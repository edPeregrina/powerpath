"""Shared realised-state cache backends for societal-access postprocessing."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


REALIZED_STATE_CACHE_SCHEMA_VERSION = "2.0.0"


class SharedRealizedStateCacheBackend(Protocol):
    """Backend protocol for cross-experiment realised-state reuse."""

    def get(self, cache_key: str) -> Optional[Dict[str, float]]:
        """Return cached societal scalar fields for *cache_key* when present."""

    def set_if_absent(self, cache_key: str, fields: Dict[str, float]) -> bool:
        """Atomically write *fields* when *cache_key* is missing."""

    def get_stats(self) -> Dict[str, int]:
        """Return backend counters for observability."""


class SQLiteSharedRealizedStateCache:
    """Transactional SQLite backend for shared realised-state caching.

    This backend provides atomic reads/writes and deterministic conflict
    behaviour through SQLite transactions and a unique primary key.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        namespace: str = "default",
        schema_version: str = REALIZED_STATE_CACHE_SCHEMA_VERSION,
        timeout_seconds: float = 30.0,
        busy_timeout_ms: int = 30000,
        max_retries: int = 5,
        retry_backoff_seconds: float = 0.05,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace)
        self.schema_version = str(schema_version)
        self.timeout_seconds = float(timeout_seconds)
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._stats_lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._conn_pid: Optional[int] = None
        self._stats: Dict[str, int] = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "write_conflicts": 0,
            "lock_retries": 0,
            "errors": 0,
        }
        self._init_db()

    def _add_stat(self, name: str, value: int = 1) -> None:
        with self._stats_lock:
            self._stats[name] = self._stats.get(name, 0) + value

    def _connect(self) -> sqlite3.Connection:
        current_pid = os.getpid()
        if self._conn is not None and self._conn_pid != current_pid:
            self.close()

        if self._conn is None:
            for attempt in range(self.max_retries + 1):
                conn: Optional[sqlite3.Connection] = None
                try:
                    conn = sqlite3.connect(
                        str(self.db_path),
                        timeout=self.timeout_seconds,
                        isolation_level=None,
                        check_same_thread=False,
                    )
                    conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
                    conn.execute("PRAGMA journal_mode = WAL")
                    conn.execute("PRAGMA synchronous = FULL")
                    self._conn = conn
                    self._conn_pid = current_pid
                    break
                except Exception as exc:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                    if self._is_lock_error(exc) and attempt < self.max_retries:
                        self._add_stat("lock_retries")
                        time.sleep(self.retry_backoff_seconds * (attempt + 1))
                        continue
                    self._add_stat("errors")
                    raise
        return self._conn

    def _init_db(self) -> None:
        for attempt in range(self.max_retries + 1):
            conn: Optional[sqlite3.Connection] = None
            try:
                conn = self._connect()
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS realized_state_cache (
                        namespace TEXT NOT NULL,
                        schema_version TEXT NOT NULL,
                        cache_key TEXT NOT NULL,
                        payload BLOB NOT NULL,
                        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                        PRIMARY KEY (namespace, schema_version, cache_key)
                    )
                    """
                )
                return
            except Exception as exc:
                if self._is_lock_error(exc) and attempt < self.max_retries:
                    self._add_stat("lock_retries")
                    self.close()
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                self._add_stat("errors")
                raise

    def _is_lock_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return "database is locked" in msg or "database table is locked" in msg

    def get(self, cache_key: str) -> Optional[Dict[str, float]]:
        self._add_stat("lookups")
        for attempt in range(self.max_retries + 1):
            conn: Optional[sqlite3.Connection] = None
            try:
                conn = self._connect()
                row = conn.execute(
                    """
                    SELECT payload
                    FROM realized_state_cache
                    WHERE namespace = ? AND schema_version = ? AND cache_key = ?
                    """,
                    (self.namespace, self.schema_version, str(cache_key)),
                ).fetchone()
                if row is None:
                    self._add_stat("misses")
                    return None
                payload = json.loads(row[0])
                if not isinstance(payload, dict):
                    self._add_stat("errors")
                    return None
                self._add_stat("hits")
                return {str(k): float(v) for k, v in payload.items()}
            except Exception as exc:
                if self._is_lock_error(exc) and attempt < self.max_retries:
                    self._add_stat("lock_retries")
                    self.close()
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                self._add_stat("errors")
                raise

    def set_if_absent(self, cache_key: str, fields: Dict[str, float]) -> bool:
        payload = json.dumps(
            {str(k): float(v) for k, v in dict(fields).items()},
            sort_keys=True,
            separators=(",", ":"),
        )
        for attempt in range(self.max_retries + 1):
            conn: Optional[sqlite3.Connection] = None
            try:
                conn = self._connect()
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO realized_state_cache
                    (namespace, schema_version, cache_key, payload)
                    VALUES (?, ?, ?, ?)
                    """,
                    (self.namespace, self.schema_version, str(cache_key), payload),
                )
                conn.execute("COMMIT")
                if cur.rowcount == 1:
                    self._add_stat("writes")
                    return True
                self._add_stat("write_conflicts")
                return False
            except Exception as exc:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                if self._is_lock_error(exc) and attempt < self.max_retries:
                    self._add_stat("lock_retries")
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                self._add_stat("errors")
                raise
        self._add_stat("errors")
        return False

    def get_stats(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def close(self) -> None:
        """Close the underlying SQLite connection if open."""
        conn = self._conn
        self._conn = None
        self._conn_pid = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle-safe state for spawn-based multiprocessing."""
        self.close()
        state = dict(self.__dict__)
        state["_stats_lock"] = None
        state["_conn"] = None
        state["_conn_pid"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._stats_lock = threading.Lock()
        self._conn = None
        self._conn_pid = None


def build_shared_realized_state_cache_from_config(
    cache_config: Optional[Dict[str, Any]],
    *,
    default_db_path: Optional[str | Path] = None,
) -> Optional[SharedRealizedStateCacheBackend]:
    """Build a shared realised-state cache backend from a config dict."""
    if not cache_config:
        return None

    if not cache_config.get("enabled", False):
        return None

    backend = str(cache_config.get("backend", "sqlite")).lower()
    if backend != "sqlite":
        raise ValueError(f"Unsupported shared realized-state cache backend: {backend!r}")

    configured_path = cache_config.get("path")
    if configured_path is None and default_db_path is None:
        raise ValueError("Shared realized-state cache requires a database path.")

    db_path = configured_path if configured_path is not None else default_db_path
    return SQLiteSharedRealizedStateCache(
        db_path=db_path,
        namespace=cache_config.get("namespace", "default"),
        schema_version=cache_config.get(
            "schema_version", REALIZED_STATE_CACHE_SCHEMA_VERSION
        ),
        timeout_seconds=float(cache_config.get("timeout_seconds", 30.0)),
        busy_timeout_ms=int(cache_config.get("busy_timeout_ms", 30000)),
        max_retries=int(cache_config.get("max_retries", 5)),
        retry_backoff_seconds=float(cache_config.get("retry_backoff_seconds", 0.05)),
    )
