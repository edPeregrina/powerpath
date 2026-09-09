import sqlite3

from src.realized_state_cache import SQLiteSharedRealizedStateCache


class _PragmaFailingConnection:
    def __init__(self, *, fail_on_journal_mode: bool) -> None:
        self.fail_on_journal_mode = fail_on_journal_mode
        self.closed = False

    def execute(self, sql: str):
        if self.fail_on_journal_mode and "journal_mode = WAL" in sql:
            raise sqlite3.OperationalError("database is locked")
        return object()

    def close(self) -> None:
        self.closed = True


class _CreateTableFailingConnection:
    def __init__(self) -> None:
        self.create_attempts = 0

    def execute(self, sql: str):
        if "CREATE TABLE IF NOT EXISTS realized_state_cache" in sql:
            self.create_attempts += 1
            if self.create_attempts == 1:
                raise sqlite3.OperationalError("database is locked")
        return object()


def test_connect_retries_when_journal_mode_pragma_is_locked(monkeypatch, tmp_path):
    monkeypatch.setattr(SQLiteSharedRealizedStateCache, "_init_db", lambda self: None)
    backend = SQLiteSharedRealizedStateCache(
        tmp_path / "connect_retry.sqlite",
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    first = _PragmaFailingConnection(fail_on_journal_mode=True)
    second = _PragmaFailingConnection(fail_on_journal_mode=False)
    connect_calls = iter([first, second])

    monkeypatch.setattr("src.realized_state_cache.sqlite3.connect", lambda *args, **kwargs: next(connect_calls))
    conn = backend._connect()

    assert conn is second
    assert first.closed is True
    assert backend.get_stats()["lock_retries"] == 1


def test_init_db_retries_create_table_when_locked(monkeypatch, tmp_path):
    conn = _CreateTableFailingConnection()
    monkeypatch.setattr(SQLiteSharedRealizedStateCache, "_connect", lambda self: conn)
    monkeypatch.setattr("src.realized_state_cache.time.sleep", lambda *_args, **_kwargs: None)

    backend = SQLiteSharedRealizedStateCache(
        tmp_path / "init_retry.sqlite",
        max_retries=2,
        retry_backoff_seconds=0.0,
    )

    assert conn.create_attempts == 2
    assert backend.get_stats()["lock_retries"] == 1


def test_set_if_absent_retries_when_connect_is_locked(monkeypatch, tmp_path):
    monkeypatch.setattr(SQLiteSharedRealizedStateCache, "_init_db", lambda self: None)
    backend = SQLiteSharedRealizedStateCache(
        tmp_path / "write_connect_retry.sqlite",
        namespace="retry_ns",
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    conn = sqlite3.connect(":memory:", isolation_level=None)
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

    calls = {"count": 0}

    def _connect_with_first_lock():
        calls["count"] += 1
        if calls["count"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return conn

    monkeypatch.setattr(backend, "_connect", _connect_with_first_lock)

    wrote = backend.set_if_absent("k", {"value": 1.0})
    row = conn.execute(
        "SELECT COUNT(*) FROM realized_state_cache WHERE namespace = ? AND cache_key = ?",
        (backend.namespace, "k"),
    ).fetchone()[0]

    assert wrote is True
    assert row == 1
    assert backend.get_stats()["lock_retries"] == 1
    conn.close()
