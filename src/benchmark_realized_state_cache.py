"""CLI benchmark runner for societal realized-state cache telemetry.

Usage:
    python -m src.benchmark_realized_state_cache \
      --factory my_package.my_module:build_ema_context \
      --scenarios 10 \
      --n-processes 4 \
      --out-json /tmp/cache_benchmark.json \
      --out-csv /tmp/cache_benchmark.csv

Factory contract:
    build_ema_context() -> dict with:
      - model: ema_workbench Model
      - policies: list[Policy]
      - optional uncertainty_sampling
      - optional scenarios (reused across all benchmark modes)
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import random
import tempfile
import time
from contextlib import nullcontext
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Tuple

import numpy as np
from ema_workbench import MultiprocessingEvaluator, SequentialEvaluator, Samplers


def _load_factory(factory_spec: str):
    module_name, func_name = factory_spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _iter_constants(model) -> Iterable[Any]:
    constants = getattr(model, "constants", None)
    if constants is None:
        return []
    try:
        return list(constants)
    except Exception:
        return []


def _get_constant(model, name: str):
    constants = getattr(model, "constants", None)
    if constants is None:
        return None
    try:
        return constants[name]
    except Exception:
        pass
    for const in _iter_constants(model):
        if getattr(const, "name", None) == name:
            return const
    return None


def _configure_shared_cache(
    model,
    *,
    enabled: bool,
    cache_db_path: Path,
    namespace: str,
    schema_version: str,
    telemetry: MutableMapping[str, int] | None = None,
    telemetry_lock: Any | None = None,
) -> MutableMapping[str, int]:
    if telemetry is None:
        telemetry = {}
    const = _get_constant(model, "societal_access_config")
    if const is None:
        raise RuntimeError("Model constant 'societal_access_config' is required for cache benchmarking")
    config = copy.deepcopy(getattr(const, "value", {}) or {})
    config.pop("shared_realized_state_cache", None)
    config["cache_telemetry"] = telemetry
    config["cache_telemetry_lock"] = telemetry_lock
    if enabled:
        config["shared_realized_state_cache_config"] = {
            "enabled": True,
            "backend": "sqlite",
            "path": str(cache_db_path),
            "namespace": namespace,
            "schema_version": schema_version,
        }
    else:
        config["shared_realized_state_cache_config"] = {"enabled": False}
    const.value = config
    return telemetry


def _run_one(
    *,
    evaluator_cls,
    evaluator_kwargs: Dict[str, Any],
    factory_spec: str,
    scenarios: int,
    shared_cache_enabled: bool,
    cache_db_path: Path,
    namespace: str,
    schema_version: str,
    shared_scenarios: Any = None,
    sampling_seed: int | None = 42,
) -> Dict[str, Any]:
    context = _load_factory(factory_spec)()
    model = context["model"]
    policies = context["policies"]
    uncertainty_sampling = context.get("uncertainty_sampling", Samplers.LHS)
    scenarios_arg = (
        copy.deepcopy(shared_scenarios) if shared_scenarios is not None else scenarios
    )
    manager_context = Manager() if evaluator_cls is MultiprocessingEvaluator else nullcontext()
    with manager_context as manager:
        telemetry_store: MutableMapping[str, int] = manager.dict() if manager is not None else {}
        telemetry_lock = manager.Lock() if manager is not None else None
        telemetry = _configure_shared_cache(
            model,
            enabled=shared_cache_enabled,
            cache_db_path=cache_db_path,
            namespace=namespace,
            schema_version=schema_version,
            telemetry=telemetry_store,
            telemetry_lock=telemetry_lock,
        )
        started = time.perf_counter()
        if shared_scenarios is None and sampling_seed is not None:
            random.seed(sampling_seed)
            np.random.seed(sampling_seed)
        with evaluator_cls(model, **evaluator_kwargs) as evaluator:
            experiments, _outcomes = evaluator.perform_experiments(
                scenarios=scenarios_arg,
                policies=policies,
                uncertainty_sampling=uncertainty_sampling,
            )
        elapsed = time.perf_counter() - started
        telemetry_snapshot = {k: int(v) for k, v in telemetry.items()}
    try:
        scenarios_count = int(len(scenarios_arg))
    except Exception:
        scenarios_count = int(scenarios)
    record = {
        "evaluator": evaluator_cls.__name__,
        "shared_cache_enabled": bool(shared_cache_enabled),
        "seconds": float(elapsed),
        "experiments": int(len(experiments)),
        "scenarios": scenarios_count,
        "policies": int(len(policies)),
        "scenario_source": (
            "factory_shared_set" if shared_scenarios is not None else f"sampler_seed_{sampling_seed}"
        ),
    }
    record.update(telemetry_snapshot)
    return record


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    columns: List[str] = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _reset_cache_db(cache_db_path: Path) -> None:
    for path in (
        cache_db_path,
        Path(f"{cache_db_path}-shm"),
        Path(f"{cache_db_path}-wal"),
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _evict_worker_shared_realized_state_cache_backends() -> None:
    try:
        from src import simulation as simulation_module
    except Exception:
        return

    with simulation_module._WORKER_SHARED_REALIZED_STATE_CACHE_LOCK:
        for backend in simulation_module._WORKER_SHARED_REALIZED_STATE_CACHE_BACKENDS.values():
            close = getattr(backend, "close", None)
            if callable(close):
                close()
        simulation_module._WORKER_SHARED_REALIZED_STATE_CACHE_BACKENDS.clear()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark societal realized-state cache behavior.")
    parser.add_argument("--factory", required=True, help="Dotted factory path module:function")
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--n-processes", type=int, default=4)
    parser.add_argument("--cache-db")
    parser.add_argument("--namespace", default="societal")
    parser.add_argument("--schema-version", default="2.0.0")
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "seq_no_cache",
            "seq_shared_cache",
            "mp_no_cache",
            "mp_shared_cache",
        ],
        choices=["seq_no_cache", "seq_shared_cache", "mp_no_cache", "mp_shared_cache"],
    )
    args = parser.parse_args()
    base_context = _load_factory(args.factory)()
    shared_scenarios = base_context.get("scenarios")
    auto_cache_db_path: Path | None = None
    if args.cache_db:
        cache_db_path = Path(args.cache_db).resolve()
        existing_cache_paths = [
            path
            for path in (
                cache_db_path,
                Path(f"{cache_db_path}-shm"),
                Path(f"{cache_db_path}-wal"),
            )
            if path.exists()
        ]
        if existing_cache_paths:
            parser.error(
                "--cache-db points to an existing cache database path; "
                "provide a new path to avoid overwriting existing cache data."
            )
        cache_db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        with tempfile.NamedTemporaryFile(
            prefix="powerpath-benchmark-cache-",
            suffix=".sqlite",
            delete=False,
        ) as tmp_cache_db:
            auto_cache_db_path = Path(tmp_cache_db.name)
        auto_cache_db_path.unlink(missing_ok=True)
        cache_db_path = auto_cache_db_path
    runs: List[Tuple[Any, Dict[str, Any], bool]] = []
    if "seq_no_cache" in args.modes:
        runs.append((SequentialEvaluator, {}, False))
    if "seq_shared_cache" in args.modes:
        runs.append((SequentialEvaluator, {}, True))
    if "mp_no_cache" in args.modes:
        runs.append((MultiprocessingEvaluator, {"n_processes": args.n_processes}, False))
    if "mp_shared_cache" in args.modes:
        runs.append((MultiprocessingEvaluator, {"n_processes": args.n_processes}, True))

    try:
        records: List[Dict[str, Any]] = []
        for evaluator_cls, evaluator_kwargs, shared_cache_enabled in runs:
            if shared_cache_enabled:
                _evict_worker_shared_realized_state_cache_backends()
                _reset_cache_db(cache_db_path)
            records.append(
                _run_one(
                    evaluator_cls=evaluator_cls,
                    evaluator_kwargs=evaluator_kwargs,
                    factory_spec=args.factory,
                    scenarios=args.scenarios,
                    shared_cache_enabled=shared_cache_enabled,
                    cache_db_path=cache_db_path,
                    namespace=args.namespace,
                    schema_version=args.schema_version,
                    shared_scenarios=shared_scenarios,
                    sampling_seed=args.sampling_seed,
                )
            )

        out_json = Path(args.out_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
        if args.out_csv:
            _write_csv(Path(args.out_csv).resolve(), records)
        print(json.dumps(records, indent=2))
        return 0
    finally:
        if auto_cache_db_path is not None:
            _reset_cache_db(auto_cache_db_path)


if __name__ == "__main__":
    raise SystemExit(main())
