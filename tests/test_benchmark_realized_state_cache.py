import importlib
import sys
import types
from pathlib import Path


def test_benchmark_main_uses_temporary_cache_db_by_default(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda *_args, **_kwargs: None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "ema_workbench",
        types.SimpleNamespace(
            MultiprocessingEvaluator=object(),
            SequentialEvaluator=object(),
            Samplers=types.SimpleNamespace(LHS="lhs"),
        ),
    )
    benchmark_module = importlib.import_module("src.benchmark_realized_state_cache")
    captured_paths = []

    monkeypatch.setattr(
        benchmark_module,
        "_load_factory",
        lambda _spec: lambda: {"model": object(), "policies": [], "scenarios": []},
    )
    monkeypatch.setattr(benchmark_module, "_evict_worker_shared_realized_state_cache_backends", lambda: None)
    monkeypatch.setattr(benchmark_module, "_reset_cache_db", lambda _path: None)

    def _fake_run_one(**kwargs):
        captured_paths.append(kwargs["cache_db_path"])
        return {"shared_cache_enabled": kwargs["shared_cache_enabled"], "seconds": 0.0}

    monkeypatch.setattr(benchmark_module, "_run_one", _fake_run_one)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_realized_state_cache",
            "--factory",
            "tests.fake:build_context",
            "--out-json",
            str(tmp_path / "benchmark.json"),
            "--modes",
            "seq_shared_cache",
        ],
    )

    exit_code = benchmark_module.main()

    assert exit_code == 0
    assert len(captured_paths) == 1
    assert captured_paths[0].name.startswith("powerpath-benchmark-cache-")
    assert captured_paths[0].suffix == ".sqlite"
    assert captured_paths[0] != Path("data/interim/societal_realized_state_cache.sqlite").resolve()


def test_benchmark_main_rejects_existing_user_cache_db(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "numpy",
        types.SimpleNamespace(random=types.SimpleNamespace(seed=lambda *_args, **_kwargs: None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "ema_workbench",
        types.SimpleNamespace(
            MultiprocessingEvaluator=object(),
            SequentialEvaluator=object(),
            Samplers=types.SimpleNamespace(LHS="lhs"),
        ),
    )
    benchmark_module = importlib.import_module("src.benchmark_realized_state_cache")
    existing_cache_db = tmp_path / "existing-cache.sqlite"
    existing_cache_db.write_text("seed", encoding="utf-8")

    monkeypatch.setattr(
        benchmark_module,
        "_load_factory",
        lambda _spec: lambda: {"model": object(), "policies": [], "scenarios": []},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_realized_state_cache",
            "--factory",
            "tests.fake:build_context",
            "--out-json",
            str(tmp_path / "benchmark.json"),
            "--cache-db",
            str(existing_cache_db),
            "--modes",
            "seq_shared_cache",
        ],
    )

    try:
        benchmark_module.main()
        assert False, "expected main() to exit via parser.error for existing --cache-db path"
    except SystemExit as exc:
        assert exc.code == 2
