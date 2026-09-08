import copy
import multiprocessing as mp
import pickle
import sqlite3
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from src.realized_state_cache import SQLiteSharedRealizedStateCache
import src.societal_access as societal_access_module
from src.societal_access import postprocess_societal_access_results


def _make_assets():
    return gpd.GeoDataFrame(
        {"type": ["hospital", "school"]},
        geometry=[Point(2, 2), Point(12, 2)],
        crs="EPSG:28992",
    )


def _make_population():
    return gpd.GeoDataFrame(
        {
            "cell_id": ["left", "right"],
            "aantal_inwoners": [100, 100],
            "aantal_inwoners_65_jaar_en_ouder": [20, 20],
            "aantal_inwoners_0_tot_15_jaar": [25, 25],
        },
        geometry=[box(0, 0, 8, 8), box(10, 0, 18, 8)],
        crs="EPSG:28992",
    )


def _make_islands(base_ids=(1, 2)):
    return gpd.GeoDataFrame(
        {"island_id": [base_ids[0], base_ids[1]]},
        geometry=[box(0, 0, 9, 9), box(9, 0, 19, 9)],
        crs="EPSG:28992",
    )


def _common_kwargs(shared_cache=None, cache_telemetry=None):
    return dict(
        gdf_assets=_make_assets(),
        pop_grid_gdf=_make_population(),
        cell_id_column="cell_id",
        taxonomy={"hospital": "hospital", "school": "education"},
        all_functions=["hospital", "education"],
        allocation_cache={},
        islands_gdf_cache={},
        shared_realized_state_cache=shared_cache,
        cache_telemetry=cache_telemetry,
    )


def test_shared_cache_is_label_invariant_across_island_id_renumbering(tmp_path):
    db_path = tmp_path / "shared_state.sqlite"
    shared_cache = SQLiteSharedRealizedStateCache(db_path)
    telemetry = {}

    islands_a = _make_islands((1, 2))
    islands_b = _make_islands((22, 24))

    summary_a = [{"timestep": 0, "map": 0}]
    detailed_a = [{
        "timestep": 0,
        "map": 0,
        "road_state_key": "roads_a",
        "operational": np.array([True, False]),
        "island_id": np.array([1, 2]),
    }]
    kwargs_a = _common_kwargs(shared_cache=shared_cache, cache_telemetry=telemetry)
    kwargs_a["islands_gdf_cache"] = {"roads_a": islands_a}
    out_a, _ = postprocess_societal_access_results(
        summary_results=copy.deepcopy(summary_a),
        detailed_results=detailed_a,
        **kwargs_a,
    )

    summary_b = [{"timestep": 0, "map": 0}]
    detailed_b = [{
        "timestep": 0,
        "map": 0,
        "road_state_key": "roads_b",
        "operational": np.array([True, False]),
        "island_id": np.array([22, 24]),
    }]
    kwargs_b = _common_kwargs(shared_cache=shared_cache, cache_telemetry=telemetry)
    kwargs_b["islands_gdf_cache"] = {"roads_b": islands_b}
    out_b, _ = postprocess_societal_access_results(
        summary_results=copy.deepcopy(summary_b),
        detailed_results=detailed_b,
        **kwargs_b,
    )

    assert out_a[0]["societal_access_pct__hospital__total"] == out_b[0]["societal_access_pct__hospital__total"]
    stats = shared_cache.get_stats()
    assert stats["writes"] == 1
    assert stats["hits"] >= 1
    assert telemetry.get("shared_hits", 0) >= 1


class _FailingSharedCache:
    def get(self, _key):
        raise RuntimeError("read failed")

    def set_if_absent(self, _key, _fields):
        raise RuntimeError("write failed")


def test_shared_cache_failure_falls_back_when_fail_hard_disabled():
    kwargs = _common_kwargs(shared_cache=_FailingSharedCache(), cache_telemetry={})
    kwargs["islands_gdf_cache"] = {"roads": _make_islands((1, 2))}

    updated, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "roads",
            "operational": np.array([True, False]),
            "island_id": np.array([1, 2]),
        }],
        shared_cache_fail_hard=False,
        **kwargs,
    )

    assert np.isfinite(updated[0]["societal_access_pct__hospital__total"])


def test_shared_cache_failure_raises_when_fail_hard_enabled():
    kwargs = _common_kwargs(shared_cache=_FailingSharedCache(), cache_telemetry={})
    kwargs["islands_gdf_cache"] = {"roads": _make_islands((1, 2))}

    with pytest.raises(RuntimeError):
        postprocess_societal_access_results(
            summary_results=[{"timestep": 0, "map": 0}],
            detailed_results=[{
                "timestep": 0,
                "map": 0,
                "road_state_key": "roads",
                "operational": np.array([True, False]),
                "island_id": np.array([1, 2]),
            }],
            shared_cache_fail_hard=True,
            **kwargs,
        )


def _contended_insert_worker(db_path: str, value: int) -> None:
    backend = SQLiteSharedRealizedStateCache(Path(db_path), namespace="contention")
    backend.set_if_absent("shared-key", {"value": float(value)})


def test_sqlite_shared_cache_is_atomic_under_multiprocess_contention(tmp_path):
    db_path = tmp_path / "contended.sqlite"
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_contended_insert_worker, args=(str(db_path), i))
        for i in range(8)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        assert proc.exitcode == 0

    backend = SQLiteSharedRealizedStateCache(db_path, namespace="contention")
    row = backend.get("shared-key")
    assert isinstance(row, dict)
    assert "value" in row

    conn = sqlite3.connect(str(db_path))
    try:
        count = conn.execute(
            """
            SELECT COUNT(*) FROM realized_state_cache
            WHERE namespace = ? AND schema_version = ? AND cache_key = ?
            """,
            ("contention", "1.0.0", "shared-key"),
        ).fetchone()[0]
    finally:
        conn.close()

    assert count == 1


def test_shared_key_builder_not_used_when_shared_cache_disabled(monkeypatch):
    kwargs = _common_kwargs(shared_cache=None, cache_telemetry={})
    kwargs["islands_gdf_cache"] = {"roads": _make_islands((1, 2))}

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("shared key builder should not run when shared cache is disabled")

    monkeypatch.setattr(
        societal_access_module,
        "_build_realized_state_cache_key",
        _fail_if_called,
    )

    updated, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "roads",
            "operational": np.array([True, False]),
            "island_id": np.array([1, 2]),
        }],
        **kwargs,
    )
    assert np.isfinite(updated[0]["societal_access_pct__hospital__total"])


def test_sqlite_shared_cache_backend_is_pickle_safe(tmp_path):
    backend = SQLiteSharedRealizedStateCache(tmp_path / "pickle_safe.sqlite")
    payload = pickle.dumps(backend)
    restored = pickle.loads(payload)
    assert isinstance(restored, SQLiteSharedRealizedStateCache)
