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


def _single_timestep(road_state_key: str, island_ids: tuple[int, int], operational: np.ndarray):
    return (
        [{"timestep": 0, "map": 0}],
        [{
            "timestep": 0,
            "map": 0,
            "road_state_key": road_state_key,
            "operational": operational,
            "island_id": np.array([island_ids[0], island_ids[1]]),
        }],
    )


def _run_once(
    *,
    shared_cache=None,
    telemetry=None,
    islands_ids=(1, 2),
    road_state_key="roads",
    operational=None,
    pop_grid_gdf=None,
    all_functions=None,
    reference_group="total",
):
    if operational is None:
        operational = np.array([True, False])
    summary, detailed = _single_timestep(road_state_key, islands_ids, operational)
    kwargs = _common_kwargs(shared_cache=shared_cache, cache_telemetry=telemetry)
    kwargs["islands_gdf_cache"] = {road_state_key: _make_islands(islands_ids)}
    if pop_grid_gdf is not None:
        kwargs["pop_grid_gdf"] = pop_grid_gdf
    if all_functions is not None:
        kwargs["all_functions"] = all_functions
    result, _ = postprocess_societal_access_results(
        summary_results=copy.deepcopy(summary),
        detailed_results=detailed,
        reference_group=reference_group,
        **kwargs,
    )
    return result


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


def test_cache_off_vs_sqlite_shared_cache_outputs_are_equivalent(tmp_path):
    cache_off = _run_once(shared_cache=None, telemetry={})
    shared_cache = SQLiteSharedRealizedStateCache(tmp_path / "equivalence.sqlite")
    _run_once(shared_cache=shared_cache, telemetry={})
    cache_on = _run_once(shared_cache=shared_cache, telemetry={})
    keys = [k for k in cache_off[0].keys() if k.startswith("societal_")]
    assert keys
    for key in keys:
        left = float(cache_off[0][key])
        right = float(cache_on[0][key])
        if np.isnan(left) and np.isnan(right):
            continue
        assert left == pytest.approx(right)


def test_shared_cache_hits_when_identical_state_repeats(tmp_path):
    shared_cache = SQLiteSharedRealizedStateCache(tmp_path / "repeat.sqlite")
    first_telemetry = {}
    second_telemetry = {}
    _run_once(shared_cache=shared_cache, telemetry=first_telemetry)
    _run_once(shared_cache=shared_cache, telemetry=second_telemetry)
    assert first_telemetry.get("shared_misses", 0) >= 1
    assert second_telemetry.get("shared_hits", 0) >= 1
    assert second_telemetry.get("shared_misses", 0) == 0


@pytest.mark.parametrize(
    "variant",
    [
        "operational_signature",
        "population_distribution",
        "reference_group",
        "function_set",
    ],
)
def test_shared_cache_key_sensitivity_matrix_misses_on_determinant_change(tmp_path, variant):
    shared_cache = SQLiteSharedRealizedStateCache(tmp_path / f"sensitivity_{variant}.sqlite")
    _run_once(shared_cache=shared_cache, telemetry={})
    telemetry = {}

    if variant == "operational_signature":
        _run_once(
            shared_cache=shared_cache,
            telemetry=telemetry,
            operational=np.array([False, True]),
        )
    elif variant == "population_distribution":
        pop_alt = _make_population()
        pop_alt.loc[0, "aantal_inwoners"] = 140
        pop_alt.loc[1, "aantal_inwoners"] = 60
        _run_once(shared_cache=shared_cache, telemetry=telemetry, pop_grid_gdf=pop_alt)
    elif variant == "reference_group":
        _run_once(shared_cache=shared_cache, telemetry=telemetry, reference_group="elderly")
    elif variant == "function_set":
        _run_once(
            shared_cache=shared_cache,
            telemetry=telemetry,
            all_functions=["hospital"],
        )
    else:
        raise AssertionError(f"Unhandled variant {variant}")

    assert telemetry.get("shared_hits", 0) == 0
    assert telemetry.get("shared_misses", 0) >= 1


def test_shared_cache_no_false_hit_from_neighboring_seeded_entry(tmp_path):
    shared_cache = SQLiteSharedRealizedStateCache(tmp_path / "false_hit.sqlite")
    shared_cache.set_if_absent(
        "neighboring-key",
        {
            "societal_access_pct__hospital__total": 999.0,
            "societal_access_relative_access__hospital__total": 999.0,
        },
    )

    telemetry = {}
    changed = _run_once(
        shared_cache=shared_cache,
        telemetry=telemetry,
        operational=np.array([False, True]),
    )
    assert telemetry.get("shared_hits", 0) == 0
    assert float(changed[0]["societal_access_pct__hospital__total"]) != pytest.approx(999.0)


def _spawn_backend_worker(backend: SQLiteSharedRealizedStateCache, queue) -> None:
    backend.set_if_absent("spawn-key", {"value": 1.0})
    queue.put(backend.get("spawn-key") is not None)


def test_sqlite_shared_cache_spawn_roundtrip_serialization_smoke(tmp_path):
    backend = SQLiteSharedRealizedStateCache(tmp_path / "spawn_smoke.sqlite")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_spawn_backend_worker, args=(backend, queue))
    proc.start()
    proc.join(timeout=30)
    assert proc.exitcode == 0
    assert queue.get(timeout=5) is True
