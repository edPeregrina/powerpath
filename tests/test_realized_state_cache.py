import copy
import multiprocessing as mp
import pickle
import sqlite3
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from src.realized_state_cache import (
    REALIZED_STATE_CACHE_SCHEMA_VERSION,
    SQLiteSharedRealizedStateCache,
    build_shared_realized_state_cache_from_config,
)
import src.societal_access as societal_access_module
import src.simulation as simulation_module
from src.societal_access import (
    _build_realized_state_cache_key,
    _build_service_area_population_maps,
    _service_area_pop_map_cache_key,
    postprocess_societal_access_results,
)


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


def _common_kwargs(shared_cache=None, cache_telemetry=None, cache_telemetry_lock=None):
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
        cache_telemetry_lock=cache_telemetry_lock,
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
    telemetry_lock=None,
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
    kwargs = _common_kwargs(
        shared_cache=shared_cache,
        cache_telemetry=telemetry,
        cache_telemetry_lock=telemetry_lock,
    )
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


class _LockAwareTelemetry(dict):
    def __init__(self):
        super().__init__()
        self.lock_held = False

    def get(self, key, default=None):
        assert self.lock_held
        return super().get(key, default)

    def __setitem__(self, key, value):
        assert self.lock_held
        super().__setitem__(key, value)


class _RecordingLock:
    def __init__(self, telemetry):
        self.telemetry = telemetry

    def __enter__(self):
        self.telemetry.lock_held = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.telemetry.lock_held = False
        return False


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


def test_cache_telemetry_updates_hold_provided_lock():
    telemetry = _LockAwareTelemetry()
    lock = _RecordingLock(telemetry)

    updated = _run_once(telemetry=telemetry, telemetry_lock=lock)

    assert np.isfinite(updated[0]["societal_access_pct__hospital__total"])
    assert dict.get(telemetry, "local_hits", 0) >= 0


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
    try:
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=30)
            assert proc.exitcode == 0
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
            proc.join()

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
            ("contention", REALIZED_STATE_CACHE_SCHEMA_VERSION, "shared-key"),
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


def test_sqlite_shared_cache_backend_reopens_connection_after_pid_change(monkeypatch, tmp_path):
    backend = SQLiteSharedRealizedStateCache(tmp_path / "pid_reopen.sqlite")
    original_conn = backend._connect()
    first_pid = backend._conn_pid
    pid_stream = iter([first_pid, first_pid + 1])
    monkeypatch.setattr("src.realized_state_cache.os.getpid", lambda: next(pid_stream))

    reused_conn = backend._connect()
    reopened_conn = backend._connect()

    assert reused_conn is original_conn
    assert reopened_conn is not original_conn
    assert backend._conn_pid == first_pid + 1
    with pytest.raises(sqlite3.ProgrammingError):
        original_conn.execute("SELECT 1")


def test_worker_local_shared_backend_is_reused_per_normalized_config(monkeypatch, tmp_path):
    calls = []
    created_backend = object()

    def _fake_builder(cache_config, *, default_db_path=None):
        calls.append((cache_config, default_db_path))
        return created_backend

    monkeypatch.setattr(
        simulation_module,
        "build_shared_realized_state_cache_from_config",
        _fake_builder,
    )
    simulation_module._WORKER_SHARED_REALIZED_STATE_CACHE_BACKENDS.clear()

    cfg = {"enabled": True, "backend": "sqlite", "namespace": "ns"}
    backend_first = simulation_module._get_worker_shared_realized_state_cache_backend(
        cfg,
        default_db_path=tmp_path / "cache.sqlite",
    )
    backend_second = simulation_module._get_worker_shared_realized_state_cache_backend(
        dict(cfg),
        default_db_path=tmp_path / "." / "cache.sqlite",
    )

    assert backend_first is created_backend
    assert backend_second is created_backend
    assert len(calls) == 1


def test_build_shared_backend_expands_user_path(monkeypatch, tmp_path):
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    cache = build_shared_realized_state_cache_from_config(
        {"enabled": True, "backend": "sqlite", "path": "~/cache.sqlite"}
    )
    try:
        assert cache is not None
        assert cache.db_path == (fake_home / "cache.sqlite").resolve()
    finally:
        cache.close()


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


def test_cache_telemetry_omits_cumulative_backend_stats(tmp_path):
    shared_cache = SQLiteSharedRealizedStateCache(tmp_path / "backend_stats.sqlite")
    telemetry = {}
    _run_once(shared_cache=shared_cache, telemetry=telemetry)
    assert not any(key.startswith("shared_backend_") for key in telemetry)


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
    try:
        proc.start()
        proc.join(timeout=30)
        assert proc.exitcode == 0
        assert queue.get(timeout=5) is True
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join()


def test_realized_state_key_changes_with_service_area_maps_digest():
    key_a = _build_realized_state_cache_key(
        frozen_island_function_map={1: frozenset({"hospital"})},
        operational_asset_ids_by_function={"hospital": {0}},
        island_pop=societal_access_module.pd.DataFrame(
            {"total_weighted": [100.0]},
            index=societal_access_module.pd.Index([1], name="island_id"),
        ),
        total_pop={"total": 100.0},
        available_group_cols={"total": "total_weighted"},
        all_functions=["hospital"],
        pop_group_columns={"total": "aantal_inwoners"},
        reference_group="total",
        service_area_function_provider_types={"electricity": frozenset({"msls"})},
        allocation_df=societal_access_module.pd.DataFrame(),
        service_area_population_maps_digest="digest_a",
    )
    key_b = _build_realized_state_cache_key(
        frozen_island_function_map={1: frozenset({"hospital"})},
        operational_asset_ids_by_function={"hospital": {0}},
        island_pop=societal_access_module.pd.DataFrame(
            {"total_weighted": [100.0]},
            index=societal_access_module.pd.Index([1], name="island_id"),
        ),
        total_pop={"total": 100.0},
        available_group_cols={"total": "total_weighted"},
        all_functions=["hospital"],
        pop_group_columns={"total": "aantal_inwoners"},
        reference_group="total",
        service_area_function_provider_types={"electricity": frozenset({"msls"})},
        allocation_df=societal_access_module.pd.DataFrame(),
        service_area_population_maps_digest="digest_b",
    )
    assert key_a != key_b


def test_realized_state_key_distinguishes_unassigned_island_sentinel():
    shared_kwargs = dict(
        operational_asset_ids_by_function={"hospital": {"provider_a"}},
        total_pop={"total": 100.0},
        available_group_cols={"total": "total_weighted"},
        all_functions=["hospital"],
        pop_group_columns={"total": "aantal_inwoners"},
        reference_group="total",
        service_area_function_provider_types={"electricity": frozenset({"msls"})},
        allocation_df=societal_access_module.pd.DataFrame(),
        service_area_population_maps_digest="same_digest",
    )
    key_unassigned = _build_realized_state_cache_key(
        frozen_island_function_map={-1: frozenset({"hospital"})},
        island_pop=societal_access_module.pd.DataFrame(
            {"total_weighted": [100.0]},
            index=societal_access_module.pd.Index([-1], name="island_id"),
        ),
        **shared_kwargs,
    )
    key_regular = _build_realized_state_cache_key(
        frozen_island_function_map={1: frozenset({"hospital"})},
        island_pop=societal_access_module.pd.DataFrame(
            {"total_weighted": [100.0]},
            index=societal_access_module.pd.Index([1], name="island_id"),
        ),
        **shared_kwargs,
    )
    assert key_unassigned != key_regular


def test_service_area_pop_map_cache_key_uses_population_values():
    assets = gpd.GeoDataFrame(
        {"type": ["msls", "msls"]},
        geometry=[Point(0, 0), Point(10, 10)],
        crs="EPSG:28992",
    )
    pop_1 = gpd.GeoDataFrame(
        {
            "cell_id": ["c1", "c2"],
            "aantal_inwoners": [100.0, 50.0],
        },
        geometry=[box(0, 0, 3, 3), box(8, 8, 12, 12)],
        crs="EPSG:28992",
    )
    pop_2 = pop_1.copy()
    pop_2.loc[0, "aantal_inwoners"] = 999.0

    working_assets = assets[["type", "geometry"]].copy()
    asset_digest = societal_access_module._service_area_asset_state_digest(working_assets)
    spec = {"electricity": frozenset({"msls"})}
    key_1 = _service_area_pop_map_cache_key(
        asset_digest,
        pop_1,
        {"total": "aantal_inwoners"},
        spec,
        "type",
    )
    key_2 = _service_area_pop_map_cache_key(
        asset_digest,
        pop_2,
        {"total": "aantal_inwoners"},
        spec,
        "type",
    )
    assert key_1 != key_2


def test_service_area_pop_map_cache_key_uses_asset_state_digest():
    pop = gpd.GeoDataFrame(
        {
            "cell_id": ["c1", "c2"],
            "aantal_inwoners": [100.0, 50.0],
        },
        geometry=[box(0, 0, 3, 3), box(8, 8, 12, 12)],
        crs="EPSG:28992",
    )
    spec = {"electricity": frozenset({"msls"})}

    assets_a = gpd.GeoDataFrame(
        {"type": ["msls", "msls"]},
        geometry=[Point(0, 0), Point(10, 10)],
        crs="EPSG:28992",
    )
    assets_b = gpd.GeoDataFrame(
        {"type": ["msls", "msls"]},
        geometry=[Point(0, 10), Point(10, 0)],  # same bbox/count, different layout
        crs="EPSG:28992",
    )
    digest_a = societal_access_module._service_area_asset_state_digest(
        assets_a[["type", "geometry"]].copy()
    )
    digest_b = societal_access_module._service_area_asset_state_digest(
        assets_b[["type", "geometry"]].copy()
    )
    key_a = _service_area_pop_map_cache_key(
        digest_a,
        pop,
        {"total": "aantal_inwoners"},
        spec,
        "type",
    )
    key_b = _service_area_pop_map_cache_key(
        digest_b,
        pop,
        {"total": "aantal_inwoners"},
        spec,
        "type",
    )
    assert key_a != key_b


def test_service_area_population_map_memo_does_not_reuse_on_population_change():
    societal_access_module._SERVICE_AREA_POP_MAP_CACHE.clear()
    assets = gpd.GeoDataFrame(
        {"type": ["msls", "msls"]},
        geometry=[Point(0, 0), Point(10, 10)],
        crs="EPSG:28992",
    )
    pop_1 = gpd.GeoDataFrame(
        {
            "cell_id": ["c1", "c2"],
            "aantal_inwoners": [100.0, 50.0],
        },
        geometry=[box(0, 0, 3, 3), box(8, 8, 12, 12)],
        crs="EPSG:28992",
    )
    pop_2 = pop_1.copy()
    pop_2.loc[:, "aantal_inwoners"] = [1.0, 2.0]

    maps_1 = _build_service_area_population_maps(
        assets,
        pop_1,
        {"total": "aantal_inwoners"},
        service_area_function_provider_types={"electricity": frozenset({"msls"})},
    )
    maps_2 = _build_service_area_population_maps(
        assets,
        pop_2,
        {"total": "aantal_inwoners"},
        service_area_function_provider_types={"electricity": frozenset({"msls"})},
    )
    sum_1 = sum(maps_1["electricity"]["total"].values())
    sum_2 = sum(maps_2["electricity"]["total"].values())
    assert sum_1 != sum_2


def test_service_area_population_maps_pass_asset_state_digest_to_voronoi(monkeypatch):
    societal_access_module._SERVICE_AREA_POP_MAP_CACHE.clear()
    assets = gpd.GeoDataFrame(
        {"type": ["msls", "msls", "msls", "msls"]},
        geometry=[Point(0, 0), Point(10, 0), Point(0, 10), Point(10, 10)],
        crs="EPSG:28992",
    )
    pop = gpd.GeoDataFrame(
        {
            "cell_id": ["c1", "c2"],
            "aantal_inwoners": [100.0, 50.0],
        },
        geometry=[box(0, 0, 3, 3), box(8, 8, 12, 12)],
        crs="EPSG:28992",
    )
    captured = {}

    def _fake_create_voronoi_for_asset_type(gdf_assets, asset_type, boundary=None, **kwargs):
        captured["asset_type"] = asset_type
        captured["asset_cache_key"] = kwargs.get("asset_cache_key")
        return gpd.GeoDataFrame(
            {"asset_id": list(gdf_assets.index)},
            geometry=gdf_assets.geometry,
            crs=gdf_assets.crs,
        )

    def _fake_build_voronoi_service_area_map(voronoi_gdf, pop_assets):
        provider_id = voronoi_gdf["asset_id"].iloc[0]
        return {provider_id: list(pop_assets.index)}

    monkeypatch.setattr("src.impacts.create_voronoi_for_asset_type", _fake_create_voronoi_for_asset_type)
    monkeypatch.setattr("src.utils.build_voronoi_service_area_map", _fake_build_voronoi_service_area_map)

    maps = _build_service_area_population_maps(
        assets,
        pop,
        {"total": "aantal_inwoners"},
        service_area_function_provider_types={"electricity": frozenset({"msls"})},
    )

    expected_digest = societal_access_module._service_area_asset_state_digest(
        assets[["type", "geometry"]].copy()
    )
    assert captured["asset_type"] == "msls"
    assert captured["asset_cache_key"] == expected_digest
    assert maps["electricity"]["total"]
