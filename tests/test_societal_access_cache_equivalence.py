"""Equivalence tests for the societal-access postprocessing caches (work item 8).

These tests exercise the caches added by work items 1-5 in
``postprocess_societal_access_results`` / ``_build_service_area_population_maps``
/ ``_apply_service_area_societal_scalars`` and confirm the hard constraint from
the optimisation task: results must be *exactly* (bit-identical) equal whether
or not a cache hit occurs.

The approach: run the same set of timesteps twice.

1. **Batched** — a single call to ``postprocess_societal_access_results`` with
   all timesteps at once.  Because several timesteps intentionally share the
   same ``(allocation_cache_key, frozen_island_function_map, op_signature)``
   combination, this run exercises real cache hits in the function-local
   ``_scalar_fields_cache`` (work item 4) and ``_pop_alloc_cache``.
2. **Per-timestep (cache-disabled reference)** — one call to
   ``postprocess_societal_access_results`` *per timestep*.  Because the
   function-local caches are created fresh on every call (they must not be
   module-level per the task's constraints), every one of these calls is a
   guaranteed cache miss — a "caching disabled" reference computation.

The two must produce exactly equal societal fields for every timestep.
"""

import copy

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from src.societal_access import postprocess_societal_access_results


def _make_islands_gdf():
    return gpd.GeoDataFrame(
        {"island_id": [1, 2]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)],
        crs="EPSG:28992",
    )


def _make_population_gdf():
    return gpd.GeoDataFrame(
        {
            "cell_id": ["a", "b", "c", "d"],
            "aantal_inwoners": [100, 50, 30, 20],
            "aantal_inwoners_65_jaar_en_ouder": [20, 10, 6, 4],
            "aantal_inwoners_0_tot_15_jaar": [25, 10, 5, 2],
            "aantal_inwoners_25_tot_45_jaar": [40, 20, 10, 5],
        },
        geometry=[
            box(1, 1, 4, 4),
            box(6, 6, 9, 9),
            box(11, 1, 14, 4),
            box(16, 6, 19, 9),
        ],
        crs="EPSG:28992",
    )


def _make_assets_gdf():
    """Three MSLS providers (nearest-neighbour service-area path) + two
    hospitals (island-membership path), covering both service-area and
    generic societal-access code paths touched by work items 1-5."""
    return gpd.GeoDataFrame(
        {"type": ["msls", "msls", "msls", "hospital", "hospital"]},
        geometry=[
            Point(2, 2),
            Point(7, 7),
            Point(12, 2),
            Point(3, 3),
            Point(13, 3),
        ],
        crs="EPSG:28992",
    )


def _build_fixture():
    """Several timesteps with deliberately repeated operational states across
    two distinct road states, to exercise both cache hits and misses in the
    function-local pop-allocation and scalar caches."""
    islands = _make_islands_gdf()
    islands_gdf_cache = {"state_a": islands, "state_b": islands}

    op_pattern_1 = np.array([True, True, True, True, True])
    isl_1 = np.array([1, 1, 2, 1, 2])

    op_pattern_2 = np.array([False, True, True, True, False])
    isl_2 = np.array([1, 1, 2, 1, 2])

    op_pattern_3 = np.array([False, False, False, False, False])
    isl_3 = np.array([1, 1, 2, 1, 2])

    # (road_state_key, operational, island_id) per timestep, with intentional
    # repeats: ts0==ts2==ts5 (state_a/pattern_1), ts1==ts4 (state_a/pattern_2),
    # ts3==ts6 (state_b/pattern_1, i.e. same op state as ts0 but a different
    # allocation_cache_key -- exercises work item 4's key extension).
    timeline = [
        ("state_a", op_pattern_1, isl_1),
        ("state_a", op_pattern_2, isl_2),
        ("state_a", op_pattern_1, isl_1),
        ("state_b", op_pattern_1, isl_1),
        ("state_a", op_pattern_2, isl_2),
        ("state_a", op_pattern_1, isl_1),
        ("state_b", op_pattern_1, isl_1),
        ("state_a", op_pattern_3, isl_3),
    ]

    summary_results = [{"timestep": i, "map": 0} for i in range(len(timeline))]
    detailed_results = [
        {
            "timestep": i,
            "map": 0,
            "road_state_key": road_state_key,
            "operational": operational,
            "island_id": island_id,
        }
        for i, (road_state_key, operational, island_id) in enumerate(timeline)
    ]
    return islands_gdf_cache, summary_results, detailed_results


_COMMON_KWARGS = dict(
    gdf_assets=_make_assets_gdf(),
    pop_grid_gdf=_make_population_gdf(),
    cell_id_column="cell_id",
    taxonomy={"msls": "electricity", "hospital": "hospital"},
    all_functions=["electricity", "hospital"],
    nearest_max_distance=200.0,
)


def test_batched_cached_run_matches_per_timestep_uncached_reference():
    islands_gdf_cache, summary_results, detailed_results = _build_fixture()

    # --- Batched run: exercises cache hits across repeated timesteps ---
    batched_summary, _ = postprocess_societal_access_results(
        summary_results=copy.deepcopy(summary_results),
        detailed_results=detailed_results,
        allocation_cache={},
        islands_gdf_cache=islands_gdf_cache,
        **_COMMON_KWARGS,
    )

    # --- Per-timestep reference run: fresh function-local caches every call,
    # i.e. every call is a guaranteed cache miss ("caching disabled"). ---
    reference_summary = []
    for i in range(len(summary_results)):
        single_summary, _ = postprocess_societal_access_results(
            summary_results=[copy.deepcopy(summary_results[i])],
            detailed_results=[detailed_results[i]],
            allocation_cache={},
            islands_gdf_cache=islands_gdf_cache,
            **_COMMON_KWARGS,
        )
        reference_summary.append(single_summary[0])

    assert len(batched_summary) == len(reference_summary)
    for i, (batched_row, reference_row) in enumerate(zip(batched_summary, reference_summary)):
        assert set(batched_row.keys()) == set(reference_row.keys()), f"timestep {i}: field set differs"
        for key in reference_row:
            batched_value = batched_row[key]
            reference_value = reference_row[key]
            if isinstance(reference_value, float) and np.isnan(reference_value):
                assert isinstance(batched_value, float) and np.isnan(batched_value), (
                    f"timestep {i}, field {key!r}: expected NaN, got {batched_value!r}"
                )
            else:
                assert batched_value == reference_value, (
                    f"timestep {i}, field {key!r}: {batched_value!r} != {reference_value!r}"
                )

    # Sanity: repeated timesteps really did produce identical (non-trivial)
    # societal fields, i.e. the fixture is exercising real cache hits rather
    # than incidentally-equal NaNs.
    ts0_pct = batched_summary[0]["societal_access_pct__electricity__total"]
    ts2_pct = batched_summary[2]["societal_access_pct__electricity__total"]
    ts5_pct = batched_summary[5]["societal_access_pct__electricity__total"]
    assert not np.isnan(ts0_pct)
    assert ts0_pct == ts2_pct == ts5_pct


def test_cache_hit_does_not_return_aliased_mutable_dict():
    """Mutating one timestep's merged fields must not affect another
    timestep's fields, even when both hit the same scalar cache entry."""
    islands_gdf_cache, summary_results, detailed_results = _build_fixture()

    batched_summary, _ = postprocess_societal_access_results(
        summary_results=copy.deepcopy(summary_results),
        detailed_results=detailed_results,
        allocation_cache={},
        islands_gdf_cache=islands_gdf_cache,
        **_COMMON_KWARGS,
    )

    # ts0 and ts2 share (allocation_cache_key, frozen_island_function_map,
    # op_signature) -- ts2 is a cache hit for whatever ts0 computed/stored.
    key = "societal_access_pct__electricity__total"
    original_value = batched_summary[2][key]

    # Mutate ts0's dict in place -- this must not leak into ts2's dict.
    batched_summary[0][key] = -999.0
    batched_summary[0]["some_new_mutation_marker"] = True

    assert batched_summary[2][key] == original_value
    assert "some_new_mutation_marker" not in batched_summary[2]
