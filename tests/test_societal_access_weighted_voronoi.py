"""End-to-end regression tests for the weighted-Voronoi ("default school")
service-provision pipeline in ``src/societal_access.py``.

Unlike ``tests/test_impacts_weighted_voronoi.py`` (which only exercises the
power-diagram primitive in isolation), these tests go through
``postprocess_societal_access_results`` with ``method="weighted_voronoi"``
configured, and specifically assert the property the schools-weighted-
voronoi branch was built to guarantee: a population cell's fixed catchment
("default school") assignment is purely geometric/weight-driven and may
legitimately straddle a road-network island boundary, but access still
requires the assigned provider to be BOTH operational AND on the same
island as the cell this timestep -- not a pure operational-status bypass
like electricity's "simple_voronoi".

Geometry setup (see ``_make_two_school_scenario``)
---------------------------------------------------
Two schools sit on the x-axis: A at x=0 (few pupils) and B at x=100 (many
pupils). With no weighting the power-diagram bisector is the midpoint
x=50. A large pupil-count imbalance pulls that bisector well below x=50 --
tuned here so a population cell at x=45 is catchment-assigned to school B
even though x=45 is on the *west* side of where roads are later severed
(at x=50). This is deliberate: it is the simplest geometry that produces a
"default school on the other side of a future road cut" situation to test.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from src.societal_access import (
    get_or_build_allocation,
    postprocess_societal_access_results,
)

# Tuned so the weighted power-diagram bisector between school A (x=0, weight
# 10) and school B (x=100, weight 100000) sits just below x=45 -- see module
# docstring. Not a universally-calibrated constant; specific to this test's
# geometry/weights only.
_WEIGHT_SCALE_FACTOR = 1000.0

_SERVICE_AREA_CONFIG = {
    "education": {
        "provider_types": {"school"},
        "method": "weighted_voronoi",
        "weight_column": "pupils",
        "weight_scale_factor": _WEIGHT_SCALE_FACTOR,
    }
}


def _make_two_school_scenario():
    schools = gpd.GeoDataFrame(
        {"type": ["school", "school"], "pupils": [10, 100000]},
        geometry=[Point(0, 0), Point(100, 0)],
        crs="EPSG:28992",
    )
    pop = gpd.GeoDataFrame(
        {
            "cell_id": ["straddling_cell"],
            "aantal_inwoners": [100],
            "aantal_inwoners_65_jaar_en_ouder": [20],
            "aantal_inwoners_0_tot_15_jaar": [15],
        },
        geometry=[box(44, -1, 46, 1)],
        crs="EPSG:28992",
    )
    return schools, pop


def test_weighted_catchment_grants_full_access_when_fully_connected():
    """Sanity-check the tuned geometry under full connectivity: regardless
    of *which* school the x=45 cell is catchment-assigned to (see module
    docstring -- it should be the high-pupil school B, not the nearer
    school A), with a single island and both schools operational, access
    must be 100%.
    """
    schools, pop = _make_two_school_scenario()
    single_island = gpd.GeoDataFrame(
        {"island_id": [1]}, geometry=[box(-10, -10, 210, 10)], crs="EPSG:28992"
    )
    allocation_cache = {}
    get_or_build_allocation(
        allocation_cache,
        pop,
        "cell_id",
        single_island,
        road_state_key="connected",
    )
    updated, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "connected",
            "operational": np.array([True, True]),
            "island_id": np.array([1, 1]),
        }],
        gdf_assets=schools,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"school": "education"},
        all_functions=["education"],
        service_area_function_provider_types=_SERVICE_AREA_CONFIG,
    )
    result = updated[0]
    assert result["societal_access_pct__education__total"] == pytest.approx(100.0)


def test_access_requires_same_island_not_just_operational_status():
    """The core regression: when roads sever at x=50, school B (x=100) ends
    up on a *different* island from the pop cell at x=45 (west island), even
    though school B itself remains fully operational. Access must drop to
    0% -- if the implementation regressed to an electricity-style pure
    operational-status bypass, this would incorrectly read 100%.
    """
    schools, pop = _make_two_school_scenario()
    severed_islands = gpd.GeoDataFrame(
        {"island_id": [1, 2]},
        geometry=[box(-10, -10, 50, 10), box(50, -10, 210, 10)],
        crs="EPSG:28992",
    )

    allocation_cache = {}
    get_or_build_allocation(
        allocation_cache,
        pop,
        "cell_id",
        severed_islands,
        road_state_key="severed",
    )
    updated, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "severed",
            # Both schools operational -- school B (high-weight, the cell's
            # assigned provider) is NOT damaged; the only thing that changed
            # is road connectivity between the cell and school B.
            "operational": np.array([True, True]),
            # School A (index 0) on island 1 (west); school B (index 1) on
            # island 2 (east) -- matches the severed_islands geometry.
            "island_id": np.array([1, 2]),
        }],
        gdf_assets=schools,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"school": "education"},
        all_functions=["education"],
        service_area_function_provider_types=_SERVICE_AREA_CONFIG,
    )
    result = updated[0]
    assert result["societal_access_pct__education__total"] == pytest.approx(0.0)


_REMOTE_WEIGHT = 100010.0  # == 10 + 100000 (the two spatial schools' pupils),
# chosen so remote_fraction = _REMOTE_WEIGHT / (10 + 100000 + _REMOTE_WEIGHT)
# == 0.5 exactly -- makes the blend arithmetic easy to assert.

_SERVICE_AREA_CONFIG_WITH_REMOTE = {
    "education": {
        "provider_types": {"school"},
        "method": "weighted_voronoi",
        "weight_column": "pupils",
        "weight_scale_factor": _WEIGHT_SCALE_FACTOR,
        "exclude_column": "is_remote",
    }
}


def _make_three_school_plus_remote_scenario():
    """Same two spatial schools as ``_make_two_school_scenario`` (A at x=0,
    few pupils; B at x=100, many pupils) plus a third, non-spatial
    "default school" entry flagged ``is_remote=True`` -- this provider is
    excluded from the power-diagram tessellation entirely (see
    ``_build_weighted_voronoi_cell_assignments`` docstring) and instead
    contributes a scalar ``remote_fraction`` of the population that is
    unconditionally "already served" regardless of road/hazard state.
    """
    schools = gpd.GeoDataFrame(
        {
            "type": ["school", "school", "school"],
            "pupils": [10, 100000, _REMOTE_WEIGHT],
            "is_remote": [False, False, True],
        },
        geometry=[Point(0, 0), Point(100, 0), Point(500, 500)],
        crs="EPSG:28992",
    )
    pop = gpd.GeoDataFrame(
        {
            "cell_id": ["straddling_cell"],
            "aantal_inwoners": [100],
            "aantal_inwoners_65_jaar_en_ouder": [20],
            "aantal_inwoners_0_tot_15_jaar": [15],
        },
        geometry=[box(44, -1, 46, 1)],
        crs="EPSG:28992",
    )
    return schools, pop


def test_remote_school_fraction_excluded_from_tessellation_geometry():
    """The remote provider must have zero effect on the spatial tessellation:
    with full connectivity and both spatial schools operational, the
    straddling cell is still 100% accessible (matches
    ``test_weighted_catchment_grants_full_access_when_fully_connected``)."""
    schools, pop = _make_three_school_plus_remote_scenario()
    single_island = gpd.GeoDataFrame(
        {"island_id": [1]}, geometry=[box(-10, -10, 510, 510)], crs="EPSG:28992"
    )
    allocation_cache = {}
    get_or_build_allocation(
        allocation_cache,
        pop,
        "cell_id",
        single_island,
        road_state_key="connected_remote",
    )
    updated, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "connected_remote",
            "operational": np.array([True, True, True]),
            "island_id": np.array([1, 1, 1]),
        }],
        gdf_assets=schools,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"school": "education"},
        all_functions=["education"],
        service_area_function_provider_types=_SERVICE_AREA_CONFIG_WITH_REMOTE,
    )
    result = updated[0]
    assert result["societal_access_pct__education__total"] == pytest.approx(100.0)


def test_remote_school_fraction_stays_accessible_when_roads_severed():
    """Core remote-fraction regression: roads sever at x=50 (school B ends
    up on the opposite island from the pop cell, same as
    ``test_access_requires_same_island_not_just_operational_status``), so
    the *spatial* catchment portion drops to 0% -- but with
    ``remote_fraction == 0.5`` (tuned via ``_REMOTE_WEIGHT``), half of the
    population is unconditionally served by the remote/default-school
    bucket regardless of road state, so total access must be exactly 50%,
    not 0%.
    """
    schools, pop = _make_three_school_plus_remote_scenario()
    severed_islands = gpd.GeoDataFrame(
        {"island_id": [1, 2]},
        geometry=[box(-10, -10, 50, 10), box(50, -10, 510, 510)],
        crs="EPSG:28992",
    )
    allocation_cache = {}
    get_or_build_allocation(
        allocation_cache,
        pop,
        "cell_id",
        severed_islands,
        road_state_key="severed_remote",
    )
    updated, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "severed_remote",
            "operational": np.array([True, True, True]),
            # School A (index 0) island 1 (west); school B (index 1) and
            # the remote provider (index 2) island 2 (east) -- matches
            # severed_islands geometry.
            "island_id": np.array([1, 2, 2]),
        }],
        gdf_assets=schools,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"school": "education"},
        all_functions=["education"],
        service_area_function_provider_types=_SERVICE_AREA_CONFIG_WITH_REMOTE,
    )
    result = updated[0]
    assert result["societal_access_pct__education__total"] == pytest.approx(50.0)


def test_default_config_education_still_uses_base_island_method():
    """Without an explicit ``weighted_voronoi`` entry for "education",
    behaviour must be completely unchanged (base shared-island method,
    *any* operational school on the same island grants access) -- guards
    against the new pipeline accidentally engaging by default.
    """
    schools, pop = _make_two_school_scenario()
    single_island = gpd.GeoDataFrame(
        {"island_id": [1]}, geometry=[box(-10, -10, 210, 10)], crs="EPSG:28992"
    )
    allocation_cache = {}
    get_or_build_allocation(
        allocation_cache,
        pop,
        "cell_id",
        single_island,
        road_state_key="connected_default",
    )
    updated, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "connected_default",
            "operational": np.array([False, True]),
            "island_id": np.array([1, 1]),
        }],
        gdf_assets=schools,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"school": "education"},
        all_functions=["education"],
        # No service_area_function_provider_types override -> defaults to
        # SERVICE_AREA_FUNCTION_PROVIDER_TYPES (electricity-only), so
        # "education" must fall back to the base island method.
    )
    result = updated[0]
    assert result["societal_access_pct__education__total"] == pytest.approx(100.0)
