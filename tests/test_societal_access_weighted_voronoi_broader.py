"""Broader regression tests for the schools-weighted-voronoi work:

1. Weighted allocation correctness across *multiple* population cells (not
   just the single straddling-cell geometry used by
   ``tests/test_societal_access_weighted_voronoi.py``), and remote-fraction
   accumulation across *multiple* ``provider_types`` mapping to the same
   function.
2. Empty-cell/swallowed-site handling at the ``societal_access.py`` level
   (the primitive-level behaviour is already covered by
   ``tests/test_impacts_weighted_voronoi.py``; this confirms the full
   pipeline degrades gracefully rather than crashing/NaN-ing).
3. ``catchment_assignment_mode="dynamic_reachable"`` is explicitly
   unimplemented and must raise ``NotImplementedError`` rather than
   silently behaving like ``"fixed"``.
4. An explicit regression proving all three service-area methods (base
   island method for hospitals, ``simple_voronoi`` for electricity,
   ``weighted_voronoi`` for schools) coexist correctly within a single
   ``postprocess_societal_access_results`` call without interfering with
   each other.
"""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from src.societal_access import (
    _build_weighted_voronoi_cell_assignments,
    get_or_build_allocation,
    postprocess_societal_access_results,
)

# Same triangle geometry as tests/test_impacts_weighted_voronoi.py (kept in
# sync deliberately -- see that module's docstring for why collinear layouts
# were rejected and where these weight-scale-factor thresholds come from).
PUPIL_COUNTS = [100, 1000, 10000]
SCHOOL_POINTS = [Point(0, 0), Point(500, 0), Point(250, 450)]
_MODERATE_SCALE = 1.0e5  # all three sites stay non-empty (monotonic areas)
_SWALLOWING_SCALE = 1.8e5  # fully swallows the 100-pupil site (asset_id 0)


def _make_three_schools(weight_column="pupil_count"):
    return gpd.GeoDataFrame(
        {"type": ["school"] * 3, weight_column: PUPIL_COUNTS},
        geometry=SCHOOL_POINTS,
        crs="EPSG:28992",
    )


def _make_pop_cells(cell_ids, points):
    return gpd.GeoDataFrame(
        {
            "cell_id": cell_ids,
            "aantal_inwoners": [100] * len(points),
            "aantal_inwoners_65_jaar_en_ouder": [20] * len(points),
            "aantal_inwoners_0_tot_15_jaar": [15] * len(points),
        },
        # Small (2x2) boxes centred on each point rather than bare points --
        # build_voronoi_service_area_map operates on polygon geometries.
        geometry=[box(x - 1, y - 1, x + 1, y + 1) for x, y in points],
        crs="EPSG:28992",
    )


def test_weighted_allocation_assigns_each_cell_to_its_own_nearby_school():
    """A population cell placed at the *representative point* of a given
    school's own power-diagram cell (computed independently via
    ``create_weighted_voronoi_for_asset_type`` at the same scale factor)
    must be catchment-assigned to that school.

    Note this deliberately does NOT use points merely "near" each school's
    site: with a large pupil-count imbalance even a moderate scale factor
    can shift a site's cell well away from its own location (see
    ``_WEIGHT_SCALE_FACTOR``/geometry tuning note in
    ``tests/test_societal_access_weighted_voronoi.py`` -- a cell literally
    1 unit from the low-pupil school can still end up dominated by the
    high-pupil school). Representative points are the only way to pick
    points guaranteed to fall inside each site's own cell for an arbitrary
    scale factor."""
    schools = _make_three_schools()
    # Representative (interior) points of each school's own power-diagram
    # cell at _MODERATE_SCALE, precomputed via create_weighted_voronoi_for_
    # asset_type with the same inputs -- see test module docstring.
    rep_points = [
        (-57.83820658926969, -46.512936549461244),
        (533.5138822649454, -46.51293654946122),
        (250.0, 331.3649452362364),
    ]
    pop = _make_pop_cells(["near_0", "near_1", "near_2"], rep_points)
    config = {
        "education": {
            "provider_types": {"school"},
            "method": "weighted_voronoi",
            "weight_column": "pupil_count",
            "weight_scale_factor": _MODERATE_SCALE,
        }
    }
    cell_assignments, remote_fraction = _build_weighted_voronoi_cell_assignments(
        gdf_assets=schools,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        service_area_function_provider_types=config,
    )

    assignment = cell_assignments["education"]
    assert assignment["near_0"] == schools.index[0]
    assert assignment["near_1"] == schools.index[1]
    assert assignment["near_2"] == schools.index[2]
    assert remote_fraction["education"] == pytest.approx(0.0)


def test_remote_fraction_accumulates_across_multiple_provider_types():
    """When more than one ``provider_type`` maps to the same function (e.g.
    ``school`` + ``school_secondary`` both -> "education", mirroring the
    real taxonomy's school/basisonderwijs/voortgezet_onderwijs -> education
    collapse), ``remote_fraction`` must sum numerator and denominator
    across *all* of them before dividing once -- not average per-type
    fractions or only account for the first type encountered."""
    assets = gpd.GeoDataFrame(
        {
            "type": ["school", "school", "school_secondary", "school_secondary"],
            "pupil_count": [10, 90, 100, 200],
            "is_remote": [False, True, False, True],
        },
        geometry=[Point(0, 0), Point(500, 0), Point(1000, 0), Point(1500, 0)],
        crs="EPSG:28992",
    )
    pop = _make_pop_cells(["c0"], [(0, 0)])
    config = {
        "education": {
            "provider_types": {"school", "school_secondary"},
            "method": "weighted_voronoi",
            "weight_column": "pupil_count",
            "weight_scale_factor": 1.0,
            "exclude_column": "is_remote",
        }
    }
    _cell_assignments, remote_fraction = _build_weighted_voronoi_cell_assignments(
        gdf_assets=assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        service_area_function_provider_types=config,
    )

    # total weight = 10 + 90 + 100 + 200 = 400; remote weight = 90 + 200 = 290
    assert remote_fraction["education"] == pytest.approx(290.0 / 400.0)


def test_swallowed_site_end_to_end_access_still_well_defined():
    """At the swallowing scale factor (1.8e5 -- see
    ``test_large_scale_factor_can_swallow_low_weight_site`` in
    tests/test_impacts_weighted_voronoi.py), the 100-pupil school's cell
    vanishes entirely. A population cell sitting right next to that now-
    swallowed school must still get a well-defined (non-NaN, non-crash)
    catchment assignment to one of the surviving providers, and end-to-end
    access must reflect that surviving provider's operational/island state
    -- not silently drop the cell or error out."""
    schools = _make_three_schools()
    pop = _make_pop_cells(["near_swallowed"], [(1, 0)])
    config = {
        "education": {
            "provider_types": {"school"},
            "method": "weighted_voronoi",
            "weight_column": "pupil_count",
            "weight_scale_factor": _SWALLOWING_SCALE,
        }
    }

    cell_assignments, _remote_fraction = _build_weighted_voronoi_cell_assignments(
        gdf_assets=schools,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        service_area_function_provider_types=config,
    )
    assigned_provider = cell_assignments["education"]["near_swallowed"]
    # The swallowed (100-pupil, asset_id 0) school must never receive the
    # assignment -- it has no cell territory left to claim.
    assert assigned_provider != schools.index[0]
    assert assigned_provider in (schools.index[1], schools.index[2])

    single_island = gpd.GeoDataFrame(
        {"island_id": [1]}, geometry=[box(-10, -10, 600, 500)], crs="EPSG:28992"
    )
    allocation_cache = {}
    get_or_build_allocation(
        allocation_cache, pop, "cell_id", single_island, road_state_key="swallowed_connected"
    )
    operational = np.array([True, True, True])
    island_ids = np.array([1, 1, 1])
    updated, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "swallowed_connected",
            "operational": operational,
            "island_id": island_ids,
        }],
        gdf_assets=schools,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"school": "education"},
        all_functions=["education"],
        service_area_function_provider_types=config,
    )
    result = updated[0]
    assert result["societal_access_pct__education__total"] == pytest.approx(100.0)

    # Now take the *assigned* (surviving) provider offline: access for this
    # cell must drop to 0% -- proving the swallowed geometry still routes
    # through the normal operational/island gating, it didn't just get
    # silently defaulted to "always accessible".
    operational_down = operational.copy()
    operational_down[schools.index.get_loc(assigned_provider)] = False
    updated_down, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "swallowed_connected",
            "operational": operational_down,
            "island_id": island_ids,
        }],
        gdf_assets=schools,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"school": "education"},
        all_functions=["education"],
        service_area_function_provider_types=config,
    )
    assert updated_down[0]["societal_access_pct__education__total"] == pytest.approx(0.0)


def test_dynamic_reachable_mode_raises_not_implemented():
    """``catchment_assignment_mode="dynamic_reachable"`` must fail loudly,
    not silently behave like ``"fixed"`` (a real correctness risk since
    nothing in ``_build_weighted_voronoi_cell_assignments`` otherwise
    branches on this value)."""
    schools = _make_three_schools()
    pop = _make_pop_cells(["c0"], [(1, 0)])
    config = {
        "education": {
            "provider_types": {"school"},
            "method": "weighted_voronoi",
            "weight_column": "pupil_count",
            "catchment_assignment_mode": "dynamic_reachable",
        }
    }
    with pytest.raises(NotImplementedError, match="dynamic_reachable"):
        _build_weighted_voronoi_cell_assignments(
            gdf_assets=schools,
            pop_grid_gdf=pop,
            cell_id_column="cell_id",
            service_area_function_provider_types=config,
        )


def test_hospital_electricity_and_school_methods_coexist_independently():
    """Single ``postprocess_societal_access_results`` call mixing all three
    formalised service-area methods at once: hospital (no entry -> base
    island/shared-reachability method), electricity (``simple_voronoi``
    over msls stations, the pre-existing exception), and school
    (``weighted_voronoi``, this branch's new addition). Each function's
    access percentage must reflect only its own method/providers and must
    not be perturbed by the other two being present in the same call."""
    assets = gpd.GeoDataFrame(
        {
            "type": ["hospital", "msls", "school", "school"],
            "pupil_count": [np.nan, np.nan, 10, 100000],
        },
        geometry=[Point(45, 0), Point(45, 0), Point(0, 0), Point(100, 0)],
        crs="EPSG:28992",
    )
    pop = _make_pop_cells(["c0"], [(45, 0)])
    config = {
        "electricity": {"provider_types": {"msls"}, "method": "simple_voronoi"},
        "education": {
            "provider_types": {"school"},
            "method": "weighted_voronoi",
            "weight_column": "pupil_count",
            "weight_scale_factor": 1000.0,
        },
    }
    single_island = gpd.GeoDataFrame(
        {"island_id": [1]}, geometry=[box(-10, -10, 110, 10)], crs="EPSG:28992"
    )
    allocation_cache = {}
    get_or_build_allocation(
        allocation_cache, pop, "cell_id", single_island, road_state_key="mixed_methods"
    )

    # hospital operational, msls operational, school A (10 pupils, x=0) down,
    # school B (100000 pupils, x=100) up -> at x=45 the weighted power-diagram
    # bisector (same tuning as tests/test_societal_access_weighted_voronoi.py)
    # catchment-assigns the cell to the high-pupil school B, so hospital/
    # electricity/education should all read 100%.
    updated, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "mixed_methods",
            "operational": np.array([True, True, False, True]),
            "island_id": np.array([1, 1, 1, 1]),
        }],
        gdf_assets=assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"hospital": "hospital", "msls": "electricity", "school": "education"},
        all_functions=["hospital", "electricity", "education"],
        service_area_function_provider_types=config,
    )
    result = updated[0]
    assert result["societal_access_pct__hospital__total"] == pytest.approx(100.0)
    assert result["societal_access_pct__electricity__total"] == pytest.approx(100.0)
    assert result["societal_access_pct__education__total"] == pytest.approx(100.0)

    # Now take the msls station down: electricity must drop to 0%, while
    # hospital and education -- unaffected by msls -- must be untouched.
    updated_msls_down, _ = postprocess_societal_access_results(
        summary_results=[{"timestep": 0, "map": 0}],
        detailed_results=[{
            "timestep": 0,
            "map": 0,
            "road_state_key": "mixed_methods",
            "operational": np.array([True, False, False, True]),
            "island_id": np.array([1, 1, 1, 1]),
        }],
        gdf_assets=assets,
        pop_grid_gdf=pop,
        cell_id_column="cell_id",
        allocation_cache=allocation_cache,
        taxonomy={"hospital": "hospital", "msls": "electricity", "school": "education"},
        all_functions=["hospital", "electricity", "education"],
        service_area_function_provider_types=config,
    )
    result_down = updated_msls_down[0]
    assert result_down["societal_access_pct__hospital__total"] == pytest.approx(100.0)
    assert result_down["societal_access_pct__electricity__total"] == pytest.approx(0.0)
    assert result_down["societal_access_pct__education__total"] == pytest.approx(100.0)
