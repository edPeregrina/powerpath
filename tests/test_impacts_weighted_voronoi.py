"""Smoke/regression tests for `create_weighted_voronoi_for_asset_type` (the
power-diagram primitive backing the schools-weighted-voronoi "weighted_voronoi"
service-area method). Exercises a dummy 100/1000/10000 pupil-count scenario
with weight-scale factors realistic for this geometry (schools ~500m apart in
EPSG:28992, so dist^2 is on the order of 10^5), checking three behaviours:

1. `weight_scale_factor=0` must degenerate to the unweighted Voronoi partition
   (equal-ish cell areas), confirming the weighting term is additive and
   neutral at zero.
2. A realistic non-zero scale factor must produce a *visible* monotonic
   ordering of cell areas (more pupils -> more area), without fully
   swallowing any site.
3. A deliberately large scale factor must demonstrate the known, accepted
   power-diagram phenomenon where a low-weight site's cell becomes **empty**
   (fully dominated by a higher-weight neighbour) -- and that this is
   reported via `diagnostics["empty_providers"]`, not a crash or silent zero.
"""
import geopandas as gpd
import pytest
from shapely.geometry import Point

from src.impacts import create_weighted_voronoi_for_asset_type

PUPIL_COUNTS = [100, 1000, 10000]
# Schools in a roughly equilateral triangle, ~500m apart -- a realistic
# neighbourhood-scale distance in EPSG:28992 (meters), giving dist^2 on the
# order of 2.5e5 m^2 between neighbours. A collinear layout was tried first
# and rejected: it hits a scipy.spatial.QhullError (degenerate ConvexHull on
# near-collinear intersection vertices) and produces asymmetric domination
# patterns that don't reflect a realistic 2D catchment. Weight-scale factors
# below were empirically tuned against *this* geometry and will need
# re-tuning if this geometry changes.
SCHOOL_POINTS = [Point(0, 0), Point(500, 0), Point(250, 450)]


def _make_schools(pupil_counts=PUPIL_COUNTS, points=SCHOOL_POINTS):
    return gpd.GeoDataFrame(
        {"type": ["school"] * len(points), "pupil_count": pupil_counts},
        geometry=points,
        crs="EPSG:28992",
    )


def test_weight_scale_factor_zero_degenerates_to_unweighted_positions():
    schools = _make_schools()
    # asset_id 0 (100 pupils) and asset_id 1 (1000 pupils) sit symmetrically
    # about the triangle's x=250 axis of symmetry (asset_id 2 sits on that
    # axis). Despite wildly different pupil counts, weight_scale_factor=0
    # must null out the weighting term entirely, so their cell areas should
    # be governed purely by position and come out equal -- confirming the
    # weighting is additive/neutral at zero rather than, say, a multiplier
    # that could leave a residual effect even at "zero".
    #
    # (create_voronoi_for_asset_type -- the plain-Voronoi primitive -- can't
    # be used as a reference here: with exactly 3 non-collinear points every
    # scipy.spatial.Voronoi region is unbounded, and that function's
    # `-1 in region` guard drops all three, returning an empty GeoDataFrame.
    # That's a pre-existing limitation of the unweighted primitive, out of
    # scope for this branch.)
    weighted = create_weighted_voronoi_for_asset_type(
        schools, "school", weight_column="pupil_count", weight_scale_factor=0.0, use_cache=False,
    )

    areas = weighted.set_index("asset_id").geometry.area
    assert areas.loc[0] == pytest.approx(areas.loc[1], rel=1e-6)


def test_moderate_scale_factor_gives_visible_monotonic_weighting():
    schools = _make_schools()
    # 1e5 is below the ~1.8e5 threshold (for this geometry) at which the
    # 100-pupil school starts getting swallowed -- all three cells stay
    # non-empty while still showing a clear monotonic area ordering.
    weighted, diagnostics = create_weighted_voronoi_for_asset_type(
        schools,
        "school",
        weight_column="pupil_count",
        weight_scale_factor=1.0e5,
        use_cache=False,
        return_diagnostics=True,
    )

    assert diagnostics["empty_providers"] == []
    areas = weighted.set_index("asset_id").geometry.area
    # asset_id 0/1/2 correspond to pupil counts 100/1000/10000 respectively.
    assert areas.loc[0] < areas.loc[1] < areas.loc[2]


def test_large_scale_factor_can_swallow_low_weight_site():
    schools = _make_schools()
    # 1.8e5 (for this geometry) fully dominates the 100-pupil school (its
    # cell vanishes) while the 1000-pupil school still retains a sliver --
    # demonstrating the known, accepted power-diagram "swallowed site"
    # phenomenon is handled gracefully (surfaced via diagnostics) rather
    # than crashing or silently dropping every low-weight provider at once.
    weighted, diagnostics = create_weighted_voronoi_for_asset_type(
        schools,
        "school",
        weight_column="pupil_count",
        weight_scale_factor=1.8e5,
        use_cache=False,
        return_diagnostics=True,
    )

    assert diagnostics["empty_providers"] == [0]
    assert 0 not in set(weighted["asset_id"])
    assert set(weighted["asset_id"]) == {1, 2}


def test_exclude_column_removes_remote_school_from_tessellation():
    schools = _make_schools()
    schools["school_remote"] = [False, False, True]

    weighted = create_weighted_voronoi_for_asset_type(
        schools,
        "school",
        weight_column="pupil_count",
        weight_scale_factor=1.0e5,
        exclude_column="school_remote",
        use_cache=False,
    )

    assert 2 not in set(weighted["asset_id"])
    assert set(weighted["asset_id"]) <= {0, 1}
