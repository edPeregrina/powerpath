"""Tests for the config.py wiring of the 3 independent weighted-Voronoi
education functions (education_bo/education_so/education_vso), which
mirror config.py's production taxonomy -> service_area_function_provider_types
construction used by src/simulation.py."""
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import get_config, get_societal_access_config


@pytest.fixture
def pop_grid_gdf():
    return gpd.GeoDataFrame(
        {"cell_id": [0]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:28992"
    )


def _build(pop_grid_gdf, asset_types=None):
    config = get_config()
    return get_societal_access_config(config, pop_grid_gdf, asset_types=asset_types)


def test_school_asset_types_map_to_independent_education_functions(pop_grid_gdf):
    sa_cfg = _build(pop_grid_gdf)
    taxonomy = sa_cfg["taxonomy"]
    assert taxonomy["school"] == "education_bo"
    assert taxonomy["school_so"] == "education_so"
    assert taxonomy["school_vso"] == "education_vso"
    # 3 independent functions, not one shared pool -- they don't compete for pupils.
    assert len({taxonomy["school"], taxonomy["school_so"], taxonomy["school_vso"]}) == 3


def test_school_functions_wired_as_weighted_voronoi_with_distinct_scale_factors(pop_grid_gdf):
    sa_cfg = _build(pop_grid_gdf)
    provider_cfg = sa_cfg["service_area_function_provider_types"]

    for function_name, asset_type in (
        ("education_bo", "school"),
        ("education_so", "school_so"),
        ("education_vso", "school_vso"),
    ):
        cfg = provider_cfg[function_name]
        assert cfg["method"] == "weighted_voronoi"
        assert cfg["provider_types"] == {asset_type}
        assert cfg["weight_column"] == "pupils"
        assert isinstance(cfg["weight_scale_factor"], float)

    scale_factors = {provider_cfg[f]["weight_scale_factor"] for f in ("education_bo", "education_so", "education_vso")}
    assert len(scale_factors) == 3  # each bucket gets its own, not a shared default

    # electricity keeps its pre-existing simple_voronoi wiring, unaffected by schools.
    assert provider_cfg["electricity"] == {"provider_types": {"msls"}, "method": "simple_voronoi"}


def test_school_functions_dropped_when_asset_type_not_requested(pop_grid_gdf):
    """asset_types filtering (used to restrict the taxonomy to asset types
    actually present in a run's data) must also drop the corresponding
    service_area_function_provider_types entry, not leave a dangling one."""
    sa_cfg = _build(pop_grid_gdf, asset_types={"hospital"})
    assert "school" not in sa_cfg["taxonomy"]
    assert "education_bo" not in sa_cfg["service_area_function_provider_types"]
    assert "education_so" not in sa_cfg["service_area_function_provider_types"]
    assert "education_vso" not in sa_cfg["service_area_function_provider_types"]
    # electricity is always kept regardless of asset_types.
    assert "electricity" in sa_cfg["service_area_function_provider_types"]
