import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from src.adaptation import simulate_asset_damage_recovery_access_breakdown_ema


def test_ema_wrapper_emits_configured_societal_metrics_when_missing(monkeypatch):
    def _fake_simulation(*args, **kwargs):
        timestep_summary = [{
            "timestep": 0,
            "map": 0,
            "operational_count": 1,
            "accessible_count": 1,
            "unreachable_count": 0,
            "flooded_count": 0,
            "crews_assigned_count": 0,
            "avg_damage_ratio": 0.0,
            "avg_repair_time": 0.0,
        }]
        return [(1, timestep_summary, [{}])], {}, {}

    monkeypatch.setattr(
        "src.simulation.simulate_asset_damage_recovery_access_breakdown",
        _fake_simulation,
    )

    assets = gpd.GeoDataFrame(
        {"type": ["msls"]},
        geometry=[Point(0, 0)],
        crs="EPSG:28992",
    )

    result = simulate_asset_damage_recovery_access_breakdown_ema(
        gdf_assets=assets,
        hazard_maps=[None],
        timestep_output=True,
        asset_population_map={},
        asset_to_lu={},
        societal_access_config={
            "pop_group_columns": {
                "total": "aantal_inwoners",
                "elderly": "aantal_inwoners_65_jaar_en_ouder",
                "children": "aantal_inwoners_0_tot_15_jaar",
            },
            "all_functions": ["electricity"],
            "reference_group": "total",
        },
    )

    assert "societal_access_pct__electricity__total" in result
    assert "societal_total_population__total" in result
    assert "societal_equity_absolute_gap__electricity__elderly" in result
    assert result["societal_access_pct__electricity__total"].shape == (1,)
    assert np.isnan(result["societal_access_pct__electricity__total"][0])
