import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.damage_recovery import _build_failure_probability, default_fragility_function
from src.dependency_evaluator import (
    evaluate_dependencies,
    evaluate_dependencies_from_graph,
)
from src.dependency_knowledge_graph import DependencyKnowledgeGraph
from src.utils import build_voronoi_service_area_map

CRS = "EPSG:28992"


def test_legacy_default_rules_only_block_flooded_roads():
    operational, report = evaluate_dependencies(
        np.array([True, True], dtype=bool),
        np.array(["road", "msls"]),
        flooded_mask=np.array([True, True], dtype=bool),
        enable_default_rules=True,
        return_report=True,
    )

    assert operational.tolist() == [False, True]
    assert report["blocked_count"] == 1
    assert report["active_rules"] == ["road:flooded"]


def test_legacy_disabled_rules_emit_warning():
    _, report = evaluate_dependencies(
        np.array([True], dtype=bool),
        np.array(["road"]),
        flooded_mask=np.array([True], dtype=bool),
        enable_default_rules=False,
        return_report=True,
    )

    assert "warning" in report


def test_repair_below_threshold_restores_at_threshold():
    kg = DependencyKnowledgeGraph.from_config(
        [
            {
                "hazard_type": "flooding",
                "asset_type_a": "ls",
                "asset_type_b": None,
                "relationship": "direct",
                "parameters": {
                    "hazard_blocks_operation": False,
                    "return_to_operational": {"trigger": "repair_below", "threshold": 2.0},
                },
            }
        ]
    )

    updated = evaluate_dependencies_from_graph(
        np.array([False], dtype=bool),
        np.array(["ls"]),
        "flooding",
        kg,
        flooded_mask=np.array([False], dtype=bool),
        repair_time=np.array([2.0], dtype=float),
    )

    assert updated.tolist() == [True]


def test_delayed_trigger_uses_named_wait_vector():
    kg = DependencyKnowledgeGraph.from_config(
        [
            {
                "hazard_type": "flooding",
                "asset_type_a": "hospital",
                "asset_type_b": None,
                "relationship": "direct",
                "parameters": {
                    "hazard_blocks_operation": False,
                    "return_to_operational": {
                        "trigger": "delayed",
                        "delay_steps": 2,
                        "wait_vector": "dependency_wait",
                    },
                },
            }
        ]
    )

    still_blocked = evaluate_dependencies_from_graph(
        np.array([False], dtype=bool),
        np.array(["hospital"]),
        "flooding",
        kg,
        flooded_mask=np.array([False], dtype=bool),
        repair_time=np.array([0.0], dtype=float),
        wait_vectors={"dependency_wait": np.array([1.0], dtype=float)},
    )
    restored = evaluate_dependencies_from_graph(
        np.array([False], dtype=bool),
        np.array(["hospital"]),
        "flooding",
        kg,
        flooded_mask=np.array([False], dtype=bool),
        repair_time=np.array([0.0], dtype=float),
        wait_vectors={"dependency_wait": np.array([0.0], dtype=float)},
    )

    assert still_blocked.tolist() == [False]
    assert restored.tolist() == [True]


def test_probability_curve_fragility_model_supported():
    probability = _build_failure_probability(
        np.array([0.0, 0.5, 1.0], dtype=float),
        {
            "mode": "probability_curve",
            "intensity_values": [0.0, 1.0],
            "failure_probabilities": [0.0, 1.0],
        },
        major_timestep=24,
    )

    assert np.allclose(probability, [0.0, 0.5, 1.0])


def test_fragility_exclusions_skip_selected_asset_type():
    result = default_fragility_function(
        np.array([1.0, 1.0], dtype=float),
        np.array(["hospital", "hospital"]),
        major_timestep=24,
        fragility_models={
            "hospital": {
                "mode": "probability_curve",
                "intensity_values": [0.0, 1.0],
                "failure_probabilities": [1.0, 1.0],
            }
        },
        fragility_exclusions={"hospital": True},
    )

    assert result.tolist() == [1, 1]


def test_voronoi_service_area_map_can_return_diagnostics():
    voronoi_gdf = gpd.GeoDataFrame(
        {
            "asset_id": [0, 1],
            "geometry": [box(0, 0, 10, 10), box(10, 0, 20, 10)],
        },
        crs=CRS,
    )
    secondary = gpd.GeoDataFrame(
        {"geometry": [box(1, 1, 4, 4), Point(30, 30)]},
        index=[100, 101],
        crs=CRS,
    )

    service_area_map, diagnostics = build_voronoi_service_area_map(
        voronoi_gdf,
        secondary,
        return_diagnostics=True,
    )

    assert service_area_map[0] == [100]
    assert diagnostics["resolved_by_overlap"] == 1
    assert diagnostics["resolved_by_nearest"] == 1
    assert diagnostics["unresolved"] == 0
