import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config
from src.damage_recovery import _build_failure_probability, default_fragility_function
from src.dependency_evaluator import (
    activate_delayed_trigger_waits,
    evaluate_dependencies,
    evaluate_dependencies_from_graph,
)
from src.dependency_knowledge_graph import (
    DependencyKnowledgeGraph,
    build_default_knowledge_graph,
)
from src.simulation import (
    SimulationState,
    _initialize_simulation,
    _update_operational_state,
    _update_repair_progress,
)
from src.utils import build_service_area_map_from_rules, build_voronoi_service_area_map

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


def test_area_dependency_blocks_assets_supplied_by_failed_asset():
    operational, report = evaluate_dependencies(
        np.array([False, True, True], dtype=bool),
        np.array(["msls", "hospital", "hospital"]),
        area_dependencies=[
            {
                "supplier_index": 0,
                "dependent_indices": [1, 2],
            }
        ],
        enable_default_rules=False,
        return_report=True,
    )

    assert operational.tolist() == [False, False, False]
    assert report["area_blocked_count"] == 2


def test_pairwise_dependency_blocks_dependent_of_failed_asset():
    operational, report = evaluate_dependencies(
        np.array([False, True], dtype=bool),
        np.array(["msls", "hospital"]),
        pairwise_dependencies=[(0, 1)],
        enable_default_rules=False,
        return_report=True,
    )

    assert operational.tolist() == [False, False]
    assert report["pairwise_blocked_count"] == 1


def test_area_dependency_observes_same_timestep_default_blocking():
    operational = evaluate_dependencies(
        np.array([True, True], dtype=bool),
        np.array(["road", "hospital"]),
        flooded_mask=np.array([True, False], dtype=bool),
        area_dependencies={0: [1]},
        enable_default_rules=True,
    )

    assert operational.tolist() == [False, False]


def test_dependency_only_outage_restores_after_supplier_recovers():
    dependencies = {0: [1]}
    first_operational, first_report = evaluate_dependencies(
        np.array([False, True], dtype=bool),
        np.array(["msls", "hospital"]),
        area_dependencies=dependencies,
        enable_default_rules=False,
        return_report=True,
    )
    restored, second_report = evaluate_dependencies(
        np.array([True, first_operational[1]], dtype=bool),
        np.array(["msls", "hospital"]),
        area_dependencies=dependencies,
        previous_dependency_blocked_mask=first_report[
            "dependency_blocked_mask"
        ],
        enable_default_rules=False,
        return_report=True,
    )

    assert restored.tolist() == [True, True]
    assert not second_report["dependency_blocked_mask"].any()


def test_mixed_dependency_chain_propagates_in_same_timestep():
    operational = evaluate_dependencies(
        np.array([False, True, True], dtype=bool),
        np.array(["msls", "hospital", "school"]),
        area_dependencies={1: [2]},
        pairwise_dependencies=[(0, 1)],
        enable_default_rules=False,
    )

    assert operational.tolist() == [False, False, False]


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


def test_delayed_trigger_counts_down_without_repair_crew():
    rule = {
        "hazard_type": "flooding",
        "asset_type_a": "hospital",
        "asset_type_b": None,
        "relationship": "direct",
        "parameters": {
            "hazard_blocks_operation": True,
            "return_to_operational": {
                "trigger": "delayed",
                "delay_steps": 2,
                "wait_vector": "dependency_wait",
            },
        },
    }
    kg = DependencyKnowledgeGraph.from_config([rule])
    config = get_config()
    config["dependency_parameters"]["knowledge_graph"] = [rule]
    state = SimulationState(None, 1)
    asset_type = np.array(["hospital"])
    flooded = np.array([True])

    activate_delayed_trigger_waits(
        state.operational,
        asset_type,
        "flooding",
        kg,
        flooded_mask=flooded,
        wait_vectors=state.recovery_wait_vectors,
        active_masks=state.recovery_delay_active,
    )
    _update_repair_progress(state, flooded, elapsed_time=1.0)
    _update_operational_state(
        state, asset_type, flooded, config, repair_threshold=0.0, knowledge_graph=kg
    )

    assert state.recovery_wait_vectors["dependency_wait"].tolist() == [1.0]
    assert not state.repair_crews_assigned[0]
    assert not state.operational[0]

    activate_delayed_trigger_waits(
        state.operational,
        asset_type,
        "flooding",
        kg,
        flooded_mask=np.array([False]),
        wait_vectors=state.recovery_wait_vectors,
        active_masks=state.recovery_delay_active,
    )
    _update_repair_progress(state, np.array([False]), elapsed_time=1.0)
    _update_operational_state(
        state,
        asset_type,
        np.array([False]),
        config,
        repair_threshold=0.0,
        knowledge_graph=kg,
    )

    assert state.recovery_wait_vectors["dependency_wait"].tolist() == [0.0]
    assert state.operational[0]


def test_service_area_rule_does_not_restore_disrupted_supplier():
    kg = DependencyKnowledgeGraph.from_config(
        [
            {
                "hazard_type": "flooding",
                "asset_type_a": "msls",
                "asset_type_b": "hospital",
                "relationship": "service_area",
                "parameters": {
                    "hazard_blocks_operation": False,
                    "return_to_operational": {"trigger": "immediate"},
                },
            }
        ]
    )

    updated = evaluate_dependencies_from_graph(
        np.array([False, True], dtype=bool),
        np.array(["msls", "hospital"]),
        "flooding",
        kg,
        service_area_map={0: [1]},
    )

    assert updated.tolist() == [False, False]


def test_default_graph_propagates_asset_192_failure_to_asset_246():
    num_assets = 248
    operational = np.ones(num_assets, dtype=bool)
    operational[192] = False
    asset_type = np.full(num_assets, "road", dtype=object)
    asset_type[192] = "msls"
    asset_type[245:248] = "hospital"
    repair_time = np.zeros(num_assets, dtype=float)
    repair_time[192] = 10.0

    updated, report = evaluate_dependencies_from_graph(
        operational,
        asset_type,
        "flooding",
        build_default_knowledge_graph(),
        repair_time=repair_time,
        service_area_map={192: [245, 246], 137: [247]},
        return_report=True,
    )

    assert not updated[192]
    assert not updated[246]
    assert updated[247]
    assert report["service_area_blocked_count"] == 2


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


def test_build_service_area_map_from_rules_handles_small_primary_sets():
    gdf_assets = gpd.GeoDataFrame(
        {
            "type": ["msls", "msls", "hospital", "school"],
            "geometry": [
                Point(0, 0),
                Point(10, 0),
                box(-1, -1, 1, 1),
                box(9, -1, 11, 1),
            ],
        },
        crs=CRS,
    )
    rules = [
        {
            "hazard_type": "flooding",
            "asset_type_a": "msls",
            "asset_type_b": "hospital",
            "relationship": "service_area",
            "parameters": {
                "hazard_blocks_operation": False,
                "return_to_operational": {"trigger": "immediate"},
            },
        },
        {
            "hazard_type": "flooding",
            "asset_type_a": "msls",
            "asset_type_b": "school",
            "relationship": "service_area",
            "parameters": {
                "hazard_blocks_operation": False,
                "return_to_operational": {"trigger": "immediate"},
            },
        },
    ]

    service_area_map = build_service_area_map_from_rules(gdf_assets, rules)

    assert service_area_map == {0: [2], 1: [3]}


def test_initialize_simulation_auto_builds_service_area_map(tmp_path):
    gdf_assets = gpd.GeoDataFrame(
        {
            "type": ["msls", "msls", "hospital", "school"],
            "geometry": [
                Point(0, 0),
                Point(10, 0),
                box(-1, -1, 1, 1),
                box(9, -1, 11, 1),
            ],
        },
        crs=CRS,
    )
    rules = [
        {
            "hazard_type": "flooding",
            "asset_type_a": "msls",
            "asset_type_b": "hospital",
            "relationship": "service_area",
            "parameters": {
                "hazard_blocks_operation": False,
                "return_to_operational": {"trigger": "immediate"},
            },
        },
        {
            "hazard_type": "flooding",
            "asset_type_a": "msls",
            "asset_type_b": "school",
            "relationship": "service_area",
            "parameters": {
                "hazard_blocks_operation": False,
                "return_to_operational": {"trigger": "immediate"},
            },
        },
    ]
    config = get_config(root_dir=tmp_path)
    config["simulation_config"]["accessibility_model"] = None
    config["dependency_parameters"]["knowledge_graph"] = rules
    config["dependency_parameters"]["service_area_map"] = None

    init = _initialize_simulation(
        gdf_assets,
        hazard_maps=[],
        recovery_parameters=None,
        root_dir=tmp_path,
        config=config,
        repair_crew_assignment_method="random",
        verbose=False,
    )

    assert init["config"]["dependency_parameters"]["service_area_map"] == {0: [2], 1: [3]}


def test_build_service_area_map_from_rules_falls_back_when_voronoi_is_incomplete():
    gdf_assets = gpd.GeoDataFrame(
        {
            "type": ["msls", "msls", "msls", "msls", "msls", "hospital", "hospital"],
            "geometry": [
                Point(0, 0),
                Point(10, 0),
                Point(0, 10),
                Point(10, 10),
                Point(5, 5),
                box(-1, -1, 1, 1),
                box(9, 9, 11, 11),
            ],
        },
        crs=CRS,
    )
    rules = [
        {
            "hazard_type": "flooding",
            "asset_type_a": "msls",
            "asset_type_b": "hospital",
            "relationship": "service_area",
            "parameters": {
                "hazard_blocks_operation": False,
                "return_to_operational": {"trigger": "immediate"},
            },
        }
    ]

    service_area_map = build_service_area_map_from_rules(gdf_assets, rules)

    assert service_area_map == {0: [5], 3: [6]}
