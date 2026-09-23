"""
PowerPath Infrastructure Resilience Analysis Configuration
==========================================================

This module contains all configuration parameters for the electricity infrastructure
damage recovery and accessibility analysis simulation.

Usage:
    from config import get_config
    config = get_config()
"""

from pathlib import Path
import os
import shutil

from src.dependency_knowledge_graph import build_default_knowledge_graph


def get_config(root_dir=None, hazard_dir_override=None):
    """
    Get configuration dictionary with all simulation parameters.
    
    Args:
        root_dir (Path, optional): Override for root directory. Defaults to repository root.
        hazard_dir_override (Path, optional): Override for hazard data directory.
    
    Returns:
        dict: Configuration dictionary containing all simulation parameters
    """
    
    # Determine root directory
    if root_dir is None:
        # Try to find repository root by looking for common repo files
        current_path = Path(__file__).parent
        while current_path != current_path.parent:
            if any((current_path / marker).exists() for marker in ['.git', 'README.md', 'setup.py', 'pyproject.toml']):
                root_dir = current_path
                break
            current_path = current_path.parent
        
        # Fallback to parent of config file location
        if root_dir is None:
            root_dir = Path(__file__).parent
    else:
        root_dir = Path(root_dir)
    
    # Base configuration
    config = {
        'root_dir': root_dir,
        'data_dir': root_dir / 'raw_data/ZH_Delfland',
        'electricity_dir': root_dir / 'raw_data/ZH_Delfland/electricity',

        # Simulation configuration
        'simulation_config': {
            # number_repair_crews supports:
            #   - scalar global crews:          10
            #   - island-keyed crews:           {1: 3, 2: 5}
            #   - global type-specific crews:   {"msls": 4, "hospital": 2}
            #   - island + type-specific crews: {1: {"msls": 3, "hospital": 1},
            #                                    2: {"msls": 2, "hospital": 4}}
            # An optional "*" key inside a type-specific dict defines an
            # explicit default/untyped pool; asset types not covered by any
            # type-specific pool otherwise receive zero crews.
            'number_repair_crews': 20,
            'repair_crew_assignment_method': 'islands',  # Options: 'islands', 'islands lowest repair time', 'lowest repair time', 'highest repair time', 'random'
            'flood_threshold': 0.2,
            'verbose': True,
            'accessibility_model': None  # Default to None, can be set to grid_hex.accessibility_model
        },
        
        # Recovery model parameters
        'recovery_parameters': {
            'repair_time_coefficients': [702.72, 3.14, 1.9891],  # [a, b, c] for quadratic: repair_time = a*DR² + b*DR + c
            'damage_ratio_coefficients': (0.0468, 0.0077),  # (m, n) for linear: damage_ratio = m*hazard + n
            # Optional per-asset fragility overrides:
            #   {'hospital': {'mode': 'probability_curve',
            #                 'intensity_values': [0.0, 0.5, 1.0],
            #                 'failure_probabilities': [0.0, 0.4, 1.0]}}
            'fragility_models': {},
            # 'time_step_hours': 1,
            'damage_threshold': 0.01,    # Minimum damage ratio to consider asset damaged
            'repair_threshold': 2.0      # Minimum repair time threshold for repairable assets
        },
        
        # Analysis configuration
        'analysis_config': {
            'hazard_extraction_method': 'max',  # Options: 'max', 'mean', 'median', 'centroid'
            'max_simulation_days': None,  # None for all available days, integer to limit
            'cache_enabled': True,        # Enable/disable result caching for performance
            'performance_monitoring': False  # Enable detailed performance monitoring
        },

        # Service-node configuration for societal access analysis.
        #
        # 'taxonomy'         : mapping of node-type string → function-category label.
        #                      Add entries for new service types without touching the
        #                      graph or metrics code.
        # 'population_groups': mapping of display label → CBS population column name.
        #                      Extend with any column present in the population grid.
        #                      Reference: CBS, "Statistische gegevens per vierkant en
        #                      postcode 2022, 2023, 2024 – Beschrijving cijfers"
        #                      https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024/4-beschrijving-cijfers
        # 'reference_group'  : the population group used as the baseline when computing
        #                      equity gaps (must be a key in 'population_groups').
        # 'service_area_function_provider_types' (optional):
        #                      override for function categories that should use
        #                      service-area/Voronoi assignment instead of island
        #                      connectivity, e.g.
        #                      {'electricity': frozenset({'msls'}),
        #                       'health': frozenset({'hospital', 'clinic'})}
        #                      Provider type strings are exact tokens and must
        #                      match asset "type" values exactly.
        #                      If omitted or None, defaults from
        #                      src.societal_access.SERVICE_AREA_FUNCTION_PROVIDER_TYPES
        #                      are used.
        'service_node_config': {
            'taxonomy': {
                # Health
                'hospital': 'health',
                'clinic': 'health',
                'doctors': 'health',
                'apotheek': 'health',
                # Emergency response
                'fire_station': 'emergency_response',
                'brandweerkazerne': 'emergency_response',
                'emergency_operations_centre': 'emergency_response',
                # Climate resilience
                'cooling_centre': 'climate_resilience',
                'water_supply_point': 'climate_resilience',
                'drinking_water': 'climate_resilience',
                # Social continuity
                'school': 'education',
                'basisonderwijs': 'education',
                'voortgezet_onderwijs': 'education',
                'repair_depot': 'repair_logistics',
            },
            'population_groups': {
                'total':       'aantal_inwoners',
                'elderly':     'aantal_inwoners_65_jaar_en_ouder',
                'children':    'aantal_inwoners_0_tot_15_jaar',
                'working_age': 'aantal_inwoners_25_tot_45_jaar',
            },
            'reference_group': 'total',
        },

        # Dependency parameters -- type-level knowledge-graph rules (see
        # src.dependency_knowledge_graph) plus the runtime knobs that consume
        # them in the simulation loop.
        #
        # 'hazard_type' : the active hazard (e.g. "flooding") used to look up
        #                 hazard-availability rules from 'knowledge_graph'.
        #
        # 'knowledge_graph' : a list of type-level rule dicts (asset *types*
        #                 only -- never asset IDs/indices), each either:
        #                   {"relation": "hazard", "hazard_type": ..., "source_type": ...,
        #                    "hazard_blocks_operation": bool, "return_to_operational": {...}}
        #                 or:
        #                   {"relation": "dependency", "source_type": ..., "target_type": ...,
        #                    "topology": "direct" | "voronoi" | "radius",
        #                    "availability_policy": "exclusive" | "any" | "at_least_n",
        #                    "radius_m": float (radius topology only),
        #                    "minimum_available": int (at_least_n only)}
        #                 Defaults to
        #                 ``src.dependency_knowledge_graph.build_default_knowledge_graph()``
        #                 -- the msls/ms/ls/hospital return-to-operational rules
        #                 plus a msls -> hospital direct dependency rule -- unless
        #                 explicitly overridden. Pass an explicit list (including
        #                 ``[]`` to opt out entirely) to take control of the rules
        #                 yourself; see ``book/Use_Case_sample_knowledge_graph.ipynb``
        #                 for a worked example. NOTE: the built-in default rule uses
        #                 topology="direct" for msls -> hospital, which requires at
        #                 most one msls asset; datasets with multiple substations
        #                 must override this rule with topology="voronoi" or
        #                 "radius" (or supply precomputed 'dependency_edges').
        #
        # 'dependency_edges' : optional caller-precomputed list of runtime
        #                 :class:`~src.dependency_topology.DependencyEdge`
        #                 instances. When left as ``None`` (the default), the
        #                 simulation expands 'knowledge_graph' against
        #                 gdf_assets once via
        #                 :func:`src.dependency_topology.expand_dependency_edges`.
        #                 When provided, this precomputed value is preserved
        #                 and never overwritten by the simulation.
        #
        # 'dependency_restart_delay_steps' : number of simulation steps a
        #                 restored dependency must remain continuously
        #                 available before the assets it gates resume
        #                 operating (see restart_wait::<edge_key> in
        #                 src.dependency_evaluator). Kept separate from
        #                 hazard-recovery waits.
        #
        # Road availability remains governed exclusively by the existing
        # road-graph exposure filtering and is not part of this dependency model.
        # Substation/hospital structural damage remains governed by fragility
        # and repair completion in the simulation loop; dependency blocking
        # never mutates damage, repair time, or fragility state.
        'dependency_parameters': {
            'hazard_type': 'flooding',  # active hazard type for graph look-up
            'knowledge_graph': build_default_knowledge_graph().to_config(),
            'dependency_edges': None,
            'dependency_restart_delay_steps': 0.0,
        }
    }
    
    # Set hazard directory with override capability
    if hazard_dir_override:
        config['hazard_dir'] = Path(hazard_dir_override)
    else:
        # Default hazard directory paths (in order of preference)
        hazard_dir_options = [
            root_dir / 'raw_data' / 'ZH_Delfland_interpolated_timesteps_tif' / 'hazard_maps_ZH_Delfland',
            root_dir / 'data' / 'static' / 'hazard' / 'processed',
            root_dir / 'data' / 'hazard',
            root_dir / 'hazard_data'
        ]
        
        # Use the first existing directory
        config['hazard_dir'] = None
        for hazard_path in hazard_dir_options:
            if hazard_path.exists():
                config['hazard_dir'] = hazard_path
                break
        
        # If no existing directory found, use the primary option
        if config['hazard_dir'] is None:
            config['hazard_dir'] = hazard_dir_options[0]
    
    # Create hazard-specific directory structure
    hazard_dir_name = config['hazard_dir'].name
    config['interim_dir'] = config['root_dir'] / 'data' / 'interim' / f'interim_{hazard_dir_name}'
    config['output_dir'] = config['root_dir'] / 'data' / 'output' / f'output_{hazard_dir_name}'
    
    return config


def _validate_number_repair_crews(number_repair_crews):
    """Validate the shape of ``simulation_config['number_repair_crews']``.

    Mirrors the shape rules enforced at runtime by
    ``src.simulation._normalize_number_repair_crews_config`` (scalar,
    island-keyed dict, type-specific dict, or island+type nested dict), but
    is intentionally self-contained here to avoid a config<->simulation
    circular import. Returns a list of human-readable error strings; an
    empty list means the value is valid.
    """
    errors = []
    if isinstance(number_repair_crews, bool):
        return [f"number_repair_crews must be an int or dict, not a bool: {number_repair_crews!r}"]
    if isinstance(number_repair_crews, int):
        if number_repair_crews < 0:
            errors.append("number_repair_crews scalar value must be non-negative.")
        return errors
    if not isinstance(number_repair_crews, dict):
        return [
            "number_repair_crews must be an int, dict[island_id, int], "
            "dict[asset_type, int], or dict[island_id, dict[asset_type, int]]; "
            f"got {type(number_repair_crews).__name__}."
        ]
    if not number_repair_crews:
        return errors

    keys = list(number_repair_crews.keys())
    is_int_key = lambda k: isinstance(k, int) and not isinstance(k, bool)
    all_int_keys = all(is_int_key(k) for k in keys)
    all_str_keys = all(isinstance(k, str) for k in keys)

    if not all_int_keys and not all_str_keys:
        errors.append(
            "number_repair_crews dict keys must be all island IDs (int) or all "
            "asset-type strings (str); mixed key types are not supported."
        )
        return errors

    values = list(number_repair_crews.values())
    value_is_dict = [isinstance(v, dict) for v in values]

    if all_int_keys:
        if any(value_is_dict) and not all(value_is_dict):
            errors.append(
                "number_repair_crews island-keyed dict values must be all plain "
                "counts (int) or all per-type dicts, not a mix."
            )
            return errors
        if all(value_is_dict):
            for island_id, type_counts in number_repair_crews.items():
                for asset_type, count in type_counts.items():
                    if not isinstance(asset_type, str):
                        errors.append(
                            f"number_repair_crews island {island_id}: inner keys must "
                            f"be asset-type strings, got {asset_type!r}."
                        )
                    elif not isinstance(count, int) or isinstance(count, bool) or count < 0:
                        errors.append(
                            f"number_repair_crews island {island_id}, type '{asset_type}': "
                            f"crew count must be a non-negative int, got {count!r}."
                        )
        else:
            for island_id, count in number_repair_crews.items():
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    errors.append(
                        f"number_repair_crews island {island_id}: crew count must be "
                        f"a non-negative int, got {count!r}."
                    )
    else:
        for asset_type, count in number_repair_crews.items():
            if isinstance(count, dict):
                errors.append(
                    f"number_repair_crews type '{asset_type}': global type-specific "
                    "values must be plain counts (int); use the island+type nested "
                    "form for per-island type pools."
                )
            elif not isinstance(count, int) or isinstance(count, bool) or count < 0:
                errors.append(
                    f"number_repair_crews type '{asset_type}': crew count must be a "
                    f"non-negative int, got {count!r}."
                )

    return errors


def validate_config(config):
    """
    Check for required directories and simulation_config shape validity.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        tuple: (is_valid, missing_directories, warnings)
    """
    missing_dirs = []
    
    # Check required directories
    required_dirs = ['electricity_dir', 'hazard_dir']
    for key in required_dirs:
        path = config[key]
        if not path.exists():
            missing_dirs.append(f"{key}: {path}")

    warnings = _validate_number_repair_crews(
        config.get('simulation_config', {}).get('number_repair_crews', 0)
    )

    is_valid = len(missing_dirs) == 0 and len(warnings) == 0
    
    return is_valid, missing_dirs, warnings


def setup_directories(config):
    """
    Create necessary directories if they don't exist.
    
    Args:
        config (dict): Configuration dictionary
    """
    # Create interim and output directories
    config['interim_dir'].mkdir(parents=True, exist_ok=True)
    config['output_dir'].mkdir(parents=True, exist_ok=True)

def setup_dev_directories(config, remove_cache=False, remove_output=False):
    """
    Create necessary directories if they don't exist.
    
    Args:
        config (dict): Configuration dictionary
    """
    # Remove directories if requested
    if remove_cache and config['interim_dir'].exists():
        shutil.rmtree(config['interim_dir'])
    if remove_output and config['output_dir'].exists():
        shutil.rmtree(config['output_dir'])

    # Always create directories, ignore if they exist
    config['interim_dir'].mkdir(parents=True, exist_ok=True)
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    print(f"Created directories: {config['interim_dir']}, {config['output_dir']}")

def get_simulation_params(config):
    simulation_params = {
    'flood_threshold': config['simulation_config']['flood_threshold'],
    'number_repair_crews': config['simulation_config']['number_repair_crews'],
    'repair_crew_assignment_method': config['simulation_config']['repair_crew_assignment_method'],
    'verbose': config['simulation_config']['verbose'],
    'recovery_parameters': config['recovery_parameters'],
    'config': config  # Pass entire config for directory management
    }
    # Convert string 'None' to actual None
    for key, value in simulation_params.items():
        if isinstance(value, str) and value.lower() == 'none':
            simulation_params[key] = None
    return simulation_params

def print_config_summary(config):
    """
    Print a summary of the current configuration.
    
    Args:
        config (dict): Configuration dictionary
    """
    print("\nConfiguration Summary")
    print(f"Root directory: {config['root_dir']}")
    print(f"Assets data: {config['electricity_dir']}")
    print(f"Hazard data: {config['hazard_dir']}")
    print(f"Interim directory: {config['interim_dir']}")
    print(f"Output directory: {config['output_dir']}")
    print()
    
    print("Simulation Configuration:")
    for key, value in config['simulation_config'].items():
        print(f"  {key}: {value}")
    print()
    
    print("Recovery Parameters:")
    for key, value in config['recovery_parameters'].items():
        print(f"  {key}: {value}")
    print()
    
    print("Analysis Configuration:")
    for key, value in config['analysis_config'].items():
        print(f"  {key}: {value}")


# Environment-specific configurations

def get_development_config():
    """Get configuration optimized for development/testing."""
    config = get_config()
    config['analysis_config']['max_simulation_days'] = 3  # Limit for faster testing
    config['analysis_config']['performance_monitoring'] = True
    config['simulation_config']['verbose'] = True

    # Smallest data samples directories
    config['data_dir'] = config['root_dir'] / 'data' / 'test_samples'
    config['electricity_dir'] = config['data_dir'] / 'electricity'

    config['simulation_config']['number_repair_crews'] = 5  # Fewer crews for faster testing

    config['hazard_dir'] = config['data_dir'] / 'test_hazard_timesteps'

    # Update interim and output directories to match the new hazard_dir
    hazard_dir_name = config['hazard_dir'].name
    config['interim_dir'] = config['root_dir'] / 'data' / 'interim' / f'interim_{hazard_dir_name}'
    config['output_dir'] = config['root_dir'] / 'data' / 'output' / f'output_{hazard_dir_name}'

    return config

def get_production_config():
    """Get configuration optimized for production runs."""
    config = get_config()
    config['analysis_config']['max_simulation_days'] = None  # Use all available days
    config['analysis_config']['performance_monitoring'] = False
    config['simulation_config']['verbose'] = False
    return config

if __name__ == "__main__":
    # Example usage and testing
    config = get_config()
    
    # Validate configuration
    is_valid, missing_dirs, warnings = validate_config(config)
    
    if not is_valid:
        print("Configuration validation failed!")
        print(f"Missing directories: {missing_dirs}")
    
    if warnings:
        print(f"Configuration warnings: {warnings}")
    
    # Print configuration summary
    print_config_summary(config)
    
    # Setup directories
    setup_directories(config)
    
    if is_valid:
        print("\n✅ Configuration is valid and ready for use!")
    else:
        print(f"\n❌ Configuration has issues that need to be resolved.")
        print("Missing directories:")
        for missing_dir in missing_dirs:
            print(f"  - {missing_dir}")
