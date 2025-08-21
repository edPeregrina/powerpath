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
        'data_dir': root_dir / 'raw_data' / 'ZH_Delfland_2',
        'electricity_dir': root_dir / 'raw_data' / 'ZH_Delfland_2' / 'electricity',
        # 'electricity_dir': root_dir / 'data' / 'electricity',
        
        # Simulation configuration
        'simulation_config': {
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
            'time_step_hours': 1,
            'damage_threshold': 0.01,    # Minimum damage ratio to consider asset damaged
            'repair_threshold': 2.0      # Minimum repair time threshold for repairable assets
        },
        
        # Analysis configuration
        'analysis_config': {
            'hazard_extraction_method': 'max',  # Options: 'max', 'mean', 'median', 'centroid'
            'max_simulation_days': None,  # None for all available days, integer to limit
            'cache_enabled': True,        # Enable/disable result caching for performance
            'performance_monitoring': False  # Enable detailed performance monitoring
        }
    }
    
    # Set hazard directory with override capability
    if hazard_dir_override:
        config['hazard_dir'] = Path(hazard_dir_override)
    else:
        # Default hazard directory paths (in order of preference)
        hazard_dir_options = [
            root_dir / 'raw_data' / 'ZH_Delfland' / 'hazard_maps_ZH_Delfland',
            # Path(r'N:\Projects\11209000\11209175\B. Measurements and calculations\Data\timeseries_data\reprojected'),
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


def validate_config(config):
    """
    Check for required directories.
    
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
    
    is_valid = len(missing_dirs) == 0
    
    return is_valid, missing_dirs


def setup_directories(config):
    """
    Create necessary directories if they don't exist.
    
    Args:
        config (dict): Configuration dictionary
    """
    # Create interim and output directories
    config['interim_dir'].mkdir(parents=True, exist_ok=True)
    config['output_dir'].mkdir(parents=True, exist_ok=True)


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


# Configuration presets for different scenarios
def get_high_resilience_config():
    """Configuration for high resilience scenario (more crews, faster repair)."""
    config = get_config()
    config['simulation_config']['number_repair_crews'] = 20
    config['recovery_parameters']['repair_time_coefficients'] = [351.36, 1.57, 0.995]  # Faster repair
    return config


def get_low_resilience_config():
    """Configuration for low resilience scenario (fewer crews, slower repair)."""
    config = get_config()
    config['simulation_config']['number_repair_crews'] = 5
    config['recovery_parameters']['repair_time_coefficients'] = [1405.44, 6.28, 3.978]  # Slower repair
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











# # Base data path (used across all notebooks)
# BASE_DATA_PATH = Path('C:/data')

# # Centralized data paths for all notebooks
# data_paths = {
#     'base_data_path': BASE_DATA_PATH,
    
#     # Electrical network data paths (from visualizations_telectrified.ipynb)
#     'high_tension_stations': BASE_DATA_PATH / r'raw\StedinData\Hoogspanningsstations\Hoogspanningsstations.shp',
#     'high_tension_network': BASE_DATA_PATH / r'raw\StedinData\Hoogspanningsverbindingen\Hoogspanningsverbindingen.shp',
#     'mid_tension_stations': BASE_DATA_PATH / r'raw\StedinData\Middenspanningsstations\Middenspanningsstations.shp',
#     'mid_tension_network': BASE_DATA_PATH / r'raw\StedinData\Middenspanningsverbindingen\Middenspanningsverbindingen.shp',
#     'mid_low_tension_stations': BASE_DATA_PATH / r'raw\StedinData\MiddenLaagspanningsstations\MiddenLaagspanningsstations.shp',
#     'low_tension_stations': BASE_DATA_PATH / r'raw\StedinData\Laagspanningsstations\Laagspanningsstations.shp',
#     'low_tension_network': BASE_DATA_PATH / r'raw\StedinData\Laagspanningsverbindingen\Laagspanningsverbindingen.shp',
    
#     # Telecom data paths (from scrape_masts.ipynb)
#     'telecom_c2000_masts': BASE_DATA_PATH / r'raw\TelecomData\c2000masten.geojson'
# }

# # Miraca color palette for visualizations
# miraca_colors = {
#     'primary blue': '#4069F6',
#     'accent green': '#64F4C0',
#     'white': '#FFFFFF',
#     'black': '#171E37',
#     'blue_900': '#233778',
#     'blue_800': '#2A4396',
#     'blue_700': '#314EB3',
#     'blue_600': '#385AD1',
#     'blue_500': '#4069F6',
#     'blue_400': '#6687F8',
#     'blue_300': '#94ABFA',
#     'blue_200': '#C2CFFC',
#     'blue_100': '#E0E7FE',
#     'green_900': '#429787',
#     'green_800': '#4CB499',
#     'green_700': '#56CEA9',
#     'green_600': '#5ADBB1',
#     'green_500': '#64F4C0',
#     'green_400': '#9CF8D7',
#     'green_300': '#B5FAE1',
#     'green_200': '#CDFCEB',
#     'green_100': '#E0FDF2',
#     'grey_900': '#373D52',
#     'grey_800': '#545866',
#     'grey_700': '#676B7A',
#     'grey_600': '#7B7F8F',
#     'grey_500': '#8F94A3',
#     'grey_400': '#A5A9B8',
#     'grey_300': '#BCBFCC',
#     'grey_200': '#D3D6E0',
#     'grey_100': '#EBEDF5',
#     'red_danger': '#ED5861',
#     'yellow_alert': '#F8CD48',
#     'green_success': '#72DA95'
# }

# # Optional: Create configparser configuration for other tools that might need it
# def create_config_file():
#     """Create a .ini configuration file with the data paths and colors."""
#     config = configparser.ConfigParser()
    
#     config['DEFAULT'] = {}
    
#     config['PATHS'] = {key: str(value) for key, value in data_paths.items()}
    
#     config['COLORS'] = miraca_colors
    
#     with open('config_plotting.ini', 'w') as configfile:
#         config.write(configfile)
    
#     return config

# # Uncomment the line below if you want to automatically create the .ini file
# create_config_file()