import configparser
from pathlib import Path

# Base data path (used across all notebooks)
BASE_DATA_PATH = Path('C:/data')

# Centralized data paths for all notebooks
data_paths = {
    'base_data_path': BASE_DATA_PATH,
    
    # Electrical network data paths (from visualizations_telectrified.ipynb)
    'high_tension_stations': BASE_DATA_PATH / r'raw\StedinData\Hoogspanningsstations\Hoogspanningsstations.shp',
    'high_tension_network': BASE_DATA_PATH / r'raw\StedinData\Hoogspanningsverbindingen\Hoogspanningsverbindingen.shp',
    'mid_tension_stations': BASE_DATA_PATH / r'raw\StedinData\Middenspanningsstations\Middenspanningsstations.shp',
    'mid_tension_network': BASE_DATA_PATH / r'raw\StedinData\Middenspanningsverbindingen\Middenspanningsverbindingen.shp',
    'mid_low_tension_stations': BASE_DATA_PATH / r'raw\StedinData\MiddenLaagspanningsstations\MiddenLaagspanningsstations.shp',
    'low_tension_stations': BASE_DATA_PATH / r'raw\StedinData\Laagspanningsstations\Laagspanningsstations.shp',
    'low_tension_network': BASE_DATA_PATH / r'raw\StedinData\Laagspanningsverbindingen\Laagspanningsverbindingen.shp',
    
    # Telecom data paths (from scrape_masts.ipynb)
    'telecom_c2000_masts': BASE_DATA_PATH / r'raw\TelecomData\c2000masten.geojson'
}

# Miraca color palette for visualizations
miraca_colors = {
    'primary blue': '#4069F6',
    'accent green': '#64F4C0',
    'white': '#FFFFFF',
    'black': '#171E37',
    'blue_900': '#233778',
    'blue_800': '#2A4396',
    'blue_700': '#314EB3',
    'blue_600': '#385AD1',
    'blue_500': '#4069F6',
    'blue_400': '#6687F8',
    'blue_300': '#94ABFA',
    'blue_200': '#C2CFFC',
    'blue_100': '#E0E7FE',
    'green_900': '#429787',
    'green_800': '#4CB499',
    'green_700': '#56CEA9',
    'green_600': '#5ADBB1',
    'green_500': '#64F4C0',
    'green_400': '#9CF8D7',
    'green_300': '#B5FAE1',
    'green_200': '#CDFCEB',
    'green_100': '#E0FDF2',
    'grey_900': '#373D52',
    'grey_800': '#545866',
    'grey_700': '#676B7A',
    'grey_600': '#7B7F8F',
    'grey_500': '#8F94A3',
    'grey_400': '#A5A9B8',
    'grey_300': '#BCBFCC',
    'grey_200': '#D3D6E0',
    'grey_100': '#EBEDF5',
    'red_danger': '#ED5861',
    'yellow_alert': '#F8CD48',
    'green_success': '#72DA95'
}

# Optional: Create configparser configuration for other tools that might need it
def create_config_file():
    """Create a .ini configuration file with the data paths and colors."""
    config = configparser.ConfigParser()
    
    config['DEFAULT'] = {}
    
    config['PATHS'] = {key: str(value) for key, value in data_paths.items()}
    
    config['COLORS'] = miraca_colors
    
    with open('config_plotting.ini', 'w') as configfile:
        config.write(configfile)
    
    return config

# Uncomment the line below if you want to automatically create the .ini file
create_config_file()