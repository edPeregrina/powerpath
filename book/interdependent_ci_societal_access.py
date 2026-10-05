# Configuration and File Paths
import sys
import time
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import geopandas as gpd
from pickle import dump, load

from config import get_config, get_development_config, validate_config, setup_directories, print_config_summary
from src.simulation import simulate_asset_damage_recovery_access_breakdown
from src.data_loader import load_hazard_maps, load_electricity_assets
from src.visualisations import *
from src.impacts import prepare_population_impact_data, prepare_land_use_impact_data, create_voronoi_for_asset_type
from src.caching import load_simulation_caches

# Get configuration (currently from config.py)
config = get_config()
# config = get_development_config()

setup_directories(config)

# Print summary
print_config_summary(config)

# Set the hazard extraction method constant
HAZARD_EXTRACTION_METHOD = config['analysis_config']['hazard_extraction_method']
print(f"Hazard extraction method set to: {HAZARD_EXTRACTION_METHOD}")

# Load data
gdf_assets = load_electricity_assets(config['electricity_dir'], asset_types=['msls'])

hazard_maps = load_hazard_maps(config['hazard_dir'], max_days=None)  
print(f"\nHazard maps loaded:")
for i, hm in enumerate(hazard_maps):
    if i < 5:
        print(f"  -{hm}")
    elif i == 5:
        print(f"  and {len(hazard_maps) - 3} more")
        break
# Configure simulation parameters from config
simulation_params = {
    'flood_threshold': config['simulation_config']['flood_threshold'],
    'number_repair_crews': config['simulation_config']['number_repair_crews'],
    'repair_crew_assignment_method': config['simulation_config']['repair_crew_assignment_method'],
    'verbose': config['simulation_config']['verbose'],
    'damage_ratio_coefficients': config['recovery_parameters']['damage_ratio_coefficients'],
    'repair_time_coefficients': config['recovery_parameters']['repair_time_coefficients'],
    'damage_threshold': config['recovery_parameters']['damage_threshold'],
    'repair_threshold': config['recovery_parameters']['repair_threshold'],
    'config': config  # Pass entire config for directory management
}

accessibility_model = None#grid_hex.accessibility_model
simulation_params['accessibility_model'] = accessibility_model

print(f"\nSimulation configuration:")
for key, value in simulation_params.items():
    if key != 'config':  # Don't print the entire config
        print(f"  {key}: {value}")

print(f"\nDirectory structure:")
print(f"  Interim: {config['interim_dir']}")
print(f"  Output: {config['output_dir']}")
print(f"  Cache will be organized by hazard directory: {Path(config['hazard_dir']).name}")



simulation_params['verbose'] = True  # Set verbose to False for cleaner output
simulation_params['flood_threshold'] = 0.2

execution_id = int(time.time())
print(f"***Starting simulation execution {execution_id}***")

max_days = None#config['analysis_config']['max_simulation_days']

caches = load_simulation_caches(config['interim_dir'], config['hazard_dir'])
print(f"Loaded caches: {[k for k in caches.keys() if caches[k] is not None]}")

# Run the  simulation
results_df, final_state, cache_updated = simulate_asset_damage_recovery_access_breakdown(
    gdf_assets=gdf_assets,
    hazard_maps=hazard_maps[:max_days],
    number_repair_crews=simulation_params['number_repair_crews'],
    repair_crew_assignment_method=simulation_params['repair_crew_assignment_method'],
    flood_threshold=simulation_params['flood_threshold'],
    recovery_parameters=config['recovery_parameters'],
    root_dir=config['root_dir'],
    verbose=simulation_params['verbose'], 
    timestep_output=True, 
    execution_id=execution_id,
    config=config, 
    major_timestep=6,
    accessibility_cache=caches.get('accessibility_cache'),
    hazard_extraction_cache=caches.get('hazard_extraction_cache'),
    overlap_cache=caches.get('overlap_cache'),
    island_cache=caches.get('island_cache'),
    fragility_param_k=config['simulation_config'].get('fragility_param_k', 5.0)
)


print(f"\n***Completed simulation execution {execution_id}***")
detailed_results = results_df[0][2]
print('Hours in simulation: ', len(detailed_results))
root_dir = Path(config['root_dir'])

study_area_path = root_dir / 'data' / 'utilities' / 'stedin_area.geojson'
study_area = gpd.read_file(study_area_path, driver='GeoJSON').to_crs("EPSG:28992")

buffer_distance_meters = 1000
population_buffer = study_area.buffer(buffer_distance_meters)
population_buffer.set_crs("EPSG:28992")

# Create Voronoi polygons for service areas
print("Creating Voronoi polygons for service areas...")
voronoi_polygons_by_type = {}
for asset_type in gdf_assets['type'].unique():
    voronoi_gdf = create_voronoi_for_asset_type(
        gdf_assets, 
        asset_type, 
        boundary=study_area.geometry.unary_union
    )
    voronoi_polygons_by_type[asset_type] = voronoi_gdf

# Load and prepare population data
print("Loading population data...")
population_data_path = root_dir / "data" / "population" / "cbs_vk100_2024_v1.gpkg"
population_data = gpd.read_file(
    population_data_path,
    driver='GPKG',
    bbox=tuple(voronoi_gdf.total_bounds)  # Only load data within bounding box
).to_crs("EPSG:28992")

# Filter out zero population and invalid geometries
population_data = population_data[
    (population_data['aantal_inwoners'] > 0) & 
    (~population_data.geometry.isna())
].dropna(subset=['geometry'])

# Clip to study area
population_study_area = gpd.clip(population_data, population_buffer)
population_above_0 = population_study_area[['aantal_inwoners', 'geometry']].set_crs("EPSG:28992")

# Prepare population impact mapping
print("Preparing population impact data...")
asset_population_map = prepare_population_impact_data(
    population_data=population_above_0,
    voronoi_gdf=voronoi_polygons_by_type['msls']
)
print(f"Population data prepared for {len(asset_population_map)} assets")
voronoi_gdf['pop_map'] = [asset_population_map[x] for x in voronoi_gdf['asset_id']]

import pickle

# Load land use data path
land_use_path = root_dir / 'data' / 'land_use' / 'CBS_Publicatiebestand_BBG2017_v1.gpkg'

# Cache path for asset_to_lu (the complete pre-computed lookup)
asset_to_lu_cache_path = config['interim_dir'] / f"{land_use_path.stem}_asset_to_lu.pkl"

# Try to load pre-computed asset_to_lu
if asset_to_lu_cache_path.exists():
    print(f"Loading asset_to_lu from cache: {asset_to_lu_cache_path}")
    with open(asset_to_lu_cache_path, 'rb') as f:
        asset_to_lu = pickle.load(f)
    print(f"Loaded asset_to_lu for {len(asset_to_lu)} assets from cache")

    asset_land_use_map = {
        aid: {lu_type: area for lu_type, area, voll_rate in lu_entries}
        for aid, lu_entries in asset_to_lu.items()
    }
    
else:
    print("Computing asset_to_lu from scratch...")
    
    # Only import when cache miss occurs
    from src.impacts import update_voll_rates, prepare_land_use_impact_data, VOLL_PER_SQM
    
    # Load land use data
    land_use_data = gpd.read_file(land_use_path, driver='GPKG')
    
    # Calculate VOLL rates (populates VOLL_PER_SQM)
    update_voll_rates(land_use_data)
    print(f"VOLL rates calculated for {len(VOLL_PER_SQM)} categories")
    
    # Compute asset-land use intersections (the expensive operation)
    asset_land_use_map = prepare_land_use_impact_data(
        land_use_data, 
        voronoi_gdf=voronoi_polygons_by_type['msls']
    )
    print(f"Computed intersections for {len(asset_land_use_map)} assets")
    
    # Build asset_to_lu with VOLL rates included
    asset_to_lu = {}
    for aid, lu_dict in asset_land_use_map.items():
        asset_to_lu[aid] = [(lu_type, area, VOLL_PER_SQM[lu_type]) 
                             for lu_type, area in lu_dict.items() 
                             if lu_type in VOLL_PER_SQM]
    
    # Cache the complete lookup
    with open(asset_to_lu_cache_path, 'wb') as f:
        pickle.dump(asset_to_lu, f)
    print(f"Saved asset_to_lu to cache: {asset_to_lu_cache_path}")

# Verify it worked
print(f"\nasset_to_lu ready with {len(asset_to_lu)} assets")
sample_asset = list(asset_to_lu.keys())[0]
print(f"\nSample asset {sample_asset} has {len(asset_to_lu[sample_asset])} land use entries:")
for lu_type, area, voll_rate in asset_to_lu[sample_asset][:3]:
    print(f"  {lu_type}: {area:.1f} m² @ €{voll_rate:.6f}/m²/h")
from ema_workbench import Model, RealParameter, CategoricalParameter, IntegerParameter, Constant, TimeSeriesOutcome
from ema_workbench import em_framework, SequentialEvaluator, MultiprocessingEvaluator, Samplers, ema_logging, Policy
from shapely.geometry import box
from src.impacts import CONSUMPTION_PER_SQM, VOLL_PER_SQM
from src.adaptation import simulate_asset_damage_recovery_access_breakdown_ema

# Use calculated rates 
print("\nUsing calculated consumption rates [MWh/m²/h]:")
for category, rate in CONSUMPTION_PER_SQM.items():
    print(f"  {category}: {rate:.10f}")

print("\nUsing calculated VOLL rates [€/m²/h]:")
for category, rate in VOLL_PER_SQM.items():
    print(f"  {category}: {rate:.6f}")

# Pre-compute asset_to_lu lookup table once
print("\nPre-computing asset_to_lu lookup table...")
asset_to_lu = {}  # {asset_id: [(lu_type, area, voll_rate), ...]}
for aid, lu_dict in asset_land_use_map.items():
    asset_to_lu[aid] = [(lu_type, area, VOLL_PER_SQM[lu_type]) 
                         for lu_type, area in lu_dict.items() 
                         if lu_type in VOLL_PER_SQM]
print(f"Pre-computed lookup for {len(asset_to_lu)} assets")

ema_logging.log_to_stderr(ema_logging.INFO)

# Load caches
caches = load_simulation_caches(config['interim_dir'], config['hazard_dir'])
print(f"Loaded caches: {[k for k in caches.keys() if caches[k] is not None]}")

max_days = None
execution_id = int(time.time())

# Set up EMA model
model = Model('ElectricitySimulation', function=simulate_asset_damage_recovery_access_breakdown_ema)

model.uncertainties = [
    IntegerParameter('number_repair_crews', 10, 20),
    RealParameter('fragility_param_k', 5.0, 7.5)
]

keep_3d_vars = []  # No per-asset arrays
keep_2d_vars = ['flooded', 'operational', 'unreachable', 'damage_ratio', 'repair_time']  # Aggregated

model.constants = [
    Constant('flood_threshold', 0.2),
    Constant('gdf_assets', gdf_assets),
    Constant('hazard_maps', hazard_maps[:max_days]),
    Constant('recovery_parameters', config['recovery_parameters']),
    Constant('root_dir', config['root_dir']),
    Constant('verbose', False),
    Constant('timestep_output', True),
    Constant('execution_id', execution_id),
    Constant('config', config),
    Constant('major_timestep', 6),
    # Cache parameters
    Constant('accessibility_cache', caches.get('accessibility_cache')),
    Constant('hazard_extraction_cache', caches.get('hazard_extraction_cache')),
    Constant('overlap_cache', caches.get('overlap_cache')),
    Constant('island_cache', caches.get('island_cache')),
    # Impact data
    Constant('asset_population_map', asset_population_map),
    Constant('asset_to_lu', asset_to_lu),
    # Dimensionality control 
    Constant('keep_3d_vars', []),  # 
    Constant('keep_2d_vars', ['flooded', 'operational', 'unreachable', 'damage_ratio', 'repair_time']) 
]



# Build outcomes list
outcomes_list = []

# Add 3D outcomes (none in this case)
for var in keep_3d_vars:
    outcomes_list.append(TimeSeriesOutcome(var))

# Add 2D outcomes (aggregated across assets)
for var in keep_2d_vars:
    outcomes_list.append(TimeSeriesOutcome(var))

# Add mandatory 1D outcomes (always present)
outcomes_list.extend([
    TimeSeriesOutcome('timesteps'),
    # Population impact outcomes
    TimeSeriesOutcome('affected_population'),
    TimeSeriesOutcome('served_population'),
    TimeSeriesOutcome('affected_population_ratio'),
    # Monetary impacts
    TimeSeriesOutcome('monetary_impact_total'),
    TimeSeriesOutcome('monetary_impact_residential'),
    TimeSeriesOutcome('monetary_impact_commercial'),
    TimeSeriesOutcome('monetary_impact_industrial'),
    TimeSeriesOutcome('monetary_impact_transport'),
    TimeSeriesOutcome('monetary_impact_public_sector'),
])

model.outcomes = outcomes_list

# Set up policies
l1_pumping_area = root_dir / 'data' / 'test_samples' / 'adaptation' / 'high_exp_peilgeb_28992.geojson'#'l1_adapt_polygon.geojson' #0.3m
l2_asset_barriers = root_dir / 'data' / 'test_samples' / 'adaptation' / 'exp_subs_outside_hexp_peilgeb_28992.geojson'#'l2_adapt_polygon.geojson' #0.5m


policies = [
    Policy('baseline_road', repair_crew_assignment_method='islands'),    
    Policy('population_impacts_road', repair_crew_assignment_method='population impacts islands'),
    Policy('monetary_impacts_road', repair_crew_assignment_method='monetary impacts islands'),    
]

with SequentialEvaluator(model) as evaluator:
    experiments, outcomes = evaluator.perform_experiments(
        scenarios=1,
        policies=policies,
        uncertainty_sampling=Samplers.LHS
    )
    
experiments_df = pd.DataFrame(experiments)
print(f"\nCompleted all experiments with execution id {execution_id}. Total runs: {len(experiments)}")
from pathlib import Path
import itertools

# Set up paths to L1 and L2 adaptation files
l1_dir = root_dir / 'data' / 'test_samples' / 'adaptation' / 'L1_peilgeb_policies_separate'
l2_dir = root_dir / 'data' / 'test_samples' / 'adaptation' / 'L2_asset_barriers'

# Collect all geojson files from L1 and L2 directories
l1_files = sorted(l1_dir.glob('*.geojson')) if l1_dir.exists() else []
l2_files = sorted(l2_dir.glob('*.geojson')) if l2_dir.exists() else []

print(f"Found L1 files: {len(l1_files)}")
print(f"Found L2 files: {len(l2_files)}")

insp_file = l1_files[2]
print(f"\nInspecting L1 adaptation file: {insp_file}")
l1_data_inspect = gpd.GeoDataFrame.from_file(insp_file)
print("CRS of file: ", l1_data_inspect.crs)
print("Number of features:", len(l1_data_inspect))
l1_data_inspect.head(3)
insp_file = l2_files[3]
print(f"\nInspecting L2 adaptation file: {insp_file}")
l2_data_inspect = gpd.GeoDataFrame.from_file(insp_file)
print("CRS of file: ", l2_data_inspect.crs)
print("Number of features:", len(l2_data_inspect))
l2_data_inspect.head(3)

# Base policies (no adaptation)
policies = [
    Policy('population_impacts_policy', 
           repair_crew_assignment_method='population impacts islands'),
    Policy('monetary_impacts_policy', 
           repair_crew_assignment_method='monetary impacts islands'),    
    Policy('lowest_repair_time_islands_policy', 
           repair_crew_assignment_method='lowest repair time islands'),
    Policy('highest_repair_time_islands_policy', 
           repair_crew_assignment_method='highest repair time islands'),
]

# Create adaptation policies: all L1 x L2 combinations
adaptation_policies = []

# Add baseline (no adaptation)
adaptation_policies.append(
    Policy('baseline', 
           repair_crew_assignment_method='monetary impacts islands')
)

# Add L1-only policies
for l1_file in l1_files:
    policy_name = f"L1_only_{l1_file.stem}"
    adaptation_policies.append(
        Policy(policy_name,
               repair_crew_assignment_method='monetary impacts islands',
               l1_area_geojson=str(l1_file),
               l1_active_timesteps=list(range(0, 198))
               )
    )

# Add L2-only policies
for l2_file in l2_files:
    policy_name = f"L2_only_{l2_file.stem}"
    adaptation_policies.append(
        Policy(policy_name,
               repair_crew_assignment_method='monetary impacts islands',
               l2_asset_geojson=str(l2_file),
               l2_active_timesteps=list(range(0, 198))
               ) 
    )

# Add L1 + L2 combined policies (all combinations)
for l1_file, l2_file in itertools.product(l1_files, l2_files):
    policy_name = f"L1L2_{l1_file.stem}_{l2_file.stem}"
    adaptation_policies.append(
        Policy(policy_name,
               repair_crew_assignment_method='monetary impacts islands',
               l1_area_geojson=str(l1_file),
               l1_active_timesteps=list(range(0, 198)),
               l2_asset_geojson=str(l2_file),
               l2_active_timesteps=list(range(0, 198))
                )
    )

# Combine all policies
policies.extend(adaptation_policies)

print(f"\nTotal policies to evaluate: {len(policies)}")
print(f"  - Base repair policies: 4")
print(f"  - Baseline (no adaptation): 1")
print(f"  - L1-only policies: {len(l1_files)}")
print(f"  - L2-only policies: {len(l2_files)}")
print(f"  - L1+L2 combined: {len(l1_files) * len(l2_files)}")

print("\nPolicy list:")
for i, policy in enumerate(policies, 1):
    print(f"  {i}. {policy.name}")

# Run EMA experiments
with SequentialEvaluator(model) as evaluator:
    experiments, outcomes = evaluator.perform_experiments(
        scenarios=100,
        policies=policies,
        uncertainty_sampling=Samplers.LHS
    )

experiments_df = pd.DataFrame(experiments)
print(f"\nCompleted all experiments with execution id {execution_id}. Total runs: {len(experiments)}")

# Summary: count policies by type
print("\n" + "="*70)
print("ADAPTATION POLICY EXPLORATION SUMMARY")
print("="*70)

for policy_name in adaptation_policies:
    policy_data = experiments_df[experiments_df['policy'] == policy_name.name]
    if len(policy_data) > 0:
        print(f"\n{policy_name.name}:")
        print(f"  Runs: {len(policy_data)}")
from pathlib import Path
import itertools


# L1 files (top10 / top20 / top30 × 0.2m & 0.4m)
l1_dir = Path(r"C:\repos\powerpath\powerpath\data\test_samples\adaptation\L1_peilgeb_policies_separate")
l1_files = sorted(list(l1_dir.glob("policy_L1_area_*.geojson")))

# L2 files (top10 / top20 / top30 × 0.5m & 1.0m)
l2_dir = Path(r"C:\repos\powerpath\powerpath\data\test_samples\adaptation\L2_asset_barriers")
l2_files = sorted(list(l2_dir.glob("*.geojson")))

print(f"Found {len(l1_files)} L1 files")
print(f"Found {len(l2_files)} L2 files")

# === BASE POLICIES ===
policies = [
    Policy('population_impacts_policy', repair_crew_assignment_method='population impacts islands'),
    Policy('monetary_impacts_policy', repair_crew_assignment_method='monetary impacts islands'),
    Policy('lowest_repair_time_islands_policy', repair_crew_assignment_method='lowest repair time islands'),
    Policy('highest_repair_time_islands_policy', repair_crew_assignment_method='highest repair time islands'),
]

# === ADAPTATION POLICIES ===
adaptation_policies = []

# Baseline
adaptation_policies.append(
    Policy('baseline',
           repair_crew_assignment_method='monetary impacts islands')
)

# === L1-only ===
for l1_file in l1_files:
    policy_name = f"L1_only_{l1_file.stem}"
    adaptation_policies.append(
        Policy(policy_name,
               repair_crew_assignment_method='monetary impacts islands',
               l1_area_geojson=str(l1_file),
               l1_active_timesteps=list(range(0, 198))
               )
    )

# === L2-only ===
for l2_file in l2_files:
    policy_name = f"L2_only_{l2_file.stem}"
    adaptation_policies.append(
        Policy(policy_name,
               repair_crew_assignment_method='monetary impacts islands',
               l2_asset_geojson=str(l2_file),
               l2_active_timesteps=list(range(0, 198))
               )
    )

# === L1 + L2 combinations ===
for l1_file, l2_file in itertools.product(l1_files, l2_files):
    policy_name = f"L1L2_{l1_file.stem}_{l2_file.stem}"
    adaptation_policies.append(
        Policy(policy_name,
               repair_crew_assignment_method='monetary impacts islands',
               l1_area_geojson=str(l1_file),
               l1_active_timesteps=list(range(0, 198)),
               l2_asset_geojson=str(l2_file),
               l2_active_timesteps=list(range(0, 198))
               )
    )

# Combine
policies.extend(adaptation_policies)

# Summary print
print(f"\nTotal policies to evaluate: {len(policies)}")
print(f"  - Base repair policies: 4")
print(f"  - Baseline (no adaptation): 1")
print(f"  - L1-only policies: {len(l1_files)}")
print(f"  - L2-only policies: {len(l2_files)}")
print(f"  - L1+L2 combined: {len(l1_files) * len(l2_files)}")

print("\nPolicy list:")
for i, policy in enumerate(policies, 1):
    print(f"{i}. {policy.name}")

# === RUN ===
with SequentialEvaluator(model) as evaluator:
    experiments, outcomes = evaluator.perform_experiments(
        scenarios=100,
        policies=policies,
        uncertainty_sampling=Samplers.LHS
    )

experiments_df = pd.DataFrame(experiments)
print(f"\nCompleted all experiments. Total runs: {len(experiments)}")

# Summary
print("\n" + "="*70)
print("ADAPTATION POLICY EXPLORATION SUMMARY")
print("="*70)

for policy_name in adaptation_policies:
    policy_data = experiments_df[experiments_df['policy'] == policy_name.name]
    print(f"\n{policy_name.name}:")
    print(f"  Runs: {len(policy_data)}")
from pathlib import Path
import itertools


# L1 files (top10 / top20 / top30 × 0.2m & 0.4m)
l1_dir = Path(r"C:\repos\powerpath\powerpath\data\test_samples\adaptation\L1_peilgeb_policies_separate")
l1_files = sorted(list(l1_dir.glob("policy_L1_area_*.geojson")))

# L2 files (top10 / top20 / top30 × 0.5m & 1.0m)
l2_dir = Path(r"C:\repos\powerpath\powerpath\data\test_samples\adaptation\L2_asset_barriers")
l2_files = sorted(list(l2_dir.glob("*.geojson")))
print("Repeating previous experiment with no accessibility constrains")
print(f"Found {len(l1_files)} L1 files")
print(f"Found {len(l2_files)} L2 files")

# === BASE POLICIES ===
policies = [
    Policy('monetary_impacts_policy', repair_crew_assignment_method='monetary impacts islands'),
    Policy('monetary_impacts_unconstrained', repair_crew_assignment_method='monetary impacts'),
]

# === ADAPTATION POLICIES ===
adaptation_policies = []

# Baseline
adaptation_policies.append(
    Policy('baseline_no_prioritisation',
           repair_crew_assignment_method='islands')
)

# === L1-only ===
for l1_file in l1_files:
    policy_name = f"L1_only_{l1_file.stem}"
    adaptation_policies.append(
        Policy(policy_name,
               repair_crew_assignment_method='monetary impacts islands',
               l1_area_geojson=str(l1_file),
               l1_active_timesteps=list(range(0, 198))
               )
    )

# === L2-only ===
for l2_file in l2_files:
    policy_name = f"L2_only_{l2_file.stem}"
    adaptation_policies.append(
        Policy(policy_name,
               repair_crew_assignment_method='monetary impacts islands',
               l2_asset_geojson=str(l2_file),
               l2_active_timesteps=list(range(0, 198))
               )
    )

# === L1 + L2 combinations ===
for l1_file, l2_file in itertools.product(l1_files, l2_files):
    policy_name = f"L1L2_{l1_file.stem}_{l2_file.stem}"
    adaptation_policies.append(
        Policy(policy_name,
               repair_crew_assignment_method='monetary impacts islands',
               l1_area_geojson=str(l1_file),
               l1_active_timesteps=list(range(0, 198)),
               l2_asset_geojson=str(l2_file),
               l2_active_timesteps=list(range(0, 198))
               )
    )

# Combine
policies.extend(adaptation_policies)

# Summary print
print(f"\nTotal policies to evaluate: {len(policies)}")
print(f"  - Base repair policies: 2")
print(f"  - Baseline (no adaptation): 1")
print(f"  - L1-only policies: {len(l1_files)}")
print(f"  - L2-only policies: {len(l2_files)}")
print(f"  - L1+L2 combined: {len(l1_files) * len(l2_files)}")

print("\nPolicy list:")
for i, policy in enumerate(policies, 1):
    print(f"{i}. {policy.name}")

# === RUN ===
with SequentialEvaluator(model) as evaluator:
    experiments, outcomes = evaluator.perform_experiments(
        scenarios=100,
        policies=policies,
        uncertainty_sampling=Samplers.LHS
    )

experiments_df = pd.DataFrame(experiments)
print(f"\nCompleted all experiments. Total runs: {len(experiments)}")

# Summary
print("\n" + "="*70)
print("ADAPTATION POLICY EXPLORATION SUMMARY")
print("="*70)

for policy_name in adaptation_policies:
    policy_data = experiments_df[experiments_df['policy'] == policy_name.name]
    print(f"\n{policy_name.name}:")
    print(f"  Runs: {len(policy_data)}")
op_path= config['output_dir']
#display files in output directory ordered by date that start with ema_outcomes
output_files_list = sorted(Path(op_path).glob('ema_outcomes_*.pkl'), key=lambda x: x.stat().st_mtime, reverse=True)
print(f"\nEMA outcomes files in output directory ({op_path}):")
for i, file in enumerate(output_files_list):
    if i < 5:
        print(f"  -{file.name} (modified: {time.ctime(file.stat().st_mtime)})")
    elif i == 5:
        print(f"  and {len(output_files_list) - 5} more")
        break
from ema_workbench.util.utilities import save_results
out_dir = config['output_dir'] / "ema_exports"

save_results(
    (experiments, outcomes),
    out_dir / f'ema_results_{execution_id}_combinations_4.tar.gz'
)

peilgeb_source_path = r"C:\repos\powerpath\data\test_samples\adaptation\peilgeb_with_exposure_28992.geojson"

gdf_peilgeb = gpd.GeoDataFrame.from_file(peilgeb_source_path)

gdf_peilgeb.head()

# Export each feature separately with depth reductions

output_dir = Path(r"C:\repos\powerpath\data\test_samples\adaptation\L1_peilgeb")
output_dir.mkdir(parents=True, exist_ok=True)

depth_reductions = [0.2, 0.4]  # in meters

for idx, (feature_idx, feature) in enumerate(gdf_peilgeb.iterrows()):
    feature_id = feature.get('id', idx)
    for depth_red in depth_reductions:
        gdf_single = gdf_peilgeb.iloc[[feature_idx]].copy()
        # Remove any existing 'depth_red' column to avoid dtype issues
        if 'depth_red' in gdf_single.columns:
            gdf_single = gdf_single.drop(columns=['depth_red'])
        gdf_single['depth_red'] = depth_red
        filename = f"peilgeb_{feature_id}_red_{depth_red}.geojson"
        output_path = output_dir / filename
        gdf_single.to_file(output_path, driver='GeoJSON')
        print(f"Exported: {filename}")

print(f"\nCompleted! {len(gdf_peilgeb)} features x {len(depth_reductions)} depth reductions = {len(gdf_peilgeb) * len(depth_reductions)} files")
# ----------------------------
# User inputs
# ----------------------------
peilgeb_source_path = r'C:\repos\powerpath\data\test_samples\adaptation\peilgeb_with_exposure_28992.geojson'


output_dir = Path(r'C:\repos\powerpath\data\test_samples\adaptation\L1_peilgeb_policies_separate')
output_dir.mkdir(parents=True, exist_ok=True)

shares = [0.10, 0.20, 0.30]   # 10%, 20%, 30%
l1_depths = [0.2, 0.4]        # meters (drainage reductions)

# ----------------------------
# Load geodata
# ----------------------------
gdf_peilgeb = gpd.read_file(peilgeb_source_path)
if 'id' not in gdf_peilgeb.columns:
    raise ValueError('gdf_peilgeb must contain an \"id\" column that identifies each peilgebied.')

# Ensure integer IDs for matching
gdf_peilgeb['id'] = gdf_peilgeb['id'].astype(int)

# ----------------------------
# Helpers
# ----------------------------
def load_and_rank_areas(csv_path, l1_depth):
    df = pd.read_csv(csv_path)
    d = df[(df['l1_depth'] == l1_depth) &
           (df['col_label'] == 'L1_only') &
           (~df['row_label'].str.contains('L2_only', na=False))].copy()
    # Extract numeric id from row_label like 'L1_305'
    d['id'] = d['row_label'].str.extract(r'L1_(\d+)$')[0].astype(int)
    # Average across potential duplicates, then rank by mean benefit
    d = d.groupby('id', as_index=False)['benefit_vs_reference'].mean()
    d = d.rename(columns={'benefit_vs_reference': 'benefit'})
    d = d.sort_values('benefit', ascending=False).reset_index(drop=True)
    return d

def select_top_ids(rank_df, share):
    n = len(rank_df)
    k_n = max(1, int(round(n * share)))
    return rank_df.head(k_n)['id'].tolist(), k_n

def export_policy_subset(gdf_base, ids, l1_depth, label_prefix, share):
    subset = gdf_base[gdf_base['id'].isin(ids)].copy()
    subset['depth_red'] = l1_depth
    fname = f'policy_{label_prefix}_top{int(share*100)}_red_{l1_depth:.1f}.geojson'
    subset.to_file(output_dir / fname, driver='GeoJSON')
    print(f'Exported: {fname}  (features: {len(subset)})')

# ----------------------------
# Do exports
# ----------------------------
for l1d in l1_depths:
    # Rank peilgebieden for this drainage depth using monetary and population benefits
    rank_mon = load_and_rank_areas(csv_path_monetary, l1d)
    rank_pop = load_and_rank_areas(csv_path, l1d)

    # Check for id mismatches (non-fatal, just informative)
    missing = (set(rank_mon['id']) | set(rank_pop['id'])) - set(gdf_peilgeb['id'])
    if missing:
        print(f'Warning: {len(missing)} ranked ids not found in geodata for depth {l1d}: {sorted(list(missing))[:10]}...')

    # Export separate files for each share and metric
    for s in shares:
        mon_ids, _ = select_top_ids(rank_mon, s)
        export_policy_subset(gdf_peilgeb, mon_ids, l1d, label_prefix='L1_area_monetary', share=s)

        pop_ids, _ = select_top_ids(rank_pop, s)
        export_policy_subset(gdf_peilgeb, pop_ids, l1d, label_prefix='L1_area_population', share=s)

print('Completed exporting all separate GeoJSONs.')
# Extract maximum hazard value for each asset from the detailed results
detailed_results = results_df[0][2]

# Get hazard_value for each asset at each timestep
num_assets = len(gdf_assets)
hazard_values_per_asset = {}

# Initialize all assets with 0
for asset_id in range(num_assets):
    hazard_values_per_asset[asset_id] = 0

# Iterate through all timesteps
for t, timestep_data in enumerate(detailed_results):
    if 'hazard_value' in timestep_data:
        hazard_array = timestep_data['hazard_value']
        
        # hazard_array is 1D with length = num_assets
        # Index directly: hazard_array[i] is the hazard for asset i
        for asset_idx in range(len(hazard_array)):
            val = float(hazard_array[asset_idx])
            if val > hazard_values_per_asset[asset_idx]:
                hazard_values_per_asset[asset_idx] = val

print(f"\nHazard statistics:")
print(f"  Assets with hazard > 0: {sum(1 for v in hazard_values_per_asset.values() if v > 0)}")
print(f"  Max hazard across all assets: {max(hazard_values_per_asset.values()):.4f}m")
print(f"  Min hazard across all assets: {min(hazard_values_per_asset.values()):.4f}m")

# Add maximum hazard to gdf_assets
gdf_assets['max_hazard'] = [hazard_values_per_asset[i] for i in range(num_assets)]

# Filter assets with exposure > 0.2m (20cm)
gdf_exposed = gdf_assets[gdf_assets['max_hazard'] > 0.2].copy()
gdf_exposed = gdf_exposed.to_crs(epsg=28992)  
print(f"\nAssets with exposure > 20cm: {len(gdf_exposed)} out of {len(gdf_assets)}")
if len(gdf_exposed) > 0:
    print(f"Hazard range: {gdf_exposed['max_hazard'].min():.3f}m to {gdf_exposed['max_hazard'].max():.3f}m")
else:
    print("No assets with exposure > 20cm found!")

# Calculate percentile thresholds
percentile_10 = gdf_exposed['max_hazard'].quantile(0.90)  # Top 10% = 90th percentile
percentile_20 = gdf_exposed['max_hazard'].quantile(0.80)  # Top 20% = 80th percentile
percentile_30 = gdf_exposed['max_hazard'].quantile(0.70)  # Top 30% = 70th percentile

print(f"\nPercentile thresholds:")
print(f"  Top 10%: >= {percentile_10:.3f}m")
print(f"  Top 20%: >= {percentile_20:.3f}m")
print(f"  Top 30%: >= {percentile_30:.3f}m")

# Create output directory
output_dir = Path(r"C:\repos\powerpath\data\test_samples\adaptation\L2_asset_barriers")
output_dir.mkdir(parents=True, exist_ok=True)

depth_reduction = 1.0  # meters

# Create GeoDataFrames for each percentile group
groups = {
    'top_10': gdf_exposed[gdf_exposed['max_hazard'] >= percentile_10],
    'top_20': gdf_exposed[gdf_exposed['max_hazard'] >= percentile_20],
    'top_30': gdf_exposed[gdf_exposed['max_hazard'] >= percentile_30]
}

# Export each group
for group_name, gdf_group in groups.items():
    gdf_group_copy = gdf_group.copy()
    gdf_group_copy['depth_red'] = depth_reduction
    
    filename = f"exp_subs_{group_name}_{depth_reduction}m.geojson"
    output_path = output_dir / filename
    
    gdf_group_copy.to_file(output_path, driver='GeoJSON')
    
    print(f"\nExported {group_name}:")
    print(f"  File: {filename}")
    print(f"  Assets: {len(gdf_group_copy)}")
    print(f"  Hazard range: {gdf_group_copy['max_hazard'].min():.3f}m to {gdf_group_copy['max_hazard'].max():.3f}m")
# ── Export top-10/20/30% most critical substations by monetary & population impact ──
import numpy as np
import geopandas as gpd
from pathlib import Path

# ── 1. Score every exposed asset ──
# Monetary score: sum of (area × VOLL_PER_SQM) across land-use types
monetary_score = {}
for aid, lu_list in asset_to_lu.items():
    monetary_score[aid] = sum(area * rate for _, area, rate in lu_list)

# Population score: directly from the population map
population_score = dict(asset_population_map)

# ── 2. Restrict to exposed substations (max_hazard > 0.2 m) ──
exposed_ids = set(gdf_exposed.index)

# Build a GeoDataFrame of exposed assets with scores
gdf_exp = gdf_exposed.copy()
gdf_exp['monetary_score'] = gdf_exp.index.map(lambda aid: monetary_score.get(aid, 0))
gdf_exp['population_score'] = gdf_exp.index.map(lambda aid: population_score.get(aid, 0))

print(f"Exposed substations: {len(gdf_exp)}")
print(f"  Monetary score range: {gdf_exp['monetary_score'].min():.2f} – {gdf_exp['monetary_score'].max():.2f}")
print(f"  Population score range: {gdf_exp['population_score'].min():.0f} – {gdf_exp['population_score'].max():.0f}")

# ── 3. Identify top-N% by each metric ──
output_dir = Path(r"C:\repos\powerpath\data\test_samples\adaptation\L2_asset_barriers")
output_dir.mkdir(parents=True, exist_ok=True)

depth_reductions = [0.5, 1.0]
percentiles = {'top10': 0.90, 'top20': 0.80, 'top30': 0.70}

for metric_name, score_col, prefix in [
    ('monetary',   'monetary_score',   'mon_subs'),
    ('population', 'population_score', 'pop_subs'),
]:
    print(f"\n{'='*60}")
    print(f"Metric: {metric_name}")

    for pct_label, quantile in percentiles.items():
        threshold = gdf_exp[score_col].quantile(quantile)
        gdf_top = gdf_exp[gdf_exp[score_col] >= threshold].copy()

        print(f"\n  {pct_label} (>= {threshold:.2f}): {len(gdf_top)} assets")

        for depth_red in depth_reductions:
            gdf_out = gdf_top[['geometry', score_col]].copy()
            gdf_out['id'] = gdf_top.index
            gdf_out['depth_red'] = depth_red

            # Ensure CRS 28992
            if gdf_out.crs is None or gdf_out.crs.to_epsg() != 28992:
                gdf_out = gdf_out.to_crs(epsg=28992)

            filename = f"{prefix}_{pct_label}_{depth_red}m.geojson"
            gdf_out.to_file(output_dir / filename, driver='GeoJSON')
            print(f"    Exported: {filename}  ({len(gdf_out)} assets, depth_red={depth_red})")

print(f"\nAll files saved to: {output_dir}")