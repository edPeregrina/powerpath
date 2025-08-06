# Sensitivity Analysis and plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calculate_performance_metrics_over_time(timestep_results):
    """
    Calculate performance metrics averaged over time (area under the curve)
    This represents the total "loss of performance" during the simulation period.
    """
    if not timestep_results:
        return None, None, None
    
    # Convert timestep results to DataFrame
    df_timesteps = pd.DataFrame(timestep_results)
    
    # Group by timestep and calculate averages
    timestep_metrics = df_timesteps.groupby('timestep').agg({
        'operational': 'mean',  # Average operational status at each timestep
        'damage_ratio': 'mean',  # Average damage ratio at each timestep
        'repair_time': 'sum'     # Total repair backlog at each timestep
    }).reset_index()
    
    # Calculate area under the curve (average over time)
    avg_operational_over_time = timestep_metrics['operational'].mean() * 100  # Convert to percentage
    avg_damage_ratio_over_time = timestep_metrics['damage_ratio'].mean()
    avg_repair_backlog_over_time = timestep_metrics['repair_time'].mean()
    
    # Calculate performance loss metrics
    perfect_performance = 100.0  # 100% operational would be perfect
    performance_loss_area = perfect_performance - avg_operational_over_time
    
    return {
        'avg_operational_over_time': avg_operational_over_time,
        'avg_damage_ratio_over_time': avg_damage_ratio_over_time, 
        'avg_repair_backlog_over_time': avg_repair_backlog_over_time,
        'performance_loss_area': performance_loss_area,
        'timestep_metrics': timestep_metrics
    }

def plot_time_averaged_sensitivity_analysis(sensitivity_results):
    """
    Plot comprehensive sensitivity analysis using time-averaged performance metrics.
    
    Parameters:
    sensitivity_results: List of dictionaries containing simulation results with timestep data
    """
    import pandas as pd
    
    print("Starting sensitivity analysis plotting (Average Performance Over Time)...")
    print(f"sensitivity_results available: {len(sensitivity_results)} results")
    
    # Extract time-averaged performance metrics from all results
    performance_loss_values = []
    operational_avg_values = []
    damage_ratio_avg_values = []
    repair_backlog_avg_values = []
    
    for i, result in enumerate(sensitivity_results):
        if i < 3:  # Print first 3 for verification
            print(f"Result {i}: ", end="")
        
        try:
            if 'timestep_results' not in result:
                if i < 3:
                    print("No timestep_results found")
                continue
                
            timestep_data = result['timestep_results']
            
            # Convert dictionary to DataFrame
            df = pd.DataFrame(timestep_data)
            
            # Calculate time-averaged metrics (aggregate by timestep first)
            timestep_summary = df.groupby('timestep').agg({
                'operational': 'mean',
                'damage_ratio': 'mean',
                'repair_time': 'mean'
            }).reset_index()
            
            avg_operational = timestep_summary['operational'].mean() * 100  # Convert to percentage
            avg_damage_ratio = timestep_summary['damage_ratio'].mean()
            avg_repair_backlog = timestep_summary['repair_time'].mean()  # Using repair_time as proxy for backlog
            
            # Performance loss area = average loss over time
            performance_loss = (100 - avg_operational)  # Percentage loss
            
            operational_avg_values.append(avg_operational)
            damage_ratio_avg_values.append(avg_damage_ratio)
            repair_backlog_avg_values.append(avg_repair_backlog)
            performance_loss_values.append(performance_loss)
            
            if i < 3:
                print(f"{avg_operational:.2f}% avg operational, {avg_damage_ratio:.6f} avg damage ratio, {performance_loss:.2f}% performance loss")
                
        except Exception as e:
            if i < 3:
                print(f"Error processing result {i}: {e}")
            continue
    
    if len(operational_avg_values) == 0:
        print("No valid data found! Check the timestep_results structure.")
        return
    
    print(f"Total values extracted: {len(operational_avg_values)}")
    print(f"Average operational range: {min(operational_avg_values):.2f}% to {max(operational_avg_values):.2f}%")
    print(f"Performance loss range: {min(performance_loss_values):.2f}% to {max(performance_loss_values):.2f}%")
    print(f"Average damage ratio range: {min(damage_ratio_avg_values):.6f} to {max(damage_ratio_avg_values):.6f}")
    
    # Duplicate the ranges for verification
    print(f"Total values extracted: {len(operational_avg_values)}")
    print(f"Average operational range: {min(operational_avg_values):.2f}% to {max(operational_avg_values):.2f}%")
    print(f"Performance loss range: {min(performance_loss_values):.2f}% to {max(performance_loss_values):.2f}%")
    print(f"Average damage ratio range: {min(damage_ratio_avg_values):.6f} to {max(damage_ratio_avg_values):.6f}")
    
    # Create comprehensive visualization for time-averaged performance
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sensitivity Analysis: Average Performance Over Time (Loss of Performance Area)', fontsize=16, fontweight='bold')
    
    # Row 1: Performance distributions
    # Plot 1: Average Operational Status Distribution (zoomed out for better interpretability)
    axes[0, 0].hist(operational_avg_values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(operational_avg_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(operational_avg_values):.2f}%')
    axes[0, 0].set_title('Average Operational Status Distribution')
    axes[0, 0].set_xlabel('Average Operational Assets Over Time (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    # Zoom out the x-axis for better interpretability
    x_min, x_max = min(operational_avg_values), max(operational_avg_values)
    x_range = x_max - x_min
    axes[0, 0].set_xlim(x_min - 0.3 * x_range, x_max + 0.3 * x_range)
    
    # Plot 2: Performance Loss by Flood Threshold (box plot)
    flood_thresholds = [0.2, 0.3, 0.4, 0.5]
    groups = []
    for ft in flood_thresholds:
        group_values = []
        for result in sensitivity_results:
            if result['flood_threshold'] == ft and 'timestep_results' in result:
                timestep_data = result['timestep_results']
                df = pd.DataFrame(timestep_data)
                timestep_summary = df.groupby('timestep')['operational'].mean()
                avg_operational = timestep_summary.mean() * 100
                performance_loss = 100 - avg_operational
                group_values.append(performance_loss)
        groups.append(group_values)
    
    bp1 = axes[0, 1].boxplot(groups, labels=flood_thresholds, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightcoral')
    axes[0, 1].set_title('Performance Loss by Flood Threshold')
    axes[0, 1].set_xlabel('Flood Threshold')
    axes[0, 1].set_ylabel('Performance Loss Area (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Performance Loss vs Repair Threshold (scatter plot)
    repair_thresholds = []
    loss_for_scatter = []
    flood_for_color = []
    
    for result in sensitivity_results:
        if 'timestep_results' in result:
            timestep_data = result['timestep_results']
            df = pd.DataFrame(timestep_data)
            timestep_summary = df.groupby('timestep')['operational'].mean()
            avg_operational = timestep_summary.mean() * 100
            performance_loss = 100 - avg_operational
            repair_thresholds.append(result['repair_threshold'])
            loss_for_scatter.append(performance_loss)
            flood_for_color.append(result['flood_threshold'])
    
    scatter1 = axes[0, 2].scatter(repair_thresholds, loss_for_scatter, c=flood_for_color, cmap='viridis', alpha=0.7)
    axes[0, 2].set_title('Performance Loss vs Repair Threshold')
    axes[0, 2].set_xlabel('Repair Threshold')
    axes[0, 2].set_ylabel('Performance Loss Area (%)')
    axes[0, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 2], label='Flood Threshold')
    
    # Row 2: Time-averaged Damage and Repair Analysis
    # Plot 4: Average Damage Ratio Distribution
    axes[1, 0].hist(damage_ratio_avg_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].axvline(np.mean(damage_ratio_avg_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(damage_ratio_avg_values):.6f}')
    axes[1, 0].set_title('Average Damage Ratio Distribution')
    axes[1, 0].set_xlabel('Average Damage Ratio Over Time')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 5: Average Repair Backlog by Flood Threshold
    repair_groups = []
    for ft in flood_thresholds:
        group_values = []
        for result in sensitivity_results:
            if result['flood_threshold'] == ft and 'timestep_results' in result:
                timestep_data = result['timestep_results']
                df = pd.DataFrame(timestep_data)
                timestep_summary = df.groupby('timestep')['repair_time'].mean()
                avg_repair_backlog = timestep_summary.mean()
                group_values.append(avg_repair_backlog)
        repair_groups.append(group_values)
    
    bp2 = axes[1, 1].boxplot(repair_groups, labels=flood_thresholds, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('orange')
    axes[1, 1].set_title('Average Repair Backlog by Flood Threshold')
    axes[1, 1].set_xlabel('Flood Threshold')
    axes[1, 1].set_ylabel('Average Repair Backlog (hours)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics table with time-averaged metrics
    axes[1, 2].axis('off')
    summary_text = f"""TIME-AVERAGED PERFORMANCE METRICS

Total Runs: {len(operational_avg_values)}

Average Operational Status (%):
  Mean: {np.mean(operational_avg_values):.3f}
  Std Dev: {np.std(operational_avg_values):.3f}
  Min: {np.min(operational_avg_values):.3f}
  Max: {np.max(operational_avg_values):.3f}

Performance Loss Area (%):
  Mean: {np.mean(performance_loss_values):.3f}
  Std Dev: {np.std(performance_loss_values):.3f}
  Min: {np.min(performance_loss_values):.3f}
  Max: {np.max(performance_loss_values):.3f}

Average Damage Ratio:
  Mean: {np.mean(damage_ratio_avg_values):.8f}
  Std Dev: {np.std(damage_ratio_avg_values):.8f}

Average Repair Backlog (hours):
  Mean: {np.mean(repair_backlog_avg_values):.3f}
  Std Dev: {np.std(repair_backlog_avg_values):.3f}"""
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, fontsize=10, 
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 2].set_title('Time-Averaged Statistics')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis for time-averaged metrics
    print("\n" + "="*60)
    print("DETAILED TIME-AVERAGED SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Create DataFrame for analysis
    analysis_data = []
    for result in sensitivity_results:
        if 'timestep_results' in result:
            timestep_data = result['timestep_results']
            df = pd.DataFrame(timestep_data)
            timestep_summary = df.groupby('timestep').agg({
                'operational': 'mean',
                'damage_ratio': 'mean',
                'repair_time': 'mean'
            })
            
            avg_operational = timestep_summary['operational'].mean() * 100
            performance_loss = 100 - avg_operational
            
            analysis_data.append({
                'flood_threshold': result['flood_threshold'],
                'repair_threshold': result['repair_threshold'],
                'avg_operational': avg_operational,
                'performance_loss': performance_loss,
                'avg_damage_ratio': timestep_summary['damage_ratio'].mean(),
                'avg_repair_backlog': timestep_summary['repair_time'].mean()
            })
    
    # Parameter sensitivity analysis for time-averaged metrics
    print(f"\nParameter Sensitivity (Time-Averaged Metrics):")
    
    for metric in ['avg_operational', 'performance_loss', 'avg_damage_ratio', 'avg_repair_backlog']:
        print(f"\n=== {metric.upper().replace('_', ' ')} ANALYSIS ===")
        
        param_df = pd.DataFrame(analysis_data)
        
        for param in ['flood_threshold', 'repair_threshold']:
            print(f"\nEffect of {param}:")
            grouped = param_df.groupby(param)[metric].agg(['mean', 'std']).reset_index()
            
            for _, row in grouped.iterrows():
                print(f"  {param} = {row[param]}: mean = {row['mean']:.4f}, std = {row['std']:.4f}")
    
    # Find best and worst performers (time-averaged)
    df_analysis = pd.DataFrame(analysis_data)
    
    best_result = df_analysis.loc[df_analysis['avg_operational'].idxmax()]
    worst_result = df_analysis.loc[df_analysis['avg_operational'].idxmin()]
    
    print(f"\n" + "="*50)
    print("BEST PERFORMING CONFIGURATION (Time-Averaged):")
    print(f"  Flood threshold: {best_result['flood_threshold']}")
    print(f"  Repair threshold: {best_result['repair_threshold']}")
    print(f"  Average operational: {best_result['avg_operational']:.2f}%")
    print(f"  Performance loss: {best_result['performance_loss']:.2f}%")
    
    print(f"\nWORST PERFORMING CONFIGURATION (Time-Averaged):")
    print(f"  Flood threshold: {worst_result['flood_threshold']}")
    print(f"  Repair threshold: {worst_result['repair_threshold']}")
    print(f"  Average operational: {worst_result['avg_operational']:.2f}%")
    print(f"  Performance loss: {worst_result['performance_loss']:.2f}%")
    
    print(f"\nTime-averaged sensitivity analysis completed successfully!")
    print(f"Performance Loss Area represents the average deviation from 100% operational status over the entire simulation period.")
    print(f"This captures the cumulative impact of flooding events and recovery under different disruption/repair assumptions.")



# Function to run sensitivity analysis
def run_sensitivity_analysis(flood_thresholds, repair_thresholds, damage_ratio_coefficients, repair_time_coefficients):
    import tqdm
    #from powerpath.simulation import simulate_asset_damage_recovery

    results = []
    for flood_threshold in tqdm.tqdm(flood_thresholds, desc="Flood Thresholds"):
        # Uncomment the next line if you want to see the flood threshold being processed
        # print(f"Processing flood threshold: {flood_threshold}")
        
        # Run the simulation for each combination of parameters
        # gdf_msls_2 is assumed to be defined in the context where this function is called
        # If not, you need to define it or pass it as an argument
        #flood_thresholds:
        for repair_threshold in repair_thresholds:
            for damage_ratio_coeff in damage_ratio_coefficients:
                for repair_time_coeff in repair_time_coefficients:
                    gdf_msls_timesteps, timestep_results = simulate_asset_damage_recovery(
                        gdf_msls_2,  # GeoDataFrame with station geometries
                        hazard_path,  # Path to hazard maps
                        timesteps=np.arange(0, 7*24+1, 1),  # 24-hour timesteps for 7 days
                        verbose=False,  # Suppress progress messages
                        record_results=True,  # Record detailed results for time series analysis
                        flood_threshold=flood_threshold,
                        repair_threshold=repair_threshold,
                        damage_ratio_coefficients=damage_ratio_coeff,
                        repair_time_coefficients=repair_time_coeff
                    )
                    results.append({
                        'flood_threshold': flood_threshold,
                        'repair_threshold': repair_threshold,
                        'damage_ratio_coefficients': damage_ratio_coeff,
                        'repair_time_coefficients': repair_time_coeff,
                        'results': gdf_msls_timesteps,
                        'timestep_results': timestep_results  # Store timestep data for time series analysis
                    })
    return results



# Sensitivity analysis flood threshold, repair threshold

flood_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
repair_thresholds = [1.9, 1.9891, 2.0, 2.1]
damage_ratio_coefficients = [(0.0468, 0.0077)]  
repair_time_coefficients = [(702.72, 3.14, 1.9891)]

# Run the sensitivity analysis
sensitivity_results = run_sensitivity_analysis(
    flood_thresholds, 
    repair_thresholds, 
    damage_ratio_coefficients, 
    repair_time_coefficients
)

# Display summary of the results
print(f"Sensitivity analysis complete! Generated {len(sensitivity_results)} parameter combinations.")

for i, result in enumerate(sensitivity_results[:3]):  # Show first 3 for brevity
    print(f"\nResult {i+1}:")
    print(f"  Flood Threshold: {result['flood_threshold']}")
    print(f"  Repair Threshold: {result['repair_threshold']}")
    print(f"  Damage Ratio Coefficients: {result['damage_ratio_coefficients']}")
    print(f"  Repair Time Coefficients: {result['repair_time_coefficients']}")
    gdf = result['results']
    print(f"  Final operational assets: {gdf['operational'].mean()*100:.1f}%")
    print(f"  Final average damage ratio: {gdf['damage_ratio'].mean():.4f}")

print(f"\n... and {len(sensitivity_results)-3} more combinations.")


# Debug: Check the structure of timestep_results
if 'sensitivity_results' in locals() and sensitivity_results:
    print("Debugging timestep_results structure...")
    
    for i, result in enumerate(sensitivity_results[:1]):  # Check first result
        print(f"\nResult {i}:")
        print(f"  Keys: {result.keys()}")
        
        if 'timestep_results' in result:
            timestep_data = result['timestep_results']
            print(f"  timestep_results type: {type(timestep_data)}")
            
            if isinstance(timestep_data, dict):
                print(f"  timestep_results keys: {list(timestep_data.keys())}")
                
                # Check each key's structure
                for key in list(timestep_data.keys())[:5]:  # First 5 keys
                    value = timestep_data[key]
                    print(f"    {key}: type={type(value)}, length={len(value) if hasattr(value, '__len__') else 'N/A'}")
                    if hasattr(value, '__len__') and len(value) > 0:
                        print(f"      Sample values: {value[:3] if isinstance(value, list) else 'N/A'}")
            else:
                print(f"  timestep_results content: {timestep_data}")
        else:
            print("  No timestep_results found")
else:
    print("sensitivity_results not found")

# Run the plotting function if sensitivity_results is available
if 'sensitivity_results' in locals() and sensitivity_results:
    plot_time_averaged_sensitivity_analysis(sensitivity_results)
else:
    print("sensitivity_results not found. Please run the sensitivity analysis first.")

