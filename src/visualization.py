"""
Visualization functions for electricity infrastructure resilience analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

def plot_simulation_results_summary(results_df, gdf_assets, config=None, save_path=None):
    """
    Create a 2x2 summary plot of simulation results.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results
        gdf_assets: GeoDataFrame of assets (used for total count)
        config (dict, optional): Configuration dictionary
        save_path (str/Path, optional): Path to save the plot
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convert timestep to hours for x-axis
    time_hours = results_df['timestep']
    total_assets = len(gdf_assets)
    
    # Plot 1: Operational vs Total Assets
    axes[0, 0].plot(time_hours, results_df['operational_count'], 'b-', linewidth=2, label='Operational')
    axes[0, 0].axhline(y=total_assets, color='k', linestyle='--', linewidth=1, label=f'Total Assets ({total_assets})')
    axes[0, 0].set_xlabel('Time (Hours)')
    axes[0, 0].set_ylabel('Number of Assets')
    axes[0, 0].set_title('Operational Assets Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add day markers as vertical lines
    for day in range(1, int(time_hours.max()//24) + 1):
        axes[0, 0].axvline(x=day*24, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Plot 2: Accessibility and Flooding
    axes[0, 1].plot(time_hours, results_df['accessible_count'], 'g-', linewidth=2, label='Accessible')
    axes[0, 1].plot(time_hours, results_df['flooded_count'], 'orange', linewidth=2, label='Flooded')
    axes[0, 1].set_xlabel('Time (Hours)')
    axes[0, 1].set_ylabel('Number of Assets')
    axes[0, 1].set_title('Accessibility and Flooding')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add day markers
    for day in range(1, int(time_hours.max()//24) + 1):
        axes[0, 1].axvline(x=day*24, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Plot 3: Damage and Repair Activity
    axes[1, 0].plot(time_hours, results_df['damaged_count'], 'r-', linewidth=2, label='Damaged')
    axes[1, 0].plot(time_hours, results_df['crews_assigned_count'], 'purple', linewidth=2, label='Crews Assigned')
    axes[1, 0].set_xlabel('Time (Hours)')
    axes[1, 0].set_ylabel('Number of Assets/Crews')
    axes[1, 0].set_title('Damage and Repair Crew Assignment')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add day markers
    for day in range(1, int(time_hours.max()//24) + 1):
        axes[1, 0].axvline(x=day*24, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Plot 4: Repair Backlog and Average Repair Time
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    # Use actual repair backlog if available, otherwise fallback to estimated
    if 'total_repair_backlog' in results_df.columns:
        line1 = ax1.plot(time_hours, results_df['total_repair_backlog'], 'orange', linewidth=2, label='Total Repair Backlog (hrs)')
        ax1.set_ylabel('Total Repair Backlog (hours)', color='orange')
    else:
        # Fallback to average damage ratio if backlog not available
        line1 = ax1.plot(time_hours, results_df['avg_damage_ratio'], 'red', linewidth=2, label='Avg Damage Ratio')
        ax1.set_ylabel('Average Damage Ratio', color='red')
    
    line2 = ax2.plot(time_hours, results_df['avg_repair_time'], 'blue', linewidth=2, label='Avg Repair Time (hrs)')
    
    ax1.set_xlabel('Time (Hours)')
    ax2.set_ylabel('Average Repair Time (hours)', color='blue')
    ax1.set_title('Repair Backlog and Average Repair Time')
    
    # Add day markers
    for day in range(1, int(time_hours.max()//24) + 1):
        ax1.axvline(x=day*24, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig


def plot_detailed_analysis_panels(results_df, gdf_assets, config=None, save_path=None):
    """
    Create a detailed 6-panel analysis visualization.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results
        gdf_assets: GeoDataFrame of assets (used for total count)
        config (dict, optional): Configuration dictionary
        save_path (str/Path, optional): Path to save the plot
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    total_assets = len(gdf_assets)
    max_crews = config['simulation_config']['number_repair_crews'] if config else 10
    
    # Prepare data for visualization
    timestep_metrics = results_df.copy()
    
    # Convert counts to percentages
    timestep_metrics['operational'] = timestep_metrics['operational_count'] / total_assets
    timestep_metrics['accessible'] = timestep_metrics['accessible_count'] / total_assets
    
    # Damage and repair metrics 
    timestep_metrics['damage_ratio'] = timestep_metrics['avg_damage_ratio']
    timestep_metrics['repair_time'] = timestep_metrics.get('total_repair_backlog', 
                                                          timestep_metrics['avg_repair_time'] * timestep_metrics['damaged_count'])
    
    # Flooding percentage data 
    timestep_flooding = results_df.copy()
    timestep_flooding['hazard_value'] = (timestep_flooding['flooded_count'] / total_assets * 100)
    
    # Create the enhanced visualization with 6 panels
    fig, axes = plt.subplots(6, 1, figsize=(12, 24))
    fig.suptitle('Asset Performance Metrics Over Time', 
                 fontsize=16, fontweight='bold', y=0.99)
    plt.subplots_adjust(top=0.96, bottom=0.04, hspace=0.8)
    
    # Day markers
    day_markers = timestep_metrics[timestep_metrics['timestep'] % 24 == 0]['timestep']
    
    # Panel 1: Percentage Flooded 
    ax1 = axes[0]
    ax1.plot(timestep_flooding['timestep'], timestep_flooding['hazard_value'], 
             'purple', linewidth=2, label='Percentage Flooded')
    ax1.fill_between(timestep_flooding['timestep'], timestep_flooding['hazard_value'], 
                     alpha=0.3, color='purple')
    ax1.set_xlabel('Timestep (hours)')
    ax1.set_ylabel('Percentage Flooded (%)')
    ax1.set_title('Panel 1: Percentage of Assets Flooded')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(timestep_flooding['hazard_value']) * 1.1 if max(timestep_flooding['hazard_value']) > 0 else 5)
    
    for day_marker in day_markers:
        ax1.axvline(x=day_marker, color='red', linestyle='--', alpha=0.5)
    
    # Panel 2: Operational Status (%)
    ax2 = axes[1]
    y_limit_min = max(0, (timestep_metrics['operational'] * 100).min() - 5)
    ax2.plot(timestep_metrics['timestep'], timestep_metrics['operational'] * 100, 
             'b-', linewidth=2, label='Operational Assets')
    ax2.fill_between(timestep_metrics['timestep'], timestep_metrics['operational'] * 100, 
                     alpha=0.3, color='blue')
    ax2.set_xlabel('Timestep (hours)')
    ax2.set_ylabel('Operational Assets (%)')
    ax2.set_title('Panel 2: Average Operational Status')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(y_limit_min, 100)
    
    for day_marker in day_markers:
        ax2.axvline(x=day_marker, color='red', linestyle='--', alpha=0.5)
    
    # Panel 3: Average Damage Ratio
    ax3 = axes[2]
    ax3.plot(timestep_metrics['timestep'], timestep_metrics['damage_ratio'], 
             'r-', linewidth=2, label='Damage Ratio')
    ax3.fill_between(timestep_metrics['timestep'], timestep_metrics['damage_ratio'], 
                     alpha=0.3, color='red')
    ax3.set_xlabel('Timestep (hours)')
    ax3.set_ylabel('Damage Ratio')
    ax3.set_title('Panel 3: Average Damage Ratio')
    ax3.grid(True, alpha=0.3)
    
    for day_marker in day_markers:
        ax3.axvline(x=day_marker, color='red', linestyle='--', alpha=0.5)
    
    # Panel 4: Repair Backlog
    ax4 = axes[3]
    actual_repair_backlog = timestep_metrics['repair_time']
    
    ax4.plot(timestep_metrics['timestep'], actual_repair_backlog, 
             'orange', linewidth=2, label='Repair Backlog')
    ax4.fill_between(timestep_metrics['timestep'], actual_repair_backlog, 
                     alpha=0.3, color='orange')
    ax4.set_xlabel('Timestep (hours)')
    ax4.set_ylabel('Total Repair Hours Remaining')
    ax4.set_title('Panel 4: Repair Backlog')
    ax4.grid(True, alpha=0.3)
    
    for day_marker in day_markers:
        ax4.axvline(x=day_marker, color='red', linestyle='--', alpha=0.5)
    
    # Panel 5: Repair Crew Utilization
    ax5 = axes[4]
    active_crews = timestep_metrics['crews_assigned_count']
    idle_crews = max_crews - active_crews
    
    # Create stacked area plot
    ax5.fill_between(timestep_metrics['timestep'], 0, idle_crews, 
                    alpha=0.7, color='lightcoral', label='Idle Crews')
    ax5.fill_between(timestep_metrics['timestep'], idle_crews, max_crews, 
                    alpha=0.7, color='darkgreen', label='Active Crews')
    
    ax5.plot(timestep_metrics['timestep'], idle_crews, 
            'r-', linewidth=1, alpha=0.8)
    
    ax5.set_xlabel('Timestep (hours)')
    ax5.set_ylabel('Number of Repair Crews')
    ax5.set_title('Panel 5: Repair Crew Utilization (Idle vs Active)')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, max_crews)
    ax5.legend(loc='upper right')
    
    for day_marker in day_markers:
        ax5.axvline(x=day_marker, color='red', linestyle='--', alpha=0.5)
    
    # Panel 6: Average Accessibility
    ax6 = axes[5]
    ax6.plot(timestep_metrics['timestep'], timestep_metrics['accessible'] * 100, 
             'g-', linewidth=2, label='Accessible Assets')
    ax6.fill_between(timestep_metrics['timestep'], timestep_metrics['accessible'] * 100, 
                     alpha=0.3, color='green')
    ax6.set_xlabel('Timestep (hours)')
    ax6.set_ylabel('Accessible Assets (%)')
    ax6.set_title('Panel 6: Average Accessibility')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 105)
    
    for day_marker in day_markers:
        ax6.axvline(x=day_marker, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig

def print_simulation_summary(results_df, gdf_assets, config=None):
    """
    Print comprehensive simulation summary statistics.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results
        gdf_assets: GeoDataFrame of assets
        config (dict, optional): Configuration dictionary
    """
    total_assets = len(gdf_assets)
    
    print("=" * 70)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 70)
    
    # Basic metrics
    print(f"Total assets: {total_assets}")
    print(f"Simulation duration: {len(results_df)} hours ({len(results_df) // 24} days)")
    print(f"Final operational rate: {results_df['operational_count'].iloc[-1] / total_assets * 100:.1f}%")
    print(f"Peak damaged assets: {results_df['damaged_count'].max()}")
    print(f"Peak flooded assets: {results_df['flooded_count'].max()}")
    print(f"Total crew assignments: {results_df['crews_assigned_count'].sum()}")
    
    # Recovery timeline
    print("\nRECOVERY TIMELINE (Daily Summary)")
    print("=" * 40)
    for day in range(int(results_df['day'].max()) + 1):
        day_data = results_df[results_df['day'] == day]
        if len(day_data) > 0:
            end_of_day = day_data.iloc[-1]
            operational_rate = end_of_day['operational_count'] / total_assets * 100
            flooded_count = end_of_day['flooded_count']
            damaged_count = end_of_day['damaged_count']
            print(f"Day {day}: {operational_rate:.1f}% operational, {flooded_count} flooded, {damaged_count} damaged")
    
    # Repair backlog analysis
    print("\n" + "="*70)
    print("REPAIR BACKLOG ANALYSIS")
    print("="*70)
    
    sample_indices = range(0, len(results_df), 24)
    print("Hour | Flooded | Damaged | Crews | Backlog | Effective?")
    print("-" * 55)
    
    for i in sample_indices:
        row = results_df.iloc[i]
        backlog = row.get('total_repair_backlog', row['avg_repair_time'] * row['damaged_count'])
        effective = "Yes" if row['flooded_count'] == 0 and row['crews_assigned_count'] > 0 else "No"
        print(f"{row['timestep']:4} | {row['flooded_count']:7} | {row['damaged_count']:7} | {row['crews_assigned_count']:5} | {backlog:7.1f} | {effective}")
    
    # Performance metrics
    flooding_periods = results_df[results_df['flooded_count'] > 0]
    clear_periods = results_df[results_df['flooded_count'] == 0]
    
    print(f"\nSUMMARY:")
    if len(flooding_periods) > 0:
        avg_crews_flood = flooding_periods['crews_assigned_count'].mean()
        print(f"During flooding ({len(flooding_periods)} hours): {avg_crews_flood:.1f} crews assigned on average")
    
    if len(clear_periods) > 0:
        avg_crews_clear = clear_periods['crews_assigned_count'].mean()
        print(f"During clear periods ({len(clear_periods)} hours): {avg_crews_clear:.1f} crews assigned on average")
    
    if 'total_repair_backlog' in results_df.columns:
        initial_backlog = results_df['total_repair_backlog'].iloc[0]
        final_backlog = results_df['total_repair_backlog'].iloc[-1]
        peak_backlog = results_df['total_repair_backlog'].max()
        print(f"Repair backlog: Start={initial_backlog:.1f}hrs, Peak={peak_backlog:.1f}hrs, End={final_backlog:.1f}hrs")
    
    print("=" * 70)

def create_comprehensive_visualization(results_df, gdf_assets, config=None, save_path=None):
    """
    Create comprehensive 2x3 visualization for publication/reporting.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing simulation results
        gdf_assets: GeoDataFrame of assets
        config (dict, optional): Configuration dictionary
        save_path (str/Path, optional): Path to save the plot
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Electricity Infrastructure Resilience Analysis', fontsize=16, fontweight='bold')
    
    total_assets = len(gdf_assets)
    timesteps_hours = results_df.index if hasattr(results_df, 'index') else range(len(results_df))
    
    # Panel 1: Operational Assets Over Time
    ax1 = axes[0, 0]
    if 'operational_count' in results_df.columns:
        ax1.plot(timesteps_hours, results_df['operational_count'], 'g-', linewidth=2, label='Operational Assets')
        ax1.set_title('(a) Operational Assets Over Time', fontweight='bold')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Number of Assets')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Panel 2: Flooded Assets Over Time  
    ax2 = axes[0, 1]
    if 'flooded_count' in results_df.columns:
        ax2.plot(timesteps_hours, results_df['flooded_count'], 'b-', linewidth=2, label='Flooded Assets')
        ax2.fill_between(timesteps_hours, results_df['flooded_count'], alpha=0.3, color='blue')
        ax2.set_title('(b) Flooded Assets Over Time', fontweight='bold')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Number of Assets')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Panel 3: Repair Activities
    ax3 = axes[0, 2]
    if 'crews_assigned_count' in results_df.columns and 'damaged_count' in results_df.columns:
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(timesteps_hours, results_df['crews_assigned_count'], 'green', linewidth=2, label='Crews Assigned')
        line2 = ax3_twin.plot(timesteps_hours, results_df['damaged_count'], 'red', linewidth=2, label='Damaged Assets', linestyle='--')
        
        ax3.set_title('(c) Repair Activities Over Time', fontweight='bold')
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Active Repair Crews', color='green')
        ax3_twin.set_ylabel('Damaged Assets', color='red')
        ax3.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
    
    # Panel 4: Asset Status Comparison
    ax4 = axes[1, 0]
    if 'damaged_count' in results_df.columns and 'operational_count' in results_df.columns:
        ax4.plot(timesteps_hours, results_df['damaged_count'], 'r-', linewidth=2, label='Damaged Assets')
        ax4.plot(timesteps_hours, results_df['operational_count'], 'g-', linewidth=2, label='Operational Assets')
        ax4.set_title('(d) Asset Status Comparison', fontweight='bold')
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Number of Assets')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # Panel 5: Cumulative Repairs
    ax5 = axes[1, 1]
    if 'crews_assigned_count' in results_df.columns:
        repair_work = results_df['crews_assigned_count'].copy()
        cumulative_repairs = repair_work.cumsum()
        
        ax5.plot(timesteps_hours, cumulative_repairs, 'purple', linewidth=2, label='Cumulative Repairs')
        ax5.fill_between(timesteps_hours, cumulative_repairs, alpha=0.05, color='purple')
        
        # Add flooding shading
        if 'flooded_count' in results_df.columns:
            max_repairs = cumulative_repairs.max()
            flooding_legend_added = False
            
            for day in range(0, len(timesteps_hours), 24):
                day_end = min(day + 24, len(timesteps_hours))
                day_data = results_df.iloc[day:day_end]
                
                if day_data['flooded_count'].max() > 0:
                    avg_flooding = day_data['flooded_count'].mean()
                    max_flooding = results_df['flooded_count'].max()
                    alpha_intensity = 0.15 + (avg_flooding / max_flooding * 0.25) if max_flooding > 0 else 0.2
                    
                    x_values = list(range(day, day_end))
                    if day_end < len(timesteps_hours):
                        x_values.append(day_end)
                    
                    ax5.fill_between(x_values, 0, max_repairs * 1.1,
                                   alpha=alpha_intensity, color='darkblue', 
                                   label='Flooding Period' if not flooding_legend_added else "")
                    
                    if not flooding_legend_added:
                        flooding_legend_added = True
        
        ax5.set_title('(e) Cumulative Repairs Executed', fontweight='bold')
        ax5.set_xlabel('Time (hours)')
        ax5.set_ylabel('Cumulative Repair Work')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        max_repairs = cumulative_repairs.max()
        ax5.text(0.7, 0.9, f'Total Repairs: {max_repairs:.0f}', 
                transform=ax5.transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Panel 6: Recovery Effectiveness
    ax6 = axes[1, 2]
    if 'operational_count' in results_df.columns:
        operational_percentage = (results_df['operational_count'] / total_assets * 100) if total_assets > 0 else results_df['operational_count']
        
        ax6.plot(timesteps_hours, operational_percentage, 'purple', linewidth=2, label='Operational %')
        ax6.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Full Operation')
        ax6.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% Threshold')
        ax6.set_title('(f) Recovery Effectiveness', fontweight='bold')
        ax6.set_xlabel('Time (hours)')
        ax6.set_ylabel('Operational Percentage (%)')
        ax6.set_ylim(0, 105)
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    plt.tight_layout()
    
    if save_path:
        # Convert Path object to string for string operations
        save_path_str = str(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    return fig



def save_all_visualizations(results_df, gdf_assets, config, output_dir):
    """
    Generate and save all visualization types to the output directory.
    
    Args:
        results_df (pd.DataFrame): Simulation results
        gdf_assets: GeoDataFrame of assets
        config (dict): Configuration dictionary
        output_dir (Path): Directory to save outputs
    
    Returns:
        list: List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_files = []
    
    # Create a single PDF with all figures
    pdf_path = output_dir / f"all_visualizations_{timestamp}.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Summary plot (2x2)
        fig1 = plot_simulation_results_summary(results_df, gdf_assets, config)
        fig1.suptitle('Simulation Results Summary', fontsize=16, fontweight='bold')
        pdf.savefig(fig1, bbox_inches='tight', facecolor='white')
        
        # Also save individual PNG
        summary_path = output_dir / f"simulation_summary_{timestamp}.png"
        fig1.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
        saved_files.append(summary_path)
        plt.close(fig1)
        
        # Page 2: Detailed analysis (6-panel)
        fig2 = plot_detailed_analysis_panels(results_df, gdf_assets, config)
        pdf.savefig(fig2, bbox_inches='tight', facecolor='white')
        
        # Also save individual PNG
        detailed_path = output_dir / f"detailed_analysis_{timestamp}.png"
        fig2.savefig(detailed_path, dpi=300, bbox_inches='tight', facecolor='white')
        saved_files.append(detailed_path)
        plt.close(fig2)
        
        # Page 3: Comprehensive visualization (2x3)
        fig3 = create_comprehensive_visualization(results_df, gdf_assets, config)
        pdf.savefig(fig3, bbox_inches='tight', facecolor='white')
        
        # Also save individual PNG
        comprehensive_path = output_dir / f"comprehensive_visualization_{timestamp}.png"
        fig3.savefig(comprehensive_path, dpi=300, bbox_inches='tight', facecolor='white')
        saved_files.append(comprehensive_path)
        plt.close(fig3)
        
        # Add metadata to PDF
        d = pdf.infodict()
        d['Title'] = 'Electricity Resilience Analysis with Accessibility'
        d['Author'] = 'D. Peregrina, L. Meijer'
        d['Subject'] = 'Infrastructure Resilience Visualization Results'
        d['Keywords'] = 'Infrastructure, Resilience, Flooding, Recovery, Analysis'
        d['CreationDate'] = datetime.now()
    
    saved_files.append(pdf_path)
    
    print(f"Saved {len(saved_files)} files to {output_dir}")
    print(f"   Combined PDF: {pdf_path.name}")
    print(f"   Individual PNGs: {len(saved_files)-1} files")
    
    return saved_files

