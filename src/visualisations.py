"""
Visualization functions for electricity infrastructure resilience analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from ema_workbench.analysis.plotting import lines
from matplotlib.patches import Wedge
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import contextily as ctx


# ============================================================================
# Core Configuration
# ============================================================================

MONETARY_OUTCOMES = [
    'monetary_impact_total',
    'monetary_impact_residential',
    'monetary_impact_commercial',
    'monetary_impact_industrial',
    'monetary_impact_transport',
    'monetary_impact_public_sector',
]

NON_MONETARY_OUTCOMES = [
    'flooded',
    'operational',
    'unreachable',
    'accessible',
    'crew_assigned',
    'affected_population',
    'served_population',
    'affected_population_ratio',
]

DEFAULT_COLORS = {
    'repair_crews': 'orange',
    'fragility': 'purple',
    'policy': 'green',
    'default': 'red',
}


# ============================================================================
# Helper Functions
# ============================================================================

def _create_int_bins(min_val: int, max_val: int, bin_size: int) -> List[Tuple[int, int]]:
    """Create non-overlapping bins for integer ranges."""
    bins = []
    start = min_val
    while start <= max_val:
        end = min(start + bin_size - 1, max_val)
        bins.append((start, end))
        start = end + 1
    return bins


def _create_float_bins(min_val: float, max_val: float, bin_size: float) -> List[Tuple[float, float]]:
    """Create bins for float ranges."""
    bins = []
    val = min_val
    while val < max_val:
        bins.append((val, min(val + bin_size, max_val)))
        val += bin_size
    return bins


def _aggregate_outcomes(outcomes: Dict, outcomes_to_plot: List[str]) -> None:
    """Aggregate 3D outcome arrays to 2D."""
    for key in outcomes_to_plot:
        if key not in outcomes:
            continue
            
        arr = outcomes[key]
        if arr.ndim == 3:
            if key.startswith('monetary_impact_'):
                outcomes[key] = arr.sum(axis=2)  # Sum monetary impacts
            else:
                outcomes[key] = arr.mean(axis=2)  # Mean for others


def _filter_valid_bins(
    experiments_df: pd.DataFrame,
    outcomes: Dict,
    outcomes_to_show: List[str],
    group_by: str,
    bin_ranges: List,
    is_categorical: bool
) -> List:
    """Filter bins to ensure they contain valid data."""
    valid_bins = []
    
    for bin_spec in bin_ranges:
        # Create mask
        if is_categorical:
            mask = experiments_df[group_by].astype(str) == str(bin_spec)
        else:
            mask = (experiments_df[group_by] >= bin_spec[0]) & (experiments_df[group_by] <= bin_spec[1])
        
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue
        
        # Check data validity
        valid = True
        for outcome_name in outcomes_to_show:
            if outcome_name not in outcomes:
                valid = False
                break
            data = outcomes[outcome_name][indices]
            if data.size == 0 or np.all(np.isnan(data)):
                valid = False
                break
        
        if valid:
            valid_bins.append(bin_spec)
    
    return valid_bins


def _apply_highlighting(
    axes: Dict,
    highlight_bin_idx: int,
    highlight_color: str,
    n_outcomes: int
) -> None:
    """Apply highlighting to specific bin in all axes."""
    for ax in axes.values():
        lines_list = ax.get_lines()
        if len(lines_list) == 0:
            continue
        
        n_bins = len(lines_list) // n_outcomes if n_outcomes > 0 else 0
        if n_bins == 0:
            continue
        
        # Grey out all lines first
        for line in lines_list:
            line.set_color('lightgrey')
            line.set_alpha(0.3)
            line.set_linewidth(0.2)
            line.set_zorder(1)
        
        # Highlight selected bin
        start_idx = highlight_bin_idx * n_outcomes
        end_idx = start_idx + n_outcomes
        
        for i in range(start_idx, min(end_idx, len(lines_list))):
            line = lines_list[i]
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            line.remove()
            ax.plot(xdata, ydata, 
                   color=highlight_color,
                   alpha=0.9,
                   linewidth=0.5,
                   zorder=10)


# ============================================================================
# Main Plotting Function
# ============================================================================

def plot_ema_outcomes(
    experiments_df: pd.DataFrame,
    outcomes: Dict,
    outcomes_to_plot: List[str],
    group_by: str,
    highlight_group: Optional[Union[str, Tuple]] = None,
    grouping_specifiers: Optional[List] = None,
    show_envelope: bool = False,
    highlight_color: str = 'red',
    figure_size: Tuple[int, int] = (12, 8),
    format_as_currency: bool = False
) -> Tuple[plt.Figure, Dict]:
    """
    Plot EMA outcomes with optional grouping and highlighting.
    """
    from ema_workbench.analysis.plotting import lines
    
    # Aggregate outcomes
    _aggregate_outcomes(outcomes, outcomes_to_plot)
    
    # Determine grouping type
    column_data = experiments_df[group_by]
    is_categorical = pd.api.types.is_object_dtype(column_data) or \
                     pd.api.types.is_categorical_dtype(column_data)
    
    # Auto-generate bins if needed
    if grouping_specifiers is None:
        if is_categorical:
            grouping_specifiers = sorted(column_data.unique().astype(str))
        else:
            min_val = column_data.min()
            max_val = column_data.max()
            
            # Integer bins for repair crews, float bins for fragility
            if 'repair_crews' in group_by or column_data.dtype == np.int64:
                bin_size = max(1, (max_val - min_val) // 5)
                grouping_specifiers = _create_int_bins(int(min_val), int(max_val), int(bin_size))
            else:
                bin_size = (max_val - min_val) / 5
                grouping_specifiers = _create_float_bins(float(min_val), float(max_val), float(bin_size))
    
    # Validate bins
    valid_bins = _filter_valid_bins(
        experiments_df, outcomes, outcomes_to_plot,
        group_by, grouping_specifiers, is_categorical
    )
    
    if not valid_bins:
        print(f"No valid bins found for {group_by}")
        return None, None
    
    # Create plots
    try:
        fig, axes = lines(
            experiments_df,
            outcomes,
            outcomes_to_show=outcomes_to_plot,
            group_by=group_by,
            grouping_specifiers=valid_bins,
            titles={outcome: f"{outcome.replace('_', ' ').title()}" 
                    for outcome in outcomes_to_plot},
            legend=True,
            show_envelope=show_envelope
        )
    except Exception as e:
        print(f"Error creating plots: {e}")
        return None, None
    
    # Apply highlighting if requested
    if highlight_group is not None:
        highlight_idx = None
        for idx, bin_spec in enumerate(valid_bins):
            if is_categorical:
                if str(bin_spec) == str(highlight_group):
                    highlight_idx = idx
                    break
            else:
                if isinstance(highlight_group, tuple):
                    if highlight_group == bin_spec:
                        highlight_idx = idx
                        break
        
        if highlight_idx is not None:
            _apply_highlighting(axes, highlight_idx, highlight_color, len(outcomes_to_plot))
    
    # Style plots
    for ax in axes.values():
        if format_as_currency:
            ax.set_ylabel('Economic Impact (€)')
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        ax.grid(True, alpha=0.3)
    
    fig.set_size_inches(*figure_size)
    plt.tight_layout()
    
    return fig, axes


# ============================================================================
# Specification Parser
# ============================================================================

def parse_plot_specification(spec: str) -> Dict:
    """
    Parse plot specification string.
    
    Examples:
        'monetary_industrial_highlighted-repair-crews-1-10'
        'population'
        'population_highlighted-fragility-5.0-5.5'
        'operational_highlighted-policy-monetary_impacts_policy'
    """
    parts = spec.split('_highlighted-')
    base = parts[0]
    
    result = {
        'highlight_info': None,
        'group_by': None,
        'outcome_names': [],
        'outcome_type': None
    }
    
    # Parse highlighting
    if len(parts) == 2:
        highlight_parts = parts[1].split('-')
        
        # repair-crews-1-10
        if len(highlight_parts) >= 4 and highlight_parts[0] == 'repair' and highlight_parts[1] == 'crews':
            result['group_by'] = 'number_repair_crews'
            result['highlight_info'] = (int(highlight_parts[2]), int(highlight_parts[3]))
        
        # fragility-5.0-5.5
        elif len(highlight_parts) >= 3 and highlight_parts[0] == 'fragility':
            result['group_by'] = 'fragility_param_k'
            result['highlight_info'] = (float(highlight_parts[1]), float(highlight_parts[2]))
        
        # policy-policy_name
        elif highlight_parts[0] == 'policy':
            result['group_by'] = 'policy'
            result['highlight_info'] = '-'.join(highlight_parts[1:])
    
    # Parse outcome type
    if base.startswith('monetary_'):
        result['outcome_type'] = 'monetary'
        category = base.replace('monetary_', '')
        
        if category == 'all':
            result['outcome_names'] = MONETARY_OUTCOMES
        elif f'monetary_impact_{category}' in MONETARY_OUTCOMES:
            result['outcome_names'] = [f'monetary_impact_{category}']
        else:
            result['outcome_names'] = [MONETARY_OUTCOMES[0]]  # Default to total
    
    elif base == 'population':
        result['outcome_type'] = 'population'
        result['outcome_names'] = ['affected_population', 'served_population']
    
    elif base == 'operational':
        result['outcome_type'] = 'operational'
        result['outcome_names'] = ['operational', 'flooded', 'unreachable']
    
    else:
        result['outcome_type'] = 'other'
        result['outcome_names'] = ['flooded', 'operational', 'affected_population']
    
    # Default group_by if not specified
    if result['group_by'] is None:
        result['group_by'] = 'number_repair_crews' if result['outcome_type'] == 'monetary' else 'policy'
    
    return result


# ============================================================================
# Batch Plot Generator
# ============================================================================

def generate_ema_plots(
    experiments_df: pd.DataFrame,
    outcomes: Dict,
    plot_specs: List[str],
    output_dir: Optional[str] = None,
    show_plots: bool = True,
    figure_size: Tuple[int, int] = (10, 6)
) -> List[Tuple[plt.Figure, Dict]]:
    """
    Generate multiple EMA plots from specification strings.
    """
    results = []
    
    for spec in plot_specs:
        print(f"\n{'='*80}")
        print(f"Generating plot: {spec}")
        print(f"{'='*80}")
        
        try:
            parsed = parse_plot_specification(spec)
        except Exception as e:
            print(f"Error parsing spec: {e}")
            continue
        
        # Determine color
        if parsed['highlight_info']:
            if 'repair_crews' in parsed['group_by']:
                color = DEFAULT_COLORS['repair_crews']
            elif 'fragility' in parsed['group_by']:
                color = DEFAULT_COLORS['fragility']
            elif 'policy' in parsed['group_by']:
                color = DEFAULT_COLORS['policy']
            else:
                color = DEFAULT_COLORS['default']
        else:
            color = DEFAULT_COLORS['default']
        
        # Generate plot
        try:
            fig, axes = plot_ema_outcomes(
                experiments_df=experiments_df,
                outcomes=outcomes,
                outcomes_to_plot=parsed['outcome_names'],
                group_by=parsed['group_by'],
                highlight_group=parsed['highlight_info'],
                show_envelope=False,  # Simplified - no envelope
                highlight_color=color,
                format_as_currency=parsed['outcome_type'] == 'monetary',
                figure_size=figure_size
            )
        except Exception as e:
            print(f"Error generating plot: {e}")
            continue
        
        if fig is not None:
            results.append((fig, axes))
            
            # Save if requested
            if output_dir:
                try:
                    output_path = Path(output_dir) / f"{spec.replace('_', '-')}.png"
                    fig.savefig(output_path, dpi=150, bbox_inches='tight')
                    print(f"Saved to {output_path}")
                except Exception as e:
                    print(f"Error saving: {e}")
            
            # Show or close
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
    
    return results


# ============================================================================
# Convenience Functions
# ============================================================================

def plot_monetary_by_repair_crews(
    experiments_df: pd.DataFrame,
    outcomes: Dict,
    highlight_range: Optional[Tuple[int, int]] = None
) -> Tuple[plt.Figure, Dict]:
    """Quick plot of monetary outcomes grouped by repair crews."""
    return plot_ema_outcomes(
        experiments_df=experiments_df,
        outcomes=outcomes,
        outcomes_to_plot=MONETARY_OUTCOMES,
        group_by='number_repair_crews',
        highlight_group=highlight_range,
        format_as_currency=True,
        highlight_color=DEFAULT_COLORS['repair_crews'],
        figure_size=(10, 8)
    )


def plot_operational_by_policy(
    experiments_df: pd.DataFrame,
    outcomes: Dict,
    highlight_policy: Optional[str] = None
) -> Tuple[plt.Figure, Dict]:
    """Quick plot of operational metrics grouped by policy."""
    return plot_ema_outcomes(
        experiments_df=experiments_df,
        outcomes=outcomes,
        outcomes_to_plot=['flooded', 'operational', 'unreachable', 'affected_population'],
        group_by='policy',
        highlight_group=highlight_policy,
        highlight_color=DEFAULT_COLORS['policy'],
        figure_size=(10, 6)
    )


def plot_population_impacts(
    experiments_df: pd.DataFrame,
    outcomes: Dict,
    group_by: str = 'policy'
) -> Tuple[plt.Figure, Dict]:
    """Quick plot of population impact metrics."""
    return plot_ema_outcomes(
        experiments_df=experiments_df,
        outcomes=outcomes,
        outcomes_to_plot=['affected_population', 'served_population', 'affected_population_ratio'],
        group_by=group_by,
        figure_size=(10, 5)
    )

# ---------------------------------------------------------------------------
# Legacy functions
# ---------------------------------------------------------------------------
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
            unreachable_count = end_of_day['unreachable_count']
            print(f"Day {day}: {operational_rate:.1f}% operational, {int(flooded_count)} flooded, {int(unreachable_count)} unreachable, {int(damaged_count)} damaged.")

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


def plot_population_and_landuse(
    population_above_0,
    sizes,
    voronoi_gdf,
    gdf_assets,
    asset_land_use_map,
):
    """
    Plot population distribution and land use distribution as pie charts on Voronoi polygons.

    Parameters
    ----------
    population_above_0 : GeoDataFrame
        Population polygons with nonzero population.
    sizes : Series
        Marker sizes for population points.
    voronoi_gdf : GeoDataFrame
        Voronoi polygons with asset_id and pop_map columns.
    gdf_assets : GeoDataFrame
        Asset polygons.
    asset_land_use_map : dict
        Mapping from asset_id to land use dict.
    """
    categories = ['residential', 'commercial', 'industrial', 'transport', 'public_sector']
    colors = ['#ede342', '#d34552', '#a52cbe', '#30123b', '#4770e8']
    crs = voronoi_gdf.crs

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Population distribution
    centroids = population_above_0.geometry.centroid
    ax[0].scatter(centroids.x, centroids.y, s=sizes, alpha=0.6, c='darkblue', edgecolors='k', linewidth=0.1)
    voronoi_gdf.plot(ax=ax[0], alpha=0.5, edgecolor='k', column='pop_map', cmap='viridis', legend=True, lw=0.1)
    ax[0].set_title('Population Distribution & Voronoi Service Areas')
    ax[0].set_aspect('equal')

    # Right plot: Land use distribution as pie charts
    voronoi_gdf.plot(ax=ax[1], alpha=0.3, edgecolor='k', facecolor='lightgray', lw=0.1)

    # Get asset centroids for plotting
    asset_centroids = gdf_assets.to_crs(crs).geometry.centroid

    # Calculate max_total from ALL assets (not just those in voronoi_gdf order)
    all_totals = []
    for aid in asset_land_use_map.keys():
        total = sum([asset_land_use_map[aid].get(cat, 0) for cat in categories])
        if total > 0:
            all_totals.append(total)
    max_total = max(all_totals) if all_totals else 1.0

    # Track statistics
    assets_with_data = 0
    assets_without_data = 0
    assets_not_in_map = 0

    # Plot ALL assets (remove max_pies limit)
    for idx, row in voronoi_gdf.iterrows():
        asset_id = row['asset_id']

        if asset_id not in asset_land_use_map:
            assets_not_in_map += 1
            continue

        lu_data = asset_land_use_map[asset_id]
        values = [lu_data.get(cat, 0) for cat in categories]
        total = sum(values)

        # Get centroid position
        centroid = asset_centroids[gdf_assets.index == asset_id].iloc[0]

        if total > 0:
            assets_with_data += 1

            # Scale radius based on total
            radius = (total / max_total) * 150

            # Create wedges
            angles = [0]
            for val in values:
                angles.append(angles[-1] + (val / total) * 360 if total > 0 else angles[-1])

            for i, (cat, color) in enumerate(zip(categories, colors)):
                if values[i] > 0:
                    wedge = Wedge(
                        (centroid.x, centroid.y),
                        radius,
                        angles[i],
                        angles[i+1],
                        facecolor=color,
                        edgecolor='k',
                        linewidth=0.1,
                        alpha=0.8
                    )
                    ax[1].add_patch(wedge)
        else:
            assets_without_data += 1
            # Plot a small marker for assets with no land use data
            ax[1].plot(centroid.x, centroid.y, 'x', color='black', markersize=4, alpha=0.8)

    # Print diagnostics
    print(f"\nLand Use Visualization Statistics:")
    print(f"  Total voronoi polygons: {len(voronoi_gdf)}")
    print(f"  Assets with land use data: {assets_with_data}")
    print(f"  Assets without land use data: {assets_without_data}")
    print(f"  Assets not in mapping: {assets_not_in_map}")
    print(f"  Total in asset_land_use_map: {len(asset_land_use_map)}")

    ax[1].set_title(f'Land Use Distribution ({assets_with_data} assets with data, {assets_without_data} without)')
    ax[1].set_aspect('equal')
    ax[1].autoscale()

    legend_elements = [Patch(facecolor=colors[i], label=categories[i].replace('_', ' ').title())
                       for i in range(len(categories))]
    # Use Line2D for marker-based legend entry
    legend_elements.append(Line2D([0], [0], marker='x', color='w', markerfacecolor='black',
                                   markersize=8, label='No land use data'))
    ax[1].legend(handles=legend_elements, loc='upper right')

    # Add basemap to both subplots
    ctx.add_basemap(ax[0], crs=crs, source=ctx.providers.OpenStreetMap.Mapnik, zoom=15, alpha=0.5)
    ctx.add_basemap(ax[1], crs=crs, source=ctx.providers.OpenStreetMap.Mapnik, zoom=15, alpha=0.5)

    plt.tight_layout()
    plt.show()



def aggregate_outcomes(outcomes, outcomes_to_show):
    """
    Aggregate 3D outcomes (n_experiments, n_timesteps, n_assets) to 2D.
    Leave 1D and 2D outcomes unchanged.
    Returns a dict indicating which outcomes were aggregated.
    """
    aggregation_info = {}
    
    for key in outcomes_to_show:
        if key not in outcomes:
            print(f"Warning: Outcome '{key}' not found in outcomes dictionary")
            aggregation_info[key] = 'missing'
            continue
        
        arr = outcomes[key]
        original_shape = arr.shape
        
        if arr.ndim == 3:  # (n_experiments, n_timesteps, n_assets)
            print(f"Aggregating 3D outcome '{key}' from {arr.shape} -> mean across assets")
            outcomes[key] = arr.mean(axis=2)  # -> (n_experiments, n_timesteps)
            aggregation_info[key] = f'aggregated from {original_shape}'
        elif arr.ndim == 2:  # (n_experiments, n_timesteps)
            aggregation_info[key] = 'already 2D'
        elif arr.ndim == 1:  # (n_timesteps,) - Used only to handle no impacts case
            print(f"Outcome '{key}' is 1D {arr.shape}!!!")
            # Broadcast to 2D by repeating for all experiments
            n_experiments = len(outcomes[list(outcomes.keys())[0]])
            outcomes[key] = np.tile(arr, (n_experiments, 1))
            aggregation_info[key] = f'broadcast from {original_shape}'
        else:
            print(f"Warning: Outcome '{key}' has unexpected shape {arr.shape}")
            aggregation_info[key] = f'unexpected shape {original_shape}'
    
    return aggregation_info


def sample_experiments(var_val, experiments_df):    
    """Sample experiments based on variable values."""
    mask = pd.Series([True] * len(experiments_df))
    for var, val in var_val.items():
        if isinstance(val, tuple) and len(val) == 2:
            # Range filter
            mask &= (experiments_df[var] >= val[0]) & (experiments_df[var] <= val[1])
        elif isinstance(val, list):
            # List of values filter
            mask &= experiments_df[var].isin(val)
        else:
            # Exact match filter
            mask &= (experiments_df[var] == val)
    return experiments_df[mask]

# Bin creation helpers
def create_float_bins(min_val, max_val, bin_size):
    bins = []
    val = min_val
    while val < max_val:
        bins.append((val, val + bin_size))
        val += bin_size
    return bins

def create_bins(min_val, max_val, bin_size):
    bins = []
    start = min_val
    while start <= max_val:
        end = min(start + bin_size - 1, max_val)
        bins.append((start, end))
        start = end + 1
    return bins

def filter_valid_bins_for_outcomes(experiments_df, outcomes, outcomes_to_show, group_by, bin_ranges):
    """Filter bins to ensure they contain valid data for all specified outcomes."""
    valid_bins = []
    for bin_range in bin_ranges:
        mask = (experiments_df[group_by] >= bin_range[0]) & (experiments_df[group_by] <= bin_range[1])
        experiment_indices = np.where(mask)[0]
        count = len(experiment_indices)
        
        if count == 0:
            print(f"Bin {bin_range}: No experiments")
            continue
        
        all_valid = True
        for outcome_name in outcomes_to_show:
            if outcome_name not in outcomes:
                print(f"Warning: Outcome '{outcome_name}' not found")
                all_valid = False
                break
            
            outcome_data = outcomes[outcome_name][experiment_indices]
            
            if outcome_data.size == 0 or np.all(np.isnan(outcome_data)):
                print(f"Bin {bin_range}: Empty/NaN data for '{outcome_name}'")
                all_valid = False
                break
        
        if all_valid:
            valid_bins.append(bin_range)
            print(f"Bin {bin_range}: {count} experiments - VALID")
        else:
            print(f"Bin {bin_range}: {count} experiments - INVALID")
    
    return valid_bins

# Better label formatting function
def format_outcome_label(outcome_name):
    """Convert outcome names to readable labels."""
    # Handle monetary outcomes
    if outcome_name.startswith('monetary_impact_'):
        category = outcome_name.replace('monetary_impact_', '')
        if category == 'total':
            return 'Total Economic Impact'
        else:
            return f'{category.replace("_", " ").title()} Economic Impact'
    
    # Handle population outcomes
    elif 'population' in outcome_name:
        return outcome_name.replace('_', ' ').title()
    
    # Handle asset-level outcomes
    elif outcome_name in ['flooded', 'operational', 'unreachable', 'accessible', 'crew_assigned']:
        return f'{outcome_name.title()} Assets (Mean)'
    
    elif outcome_name in ['damage_ratio', 'repair_time', 'hazard_value']:
        return f'{outcome_name.replace("_", " ").title()} (Mean)'
    
    else:
        return outcome_name.replace('_', ' ').title()
    

def plot_grouped_outcomes(
    group_by,
    experiments_df,
    outcomes,
    outcomes_to_show,
    var_val=None,
    bin_ranges=None,
    monetary_outcomes=None,
    population_outcomes=None,
    asset_outcomes_3d=None,
    asset_outcomes_2d=None,
):
    """
    Plot grouped outcomes for a given grouping variable.

    Parameters
    ----------
    group_by : str
        The variable to group by (e.g., "policy", "number_repair_crews", etc.)
    experiments_df : pd.DataFrame
        DataFrame of experiments.
    outcomes : dict
        Dictionary of outcome arrays.
    outcomes_to_show : list
        List of outcome names to plot.
    var_val : dict, optional
        Dictionary of filters to apply to experiments_df.
    bin_ranges : list or None, optional
        Bin ranges for grouping (if not grouping by policy).
    monetary_outcomes, population_outcomes, asset_outcomes_3d, asset_outcomes_2d : list, optional
        Lists of outcome names for special formatting.
    """
    # Filter experiments DataFrame for the desired value
    experiments_sample = sample_experiments(var_val, experiments_df) if var_val else experiments_df
    sample_indices = experiments_sample.index.values

    # Filter outcomes for these indices
    outcomes_for_sample = {k: v[sample_indices] for k, v in outcomes.items()}

    # Validate bins
    if group_by == "policy":
        valid_bins = None
        bins = None
    else:
        valid_bins = filter_valid_bins_for_outcomes(
            experiments_sample, 
            outcomes_for_sample, 
            outcomes_to_show,
            group_by, 
            bin_ranges
        )
        bins = bin_ranges

    # Plot outcomes
    if valid_bins:
        try:
            fig, axes = lines(
                experiments_sample,
                outcomes_for_sample,
                outcomes_to_show=outcomes_to_show,
                group_by=group_by,
                titles={outcome: format_outcome_label(outcome) for outcome in outcomes_to_show},
                grouping_specifiers=valid_bins,
                legend=True,
                show_envelope=False,
            )
        except ValueError as e:
            print(f"Error: {e}\nRetrying without envelope...")
            fig, axes = lines(
                experiments_sample,
                outcomes_for_sample,
                outcomes_to_show=outcomes_to_show,
                group_by=group_by,
                titles={outcome: format_outcome_label(outcome) for outcome in outcomes_to_show},
                grouping_specifiers=valid_bins,
                legend=True,
                show_envelope=False,
            )
    else:
        fig, axes = lines(
                experiments_sample,
                outcomes_for_sample,
                outcomes_to_show=outcomes_to_show,
                group_by=group_by,
                titles={outcome: format_outcome_label(outcome) for outcome in outcomes_to_show},
                legend=True,
                show_envelope=True,
            )
    # Style plots differently for different outcome types
    for outcome_name, ax in axes.items():
        for line in ax.get_lines():
            line.set_linewidth(0.1)
            line.set_alpha(0.8)
        
        # Special formatting for monetary outcomes (cumulative, so should be monotonically increasing)
        if monetary_outcomes and outcome_name in monetary_outcomes:
            ax.set_ylabel('Cumulative Economic Impact (€)')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))
            
        elif population_outcomes and outcome_name in population_outcomes:
            ax.set_ylabel('Population')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
        elif asset_outcomes_3d and outcome_name in asset_outcomes_3d:
            # These are mean counts of binary states (flooded/operational)
            ax.set_ylabel('Mean Asset Count')
            
        elif asset_outcomes_2d and outcome_name in asset_outcomes_2d:
            if 'operational' in outcome_name:
                ax.set_ylabel('Operational Assets')
            elif 'flooded' in outcome_name:
                ax.set_ylabel('Flooded Assets')
            # These are already aggregated means (damage_ratio, repair_time)
            elif 'damage' in outcome_name:
                ax.set_ylabel('Mean Damage Ratio')
            elif 'time' in outcome_name:
                ax.set_ylabel('Mean Repair Time (hours)')
            else:
                ax.set_ylabel('Mean Value')        
        else:
            ax.set_ylabel('Value')
        
        # Common styling
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('Timestep (hours)')

    fig.set_size_inches(14, len(axes)*4)
    plt.show()

    # Print summary of what was actually plotted
    print("\n")
    print("Plot summary:")
    print(f"\nGrouped by: {group_by}")
    print(f"Number of bins: {len(valid_bins) if valid_bins is not None else 'N/A'}")
    print(f"\nOutcomes plotted:")
    for outcome in outcomes_to_show:
        dim = 'missing'
        if outcome in outcomes:
            shape = outcomes[outcome].shape
            if outcomes[outcome].ndim == 2:
                dim = f'2D ({shape[0]} experiments x {shape[1]} timesteps)'
            elif outcomes[outcome].ndim == 3:
                dim = f'3D ({shape[0]} x {shape[1]} x {shape[2]}) - Should be aggregated'
            else:
                dim = f'{outcomes[outcome].ndim}D {shape}'
        
        category = ''
        if monetary_outcomes and outcome in monetary_outcomes:
            category = 'Monetary (cumulative)'
        elif population_outcomes and outcome in population_outcomes:
            category = 'Population'
        elif asset_outcomes_3d and outcome in asset_outcomes_3d:
            category = 'Asset (3D→2D aggregated)'
        elif asset_outcomes_2d and outcome in asset_outcomes_2d:
            category = 'Asset (2D pre-aggregated)'
        
        print(f"  • {outcome:40s} {category:30s} {dim}")

    print("\n")
    if monetary_outcomes:
        print("Monetary impact")
        for mon_outcome in monetary_outcomes:
            if mon_outcome in outcomes:
                final_values = outcomes[mon_outcome][:, -1]  # Last timestep
                print(f"\n{mon_outcome}:")
                print(f"  Shape: {outcomes[mon_outcome].shape}")
                print(f"  Min final value: €{final_values.min():,.2f}")
                print(f"  Max final value: €{final_values.max():,.2f}")
                print(f"  Mean final value: €{final_values.mean():,.2f}")
                print(f"  Non-zero scenarios: {(final_values > 0).sum()} / {len(final_values)}")
            else:
                print(f"\n{mon_outcome}: NOT FOUND IN OUTCOMES!")

def compare_adaptation_policies(experiments_df, outcomes):
    """
    Simple comparison of adaptation policies.
    """
    
    print("Impact results - all experiments (n={}):".format(len(experiments_df)))
    print("Number of policies: {}".format(experiments_df['policy'].nunique()))
    
    monetary_outcomes = [
        'monetary_impact_total',
        'monetary_impact_residential',
        'monetary_impact_commercial',
        'monetary_impact_industrial',
        'monetary_impact_transport',
        'monetary_impact_public_sector',
    ]
    
    # Print summary for each monetary outcome 
    for outcome in monetary_outcomes:
        if outcome in outcomes:      
            # Calculate hourly impacts
            hourly_impacts = np.diff(outcomes[outcome], axis=1, prepend=0)
            peak_hourly = np.max(hourly_impacts)
            
            # Final cumulative impact per scenario
            final_cumulative_per_scenario = outcomes[outcome][:, -1]
            total_impact = np.mean(final_cumulative_per_scenario)
            median_impact = np.median(final_cumulative_per_scenario)
            
            print(f"\n{outcome.replace('_', ' ').title()}:")
            print(f"  - Peak hourly impact: €{peak_hourly:,.2f}")
            print(f"  - Total impact (mean): €{total_impact:,.2f}")
            print(f"  - Total impact (median): €{median_impact:,.2f}")
    
    # Compare by policy
    policy_names = experiments_df['policy'].unique()
    
    if len(policy_names) > 1:
        print("\n\nAdaptation Policy Comparison")

        for policy in policy_names:
            policy_idx = experiments_df['policy'] == policy
            # Get final cumulative values for this policy
            final_values = outcomes['monetary_impact_total'][policy_idx, -1]
            policy_total = np.mean(final_values)
            policy_std = np.std(final_values)
            print(f"  {policy}: €{policy_total:,.2f} (±€{policy_std:,.2f})")
        print("")
        
        for policy in policy_names:

            print("Detailed results for policy: {}".format(policy))
            # Get integer indices instead of boolean mask
            policy_idx = np.where(experiments_df['policy'] == policy)[0]
            n_scenarios = len(policy_idx)
            
            print(f"-- {policy} (n={n_scenarios}):")
            
            # Monetary impacts
            final_values = outcomes['monetary_impact_total'][policy_idx, -1]
            policy_total = np.mean(final_values)
            policy_std = np.std(final_values)
            print(f"  Total monetary impact: €{policy_total:,.2f} (±€{policy_std:,.2f})")
            
            # Operational metrics
            peak_flooded = outcomes['flooded'][policy_idx].max(axis=1).mean()
            min_operational = outcomes['operational'][policy_idx].min(axis=1).mean()
            try:
                peak_unreachable = outcomes['unreachable'][policy_idx].max(axis=1).mean()
            except KeyError:
                peak_unreachable = None
            
            print(f"  Peak flooded assets: {peak_flooded:.1f}")
            print(f"  Min operational assets: {min_operational:.1f}")
            if peak_unreachable is not None:
                print(f"  Peak unreachable assets: {peak_unreachable:.1f}")
            
            # Population impact
            if 'affected_population' in outcomes:
                peak_affected_pop = outcomes['affected_population'][policy_idx].max(axis=1).mean()
                print(f"  Peak affected population: {peak_affected_pop:,.0f}")
            print("")
