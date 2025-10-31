"""
EMA Workbench visualization utilities for simulation outcomes.
Simplified and focused on core functionality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path


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