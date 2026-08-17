"""
Functions for calculating damage ratios and repair times based on hazard values.
The operations should be vectorized for performance, especially for large datasets.
Includes default implementations for damage ratio and repair time.

Refs:
Movahednia, Mohadese, et al. ‘Power Grid Resilience Enhancement via Protecting Electrical Substations Against Flood Hazards: A Stochastic Framework’. IEEE Transactions on Industrial Informatics, vol. 18, no. 3, Mar. 2022, pp. 2132–43. Crossref, https://doi.org/10.1109/tii.2021.3100079.
Sánchez-Muñoz, Daniel, et al. ‘Electrical Grid Risk Assessment Against Flooding in Barcelona and Bristol Cities’. Sustainability, vol. 12, no. 4, Feb. 2020, p. 1527. Crossref, https://doi.org/10.3390/su12041527.

"""
from collections.abc import Mapping

import numpy as np


# Helper functions for damage and repair calculations
def default_damage_ratio_function(hazard_values, coefficients):
    """Calculate damage ratio from hazard values using linear function"""
    m, n = coefficients
    return m * hazard_values + n

def default_repair_time_function(damage_ratios, coefficients):
    """Calculate repair time from damage ratios using polynomial function"""
    a, b, c = coefficients
    return a * (damage_ratios ** 2) + b * damage_ratios + c

def vectorized_damage_ratio_solver(repair_times, coefficients):
    """
    Vectorized solver for quadratic function: repair_time = a*DR² + b*DR + c
    Solve for DR using quadratic formula

    Args:
        repair_times (np.ndarray): Array of repair times (for each asset).
        coefficients (tuple): Coefficients (a, b, c) of the quadratic equation.

    Returns:
        np.ndarray: Array of damage ratios (DR) corresponding to each repair time.
    """
    a, b, c = coefficients
    
    # Quadratic equation: a*DR² + b*DR + (c - repair_time) = 0
    # Using quadratic formula: DR = (-b ± √(b² - 4a(c-repair_time))) / 2a
    
    discriminant = b**2 - 4*a*(c - repair_times)
    
    # Handle negative discriminants (no real solution)
    valid_mask = discriminant >= 0
    damage_ratios = np.zeros_like(repair_times, dtype=np.float64)
    
    if np.any(valid_mask):
        sqrt_disc = np.sqrt(discriminant[valid_mask])
        # Take positive root (damage ratio should be positive)
        damage_ratios[valid_mask] = (-b + sqrt_disc) / (2*a)
    
    # Clamp to valid range [0, 1]
    return np.clip(damage_ratios, 0.0, 1.0)

def _per_timestep_failure_probability(failure_probability, major_timestep):
    """Convert daily failure probability to the active timestep probability."""
    failure_probability = np.clip(np.asarray(failure_probability, dtype=np.float64), 0.0, 1.0)
    timesteps_per_day = 24 / major_timestep if major_timestep not in (None, 0) else 1
    if timesteps_per_day == 1:
        return failure_probability
    return 1.0 - np.power(1.0 - failure_probability, 1.0 / timesteps_per_day)


def _resolve_fragility_exclusions(hazard_values, asset_type, fragility_exclusions=None):
    """Return a boolean mask where fragility should be skipped."""
    exclusion_mask = np.zeros_like(hazard_values, dtype=bool)
    if fragility_exclusions is None:
        return exclusion_mask

    if isinstance(fragility_exclusions, np.ndarray):
        return np.asarray(fragility_exclusions, dtype=bool)

    if isinstance(fragility_exclusions, Mapping):
        for key, value in fragility_exclusions.items():
            key_mask = asset_type == str(key)
            if isinstance(value, bool):
                if value:
                    exclusion_mask |= key_mask
            else:
                exclusion_mask |= key_mask & (hazard_values <= float(value))
        return exclusion_mask

    raise TypeError("fragility_exclusions must be a boolean array or a mapping of asset-type rules.")


def _build_failure_probability(hazard_values, model, *, k=None, major_timestep=24):
    """Build failure probabilities from an explicit fragility model."""
    mode = model.get("mode", model.get("regime", "depth_logistic"))
    activation_threshold = float(model.get("activation_threshold", 0.0))
    failure_probability = np.zeros_like(hazard_values, dtype=np.float64)
    active_mask = hazard_values > activation_threshold

    if not np.any(active_mask):
        return failure_probability

    if mode == "probability_curve":
        intensity_values = np.asarray(model["intensity_values"], dtype=np.float64)
        probability_values = np.asarray(model["failure_probabilities"], dtype=np.float64)
        interpolated = np.interp(
            hazard_values[active_mask],
            intensity_values,
            probability_values,
            left=probability_values[0],
            right=probability_values[-1],
        )
        failure_probability[active_mask] = interpolated
        return _per_timestep_failure_probability(failure_probability, major_timestep)

    steepness = model.get("steepness", k)
    if steepness is None:
        low, high = model.get("steepness_range", (5.0, 7.5))
        steepness = np.random.uniform(low, high)
    steepness = float(steepness)

    median_failure_depth = float(model.get("median_failure_depth", 0.0))
    failure_probability[active_mask] = 1.0 / (
        1.0 + np.exp(-steepness * (hazard_values[active_mask] - median_failure_depth))
    )
    return _per_timestep_failure_probability(failure_probability, major_timestep)


def _sample_fragility_operational_status(
    hazard_values,
    *,
    model,
    k=None,
    major_timestep=24,
):
    """Sample operational status from an explicit fragility model."""
    failure_probability = _build_failure_probability(
        hazard_values,
        model,
        k=k,
        major_timestep=major_timestep,
    )
    random_values = np.random.random(size=hazard_values.shape)
    return (random_values >= failure_probability).astype(int)


def hospital_fragility_function(
    hazard_values,
    asset_type,
    k=None,
    major_timestep=24,
    model=None,
):
    """Evaluate hospital fragility using an explicit model definition."""
    hospital_model = model or {
        "mode": "depth_logistic",
        "median_failure_depth": 0.5,
        "steepness_range": (5.0, 7.5),
        "activation_threshold": 0.0,
    }
    return _sample_fragility_operational_status(
        np.asarray(hazard_values, dtype=np.float64),
        model=hospital_model,
        k=k,
        major_timestep=major_timestep,
    )


def default_fragility_function(
    hazard_values,
    asset_type,
    k=None,
    major_timestep=24,
    fragility_models: Mapping[str, Mapping[str, object]] | None = None,
    fragility_exclusions=None,
):
    """
    Calculate binary operational status from hazard values using fragility curve.
    Failure probability is determined (by default daily, major_timestep=24 hours) and sampled for each asset.
    Returns 1 for operational, 0 for failed, based on probabilistic sampling.
    
    Following NKWK, a median failure depth (d_m) by voltage is considered - 0.3m for ls, 0.6m for msls
    The equation used follows: 
        P_f(d) = 1/(1 + exp(-k*(d - d_m)))
    
    if k is not given, it is determined each run as a value between 5.0 and 7.5 for a hardened and softened curve (Boreel)

    To adjust for non-daily timesteps, the failure probability is calculated as:

        P_f_timestep = 1 - ((exp(-k*(d - d_m))) / (1 + exp(-k*(d - d_m))))^(major_timestep/24)

    Explicit fragility overrides can be passed through ``fragility_models``. Each
    asset-type entry may use either:

    - ``mode="depth_logistic"`` with ``median_failure_depth`` and optional
      ``steepness`` / ``steepness_range``
    - ``mode="probability_curve"`` with paired ``intensity_values`` and
      ``failure_probabilities`` arrays

    ``fragility_exclusions`` can be a boolean mask or a mapping of asset types to
    boolean/threshold exclusion rules.
    """
    hazard_values = np.asarray(hazard_values, dtype=np.float64)
    asset_type = np.asarray(asset_type)
    fragility_models = dict(fragility_models or {})
    exclusion_mask = _resolve_fragility_exclusions(
        hazard_values,
        asset_type,
        fragility_exclusions=fragility_exclusions,
    )

    operational_status = np.ones_like(hazard_values, dtype=int)
    for asset_name in np.unique(asset_type):
        asset_mask = (asset_type == asset_name) & ~exclusion_mask
        if not np.any(asset_mask):
            continue

        model = fragility_models.get(str(asset_name))
        if model is None:
            if str(asset_name) == "hospital":
                model = {
                    "mode": "depth_logistic",
                    "median_failure_depth": 0.5,
                    "steepness_range": (5.0, 7.5),
                    "activation_threshold": 0.0,
                }
            else:
                model = {
                    "mode": "depth_logistic",
                    "median_failure_depth": 0.3 if str(asset_name) == "ls" else 0.6 if str(asset_name) == "msls" else 0.0,
                    "steepness_range": (5.0, 7.5),
                    "activation_threshold": 0.0,
                }

        operational_status[asset_mask] = _sample_fragility_operational_status(
            hazard_values[asset_mask],
            model=model,
            k=k,
            major_timestep=major_timestep,
        )

    return operational_status