"""
Functions for calculating damage ratios and repair times based on hazard values.
The operations should be vectorized for performance, especially for large datasets.
Includes default implementations for damage ratio and repair time.

Refs:
Movahednia, Mohadese, et al. ‘Power Grid Resilience Enhancement via Protecting Electrical Substations Against Flood Hazards: A Stochastic Framework’. IEEE Transactions on Industrial Informatics, vol. 18, no. 3, Mar. 2022, pp. 2132–43. Crossref, https://doi.org/10.1109/tii.2021.3100079.
Sánchez-Muñoz, Daniel, et al. ‘Electrical Grid Risk Assessment Against Flooding in Barcelona and Bristol Cities’. Sustainability, vol. 12, no. 4, Feb. 2020, p. 1527. Crossref, https://doi.org/10.3390/su12041527.

"""
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

def hospital_fragility_function(hazard_values, asset_type, k=None, major_timestep=24):
    """Placeholder fragility function for structural flood damage to hospitals.

    **This is a placeholder** to be replaced once evidence-based depth-damage
    data for hospitals is available.  Currently uses a logistic fragility curve
    identical in shape to the substation curve but with a median failure depth of
    ``d_m = 0.5 m``.  This is an arbitrary interim value; the real curve should
    be calibrated from empirical data or expert elicitation.

    The function signature is intentionally identical to
    :func:`default_fragility_function` so it can be swapped in without changing
    call sites.

    Args:
        hazard_values (np.ndarray): Flood depth values for each hospital asset.
        asset_type (np.ndarray): Asset-type labels (used for future type-specific
            branching within this function).
        k (float, optional): Steepness parameter of the logistic curve.  If
            ``None``, a value is sampled uniformly from ``[5.0, 7.5]`` each call.
        major_timestep (int): Simulation hours per hazard-map update; used to
            scale the daily failure probability to the actual timestep.

    Returns:
        np.ndarray: Integer array of binary operational status (1 = operational,
        0 = structurally damaged by flooding).
    """
    failure_probability = np.zeros_like(hazard_values, dtype=np.float64)

    if k is None:
        k = np.random.uniform(5, 7.5)

    hazard_mask = hazard_values > 0
    # Placeholder median failure depth for hospitals (0.5 m – to be calibrated).
    d_m = np.full_like(hazard_values, 0.5)

    timesteps_per_day = 24 / major_timestep if major_timestep is not None else 1
    if timesteps_per_day == 1:
        failure_probability[hazard_mask] = 1 / (
            1 + np.exp(-k * (hazard_values[hazard_mask] - d_m[hazard_mask]))
        )
    else:
        failure_probability[hazard_mask] = 1 - (
            (np.exp(-k * (hazard_values[hazard_mask] - d_m[hazard_mask])))
            / (1 + np.exp(-k * (hazard_values[hazard_mask] - d_m[hazard_mask])))
        ) ** (1 / timesteps_per_day)

    random_values = np.random.random(size=hazard_values.shape)
    return (random_values >= failure_probability).astype(int)


def default_fragility_function(hazard_values, asset_type, k=None, major_timestep=24):
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

    For asset types not explicitly modelled (e.g. ``'hospital'``), the function
    dispatches to the appropriate specialised fragility function so that all asset
    types receive correct treatment without requiring changes at the call site.
    """

    # Dispatch hospital assets to the dedicated placeholder function.
    hospital_mask = asset_type == 'hospital'
    if np.any(hospital_mask):
        operational_status = np.ones_like(hazard_values, dtype=int)

        # Non-hospital assets via the standard substation curve.
        non_hospital_mask = ~hospital_mask
        if np.any(non_hospital_mask):
            operational_status[non_hospital_mask] = default_fragility_function(
                hazard_values[non_hospital_mask],
                asset_type[non_hospital_mask],
                k=k,
                major_timestep=major_timestep,
            )

        # Hospital assets via the placeholder hospital curve.
        operational_status[hospital_mask] = hospital_fragility_function(
            hazard_values[hospital_mask],
            asset_type[hospital_mask],
            k=k,
            major_timestep=major_timestep,
        )
        return operational_status

    failure_probability = np.zeros_like(hazard_values, dtype=np.float64)

    if k is None:
        k = np.random.uniform(5, 7.5)

    hazard_mask = hazard_values > 0
    ls_mask = asset_type == 'ls'
    msls_mask = asset_type == 'msls'

    d_m = np.where(ls_mask, 0.3, np.where(msls_mask, 0.6, 0))  # Default median depth for other types

    timesteps_per_day = 24 / major_timestep if major_timestep is not None else 1
    # Calculate failure probability only for positive hazard values
    if timesteps_per_day == 1:
        failure_probability[hazard_mask] = 1 / (1 + np.exp(-k * (hazard_values[hazard_mask] - d_m[hazard_mask])))
    
    else:
        failure_probability[hazard_mask] = 1 - ( (np.exp(-k * (hazard_values[hazard_mask] - d_m[hazard_mask]))) / 
                                                  (1 + np.exp(-k * (hazard_values[hazard_mask] - d_m[hazard_mask]))) 
                                                  )**(1/timesteps_per_day)
    # Generate random values for each asset
    random_values = np.random.random(size=hazard_values.shape)
    
    # Binary decision: 0 = failed, 1 = operational
    # Asset fails if random value < failure probability
    operational_status = (random_values >= failure_probability).astype(int)
    
    return operational_status