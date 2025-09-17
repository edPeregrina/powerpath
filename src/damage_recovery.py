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

def default_fragility_function(hazard_values, asset_type):
    """
    Calculate binary operational status from hazard values using fragility curve.
    Returns 1 for operational, 0 for failed, based on probabilistic sampling.
    
    Following NKWK, a median failure depth (d_m) by voltage is considered - 0.3m for ls, 0.6m for msls
    The equation used follows: P_f(d) = 1/(1 + exp(-k*(d - d_m)))
    
    k is determined each run as a value between 5 and 7.5 for a hardened and softened curve (Boreel)
    """
    failure_probability = np.zeros_like(hazard_values, dtype=np.float64)

    k = np.random.uniform(5, 7.5)

    hazard_mask = hazard_values > 0
    ls_mask = asset_type == 'ls'
    msls_mask = asset_type == 'msls'

    d_m = np.where(ls_mask, 0.3, np.where(msls_mask, 0.6, 0))  # Default median depth for other types
    
    # Calculate failure probability only for positive hazard values
    failure_probability[hazard_mask] = 1 / (1 + np.exp(-k * (hazard_values[hazard_mask] - d_m[hazard_mask])))
    
    # Generate random values for each asset
    random_values = np.random.random(size=hazard_values.shape)
    
    # Binary decision: 0 = failed, 1 = operational
    # Asset fails if random value < failure probability
    operational_status = (random_values >= failure_probability).astype(int)
    
    return operational_status