# physics_model.py
"""
Physics Model Module - Forrestal & Jones Models
"""

import numpy as np
import math

# Constants
RHO_TARGET = 2300.0  # kg/m^3
Q_STEEL = 850000.0  # J/kg


def calculate_geometry_N(psi):
    """Calculate nose shape factor N."""
    if psi <= 0: return 0.1
    val = (8 * psi - 1) / (24 * psi * psi)
    return max(val, 0.1)


def calculate_geometry_M(psi):
    """Calculate geometric constant M (Jones 2002)."""
    if psi < 0.5: return 0.0

    sin_phi_0 = (2 * psi - 1) / (2 * psi)
    sin_phi_0 = max(-1.0, min(1.0, sin_phi_0))
    phi_0 = math.asin(sin_phi_0)

    val = (1.0 / math.cos(phi_0)) + math.tan(phi_0)
    if val <= 1e-9: return 0.0

    return 2 * psi * math.log(val)


def calculate_forrestal_depth(m, d_mm, v, fc_mpa, psi, rho_t=RHO_TARGET):
    """Calculate rigid body penetration depth (Forrestal)."""
    d = d_mm / 1000.0
    fc = fc_mpa * 1e6

    if fc_mpa <= 0: return 0.0
    S = 82.6 * (fc_mpa ** -0.544)
    N = calculate_geometry_N(psi)

    area = np.pi * (d ** 2) / 4.0
    denominator = area * N * rho_t

    if denominator <= 1e-9: return 0.0

    term_inner = (N * rho_t * (v ** 2)) / (S * fc)
    return (m / denominator) * math.log(1 + term_inner)


def calculate_jones_mass_loss(d_mm, depth_m, fc_mpa, psi, Q=Q_STEEL):
    """Calculate mass loss (Jones 2002)."""
    d = d_mm / 1000.0
    a = d / 2.0
    tau_0 = 0.15 * (fc_mpa * 1e6)
    M = calculate_geometry_M(psi)

    return (np.pi * (a ** 2) * tau_0 * M * depth_m) / Q


def hybrid_physics_model(m, d, l, v, fc, psi, crh=None, rho_t=RHO_TARGET):
    """
    Combined physics interface.
    Uses Forrestal for v < 800 and Jones (mass loss corrected) for 800 <= v < 1500.
    """
    if crh is not None:
        psi = crh

    if v >= 1500:
        return None

    # Baseline rigid depth
    depth_rigid = calculate_forrestal_depth(m, d, v, fc, psi, rho_t)

    if v < 800.0:
        return max(depth_rigid, 0.0)
    else:
        # Jones Region: Correct depth for mass loss
        mass_loss = calculate_jones_mass_loss(d, depth_rigid, fc, psi)

        # Calculate effective mass driving penetration
        mass_remaining = max(m - mass_loss, m * 0.1)
        m_effective = (m + mass_remaining) / 2.0

        # Scale depth by effective mass ratio (P ~ m)
        depth_corrected = depth_rigid * (m_effective / m)
        return max(depth_corrected, 0.0)
