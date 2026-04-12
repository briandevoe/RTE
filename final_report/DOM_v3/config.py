"""
config.py — All user-facing parameters and method flags for DOM_v2.

Edit this file to change inputs, quadrature method, or sweep scheme.
All other modules import from here — no other file needs to be touched
for a standard parameter sweep.
"""

# =============================================================================
# SPECTRAL PARAMETERS
# =============================================================================
# RED band
# rho_L = 0.06     # leaf reflectance
# tau_L = 0.04     # leaf transmittance
# rho_g = 0.10     # ground reflectance

# NIR band
rho_L = 0.45     # leaf reflectance
tau_L = 0.45     # leaf transmittance
rho_g = 0.15     # ground reflectance

# =============================================================================
# ILLUMINATION GEOMETRY
# =============================================================================
theta_solar = 140.0   # solar polar angle from UPWARD zenith (deg)
                      # must be in (90, 180]: 180 = overhead, 140 = 40-deg elevation
phi_solar   =   0.0   # solar azimuth (deg)
f_dir       =   0.70  # fraction of F_in that is direct solar beam
F_in        =   1.0   # total downwelling irradiance at canopy top [W m-2]

# =============================================================================
# CANOPY PARAMETERS
# =============================================================================
LAI = 1.5   # leaf area index

# =============================================================================
# NUMERICAL RESOLUTION
# =============================================================================
N       = 16    # polar quadrature directions on full sphere (must be even)
M       = 16    # azimuthal directions
K       = 50    # canopy layers

# =============================================================================
# SOLVER CONVERGENCE
# =============================================================================
TOL      = 0.01   # relative convergence tolerance on scalar irradiance
MAX_ITER = 500    # safety cap on source iterations

# =============================================================================
# METHOD FLAGS  — swap these without touching any other file
# =============================================================================

# Quadrature method for polar directions:
#   "gauss_legendre"  — nodes = zeros of P_N on (-1,1)  [default, NumPy built-in]
#   "double_gauss"    — separate G-L on each hemisphere; better flux accuracy
#   "gauss_lobatto"   — includes endpoints ±1
#   "uniform"         — evenly spaced, equal weights (for debugging only)
QUADRATURE = "gauss_legendre"

# Sweep scheme for Diamond-Difference parameter alpha:
#   "diamond_diff"  — alpha=0.5; 2nd-order, may produce negative intensities
#   "step"          — alpha=1.0; 1st-order upwind, always positive
SWEEP = "diamond_diff"

# Diamond-difference alpha (overridden by SWEEP if set to "step")
ALPHA_DD = 0.5
