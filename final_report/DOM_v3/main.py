# Discrete Ordinates Method (DOM) for Radiative Transfer


##################################
# imports and configs
####################################
import math
import sys
import os
import numpy as np

# set directory to current file so DOM_v3 modules find each other
sys.path.insert(0, os.path.dirname(__file__))

from step3_uncollided_rewrite import solve_uncollided
from step2_phase_rewrite      import precompute_G_qq, precompute_G_sol
from step4_collided_rewrite   import solve_collided
from step5_energy_rewrite     import energy_balance
from step6_brf_rewrite        import brf_at_view

import config as cfg
# configs
N=16
M=16

# Derived quantities 
omega_L   = cfg.rho_L + cfg.tau_L
G         = 0.5          # uniform leaf angle distribution
dL        = cfg.LAI / cfg.K
solar_mu  = math.cos(math.radians(cfg.theta_solar))
solar_phi = math.radians(cfg.phi_solar)
alpha     = 0.5 if cfg.SWEEP == "diamond_diff" else 1.0

#######################################
# Step 1: Discretize
#######################################

# Legendre-Gauss quadrature for polar angles (mu) and weights (w_mu)
# 1-based: index 0 unused, valid at indices 1..N
_nodes, _weights = np.polynomial.legendre.leggauss(N)
mu   = np.zeros(N + 1);  mu[1:]   = _nodes
w_mu = np.zeros(N + 1);  w_mu[1:] = _weights

# Uniform quadrature for azimuthal angles (phi) and weights (w_phi)
phi = np.zeros(M + 1)
phi[1:] = (np.arange(M) / M) * 2.0 * np.pi
w_phi = 2.0 * np.pi / M


#######################################
# Step 2: Phase function tables
#######################################

# Step 2: Phase tables 
G_qq  = precompute_G_qq(mu, phi, cfg.N, cfg.M, omega_L, cfg.tau_L)
G_sol = precompute_G_sol(solar_mu, solar_phi, mu, phi, cfg.N, cfg.M, omega_L, cfg.tau_L)


#######################################
# Step 3: Uncollided
#######################################

I0_dir, I0, Fd_dir, Fd_dif, Fu_unc, Q = solve_uncollided(
    mu, phi, w_mu, w_phi, cfg.N, cfg.M, cfg.K,
    solar_mu, cfg.f_dir, cfg.F_in, cfg.rho_g, G, dL,G_qq, G_sol)


#######################################
# Step 4: Collided field (iterative) 
#######################################

IC, Fd_col, Fu_col = solve_collided(
    mu, phi, w_mu, w_phi, cfg.N, cfg.M, cfg.K,
    G, dL, cfg.rho_g, Q, G_qq, omega_L,
    tol=cfg.TOL, max_iter=cfg.MAX_ITER, alpha=alpha)

#######################################
# Step 5: Energy balance
#######################################    
result = energy_balance(
    I0_dir, I0, IC,
    Fd_dir, Fd_dif, Fu_unc, Fd_col, Fu_col,
    mu, w_mu, w_phi, cfg.N, cfg.M, cfg.K,
    G, dL, omega_L, cfg.rho_g, cfg.F_in)


#######################################
# Step 6: BRF at sample view directions 
#######################################
for vz in [10, 30, 45, 60]:
    brf = brf_at_view(
        vz, 180.0,
        I0, IC, I0_dir, mu, phi, w_mu, w_phi,
        cfg.N, cfg.M, cfg.K, G, dL,
        solar_mu, G_sol, omega_L, cfg.tau_L,
        cfg.rho_g, cfg.F_in, alpha)



