

import math
import sys
import os

# Add parent directory to path so DOM_v2 modules find each other
sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from step1_quadrature import get_polar_quadrature, get_azimuth_quadrature
from step2_phase      import precompute_G_qq, precompute_G_sol
from step3_uncollided import solve_uncollided
from step4_collided   import solve_collided
from step5_energy     import energy_balance
from step6_brf        import brf_at_view


def main():
    print("=" * 60)
    print("  DOM_v2 : Leaf Canopy Radiative Transfer")
    print("=" * 60)
    print(f"  Quadrature : {cfg.QUADRATURE}  (N={cfg.N}, M={cfg.M})")
    print(f"  Sweep      : {cfg.SWEEP}  (alpha={cfg.ALPHA_DD})")
    print(f"  Band       : rho_L={cfg.rho_L}, tau_L={cfg.tau_L}")
    print(f"  LAI={cfg.LAI}, K={cfg.K}, f_dir={cfg.f_dir}")
    print()

    # Derived quantities 
    omega_L   = cfg.rho_L + cfg.tau_L
    G         = 0.5          # uniform leaf angle distribution
    dL        = cfg.LAI / cfg.K
    solar_mu  = math.cos(math.radians(cfg.theta_solar))
    solar_phi = math.radians(cfg.phi_solar)
    alpha     = 0.5 if cfg.SWEEP == "diamond_diff" else 1.0

    # Step 1: Quadrature 
    mu,  w_mu  = get_polar_quadrature(cfg.N, cfg.QUADRATURE)
    phi, w_phi = get_azimuth_quadrature(cfg.M)

    # Step 2: Phase tables 
    G_qq  = precompute_G_qq(mu, phi, cfg.N, cfg.M, omega_L, cfg.tau_L)
    G_sol = precompute_G_sol(solar_mu, solar_phi,
                             mu, phi, cfg.N, cfg.M, omega_L, cfg.tau_L)

    # Step 3: Uncollided field + First Collision Source 
    print("\nSolving uncollided field ...")
    I0_dir, I0, Fd_dir, Fd_dif, Fu_unc, Q = solve_uncollided(
        mu, phi, w_mu, w_phi, cfg.N, cfg.M, cfg.K,
        solar_mu, cfg.f_dir, cfg.F_in, cfg.rho_g, G, dL,
        G_qq, G_sol)

    # Step 4: Collided field (iterative) 
    print("\nSolving collided field ...")
    IC, Fd_col, Fu_col = solve_collided(
        mu, phi, w_mu, w_phi, cfg.N, cfg.M, cfg.K,
        G, dL, cfg.rho_g, Q, G_qq, omega_L,
        tol=cfg.TOL, max_iter=cfg.MAX_ITER, alpha=alpha)

    # Step 5: Energy balance    
    result = energy_balance(
        I0_dir, I0, IC,
        Fd_dir, Fd_dif, Fu_unc, Fd_col, Fu_col,
        mu, w_mu, w_phi, cfg.N, cfg.M, cfg.K,
        G, dL, omega_L, cfg.rho_g, cfg.F_in)

    # Step 6: BRF at sample view directions 
    print("\n=== BRF / HDRF (backscatter plane, phi_v=180 deg) ===")
    for vz in [10, 30, 45, 60]:
        brf = brf_at_view(
            vz, 180.0,
            I0, IC, I0_dir, mu, phi, w_mu, w_phi,
            cfg.N, cfg.M, cfg.K, G, dL,
            solar_mu, G_sol, omega_L, cfg.tau_L,
            cfg.rho_g, cfg.F_in, alpha)
        print(f"  view zenith = {vz:2d} deg  ->  BRF = {brf:.4f}")

    return result


if __name__ == "__main__":
    main()
