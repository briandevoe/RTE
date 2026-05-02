"""
figure10.py -- Solar polar angle effect on BHR and Transmittance (NIR, two LAI)

Sweeps solar polar angle (theta_solar) from 100 to 175 degrees (SZA 5–80°)
for two LAI values (1.5 and 4.0), NIR band only.

theta_solar convention (DOM_v3): angle from UPWARD zenith for solar direction.
    theta_solar = 180 - SZA  (e.g. SZA=40 -> theta_solar=140)

Fixed conditions: NIR band, fdir=0.70, rho_g=0.20, uniform LAD (G=0.5)

Run from the repo root:
    python code/figure10.py
"""

import sys, os, io, math, contextlib
import numpy as np
import matplotlib.pyplot as plt

DOM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DOM_v3'))
sys.path.insert(0, DOM_DIR)

from step2_phase_rewrite      import precompute_G_qq, precompute_G_sol
from step3_uncollided_rewrite import solve_uncollided
from step4_collided_rewrite   import solve_collided

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N, M, K   = 16, 16, 50
G         = 0.5
ALPHA     = 0.5
F_IN      = 1.0
F_DIR     = 0.70
PHI_SOLAR = 0.0

RHO_L = 0.525
TAU_L = 0.45
RHO_G = 0.20
OMEGA_L = RHO_L + TAU_L

LAI_LIST   = [1.5, 4.0]
LAI_STYLES = {
    1.5: dict(color='#1f77b4', ls='-',  lw=2, marker='o', label='LAI = 1.5'),
    4.0: dict(color='#d62728', ls='--', lw=2, marker='s', label='LAI = 4.0'),
}

# Sweep theta_solar from 92 to 178 (SZA from 2 to 88 deg)
# Near-grazing (low theta_solar) produces steep BHR upturn
THETA_SOL_ARR = np.array([92, 95, 100, 110, 120, 130, 140, 150, 160, 170, 178])

# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------
_nd, _wt = np.polynomial.legendre.leggauss(N)
MU    = np.zeros(N + 1);  MU[1:]    = _nd
W_MU  = np.zeros(N + 1);  W_MU[1:]  = _wt
PHI_Q = np.zeros(M + 1);  PHI_Q[1:] = (np.arange(M) / M) * 2 * np.pi
W_PHI = 2 * np.pi / M

solar_phi = math.radians(PHI_SOLAR)


def sweep_theta_solar(lai):
    """Return (BHR_arr, Trans_arr) over THETA_SOL_ARR for NIR band."""
    dL = lai / K
    BHR   = np.zeros(len(THETA_SOL_ARR))
    Trans = np.zeros(len(THETA_SOL_ARR))

    with contextlib.redirect_stdout(io.StringIO()):
        G_qq = precompute_G_qq(MU, PHI_Q, N, M, OMEGA_L, TAU_L)

    for idx, theta_solar in enumerate(THETA_SOL_ARR):
        sol_mu = math.cos(math.radians(theta_solar))
        with contextlib.redirect_stdout(io.StringIO()):
            G_sol = precompute_G_sol(sol_mu, solar_phi, MU, PHI_Q, N, M, OMEGA_L, TAU_L)
            I0_dir, I0, Fd_dir, Fd_dif, Fu_unc, Q = solve_uncollided(
                MU, PHI_Q, W_MU, W_PHI, N, M, K,
                sol_mu, F_DIR, F_IN, RHO_G, G, dL, G_qq, G_sol)
            IC, Fd_col, Fu_col = solve_collided(
                MU, PHI_Q, W_MU, W_PHI, N, M, K,
                G, dL, RHO_G, Q, G_qq, OMEGA_L,
                tol=0.01, max_iter=500, alpha=ALPHA)
        Fd_tot = Fd_dir + Fd_dif + Fd_col
        Fu_tot = Fu_unc + Fu_col
        BHR[idx]   = float(Fu_tot[1])                      / F_IN
        Trans[idx] = float(Fd_tot[K+1] - Fu_tot[K+1])     / F_IN
    return BHR, Trans


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("=== Figure 10: NIR BHR and Transmittance vs Solar Polar Angle ===")
results = {}
for lai in LAI_LIST:
    print(f"[LAI={lai}] sweeping {len(THETA_SOL_ARR)} solar angles...")
    bhr, trans = sweep_theta_solar(lai)
    results[lai] = (bhr, trans)
    print(f"  BHR: {bhr.min():.3f}–{bhr.max():.3f},  Trans: {trans.min():.3f}–{trans.max():.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, quantity in zip(axes, ['BHR', 'Transmittance']):
    for lai in LAI_LIST:
        bhr, trans = results[lai]
        data = bhr if quantity == 'BHR' else trans
        s = LAI_STYLES[lai]
        ax.plot(THETA_SOL_ARR, data,
                color=s['color'], linestyle=s['ls'], linewidth=s['lw'],
                marker=s['marker'], markersize=6, label=s['label'])

    ax.set_xlabel('Solar Polar Angle (deg from upward zenith)', fontsize=12)
    ax.set_ylabel(quantity, fontsize=12)
    ax.set_title(f'NIR — {quantity}\n'
                 f'$\\rho_L$={RHO_L}, $\\tau_L$={TAU_L}, '
                 f'$\\omega_L$={OMEGA_L:.3f}, $\\rho_g$={RHO_G}', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(180, 90)   # reversed: overhead sun (180) at left, horizon (90) at right
    ax.set_ylim(0, None)
    ax.set_xticks([180, 165, 150, 135, 120, 105, 90])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

fig.suptitle(
    'Fig. 10 — Solar Polar Angle Effect on NIR BHR and Transmittance\n'
    f'fdir={F_DIR}, $\\rho_g$={RHO_G}, uniform LAD (G=0.5)',
    fontsize=13, fontweight='bold')
plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure10_BHR_Trans_vs_SZA.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.show()
