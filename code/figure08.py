"""
figure08.py -- BHR as a function of LAI and soil brightness

Sweeps LAI from 0.1 to 6.0 for two soil reflectance conditions:
  - Dark soil:   rho_g = 0.01
  - Bright soil: rho_g = 0.30

Produces 2 subplots (RED and NIR), each showing BHR vs LAI for both soils.

Fixed conditions: SZA = 40 deg, fdir = 1.0, uniform LAD (G = 0.5)

Run from the repo root:
    python code/figure08.py
"""

import sys, os, io, math, contextlib
import numpy as np
import matplotlib.pyplot as plt

DOM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       '..', 'final_report', 'DOM_v3'))
sys.path.insert(0, DOM_DIR)

from step2_phase_rewrite      import precompute_G_qq, precompute_G_sol
from step3_uncollided_rewrite import solve_uncollided
from step4_collided_rewrite   import solve_collided

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N, M, K     = 16, 16, 50
G           = 0.5
ALPHA       = 0.5
F_IN        = 1.0
THETA_SOLAR = 140.0     # SZA = 40 deg
PHI_SOLAR   = 0.0
F_DIR       = 1.0

BANDS = {
    'RED': dict(rho_L=0.06, tau_L=0.04),
    'NIR': dict(rho_L=0.525, tau_L=0.45),
}

SOILS = {
    'Dark soil ($\\rho_g=0.01$)':   0.01,
    'Bright soil ($\\rho_g=0.30$)': 0.30,
}

LAI_ARR = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])

# ---------------------------------------------------------------------------
# Quadrature (fixed)
# ---------------------------------------------------------------------------
_nd, _wt = np.polynomial.legendre.leggauss(N)
MU    = np.zeros(N + 1);  MU[1:]    = _nd
W_MU  = np.zeros(N + 1);  W_MU[1:]  = _wt
PHI_Q = np.zeros(M + 1);  PHI_Q[1:] = (np.arange(M) / M) * 2 * np.pi
W_PHI = 2 * np.pi / M

solar_mu  = math.cos(math.radians(THETA_SOLAR))
solar_phi = math.radians(PHI_SOLAR)


def sweep_lai(rho_L, tau_L, rho_g, G_qq, G_sol):
    """
    Sweep over LAI_ARR at fixed spectral and solar parameters.
    Returns array of BHR values.
    G_qq and G_sol are precomputed outside and reused here for efficiency.
    """
    omega_L = rho_L + tau_L
    BHR = np.zeros(len(LAI_ARR))
    for idx, lai in enumerate(LAI_ARR):
        dL = lai / K
        with contextlib.redirect_stdout(io.StringIO()):
            I0_dir, I0, Fd_dir, Fd_dif, Fu_unc, Q = solve_uncollided(
                MU, PHI_Q, W_MU, W_PHI, N, M, K,
                solar_mu, F_DIR, F_IN, rho_g, G, dL, G_qq, G_sol)
            IC, Fd_col, Fu_col = solve_collided(
                MU, PHI_Q, W_MU, W_PHI, N, M, K,
                G, dL, rho_g, Q, G_qq, omega_L,
                tol=0.01, max_iter=500, alpha=ALPHA)
        Fu_tot = Fu_unc + Fu_col
        BHR[idx] = float(Fu_tot[1]) / F_IN
    return BHR


# ---------------------------------------------------------------------------
# Compute BHR for each band x soil combination
# ---------------------------------------------------------------------------
print("=== Figure 08: BHR vs LAI and Soil Brightness ===")
results = {}   # key: (band, soil_label) -> BHR array

for band_name, bp in BANDS.items():
    rho_L, tau_L = bp['rho_L'], bp['tau_L']
    omega_L = rho_L + tau_L

    print(f"[{band_name}] precomputing phase tables...")
    with contextlib.redirect_stdout(io.StringIO()):
        G_qq  = precompute_G_qq(MU, PHI_Q, N, M, omega_L, tau_L)
        G_sol = precompute_G_sol(solar_mu, solar_phi, MU, PHI_Q, N, M, omega_L, tau_L)

    for soil_label, rho_g in SOILS.items():
        print(f"  [{band_name}, {soil_label[:4]}] sweeping {len(LAI_ARR)} LAI values...")
        bhr = sweep_lai(rho_L, tau_L, rho_g, G_qq, G_sol)
        results[(band_name, soil_label)] = bhr
        print(f"    BHR range: {bhr.min():.3f} – {bhr.max():.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
COLORS = {
    'Dark soil ($\\rho_g=0.01$)':   ('#2c7bb6', '--'),
    'Bright soil ($\\rho_g=0.30$)': ('#d7191c', '-'),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

for ax, band_name in zip(axes, ['RED', 'NIR']):
    bp = BANDS[band_name]
    for soil_label, rho_g in SOILS.items():
        color, ls = COLORS[soil_label]
        bhr = results[(band_name, soil_label)]
        ax.plot(LAI_ARR, bhr, color=color, linestyle=ls, linewidth=2,
                marker='o', markersize=5, label=soil_label)

    ax.set_xlabel('Leaf Area Index (LAI)', fontsize=12)
    ax.set_ylabel('BHR (Bihemispherical Reflectance)', fontsize=12)
    ax.set_title(
        f'{band_name} Band\n'
        f'($\\rho_L$={bp["rho_L"]}, $\\tau_L$={bp["tau_L"]})',
        fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, LAI_ARR[-1])
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

fig.suptitle(
    'Fig. 08 — BHR as a Function of LAI and Soil Brightness\n'
    f'SZA=40°, fdir=1.0, uniform LAD',
    fontsize=13, fontweight='bold')
plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure08_BHR_vs_LAI_soil.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.show()
