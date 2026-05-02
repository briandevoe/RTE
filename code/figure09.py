"""
figure09.py -- LAI effect on BHR and Transmittance

Sweeps LAI from 0.1 to 6.0 and plots both BHR and Transmittance for
RED and NIR bands on the same axes.

Transmittance = total downwelling irradiance at ground / F_in
BHR           = total upwelling irradiance at canopy top / F_in

Fixed conditions: SZA = 40 deg, fdir = 1.0, rho_g = 0.10 (RED) / 0.15 (NIR),
                  uniform LAD (G = 0.5)

Run from the repo root:
    python code/figure09.py
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
    'RED': dict(rho_L=0.06, tau_L=0.04, rho_g=0.10),
    'NIR': dict(rho_L=0.525, tau_L=0.45, rho_g=0.20),
}

LAI_ARR = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])

# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------
_nd, _wt = np.polynomial.legendre.leggauss(N)
MU    = np.zeros(N + 1);  MU[1:]    = _nd
W_MU  = np.zeros(N + 1);  W_MU[1:]  = _wt
PHI_Q = np.zeros(M + 1);  PHI_Q[1:] = (np.arange(M) / M) * 2 * np.pi
W_PHI = 2 * np.pi / M

solar_mu  = math.cos(math.radians(THETA_SOLAR))
solar_phi = math.radians(PHI_SOLAR)


def sweep_lai(rho_L, tau_L, rho_g, G_qq, G_sol):
    """Return (BHR_arr, Trans_arr) over LAI_ARR."""
    omega_L = rho_L + tau_L
    BHR   = np.zeros(len(LAI_ARR))
    Trans = np.zeros(len(LAI_ARR))
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
        Fd_tot = Fd_dir + Fd_dif + Fd_col
        Fu_tot = Fu_unc + Fu_col
        BHR[idx]   = float(Fu_tot[1])                              / F_IN
        Trans[idx] = float(Fd_tot[K+1] - Fu_tot[K+1])             / F_IN
    return BHR, Trans


# ---------------------------------------------------------------------------
# Run sweeps
# ---------------------------------------------------------------------------
print("=== Figure 09: BHR and Transmittance vs LAI ===")
results = {}   # key: band_name -> (BHR, Trans)

for band_name, bp in BANDS.items():
    rho_L, tau_L, rho_g = bp['rho_L'], bp['tau_L'], bp['rho_g']
    omega_L = rho_L + tau_L

    print(f"[{band_name}] precomputing phase tables...")
    with contextlib.redirect_stdout(io.StringIO()):
        G_qq  = precompute_G_qq(MU, PHI_Q, N, M, omega_L, tau_L)
        G_sol = precompute_G_sol(solar_mu, solar_phi, MU, PHI_Q, N, M, omega_L, tau_L)

    print(f"[{band_name}] sweeping {len(LAI_ARR)} LAI values...")
    bhr, trans = sweep_lai(rho_L, tau_L, rho_g, G_qq, G_sol)
    results[band_name] = (bhr, trans)
    print(f"  BHR: {bhr.min():.3f}–{bhr.max():.3f},  Trans: {trans.min():.3f}–{trans.max():.3f}")

# ---------------------------------------------------------------------------
# Plot: BHR and Transmittance on the same axes
# ---------------------------------------------------------------------------
BAND_STYLE = {
    'RED': dict(color='#d62728', marker='o'),
    'NIR': dict(color='#2ca02c', marker='s'),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left panel: BHR vs LAI ---
ax = axes[0]
for band_name, (bhr, _) in results.items():
    s = BAND_STYLE[band_name]
    bp = BANDS[band_name]
    ax.plot(LAI_ARR, bhr, color=s['color'], marker=s['marker'],
            linewidth=2, markersize=6,
            label=f'{band_name} ($\\rho_L$={bp["rho_L"]}, $\\tau_L$={bp["tau_L"]})')
ax.set_xlabel('Leaf Area Index (LAI)', fontsize=12)
ax.set_ylabel('BHR', fontsize=12)
ax.set_title('BHR vs LAI', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(0, LAI_ARR[-1])
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

# --- Right panel: Transmittance vs LAI ---
ax = axes[1]
for band_name, (_, trans) in results.items():
    s = BAND_STYLE[band_name]
    bp = BANDS[band_name]
    ax.plot(LAI_ARR, trans, color=s['color'], marker=s['marker'],
            linewidth=2, markersize=6,
            label=f'{band_name} ($\\rho_L$={bp["rho_L"]}, $\\tau_L$={bp["tau_L"]})')
ax.set_xlabel('Leaf Area Index (LAI)', fontsize=12)
ax.set_ylabel('Transmittance', fontsize=12)
ax.set_title('Transmittance vs LAI', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(0, LAI_ARR[-1])
ax.set_ylim(0, None)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

fig.suptitle(
    'Fig. 09 — LAI Effect on BHR and Transmittance\n'
    f'SZA=40°, fdir=1.0, uniform LAD',
    fontsize=13, fontweight='bold')
plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure09_BHR_Trans_vs_LAI.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.show()
