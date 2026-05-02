"""
figure06.py -- HDRF hemispherical polar map, 4 panels (fdir = 0.7)

Produces a 2x2 grid of polar hemispherical plots showing HDRF at
45 x 90 view directions for RED and NIR bands at LAI = 1.5 and 4.0.

Layout:
    [RED  LAI=1.5]  [RED  LAI=4.0]
    [NIR  LAI=1.5]  [NIR  LAI=4.0]

Polar orientation: 0 deg at top (forward-scatter direction),
                   180 deg at bottom (sun's sky position = hot-spot).

Convention: phi_solar=0 means beam travels toward phi=0 (top of plot).
            Sun's sky position is phi=180 (bottom), consistent with image9.jpg.

Run from the repo root:
    python code/figure06.py
"""

import sys, os, io, math, contextlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DOM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       '..', 'final_report', 'DOM_v3'))
sys.path.insert(0, DOM_DIR)

from step2_phase_rewrite      import precompute_G_qq, precompute_G_sol
from step3_uncollided_rewrite import solve_uncollided
from step4_collided_rewrite   import solve_collided
from step6_brf_rewrite        import brf_at_view

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N, M, K     = 16, 16, 50
G           = 0.5
ALPHA       = 0.5
F_IN        = 1.0
THETA_SOLAR = 140.0     # SZA = 40 deg
PHI_SOLAR   = 0.0       # beam toward phi=0; sun sky position at phi=180
F_DIR       = 0.70      # mixed illumination -> HDRF

SZA_DEG     = 180.0 - THETA_SOLAR   # 40 deg

BANDS = {
    'RED': dict(rho_L=0.06, tau_L=0.04, rho_g=0.10, cmap='YlOrRd'),
    'NIR': dict(rho_L=0.525, tau_L=0.45, rho_g=0.20, cmap='Blues'),
}
LAI_LIST = [1.5, 4.0]

# View angle grid: 45 zenith x 90 azimuth
VZ_DEG = np.linspace(2, 75, 45)           # cap at 75 to avoid limb singularity
VA_DEG = np.linspace(0, 360 - 360/90, 90)

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


def compute_hdrf_grid(band_name, lai):
    """Run DOM and compute HDRF on VZ_DEG x VA_DEG grid."""
    p = BANDS[band_name]
    rho_L, tau_L, rho_g = p['rho_L'], p['tau_L'], p['rho_g']
    omega_L = rho_L + tau_L
    dL = lai / K

    with contextlib.redirect_stdout(io.StringIO()):
        G_qq  = precompute_G_qq(MU, PHI_Q, N, M, omega_L, tau_L)
        G_sol = precompute_G_sol(solar_mu, solar_phi, MU, PHI_Q, N, M, omega_L, tau_L)
        I0_dir, I0, Fd_dir, Fd_dif, Fu_unc, Q = solve_uncollided(
            MU, PHI_Q, W_MU, W_PHI, N, M, K,
            solar_mu, F_DIR, F_IN, rho_g, G, dL, G_qq, G_sol)
        IC, Fd_col, Fu_col = solve_collided(
            MU, PHI_Q, W_MU, W_PHI, N, M, K,
            G, dL, rho_g, Q, G_qq, omega_L,
            tol=0.01, max_iter=500, alpha=ALPHA)

    # Textbook Ch4 Eq(4b): total E_i = F_in = 1.0 exactly (direct + diffuse sum to F_in).
    # step6 divides by mu_v*F_in; correct BRF divides by F_in => multiply raw by mu_v only.
    E_i_norm = 1.0

    HDRF = np.zeros((len(VZ_DEG), len(VA_DEG)))
    for i, vz in enumerate(VZ_DEG):
        if i % 9 == 0:
            print(f"    {int(100*i/len(VZ_DEG))}%...")
        mu_v = math.cos(math.radians(vz))
        for j, va in enumerate(VA_DEG):
            raw = brf_at_view(
                vz, va, I0, IC, I0_dir, MU, PHI_Q, W_MU, W_PHI,
                N, M, K, G, dL, solar_mu, G_sol, omega_L, tau_L,
                rho_g, F_IN, ALPHA)
            HDRF[i, j] = raw * mu_v / E_i_norm
    return HDRF


# ---------------------------------------------------------------------------
# Compute all 4 panels
# ---------------------------------------------------------------------------
print("=== Figure 06: HDRF Polar Map (fdir=0.70) ===")
HDRF_grids = {}
for band_name in ['RED', 'NIR']:
    for lai in LAI_LIST:
        print(f"[{band_name}, LAI={lai}] running DOM + HDRF grid...")
        HDRF_grids[(band_name, lai)] = compute_hdrf_grid(band_name, lai)
        g = HDRF_grids[(band_name, lai)]
        print(f"  HDRF range: {g.min():.4f} – {g.max():.4f}")

# ---------------------------------------------------------------------------
# Plot: 2x2 polar subplots
# ---------------------------------------------------------------------------
# Close the azimuth ring so contourf has no seam at 0°/360°
VA_RAD_open = np.deg2rad(VA_DEG)
VA_RAD = np.append(VA_RAD_open, 2 * np.pi)         # 91 values, closes the circle
THETA_M, R_M = np.meshgrid(VA_RAD, VZ_DEG)         # shape (45, 91)

fig, axes = plt.subplots(2, 2, figsize=(13, 11),
                         subplot_kw={'projection': 'polar'},
                         constrained_layout=True)
fig.suptitle(
    f'HDRF — DOM  |  Sun: $\\phi$={PHI_SOLAR:.0f}°, SZA={SZA_DEG:.0f}°  |  '
    f'$f_{{dir}}$={F_DIR},  N={N}, M={M}, K={K}',
    fontsize=11)

panel_order = [('RED', 1.5), ('RED', 4.0), ('NIR', 1.5), ('NIR', 4.0)]

for ax, (band_name, lai) in zip(axes.flatten(), panel_order):
    cmap  = BANDS[band_name]['cmap']
    HDRF  = HDRF_grids[(band_name, lai)]

    # Polar orientation: 0 at top, clockwise
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 75)
    ax.set_yticks([20, 40, 60])
    ax.set_yticklabels(['20°', '40°', '60°'], fontsize=8, color='gray')
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(['0°', '90°', '180°', '270°'], fontsize=9)

    # Wrap first column to close the circle (91st column = repeat of 0°)
    HDRF_c = np.concatenate([HDRF, HDRF[:, :1]], axis=1)
    c = ax.contourf(THETA_M, R_M, HDRF_c, levels=25, cmap=cmap)
    cb = plt.colorbar(c, ax=ax, pad=0.08, shrink=0.7, label='HDRF')
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    # Sun marker at phi=180 (bottom of plot), r=SZA=40
    ax.plot(np.radians(180.0), SZA_DEG, 'y*',
            markersize=14, markeredgecolor='k', markeredgewidth=0.5, zorder=5)
    ax.text(np.radians(180.0), SZA_DEG + 11, 'Sun',
            ha='center', fontsize=8, color='yellow',
            fontweight='bold', zorder=5)

    # Forward scatter label at top
    ax.text(0, 48, 'Forward\nscatter',
            ha='center', va='center', fontsize=8, color='white',
            fontweight='bold', zorder=5)

    ax.set_title(f'{band_name} band — LAI = {lai}',
                 fontsize=11, pad=10)
    # min/max annotation at top-left corner of axes bounding box
    ax.text(0.02, 0.98, f'min={HDRF.min():.3f}  max={HDRF.max():.3f}',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=7.5, color='dimgray', zorder=6)

out_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure06_HDRF_polar_map.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.show()
