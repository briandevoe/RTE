"""
figure07.py -- HDRF principal-plane cross-section, 2 panels (fdir = 0.7)

Plots HDRF vs. view zenith angle in the principal plane for RED and NIR bands,
at LAI = 1.5 and LAI = 4.0. Two panels: RED (left) and NIR (right).

Note: the Y-axis is labeled "BRF" (as the professor's code outputs), but the
quantity is HDRF because fdir = 0.7 (not pure direct beam). This matches slide 8:
"Although the Y axis says BRF, it is really HDRF because fdir=0.7."

Convention (phi_solar=0 = beam direction):
  Backscatter (toward sun, phi_view=180) → negative VZA axis.
  Forward scatter (away from sun, phi_view=0) → positive VZA axis.
  Hot-spot expected near VZA = -SZA = -40 deg.

Run from the repo root:
    python code/figure07.py
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
from step6_brf_rewrite        import brf_at_view

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N, M, K     = 16, 16, 50
G           = 0.5
ALPHA       = 0.5
F_IN        = 1.0
THETA_SOLAR = 140.0     # SZA = 40 deg
PHI_SOLAR   = 0.0
F_DIR       = 0.70      # mixed illumination -> HDRF (but labeled BRF per code)

SZA_DEG = 180.0 - THETA_SOLAR   # 40 deg

BANDS = {
    'RED': dict(rho_L=0.06, tau_L=0.04, rho_g=0.10),
    'NIR': dict(rho_L=0.525, tau_L=0.45, rho_g=0.20),
}
LAI_LIST = [1.5, 4.0]
LAI_STYLES = {
    1.5: dict(color='#1f77b4', ls='-',  lw=2,   label='LAI = 1.5'),
    4.0: dict(color='#d62728', ls='--', lw=2,   label='LAI = 4.0'),
}

# Principal plane view angles (degrees)
VZA_ABS = np.linspace(2, 80, 40)   # zenith magnitudes 2..80

# Backscatter: phi_view=180 deg (looking toward sun's sky position)
PHI_BACK = 180.0
# Forward scatter: phi_view=0 deg (looking away from sun)
PHI_FWD  = 0.0

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


def compute_principal_plane(band_name, lai):
    """Return (vza_axis, hdrf_axis) for the principal plane.

    step6_brf_rewrite computes pi*I_v / (mu_v * F_in), but the standard
    HDRF definition normalises by total incident irradiance E_i (not mu_v).
    We correct by multiplying each point by mu_v / E_i_norm where
        E_i_norm = fdir*|mu_solar| + (1-fdir)   [normalised to F_in=1]
    """
    p = BANDS[band_name]
    rho_L, tau_L, rho_g = p['rho_L'], p['tau_L'], p['rho_g']
    omega_L = rho_L + tau_L
    dL = lai / K

    # From textbook Ch4 Eq(4b): I_o = (f_dir/|mu_o|)*F_in*delta + (1-f_dir)/pi*F_in
    # => E_i = f_dir*F_in + (1-f_dir)*F_in = F_in = 1.0 exactly.
    # step6 divides by mu_v*F_in; standard BRF divides by F_in => multiply by mu_v only.
    E_i_norm = 1.0

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

    mu_v_arr = np.cos(np.radians(VZA_ABS))   # correction per angle

    brf_back = np.array([
        brf_at_view(vz, PHI_BACK, I0, IC, I0_dir, MU, PHI_Q, W_MU, W_PHI,
                    N, M, K, G, dL, solar_mu, G_sol, omega_L, tau_L,
                    rho_g, F_IN, ALPHA)
        for vz in VZA_ABS]) * mu_v_arr / E_i_norm

    brf_fwd = np.array([
        brf_at_view(vz, PHI_FWD, I0, IC, I0_dir, MU, PHI_Q, W_MU, W_PHI,
                    N, M, K, G, dL, solar_mu, G_sol, omega_L, tau_L,
                    rho_g, F_IN, ALPHA)
        for vz in VZA_ABS]) * mu_v_arr / E_i_norm

    # negative VZA = backscatter (reversed), positive = forward scatter
    vza_axis = np.concatenate([-VZA_ABS[::-1], VZA_ABS])
    brf_axis = np.concatenate([brf_back[::-1], brf_fwd])
    return vza_axis, brf_axis


# ---------------------------------------------------------------------------
# Run all 4 combinations
# ---------------------------------------------------------------------------
print("=== Figure 07: HDRF Principal Plane (fdir=0.70) ===")
results = {}
for band_name in ['RED', 'NIR']:
    for lai in LAI_LIST:
        print(f"[{band_name}, LAI={lai}]...")
        results[(band_name, lai)] = compute_principal_plane(band_name, lai)

# ---------------------------------------------------------------------------
# Plot: 2 panels (RED left, NIR right)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

fig.suptitle(
    f'HDRF — Principal Plane  |  $f_{{dir}}$={F_DIR},  SZA={SZA_DEG:.0f}°,  '
    f'$\\phi_{{solar}}$={PHI_SOLAR:.0f}°,  N={N}, M={M}, K={K}',
    fontsize=12)

band_titles = {
    'RED': f"RED — HDRF Principal Plane\n"
           f"$\\rho_L$=0.06, $\\tau_L$=0.04, $\\omega_L$=0.10, $\\rho_g$=0.10",
    'NIR': f"NIR — HDRF Principal Plane\n"
           f"$\\rho_L$=0.525, $\\tau_L$=0.45, $\\omega_L$=0.975, $\\rho_g$=0.20",
}

for ax, band_name in zip(axes, ['RED', 'NIR']):
    for lai in LAI_LIST:
        vza_axis, brf_axis = results[(band_name, lai)]
        s = LAI_STYLES[lai]
        ax.plot(vza_axis, brf_axis,
                color=s['color'], linestyle=s['ls'], linewidth=s['lw'],
                label=s['label'])

    # Solar position line (backscatter at -SZA)
    ax.axvline(-SZA_DEG, color='orange', linestyle='--', linewidth=1.5,
               label=f'Solar ($\\theta$={SZA_DEG:.0f}° back.)')
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

    ax.set_xlabel('View Zenith Angle (degrees)\n← Back-scatter  |  Forward-scatter →',
                  fontsize=11)
    ax.set_ylabel('HDRF', fontsize=12)
    ax.set_title(band_titles[band_name], fontsize=11)
    ax.set_xlim(-82, 82)
    ax.set_ylim(0, None)
    ax.set_xticks(range(-80, 81, 20))
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate normalisation note
    ax.text(0.98, 0.97, f'fdir={F_DIR}, BRF = $\\pi I_v / F_{{in}}$',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='gray', style='italic')

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure07_HDRF_principal_plane.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.show()
