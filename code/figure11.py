"""
figure11.py -- Konza Prairie HDRF: DOM model vs measured field data

Compares DOM_v3 principal-plane HDRF output against field measurements
from Konza Prairie grassland (Prof. Myneni slides 15-16).

Konza Prairie parameters (from slide 16):
  LAI=2.2, SZA=70° (theta_solar=110°), phi_solar=0°, uniform LAD (G=0.5)
  RED: rho_L=0.1814, tau_L=0.0926, rho_g=0.0825, fdir=0.862
  NIR: rho_L=0.4525, tau_L=0.4913, rho_g=0.1363, fdir=0.931

Field data (slide 15): principal plane, negative VZA = backscatter,
positive VZA = forward scatter.

Convention (phi_solar=0 = beam direction):
  Backscatter (toward sun, phi_view=180) → negative VZA axis.
  Forward scatter (away from sun, phi_view=0) → positive VZA axis.
  Hot-spot expected near VZA = -SZA = -70 deg.

Note: DOM_v3 uses a bi-Lambertian phase function with no hot-spot singularity,
so the model will not show the sharp retro-illumination peak visible in the
field data near VZA = -70°.

Run from the repo root:
    python code/figure11.py
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
# Konza Prairie parameters
# ---------------------------------------------------------------------------
N, M, K     = 16, 16, 50
G           = 0.5
ALPHA       = 0.5
F_IN        = 1.0
LAI         = 2.2
THETA_SOLAR = 110.0     # SZA = 70 deg (theta_solar = 180 - SZA)
PHI_SOLAR   = 0.0
SZA_DEG     = 180.0 - THETA_SOLAR   # 70 deg

BANDS = {
    'RED': dict(rho_L=0.1814, tau_L=0.0926, rho_g=0.0825, fdir=0.862),
    'NIR': dict(rho_L=0.4525, tau_L=0.4913, rho_g=0.1363, fdir=0.931),
}

# ---------------------------------------------------------------------------
# Field measurements from Konza Prairie (slide 15)
# Negative VZA = backscatter, positive = forward scatter
# ---------------------------------------------------------------------------
FIELD_DATA = {
    'RED': {
        -80: 0.127, -65: 0.096, -45: 0.072, -30: 0.054,
        -20: 0.043, -10: 0.043,   0: 0.038,  10: 0.038,
         20: 0.047,  30: 0.047,  45: 0.053,  70: 0.070,  80: 0.116,
    },
    'NIR': {
        -80: 0.835, -65: 0.650, -40: 0.505, -25: 0.390,
        -15: 0.330,  -5: 0.305,  10: 0.345,  15: 0.350,
         28: 0.370,  40: 0.490,  55: 0.635,  70: 0.635,  80: 0.890,
    },
}

# Principal-plane view angles
VZA_ABS = np.linspace(2, 80, 40)
PHI_BACK = 180.0   # looking toward sun (backscatter)
PHI_FWD  = 0.0    # looking away from sun (forward scatter)

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


def compute_principal_plane(band_name):
    """Run DOM and return (vza_axis, hdrf_axis) for the principal plane."""
    p = BANDS[band_name]
    rho_L, tau_L, rho_g, fdir = p['rho_L'], p['tau_L'], p['rho_g'], p['fdir']
    omega_L = rho_L + tau_L
    dL = LAI / K

    with contextlib.redirect_stdout(io.StringIO()):
        G_qq  = precompute_G_qq(MU, PHI_Q, N, M, omega_L, tau_L)
        G_sol = precompute_G_sol(solar_mu, solar_phi, MU, PHI_Q, N, M, omega_L, tau_L)
        I0_dir, I0, Fd_dir, Fd_dif, Fu_unc, Q = solve_uncollided(
            MU, PHI_Q, W_MU, W_PHI, N, M, K,
            solar_mu, fdir, F_IN, rho_g, G, dL, G_qq, G_sol)
        IC, Fd_col, Fu_col = solve_collided(
            MU, PHI_Q, W_MU, W_PHI, N, M, K,
            G, dL, rho_g, Q, G_qq, omega_L,
            tol=0.01, max_iter=500, alpha=ALPHA)

    # From textbook Ch4 Eq(4b): total E_i = F_in = 1.0 exactly.
    # step6 divides by mu_v*F_in; multiply by mu_v to recover standard HDRF.
    E_i_norm = 1.0
    mu_v_arr = np.cos(np.radians(VZA_ABS))

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

    vza_axis  = np.concatenate([-VZA_ABS[::-1], VZA_ABS])
    hdrf_axis = np.concatenate([brf_back[::-1], brf_fwd])
    return vza_axis, hdrf_axis


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("=== Figure 11: Konza Prairie — DOM vs Field Data (SZA=70°) ===")
dom_results = {}
for band_name in ['RED', 'NIR']:
    print(f"[{band_name}]...")
    dom_results[band_name] = compute_principal_plane(band_name)

# ---------------------------------------------------------------------------
# Plot: 2 panels (RED left, NIR right)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

fig.suptitle(
    f'HDRF — Principal Plane  |  Konza Prairie: DOM vs Measured  |  '
    f'LAI={LAI}, SZA={SZA_DEG:.0f}°, N={N}, M={M}, K={K}',
    fontsize=11)

BAND_STYLE = {
    'RED': dict(model_color='#d62728',
                title='RED — Principal Plane HDRF\n'
                      r'$\rho_L$=0.1814, $\tau_L$=0.0926, '
                      r'$\omega_L$=0.274, $\rho_g$=0.0825, fdir=0.862'),
    'NIR': dict(model_color='#1f77b4',
                title='NIR — Principal Plane HDRF\n'
                      r'$\rho_L$=0.4525, $\tau_L$=0.4913, '
                      r'$\omega_L$=0.944, $\rho_g$=0.1363, fdir=0.931'),
}

for ax, band_name in zip(axes, ['RED', 'NIR']):
    s = BAND_STYLE[band_name]
    vza_axis, hdrf_axis = dom_results[band_name]

    # DOM model line
    ax.plot(vza_axis, hdrf_axis,
            color=s['model_color'], lw=2, label='DOM model')

    # Field data scatter
    fd = FIELD_DATA[band_name]
    fd_vza  = np.array(sorted(fd.keys()))
    fd_hdrf = np.array([fd[v] for v in fd_vza])
    ax.scatter(fd_vza, fd_hdrf,
               color='black', s=50, zorder=5, marker='*',
               label='Konza Prairie (measured)')

    # Solar position line at backscatter -SZA
    ax.axvline(-SZA_DEG, color='orange', ls='--', lw=1.5,
               label=f'Solar ($\\theta$={SZA_DEG:.0f}° back.)')
    ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.6)

    ax.set_xlabel('View Zenith Angle (degrees)\n← Back-scatter  |  Forward-scatter →',
                  fontsize=11)
    ax.set_ylabel('HDRF', fontsize=12)
    ax.set_title(s['title'], fontsize=10)
    ax.set_xlim(-82, 82)
    ax.set_ylim(0, None)
    ax.set_xticks(range(-80, 81, 20))
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure11_Konza_DOM_vs_field.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.show()
