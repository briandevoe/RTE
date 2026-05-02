"""
figure05.py -- Uncollided radiation profiles, NIR band

Three-panel figure showing uncollided-only radiation at two LAI values:
  Plot A: Normalised downward flux vs L/LAI (direct + diffuse sky)
  Plot B: Normalised upward flux vs L/LAI (ground-reflected)
  Plot C: Uncollided BRF in the principal plane at L=0

Uses dom_uncollided.py's solver (Beer-Lambert, no scattering).
Parameters: SZA=40°, fdir=0.7, phi=0°, N=16, M=16, K=50.

Run from the repo root:
    python code/figure05.py
"""

import sys, os
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from dom_uncollided import run_uncollided_dom_solver, make_three_panel_plots

# ---------------------------------------------------------------------------
# Parameters  (match dom_uncollided.py defaults)
# ---------------------------------------------------------------------------
N, M, K     = 16, 16, 50
F_IN        = 1.0
F_DIR       = 0.7
THETA_SOLAR = 140.0   # SZA = 40 deg
PHI_SOLAR   = 0.0
LAI_LIST    = [1.5, 4.0]

NIR_LEAF  = {'rho_L': 0.525, 'tau_L': 0.45}
NIR_RHO_G = 0.20

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("=== Figure 05: Uncollided Radiation — NIR band ===")
results = []
for lai in LAI_LIST:
    print(f"  LAI={lai}...")
    canopy = dict(LAI=lai, f_dir=F_DIR,
                  theta_o_deg=THETA_SOLAR, phi_o_deg=PHI_SOLAR,
                  rho_g=NIR_RHO_G, F_in=F_IN)
    results.append(run_uncollided_dom_solver(canopy, NIR_LEAF, N, M, K))

fig = make_three_panel_plots(results[0], results[1], 'NIR', LAI_LIST)

out_dir = os.path.join(os.path.dirname(CODE_DIR), 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure05_uncollided_NIR.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.show()
