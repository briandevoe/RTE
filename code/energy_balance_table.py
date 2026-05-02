"""
energy_balance_table.py -- Energy balance table for 4 canopy configurations

Runs DOM_v3 with fdir=1.0 (pure direct beam → DHR) for:
    RED  × LAI = 1.5
    RED  × LAI = 4.0
    NIR  × LAI = 1.5
    NIR  × LAI = 4.0

Prints a formatted table and saves it as a PNG figure for the report.

Reference (outline, Slide 2): RED LAI=1.5 should converge in ~3 iterations
with energy imbalance ~ +0.095%.

Run from the repo root:
    python code/energy_balance_table.py
"""

import sys, os, io, math, contextlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DOM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       '..', 'final_report', 'DOM_v3'))
sys.path.insert(0, DOM_DIR)

from step2_phase_rewrite      import precompute_G_qq, precompute_G_sol
from step3_uncollided_rewrite import solve_uncollided
from step4_collided_rewrite   import solve_collided
from step5_energy_rewrite     import energy_balance

# ---------------------------------------------------------------------------
# Fixed parameters (fdir=1 → pure direct beam → DHR)
# ---------------------------------------------------------------------------
N, M, K     = 16, 16, 50
G_EXT       = 0.5
ALPHA       = 0.5
F_IN        = 1.0
F_DIR       = 1.0         # pure direct beam → DHR
THETA_SOLAR = 140.0       # SZA = 40 deg
PHI_SOLAR   = 0.0

CONFIGS = [
    dict(label='RED  LAI=1.5', band='RED', lai=1.5,
         rho_L=0.06, tau_L=0.04, rho_g=0.10),
    dict(label='RED  LAI=4.0', band='RED', lai=4.0,
         rho_L=0.06, tau_L=0.04, rho_g=0.10),
    dict(label='NIR  LAI=1.5', band='NIR', lai=1.5,
         rho_L=0.45, tau_L=0.45, rho_g=0.15),
    dict(label='NIR  LAI=4.0', band='NIR', lai=4.0,
         rho_L=0.45, tau_L=0.45, rho_g=0.15),
]

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

# ---------------------------------------------------------------------------
# Run all configurations
# ---------------------------------------------------------------------------
print("=== Energy Balance Table — fdir=1.0 (DHR), SZA=40° ===\n")
rows = []
for cfg in CONFIGS:
    omega_L = cfg['rho_L'] + cfg['tau_L']
    dL      = cfg['lai'] / K

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        G_qq  = precompute_G_qq(MU, PHI_Q, N, M, omega_L, cfg['tau_L'])
        G_sol = precompute_G_sol(solar_mu, solar_phi, MU, PHI_Q, N, M, omega_L, cfg['tau_L'])
        I0_dir, I0, Fd_dir, Fd_dif, Fu_unc, Q = solve_uncollided(
            MU, PHI_Q, W_MU, W_PHI, N, M, K,
            solar_mu, F_DIR, F_IN, cfg['rho_g'], G_EXT, dL, G_qq, G_sol)
        IC, Fd_col, Fu_col = solve_collided(
            MU, PHI_Q, W_MU, W_PHI, N, M, K,
            G_EXT, dL, cfg['rho_g'], Q, G_qq, omega_L,
            tol=0.01, max_iter=500, alpha=ALPHA)
        res = energy_balance(
            I0_dir, I0, IC,
            Fd_dir, Fd_dif, Fu_unc, Fd_col, Fu_col,
            MU, W_MU, W_PHI, N, M, K,
            G_EXT, dL, omega_L, cfg['rho_g'], F_IN)

    # Extract iteration count from captured output
    output_text = buf.getvalue()
    iters = None
    for line in output_text.splitlines():
        if 'iter' in line.lower() and 'converge' in line.lower():
            try:
                iters = int([t for t in line.split() if t.isdigit()][0])
            except Exception:
                pass

    Fd_tot = res['Fd_tot']
    Fu_tot = res['Fu_tot']
    transmittance = float(Fd_tot[K + 1]) / F_IN   # downward flux at ground

    row = dict(
        label       = cfg['label'],
        band        = cfg['band'],
        lai         = cfg['lai'],
        DHR         = res['F_ref'],
        Trans       = transmittance,
        Absorptance = res['A_leaves'],
        GroundAbs   = res['F_abs_g'],
        Imbalance   = res['error_pct'],
        iters       = iters,
    )
    rows.append(row)

    print(f"[{cfg['label']}]")
    print(f"  DHR (reflectance)   : {row['DHR']:.4f}  ({100*row['DHR']:.2f}%)")
    print(f"  Transmittance       : {row['Trans']:.4f}  ({100*row['Trans']:.2f}%)")
    print(f"  Canopy absorptance  : {row['Absorptance']:.4f}  ({100*row['Absorptance']:.2f}%)")
    print(f"  Ground absorptance  : {row['GroundAbs']:.4f}  ({100*row['GroundAbs']:.2f}%)")
    print(f"  Energy imbalance    : {row['Imbalance']:+.3f}%")
    print()

# ---------------------------------------------------------------------------
# Save as formatted table figure
# ---------------------------------------------------------------------------
col_headers = ['Configuration', 'DHR', 'Transmittance', 'Canopy Abs.', 'Ground Abs.', 'Imbalance (%)']
table_data  = [
    [r['label'],
     f"{r['DHR']:.4f}",
     f"{r['Trans']:.4f}",
     f"{r['Absorptance']:.4f}",
     f"{r['GroundAbs']:.4f}",
     f"{r['Imbalance']:+.3f}%"]
    for r in rows
]

fig, ax = plt.subplots(figsize=(11, 4))
ax.axis('off')

fig.suptitle(
    'Energy Balance Table — DOM_v3\n'
    f'fdir=1.0 (pure direct beam → DHR),  SZA=40°,  '
    f'N={N}, M={M}, K={K},  uniform LAD (G=0.5)',
    fontsize=12, fontweight='bold', y=0.98)

tbl = ax.table(
    cellText=table_data,
    colLabels=col_headers,
    cellLoc='center',
    loc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 2.0)

# Header row styling
for j in range(len(col_headers)):
    tbl[0, j].set_facecolor('#2c4a7c')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Row colouring: RED rows warm, NIR rows cool
row_colors = ['#fff0ee', '#fff0ee', '#eef4ff', '#eef4ff']
for i, color in enumerate(row_colors):
    for j in range(len(col_headers)):
        tbl[i + 1, j].set_facecolor(color)

# Imbalance column: highlight if > 1%
for i, row in enumerate(rows):
    cell = tbl[i + 1, 5]
    if abs(row['Imbalance']) > 1.0:
        cell.set_facecolor('#ffe0e0')

ax.text(0.5, 0.02,
        'DHR = Directional-Hemispherical Reflectance (= BHR when fdir=1.0).  '
        'Transmittance = F↓_ground / F_in.  '
        'Imbalance = F_in − (DHR + Canopy Abs. + Ground Abs.).',
        ha='center', va='bottom', fontsize=8.5, color='dimgray',
        transform=ax.transAxes)

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'energy_balance_table.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
