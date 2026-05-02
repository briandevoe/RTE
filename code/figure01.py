"""
figure01.py -- Leaf inclination angle distribution functions

Plots g_L(theta_L) for six LADs: Planophile, Erectophile, Plagiophile,
Extremophile, Spherical, and Uniform.

Run from the repo root:
    python code/figure01.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

theta_deg = np.linspace(0, 90, 500)
theta = np.deg2rad(theta_deg)

g_uniform      = np.ones_like(theta)
g_planophile   = 3 * np.cos(theta)**2
g_erectophile  = (3/2) * np.sin(theta)**2
g_plagiophile  = (15/8) * np.sin(2*theta)**2
g_extremophile = (15/8) * np.cos(2*theta)**2
g_spherical    = (4/np.pi) * np.sin(theta)

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(theta_deg, g_planophile,   label='Planophile',   linewidth=2)
ax.plot(theta_deg, g_erectophile,  label='Erectophile',  linewidth=2)
ax.plot(theta_deg, g_plagiophile,  label='Plagiophile',  linewidth=2)
ax.plot(theta_deg, g_extremophile, label='Extremophile', linewidth=2)
ax.plot(theta_deg, g_spherical,    label='Spherical',    linewidth=2)
ax.plot(theta_deg, g_uniform,      label='Uniform',      linestyle='--', linewidth=1.5)

ax.set_xlabel(r'Leaf inclination angle $\theta_L$ (degrees)', fontsize=12)
ax.set_ylabel(r'$g_L(\theta_L)$', fontsize=12)
ax.set_title('Leaf Normal Inclination Distribution Functions', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 90)
ax.set_ylim(0, None)
plt.tight_layout()

out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure01_gauss_quadrature.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
