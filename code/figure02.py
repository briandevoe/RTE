"""
figure02.py -- G-function for various leaf angle distributions

Plots G(theta) vs beam zenith angle for Planophile, Erectophile,
Plagiophile, Extremophile, and Uniform LADs.

Run from the repo root:
    python code/figure02.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

theta_deg = np.linspace(0, 90, 500)
theta = np.deg2rad(theta_deg)

G_uniform      = 0.5 * np.ones_like(theta)
G_planophile   = 3*(1 + np.cos(theta)**2) / 8
G_erectophile  = 3*(2 + np.sin(theta)**2) / 16
G_plagiophile  = 5*(3 + np.cos(theta)**4) / 32
G_extremophile = 5*(3 - np.cos(theta)**4) / 28

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(theta_deg, G_planophile,   label='Planophile',   linewidth=2)
ax.plot(theta_deg, G_erectophile,  label='Erectophile',  linewidth=2)
ax.plot(theta_deg, G_plagiophile,  label='Plagiophile',  linewidth=2)
ax.plot(theta_deg, G_extremophile, label='Extremophile', linewidth=2)
ax.plot(theta_deg, G_uniform,      label='Uniform',      linestyle='--', linewidth=1.5)

ax.axhline(0.5, linestyle=':', color='gray', linewidth=1, label='G = 0.5')

ax.set_xlabel(r'Beam zenith angle $\theta$ (degrees)', fontsize=12)
ax.set_ylabel(r'$G(\theta)$', fontsize=12)
ax.set_title('Analytical Forms of the Canopy G-Function', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 90)
plt.tight_layout()

out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure02_LAD_verification.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
