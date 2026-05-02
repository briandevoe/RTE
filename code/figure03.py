"""
figure03.py -- Volume scattering phase function (bi-Lambertian, uniform LAD)

Plots the area scattering phase function Gamma_LD,U vs cosine of scattering
angle for tau/omega ratios from 0.0 to 0.5.

Run from the repo root:
    python code/figure03.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

cos_beta  = np.linspace(-1, 1, 500)
beta      = np.arccos(cos_beta)
omega_LD  = 1.0
tau_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
colors     = ['tab:red', 'tab:orange', 'gold', 'skyblue', 'steelblue', 'navy']

fig, ax = plt.subplots(figsize=(8, 6))

for tau_ratio, color in zip(tau_ratios, colors):
    tau_LD = tau_ratio * omega_LD
    Gamma  = (omega_LD * (np.sin(beta) - beta * cos_beta) / (3 * np.pi)
              + tau_LD * cos_beta / 3)
    ax.plot(cos_beta, Gamma, color=color, linewidth=2,
            label=rf'$\tau_{{LD}}/\omega_{{LD}} = {tau_ratio}$')

ax.set_xlabel('Cosine of Scattering Angle', fontsize=12)
ax.set_ylabel('Area Scattering Phase Function', fontsize=12)
ax.set_title(
    r'$\Gamma_{LD,U}$ for Uniform Leaf Inclination and Azimuthal Distribution'
    '\n'
    r'($\bar{g}_L = 1$,  $h_L = 1$,  $\omega_{LD} = 1$)',
    fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(-1, 1)
ax.set_ylim(0, None)
ax.grid(alpha=0.3)
plt.tight_layout()

out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'figure03_phase_function.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.show()
