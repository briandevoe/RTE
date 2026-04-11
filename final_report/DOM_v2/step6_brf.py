"""
brf.py — DOM interpolation to arbitrary view directions for DOM_v2.

Computes BRF or HDRF at any (view_zenith, view_azimuth) by sweeping the
total source upward along the view ray using the Diamond-Difference scheme.

Public API
----------
brf_at_view(view_zenith_deg, view_azimuth_deg,
            I0, IC, I0_dir, mu, phi, w_mu, w_phi,
            N, M, K, G, dL, solar_mu, G_sol, omega_L, tau_L,
            rho_g, F_in, alpha) -> float
"""

import math
import numpy as np
from step2_phase import gamma_scalar


def brf_at_view(view_zenith_deg, view_azimuth_deg,
                I0, IC, I0_dir, mu, phi, w_mu, w_phi,
                N, M, K, G, dL, solar_mu, G_sol, omega_L, tau_L,
                rho_g, F_in, alpha=0.5):
    """
    Compute BRF (or HDRF) at a single upward view direction.

    Method: Build the total source J_v for the view ray, then perform a
    Diamond-Difference upward sweep from the ground BC to canopy top.

        BRF = pi * I_v(canopy top) / (cos(view_zenith) * F_in)

    Parameters
    ----------
    view_zenith_deg  : float   view zenith from nadir (0=nadir, 90=horizon)
    view_azimuth_deg : float   view azimuth (degrees)
    I0               : ndarray (N+1,M+1,K+2)  uncollided intensity at edges
    IC               : ndarray (N+1,M+1,K+2)  collided intensity at edges
    I0_dir           : ndarray (K+2,)          direct-beam intensity at edges
    mu, phi, w_mu, w_phi : 1-based quadrature arrays
    N, M, K          : quadrature and layer counts
    G                : float   extinction coefficient
    dL               : float   layer thickness
    solar_mu         : float   solar polar cosine (negative)
    G_sol            : ndarray (N+1,M+1)  solar-to-Gauss phase (for Q1 only)
    omega_L, tau_L   : float   spectral leaf properties
    rho_g            : float   ground reflectance
    F_in             : float   total incoming irradiance
    alpha            : float   DD parameter (0.5 = DD, 1.0 = Step)

    Returns
    -------
    float : BRF or HDRF (dimensionless)
    """
    mu_v  =  math.cos(math.radians(view_zenith_deg))    # upward: mu > 0
    phi_v =  math.radians(view_azimuth_deg)

    if mu_v <= 0:
        return 0.0   # below-horizon view not supported

    I0_dir_c = 0.5 * (I0_dir[1:K+1] + I0_dir[2:K+2])
    I0_c     = 0.5 * (I0[:, :, 1:K+1] + I0[:, :, 2:K+2])
    IC_c     = 0.5 * (IC[:, :, 1:K+1] + IC[:, :, 2:K+2])

    # Source for the view ray: Q1 (solar) + scattering from all Gauss directions
    q1_v = ((1.0 / math.pi) *
            gamma_scalar(solar_mu, 0.0, mu_v, phi_v, omega_L, tau_L) *
            I0_dir_c)

    q2_v = np.zeros(K)
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            q2_v += (w_mu[n] * w_phi / math.pi *
                     gamma_scalar(mu[n], phi[m], mu_v, phi_v, omega_L, tau_L) *
                     (I0_c[n, m, :] + IC_c[n, m, :]))

    J_v = q1_v + q2_v

    # Ground BC: Lambertian reflection of total downward flux at ground
    Fd_all = 0.0
    for n in range(1, N // 2 + 1):
        for m in range(1, M + 1):
            Fd_all += w_mu[n] * w_phi * abs(mu[n]) * (I0[n, m, K+1] + IC[n, m, K+1])
    Fd_all += abs(solar_mu) * I0_dir[K+1]
    I_v    = np.zeros(K + 2)
    I_v[K+1] = (rho_g / math.pi) * Fd_all

    # Diamond-Difference upward sweep along view direction
    f_v = G * dL / mu_v                                 # positive (upward)
    c_v = (1.0 - (1.0 - alpha) * f_v) / (1.0 + alpha * f_v)
    d_v = f_v / (G * (1.0 + alpha * f_v))
    for k in range(K, 0, -1):
        I_v[k] = max(c_v * I_v[k+1] + d_v * J_v[k-1], 0.0)

    brf = math.pi * I_v[1] / (mu_v * F_in)
    return brf
