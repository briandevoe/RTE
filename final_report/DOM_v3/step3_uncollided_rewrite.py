"""
uncollided.py — Uncollided field and First Collision Source for DOM_v2.

The uncollided field I0 consists of radiation that has NEVER hit a leaf.
Because there is no scattering source, it follows Beer-Lambert exponential
decay and can be solved analytically (no iteration).

Public API
----------
solve_uncollided(mu, phi, w_mu, w_phi, N, M, K,
                 solar_mu, f_dir, F_in, rho_g, G, dL,
                 G_qq, G_sol) -> (I0_dir, I0, Fd_dir, Fd_dif, Fu_unc, Q)
"""

import math
import numpy as np


def solve_uncollided(mu, phi, w_mu, w_phi, N, M, K,
                     solar_mu, f_dir, F_in, rho_g, G, dL,
                     G_qq, G_sol):
    """
    Solve for the uncollided intensity field and First Collision Source.

    Layer indexing (1-based, K+2 edge points):
        k=1      : canopy top    (L=0)
        k=K+1    : canopy bottom (L=LAI)
    Cell centres are at k+1/2 for k=1..K.

    Parameters
    ----------
    mu, phi, w_mu, w_phi : quadrature arrays (1-based)
    N, M, K              : quadrature and layer counts
    solar_mu             : float   solar polar cosine (negative = downward)
    f_dir                : float   direct fraction of F_in
    F_in                 : float   total downwelling irradiance [W m-2]
    rho_g                : float   Lambertian ground reflectance
    G                    : float   extinction coefficient (0.5 for uniform LAD)
    dL                   : float   layer thickness = LAI / K
    G_qq                 : ndarray (N+1,M+1,N+1,M+1)  Gauss-to-Gauss phase table
    G_sol                : ndarray (N+1,M+1)           solar-to-Gauss phase column

    Returns
    -------
    I0_dir : ndarray (K+2,)          direct-beam scalar intensity at edges
    I0     : ndarray (N+1,M+1,K+2)   diffuse uncollided intensity at edges
    Fd_dir : ndarray (K+2,)          direct downward irradiance
    Fd_dif : ndarray (K+2,)          diffuse downward irradiance (uncollided)
    Fu_unc : ndarray (K+2,)          upward irradiance (uncollided, from ground)
    Q      : ndarray (N+1,M+1,K)     First Collision Source at cell centres
    """
    half       = N // 2
    abs_mu_sol = abs(solar_mu)

    # -------------------------------------------------------------------------
    # 1. Direct solar beam: Beer-Lambert sweep downward
    #
    #   I_beam = f_dir * F_in / |mu_solar|   [radiance of the solar pencil]
    #   I0_dir[k] = I_beam * exp(-G * (k-1)*dL / |mu_solar|)
    #   Recurrence: I0_dir[k+1] = I0_dir[k] * exp(-G*dL/|mu_solar|)
    # -------------------------------------------------------------------------
    I_beam    = f_dir * F_in / abs_mu_sol
    exp_sol   = math.exp(-G * dL / abs_mu_sol)
    I0_dir    = np.zeros(K + 2)
    I0_dir[1] = I_beam
    for k in range(1, K + 1):
        I0_dir[k+1] = I0_dir[k] * exp_sol

    Fd_dir = abs_mu_sol * I0_dir   # direct downward irradiance: F = |mu| * I

    # -------------------------------------------------------------------------
    # 2. Diffuse sky: Beer-Lambert sweep downward for each Gauss direction
    #
    #   I_sky = (1 - f_dir) * F_in / pi   [isotropic radiance at canopy top]
    #   For each downward direction n (mu_n < 0):
    #     I0[n,m,k] = I_sky * exp(-G*(k-1)*dL / |mu_n|)
    # -------------------------------------------------------------------------
    I0    = np.zeros((N+1, M+1, K+2))
    I_sky = (1.0 - f_dir) * F_in / math.pi
    for n in range(1, half + 1):
        exp_n = math.exp(-G * dL / abs(mu[n]))
        for m in range(1, M + 1):
            I0[n, m, 1] = I_sky
            for k in range(1, K + 1):
                I0[n, m, k+1] = I0[n, m, k] * exp_n

    # Downward diffuse irradiance: Fd_dif[k] = sum_{n,m} w_n*w_phi*|mu_n|*I0[n,m,k]
    Fd_dif = np.zeros(K + 2)
    for k in range(1, K + 2):
        for n in range(1, half + 1):
            for m in range(1, M + 1):
                Fd_dif[k] += w_mu[n] * w_phi * abs(mu[n]) * I0[n, m, k]

    # -------------------------------------------------------------------------
    # 3. Ground BC: Lambertian reflection of total downward flux at ground
    #
    #   I0_up_gnd = (rho_g / pi) * (Fd_dif[K+1] + Fd_dir[K+1])
    # -------------------------------------------------------------------------
    I0_up_gnd = (rho_g / math.pi) * (Fd_dif[K+1] + Fd_dir[K+1])

    # -------------------------------------------------------------------------
    # 4. Upward uncollided: Beer-Lambert sweep upward from ground
    #
    #   For each upward direction n (mu_n > 0):
    #     I0[n,m,K+1] = I0_up_gnd   (lower BC)
    #     I0[n,m,k]   = I0[n,m,k+1] * exp(-G*dL/|mu_n|)   (sweep upward)
    # -------------------------------------------------------------------------
    for n in range(half + 1, N + 1):
        exp_n = math.exp(-G * dL / abs(mu[n]))
        for m in range(1, M + 1):
            I0[n, m, K+1] = I0_up_gnd
            for k in range(K, 0, -1):
                I0[n, m, k] = I0[n, m, k+1] * exp_n

    # Upward uncollided irradiance
    Fu_unc = np.zeros(K + 2)
    for k in range(1, K + 2):
        for n in range(half + 1, N + 1):
            for m in range(1, M + 1):
                Fu_unc[k] += w_mu[n] * w_phi * abs(mu[n]) * I0[n, m, k]

    # -------------------------------------------------------------------------
    # 5. First Collision Source  Q = Q1 + Q2 + Q3
    #
    #   Cell-centre values: average of surrounding edges.
    #   Q1[i,j,k] = (1/pi) * G_sol[i,j] * I0_dir_c[k]          (solar beam)
    #   Q2[i,j,k] = (1/pi) * sum_{n<half} w_n*w_phi*G_qq*I0_c   (diffuse down)
    #   Q3[i,j,k] = (1/pi) * sum_{n>half} w_n*w_phi*G_qq*I0_c   (diffuse up)
    # -------------------------------------------------------------------------
    I0_dir_c = 0.5 * (I0_dir[1:K+1] + I0_dir[2:K+2])      # (K,)
    I0_c     = 0.5 * (I0[:, :, 1:K+1] + I0[:, :, 2:K+2])  # (N+1,M+1,K)

    Q = np.zeros((N+1, M+1, K))
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            q1 = (1.0 / math.pi) * G_sol[i, j] * I0_dir_c

            q2 = np.zeros(K)
            for n in range(1, half + 1):
                for m in range(1, M + 1):
                    q2 += w_mu[n] * w_phi * G_qq[n, m, i, j] * I0_c[n, m, :]

            q3 = np.zeros(K)
            for n in range(half + 1, N + 1):
                for m in range(1, M + 1):
                    q3 += w_mu[n] * w_phi * G_qq[n, m, i, j] * I0_c[n, m, :]

            Q[i, j, :] = q1 + (q2 + q3) / math.pi

    return I0_dir, I0, Fd_dir, Fd_dif, Fu_unc, Q
