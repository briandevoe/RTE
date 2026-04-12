"""
collided.py — Iterative collided-field solver for DOM_v2.

Uses Diamond-Difference (or Step) sweeps with Source Iteration until the
scalar irradiance profile converges.

Public API
----------
sweep_coeffs(mu_n, G, dL, alpha) -> (a, b, c, d)
    Pre-compute DD sweep coefficients for one polar direction.

solve_collided(mu, phi, w_mu, w_phi, N, M, K,
               G, dL, rho_g, Q, G_qq, omega_L, tol, max_iter, alpha)
    -> (IC, Fd_col, Fu_col)
"""

import math
import numpy as np


def sweep_coeffs(mu_n, G, dL, alpha=0.5):
    """
    Compute Diamond-Difference sweep coefficients for polar direction mu_n.

    Uses the SIGNED cosine (not |mu_n|) so that:
      - Downward (mu_n < 0): f < 0, a < 1, source term -b*J > 0 (adds to IC)
      - Upward   (mu_n > 0): f > 0, c < 1, source term +d*J > 0 (adds to IC)

    Downward sweep:  IC[k+1] = a * IC[k]  -  b * J[k+1/2]
    Upward   sweep:  IC[k]   = c * IC[k+1] + d * J[k+1/2]

    Parameters
    ----------
    mu_n  : float   signed polar cosine (negative=downward, positive=upward)
    G     : float   extinction coefficient
    dL    : float   layer thickness
    alpha : float   weighting parameter: 0.5 = Diamond Diff, 1.0 = Step

    Returns
    -------
    a, b : coefficients for downward sweep
    c, d : coefficients for upward sweep
    """
    f = G * dL / mu_n                            # signed optical half-thickness
    a = (1.0 + (1.0 - alpha) * f) / (1.0 - alpha * f)
    b = f / (G * (1.0 - alpha * f))
    c = (1.0 - (1.0 - alpha) * f) / (1.0 + alpha * f)
    d = f / (G * (1.0 + alpha * f))
    return a, b, c, d


def solve_collided(mu, phi, w_mu, w_phi, N, M, K,
                   G, dL, rho_g, Q, G_qq, omega_L,
                   tol=0.01, max_iter=500, alpha=0.5):
    """
    Solve the collided intensity field by Source Iteration.

    Algorithm
    ---------
    1.  Start: IC = 0, S = 0 (multiple-scatter source)
    2.  Compute total source J = Q + S
    3.  Sweep all downward directions (upper BC: IC=0 at top)
    4.  Compute ground BC from this sweep's downward IC at k=K+1
    5.  Sweep all upward directions (lower BC from step 4)
    6.  Compute new S from converged IC
    7.  Check convergence on scalar irradiance profile
    8.  Repeat until converged

    Parameters
    ----------
    mu, phi, w_mu, w_phi : 1-based quadrature arrays
    N, M, K              : quadrature and layer counts
    G                    : float   extinction coefficient
    dL                   : float   layer thickness
    rho_g                : float   Lambertian ground reflectance
    Q                    : ndarray (N+1,M+1,K)   First Collision Source
    G_qq                 : ndarray (N+1,M+1,N+1,M+1)  Gauss-to-Gauss phase
    omega_L              : float   single-scattering albedo
    tol                  : float   convergence tolerance
    max_iter             : int     maximum iterations
    alpha                : float   DD parameter (0.5=DD, 1.0=Step)

    Returns
    -------
    IC     : ndarray (N+1,M+1,K+2)   converged collided intensity at edges
    Fd_col : ndarray (K+2,)          downward collided irradiance
    Fu_col : ndarray (K+2,)          upward collided irradiance
    """
    half = N // 2

    # Pre-compute sweep coefficients for every direction
    a_c = np.zeros(N + 1)
    b_c = np.zeros(N + 1)
    c_c = np.zeros(N + 1)
    d_c = np.zeros(N + 1)
    for n in range(1, N + 1):
        a_c[n], b_c[n], c_c[n], d_c[n] = sweep_coeffs(mu[n], G, dL, alpha)

    IC      = np.zeros((N+1, M+1, K+2))
    S       = np.zeros((N+1, M+1, K))
    SI_prev = np.zeros(K)

    for it in range(1, max_iter + 1):
        J      = Q + S
        IC_new = np.zeros((N+1, M+1, K+2))

        # -- Downward sweep (mu < 0): upper BC = 0 (no collided from above) --
        for n in range(1, half + 1):
            for m in range(1, M + 1):
                for k in range(1, K + 1):
                    val = a_c[n] * IC_new[n, m, k] - b_c[n] * J[n, m, k-1]
                    IC_new[n, m, k+1] = max(val, 0.0)   # fix-up: clip negatives

        # -- Ground BC: Lambertian reflection of THIS sweep's downward flux --
        F_gnd = 0.0
        for n in range(1, half + 1):
            for m in range(1, M + 1):
                F_gnd += w_mu[n] * w_phi * abs(mu[n]) * IC_new[n, m, K+1]
        I_gnd = (rho_g / math.pi) * F_gnd

        # -- Upward sweep (mu > 0): lower BC from ground reflection --
        for n in range(half + 1, N + 1):
            for m in range(1, M + 1):
                IC_new[n, m, K+1] = I_gnd
                for k in range(K, 0, -1):
                    val = c_c[n] * IC_new[n, m, k+1] + d_c[n] * J[n, m, k-1]
                    IC_new[n, m, k] = max(val, 0.0)

        IC   = IC_new
        IC_c = 0.5 * (IC[:, :, 1:K+1] + IC[:, :, 2:K+2])   # cell centres

        # -- New multiple-scattering source --
        S_new = np.zeros((N+1, M+1, K))
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                s = np.zeros(K)
                for n in range(1, N + 1):
                    for m in range(1, M + 1):
                        s += w_mu[n] * w_phi * G_qq[n, m, i, j] * IC_c[n, m, :]
                S_new[i, j, :] = s / math.pi

        # -- Convergence: max relative change in scalar irradiance profile --
        SI_new = np.zeros(K)
        for n in range(1, N + 1):
            for m in range(1, M + 1):
                SI_new += w_mu[n] * w_phi * IC_c[n, m, :]   # no |mu| weight

        if it > 1:
            crit = float(np.max(
                np.abs(SI_new - SI_prev) / np.maximum(np.abs(SI_new), 1e-30)
            ))
        else:
            crit = 1.0   # no previous value on first iteration

        SI_prev = SI_new.copy()
        S       = S_new
        print(f"  iter {it:3d}  S={crit:.2e}")

        if it > 1 and crit < tol:
            print(f"  Converged in {it} iterations.\n")
            break
    else:
        print(f"  WARNING: did not converge in {max_iter} iterations.\n")

    # -- Irradiance profiles from converged IC --
    Fd_col = np.zeros(K + 2)
    Fu_col = np.zeros(K + 2)
    for k in range(1, K + 2):
        for n in range(1, half + 1):
            for m in range(1, M + 1):
                Fd_col[k] += w_mu[n] * w_phi * abs(mu[n]) * IC[n, m, k]
        for n in range(half + 1, N + 1):
            for m in range(1, M + 1):
                Fu_col[k] += w_mu[n] * w_phi * abs(mu[n]) * IC[n, m, k]

    return IC, Fd_col, Fu_col
