"""
phase.py — Bi-Lambertian scattering phase function and precomputed tables.

Public API
----------
gamma_scalar(mu_i, phi_i, mu_j, phi_j, omega_L, tau_L) -> float
    Single evaluation of Gamma(i -> j).

precompute_G_qq(mu, phi, N, M, omega_L, tau_L) -> ndarray (N+1,M+1,N+1,M+1)
    Full Gauss-to-Gauss phase table.

precompute_G_sol(solar_mu, solar_phi, mu, phi, N, M, omega_L, tau_L)
    -> ndarray (N+1, M+1)
    Solar-to-Gauss phase column.
"""

import math
import numpy as np


def gamma_scalar(mu_i, phi_i, mu_j, phi_j, omega_L, tau_L):
    """
    Bi-Lambertian phase function Gamma(i -> j)  [sr^-1].

    The scattering angle beta between incident direction (mu_i, phi_i) and
    scattered direction (mu_j, phi_j) satisfies:

        cos(beta) = mu_i*mu_j
                  + sqrt(1-mu_i^2)*sqrt(1-mu_j^2)*cos(phi_i - phi_j)

    The phase function:

        Gamma = (omega_L / (3*pi)) * (sin(beta) - beta*cos(beta))
              + (tau_L  / 3)       * cos(beta)

    The first term is diffuse (Lambertian) reflection from both faces;
    the second is forward transmission through the leaf.

    Parameters
    ----------
    mu_i, phi_i  : float   incident direction (polar cosine, azimuth [rad])
    mu_j, phi_j  : float   scattered direction
    omega_L      : float   single-scattering albedo = rho_L + tau_L
    tau_L        : float   leaf transmittance

    Returns
    -------
    float : Gamma value [sr^-1]
    """
    sin_i    = math.sqrt(max(0.0, 1.0 - mu_i**2))
    sin_j    = math.sqrt(max(0.0, 1.0 - mu_j**2))
    cos_beta = max(-1.0, min(1.0,
                   mu_i * mu_j + sin_i * sin_j * math.cos(phi_i - phi_j)))
    beta     = math.acos(cos_beta)
    return ((omega_L / (3.0 * math.pi)) * (math.sin(beta) - beta * cos_beta)
            + (tau_L / 3.0) * cos_beta)


def precompute_G_qq(mu, phi, N, M, omega_L, tau_L):
    """
    Pre-compute the full Gauss-to-Gauss phase table.

        G_qq[n, m, i, j] = Gamma( (mu_n, phi_m) -> (mu_i, phi_j) )

    This is the most expensive part of setup (N*M)^2 evaluations.
    Called once per spectral band and reused across all iterations.

    Parameters
    ----------
    mu, phi      : 1-based quadrature arrays
    N, M         : int   quadrature sizes
    omega_L, tau_L : float

    Returns
    -------
    G_qq : ndarray (N+1, M+1, N+1, M+1)
    """
    print(f"  Precomputing G_qq (N={N}, M={M}: {N*M*N*M} evaluations) ...")
    G_qq = np.zeros((N+1, M+1, N+1, M+1))
    for n in range(1, N+1):
        for m in range(1, M+1):
            for i in range(1, N+1):
                for j in range(1, M+1):
                    G_qq[n, m, i, j] = gamma_scalar(
                        mu[n], phi[m], mu[i], phi[j], omega_L, tau_L)
    return G_qq


def precompute_G_sol(solar_mu, solar_phi, mu, phi, N, M, omega_L, tau_L):
    """
    Pre-compute the solar-to-Gauss phase column.

        G_sol[i, j] = Gamma( (solar_mu, solar_phi) -> (mu_i, phi_j) )

    Parameters
    ----------
    solar_mu, solar_phi : float   solar direction
    mu, phi, N, M       : quadrature
    omega_L, tau_L      : float

    Returns
    -------
    G_sol : ndarray (N+1, M+1)
    """
    G_sol = np.zeros((N+1, M+1))
    for i in range(1, N+1):
        for j in range(1, M+1):
            G_sol[i, j] = gamma_scalar(
                solar_mu, solar_phi, mu[i], phi[j], omega_L, tau_L)
    return G_sol
