"""
quadrature.py — Pluggable polar and azimuthal quadrature for DOM_v2.

Public API
----------
get_polar_quadrature(N, method) -> (mu, w_mu)
    mu    : ndarray (N+1,)  polar cosines, 1-based (index 0 unused)
    w_mu  : ndarray (N+1,)  corresponding weights

get_azimuth_quadrature(M) -> (phi, w_phi)
    phi   : ndarray (M+1,)  azimuth angles in radians, 1-based
    w_phi : float           uniform azimuthal weight = 2*pi/M
"""

import numpy as np


def get_polar_quadrature(N, method="gauss_legendre"):
    """
    Build polar quadrature on the full sphere mu in (-1, 1).

    Nodes are sorted ascending (most-downward first):
        index 1 .. N//2   : mu < 0  (downward directions)
        index N//2+1 .. N : mu > 0  (upward directions)

    Parameters
    ----------
    N      : int   number of polar directions (must be even)
    method : str   one of "gauss_legendre", "double_gauss",
                   "gauss_lobatto", "uniform"

    Returns
    -------
    mu   : ndarray (N+1,)
    w_mu : ndarray (N+1,)
    """
    assert N % 2 == 0, "N must be even"

    mu   = np.zeros(N + 1)
    w_mu = np.zeros(N + 1)
    half = N // 2

    if method == "gauss_legendre":
        # Nodes = zeros of P_N; computed via Golub-Welsch (NumPy built-in).
        # Exact for polynomials up to degree 2N-1.
        nodes, weights = np.polynomial.legendre.leggauss(N)
        mu[1:N+1]   = nodes
        w_mu[1:N+1] = weights

    elif method == "double_gauss":
        # Apply separate Gauss-Legendre on each hemisphere [0,1] and [-1,0].
        # Better hemisphere flux integrals for the same N.
        # Used in DISORT and many production RT codes.
        nodes_h, weights_h = np.polynomial.legendre.leggauss(half)
        # Map from (-1,1) to (0,1): x_mapped = (x+1)/2, w_mapped = w/2
        nodes_up   = 0.5 * (nodes_h + 1.0)   # in (0,1), ascending
        weights_up = 0.5 * weights_h
        # Downward hemisphere: mirror of upward, reversed so overall array is ascending
        mu[1:half+1]      = -nodes_up[::-1]   # negative, most-negative first
        w_mu[1:half+1]    =  weights_up[::-1]
        mu[half+1:N+1]    =  nodes_up          # positive, ascending
        w_mu[half+1:N+1]  =  weights_up

    elif method == "gauss_lobatto":
        # Includes endpoints mu = ±1 (nadir + zenith).
        # Interior: Gauss-Legendre of order N-2.
        # Sacrifices one order of accuracy vs. standard G-L.
        if N < 2:
            raise ValueError("Gauss-Lobatto requires N >= 2")
        end_w = 2.0 / (N * (N - 1))
        if N == 2:
            interior_nodes   = np.array([])
            interior_weights = np.array([])
        else:
            interior_nodes, interior_weights = np.polynomial.legendre.leggauss(N - 2)
        all_nodes   = np.concatenate([[-1.0], interior_nodes, [1.0]])
        all_weights = np.concatenate([[end_w], interior_weights, [end_w]])
        mu[1:N+1]   = all_nodes
        w_mu[1:N+1] = all_weights

    elif method == "uniform":
        # Evenly spaced mu with equal weights (midpoint/rectangle rule).
        # Only 1st-order accurate. Use for debugging or comparison only.
        nodes   = np.linspace(-1.0 + 1.0/N, 1.0 - 1.0/N, N)
        weights = np.full(N, 2.0 / N)
        mu[1:N+1]   = nodes
        w_mu[1:N+1] = weights

    else:
        raise ValueError(
            f"Unknown quadrature method '{method}'. "
            "Choose from: gauss_legendre, double_gauss, gauss_lobatto, uniform"
        )

    # Sanity: weights must sum to 2 (integral of 1 over [-1,1])
    weight_sum = float(np.sum(w_mu[1:N+1]))
    assert abs(weight_sum - 2.0) < 1e-8, (
        f"Polar weights sum to {weight_sum:.6f}, expected 2.0"
    )

    return mu, w_mu


def get_azimuth_quadrature(M):
    """
    Build uniform azimuthal quadrature on [0, 2*pi).

    The rectangle rule converges exponentially for smooth periodic integrands.

    Parameters
    ----------
    M : int   number of azimuthal directions

    Returns
    -------
    phi   : ndarray (M+1,)  azimuth angles [rad], index 1..M valid
    w_phi : float           uniform weight = 2*pi / M
    """
    phi      = np.zeros(M + 1)
    phi[1:]  = (np.arange(M) / M) * 2.0 * np.pi
    w_phi    = 2.0 * np.pi / M
    return phi, w_phi
