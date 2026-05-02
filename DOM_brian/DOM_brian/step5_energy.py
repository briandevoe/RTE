

import math
import numpy as np


def scalar_irradiance(I0_dir, I0, IC, mu, w_mu, w_phi, N, M, K):
    """
    Compute scalar irradiance SI at cell centres k=1..K.

    Scalar irradiance = integral of intensity over ALL directions with
    NO cosine (|mu|) weighting. It measures energy DENSITY, not flux.

        SI[k] = I0_dir_c[k]
              + sum_{n,m} w_n * w_phi * (I0_c[n,m,k] + IC_c[n,m,k])

    The direct beam (I0_dir) is a delta function in angle, so its
    directional integral is just the scalar intensity itself.

    Parameters
    ----------
    I0_dir : ndarray (K+2,)
    I0     : ndarray (N+1,M+1,K+2)
    IC     : ndarray (N+1,M+1,K+2)
    mu, w_mu, w_phi, N, M, K : quadrature + grid

    Returns
    -------
    SI : ndarray (K,)   scalar irradiance at cell centres k=1..K
    """
    I0_dir_c = 0.5 * (I0_dir[1:K+1] + I0_dir[2:K+2])
    I0_c     = 0.5 * (I0[:, :, 1:K+1] + I0[:, :, 2:K+2])
    IC_c     = 0.5 * (IC[:, :, 1:K+1] + IC[:, :, 2:K+2])

    SI = I0_dir_c.copy()
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            SI += w_mu[n] * w_phi * (I0_c[n, m, :] + IC_c[n, m, :])
    return SI


def energy_balance(I0_dir, I0, IC,
                   Fd_dir, Fd_dif, Fu_unc, Fd_col, Fu_col,
                   mu, w_mu, w_phi, N, M, K,
                   G, dL, omega_L, rho_g, F_in):
    """
    Compute total irradiances, leaf absorption, and energy balance.

    Energy balance identity:
        F_in = F_ref + A_leaves + (1 - rho_g) * F_trans

    where:
        F_ref   = upward flux at canopy top    (= Fu_tot[1])
        A_leaves= G*(1-omega_L)*dL * sum_k SI_k
        F_trans = total downward flux at ground (= Fd_tot[K+1])
        (1-rho_g)*F_trans = ground-absorbed flux

 

    Returns
    -------
    result : dict with keys:
        Fd_tot, Fu_tot   : ndarray (K+2,)  total downward/upward irradiance
        SI               : ndarray (K,)    scalar irradiance at cell centres
        F_ref            : float   reflected flux at canopy top
        F_trans          : float   total downward flux at ground
        A_leaves         : float   leaf absorption
        F_abs_g          : float   ground absorption
        error_W          : float   absolute energy balance error
        error_pct        : float   relative energy balance error (%)
    """
    Fd_tot = Fd_dir + Fd_dif + Fd_col
    Fu_tot = Fu_unc + Fu_col

    SI       = scalar_irradiance(I0_dir, I0, IC, mu, w_mu, w_phi, N, M, K)
    A_leaves = G * (1.0 - omega_L) * dL * float(np.sum(SI))

    F_ref   = float(Fu_tot[1])       # upward at canopy top
    F_trans = float(Fd_tot[K+1])     # downward at ground
    F_abs_g = (1.0 - rho_g) * F_trans

    error_W   = F_in - (F_ref + A_leaves + F_abs_g)
    error_pct = 100.0 * error_W / F_in

    print("\n=== ENERGY BALANCE ===")
    print(f"  Incoming irradiance      : {F_in:.6f}")
    print(f"  Reflected (DHR/BHR)      : {F_ref:.6f}   ({100*F_ref/F_in:.2f} %)")
    print(f"  Canopy absorption        : {A_leaves:.6f}   ({100*A_leaves/F_in:.2f} %)")
    print(f"  Transmitted to ground    : {F_trans:.6f}   ({100*F_trans/F_in:.2f} %)")
    print(f"  Ground absorption        : {F_abs_g:.6f}   ({100*F_abs_g/F_in:.2f} %)")
    print(f"  Energy balance error     : {error_W:.6f}   ({error_pct:.2f} %)")

    return dict(
        Fd_tot=Fd_tot, Fu_tot=Fu_tot, SI=SI,
        F_ref=F_ref, F_trans=F_trans, A_leaves=A_leaves,
        F_abs_g=F_abs_g, error_W=error_W, error_pct=error_pct,
    )
