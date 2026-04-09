"""
uncollided_intensity.py  —  DOM_v3
===================================
DOM uncollided solver for a plane-parallel leaf canopy.
Follows Steps 1-6 of the DOM Full Guide (C4-P3).
Steps 7-11 (collided field, IC) are handled separately.

DOM Guide §1.2:  I(L,Ω) = I0(L,Ω) + IC(L,Ω).
I0 = uncollided: Beer-Lambert only, no iteration needed.
IC = collided:   requires iterative solver (Steps 7-11).
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 0 — INPUTS  (Guide §15, Code Map §0)
# Configuration matches dom_uncollided.py exactly.
# =============================================================================

# ── Numerical resolution ──────────────────────────────────────────────────────
N = 16   # polar Gauss-Legendre directions  (must be even; N/2 down, N/2 up)
M = 16   # uniform azimuthal directions
K = 50   # canopy layers

# ── Illumination ──────────────────────────────────────────────────────────────
F_in  = 1.0   # total incident flux [W/m²]; 1.0 so outputs are normalised
f_dir = 0.70  # fraction that is direct solar beam (0 = overcast, 1 = full sun)

# ── Solar geometry ────────────────────────────────────────────────────────────
# theta measured from the UPWARD zenith; must be >90 for a downward sun.
# 140 deg from zenith = 40-deg solar elevation.
theta_solar_deg = 140.0
phi_solar_deg   =   0.0   # solar azimuth (0 = reference plane)

# ── Canopy structure ──────────────────────────────────────────────────────────
# Both LAI values are run and plotted together (Guide §15: LAI_VALUES = [1.5, 4.0])
LAI_values = [1.5, 4.0]

# ── Spectral bands ────────────────────────────────────────────────────────────
# RED: leaves absorb most; NIR: leaves nearly transparent.
# omega_L = rho_L + tau_L is the leaf single-scattering albedo.
spectral_bands = {
    'RED': {'rho_L': 0.06,  'tau_L': 0.04,  'rho_g': 0.10},
    'NIR': {'rho_L': 0.525, 'tau_L': 0.45,  'rho_g': 0.20},
}


# =============================================================================
# STEP 1 — ANGULAR DISCRETIZATION  (Guide §4, C4-P3 Slides 7-9)
#
# N Gauss-Legendre nodes on [-1,+1], stored 1-based (index 0 unused).
#   n = 1..N/2   →  mu[n] < 0  (downward)
#   n = N/2+1..N →  mu[n] > 0  (upward)
# M cell-centred azimuth angles:  phi[j] = (j-0.5)*2π/M
# Weight checks (Guide §4.1 / Slide 9):
#   sum(w_mu) = 2.0  and  sum(mu[up]*w_mu[up]) ≈ 0.5
# =============================================================================

half = N // 2

nodes_gl, weights_gl = np.polynomial.legendre.leggauss(N)
mu   = np.zeros(N + 1);  mu[1:N+1]   = nodes_gl    # ascending: negative first
w_mu = np.zeros(N + 1);  w_mu[1:N+1] = weights_gl

assert abs(np.sum(w_mu[1:N+1]) - 2.0) < 1e-8,  "sum(w_mu) must equal 2"
assert abs(np.sum(mu[half+1:N+1] * w_mu[half+1:N+1]) - 0.5) < 1e-6, "upward hemisphere check"

# Azimuthal: cell-centred on [0, 2π)  (Guide §4.2)
w_phi = 2.0 * np.pi / M
phi_az = np.zeros(M + 1)
for j in range(1, M + 1):
    phi_az[j] = (j - 0.5) * w_phi

# Solar direction
solar_mu  = math.cos(math.radians(theta_solar_deg))   # ≈ -0.766 for 140°
solar_phi = math.radians(phi_solar_deg)
abs_mu_sol = abs(solar_mu)

# Extinction coefficient G  (Guide §4.3, C4-P3 Slide 7)
# G = 0.5 for uniform (spherical) leaf-normal distribution — exact for all directions.
# Only this line changes if a non-uniform leaf distribution is used later.
G = 0.5


# =============================================================================
# STEP 1 (cont.) — PHASE FUNCTION  (Guide §4.4)
# Γ(Ω'→Ω): probability a photon from Ω' scatters into dΩ around Ω after a leaf hit.
# Pre-computed once per spectral band (not per LAI, not per iteration).
#   G_sol[i,j]    = Γ(solar → (mu_i, phi_j))       shape (N+1, M+1)
#   G_qq[n,m,i,j] = Γ((mu_n, phi_m) → (mu_i,phi_j)) shape (N+1,M+1,N+1,M+1)
# For N=M=16: 65,536 Γ evaluations — compute once, reuse for all LAI values.
# =============================================================================

def gamma_scalar(inc_mu, inc_phi, scat_mu, scat_phi, omega_L, tau_L):
    """Leaf volume scattering phase function Γ(Ω'→Ω).  Guide §4.4 / C4-P1."""
    sin_i = math.sqrt(max(1.0 - inc_mu**2,  0.0))
    sin_s = math.sqrt(max(1.0 - scat_mu**2, 0.0))
    cos_beta = inc_mu*scat_mu + sin_i*sin_s*math.cos(inc_phi - scat_phi)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    beta = math.acos(cos_beta)
    return ((omega_L / (3.0*math.pi)) * (math.sin(beta) - beta*math.cos(beta))
            + (tau_L  / 3.0) * math.cos(beta))


# =============================================================================
# MAIN LOOP — spectral bands then LAI values
# Steps 1 (phase tables) and 2-6 are inside this structure.
# =============================================================================

for band_label, band in spectral_bands.items():

    rho_L   = band['rho_L']
    tau_L   = band['tau_L']
    rho_g   = band['rho_g']
    omega_L = rho_L + tau_L

    print(f"\n{'='*60}")
    print(f"  Band: {band_label}  (omega_L={omega_L:.3f}, rho_g={rho_g})")
    print(f"{'='*60}")

    # ── Phase function tables — once per band (Guide §4.4) ────────────────────
    print("  Building phase function tables ...")
    G_sol = np.zeros((N+1, M+1))
    for i in range(1, N+1):
        for j in range(1, M+1):
            G_sol[i, j] = gamma_scalar(solar_mu, solar_phi,
                                       mu[i], phi_az[j], omega_L, tau_L)

    # G_qq: pre-compute all (n,m)→(i,j) scattering combinations
    G_qq = np.zeros((N+1, M+1, N+1, M+1))
    for n in range(1, N+1):
        for m in range(1, M+1):
            for i in range(1, N+1):
                for j in range(1, M+1):
                    G_qq[n, m, i, j] = gamma_scalar(mu[n], phi_az[m],
                                                     mu[i], phi_az[j], omega_L, tau_L)

    # Slice out the 1-based subsets used in Steps 6 (avoids repeated index shifting)
    Gqq_dn  = G_qq[1:half+1,    1:M+1, 1:N+1, 1:M+1]  # downward source  (N/2,M,N,M)
    Gqq_up  = G_qq[half+1:N+1,  1:M+1, 1:N+1, 1:M+1]  # upward source    (N/2,M,N,M)
    Gqq_all = G_qq[1:N+1,       1:M+1, 1:N+1, 1:M+1]  # all directions   (N,M,N,M)
    Gsol    = G_sol[1:N+1, 1:M+1]                       # (N,M)

    # Quadrature weight arrays (0-based slices of 1-based w_mu)
    wdn = w_mu[1:half+1]    # (N/2,)  downward weights
    wup = w_mu[half+1:N+1]  # (N/2,)  upward weights
    wall = w_mu[1:N+1]       # (N,)

    all_results = []

    for LAI in LAI_values:

        print(f"\n  LAI = {LAI}")

        # =====================================================================
        # STEP 2 — SPATIAL DISCRETIZATION  (Guide §4.5, C4-P3 Slide 10)
        # K layers, cell edges k=1..K+1.  Grid fully described by G and dL.
        # =====================================================================
        dL = LAI / K

        # =====================================================================
        # STEP 3a — DOWNWARD SWEEP: DIRECT SOLAR  (Guide §5.2, C4-P3 Slide 12)
        # Upper BC: I_beam = f_dir*F_in/|mu_sol|  (irradiance÷|mu| = intensity)
        # Recurrence: I0_dir[k+1] = I0_dir[k] * exp(-G*dL/|mu_sol|)
        # =====================================================================
        I_beam       = f_dir * F_in / abs_mu_sol
        exp_step_sol = math.exp(-G * dL / abs_mu_sol)

        I0_dir = np.zeros(K + 2)    # 1-based; index 0 unused
        I0_dir[1] = I_beam           # upper BC at canopy top
        for k in range(1, K + 1):
            I0_dir[k+1] = I0_dir[k] * exp_step_sol

        # Direct-beam irradiance  Fd_dir[k] = |mu_sol| * I0_dir[k]
        Fd_dir = abs_mu_sol * I0_dir

        # =====================================================================
        # STEP 3b — DOWNWARD SWEEP: DIFFUSE SKY  (Guide §5.3, C4-P3 Slide 13)
        # Isotropic sky radiance: I_sky = (1-f_dir)*F_in/π  (Guide §5.1)
        # Same recurrence for all (N/2)*M downward directions.
        # =====================================================================
        I_sky = (1.0 - f_dir) * F_in / math.pi

        I0 = np.zeros((N+1, M+1, K+2))    # I0[n,m,k], 1-based in all dims

        for n in range(1, half + 1):       # downward: mu[n] < 0
            exp_step_n = math.exp(-G * dL / abs(mu[n]))
            for m in range(1, M + 1):
                I0[n, m, 1] = I_sky         # upper BC
                for k in range(1, K + 1):
                    I0[n, m, k+1] = I0[n, m, k] * exp_step_n

        # Downward diffuse irradiance — Gauss quadrature (Guide §6 / Slide 15)
        Fd_dif = np.zeros(K + 2)
        for k in range(1, K + 2):
            for n in range(1, half + 1):
                for m in range(1, M + 1):
                    Fd_dif[k] += w_mu[n] * w_phi * abs(mu[n]) * I0[n, m, k]

        # =====================================================================
        # STEP 4 — LOWER BC: LAMBERTIAN GROUND  (Guide §6, C4-P3 Slide 14)
        # F_down_gnd MUST include BOTH direct AND diffuse (common error — Guide §6).
        # Lambertian: I0_up_gnd = (rho_g/π) * F_down_gnd  (isotropic upward).
        # =====================================================================
        F_down_gnd = Fd_dif[K+1] + Fd_dir[K+1]    # both terms required
        I0_up_gnd  = (rho_g / math.pi) * F_down_gnd

        for n in range(half + 1, N + 1):   # upward: mu[n] > 0
            for m in range(1, M + 1):
                I0[n, m, K+1] = I0_up_gnd  # lower BC for upward sweep

        # =====================================================================
        # STEP 5 — UPWARD SWEEP  (Guide §7, C4-P3 Slide 16)
        # Lower BC from Step 4; recurrence: I0[n,m,k] = I0[n,m,k+1]*exp(...)
        # =====================================================================
        for n in range(half + 1, N + 1):   # upward: mu[n] > 0
            exp_step_n = math.exp(-G * dL / abs(mu[n]))
            for m in range(1, M + 1):
                for k in range(K, 0, -1):   # sweep upward: K → 1
                    I0[n, m, k] = I0[n, m, k+1] * exp_step_n

        # Upward uncollided irradiance (same Gauss quadrature as Fd_dif)
        Fu_unc = np.zeros(K + 2)
        for k in range(1, K + 2):
            for n in range(half + 1, N + 1):
                for m in range(1, M + 1):
                    Fu_unc[k] += w_mu[n] * w_phi * abs(mu[n]) * I0[n, m, k]

        # =====================================================================
        # STEP 6 — FIRST COLLISION SOURCE  Q = Q1 + Q2 + Q3
        #          (Guide §8, C4-P3 Slides 18-20)
        #
        # Cell-centre intensity = mean of surrounding edges (Guide §8.1):
        #   I0_dir_c[k] = 0.5*(I0_dir[k] + I0_dir[k+1])   k=1..K
        #   I0_c[n,m,k] = 0.5*(I0[n,m,k]  + I0[n,m,k+1])
        #
        # Q1[i,j,k] = (1/π)*G_sol[i,j]*I0_dir_c[k]              §8.2 / Slide 18
        # Q2[i,j,k] = (1/π)*Σ_{n=1..N/2}  w*w_phi*G_qq*I0_c_dn  §8.3 / Slide 19
        # Q3[i,j,k] = (1/π)*Σ_{n=N/2+1..N} w*w_phi*G_qq*I0_c_up §8.4 / Slide 20
        #
        # Source direction ranges: Q2 downward only, Q3 upward only (Guide §8).
        # Output direction i=1..N (both hemispheres) for all Q1, Q2, Q3.
        # Vectorised with numpy einsum — mathematically identical to Guide loops.
        # =====================================================================

        # Cell-centre averages
        I0_dir_c = 0.5 * (I0_dir[1:K+1] + I0_dir[2:K+2])   # shape (K,)
        I0_c     = 0.5 * (I0[1:N+1, 1:M+1, 1:K+1]
                         + I0[1:N+1, 1:M+1, 2:K+2])          # (N,M,K) 0-based

        I0_c_dn  = I0_c[:half,  :, :]    # downward source:  (N/2, M, K)
        I0_c_up  = I0_c[half:,  :, :]    # upward source:    (N/2, M, K)

        # Q1: direct solar scatters into each (i,j) direction
        # Q1[i,j,k] = (1/π) * G_sol[i,j] * I0_dir_c[k]
        Q1 = (1.0/math.pi) * np.einsum('ij,k->ijk', Gsol, I0_dir_c)

        # Q2: downward diffuse sky scatters into (i,j)
        # weighted source (N/2,M,K): w_mu[n]*w_phi * I0_c_dn
        wsrc_dn = wdn[:, np.newaxis, np.newaxis] * w_phi * I0_c_dn
        Q2 = (1.0/math.pi) * np.einsum('nmk,nmij->ijk', wsrc_dn, Gqq_dn)

        # Q3: ground-reflected upward field scatters into (i,j)
        wsrc_up = wup[:, np.newaxis, np.newaxis] * w_phi * I0_c_up
        Q3 = (1.0/math.pi) * np.einsum('nmk,nmij->ijk', wsrc_up, Gqq_up)

        Q = Q1 + Q2 + Q3   # shape (N, M, K); output i=0..N-1 (0-based = 1..N 1-based)

        # ── Results summary ───────────────────────────────────────────────────
        print(f"    Fd_dir : top={Fd_dir[1]:.4f},   ground={Fd_dir[K+1]:.4f}")
        print(f"    Fd_dif : top={Fd_dif[1]:.4f},   ground={Fd_dif[K+1]:.4f}")
        print(f"    Fu_unc : top={Fu_unc[1]:.4f},   ground={Fu_unc[K+1]:.4f}")
        print(f"    I0_up_gnd (ground BC) = {I0_up_gnd:.6f}")

        all_results.append({
            'LAI': LAI, 'K': K,
            'I0_dir': I0_dir, 'I0': I0,
            'Q': Q, 'Q1': Q1, 'Q2': Q2, 'Q3': Q3,
            'Fd_dir': Fd_dir, 'Fd_dif': Fd_dif, 'Fu_unc': Fu_unc,
        })

    # =========================================================================
    # PLOTS — three-panel figure per band (matches dom_uncollided.py structure)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"DOM Uncollided — {band_label} band  "
                 f"(N={N}, M={M}, K={K})", fontsize=13)

    colors = ['tab:blue', 'tab:orange']

    for idx, res in enumerate(all_results):
        LAI_  = res['LAI']
        K_    = res['K']
        L_norm = np.linspace(0, 1, K_+1)   # normalised depth 0→1

        # ── Plot A: Normalised downward flux vs L/LAI ─────────────────────────
        ax = axes[0]
        ax.plot(L_norm, res['Fd_dir'][1:K_+2] / F_in,
                color=colors[idx], ls='-',
                label=f"Direct,  LAI={LAI_}")
        ax.plot(L_norm, res['Fd_dif'][1:K_+2] / F_in,
                color=colors[idx], ls='dotted',
                label=f"Diffuse, LAI={LAI_}")
        ax.set_xlabel("Normalised depth  L / LAI")
        ax.set_ylabel("Normalised flux  F / F_in")
        ax.set_title("Plot A: Uncollided downward flux")
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

        # ── Plot B: Normalised upward flux vs L/LAI ───────────────────────────
        ax = axes[1]
        ax.plot(L_norm, res['Fu_unc'][1:K_+2] / F_in,
                color=colors[idx], ls='-',
                label=f"Upward, LAI={LAI_}")
        ax.set_xlabel("Normalised depth  L / LAI")
        ax.set_ylabel("Normalised flux  F / F_in")
        ax.set_title("Plot B: Uncollided upward flux")
        ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

        # ── Plot C: BRF at canopy top in the principal plane ──────────────────
        # BRF(Ω_view) = I0[n,j,k=1] / (F_in/π)  for upward G-L directions.
        # Principal plane: j closest to phi_solar (forward) and phi_solar+π (back).
        ax = axes[2]
        upward_mu  = mu[half+1:N+1]
        theta_up   = np.degrees(np.arccos(upward_mu))   # view zenith 0→90°
        sort_idx   = np.argsort(theta_up)
        theta_sort = theta_up[sort_idx]

        phi_fwd = solar_phi % (2.0*math.pi)
        phi_bck = (solar_phi + math.pi) % (2.0*math.pi)
        diffs_fwd = np.array([abs(phi_az[j] - phi_fwd) for j in range(1, M+1)])
        diffs_bck = np.array([abs(phi_az[j] - phi_bck) for j in range(1, M+1)])
        j_fwd = int(np.argmin(diffs_fwd)) + 1   # 1-based azimuth index
        j_bck = int(np.argmin(diffs_bck)) + 1

        brf_norm = F_in / math.pi
        brf_fwd  = np.array([res['I0'][half+1+s, j_fwd, 1] / brf_norm
                              for s in range(half)])[sort_idx]
        brf_bck  = np.array([res['I0'][half+1+s, j_bck, 1] / brf_norm
                              for s in range(half)])[sort_idx]

        ax.plot(-theta_sort[::-1], brf_bck[::-1],
                color=colors[idx], ls='-',  label=f"LAI={LAI_}")
        ax.plot( theta_sort,       brf_fwd,
                color=colors[idx], ls='-')
        ax.axvline(0, color='grey', lw=0.8, ls=':')

    axes[2].set_xlabel("View zenith [deg]\n← backscatter  |  forward scatter →")
    axes[2].set_ylabel("Uncollided BRF")
    axes[2].set_title("Plot C: Uncollided BRF at canopy top\n(principal plane)")
    axes[2].legend(fontsize=8);  axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"uncollided_profiles_{band_label}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Figure saved: {fname}")

print("\n" + "="*60)
print("  All bands complete.")
print("="*60)
