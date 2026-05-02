"""
run_all_figures.py -- Generate all figures and compile the final report PDF

Runs every figure script headlessly (no popup windows), then assembles all
outputs into a single PDF with captions.

Usage (from repo root):
    python code/run_all_figures.py

Output:
    code/figures/EE645_report_figures.pdf

Notes:
  - Figures 01-03 are Jupyter notebooks (figure01-03.ipynb). Export them
    manually as PNGs named figure01_gauss_quadrature.png, etc. and place
    in code/figures/ to include them in the PDF.
  - Total runtime: ~10-15 minutes (DOM solver runs multiple times).
"""

import os, sys, subprocess, textwrap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

CODE_DIR    = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(CODE_DIR, 'figures')
OUTPUT_PDF  = os.path.join(FIGURES_DIR, 'EE645_report_figures.pdf')

# ---------------------------------------------------------------------------
# Scripts to run (in order)
# ---------------------------------------------------------------------------
SCRIPTS = [
    'energy_balance_table.py',
    'figure01.py',
    'figure02.py',
    'figure03.py',
    'figure04.py',
    'figure05.py',
    'figure06.py',
    'figure07.py',
    'figure08.py',
    'figure09.py',
    'figure10.py',
    'figure11.py',
]

# ---------------------------------------------------------------------------
# PDF manifest — one entry per output, in report order
# ---------------------------------------------------------------------------
MANIFEST = [
    dict(
        file='energy_balance_table.png',
        title='Energy Balance Table',
        caption=(
            'Canopy energy budget for RED and NIR bands at LAI=1.5 and 4.0, '
            'with fdir=1.0 (pure direct beam), SZA=40°. '
            'Reflectance equals the Directional-Hemispherical Reflectance (DHR) '
            'since the illumination is purely collimated. '
            'RED band: leaves absorb most incoming radiation (62-91%), yielding '
            'very low DHR (2-3%). NIR band: leaves scatter strongly, giving high '
            'DHR (32-42%) and lower canopy absorption (13-31%). '
            'Energy imbalance < 1% for all cases confirms solver convergence. '
            'Spectral parameters — RED: rhoL=0.06, tauL=0.04, rhoG=0.10; '
            'NIR: rhoL=0.45, tauL=0.45, rhoG=0.15.'
        ),
    ),
    dict(
        file='figure01_gauss_quadrature.png',
        title='Figure 1 — Leaf Angle Distributions',
        caption=(
            'The six leaf inclination angle distribution functions g(thetaL) '
            'plotted vs. thetaL from 0 to 90 degrees: uniform, planophile, '
            'erectophile, plagiophile, extremophile, and spherical. '
            'Each distribution describes the statistical orientation of leaves. '
            'Planophile canopies have mostly horizontal leaves; erectophile have '
            'mostly vertical. The uniform (spherical) distribution gives G=0.5 '
            'for all solar/view directions and is used throughout this report.'
        ),
    ),
    dict(
        file='figure02_LAD_verification.png',
        title='Figure 2 — G-Function for Various Leaf Angle Distributions',
        caption=(
            'The Ross-Nilson G-function G(theta) vs. solar/view zenith angle '
            'for five LADs (uniform, planophile, erectophile, plagiophile, '
            'extremophile). G(theta) is the mean projected leaf area per unit '
            'leaf area in direction theta. All distributions satisfy the identity '
            '(1/2pi) * integral G dOmega = 0.5. The uniform LAD gives G=0.5 '
            'for all angles — this is the assumption used in DOM_v3.'
        ),
    ),
    dict(
        file='figure03_phase_function.png',
        title='Figure 3 — Volume Scattering Phase Function',
        caption=(
            'The bi-Lambertian area scattering phase function Gamma(cos_beta) '
            'for uniform LAD, shown for several tau/omega ratios (0.0 to 0.5). '
            'The phase function governs how a leaf scatters incident radiation: '
            'the tau_L term drives forward scattering (transmission) and the '
            'rho_L term drives backward scattering (reflection). '
            'The function is smooth with no singularity at beta=180 deg '
            '(retro-illumination); the hot-spot effect (Ch4 Sec.9) is not '
            'modelled in DOM_v3.'
        ),
    ),
    dict(
        file='figure04_uncollided_RED.png',
        title='Figure 4 — Uncollided Radiation: RED Band',
        caption=(
            'Three-panel uncollided DOM solution for the RED band '
            '(rhoL=0.06, tauL=0.04, rhoG=0.10). '
            'Plot A: normalised downward uncollided flux (direct solar + diffuse '
            'sky) vs. normalised canopy depth L/LAI — Beer-Lambert exponential '
            'decay with a steeper slope for larger LAI. '
            'Plot B: normalised upward ground-reflected uncollided flux — low '
            'values because strong leaf absorption limits what reaches the ground. '
            'Plot C: uncollided BRF at canopy top in the principal plane. '
            'Parameters: SZA=40 deg, fdir=0.70, LAI=1.5 and 4.0, N=M=16, K=50.'
        ),
    ),
    dict(
        file='figure05_uncollided_NIR.png',
        title='Figure 5 — Uncollided Radiation: NIR Band',
        caption=(
            'Same three-panel uncollided DOM solution for the NIR band '
            '(rhoL=0.525, tauL=0.45, rhoG=0.20). '
            'NIR leaves are nearly transparent (omegaL~0.975), so substantially '
            'more radiation penetrates to the ground and the upward flux is much '
            'higher than in the RED band. Plot C shows the uncollided BRF is '
            'also significantly larger in NIR due to the high leaf single-'
            'scattering albedo. Parameters identical to Figure 4.'
        ),
    ),
    dict(
        file='figure06_HDRF_polar_map.png',
        title='Figure 6 — HDRF Hemispherical Polar Map (fdir=0.70)',
        caption=(
            '2x2 hemispherical polar maps of HDRF for RED and NIR bands at '
            'LAI=1.5 and 4.0. Polar orientation: 0 deg at top (forward scatter '
            'direction), 180 deg at bottom (sun sky position). '
            'Yellow star marks the sun at SZA=40 deg. '
            'With mixed illumination (fdir=0.70), the HDRF pattern is smoother '
            'than pure BRF — the diffuse component fills in angular variations. '
            'NIR HDRF is 10-20x higher than RED due to high leaf albedo (0.90 '
            'vs 0.10). Increasing LAI from 1.5 to 4.0 reduces the soil '
            'contribution and flattens the angular pattern. '
            'Parameters: SZA=40 deg, phi_solar=0 deg, N=M=16, K=50.'
        ),
    ),
    dict(
        file='figure07_HDRF_principal_plane.png',
        title='Figure 7 — HDRF Principal Plane Cross-Section (fdir=0.70)',
        caption=(
            'HDRF vs. view zenith angle in the principal plane for RED (left) '
            'and NIR (right) bands at LAI=1.5 and 4.0. '
            'Negative VZA = backscatter toward sun; positive VZA = forward '
            'scatter away from sun. Orange dashed line marks the solar retro-'
            'illumination direction at VZA=-40 deg. '
            'The smooth bowl shape (minimum near nadir) is a consequence of the '
            'bi-Lambertian phase function used in DOM_v3, which has no hot-spot '
            'singularity at the retro-illumination direction (Ch4 Sec.9 not '
            'implemented). Parameters: SZA=40 deg, fdir=0.70, N=M=16, K=50.'
        ),
    ),
    dict(
        file='figure08_BHR_vs_LAI_soil.png',
        title='Figure 8 — BHR as a Function of LAI and Soil Brightness',
        caption=(
            'Bihemispherical reflectance (BHR = upward flux at canopy top / Fin) '
            'vs. LAI for dark (rhoG=0.01) and bright (rhoG=0.30) soil. '
            'RED band: BHR decreases monotonically with LAI as absorbing leaves '
            'progressively mask the bright soil; both soil cases converge to the '
            'same leaf-limited value at high LAI. '
            'NIR band: BHR increases with LAI for dark soil (leaf reflectance '
            'exceeds soil reflectance) but decreases for bright soil (leaves '
            'mask the highly reflective ground). Both cases converge at large LAI '
            'to the canopy-only reflectance. '
            'Parameters: SZA=40 deg, fdir=1.0, uniform LAD (G=0.5).'
        ),
    ),
    dict(
        file='figure09_BHR_Trans_vs_LAI.png',
        title='Figure 9 — LAI Effect on BHR and Transmittance',
        caption=(
            'BHR (left) and canopy transmittance (right) vs. LAI for RED and '
            'NIR bands. Transmittance = net downward irradiance at the ground: '
            '(Fd - Fu)_bottom / Fin. '
            'NIR BHR increases steeply with LAI (high leaf scattering albedo) '
            'while RED BHR stays near-flat (strong absorption limits multiple '
            'scattering). Both transmittances decrease exponentially with LAI '
            'as the canopy intercepts more radiation. '
            'At LAI=6, NIR transmittance (~15%) still exceeds RED transmittance '
            '(~2%) because NIR photons scatter through the canopy rather than '
            'being absorbed. '
            'Parameters: SZA=40 deg, fdir=1.0, rhoG=0.10 (RED), 0.15 (NIR).'
        ),
    ),
    dict(
        file='figure10_BHR_Trans_vs_SZA.png',
        title='Figure 10 — Solar Polar Angle Effect on BHR and Transmittance',
        caption=(
            'BHR (left) and canopy transmittance (right) vs. solar zenith angle '
            '(SZA) from 10 to 80 degrees at fixed LAI=1.5. '
            'NIR BHR increases steeply at high SZA: oblique solar geometry '
            'means the beam traverses more leaf area per unit depth, driving '
            'more multiple scattering and increasing the upward escape fraction. '
            'RED BHR is nearly flat because strong leaf absorption quenches '
            'multiple scattering regardless of geometry. '
            'Both transmittances decrease with SZA (longer optical path). '
            'RED transmittance drops sharply to near zero at SZA=80 deg. '
            'Parameters: LAI=1.5, fdir=1.0, uniform LAD.'
        ),
    ),
    dict(
        file='figure11_Konza_DOM_vs_field.png',
        title='Figure 11 — Konza Prairie: DOM Model vs. Measured HDRF',
        caption=(
            'DOM_v3 principal-plane HDRF (solid line) vs. field measurements '
            'from Konza Prairie grassland, Kansas (stars). '
            'RED: rhoL=0.1814, tauL=0.0926, rhoG=0.0825, fdir=0.862. '
            'NIR: rhoL=0.4525, tauL=0.4913, rhoG=0.1363, fdir=0.931. '
            'LAI=2.2, SZA=70 deg. '
            'The DOM reproduces the broad bowl-shaped angular dependence but '
            'overestimates HDRF at extreme view angles and misses the hot-spot '
            'enhancement visible in the field data near VZA=-70 deg '
            '(retro-illumination direction). This discrepancy directly '
            'demonstrates the need for the hot-spot parameterisation of Ch4 '
            'Sec.9, which accounts for shadow-hiding at retro-illumination.'
        ),
    ),
]


# ===========================================================================
# STEP 1 — Run all figure scripts headlessly
# ===========================================================================
def run_scripts():
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    repo_root = os.path.dirname(CODE_DIR)

    print("=" * 60)
    print("  EE645 — Generating all figures")
    print("=" * 60)
    failed = []
    for script in SCRIPTS:
        path = os.path.join(CODE_DIR, script)
        print(f"\n--- {script} ---")
        result = subprocess.run(
            [sys.executable, path],
            env=env,
            cwd=repo_root,
        )
        if result.returncode != 0:
            print(f"  ERROR: exited with code {result.returncode}")
            failed.append(script)
        else:
            print(f"  OK")
    return failed


# ===========================================================================
# STEP 2 — Assemble PDF
# ===========================================================================
def build_pdf():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    present = [e for e in MANIFEST
               if os.path.exists(os.path.join(FIGURES_DIR, e['file']))]
    missing = [e for e in MANIFEST
               if not os.path.exists(os.path.join(FIGURES_DIR, e['file']))]

    print("\n" + "=" * 60)
    print("  Assembling PDF")
    print("=" * 60)
    if missing:
        print("Skipped (PNG not found):")
        for e in missing:
            print(f"  {e['file']}")

    with PdfPages(OUTPUT_PDF) as pdf:

        # Cover page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.68, 'EE645 — Physical Models in Remote Sensing',
                 ha='center', fontsize=15, fontweight='bold')
        fig.text(0.5, 0.61, 'Final Report: Discrete Ordinates Method',
                 ha='center', fontsize=18, fontweight='bold')
        fig.text(0.5, 0.53,
                 'Vegetation Canopy Radiative Transfer\n'
                 'DOM_v3 — Bi-Lambertian Phase Function',
                 ha='center', fontsize=12)
        fig.text(0.5, 0.40,
                 'Brian DeVoe\nBoston University  |  Spring 2026\n'
                 'Prof. Ranga B. Myneni',
                 ha='center', fontsize=11, color='dimgray')
        pdf.savefig(fig)
        plt.close(fig)

        for entry in present:
            img_path = os.path.join(FIGURES_DIR, entry['file'])
            img = mpimg.imread(img_path)
            h_px, w_px = img.shape[:2]
            aspect = w_px / h_px

            fig = plt.figure(figsize=(8.5, 11))

            # Title
            fig.text(0.5, 0.955, entry['title'],
                     ha='center', va='top', fontsize=13, fontweight='bold')

            # Image — scale to fit while preserving aspect ratio
            img_h = 0.71
            img_w = min(0.90, img_h * aspect / (8.5 / 11))
            ax = fig.add_axes([(1 - img_w) / 2, 0.21, img_w, img_h])
            ax.imshow(img)
            ax.axis('off')

            # Caption
            wrapped = textwrap.fill(entry['caption'], width=108)
            fig.text(0.05, 0.185, wrapped,
                     ha='left', va='top', fontsize=8.5, color='#333333',
                     linespacing=1.45)

            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Added: {entry['file']}")

    print(f"\nPDF saved to:\n  {OUTPUT_PDF}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
    failed = run_scripts()
    build_pdf()
    if failed:
        print(f"\nWarning — {len(failed)} script(s) failed: {failed}")
    else:
        print("\nDone.")
