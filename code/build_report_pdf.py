"""
build_report_pdf.py -- Assemble all figures into a single PDF with captions

Reads PNGs from code/figures/ and produces a one-figure-per-page PDF.
Run after all figure scripts have been executed (or use run_all_figures.py).

Usage:
    python code/build_report_pdf.py

Output:
    code/figures/EE645_report_figures.pdf
"""

import os, textwrap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
OUTPUT_PDF  = os.path.join(FIGURES_DIR, 'EE645_report_figures.pdf')

# ---------------------------------------------------------------------------
# Figure manifest — energy balance table first, then figures in order
# ---------------------------------------------------------------------------
MANIFEST = [
    dict(
        file='energy_balance_table.png',
        title='Energy Balance Table — DOM_v3 (fdir=1.0, SZA=40°)',
        caption=(
            'Canopy energy budget for four configurations: RED and NIR bands at LAI=1.5 and 4.0. '
            'fdir=1.0 (pure direct beam) so reflectance = DHR (directional-hemispherical reflectance). '
            'DHR + canopy absorptance + ground absorptance = F_in within < 1% for all cases. '
            'RED band: leaves strongly absorb (~62–91% of incoming), low DHR (~2–3%). '
            'NIR band: leaves scatter strongly, high DHR (~32–42%), lower absorptance (~13–31%). '
            'Spectral parameters: RED ρL=0.06, τL=0.04, ρg=0.10; NIR ρL=0.45, τL=0.45, ρg=0.15.'
        ),
    ),
    dict(
        file='figure01_gauss_quadrature.png',
        title='Figure 1 — Gauss-Legendre Quadrature Verification',
        caption=(
            'Verification of the Gauss-Legendre quadrature setup for N=16 polar directions. '
            'Confirms sum of weights = 2.0 and the weighted sum over the upward hemisphere = 0.5, '
            'matching the analytic integrals required by the DOM formulation.'
        ),
    ),
    dict(
        file='figure02_LAD_verification.png',
        title='Figure 2 — Leaf Angle Distribution Verification',
        caption=(
            'The g(θL) projected-area function for six leaf angle distributions: planophile, '
            'plagiophile, erectophile, extremophile, uniform (spherical), and erectophile. '
            'The uniform (spherical) distribution gives G=0.5 for all solar directions, '
            'which is the assumption used throughout this report.'
        ),
    ),
    dict(
        file='figure03_phase_function.png',
        title='Figure 3 — Volume Scattering Phase Function',
        caption=(
            'The bi-Lambertian volume scattering phase function Γ(Ω′→Ω) vs. scattering angle β '
            'for RED (ωL=0.10) and NIR (ωL=0.90) leaf properties. '
            'The function is smooth with no singularity at β=180° (retro-illumination); '
            'the hot-spot effect (Ch4 §9) is not modelled in DOM_v3.'
        ),
    ),
    dict(
        file='figure04_uncollided_RED.png',
        title='Figure 4 — Uncollided Radiation: RED Band',
        caption=(
            'Three-panel uncollided DOM solution, RED band (ρL=0.06, τL=0.04, ρg=0.10). '
            'A: normalised downward flux (direct + diffuse sky) vs. L/LAI. '
            'B: normalised upward ground-reflected flux vs. L/LAI. '
            'C: uncollided BRF at canopy top in the principal plane. '
            'Strong leaf absorption keeps upward flux and BRF low. '
            'Parameters: SZA=40°, fdir=0.70, LAI=1.5 and 4.0, N=M=16, K=50.'
        ),
    ),
    dict(
        file='figure05_uncollided_NIR.png',
        title='Figure 5 — Uncollided Radiation: NIR Band',
        caption=(
            'Same three-panel uncollided DOM solution, NIR band (ρL=0.525, τL=0.45, ρg=0.20). '
            'Leaves are nearly transparent (ωL≈0.975), so considerably more radiation reaches '
            'the ground and is reflected upward compared to the RED band. '
            'Parameters identical to Figure 4.'
        ),
    ),
    dict(
        file='figure06_HDRF_polar_map.png',
        title='Figure 6 — HDRF Hemispherical Polar Map (fdir=0.70)',
        caption=(
            '2×2 hemispherical polar maps of HDRF for RED and NIR bands at LAI=1.5 and 4.0. '
            'Polar orientation: 0° at top (forward scatter), 180° at bottom (sun sky position). '
            'Yellow star marks the sun at SZA=40°. NIR HDRF is much larger due to high leaf '
            'single-scattering albedo (ωL=0.90). Mixed illumination: fdir=0.70. '
            'Parameters: SZA=40°, φsolar=0°, N=M=16, K=50.'
        ),
    ),
    dict(
        file='figure07_HDRF_principal_plane.png',
        title='Figure 7 — HDRF Principal Plane (fdir=0.70)',
        caption=(
            'HDRF vs. view zenith angle in the principal plane for RED (left) and NIR (right). '
            'Negative VZA = backscatter toward sun; positive VZA = forward scatter. '
            'LAI=1.5 and 4.0 shown. The smooth bowl shape (no hot-spot peak at −40°) is a known '
            'limitation of the bi-Lambertian phase function used in DOM_v3. '
            'Parameters: SZA=40°, fdir=0.70, N=M=16, K=50.'
        ),
    ),
    dict(
        file='figure08_BHR_vs_LAI_soil.png',
        title='Figure 8 — BHR vs. LAI and Soil Brightness',
        caption=(
            'Bihemispherical reflectance (BHR = Fu_top / Fin) vs. LAI for dark (ρg=0.01) '
            'and bright (ρg=0.30) soils. '
            'RED: BHR decreases with LAI as absorbing leaves mask bright soil. '
            'NIR: BHR increases with LAI for dark soil (leaf reflectance dominates) '
            'but decreases for bright soil (leaves mask the highly reflective ground). '
            'Parameters: SZA=40°, fdir=1.0, uniform LAD.'
        ),
    ),
    dict(
        file='figure09_BHR_Trans_vs_LAI.png',
        title='Figure 9 — LAI Effect on BHR and Transmittance',
        caption=(
            'BHR (left) and canopy transmittance (right) vs. LAI for RED and NIR bands. '
            'NIR BHR increases with LAI while RED BHR stays near-flat (strong absorption). '
            'Both transmittances decrease exponentially with LAI. '
            'Transmittance defined as net downward irradiance at ground: '
            '(F↓ − F↑)bottom / Fin. '
            'Parameters: SZA=40°, fdir=1.0, ρg=0.10 (RED), ρg=0.15 (NIR), uniform LAD.'
        ),
    ),
    dict(
        file='figure10_BHR_Trans_vs_SZA.png',
        title='Figure 10 — Solar Polar Angle Effect on BHR and Transmittance',
        caption=(
            'BHR (left) and transmittance (right) vs. solar zenith angle (SZA) from 10° to 80°. '
            'NIR BHR increases steeply at high SZA (oblique geometry, more multiple scattering). '
            'RED BHR is nearly flat (leaf absorption suppresses multiple scattering). '
            'Both transmittances decrease with SZA (longer path through canopy). '
            'Fixed conditions: LAI=1.5, fdir=1.0, uniform LAD.'
        ),
    ),
    dict(
        file='figure11_Konza_DOM_vs_field.png',
        title='Figure 11 — Konza Prairie: DOM Model vs. Measured HDRF',
        caption=(
            'DOM_v3 principal-plane HDRF (line) vs. Konza Prairie grassland field measurements '
            '(stars). RED: ρL=0.1814, τL=0.0926, ρg=0.0825, fdir=0.862. '
            'NIR: ρL=0.4525, τL=0.4913, ρg=0.1363, fdir=0.931. LAI=2.2, SZA=70°. '
            'The model reproduces the broad bowl shape but overestimates HDRF at extreme view '
            'angles and misses the hot-spot peak visible near VZA=−70° (retro-illumination). '
            'This discrepancy motivates the hot-spot parameterisation described in Ch4 §9.'
        ),
    ),
]

# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------
os.makedirs(FIGURES_DIR, exist_ok=True)
present  = [e for e in MANIFEST if os.path.exists(os.path.join(FIGURES_DIR, e['file']))]
missing  = [e for e in MANIFEST if not os.path.exists(os.path.join(FIGURES_DIR, e['file']))]

if missing:
    print("Missing PNGs (skipped):")
    for e in missing:
        print(f"  {e['file']}")
print(f"\nAssembling {len(present)} figure(s) into {OUTPUT_PDF} ...")

with PdfPages(OUTPUT_PDF) as pdf:

    # Cover page
    fig_cover = plt.figure(figsize=(8.5, 11))
    fig_cover.text(0.5, 0.65, 'EE645 — Physical Models in Remote Sensing',
                   ha='center', va='center', fontsize=16, fontweight='bold')
    fig_cover.text(0.5, 0.58, 'Final Report Figures',
                   ha='center', va='center', fontsize=20, fontweight='bold')
    fig_cover.text(0.5, 0.50, 'Discrete Ordinates Method (DOM_v3)\nVegetation Canopy Radiative Transfer',
                   ha='center', va='center', fontsize=13)
    fig_cover.text(0.5, 0.38, 'Brian DeVoe\nBoston University  |  Spring 2026',
                   ha='center', va='center', fontsize=11, color='dimgray')
    pdf.savefig(fig_cover)
    plt.close(fig_cover)

    # One page per figure
    for entry in present:
        img_path = os.path.join(FIGURES_DIR, entry['file'])
        img = mpimg.imread(img_path)
        h_px, w_px = img.shape[:2]
        aspect = w_px / h_px

        # Layout: title (top) + image (middle) + caption (bottom)
        fig = plt.figure(figsize=(8.5, 11))

        # Title
        fig.text(0.5, 0.95, entry['title'],
                 ha='center', va='top', fontsize=13, fontweight='bold',
                 wrap=True)

        # Image: scale to fit within page margins while preserving aspect ratio
        img_height = 0.72   # fraction of page height reserved for image
        img_width  = min(0.90, img_height * aspect / (8.5 / 11))
        img_left   = (1.0 - img_width) / 2

        ax_img = fig.add_axes([img_left, 0.20, img_width, img_height])
        ax_img.imshow(img)
        ax_img.axis('off')

        # Caption (wrapped)
        wrapped = textwrap.fill(entry['caption'], width=105)
        fig.text(0.05, 0.17, wrapped,
                 ha='left', va='top', fontsize=8.5, color='#333333',
                 linespacing=1.4)

        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Added: {entry['file']}")

print(f"\nDone. PDF saved to:\n  {OUTPUT_PDF}")
