"""
run_all_figures.py -- Generate all figures and compile the final report PDF

Runs every figure script headlessly (no popup windows), then assembles all
outputs into a single PDF (one figure per page, title only).

Usage (from repo root):
    python code/run_all_figures.py

Output:
    code/figures/EE645_report_figures.pdf

Notes:
  - Total runtime: ~10-15 minutes (DOM solver runs multiple times).
"""

import os, sys, subprocess
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
    dict(file='energy_balance_table.png',      title='Energy Balance Table'),
    dict(file='figure01_gauss_quadrature.png',  title='Figure 1 — Leaf Angle Distributions'),
    dict(file='figure02_LAD_verification.png',  title='Figure 2 — G-Function for Various Leaf Angle Distributions'),
    dict(file='figure03_phase_function.png',    title='Figure 3 — Volume Scattering Phase Function'),
    dict(file='figure04_uncollided_RED.png',    title='Figure 4 — Uncollided Radiation: RED Band'),
    dict(file='figure05_uncollided_NIR.png',    title='Figure 5 — Uncollided Radiation: NIR Band'),
    dict(file='figure06_HDRF_polar_map.png',    title='Figure 6 — HDRF Hemispherical Polar Map (fdir=0.70)'),
    dict(file='figure07_HDRF_principal_plane.png', title='Figure 7 — HDRF Principal Plane Cross-Section (fdir=0.70)'),
    dict(file='figure08_BHR_vs_LAI_soil.png',   title='Figure 8 — BHR as a Function of LAI and Soil Brightness'),
    dict(file='figure09_BHR_Trans_vs_LAI.png',  title='Figure 9 — LAI Effect on BHR and Transmittance'),
    dict(file='figure10_BHR_Trans_vs_SZA.png',  title='Figure 10 — Solar Polar Angle Effect on BHR and Transmittance'),
    dict(file='figure11_Konza_DOM_vs_field.png', title='Figure 11 — Konza Prairie: DOM Model vs. Measured HDRF'),
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
            fig.text(0.5, 0.97, entry['title'],
                     ha='center', va='top', fontsize=13, fontweight='bold')

            # Image — fill page below title
            img_h = 0.88
            img_w = min(0.95, img_h * aspect / (8.5 / 11))
            ax = fig.add_axes([(1 - img_w) / 2, 0.05, img_w, img_h])
            ax.imshow(img)
            ax.axis('off')

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
