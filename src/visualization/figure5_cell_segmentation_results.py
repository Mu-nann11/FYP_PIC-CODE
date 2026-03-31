"""
Figure 5: Cell Segmentation & Screening Results
Two-panel main view + ROI comparison for detailed visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path
from tifffile import tifffile
import warnings
warnings.filterwarnings('ignore')

# Configuration
BLOCK = "H2"  # Nuclei=5034, Cyto=6505
RESULTS_DIR = Path(r"d:\Try_munan\FYP_LAST\results")
OUTPUT_DIR = RESULTS_DIR / "figures"

# Paths
DAPI_PATH = RESULTS_DIR / "registered" / BLOCK / f"{BLOCK}_Cycle1_DAPI.tif"
CYTO_PATH = RESULTS_DIR / "segmentation" / BLOCK / f"{BLOCK}_cyto_masks.tif"
NUC_PATH = RESULTS_DIR / "segmentation" / BLOCK / f"{BLOCK}_nuclei_masks.tif"


def load_image(path):
    """Load TIFF image and convert to float32"""
    img = tifffile.imread(str(path))
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img


def find_dense_roi(dapi, cyto_masks, nuc_masks, target_size=800):
    """Find a region with good cell density for ROI display"""
    h, w = dapi.shape

    # Try multiple candidate regions
    candidates = [
        (h // 4, w // 4, target_size, target_size),
        (h // 3, w // 3, target_size, target_size),
        (h // 2, w // 4, target_size, target_size),
        (h // 2, w // 2, target_size, target_size),
        (h // 3, w // 2, target_size, target_size),
    ]

    best_roi = (0, 0, target_size, target_size)
    best_score = 0

    for y, x, rh, rw in candidates:
        # Ensure within bounds
        if y + rh > h or x + rw > w:
            continue

        # Score based on cell density
        nuc_count = np.sum(nuc_masks[y:y+rh, x:x+rw] > 0)
        cyto_count = np.sum(cyto_masks[y:y+rh, x:x+rw] > 0)
        score = nuc_count + cyto_count * 0.5

        if score > best_score:
            best_score = score
            best_roi = (y, x, rh, rw)

    return best_roi


def create_overlay(dapi, cyto_masks, nuc_masks):
    """Create RGB overlay with DAPI background, cytoplasm green, nuclei red"""
    h, w = dapi.shape
    overlay_rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Normalize DAPI for background
    dapi_norm = (dapi - dapi.min()) / (dapi.max() - dapi.min() + 1e-8)

    # DAPI as blue-tinted background
    overlay_rgb[:, :, 0] = dapi_norm * 0.7
    overlay_rgb[:, :, 1] = dapi_norm * 0.7
    overlay_rgb[:, :, 2] = dapi_norm * 0.95

    # Cytoplasm (green)
    cyto_mask = cyto_masks > 0
    overlay_rgb[cyto_mask, 0] = 0.2
    overlay_rgb[cyto_mask, 1] = 0.85
    overlay_rgb[cyto_mask, 2] = 0.2

    # Nuclei (red, on top)
    nuc_mask = nuc_masks > 0
    overlay_rgb[nuc_mask, 0] = 0.95
    overlay_rgb[nuc_mask, 1] = 0.15
    overlay_rgb[nuc_mask, 2] = 0.15

    return overlay_rgb


def create_figure():
    """Main figure creation function with ROI comparison"""
    print(f"Creating Figure 5 for block {BLOCK}...")

    # Load images
    print("Loading DAPI image...")
    dapi = load_image(DAPI_PATH)

    print("Loading cytoplasm masks...")
    cyto_masks = tifffile.imread(str(CYTO_PATH))
    n_cyto = len(np.unique(cyto_masks)) - 1

    print("Loading nuclei masks...")
    nuc_masks = tifffile.imread(str(NUC_PATH))
    n_nuc = len(np.unique(nuc_masks)) - 1

    # Create overlays
    print("Creating overlays...")
    overlay_rgb = create_overlay(dapi, cyto_masks, nuc_masks)

    # Find good ROI
    roi_y, roi_x, roi_h, roi_w = find_dense_roi(dapi, cyto_masks, nuc_masks)
    print(f"ROI selected: y={roi_y}, x={roi_x}, h={roi_h}, w={roi_w}")

    # Extract ROI regions
    dapi_roi = dapi[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    overlay_roi = overlay_rgb[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # Normalize DAPI for display
    dapi_display = (dapi - dapi.min()) / (dapi.max() - dapi.min() + 1e-8)

    # Create figure with subplot layout
    fig = plt.figure(figsize=(18, 12))

    # Main title
    fig.suptitle('[Figure 5] Cell Segmentation & Screening Results', fontsize=18, fontweight='bold', y=0.98)

    # Create grid for layout
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.15,
                          left=0.05, right=0.95, top=0.92, bottom=0.08)

    # Row 1: Main views
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(dapi_display, cmap='gray')
    ax1.set_title(f'Original DAPI', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Draw ROI rectangle on main DAPI view
    rect = Rectangle((roi_x, roi_y), roi_w, roi_h,
                      linewidth=3, edgecolor='yellow', facecolor='none')
    ax1.add_patch(rect)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(overlay_rgb)
    ax2.set_title(f'DAPI (Blue) + Cytoplasm (Green) + Nuclei (Red)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    # Draw same ROI rectangle
    rect2 = Rectangle((roi_x, roi_y), roi_w, roi_h,
                       linewidth=3, edgecolor='yellow', facecolor='none')
    ax2.add_patch(rect2)

    # Row 2: ROI zoomed views
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(dapi_roi, cmap='gray')
    ax3.set_title(f'ROI: Original DAPI (Zoomed)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(overlay_roi)
    ax4.set_title(f'ROI: Cell Segmentation Overlay (Zoomed)', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # Add legend
    legend_patches = [
        mpatches.Patch(color='#B3B3E6', label='DAPI (Background)', edgecolor='black', linewidth=1),
        mpatches.Patch(color='#33CC33', label=f'Cytoplasm (n={n_cyto})', edgecolor='black', linewidth=1),
        mpatches.Patch(color='#E63333', label=f'Nuclei (n={n_nuc})', edgecolor='black', linewidth=1),
        mpatches.Patch(color='yellow', label='ROI Selection', edgecolor='black', linewidth=1)
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=11, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.01), edgecolor='black')

    # Add subtitle
    fig.text(0.5, 0.94, f'Block: {BLOCK} | 细胞质掩膜: CPSAM (her2_wholecell_v3) | 核掩膜: Cellpose v3',
             ha='center', fontsize=10, style='italic', color='gray')

    # Add ROI coordinates annotation
    fig.text(0.02, 0.5, f'ROI区域\ny: {roi_y}\nx: {roi_x}\n大小: {roi_h}×{roi_w}',
             fontsize=9, va='center', ha='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Save outputs
    output_png = OUTPUT_DIR / f"figure5_cell_segmentation_results_{BLOCK}.png"
    output_pdf = OUTPUT_DIR / f"figure5_cell_segmentation_results_{BLOCK}.pdf"

    print(f"Saving to {output_png}...")
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"Saving to {output_pdf}...")
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')

    plt.close()
    print("Done!")

    return output_png, output_pdf


if __name__ == "__main__":
    output_png, output_pdf = create_figure()
    print(f"\nOutput files:")
    print(f"  PNG: {output_png}")
    print(f"  PDF: {output_pdf}")
