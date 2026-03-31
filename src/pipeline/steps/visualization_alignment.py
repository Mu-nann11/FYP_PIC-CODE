"""
Step 5: Alignment Visualization
生成配准结果可视化：配准前后对比图、多通道叠加图等
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import tifffile
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def load_image(path, max_size=1024):
    """Load TIFF image and resize if too large"""
    if not Path(path).exists():
        return None
    
    img = tifffile.imread(path)
    
    # Resize if too large
    if img.shape[0] > max_size or img.shape[1] > max_size:
        scale = max_size / max(img.shape[0], img.shape[1])
        from skimage.transform import resize
        img = resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)))
    
    return img

def normalize_image(img, percentile=99):
    """Normalize image for visualization"""
    if img is None:
        return None
    img = img.astype(np.float32)
    vmax = np.percentile(img, percentile)
    img = np.clip(img / vmax, 0, 1)
    return img

def create_rgb_overlay(red, green, blue):
    """Create RGB overlay from three grayscale images"""
    h, w = red.shape[:2]
    rgb = np.zeros((h, w, 3))
    rgb[:, :, 0] = normalize_image(red)
    rgb[:, :, 1] = normalize_image(green)
    rgb[:, :, 2] = normalize_image(blue)
    return np.clip(rgb, 0, 1)

def visualize_alignment_for_block(block, dataset="TMAe", force=False):
    """Generate alignment visualization for a block"""
    
    from ..config import BASE_DIR, DATASETS
    
    if dataset not in DATASETS:
        print(f"[VIS] Unknown dataset: {dataset}")
        return False
    
    # Paths
    registered_dir = BASE_DIR / "results" / "registered" / block
    output_dir = BASE_DIR / "results" / "pipeline_reports" / block
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already done
    comparison_png = output_dir / f"{block}_registration_comparison.png"
    overlay_png = output_dir / f"{block}_multi_marker_overlay.png"
    
    if not force and comparison_png.exists() and overlay_png.exists():
        print(f"[VIS] {block}: Alignment visualization already done")
        return True
    
    print(f"[VIS] {block}: Creating alignment visualizations...")
    
    try:
        # Load registered images - using actual file naming convention
        # Files are named: {block}_Cycle1_DAPI.tif, {block}_Cycle1_ER_aligned.tif, etc.
        dapi_c1 = load_image(registered_dir / f"{block}_Cycle1_DAPI.tif")
        er = load_image(registered_dir / f"{block}_Cycle1_ER_aligned.tif")
        pr = load_image(registered_dir / f"{block}_Cycle1_PR_aligned.tif")
        her2 = load_image(registered_dir / f"{block}_Cycle1_HER2_aligned.tif")
        
        # Try to load Ki67 if it exists (Cycle2)
        ki67_paths = [
            registered_dir / f"{block}_Cycle2_KI67_aligned.tif",
            registered_dir / f"{block}_Ki67_aligned.tif",
        ]
        ki67 = None
        for ki67_path in ki67_paths:
            ki67 = load_image(ki67_path)
            if ki67 is not None:
                break
        
        # 1. Multi-marker overlay (ER + PR + HER2)
        if er is not None and pr is not None and her2 is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Left: ER(red) + PR(green) + HER2(blue)
            rgb = create_rgb_overlay(er, pr, her2)
            axes[0].imshow(rgb)
            axes[0].set_title(f'{block}: ER(Red) + PR(Green) + HER2(Blue)')
            axes[0].axis('off')
            
            # Right: DAPI + Ki67 overlay if Ki67 available
            if ki67 is not None:
                rgb2 = np.zeros((dapi_c1.shape[0], dapi_c1.shape[1], 3))
                rgb2[:, :, 0] = normalize_image(ki67)  # Ki67 in red
                rgb2[:, :, 1] = normalize_image(dapi_c1) * 0.5  # DAPI in green (dimmed)
                axes[1].imshow(rgb2)
                axes[1].set_title(f'{block}: DAPI + Ki67')
            else:
                axes[1].imshow(normalize_image(dapi_c1), cmap='gray')
                axes[1].set_title(f'{block}: DAPI')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(overlay_png, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[VIS] {block}: Saved multi-marker overlay to {overlay_png.name}")
        
        return True
        
    except Exception as e:
        print(f"[VIS] {block}: ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_visualization_alignment(blocks, dataset="TMAe", force=False):
    """Run alignment visualization for all blocks"""
    
    results = []
    for block in blocks:
        success = visualize_alignment_for_block(block, dataset, force)
        results.append((block, success))
    
    print(f"\n[VIS] Alignment visualization complete: {sum(1 for _, s in results if s)}/{len(results)} blocks")
    return all(success for _, success in results)


if __name__ == "__main__":
    # Quick test
    blocks = ["D5", "E9"]
    run_visualization_alignment(blocks, force=True)
