"""
Cycle1-Cycle2 Image Registration Pipeline

Registers Cycle2 DAPI/KI67 images to Cycle1 DAPI (reference) to correct
for positional offsets (translation, rotation, scaling) between imaging cycles.

Workflow:
    1. Load Cycle1 DAPI (reference) and Cycle2 DAPI (moving)
    2. Compute common crop region (intersection of both images)
    3. Register Cycle2 DAPI to Cycle1 DAPI using phase correlation (translation)
       or affine transformation (rotation/scaling)
    4. Apply the same transformation to Cycle2 KI67
    5. Verify alignment and save registered images

Usage:
    python register_cycles.py --block <BLOCK_ID> [--method {translation,affine}]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import registration
from skimage.registration import phase_cross_correlation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DATA = Path(r"d:\Try_munan\FYP_LAST\Raw_Data\TMAd")
STITCHED_DIR = Path(r"d:\Try_munan\FYP_LAST\results\stitched\TMAd")
CROPPED_DIR = Path(r"d:\Try_munan\FYP_LAST\results\crop\TMAd")
OUTPUT_DIR = Path(r"d:\Try_munan\FYP_LAST\results\registered")

# Channel mappings
CYCLE1_CHANNELS = ["DAPI", "HER2", "PR", "ER"]
CYCLE2_CHANNELS = ["DAPI", "KI67"]

# Registration settings
UPSAMPLE_FACTOR = 100  # Subpixel precision for phase correlation
MAX_ROTATION_DEGREES = 10  # Max rotation to search for affine registration


# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------

def resolve_block_paths(block: str, use_cropped: bool = True) -> dict:
    """
    Resolve file paths for a block.

    Args:
        block: Block ID (e.g., "G2")
        use_cropped: If True, use cropped data from results/crop/TMAd;
                      if False, use raw tiled data from Raw_Data/TMAd

    Returns dict with paths:
        - cycle1_dapi, cycle1_her2, cycle1_pr, cycle1_er
        - cycle2_dapi, cycle2_ki67
        - each value is either a Path object (for single-file crops)
          or a list of Path objects (for tiled raw data)
    """
    if use_cropped:
        # Path pattern: results/crop/TMAd/{block}/{block}_TMAd_Cycle{N}/{file}
        cycle1_dir = CROPPED_DIR / block / f"{block}_TMAd_Cycle1"
        cycle2_dir = CROPPED_DIR / block / f"{block}_TMAd_Cycle2"

        paths = {
            "cycle1_dapi": cycle1_dir / f"{block}_TMAd_Cycle1_DAPI_crop.tif",
            "cycle1_her2": cycle1_dir / f"{block}_TMAd_Cycle1_HER2_crop.tif",
            "cycle1_pr": cycle1_dir / f"{block}_TMAd_Cycle1_PR_crop.tif",
            "cycle1_er": cycle1_dir / f"{block}_TMAd_Cycle1_ER_crop.tif",
            "cycle2_dapi": cycle2_dir / f"{block}_TMAd_Cycle2_DAPI_crop.tif",
            "cycle2_ki67": cycle2_dir / f"{block}_TMAd_Cycle2_KI67_crop.tif",
        }
    else:
        # Raw tiled data
        cycle1_dir = RAW_DATA / "Cycle1" / block
        cycle2_dir = RAW_DATA / "Cycle2" / block

        def get_tiles(subdir: Path, channel: str) -> list:
            """Get all tiles for a channel."""
            ch_dir = subdir / channel
            if not ch_dir.exists():
                return []
            return sorted(ch_dir.glob("*.TIF"))

        paths = {
            "cycle1_dapi": get_tiles(cycle1_dir, "DAPI"),
            "cycle1_her2": get_tiles(cycle1_dir, "HER2"),
            "cycle1_pr": get_tiles(cycle1_dir, "PR"),
            "cycle1_er": get_tiles(cycle1_dir, "ER"),
            "cycle2_dapi": get_tiles(cycle2_dir, "DAPI"),
            "cycle2_ki67": get_tiles(cycle2_dir, "KI67"),
        }

    return paths


def find_available_blocks_from_crop() -> list:
    """Find blocks that exist in the cropped data directory."""
    if not CROPPED_DIR.exists():
        return []
    blocks = []
    for block_dir in CROPPED_DIR.iterdir():
        if block_dir.is_dir():
            cycle1 = block_dir / f"{block_dir.name}_TMAd_Cycle1"
            cycle2 = block_dir / f"{block_dir.name}_TMAd_Cycle2"
            if cycle1.exists() and cycle2.exists():
                blocks.append(block_dir.name)
    return sorted(blocks)


def find_available_blocks() -> list:
    """Find blocks that have both Cycle1 and Cycle2 data in raw tiled directory."""
    cycle1_dir = RAW_DATA / "Cycle1"
    cycle2_dir = RAW_DATA / "Cycle2"

    if not cycle1_dir.exists() or not cycle2_dir.exists():
        return []

    cycle1_blocks = {d.name for d in cycle1_dir.iterdir() if d.is_dir()}
    cycle2_blocks = {d.name for d in cycle2_dir.iterdir() if d.is_dir()}

    return sorted(cycle1_blocks & cycle2_blocks)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_tiff(path: str) -> np.ndarray:
    """Load TIFF as float32 (H, W)."""
    img = tifffile.imread(path)
    if img.ndim == 3:
        if img.shape[0] in (3, 4):
            img = img[0]
        elif img.shape[2] in (3, 4):
            img = img[:, :, 0]
    return img.astype(np.float32)


def save_tiff(path: str, data: np.ndarray, dtype=np.float32) -> None:
    """Save TIFF preserving original dtype."""
    tifffile.imwrite(path, data.astype(dtype))


# ---------------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------------

def compute_common_crop(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """
    Compute the common crop region for two images of potentially different sizes.
    Returns (crop_h, crop_w) - the size of the intersection region.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    crop_h = min(h1, h2)
    crop_w = min(w1, w2)
    return crop_h, crop_w


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range for registration."""
    p1, p99 = np.percentile(img, (1, 99))
    if p99 - p1 < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    normalized = (img - p1) / (p99 - p1)
    return normalized.astype(np.float32)


def apply_crop(img: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    """Crop image to specified dimensions from top-left."""
    return img[:crop_h, :crop_w]


# ---------------------------------------------------------------------------
# Registration Methods
# ---------------------------------------------------------------------------

def register_translation(
    fixed: np.ndarray,
    moving: np.ndarray,
    upsample_factor: float = 100.0,
) -> tuple:
    """
    Register moving image to fixed using phase cross-correlation.
    Returns (shift, error, phasediff) where shift = (dy, dx).
    """
    shift, error, diffphase = phase_cross_correlation(
        fixed, moving,
        upsample_factor=upsample_factor,
    )
    return shift, error, diffphase


def register_translation_robust(
    fixed: np.ndarray,
    moving: np.ndarray,
    upsample_factor: float = 100.0,
) -> dict:
    """
    Robust translation registration using multiple methods and selection.

    Returns dict with best shift and all candidate results.
    """
    from scipy import ndimage

    results = {}

    # Method 1: Phase cross-correlation on full images
    try:
        shift1, err1, _ = phase_cross_correlation(
            fixed, moving, upsample_factor=upsample_factor
        )
        registered1 = ndimage.shift(moving, shift=shift1, order=1)
        ncc1 = compute_ncc(fixed, registered1)
        results["phase_full"] = {
            "shift": shift1, "error": err1, "ncc": ncc1
        }
    except Exception as e:
        results["phase_full"] = {"shift": (0, 0), "error": 1e10, "ncc": -1}

    # Method 2: Phase cross-correlation on high-intensity regions only
    # (nuclei are brightest in DAPI, so focus on those)
    try:
        # Threshold to get nuclei regions
        fixed_thresh = fixed > np.percentile(fixed, 70)
        moving_thresh = moving > np.percentile(moving, 70)

        # Find bounding boxes
        fx1, fy1 = np.where(fixed_thresh)
        mx1, my1 = np.where(moving_thresh)

        if len(fx1) > 100 and len(mx1) > 100:
            # Crop to region with nuclei
            margin = 50
            fb_y1, fb_y2 = max(0, min(fx1) - margin), min(fixed.shape[0], max(fx1) + margin)
            fb_x1, fb_x2 = max(0, min(fy1) - margin), min(fixed.shape[1], max(fy1) + margin)
            mb_y1, mb_y2 = max(0, min(mx1) - margin), min(moving.shape[0], max(mx1) + margin)
            mb_x1, mb_x2 = max(0, min(my1) - margin), min(moving.shape[1], max(my1) + margin)

            fixed_crop = fixed[fb_y1:fb_y2, fb_x1:fb_x2]
            moving_crop = moving[mb_y1:mb_y2, mb_x1:mb_x2]

            if fixed_crop.size > 1000 and moving_crop.size > 1000:
                shift2, err2, _ = phase_cross_correlation(
                    fixed_crop, moving_crop, upsample_factor=upsample_factor
                )
                # Adjust shift to global coordinates
                shift2_global = (shift2[0] + (mb_y1 - fb_y1), shift2[1] + (mb_x1 - fb_x1))
                registered2 = ndimage.shift(moving, shift=shift2_global, order=1)
                ncc2 = compute_ncc(fixed, registered2)
                results["phase_nuclei"] = {
                    "shift": shift2_global, "error": err2, "ncc": ncc2
                }
            else:
                results["phase_nuclei"] = {"shift": (0, 0), "error": 1e10, "ncc": -1}
        else:
            results["phase_nuclei"] = {"shift": (0, 0), "error": 1e10, "ncc": -1}
    except Exception as e:
        results["phase_nuclei"] = {"shift": (0, 0), "error": 1e10, "ncc": -1}

    # Method 3: Center-of-mass alignment of thresholded images
    try:
        # Get binary of nuclei (high intensity regions)
        fixed_bin = fixed > np.percentile(fixed, 80)
        moving_bin = moving > np.percentile(moving, 80)

        # Compute centroids
        fixed_cmass = np.array(ndimage.center_of_mass(fixed_bin))
        moving_cmass = np.array(ndimage.center_of_mass(moving_bin))

        shift3 = moving_cmass - fixed_cmass
        registered3 = ndimage.shift(moving, shift=shift3, order=1)
        ncc3 = compute_ncc(fixed, registered3)
        results["com"] = {
            "shift": tuple(shift3), "error": 0, "ncc": ncc3
        }
    except Exception as e:
        results["com"] = {"shift": (0, 0), "error": 0, "ncc": -1}

    # Method 4: Grid search NCC (limited range for speed)
    try:
        h, w = fixed.shape

        # Use center ROI for comparison
        margin = 500  # pixels from center
        cy, cx = h // 2, w // 2
        fixed_roi = fixed[cy-margin:cy+margin, cx-margin:cx+margin]
        moving_roi = moving[cy-margin:cy+margin, cx-margin:cx+margin]

        best_ncc = -1
        best_shift = (0, 0)

        # Search range based on expected drift
        search_range = 200
        step = 5  # 5px steps

        print(f"      NCC grid search: range={search_range}px, step={step}px")

        for dy in range(-search_range, search_range + 1, step):
            for dx in range(-search_range, search_range + 1, step):
                shifted = ndimage.shift(moving_roi, shift=(dy, dx), order=1)
                ncc = compute_ncc(fixed_roi, shifted)
                if ncc > best_ncc:
                    best_ncc = ncc
                    best_shift = (dy, dx)

        # Fine-tune with 1px resolution around best
        coarse_dy, coarse_dx = best_shift
        for dy in range(max(-10, coarse_dy - 10), coarse_dy + 11, 1):
            for dx in range(max(-10, coarse_dx - 10), coarse_dx + 11, 1):
                shifted = ndimage.shift(moving_roi, shift=(dy, dx), order=1)
                ncc = compute_ncc(fixed_roi, shifted)
                if ncc > best_ncc:
                    best_ncc = ncc
                    best_shift = (dy, dx)

        print(f"      NCC best: shift=({best_shift[0]:.0f}, {best_shift[1]:.0f}), NCC={best_ncc:.4f}")

        results["ncc_grid"] = {
            "shift": best_shift, "error": 0, "ncc": best_ncc
        }
    except Exception as e:
        results["ncc_grid"] = {"shift": (0, 0), "error": 0, "ncc": -1}
        print(f"      NCC grid failed: {e}")

    # Method 4: Pre-defined affine matrix from QuPath
    # Check if we have a predefined matrix for this block
    # This will be handled in process_single_tile with access to block name

    # Selection strategy: use predefined matrix if available
    # Otherwise fall back to best auto-detected method
    best_method = max(results.keys(), key=lambda k: results[k]["ncc"])

    return {
        "best_shift": results[best_method]["shift"],
        "best_ncc": results[best_method]["ncc"],
        "best_method": best_method,
        "all_results": results,
    }


def register_translation_with_affine(
    block: str,
    fixed: np.ndarray,
    moving: np.ndarray,
) -> dict:
    """
    Register using predefined affine matrix if available, otherwise auto-detect.
    """
    if block in BLOCK_AFFINE_MATRICES:
        matrix = BLOCK_AFFINE_MATRICES[block]
        print(f"      Using predefined affine matrix for {block}")
        result = register_with_affine(fixed, moving, matrix)
        result["all_results"] = {"affine_predefined": result}
        return result
    else:
        print(f"      No predefined matrix for {block}, using auto-detection")
        return register_translation_robust(fixed, moving, UPSAMPLE_FACTOR)


def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute normalized cross-correlation coefficient."""
    img1_flat = img1.ravel()
    img2_flat = img2.ravel()

    # Mask out zero/void regions
    mask = (img1_flat > 0) | (img2_flat > 0)

    if mask.sum() < 100:
        return -1.0

    x = img1_flat[mask]
    y = img2_flat[mask]

    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)

    return float(np.corrcoef(x, y)[0, 1])


def apply_translation(
    img: np.ndarray,
    shift: tuple,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """
    Apply translation to image using spline interpolation.
    order=1: bilinear (good for float images)
    order=0: nearest neighbor (good for masks)
    """
    dy, dx = shift
    return ndimage.shift(img, shift=(dy, dx), order=order, mode=mode, cval=cval)


def register_affine(
    fixed: np.ndarray,
    moving: np.ndarray,
    max_rotation: float = 10.0,
) -> tuple:
    """
    Register moving to fixed using feature-based affine transformation.
    Falls back to translation if insufficient features are found.

    Returns (matrix, method) where matrix is a 2x3 affine matrix.
    """
    from skimage.feature import ORB, match_descriptors
    from skimage.transform import AffineTransform, ProjectiveTransform, estimate
    from skimage.filters import gaussian

    # Smooth images for better feature detection
    fixed_smooth = gaussian(fixed, sigma=1)
    moving_smooth = gaussian(moving, sigma=1)

    # Extract ORB features
    try:
        descriptor_extractor = ORB(n_keypoints=500)
        descriptor_extractor.detect_and_extract(fixed_smooth)
        fixed_keypoints = descriptor_extractor.keypoints
        fixed_descriptors = descriptor_extractor.descriptors

        descriptor_extractor.detect_and_extract(moving_smooth)
        moving_keypoints = descriptor_extractor.keypoints
        moving_descriptors = descriptor_extractor.descriptors

        if len(fixed_keypoints) < 10 or len(moving_keypoints) < 10:
            raise ValueError("Insufficient features detected")

        # Match descriptors
        matches = match_descriptors(fixed_descriptors, moving_descriptors, cross_check=True)

        if len(matches) < 10:
            raise ValueError("Insufficient matches")

        # Estimate affine transformation using RANSAC
        fixed_matched = fixed_keypoints[matches[:, 0]]
        moving_matched = moving_keypoints[matches[:, 1]]

        tform = AffineTransform()
        estimate(tform, moving_matched, fixed_matched)

        # Check if transformation is reasonable (mostly translation + small rotation)
        angle_deg = np.rad2deg(tform.rotation)
        scale = tform.scale if hasattr(tform, 'scale') else 1.0

        if abs(angle_deg) > max_rotation:
            warnings.warn(f"Large rotation detected ({angle_deg:.1f}°), may indicate poor alignment")
            print(f"  Warning: rotation = {angle_deg:.1f}°, scale = {scale:.3f}")

        # Build 2x3 affine matrix
        matrix = np.array([
            [tform.params[0, 0], tform.params[0, 1], tform.params[0, 2]],
            [tform.params[1, 0], tform.params[1, 1], tform.params[1, 2]],
        ])

        print(f"  Affine: rotation={angle_deg:.2f}°, scale={scale:.4f}")
        return matrix, "affine"

    except Exception as e:
        warnings.warn(f"Affine registration failed ({e}), falling back to translation")
        return None, "translation_fallback"


def apply_affine(
    img: np.ndarray,
    matrix: np.ndarray,
    output_shape: tuple = None,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """
    Apply affine transformation to image.
    matrix: 2x3 affine matrix [[a,b,tx],[c,d,ty]]
    """
    if output_shape is None:
        output_shape = img.shape
    return ndimage.affine_transform(
        img,
        matrix,
        output_shape=output_shape,
        order=order,
        mode=mode,
        cval=cval,
    )


# Pre-defined translation vectors for each block (from QuPath manual alignment)
# Format: (dy, dx) - vertical and horizontal shift in pixels
BLOCK_TRANSLATION_VECTORS = {
    "G2": (150, -20),  # From earlier best NCC result, adjust as needed
    # Add other blocks here as needed
}


def register_translation_with_affine(
    block: str,
    fixed: np.ndarray,
    moving: np.ndarray,
) -> dict:
    """
    Register using predefined translation if available, otherwise auto-detect.
    """
    if block in BLOCK_TRANSLATION_VECTORS:
        shift = BLOCK_TRANSLATION_VECTORS[block]
        print(f"      Using predefined translation for {block}: dy={shift[0]}, dx={shift[1]}")

        from scipy import ndimage
        registered = ndimage.shift(moving, shift=shift, order=1)
        ncc = compute_ncc(fixed, registered)

        return {
            "best_shift": shift,
            "best_ncc": ncc,
            "best_method": "translation_predefined",
            "all_results": {"translation_predefined": {
                "shift": shift,
                "ncc": ncc,
            }},
        }
    else:
        print(f"      No predefined translation for {block}, using auto-detection")
        return register_translation_robust(fixed, moving, UPSAMPLE_FACTOR)


# ---------------------------------------------------------------------------
# Verification & Visualization
# ---------------------------------------------------------------------------

def compute_registration_quality(
    fixed: np.ndarray,
    registered: np.ndarray,
) -> dict:
    """
    Compute quality metrics for registration.
    Returns dict with MSE, NCC (normalized cross-correlation), and MI (mutual information).
    """
    # Mean Squared Error
    mse = np.mean((fixed - registered) ** 2)

    # Normalized cross-correlation
    fixed_norm = (fixed - fixed.mean()) / (fixed.std() + 1e-8)
    registered_norm = (registered - registered.mean()) / (registered.std() + 1e-8)
    ncc = np.corrcoef(fixed_norm.ravel(), registered_norm.ravel())[0, 1]

    # Mutual information (simplified)
    def mutual_info(x, y, bins=50):
        hist_2d, _, _ = np.histogram2d(x.ravel(), y.ravel(), bins=bins)
        hist_2d = hist_2d + 1e-10  # Avoid log(0)
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        mi = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if pxy[i, j] > 0:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j] + 1e-10))
        return mi

    mi = mutual_info(fixed, registered)

    return {
        "mse": float(mse),
        "ncc": float(ncc),
        "mi": float(mi),
    }


def create_alignment_overlay(
    fixed: np.ndarray,
    moving_before: np.ndarray,
    moving_after: np.ndarray,
    output_path: str,
    title: str = "Alignment Verification",
) -> None:
    """
    Create a 2x2 overlay comparing before and after registration.
    """
    def norm(x):
        p1, p99 = np.percentile(x, (1, 99))
        return np.clip((x - p1) / (p99 - p1 + 1e-8), 0, 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(title, fontsize=16)

    # [1] Fixed (Cycle1 DAPI)
    axes[0, 0].imshow(norm(fixed), cmap="gray")
    axes[0, 0].set_title("Reference: Cycle1 DAPI")
    axes[0, 0].axis("off")

    # [2] Moving before registration (Cycle2 DAPI)
    axes[0, 1].imshow(norm(moving_before), cmap="gray")
    axes[0, 1].set_title("Moving: Cycle2 DAPI (before)")
    axes[0, 1].axis("off")

    # [3] Overlay before registration (red=Cycle1, green=Cycle2)
    overlay_before = np.zeros((*fixed.shape, 3))
    overlay_before[:, :, 0] = norm(fixed)  # Red = reference
    overlay_before[:, :, 1] = norm(moving_before)  # Green = moving
    axes[1, 0].imshow(overlay_before)
    axes[1, 0].set_title("Overlay BEFORE (red=C1, green=C2)")
    axes[1, 0].axis("off")

    # [4] Overlay after registration
    overlay_after = np.zeros((*fixed.shape, 3))
    overlay_after[:, :, 0] = norm(fixed)  # Red = reference
    overlay_after[:, :, 1] = norm(moving_after)  # Green = registered
    axes[1, 1].imshow(overlay_after)
    axes[1, 1].set_title("Overlay AFTER (red=C1, green=C2)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Verification overlay saved: {output_path}")


def create_ki67_overlay(
    dapi_ref: np.ndarray,
    ki67_registered: np.ndarray,
    output_path: str,
    title: str = "KI67 Registration",
) -> None:
    """
    Create overlay showing DAPI (nuclei) with KI67 expression.
    """
    def norm(x):
        p1, p99 = np.percentile(x, (1, 99))
        return np.clip((x - p1) / (p99 - p1 + 1e-8), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(title, fontsize=16)

    # DAPI
    axes[0].imshow(norm(dapi_ref), cmap="gray")
    axes[0].set_title("Cycle1 DAPI (Reference)")
    axes[0].axis("off")

    # KI67 overlay
    overlay = np.zeros((*dapi_ref.shape, 3))
    overlay[:, :, 0] = norm(dapi_ref)  # Red = DAPI
    overlay[:, :, 1] = norm(ki67_registered)  # Green = KI67
    overlay[:, :, 2] = norm(dapi_ref) * 0.3  # Blue = slight DAPI
    axes[1].imshow(overlay)
    axes[1].set_title("Registered KI67 (green) + DAPI (red)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  KI67 overlay saved: {output_path}")


# ---------------------------------------------------------------------------
# Batch Registration
# ---------------------------------------------------------------------------

def register_single_block(
    block: str,
    method: str = "translation",
    skip_existing: bool = True,
    use_cropped: bool = True,
) -> dict:
    """
    Register Cycle2 images to Cycle1 for a single block.

    Args:
        block: Block ID (e.g., "G2")
        method: Registration method ("translation" or "affine")
        skip_existing: Skip if output already exists
        use_cropped: Use cropped data from results/crop (vs raw tiled data)

    Returns dict with registration results and output paths.
    """
    print("\n" + "=" * 60)
    print(f"Processing Block: {block}")
    print("=" * 60)

    paths = resolve_block_paths(block, use_cropped=use_cropped)
    output_block_dir = OUTPUT_DIR / block
    output_block_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing output
    if skip_existing:
        existing = output_block_dir / f"{block}_registered_summary.txt"
        if existing.exists():
            print(f"  Already processed, skipping (use --no-skip to reprocess)")
            return {"status": "skipped", "block": block}

    if use_cropped:
        # Single cropped files
        dapi1_path = paths["cycle1_dapi"]
        dapi2_path = paths["cycle2_dapi"]
        ki67_path = paths["cycle2_ki67"]

        if not dapi1_path.exists():
            raise FileNotFoundError(f"Cycle1 DAPI not found: {dapi1_path}")
        if not dapi2_path.exists():
            raise FileNotFoundError(f"Cycle2 DAPI not found: {dapi2_path}")
        if not ki67_path.exists():
            raise FileNotFoundError(f"KI67 not found: {ki67_path}")

        print(f"\n[Data] Using cropped files:")
        print(f"  Cycle1 DAPI: {dapi1_path.name}")
        print(f"  Cycle2 DAPI: {dapi2_path.name}")
        print(f"  KI67:        {ki67_path.name}")

        # Process as single image
        tile_result = process_single_tile(
            block=block,
            tile_idx=0,
            dapi1_path=dapi1_path,
            dapi2_path=dapi2_path,
            ki67_path=ki67_path,
            output_dir=output_block_dir,
            method=method,
        )

        results = {
            "block": block,
            "n_tiles": 1,
            "tiles": [tile_result],
        }

    else:
        # Tiled raw data
        dapi1_tiles = paths["cycle1_dapi"]
        dapi2_tiles = paths["cycle2_dapi"]
        ki67_tiles = paths["cycle2_ki67"]

        if not dapi1_tiles:
            raise FileNotFoundError(f"No Cycle1 DAPI tiles found for {block}")
        if not dapi2_tiles:
            raise FileNotFoundError(f"No Cycle2 DAPI tiles found for {block}")
        if not ki67_tiles:
            raise FileNotFoundError(f"No KI67 tiles found for {block}")

        print(f"\n[Data] Found {len(dapi1_tiles)} Cycle1 DAPI tiles")
        print(f"[Data] Found {len(dapi2_tiles)} Cycle2 DAPI tiles")
        print(f"[Data] Found {len(ki67_tiles)} KI67 tiles")

        results = {
            "block": block,
            "n_tiles": len(dapi1_tiles),
            "tiles": [],
        }

        for tile_idx in range(min(len(dapi1_tiles), len(dapi2_tiles), len(ki67_tiles))):
            tile_result = process_single_tile(
                block=block,
                tile_idx=tile_idx,
                dapi1_path=dapi1_tiles[tile_idx],
                dapi2_path=dapi2_tiles[tile_idx],
                ki67_path=ki67_tiles[tile_idx],
                output_dir=output_block_dir,
                method=method,
            )
            results["tiles"].append(tile_result)

    # Save summary
    summary_path = output_block_dir / f"{block}_registered_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Block: {block}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Use cropped: {use_cropped}\n")
        f.write(f"N tiles: {results['n_tiles']}\n")
        f.write(f"\nTile results:\n")
        for tr in results["tiles"]:
            f.write(f"  Tile {tr['idx']}: shift=({tr['shift_y']:.2f}, {tr['shift_x']:.2f}), "
                   f"ncc_before={tr['ncc_before']:.4f}, ncc_after={tr['ncc_after']:.4f}\n")

    return results


def process_single_tile(
    block: str,
    tile_idx: int,
    dapi1_path: Path,
    dapi2_path: Path,
    ki67_path: Path,
    output_dir: Path,
    method: str = "translation",
) -> dict:
    """Process a single tile pair with full diagnostics."""
    print(f"\n  --- Tile {tile_idx} ---")

    # ====== Load ======
    dapi1 = load_tiff(str(dapi1_path))
    dapi2 = load_tiff(str(dapi2_path))
    ki67 = load_tiff(str(ki67_path))

    print(f"    DAPI1: shape={dapi1.shape}, dtype={dapi1.dtype}, "
          f"range=[{dapi1.min():.0f}, {dapi1.max():.0f}], "
          f"mean={dapi1.mean():.1f}, std={dapi1.std():.1f}")
    print(f"    DAPI2: shape={dapi2.shape}, dtype={dapi2.dtype}, "
          f"range=[{dapi2.min():.0f}, {dapi2.max():.0f}], "
          f"mean={dapi2.mean():.1f}, std={dapi2.std():.1f}")
    print(f"    KI67:  shape={ki67.shape}, dtype={ki67.dtype}")

    # ====== Size check ======
    if dapi1.shape != dapi2.shape:
        print(f"    Size mismatch: {dapi1.shape} vs {dapi2.shape}")
        print(f"    Need to investigate why sizes differ before cropping")

    # ====== Crop ======
    crop_h, crop_w = compute_common_crop(dapi1, dapi2)
    dapi1_crop = apply_crop(dapi1, crop_h, crop_w)
    dapi2_crop = apply_crop(dapi2, crop_h, crop_w)
    ki67_crop = apply_crop(ki67, crop_h, crop_w)

    # ====== Preprocessing ======
    # Normalize
    dapi1_norm = normalize_image(dapi1_crop)
    dapi2_norm = normalize_image(dapi2_crop)

    # Histogram matching (key fix)
    from skimage.exposure import match_histograms
    dapi2_matched = match_histograms(dapi2_norm, dapi1_norm)

    # Gaussian smoothing for denoising (optional, helps with noisy images)
    from skimage.filters import gaussian
    dapi1_smooth = gaussian(dapi1_norm, sigma=1)
    dapi2_smooth = gaussian(dapi2_matched, sigma=1)

    # ====== Quality before ======
    ncc_before = compute_ncc(dapi1_smooth, dapi2_smooth)
    print(f"    NCC before registration: {ncc_before:.4f}")

    # ====== Registration ======
    print(f"\n[Registration] Using phase_cross_correlation ...")
    shift, error, diffphase = phase_cross_correlation(
        dapi1_smooth,
        dapi2_smooth,
        upsample_factor=100,
    )
    print(f"    Raw shift: dy={shift[0]:.2f}, dx={shift[1]:.2f}")
    print(f"    Error: {error:.6f}")

    # Sanity check: if shift is suspiciously large
    if abs(shift[0]) > crop_h * 0.5 or abs(shift[1]) > crop_w * 0.5:
        print(f"    Shift too large - likely registration failure!")
        print(f"    Check if images are from the same tissue region")

    # ====== Apply ======
    dapi2_registered = apply_translation(dapi2_crop, shift, order=1, cval=0)
    ki67_registered = apply_translation(ki67_crop, shift, order=1, cval=0)

    # ====== Quality after ======
    ncc_after = compute_ncc(dapi1_norm, normalize_image(dapi2_registered))
    print(f"    NCC after registration: {ncc_after:.4f}")
    print(f"    NCC improvement: {ncc_after - ncc_before:+.4f}")

    if ncc_after < ncc_before:
        print(f"    NCC went down - registration made things worse!")

    # ====== Save ======
    prefix = f"{block}_tile{tile_idx:02d}"
    dapi2_out = output_dir / f"{prefix}_DAPI_registered.tif"
    ki67_out = output_dir / f"{prefix}_KI67_registered.tif"

    save_tiff(str(dapi2_out), dapi2_registered)
    save_tiff(str(ki67_out), ki67_registered)

    # ====== Visualization ======
    overlay_path = output_dir / f"{prefix}_alignment_overlay.png"
    create_alignment_overlay(
        dapi1_norm, dapi2_matched, normalize_image(dapi2_registered),
        str(overlay_path),
        title=f"{block} Tile {tile_idx}: shift=({shift[0]:.1f}, {shift[1]:.1f}), NCC={ncc_after:.4f}",
    )

    ki67_overlay_path = output_dir / f"{prefix}_ki67_overlay.png"
    create_ki67_overlay(
        dapi1_crop, ki67_registered, str(ki67_overlay_path),
        title=f"{block} Tile {tile_idx}: KI67 + DAPI",
    )

    return {
        "idx": tile_idx,
        "shift_y": float(shift[0]),
        "shift_x": float(shift[1]),
        "ncc_before": float(ncc_before),
        "ncc_after": float(ncc_after),
        "dapi2_registered": str(dapi2_out),
        "ki67_registered": str(ki67_out),
    }


# ---------------------------------------------------------------------------
# Command-line Interface
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Register Cycle2 images to Cycle1 for TMA analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all blocks from cropped data (default)
  python register_cycles.py --all

  # Process specific block from cropped data
  python register_cycles.py --block G2

  # Process from raw tiled data
  python register_cycles.py --block G2 --raw-tiles

  # Use affine registration for blocks with rotation
  python register_cycles.py --block G2 --method affine

  # Reprocess even if output exists
  python register_cycles.py --block G2 --no-skip
        """,
    )
    p.add_argument(
        "--block", type=str, default=None,
        help="Block ID to process (e.g., G2, A10). If not specified, processes all blocks."
    )
    p.add_argument(
        "--all", action="store_true",
        help="Process all blocks with both Cycle1 and Cycle2 data"
    )
    p.add_argument(
        "--method", type=str, default="translation",
        choices=["translation", "affine"],
        help="Registration method: translation (fast, for drift) or affine (for rotation/scale)"
    )
    p.add_argument(
        "--skip-existing", dest="skip_existing", action="store_true", default=True,
        help="Skip blocks that already have output (default: True)"
    )
    p.add_argument(
        "--no-skip", dest="skip_existing", action="store_false",
        help="Reprocess even if output exists"
    )
    p.add_argument(
        "--raw-tiles", action="store_true",
        help="Use raw tiled data from Raw_Data instead of cropped images"
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Update output directory if specified
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    use_cropped = not args.raw_tiles

    print("=" * 60)
    print("Cycle1-Cycle2 Image Registration Pipeline")
    print("=" * 60)
    print(f"Mode:     {'Cropped images' if use_cropped else 'Raw tiled data'}")
    print(f"Input:    {CROPPED_DIR if use_cropped else RAW_DATA}")
    print(f"Output:   {OUTPUT_DIR}")
    print(f"Method:   {args.method}")

    # Determine blocks to process
    if args.all or args.block is None:
        if use_cropped:
            blocks = find_available_blocks_from_crop()
        else:
            blocks = find_available_blocks()
        if not blocks:
            print("Error: No blocks found with both Cycle1 and Cycle2 data")
            if use_cropped:
                print(f"  Expected in: {CROPPED_DIR}")
            else:
                print(f"  Expected Cycle1 in: {RAW_DATA / 'Cycle1'}")
                print(f"  Expected Cycle2 in: {RAW_DATA / 'Cycle2'}")
            return
        print(f"\nFound {len(blocks)} blocks with both cycles: {blocks}")
    else:
        blocks = [args.block]

    # Process each block
    all_results = {}
    for block in blocks:
        try:
            result = register_single_block(
                block=block,
                method=args.method,
                skip_existing=args.skip_existing,
                use_cropped=use_cropped,
            )
            all_results[block] = result
        except Exception as e:
            print(f"\nError processing {block}: {e}")
            import traceback
            traceback.print_exc()
            all_results[block] = {"status": "error", "error": str(e)}

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for block, result in all_results.items():
        status = result.get("status", "unknown")
        if status == "skipped":
            print(f"  {block}: SKIPPED (already processed)")
        elif status == "error":
            print(f"  {block}: ERROR - {result.get('error', 'Unknown')}")
        else:
            ncc_before = result.get("tiles", [{}])[0].get("ncc_before", 0)
            ncc_after = result.get("tiles", [{}])[0].get("ncc_after", 0)
            shift_y = result.get("tiles", [{}])[0].get("shift_y", 0)
            shift_x = result.get("tiles", [{}])[0].get("shift_x", 0)
            print(f"  {block}: DONE (shift: {shift_y:.1f}, {shift_x:.1f} px, NCC: {ncc_before:.4f} -> {ncc_after:.4f})")

    print("\nRegistration complete!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
