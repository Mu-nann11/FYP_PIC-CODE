#!/usr/bin/env python3
"""
CPSAM Cytoplasm-to-Nucleus Cell Segmentation Pipeline (5-channel)

Workflow:
    DAPI + HER2 merge → (2, H, W) → CellPose CPSAM → cytoplasm masks
    Each cytoplasm region → CellPose nuclei model (or Otsu fallback) → nucleus mask
    Nucleus label = cytoplasm label (1:1 mapping)
    Extract features from ALL 5 channels → save TIFF masks + CSV + PNG overlay
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

import skimage.filters
import skimage.measure
from cellpose import models


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_channel(path: str) -> np.ndarray:
    """Load a single-channel TIFF into (H, W) float32."""
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        if img.shape[0] in (3, 4):
            img = img[0]
        elif img.shape[2] in (3, 4):
            img = img[:, :, 0]
    return img.astype(np.float32)


def save_tiff(path: str, data: np.ndarray) -> None:
    """Save an integer mask as TIFF."""
    tifffile.imwrite(str(path), data.astype(np.int32), photometric="minisblack")


# ---------------------------------------------------------------------------
# Step 1 – Merge channels
# ---------------------------------------------------------------------------

def merge_channels(dapi_img: np.ndarray, her2_img: np.ndarray) -> np.ndarray:
    """
    Stack DAPI (ch0) and HER2 (ch1) into (2, H, W).
    If shapes differ, crop both to the minimum common size.
    """
    if dapi_img.shape != her2_img.shape:
        min_h = min(dapi_img.shape[0], her2_img.shape[0])
        min_w = min(dapi_img.shape[1], her2_img.shape[1])
        warnings.warn(
            f"Shape mismatch: DAPI {dapi_img.shape} vs HER2 {her2_img.shape}. "
            f"Cropping both to ({min_h}, {min_w})."
        )
        dapi_img = dapi_img[:min_h, :min_w]
        her2_img = her2_img[:min_h, :min_w]
    return np.stack([dapi_img.astype(np.float32),
                     her2_img.astype(np.float32)], axis=0)


# ---------------------------------------------------------------------------
# Step 2 – Segment cytoplasm with CPSAM
# ---------------------------------------------------------------------------

def segment_cytoplasm(
    img_2ch: np.ndarray,
    model_path: str,
    diameter: int = 30,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    channels: tuple = (0, 1),
) -> np.ndarray:
    """
    Run CPSAM (CellPose) cytoplasm segmentation.
    """
    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    masks, _, _ = model.eval(
        img_2ch,
        channels=list(channels),
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return masks.astype(np.int32)


# ---------------------------------------------------------------------------
# Step 3 – Find nucleus inside each cytoplasm region
# ---------------------------------------------------------------------------

NUCLEI_V3_ENV = r"D:\Miniconda3\envs\cellpose_nuclei\python.exe"
NUCLEI_V3_SCRIPT = str(Path(__file__).parent / "detect_nuclei_v3.py")
PY_UNBUFFERED = "-u"


def _find_nuclei_via_subprocess(
    cyto_masks: np.ndarray,
    dapi_img: np.ndarray,
    min_nuc_area: int,
    max_area_ratio: float,
    nuclei_diameter,
) -> np.ndarray:
    """
    Call the v3 environment (which has the official nuclei model) via subprocess.
    """
    import tempfile
    import subprocess
    import shutil

    tmp = Path(tempfile.gettempdir()) / "cellpose_nuclei_tmp"
    tmp.mkdir(exist_ok=True)

    cyto_path = tmp / "cyto_masks.tif"
    dapi_path = tmp / "dapi_img.tif"
    output_path = tmp / "nuclei_masks.npy"

    tifffile.imwrite(str(cyto_path), cyto_masks.astype(np.int32))
    tifffile.imwrite(str(dapi_path), dapi_img.astype(np.float32))

    cmd = [
        NUCLEI_V3_ENV,
        PY_UNBUFFERED,
        NUCLEI_V3_SCRIPT,
        "--cyto-masks", str(cyto_path),
        "--dapi-img", str(dapi_path),
        "--output-npy", str(output_path),
        "--min-nuc-area", str(min_nuc_area),
        "--max-area-ratio", str(max_area_ratio),
    ]
    if nuclei_diameter is not None:
        cmd.extend(["--nuclei-diameter", str(nuclei_diameter)])

    print("    Calling Cellpose v3 nuclei model via subprocess …")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"    STDERR: {result.stderr[-500:]}")
        raise RuntimeError(f"v3 nuclei subprocess failed (code {result.returncode})")

    nuclei_masks = np.load(str(output_path)).astype(np.int32)

    n_found = int(np.sum(np.unique(nuclei_masks) > 0))
    n_total = int(np.sum(np.unique(cyto_masks) > 0))
    print(f"    Cellpose v3 nuclei: {n_found}/{n_total} found")

    shutil.rmtree(tmp, ignore_errors=True)
    return nuclei_masks


def _find_nuclei_otsu(
    cyto_masks: np.ndarray,
    dapi_img: np.ndarray,
    min_nuc_area: int,
    max_area_ratio: float,
) -> np.ndarray:
    """Pure-Otsu nucleus detection (runs entirely in-process, no v3 needed)."""
    nuclei_masks = np.zeros_like(cyto_masks, dtype=np.int32)
    unique_labels = np.unique(cyto_masks)
    unique_labels = unique_labels[unique_labels > 0]

    found = 0
    for label in unique_labels:
        cyto_region = (cyto_masks == label)
        dapi_in_cyto = dapi_img.copy()
        dapi_in_cyto[~cyto_region] = 0
        nonzero = dapi_in_cyto[cyto_region]
        if nonzero.size == 0:
            continue

        thr = skimage.filters.threshold_otsu(nonzero)
        binary = (dapi_in_cyto > thr).astype(np.int32)

        labeled, num = skimage.measure.label(
            binary, return_num=True, connectivity=2
        )
        if num == 0:
            continue

        props = skimage.measure.regionprops(labeled)
        best_prop = max(props, key=lambda p: p.area)

        nuc_area = best_prop.area
        cyto_area = int(np.sum(cyto_region))

        if nuc_area < min_nuc_area:
            continue
        if nuc_area > cyto_area * max_area_ratio:
            continue

        nuclei_masks[labeled == best_prop.label] = label
        found += 1

    n_total = int(np.sum(np.unique(cyto_masks) > 0))
    print(f"    Otsu: {found}/{n_total} found")
    return nuclei_masks


def find_nucleus_in_cytoplasm(
    cyto_masks: np.ndarray,
    dapi_img: np.ndarray,
    min_nuc_area: int = 30,
    max_area_ratio: float = 0.8,
    use_cellpose_nuclei: bool = True,
    nuclei_diameter=None,
) -> np.ndarray:
    """
    For each cytoplasm region, locate the nucleus using Cellpose nuclei model
    (or Otsu fallback) on the DAPI channel restricted to that region.
    """
    if use_cellpose_nuclei:
        return _find_nuclei_via_subprocess(
            cyto_masks, dapi_img, min_nuc_area, max_area_ratio, nuclei_diameter
        )
    return _find_nuclei_otsu(cyto_masks, dapi_img, min_nuc_area, max_area_ratio)


# ---------------------------------------------------------------------------
# Step 4 – Align labels (cytoplasm + nucleus → complete cell)
# ---------------------------------------------------------------------------

def align_labels(
    cyto_masks: np.ndarray,
    nuclei_masks: np.ndarray,
) -> np.ndarray:
    """
    Build a complete-cell mask by overlaying nucleus and cytoplasm.
    """
    cell_masks = cyto_masks.copy()
    nucleus_pixels = nuclei_masks > 0
    cell_masks[nucleus_pixels] = nuclei_masks[nucleus_pixels]
    return cell_masks


# ---------------------------------------------------------------------------
# Step 5 – Feature extraction (5 channels)
# ---------------------------------------------------------------------------

def extract_features(
    cyto_masks: np.ndarray,
    nuclei_masks: np.ndarray,
    cell_masks: np.ndarray,
    all_channels: dict,
) -> pd.DataFrame:
    """
    Extract per-cell morphological and intensity features for ALL channels.

    Parameters
    ----------
    cyto_masks : int32 ndarray
    nuclei_masks : int32 ndarray
    cell_masks : int32 ndarray
    all_channels : dict
        {"DAPI": np.ndarray, "HER2": np.ndarray, "PR": ..., "ER": ..., "Ki67": ...}
    """
    records = []
    unique_labels = np.unique(cyto_masks)
    unique_labels = unique_labels[unique_labels > 0]

    channel_names = list(all_channels.keys())

    for label in unique_labels:
        cyto_region = cyto_masks == label
        nuc_region = nuclei_masks == label
        cyto_only_region = cyto_region & ~nuc_region

        has_nucleus = np.any(nuc_region)
        nuc_area = int(np.sum(nuc_region)) if has_nucleus else 0
        cyto_area = int(np.sum(cyto_region))
        cyto_only_area = int(np.sum(cyto_only_region))

        # ── Morphological features ──
        if has_nucleus:
            nuc_props = skimage.measure.regionprops(
                (nuclei_masks == label).astype(np.uint8)
            )[0]
            nuc_major = nuc_props.major_axis_length
            nuc_minor = nuc_props.minor_axis_length
            nuc_aspect_ratio = round(nuc_major / (nuc_minor + 1e-8), 4)
            nuc_eccentricity = round(nuc_props.eccentricity, 4)
            nuc_centroid_y = round(nuc_props.centroid[0], 2)
            nuc_centroid_x = round(nuc_props.centroid[1], 2)
        else:
            nuc_aspect_ratio = np.nan
            nuc_eccentricity = np.nan
            nuc_centroid_y = np.nan
            nuc_centroid_x = np.nan

        # Cell-level morphological features
        cell_props = skimage.measure.regionprops(
            (cell_masks == label).astype(np.uint8)
        )
        if cell_props:
            cell_area = cell_props[0].area
            cell_eccentricity = round(cell_props[0].eccentricity, 4)
            cell_centroid_y = round(cell_props[0].centroid[0], 2)
            cell_centroid_x = round(cell_props[0].centroid[1], 2)
        else:
            cell_area = cyto_area + nuc_area
            cell_eccentricity = np.nan
            cell_centroid_y = np.nan
            cell_centroid_x = np.nan

        # ── Intensity features for EVERY channel ──
        row = {
            "cell_label": label,
            "nuc_area": nuc_area,
            "cyto_area": cyto_area,
            "cyto_only_area": cyto_only_area,
            "cell_area": cell_area,
            "nuc_aspect_ratio": nuc_aspect_ratio,
            "nuc_eccentricity": nuc_eccentricity,
            "nuc_centroid_y": nuc_centroid_y,
            "nuc_centroid_x": nuc_centroid_x,
            "cell_eccentricity": cell_eccentricity,
            "cell_centroid_y": cell_centroid_y,
            "cell_centroid_x": cell_centroid_x,
            "has_nucleus": has_nucleus,
        }

        for ch_name, ch_img in all_channels.items():
            # Nucleus intensity
            if has_nucleus:
                nuc_pixels = ch_img[nuc_region]
                row[f"{ch_name}_nuc_mean"] = round(float(np.mean(nuc_pixels)), 4)
                row[f"{ch_name}_nuc_std"] = round(float(np.std(nuc_pixels)), 4)
                row[f"{ch_name}_nuc_median"] = round(float(np.median(nuc_pixels)), 4)
                row[f"{ch_name}_nuc_sum"] = round(float(np.sum(nuc_pixels)), 4)
            else:
                row[f"{ch_name}_nuc_mean"] = np.nan
                row[f"{ch_name}_nuc_std"] = np.nan
                row[f"{ch_name}_nuc_median"] = np.nan
                row[f"{ch_name}_nuc_sum"] = np.nan

            # Cytoplasm intensity (full cytoplasm including nucleus)
            if cyto_area > 0:
                cyto_pixels = ch_img[cyto_region]
                row[f"{ch_name}_cyto_mean"] = round(float(np.mean(cyto_pixels)), 4)
                row[f"{ch_name}_cyto_std"] = round(float(np.std(cyto_pixels)), 4)
            else:
                row[f"{ch_name}_cyto_mean"] = np.nan
                row[f"{ch_name}_cyto_std"] = np.nan

            # Cytoplasm-only intensity (excluding nucleus)
            if cyto_only_area > 0:
                cyto_only_pixels = ch_img[cyto_only_region]
                row[f"{ch_name}_cyto_only_mean"] = round(float(np.mean(cyto_only_pixels)), 4)
                row[f"{ch_name}_cyto_only_std"] = round(float(np.std(cyto_only_pixels)), 4)
            else:
                row[f"{ch_name}_cyto_only_mean"] = np.nan
                row[f"{ch_name}_cyto_only_std"] = np.nan

            # Net: cytoplasm_only_mean - nucleus_mean
            if has_nucleus and cyto_only_area > 0:
                net = row[f"{ch_name}_cyto_only_mean"] - row[f"{ch_name}_nuc_mean"]
                row[f"{ch_name}_net"] = round(float(net), 4)
            else:
                row[f"{ch_name}_net"] = np.nan

            # Nucleus-to-cytoplasm ratio
            if has_nucleus and cyto_only_area > 0:
                nuc_m = row[f"{ch_name}_nuc_mean"]
                cyto_m = row[f"{ch_name}_cyto_only_mean"]
                if cyto_m > 1e-8:
                    row[f"{ch_name}_nuc_cyto_ratio"] = round(nuc_m / cyto_m, 4)
                else:
                    row[f"{ch_name}_nuc_cyto_ratio"] = np.nan
            else:
                row[f"{ch_name}_nuc_cyto_ratio"] = np.nan

        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Step 6 – Visualization (5-channel)
# ---------------------------------------------------------------------------

def create_overlay(
    all_channels: dict,
    cyto_masks: np.ndarray,
    nuclei_masks: np.ndarray,
    output_path: str,
) -> None:
    """
    Generate overlay figures:
        1) all_channels.png  – each channel as greyscale
        2) dapi_cyto_nuc.png – DAPI + cytoplasm(green) + nucleus(red)
        3) dapi_markers.png  – DAPI(blue) + each marker
        4) overlay.png       – random colour per cell
    """
    def norm(x):
        p2, p98 = np.nanpercentile(x, (1, 99))
        if p98 - p2 < 1e-8:
            return np.zeros_like(x)
        return np.clip((x - p2) / (p98 - p2 + 1e-8), 0, 1)

    channel_names = list(all_channels.keys())
    n_channels = len(channel_names)
    dapi_img = all_channels["DAPI"]

    # ── Figure 1: Greyscale per channel ──
    fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 5))
    fig.suptitle(f"{Path(output_path).stem} – All Channels", fontsize=14)

    cmaps = {"DAPI": "gray", "HER2": "magma", "PR": "viridis", "ER": "inferno", "Ki67": "plasma"}
    for i, name in enumerate(channel_names):
        axes[i].imshow(norm(all_channels[name]), cmap=cmaps.get(name, "gray"))
        axes[i].set_title(name, fontsize=12)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_path.replace("_overlay.png", "_all_channels.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 2: DAPI + cytoplasm (green) + nucleus (red) ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    overlay = np.zeros((*dapi_img.shape, 3))
    overlay[:, :, 0] = norm(dapi_img)
    overlay[:, :, 1] = norm(dapi_img)
    overlay[:, :, 2] = norm(dapi_img)
    mask_cyto = cyto_masks > 0
    mask_nuc = nuclei_masks > 0
    overlay[mask_cyto, 1] = 0.6  # green
    overlay[mask_nuc, 0] = 1.0   # red
    overlay[mask_nuc, 1] = 0.0
    overlay[mask_nuc, 2] = 0.0
    ax.imshow(overlay)
    ax.set_title("DAPI + Cyto (green) + Nuc (red)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path.replace("_overlay.png", "_dapi_cyto_nuc.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 3: DAPI + each marker channel ──
    marker_channels = [ch for ch in channel_names if ch != "DAPI"]
    n_markers = len(marker_channels)
    fig, axes = plt.subplots(1, n_markers, figsize=(6 * n_markers, 6))
    if n_markers == 1:
        axes = [axes]
    fig.suptitle(f"{Path(output_path).stem} – DAPI + Markers", fontsize=14)

    marker_colors = {"HER2": 1, "PR": 0, "ER": 0, "Ki67": 0}  # green or red
    for i, name in enumerate(marker_channels):
        ov = np.zeros((*dapi_img.shape, 3))
        ov[:, :, 2] = norm(dapi_img) * 0.6  # blue = DAPI
        color_idx = marker_colors.get(name, 0)
        ov[:, :, color_idx] = norm(all_channels[name])  # marker
        axes[i].imshow(np.clip(ov, 0, 1))
        axes[i].set_title(f"DAPI (blue) + {name}", fontsize=12)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_path.replace("_overlay.png", "_dapi_markers.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 4: Cell masks with random colors ──
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(norm(dapi_img), cmap="gray", alpha=0.4)
    n_labels = min(np.max(cyto_masks) + 1, 256)
    base = plt.cm.get_cmap("tab20", 20)(np.arange(20))[:, :3]
    extra = np.random.RandomState(42).rand(256 - 20, 3) * 0.6 + 0.4
    colour_map = np.vstack([[0, 0, 0], base, extra])
    label_cmap = ListedColormap(colour_map[:n_labels])
    colored = label_cmap(np.clip(cyto_masks, 0, n_labels - 1))
    ax.imshow(colored, alpha=0.6)
    n_cells = len(np.unique(cyto_masks)) - 1
    ax.set_title(f"Cell masks ({n_cells} cells)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Step 7 – Save results
# ---------------------------------------------------------------------------

def save_results(
    output_dir: str,
    block_name: str,
    cyto_masks: np.ndarray,
    nuclei_masks: np.ndarray,
    cell_masks: np.ndarray,
    features_df: pd.DataFrame,
    all_channels: dict,
) -> None:
    """Save all output files to disk."""
    out(parents=True, exist_ok=True)

    prefix = out / block_name

    # TIFF masks
    save_t = Path(output_dir) / block_name
    out.mkdiriff(f"{prefix}_cyto_masks.tif", cyto_masks)
    save_tiff(f"{prefix}_nuclei_masks.tif", nuclei_masks)
    save_tiff(f"{prefix}_cell_masks.tif", cell_masks)

    # CSV
    features_df.to_csv(f"{prefix}_features.csv", index=False)

    # Overlays
    create_overlay(
        all_channels, cyto_masks, nuclei_masks,
        f"{prefix}_overlay.png"
    )

    print(f"\nResults saved to: {out}")
    print(f"  {block_name}_cyto_masks.tif      – {len(np.unique(cyto_masks)) - 1} cytoplasm regions")
    print(f"  {block_name}_nuclei_masks.tif     – {len(np.unique(nuclei_masks[nuclei_masks > 0]))} nuclei")
    print(f"  {block_name}_cell_masks.tif       – {len(np.unique(cell_masks)) - 1} complete cells")
    print(f"  {block_name}_features.csv         – {len(features_df)} rows × {len(features_df.columns)} columns")
    print(f"  {block_name}_overlay.png")
    print(f"  {block_name}_all_channels.png")
    print(f"  {block_name}_dapi_cyto_nuc.png")
    print(f"  {block_name}_dapi_markers.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    # 如果只传了一个参数（block name），自动拼路径
    if len(sys.argv) == 2:
        BLOCK = sys.argv[1]
        base = Path(r"d:\Try_munan\FYP_LAST\results\registered") / BLOCK
        model_path = r"D:\Try_munan\Cellpose_model\model2\models\her2_wholecell_v3"

        return argparse.Namespace(
            dapi=str(base / f"{BLOCK}_Cycle1_DAPI.tif"),
            her2=str(base / f"{BLOCK}_Cycle1_HER2_aligned.tif"),
            pr=str(base / f"{BLOCK}_Cycle1_PR_aligned.tif"),
            er=str(base / f"{BLOCK}_Cycle1_ER_aligned.tif"),
            ki67=str(base / f"{BLOCK}_KI67_aligned.tif"),
            model=model_path,
            block_name=BLOCK,
            output_dir=str(Path(r"d:\Try_munan\FYP_LAST\results\segmentation")),
            diameter=30,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            channels="0,1",
            min_nuc_area=30,
            max_area_ratio=0.8,
            use_cellpose_nuclei=True,
            nuclei_diameter=None,
        )

    # 否则走正常 argparse
    p = argparse.ArgumentParser(
        description="CPSAM cytoplasm-to-nucleus cell segmentation (5-channel)"
    )
    p.add_argument("--dapi", required=True)
    p.add_argument("--her2", required=True)
    p.add_argument("--pr", required=True)
    p.add_argument("--er", required=True)
    p.add_argument("--ki67", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--block-name", required=True)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--diameter", type=int, default=30)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument("--channels", default="0,1")
    p.add_argument("--min-nuc-area", type=int, default=30)
    p.add_argument("--max-area-ratio", type=float, default=0.8)
    p.add_argument("--use-cellpose-nuclei", action="store_true", default=True)
    p.add_argument("--no-cellpose-nuclei", dest="use_cellpose_nuclei", action="store_false")
    p.add_argument("--nuclei-diameter", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("CPSAM Cytoplasm-to-Nucleus Segmentation (5-channel)")
    print("=" * 60)

    # Parse channels
    ch0, ch1 = map(int, args.channels.split(","))

    # 1. Load ALL 5 channels
    print("\n[1/7] Loading images …")
    dapi_img = load_channel(args.dapi)
    her2_img = load_channel(args.her2)
    pr_img   = load_channel(args.pr)
    er_img   = load_channel(args.er)
    ki67_img = load_channel(args.ki67)
    print(f"    DAPI shape: {dapi_img.shape}")
    print(f"    HER2 shape: {her2_img.shape}")
    print(f"    PR   shape: {pr_img.shape}")
    print(f"    ER   shape: {er_img.shape}")
    print(f"    Ki67 shape: {ki67_img.shape}")

    # 2. Merge channels (only DAPI + HER2 for CPSAM)
    print("\n[2/7] Merging channels for CPSAM (DAPI + HER2) …")
    img_2ch = merge_channels(dapi_img, her2_img)
    merged_h, merged_w = img_2ch.shape[1], img_2ch.shape[2]

    # Sync crop ALL channels to match merged shape
    dapi_img = dapi_img[:merged_h, :merged_w]
    her2_img = her2_img[:merged_h, :merged_w]
    pr_img   = pr_img[:merged_h, :merged_w]
    er_img   = er_img[:merged_h, :merged_w]
    ki67_img = ki67_img[:merged_h, :merged_w]
    print(f"    Merged shape: {img_2ch.shape}  (ch0=DAPI, ch1=HER2)")
    print(f"    All channels cropped to: ({merged_h}, {merged_w})")

    # 3. Segment cytoplasm
    print("\n[3/7] Running CPSAM cytoplasm segmentation …")
    cyto_masks = segment_cytoplasm(
        img_2ch,
        args.model,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        channels=(ch0, ch1),
    )
    n_cyto = len(np.unique(cyto_masks)) - 1
    print(f"    Found {n_cyto} cytoplasm regions")

    # 4. Find nuclei
    nuc_method = "Cellpose nuclei" if args.use_cellpose_nuclei else "Otsu"
    print(f"\n[4/7] Finding nuclei inside cytoplasm ({nuc_method}) …")
    nuclei_masks = find_nucleus_in_cytoplasm(
        cyto_masks, dapi_img,
        min_nuc_area=args.min_nuc_area,
        max_area_ratio=args.max_area_ratio,
        use_cellpose_nuclei=args.use_cellpose_nuclei,
        nuclei_diameter=args.nuclei_diameter,
    )
    n_nuc = len(np.unique(nuclei_masks[nuclei_masks > 0]))
    print(f"    Found {n_nuc} nuclei")

    # 5. Align labels
    print("\n[5/7] Aligning labels …")
    cell_masks = align_labels(cyto_masks, nuclei_masks)
    n_cells = len(np.unique(cell_masks)) - 1
    print(f"    {n_cells} complete cells")

    # 6. Extract features from ALL 5 channels
    print("\n[6/7] Extracting features from all 5 channels …")
    all_channels = {
        "DAPI":  dapi_img,
        "HER2":  her2_img,
        "PR":    pr_img,
        "ER":    er_img,
        "Ki67":  ki67_img,
    }
    features_df = extract_features(
        cyto_masks, nuclei_masks, cell_masks,
        all_channels,
    )
    print(f"    {len(features_df)} cells × {len(features_df.columns)} features")
    if len(features_df) > 0 and "has_nucleus" in features_df.columns:
        no_nuc = (~features_df["has_nucleus"]).sum()
        if no_nuc > 0:
            print(f"    WARNING: {no_nuc} cells have no nucleus")
    else:
        print("    WARNING: No cells detected")

    # 7. Save
    print("\n[7/7] Saving results …")
    save_results(
        args.output_dir, args.block_name,
        cyto_masks, nuclei_masks, cell_masks,
        features_df, all_channels,
    )

    print("\nDone.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
