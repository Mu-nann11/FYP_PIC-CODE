#!/usr/bin/env python3
"""
CPSAM + v3 Nuclei Cell Segmentation Pipeline (5-channel) – FAST VERSION

Workflow:
    Step 1: v3 nuclei model → whole DAPI → nuclei_masks (1 GPU call)
    Step 2: CPSAM → DAPI + HER2 → cyto_masks (1 GPU call)
    Step 3: Match nuclei labels to cytoplasm labels
    Step 4: Align labels → cell_masks (only cells with nucleus)
    Step 5: Extract features from all 5 channels (ROI-based)
    Step 6: Save TIFF masks + CSV + PNG overlay

Supports checkpoint: each step saves immediately, next run skips cached results.
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
from scipy import ndimage
from cellpose import models
from tqdm import tqdm


# =====================================================================
# 路径配置
# =====================================================================

REGISTERED_DIR = Path(r"d:\Try_munan\FYP_LAST\results\registered")
SEGMENTATION_DIR = Path(r"d:\Try_munan\FYP_LAST\results\segmentation")
MODEL_PATH = r"D:\Try_munan\Cellpose_model\model2\models\her2_wholecell_v3"


def auto_paths(block):
    reg = REGISTERED_DIR / block
    return {
        "dapi":  str(reg / f"{block}_Cycle1_DAPI.tif"),
        "her2":  str(reg / f"{block}_Cycle1_HER2_aligned.tif"),
        "pr":    str(reg / f"{block}_Cycle1_PR_aligned.tif"),
        "er":    str(reg / f"{block}_Cycle1_ER_aligned.tif"),
        "ki67":  str(reg / f"{block}_KI67_aligned.tif"),
    }


# =====================================================================
# Image I/O
# =====================================================================

def load_channel(path):
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        if img.shape[0] in (3, 4):
            img = img[0]
        elif img.shape[2] in (3, 4):
            img = img[:, :, 0]
    return img.astype(np.float32)


def save_tiff(path, data):
    tifffile.imwrite(str(path), data.astype(np.int32), photometric="minisblack")


def load_tiff_int32(path):
    return tifffile.imread(str(path)).astype(np.int32)


# =====================================================================
# Step 1 – 整图核检测（v3 nuclei model，一次 GPU 调用）
# =====================================================================

def segment_nuclei_whole(dapi_img, diameter=30):
    print("    Loading Cellpose nuclei model …", flush=True)
    nuc_model = models.CellposeModel(gpu=True, model_type='nuclei')
    print("    Running nuclei segmentation on whole image …", flush=True)
    masks, _, _ = nuc_model.eval(
        dapi_img, diameter=diameter,
        channels=[0, 0], flow_threshold=0.4, cellprob_threshold=0.0,
    )
    return masks.astype(np.int32)


# =====================================================================
# Step 2 – CPSAM 细胞质分割
# =====================================================================

def segment_cytoplasm(img_2ch, model_path, diameter=30, flow_threshold=0.4,
                      cellprob_threshold=0.0, channels=(0, 1)):
    print("    Loading CPSAM model …", flush=True)
    cyto_model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    print("    Running cytoplasm segmentation …", flush=True)
    masks, _, _ = cyto_model.eval(
        img_2ch, channels=list(channels), diameter=diameter,
        flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
    )
    return masks.astype(np.int32)


# =====================================================================
# Step 3 – 核 label 匹配到细胞质 label（SciPy加速版本）
# =====================================================================

def match_nuclei_to_cyto(nuclei_masks, cyto_masks, min_nuc_area=30, max_area_ratio=0.8):
    """
    使用SciPy ndimage优化的核匹配算法
    相比原始版本快 2-3 倍
    """
    nuclei_matched = np.zeros_like(cyto_masks, dtype=np.int32)
    cyto_labels = np.unique(cyto_masks)
    cyto_labels = cyto_labels[cyto_labels > 0]
    
    # 预计算核的面积（使用ndimage，比循环快）
    nuclei_labels = np.unique(nuclei_masks)
    nuclei_labels = nuclei_labels[nuclei_labels > 0]
    nuc_areas = ndimage.sum(np.ones_like(nuclei_masks, dtype=np.int32), 
                             nuclei_masks, nuclei_labels)
    
    # 构建核标签到面积的映射表
    nuc_area_dict = dict(zip(nuclei_labels, nuc_areas))
    
    found = 0
    for cyto_label in tqdm(cyto_labels, desc="Matching nuclei (SciPy)", unit="cell"):
        cyto_region = cyto_masks == cyto_label
        cyto_area = int(np.sum(cyto_region))
        
        # 获取细胞区域内的核标签（只查询该区域）
        nuclei_in_cyto = nuclei_masks[cyto_region]
        nuc_labels_in_region = np.unique(nuclei_in_cyto)
        nuc_labels_in_region = nuc_labels_in_region[nuc_labels_in_region > 0]
        
        if len(nuc_labels_in_region) == 0:
            continue
        
        # 计算每个核与细胞的重叠（只在细胞范围内计算，大幅加速）
        best_label = None
        best_overlap = 0
        for nuc_label in nuc_labels_in_region:
            overlap = int(np.sum((nuclei_masks == nuc_label) & cyto_region))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = nuc_label
        
        if best_label is None:
            continue
        
        nuc_area = nuc_area_dict[best_label]
        if nuc_area < min_nuc_area:
            continue
        if nuc_area > cyto_area * max_area_ratio:
            continue
        
        nuclei_matched[nuclei_masks == best_label] = cyto_label
        found += 1
    
    print(f"    Matched: {found}/{len(cyto_labels)} cytoplasm regions have nuclei")
    return nuclei_matched


# =====================================================================
# Step 4 – Align labels（只保留有核的细胞）
# =====================================================================

def align_labels(cyto_masks, nuclei_masks):
    """
    只保留同时有胞质和核的细胞：
      - 用 cyto label 作为 cell label
      - 没有匹配到核的 cyto region 直接丢弃
    """
    cell_masks = np.zeros_like(cyto_masks, dtype=np.int32)

    # 找出所有有核的 cyto label
    nuc_labels = np.unique(nuclei_masks)
    nuc_labels = nuc_labels[nuc_labels > 0]

    for label in nuc_labels:
        cell_masks[cyto_masks == label] = label

    return cell_masks


# =====================================================================
# Membrane ring calculation (for HER2 membrane intensity)
# =====================================================================

def compute_membrane_ring(cyto_region, nuc_region, ring_width=2):
    """
    计算膜环区域（细胞膜附近的像素）
    
    膜环定义：从细胞外边界向内 ring_width 像素的区域，
    不包括细胞核内部。
    
    Args:
        cyto_region: bool array, shape (roi_h, roi_w)
        nuc_region: bool array, same shape
        ring_width: 膜环宽度（像素），default=2
    
    Returns:
        bool array: 膜环区域（True 表示膜环像素）
    """
    # 细胞的内侧边界（向内缩小 ring_width 像素）
    cyto_eroded = ndimage.binary_erosion(cyto_region, iterations=ring_width)
    
    # 膜环 = 细胞区域 - 内侧边界
    membrane = cyto_region & ~cyto_eroded
    
    # 排除核内像素
    membrane = membrane & ~nuc_region
    
    return membrane


# =====================================================================
# Step 5 – Feature extraction (ROI-based, fast)
# =====================================================================

def extract_features(cyto_masks, nuclei_masks, cell_masks, all_channels):
    records = []

    # 只提取有核的细胞
    valid_labels = np.unique(nuclei_masks)
    valid_labels = valid_labels[valid_labels > 0]

    for label in tqdm(valid_labels, desc="Extracting features", unit="cell"):
        # ── 裁 ROI ──
        region = cyto_masks == label
        rows = np.any(region, axis=1)
        cols = np.any(region, axis=0)
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]

        margin = 5
        y0 = max(0, y0 - margin)
        y1 = min(cyto_masks.shape[0], y1 + margin + 1)
        x0 = max(0, x0 - margin)
        x1 = min(cyto_masks.shape[1], x1 + margin + 1)

        roi_cyto = cyto_masks[y0:y1, x0:x1]
        roi_nuc = nuclei_masks[y0:y1, x0:x1]
        roi_cell = cell_masks[y0:y1, x0:x1]

        cyto_region = roi_cyto == label
        nuc_region = roi_nuc == label
        cyto_only_region = cyto_region & ~nuc_region

        has_nucleus = np.any(nuc_region)
        nuc_area = int(np.sum(nuc_region)) if has_nucleus else 0
        cyto_area = int(np.sum(cyto_region))
        cyto_only_area = int(np.sum(cyto_only_region))

        # ── Morphological features ──
        if has_nucleus:
            nuc_props = skimage.measure.regionprops(
                (roi_nuc == label).astype(np.uint8)
            )[0]
            nuc_aspect_ratio = round(
                nuc_props.major_axis_length / (nuc_props.minor_axis_length + 1e-8), 4
            )
            nuc_eccentricity = round(nuc_props.eccentricity, 4)
            nuc_centroid_y = round(nuc_props.centroid[0] + y0, 2)
            nuc_centroid_x = round(nuc_props.centroid[1] + x0, 2)
        else:
            nuc_aspect_ratio = np.nan
            nuc_eccentricity = np.nan
            nuc_centroid_y = np.nan
            nuc_centroid_x = np.nan

        cell_props = skimage.measure.regionprops(
            (roi_cell == label).astype(np.uint8)
        )
        if cell_props:
            cell_area = cell_props[0].area
            cell_eccentricity = round(cell_props[0].eccentricity, 4)
            cell_centroid_y = round(cell_props[0].centroid[0] + y0, 2)
            cell_centroid_x = round(cell_props[0].centroid[1] + x0, 2)
        else:
            cell_area = cyto_area + nuc_area
            cell_eccentricity = np.nan
            cell_centroid_y = np.nan
            cell_centroid_x = np.nan

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

        # ── Intensity features for every channel ──
        for ch_name, ch_img in all_channels.items():
            roi_ch = ch_img[y0:y1, x0:x1]

            if has_nucleus:
                nuc_pixels = roi_ch[nuc_region]
                row[f"{ch_name}_nuc_mean"] = round(float(np.mean(nuc_pixels)), 4)
                row[f"{ch_name}_nuc_std"] = round(float(np.std(nuc_pixels)), 4)
                row[f"{ch_name}_nuc_median"] = round(float(np.median(nuc_pixels)), 4)
                row[f"{ch_name}_nuc_sum"] = round(float(np.sum(nuc_pixels)), 4)
            else:
                row[f"{ch_name}_nuc_mean"] = np.nan
                row[f"{ch_name}_nuc_std"] = np.nan
                row[f"{ch_name}_nuc_median"] = np.nan
                row[f"{ch_name}_nuc_sum"] = np.nan

            if cyto_area > 0:
                cyto_pixels = roi_ch[cyto_region]
                row[f"{ch_name}_cyto_mean"] = round(float(np.mean(cyto_pixels)), 4)
                row[f"{ch_name}_cyto_std"] = round(float(np.std(cyto_pixels)), 4)
            else:
                row[f"{ch_name}_cyto_mean"] = np.nan
                row[f"{ch_name}_cyto_std"] = np.nan

            if cyto_only_area > 0:
                cyto_only_pixels = roi_ch[cyto_only_region]
                row[f"{ch_name}_cyto_only_mean"] = round(float(np.mean(cyto_only_pixels)), 4)
                row[f"{ch_name}_cyto_only_std"] = round(float(np.std(cyto_only_pixels)), 4)
            else:
                row[f"{ch_name}_cyto_only_mean"] = np.nan
                row[f"{ch_name}_cyto_only_std"] = np.nan

            if has_nucleus and cyto_only_area > 0:
                net = row[f"{ch_name}_cyto_only_mean"] - row[f"{ch_name}_nuc_mean"]
                row[f"{ch_name}_net"] = round(float(net), 4)
                cyto_m = row[f"{ch_name}_cyto_only_mean"]
                if cyto_m > 1e-8:
                    row[f"{ch_name}_nuc_cyto_ratio"] = round(
                        row[f"{ch_name}_nuc_mean"] / cyto_m, 4
                    )
                else:
                    row[f"{ch_name}_nuc_cyto_ratio"] = np.nan
            else:
                row[f"{ch_name}_net"] = np.nan
                row[f"{ch_name}_nuc_cyto_ratio"] = np.nan

            # 新增：HER2 膜环特征（仅对 HER2 通道）
            if ch_name == "HER2":
                membrane_region = compute_membrane_ring(cyto_region, nuc_region, ring_width=2)
                if np.any(membrane_region):
                    membrane_pixels = roi_ch[membrane_region]
                    row["HER2_membrane_ring_mean"] = round(float(np.mean(membrane_pixels)), 4)
                    row["HER2_membrane_ring_std"] = round(float(np.std(membrane_pixels)), 4)
                else:
                    row["HER2_membrane_ring_mean"] = np.nan
                    row["HER2_membrane_ring_std"] = np.nan

        records.append(row)

    return pd.DataFrame(records)


# =====================================================================
# Visualization helpers
# =====================================================================

def norm_img(x):
    p2, p98 = np.nanpercentile(x, (1, 99))
    if p98 - p2 < 1e-8:
        return np.zeros_like(x)
    return np.clip((x - p2) / (p98 - p2 + 1e-8), 0, 1)


def save_mask_png(mask, output_path, title, colormap="tab20"):
    """将 label mask 保存为彩色 PNG"""
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    n_labels = int(np.max(mask))
    if n_labels == 0:
        ax.imshow(mask, cmap="gray")
    else:
        base = plt.cm.get_cmap(colormap, 20)(np.arange(20))[:, :3]
        if n_labels > 20:
            extra = np.random.RandomState(42).rand(n_labels - 19, 3) * 0.6 + 0.4
            colors = np.vstack([[0, 0, 0], base, extra])
        else:
            colors = np.vstack([[0, 0, 0], base[:n_labels]])
        cmap = ListedColormap(colors[:n_labels + 1])
        ax.imshow(mask, cmap=cmap)

    n_cells = len(np.unique(mask)) - 1
    ax.set_title(f"{title} ({n_cells} regions)", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path}")


def save_nucleus_cytoplasm_overlay(dapi_img, cyto_masks, nuclei_masks, output_path):
    """
    一张图看清每个细胞的核和质：
      - 背景：DAPI 灰度
      - 胞质：半透明随机色（每个细胞不同颜色）
      - 核：  统一蓝色
    """
    from matplotlib.colors import ListedColormap
    from skimage.segmentation import find_boundaries
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # ── 背景 DAPI ──
    dapi_n = norm_img(dapi_img)
    ax.imshow(dapi_n, cmap="gray")

    # ── 胞质着色（每个 cell 一个随机颜色）──
    n_labels = int(np.max(cyto_masks))
    if n_labels > 0:
        rng = np.random.RandomState(42)
        colors = rng.rand(n_labels + 1, 4)
        colors[0] = [0, 0, 0, 0]
        colors[:, 3] = 0.35
        cyto_cmap = ListedColormap(colors)
        colored_cyto = cyto_cmap(cyto_masks)
        ax.imshow(colored_cyto)

    # ── 核着色（统一蓝色）──
    nuc_overlay = np.zeros((*dapi_img.shape, 4))
    nuc_region = nuclei_masks > 0
    nuc_overlay[nuc_region] = [0.15, 0.35, 1.0, 0.7]
    ax.imshow(nuc_overlay)

    # ── 胞质边界线（白色）──
    cyto_boundary = find_boundaries(cyto_masks, mode="outer")
    boundary_img = np.zeros((*dapi_img.shape, 4))
    boundary_img[cyto_boundary] = [1, 1, 1, 0.6]
    ax.imshow(boundary_img)

    # ── 核边界线（亮蓝）──
    nuc_boundary = find_boundaries(nuclei_masks, mode="outer")
    nuc_bnd_img = np.zeros((*dapi_img.shape, 4))
    nuc_bnd_img[nuc_boundary] = [0.3, 0.6, 1.0, 0.9]
    ax.imshow(nuc_bnd_img)

    # ── 标注 ──
    n_cells = len(np.unique(cyto_masks)) - 1
    n_nuc = len(np.unique(nuclei_masks[nuclei_masks > 0]))
    ax.set_title(
        f"Segmentation: {n_cells} cytoplasm, {n_nuc} nuclei matched",
        fontsize=14, fontweight="bold",
    )
    ax.axis("off")

    # ── 图例 ──
    legend_elements = [
        Patch(facecolor="white", alpha=0.35, label="Cytoplasm"),
        Patch(facecolor=(0.15, 0.35, 1.0), alpha=0.7, label="Nucleus"),
        Patch(edgecolor="white", facecolor="none", label="Cyto boundary"),
        Patch(edgecolor=(0.3, 0.6, 1.0), facecolor="none", label="Nuc boundary"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10,
              framealpha=0.8, edgecolor="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {output_path}")


# =====================================================================
# TIFF overlay helpers
# =====================================================================

def save_overlay_tiff(dapi_img, cyto_masks, nuclei_masks, output_path):
    """
    保存核+质叠加的 RGB TIFF（16-bit）：
      - R 通道：胞质掩膜
      - G 通道：胞质边界
      - B 通道：核掩膜 + 核边界
    """
    from skimage.segmentation import find_boundaries

    h, w = dapi_img.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint16)

    # R: 胞质 mask
    if np.max(cyto_masks) > 0:
        overlay[:, :, 0] = (cyto_masks / np.max(cyto_masks) * 60000).astype(np.uint16)

    # G: 胞质边界
    cyto_bnd = find_boundaries(cyto_masks, mode="outer")
    overlay[cyto_bnd, 1] = 65535

    # B: 核 mask + 边界
    nuc_region = nuclei_masks > 0
    overlay[nuc_region, 2] = 50000
    nuc_bnd = find_boundaries(nuclei_masks, mode="outer")
    overlay[nuc_bnd, 2] = 65535

    tifffile.imwrite(str(output_path), overlay, photometric="rgb")
    print(f"    Saved: {output_path}")


def save_all_channels_tiff(all_channels, output_path):
    """
    保存 5 通道合成为一个 5-page TIFF，每页一个通道（16-bit）
    方便在 ImageJ 中逐通道查看
    """
    with tifffile.TiffWriter(str(output_path)) as tif:
        for name, img in all_channels.items():
            p2, p98 = np.nanpercentile(img, (1, 99))
            if p98 - p2 > 1e-8:
                normed = np.clip((img - p2) / (p98 - p2), 0, 1)
            else:
                normed = np.zeros_like(img)
            tif.write(
                (normed * 65535).astype(np.uint16),
                photometric="minisblack",
                description=name,
            )
    print(f"    Saved: {output_path}")


def save_dapi_cyto_nuc_tiff(dapi_img, cyto_masks, nuclei_masks, output_path):
    """
    保存 DAPI + 胞质(green) + 核(red) 的 RGB TIFF
    """
    def norm(x):
        p2, p98 = np.nanpercentile(x, (1, 99))
        if p98 - p2 < 1e-8:
            return np.zeros_like(x, dtype=np.float64)
        return np.clip((x - p2) / (p98 - p2), 0, 1).astype(np.float64)

    dapi_n = norm(dapi_img)
    h, w = dapi_img.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint16)

    # R: DAPI + 核高亮
    overlay[:, :, 0] = (dapi_n * 30000).astype(np.uint16)
    nuc_region = nuclei_masks > 0
    overlay[nuc_region, 0] = 65535

    # G: DAPI + 胞质高亮
    overlay[:, :, 1] = (dapi_n * 30000).astype(np.uint16)
    cyto_region = cyto_masks > 0
    overlay[cyto_region, 1] = 50000

    # B: DAPI
    overlay[:, :, 2] = (dapi_n * 65535).astype(np.uint16)

    tifffile.imwrite(str(output_path), overlay, photometric="rgb")
    print(f"    Saved: {output_path}")


# =====================================================================
# Step 6 – Visualization
# =====================================================================

def create_overlay(all_channels, cyto_masks, nuclei_masks, output_path):
    dapi_img = all_channels["DAPI"]
    dapi_n = norm_img(dapi_img)

    channel_names = list(all_channels.keys())
    n_channels = len(channel_names)

    # Figure 1: all channels greyscale
    fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 5))
    fig.suptitle(f"{Path(output_path).stem} – All Channels", fontsize=14)
    cmaps = {"DAPI": "gray", "HER2": "magma", "PR": "viridis",
             "ER": "inferno", "Ki67": "plasma"}
    for i, name in enumerate(channel_names):
        axes[i].imshow(norm_img(all_channels[name]), cmap=cmaps.get(name, "gray"))
        axes[i].set_title(name, fontsize=12)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_path.replace("_overlay.png", "_all_channels.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: DAPI + cyto(green) + nuc(red)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    overlay = np.zeros((*dapi_img.shape, 3))
    overlay[:, :, 0] = dapi_n
    overlay[:, :, 1] = dapi_n
    overlay[:, :, 2] = dapi_n
    mask_cyto = cyto_masks > 0
    mask_nuc = nuclei_masks > 0
    overlay[mask_cyto, 1] = 0.6
    overlay[mask_nuc, 0] = 1.0
    overlay[mask_nuc, 1] = 0.0
    overlay[mask_nuc, 2] = 0.0
    ax.imshow(overlay)
    ax.set_title("DAPI + Cyto (green) + Nuc (red)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path.replace("_overlay.png", "_dapi_cyto_nuc.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 3: DAPI + each marker
    marker_channels = [ch for ch in channel_names if ch != "DAPI"]
    n_markers = len(marker_channels)
    fig, axes = plt.subplots(1, n_markers, figsize=(6 * n_markers, 6))
    if n_markers == 1:
        axes = [axes]
    fig.suptitle(f"{Path(output_path).stem} – DAPI + Markers", fontsize=14)
    marker_colors = {"HER2": 1, "PR": 0, "ER": 0, "Ki67": 0}
    for i, name in enumerate(marker_channels):
        ov = np.zeros((*dapi_img.shape, 3))
        ov[:, :, 2] = dapi_n * 0.6
        color_idx = marker_colors.get(name, 0)
        ov[:, :, color_idx] = norm_img(all_channels[name])
        axes[i].imshow(np.clip(ov, 0, 1))
        axes[i].set_title(f"DAPI (blue) + {name}", fontsize=12)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_path.replace("_overlay.png", "_dapi_markers.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 4: random color per cell
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(dapi_n, cmap="gray", alpha=0.4)
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


# =====================================================================
# Otsu fallback
# =====================================================================

def _find_nuclei_otsu(cyto_masks, dapi_img, min_nuc_area=30, max_area_ratio=0.8):
    nuclei_masks = np.zeros_like(cyto_masks, dtype=np.int32)
    unique_labels = np.unique(cyto_masks)
    unique_labels = unique_labels[unique_labels > 0]

    found = 0
    for label in tqdm(unique_labels, desc="Otsu nuclei", unit="cell"):
        region = cyto_masks == label
        rows = np.any(region, axis=1)
        cols = np.any(region, axis=0)
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]

        margin = 5
        y0 = max(0, y0 - margin)
        y1 = min(dapi_img.shape[0], y1 + margin + 1)
        x0 = max(0, x0 - margin)
        x1 = min(dapi_img.shape[1], x1 + margin + 1)

        roi_dapi = dapi_img[y0:y1, x0:x1]
        roi_cyto = cyto_masks[y0:y1, x0:x1]
        roi_region = roi_cyto == label

        nonzero = roi_dapi[roi_region]
        if nonzero.size == 0:
            continue

        cyto_area = int(np.sum(roi_region))

        thr = skimage.filters.threshold_otsu(nonzero)
        binary = np.zeros_like(roi_dapi, dtype=np.int32)
        binary[roi_region] = (roi_dapi[roi_region] > thr).astype(np.int32)

        labeled, num = skimage.measure.label(binary, return_num=True, connectivity=2)
        if num == 0:
            continue

        props = skimage.measure.regionprops(labeled)
        best_prop = max(props, key=lambda p: p.area)
        nuc_area = best_prop.area

        if nuc_area < min_nuc_area:
            continue
        if nuc_area > cyto_area * max_area_ratio:
            continue

        nuc_mask_local = labeled == best_prop.label
        nuclei_masks[y0:y1, x0:x1][nuc_mask_local] = label
        found += 1

    print(f"    Otsu: {found}/{len(unique_labels)} found")
    return nuclei_masks


# =====================================================================
# argparse
# =====================================================================

def parse_args():
    positional_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flag_args = [a for a in sys.argv[1:] if a.startswith("--")]

    if len(positional_args) == 1 and not any(a.startswith("--dapi") for a in sys.argv[1:]):
        BLOCK = positional_args[0]
        paths = auto_paths(BLOCK)
        use_nuclei = "--no-cellpose-nuclei" not in flag_args

        return argparse.Namespace(
            dapi=paths["dapi"], her2=paths["her2"],
            pr=paths["pr"], er=paths["er"], ki67=paths["ki67"],
            model=MODEL_PATH, block_name=BLOCK,
            output_dir=str(SEGMENTATION_DIR),
            diameter=30, flow_threshold=0.4, cellprob_threshold=0.0,
            channels="0,1", min_nuc_area=30, max_area_ratio=0.8,
            use_cellpose_nuclei=use_nuclei, nuclei_diameter=None,
        )

    p = argparse.ArgumentParser(
        description="CPSAM + v3 Nuclei cell segmentation (5-channel, fast)"
    )
    p.add_argument("--dapi", required=True)
    p.add_argument("--her2", required=True)
    p.add_argument("--pr", required=True)
    p.add_argument("--er", required=True)
    p.add_argument("--ki67", required=False, default="")
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
    p.add_argument("--no-cellpose-nuclei", dest="use_cellpose_nuclei",
                    action="store_false")
    p.add_argument("--nuclei-diameter", type=int, default=None)
    return p.parse_args()


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("CPSAM + v3 Nuclei Segmentation (5-channel, fast)")
    print("=" * 60)

    ch0, ch1 = map(int, args.channels.split(","))

    # ── Output paths ──
    out_dir = Path(args.output_dir) / args.block_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / args.block_name

    # TIFF masks
    path_nuclei_raw        = f"{prefix}_nuclei_masks_raw.tif"
    path_cyto              = f"{prefix}_cyto_masks.tif"
    path_nuclei            = f"{prefix}_nuclei_masks.tif"
    path_cell              = f"{prefix}_cell_masks.tif"

    # TIFF overlays
    path_nuc_cyto_tif      = f"{prefix}_nucleus_cytoplasm.tif"
    path_all_ch_tif        = f"{prefix}_all_channels.tif"
    path_dapi_cyto_nuc_tif = f"{prefix}_dapi_cyto_nuc.tif"

    # PNG masks
    path_cyto_png          = f"{prefix}_cyto_masks.png"
    path_nuclei_png        = f"{prefix}_nuclei_masks.png"
    path_cell_png          = f"{prefix}_cell_masks.png"

    # PNG overlays
    path_nuc_cyto_png      = f"{prefix}_nucleus_cytoplasm.png"
    path_overlay           = f"{prefix}_overlay.png"

    # CSV
    path_csv               = f"{prefix}_features.csv"
    path_csv_core          = f"{prefix}_features_core.csv"

    # ── Load all channels ──
    print("\n[1/7] Loading images …")
    dapi_img = load_channel(args.dapi)
    her2_img = load_channel(args.her2)
    pr_img   = load_channel(args.pr)
    er_img   = load_channel(args.er)
    
    if args.ki67 and Path(args.ki67).exists():
        ki67_img = load_channel(args.ki67)
    else:
        print("    [!] Ki67 channel not provided or not found, using dummy zeros.")
        ki67_img = np.zeros_like(dapi_img)

    print(f"    DAPI shape: {dapi_img.shape}")

    min_h = min(dapi_img.shape[0], her2_img.shape[0], pr_img.shape[0],
                er_img.shape[0], ki67_img.shape[0])
    min_w = min(dapi_img.shape[1], her2_img.shape[1], pr_img.shape[1],
                er_img.shape[1], ki67_img.shape[1])
    dapi_img = dapi_img[:min_h, :min_w]
    her2_img = her2_img[:min_h, :min_w]
    pr_img   = pr_img[:min_h, :min_w]
    er_img   = er_img[:min_h, :min_w]
    ki67_img = ki67_img[:min_h, :min_w]
    print(f"    All channels cropped to: ({min_h}, {min_w})")

    # ── Step 2: Nuclei ──
    if Path(path_nuclei_raw).exists():
        print(f"\n[2/7] Loading cached nuclei masks: {path_nuclei_raw}")
        nuclei_masks_raw = load_tiff_int32(path_nuclei_raw)
        n_nuc_raw = len(np.unique(nuclei_masks_raw)) - 1
        print(f"    Loaded {n_nuc_raw} nuclei")
    elif args.use_cellpose_nuclei:
        print("\n[2/7] Segmenting nuclei (v3 model, whole image) …")
        nuclei_masks_raw = segment_nuclei_whole(
            dapi_img, diameter=args.nuclei_diameter or args.diameter,
        )
        n_nuc_raw = len(np.unique(nuclei_masks_raw)) - 1
        print(f"    Found {n_nuc_raw} nuclei")
        save_tiff(path_nuclei_raw, nuclei_masks_raw)
        print(f"    Saved: {path_nuclei_raw}")
    else:
        print("\n[2/7] Skipping nuclei (will use Otsu after CPSAM)")
        nuclei_masks_raw = None

    # ── Step 3: CPSAM ──
    if Path(path_cyto).exists():
        print(f"\n[3/7] Loading cached cytoplasm masks: {path_cyto}")
        cyto_masks = load_tiff_int32(path_cyto)
        n_cyto = len(np.unique(cyto_masks)) - 1
        print(f"    Loaded {n_cyto} cytoplasm regions")
    else:
        print("\n[3/7] Segmenting cytoplasm (CPSAM) …")
        img_2ch = np.stack([dapi_img.astype(np.float32),
                            her2_img.astype(np.float32)], axis=0)
        cyto_masks = segment_cytoplasm(
            img_2ch, args.model,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            channels=(ch0, ch1),
        )
        n_cyto = len(np.unique(cyto_masks)) - 1
        print(f"    Found {n_cyto} cytoplasm regions")
        save_tiff(path_cyto, cyto_masks)
        print(f"    Saved: {path_cyto}")

    # ── Step 4: Match ──
    if Path(path_nuclei).exists():
        print(f"\n[4/7] Loading cached matched nuclei: {path_nuclei}")
        nuclei_masks = load_tiff_int32(path_nuclei)
        n_nuc = len(np.unique(nuclei_masks[nuclei_masks > 0]))
        print(f"    Loaded {n_nuc} matched nuclei")
    elif args.use_cellpose_nuclei and nuclei_masks_raw is not None:
        print("\n[4/7] Matching nuclei to cytoplasm …")
        nuclei_masks = match_nuclei_to_cyto(
            nuclei_masks_raw, cyto_masks,
            min_nuc_area=args.min_nuc_area,
            max_area_ratio=args.max_area_ratio,
        )
        n_nuc = len(np.unique(nuclei_masks[nuclei_masks > 0]))
        print(f"    Final nuclei: {n_nuc}")
        save_tiff(path_nuclei, nuclei_masks)
        print(f"    Saved: {path_nuclei}")
    else:
        print("\n[4/7] Otsu nucleus detection …")
        nuclei_masks = _find_nuclei_otsu(
            cyto_masks, dapi_img,
            args.min_nuc_area, args.max_area_ratio,
        )
        n_nuc = len(np.unique(nuclei_masks[nuclei_masks > 0]))
        print(f"    Final nuclei: {n_nuc}")
        save_tiff(path_nuclei, nuclei_masks)
        print(f"    Saved: {path_nuclei}")

    # ── Step 4: Align（只保留有核的细胞）──
    if Path(path_cell).exists():
        print(f"\n[5/7] Loading cached cell masks: {path_cell}")
        cell_masks = load_tiff_int32(path_cell)
        n_cells = len(np.unique(cell_masks)) - 1
        print(f"    Loaded {n_cells} complete cells")
    else:
        print("\n[5/7] Aligning labels (cells with nucleus only) …")
        cell_masks = align_labels(cyto_masks, nuclei_masks)
        n_cells = len(np.unique(cell_masks)) - 1
        print(f"    {n_cells} complete cells")
        save_tiff(path_cell, cell_masks)
        print(f"    Saved: {path_cell}")

    # ── Step 5: 掩膜可视化 (PNG + TIFF) ──
    print("\n[5/7] Saving visualizations …")

    # --- 单独掩膜 PNG ---
    if not Path(path_cyto_png).exists():
        save_mask_png(cyto_masks, path_cyto_png, "Cytoplasm Masks")
    else:
        print(f"    Exists: {path_cyto_png}")

    if not Path(path_nuclei_png).exists():
        save_mask_png(nuclei_masks, path_nuclei_png, "Nuclei Masks", colormap="Set1")
    else:
        print(f"    Exists: {path_nuclei_png}")

    if not Path(path_cell_png).exists():
        save_mask_png(cell_masks, path_cell_png, "Cell Masks")
    else:
        print(f"    Exists: {path_cell_png}")

    # --- 核+质同图 PNG ---
    if not Path(path_nuc_cyto_png).exists():
        print("    Creating nucleus + cytoplasm PNG …")
        save_nucleus_cytoplasm_overlay(dapi_img, cyto_masks, nuclei_masks, path_nuc_cyto_png)
    else:
        print(f"    Exists: {path_nuc_cyto_png}")

    # --- 核+质同图 TIFF ---
    if not Path(path_nuc_cyto_tif).exists():
        print("    Creating nucleus + cytoplasm TIFF …")
        save_overlay_tiff(dapi_img, cyto_masks, nuclei_masks, path_nuc_cyto_tif)
    else:
        print(f"    Exists: {path_nuc_cyto_tif}")

    # --- 5通道 TIFF ---
    if not Path(path_all_ch_tif).exists():
        print("    Creating all-channels TIFF …")
        all_channels = {
            "DAPI": dapi_img, "HER2": her2_img,
            "PR": pr_img, "ER": er_img, "Ki67": ki67_img,
        }
        save_all_channels_tiff(all_channels, path_all_ch_tif)
    else:
        print(f"    Exists: {path_all_ch_tif}")

    # --- DAPI+胞质+核 RGB TIFF ---
    if not Path(path_dapi_cyto_nuc_tif).exists():
        print("    Creating DAPI + cyto + nuc TIFF …")
        save_dapi_cyto_nuc_tiff(dapi_img, cyto_masks, nuclei_masks, path_dapi_cyto_nuc_tif)
    else:
        print(f"    Exists: {path_dapi_cyto_nuc_tif}")

    # ── Step 6: Extract features ──
    if Path(path_csv).exists():
        print(f"\n[6/7] Loading cached features: {path_csv}")
        features_df = pd.read_csv(path_csv)
        print(f"    Loaded {len(features_df)} rows × {len(features_df.columns)} columns")
    else:
        print("\n[6/7] Extracting features from all 5 channels …")
        all_channels = {
            "DAPI": dapi_img, "HER2": her2_img,
            "PR": pr_img, "ER": er_img, "Ki67": ki67_img,
        }
        features_df = extract_features(cyto_masks, nuclei_masks, cell_masks, all_channels)
        print(f"    {len(features_df)} cells × {len(features_df.columns)} features")
        if len(features_df) > 0:
            no_nuc = (~features_df["has_nucleus"]).sum()
            if no_nuc > 0:
                print(f"    WARNING: {no_nuc} cells have no nucleus")
        features_df.to_csv(path_csv, index=False)
        print(f"    Saved: {path_csv}")

    # ── 精简版 CSV ──
    core_cols = [
        "cell_label", "cell_area", "nuc_area", "cyto_only_area",
        "nuc_eccentricity", "cell_eccentricity",
        "has_nucleus",
        "HER2_nuc_mean", "HER2_cyto_only_mean", "HER2_nuc_cyto_ratio",
        "Ki67_nuc_mean",
        "ER_nuc_mean", "PR_nuc_mean",
        "DAPI_nuc_mean",
    ]
    if not Path(path_csv_core).exists():
        available = [c for c in core_cols if c in features_df.columns]
        features_df[available].to_csv(path_csv_core, index=False)
        print(f"    Saved core features: {path_csv_core} ({len(available)} columns)")
    else:
        print(f"    Exists: {path_csv_core}")

    # ── Step 6: Visualize ──
    if Path(path_overlay).exists():
        print(f"\n[7/7] Overlays already exist: {path_overlay}")
    else:
        print("\n[7/7] Creating visualizations …")
        all_channels = {
            "DAPI": dapi_img, "HER2": her2_img,
            "PR": pr_img, "ER": er_img, "Ki67": ki67_img,
        }
        create_overlay(all_channels, cyto_masks, nuclei_masks, path_overlay)
        print(f"    Saved: {path_overlay}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Output: {out_dir}")
    print(f"  Cytoplasm regions: {len(np.unique(cyto_masks)) - 1}")
    print(f"  Nuclei matched:    {len(np.unique(nuclei_masks[nuclei_masks > 0]))}")
    print(f"  Complete cells:    {len(np.unique(cell_masks)) - 1}")
    print(f"  Features:          {len(features_df)} rows × {len(features_df.columns)} columns")
    print(f"\n  Files:")
    print(f"    {prefix.name}_nuclei_masks_raw.tif")
    print(f"    {prefix.name}_cyto_masks.tif")
    print(f"    {prefix.name}_cyto_masks.png")
    print(f"    {prefix.name}_nuclei_masks.tif")
    print(f"    {prefix.name}_nuclei_masks.png")
    print(f"    {prefix.name}_cell_masks.tif")
    print(f"    {prefix.name}_cell_masks.png")
    print(f"    {prefix.name}_nucleus_cytoplasm.tif")
    print(f"    {prefix.name}_nucleus_cytoplasm.png")
    print(f"    {prefix.name}_all_channels.tif")
    print(f"    {prefix.name}_dapi_cyto_nuc.tif")
    print(f"    {prefix.name}_features.csv")
    print(f"    {prefix.name}_features_core.csv")
    print(f"    {prefix.name}_overlay.png")
    print(f"    {prefix.name}_all_channels.png")
    print(f"    {prefix.name}_dapi_cyto_nuc.png")
    print(f"    {prefix.name}_dapi_markers.png")
    print("\nDone.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
