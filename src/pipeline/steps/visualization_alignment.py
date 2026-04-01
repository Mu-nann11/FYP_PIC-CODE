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
import json

warnings.filterwarnings('ignore')

# Overlay defaults (tuned to avoid marker "film" over DAPI)
DAPI_COLOR = [0.2, 0.4, 1.0]
MARKER_COLORS = {
    "ER": [1.0, 0.0, 0.0],
    "PR": [0.0, 1.0, 0.0],
    "HER2": [1.0, 1.0, 0.0],
    "KI67": [1.0, 0.0, 1.0],
}
COMPARISON_CHANNELS = ["ER", "PR", "HER2", "KI67"]
OVERLAY_DEFAULTS = {
    "marker_threshold": 0.15,
    "marker_alpha": 0.7,
    "marker_gamma": 0.6,
    "dapi_strength": 0.8,
}


def compute_ncc(a, b):
    """计算归一化互相关（NCC）"""
    min_h = min(a.shape[0], b.shape[0])
    min_w = min(a.shape[1], b.shape[1])
    a, b = a[:min_h, :min_w], b[:min_h, :min_w]
    x, y = a.ravel(), b.ravel()
    mask = (x > 0) | (y > 0)
    if mask.sum() < 100:
        return -1.0
    x, y = x[mask], y[mask]
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float(np.corrcoef(x, y)[0, 1])


def norm(img):
    """归一化到 [0, 1]"""
    p1, p99 = np.percentile(img, (1, 99))
    if p99 - p1 < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - p1) / (p99 - p1), 0, 1).astype(np.float32)


def compute_alignment_quality_metrics(block, dataset="TMAe"):
    """
    计算配准质量指标
    返回: {
        "ncc_before": float, "ncc_after": float,
        "shift_x": float, "shift_y": float, "angle": float or None,
        "channel_metrics": {channel: {"ncc_before": float, "ncc_after": float, "shift_x": float, "shift_y": float}}
    }
    """
    from ..config import BASE_DIR

    metrics = {
        "ncc_before": None,
        "ncc_after": None,
        "shift_x": None,
        "shift_y": None,
        "angle": None,
        "channel_metrics": {}
    }

    try:
        # TMAd: 有 Cycle2，需要计算旋转配准
        if dataset == "TMAd":
            stitch_dir = BASE_DIR / "results" / "stitched" / "TMAd"
            registered_dir = BASE_DIR / "results" / "registered" / block

            # 加载 Cycle1 和 Cycle2 DAPI（未配准）
            dapi1_path = stitch_dir / "Cycle1" / block / f"{block}_TMAd_Cycle1_DAPI.tif"
            dapi2_path = stitch_dir / "Cycle2" / block / f"{block}_TMAd_Cycle2_DAPI.tif"

            # 加载配准后的 DAPI
            dapi1_reg_path = registered_dir / f"{block}_Cycle1_DAPI.tif"
            dapi2_reg_path = registered_dir / f"{block}_Cycle1_HER2_aligned.tif"  # 用 HER2 的文件名，但实际 DAPI2 可能在 merged 中

            if dapi1_path.exists() and dapi2_path.exists():
                dapi1_raw = tifffile.imread(str(dapi1_path)).astype(np.float32)
                dapi2_raw = tifffile.imread(str(dapi2_path)).astype(np.float32)

                h = min(dapi1_raw.shape[0], dapi2_raw.shape[0])
                w = min(dapi1_raw.shape[1], dapi2_raw.shape[1])
                dapi1_n = norm(dapi1_raw[:h, :w])
                dapi2_n = norm(dapi2_raw[:h, :w])

                metrics["ncc_before"] = compute_ncc(dapi1_n, dapi2_n)

            # 加载配准后的图像并计算 NCC
            merged_path = registered_dir / f"{block}_merged_5channel.tif"
            if merged_path.exists():
                merged = tifffile.imread(str(merged_path))
                if merged.ndim == 3 and merged.shape[0] >= 2:
                    dapi1_reg = merged[0].astype(np.float32)
                    dapi2_reg = merged[4].astype(np.float32) if merged.shape[0] > 4 else merged[0].astype(np.float32)

                    h = min(dapi1_reg.shape[0], dapi2_reg.shape[0])
                    w = min(dapi1_reg.shape[1], dapi2_reg.shape[1])
                    metrics["ncc_after"] = compute_ncc(
                        norm(dapi1_reg[:h, :w]),
                        norm(dapi2_reg[:h, :w])
                    )

            # 计算 Cycle1 内部通道的配准质量
            for ch in ["HER2", "PR", "ER"]:
                raw_path = stitch_dir / "Cycle1" / block / f"{block}_TMAd_Cycle1_{ch}.tif"
                reg_path = registered_dir / f"{block}_Cycle1_{ch}_aligned.tif"

                if raw_path.exists() and reg_path.exists():
                    raw = tifffile.imread(str(raw_path)).astype(np.float32)
                    reg = tifffile.imread(str(reg_path)).astype(np.float32)

                    h = min(dapi1_raw.shape[0], raw.shape[0], reg.shape[0])
                    w = min(dapi1_raw.shape[1], raw.shape[1], reg.shape[1])

                    dapi_n = norm(dapi1_raw[:h, :w])
                    raw_n = norm(raw[:h, :w])
                    reg_n = norm(reg[:h, :w])

                    metrics["channel_metrics"][ch] = {
                        "ncc_before": compute_ncc(dapi_n, raw_n),
                        "ncc_after": compute_ncc(dapi_n, reg_n)
                    }

        # TMAe: 只有 Cycle1 内部配准
        elif dataset == "TMAe":
            stitch_dir = BASE_DIR / "results" / "stitched" / "TMAe" / block
            registered_dir = BASE_DIR / "results" / "registered" / block

            # 加载 DAPI 作为参考
            dapi_path = stitch_dir / f"{block}_TMAe_DAPI.tif"
            if dapi_path.exists():
                dapi_raw = tifffile.imread(str(dapi_path)).astype(np.float32)
            else:
                return metrics

            # 计算 Cycle1 内部通道的配准质量
            for ch in ["HER2", "PR", "ER"]:
                raw_path = stitch_dir / f"{block}_TMAe_{ch}.tif"
                reg_path = registered_dir / f"{block}_Cycle1_{ch}_aligned.tif"

                if raw_path.exists() and reg_path.exists():
                    raw = tifffile.imread(str(raw_path)).astype(np.float32)
                    reg = tifffile.imread(str(reg_path)).astype(np.float32)

                    h = min(dapi_raw.shape[0], raw.shape[0], reg.shape[0])
                    w = min(dapi_raw.shape[1], raw.shape[1], reg.shape[1])

                    dapi_n = norm(dapi_raw[:h, :w])
                    raw_n = norm(raw[:h, :w])
                    reg_n = norm(reg[:h, :w])

                    metrics["channel_metrics"][ch] = {
                        "ncc_before": compute_ncc(dapi_n, raw_n),
                        "ncc_after": compute_ncc(dapi_n, reg_n)
                    }

    except Exception as e:
        print(f"[VIS] {block}: Error computing quality metrics - {e}")

    return metrics


def format_metrics_text(metrics):
    """将质量指标格式化为显示文本"""
    lines = []

    if metrics["ncc_before"] is not None and metrics["ncc_after"] is not None:
        lines.append(f"NCC: {metrics['ncc_before']:.4f} -> {metrics['ncc_after']:.4f}")

    if metrics["shift_x"] is not None and metrics["shift_y"] is not None:
        lines.append(f"Shift: ({metrics['shift_x']:.1f}, {metrics['shift_y']:.1f}) px")

    if metrics["angle"] is not None:
        lines.append(f"Angle: {metrics['angle']:.2f}°")

    if metrics["channel_metrics"]:
        lines.append("---")
        for ch, ch_metrics in metrics["channel_metrics"].items():
            if ch_metrics.get("ncc_before") is not None and ch_metrics.get("ncc_after") is not None:
                lines.append(f"{ch}: {ch_metrics['ncc_before']:.4f} -> {ch_metrics['ncc_after']:.4f}")

    return "\n".join(lines) if lines else "No metrics available"


def add_metrics_annotation(ax, metrics, position='lower_right'):
    """在图表上添加质量指标注释"""
    text = format_metrics_text(metrics)

    if position == 'lower_right':
        x, y = 0.98, 0.02
        ha, va = 'right', 'bottom'
    elif position == 'upper_left':
        x, y = 0.02, 0.98
        ha, va = 'left', 'top'
    else:
        x, y = 0.98, 0.98
        ha, va = 'right', 'top'

    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=9, verticalalignment=va, horizontalalignment=ha,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'),
            family='monospace')

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


def crop_pair_to_common(img_a, img_b):
    """Crop two images to the same top-left common size."""
    if img_a is None or img_b is None:
        return img_a, img_b
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    return img_a[:h, :w], img_b[:h, :w]


def resolve_before_paths(block, dataset, channel):
    """Resolve crop-first before paths for DAPI + marker."""
    from ..config import BASE_DIR

    if dataset == "TMAd":
        cycle = "Cycle2" if channel == "KI67" else "Cycle1"
        crop_dir = BASE_DIR / "results" / "crop" / "TMAd" / block / f"{block}_TMAd_{cycle}"
        crop_dapi = crop_dir / f"{block}_TMAd_{cycle}_DAPI_crop.tif"
        crop_marker = crop_dir / f"{block}_TMAd_{cycle}_{channel}_crop.tif"
        if crop_dapi.exists() and crop_marker.exists():
            return crop_dapi, crop_marker, "crop"

        stitched_dir = BASE_DIR / "results" / "stitched" / "TMAd" / cycle / block
        stitched_dapi = stitched_dir / f"{block}_TMAd_{cycle}_DAPI.tif"
        stitched_marker = stitched_dir / f"{block}_TMAd_{cycle}_{channel}.tif"
        if stitched_dapi.exists() and stitched_marker.exists():
            return stitched_dapi, stitched_marker, "stitched"

    if dataset == "TMAe":
        crop_dir = BASE_DIR / "results" / "crop" / "TMAe" / block / f"{block}_TMAe"
        crop_dapi = crop_dir / f"{block}_TMAe_DAPI_crop.tif"
        crop_marker = crop_dir / f"{block}_TMAe_{channel}_crop.tif"
        if crop_dapi.exists() and crop_marker.exists():
            return crop_dapi, crop_marker, "crop"

        stitched_dir = BASE_DIR / "results" / "stitched" / "TMAe" / block
        stitched_dapi = stitched_dir / f"{block}_TMAe_DAPI.tif"
        stitched_marker = stitched_dir / f"{block}_TMAe_{channel}.tif"
        if stitched_dapi.exists() and stitched_marker.exists():
            return stitched_dapi, stitched_marker, "stitched"

    return None, None, None


def resolve_after_paths(block, channel):
    """Resolve registered after paths for DAPI + marker."""
    from ..config import BASE_DIR

    reg_dir = BASE_DIR / "results" / "registered" / block
    dapi_paths = [
        reg_dir / f"{block}_Cycle1_DAPI.tif",
        reg_dir / f"{block}_DAPI.tif",
    ]
    dapi_path = next((p for p in dapi_paths if p.exists()), None)

    if channel == "KI67":
        marker_paths = [
            reg_dir / f"{block}_Cycle2_KI67_aligned.tif",
            reg_dir / f"{block}_Ki67_aligned.tif",
        ]
    else:
        marker_paths = [
            reg_dir / f"{block}_Cycle1_{channel}_aligned.tif",
            reg_dir / f"{block}_Cycle1_{channel}.tif",
            reg_dir / f"{block}_{channel}_aligned.tif",
        ]
    marker_path = next((p for p in marker_paths if p.exists()), None)

    return dapi_path, marker_path


def load_pair_images(dapi_path, marker_path):
    """Load DAPI + marker and crop to common size if needed."""
    dapi = load_image(dapi_path) if dapi_path is not None else None
    marker = load_image(marker_path) if marker_path is not None else None
    if dapi is not None and marker is not None:
        dapi, marker = crop_pair_to_common(dapi, marker)
    return dapi, marker


def create_dapi_marker_overlay(
    dapi,
    marker,
    dapi_color,
    marker_color,
    marker_threshold=0.15,
    marker_alpha=0.7,
    marker_gamma=0.6,
    dapi_strength=0.8,
):
    """Blend DAPI and marker with a threshold mask to avoid background film."""
    if dapi is None and marker is None:
        return None

    base = dapi if dapi is not None else marker
    h, w = base.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    if dapi is not None:
        dapi_norm = normalize_image(dapi)
        for c in range(3):
            rgb[:, :, c] += dapi_norm * dapi_color[c] * dapi_strength

    if marker is not None:
        marker_norm = normalize_image(marker)
        marker_enh = np.power(marker_norm, marker_gamma)
        mask = marker_enh > marker_threshold
        for c in range(3):
            rgb[:, :, c] = np.where(
                mask,
                rgb[:, :, c] + marker_enh * marker_color[c] * marker_alpha,
                rgb[:, :, c],
            )

    return np.clip(rgb, 0, 1)


def create_transparent_overlay(dapi, marker, color, alpha=0.5):
    """使用透明度混合创建单通道叠加图
    
    Args:
        dapi: DAPI核染色图像（用于背景）
        marker: 标记通道图像
        color: RGB颜色 (0-1范围)
        alpha: 透明度 (0-1)
    """
    rgb = np.zeros((*marker.shape[:2], 3))
    marker_norm = normalize_image(marker)
    # 只在marker信号 > 阈值时才显示颜色
    mask = marker_norm > 0.15
    rgb[mask] = np.array(color) * marker_norm[mask, None] * alpha
    
    # 添加DAPI背景（灰度）
    if dapi is not None:
        dapi_norm = normalize_image(dapi)
        for c in range(3):
            rgb[:, :, c] += dapi_norm * 0.3
    return np.clip(rgb, 0, 1)


def create_alpha_overlay(images_dict, alpha=0.4, gamma=0.5):
    """使用独立归一化+透明度混合创建多通道叠加图
    
    Args:
        images_dict: {channel_name: (image, rgb_color)}
        alpha: 基础透明度
        gamma: gamma校正系数 (用于增强对比度)
    
    Returns:
        RGB图像
    """
    first_img = list(images_dict.values())[0][0]
    h, w = first_img.shape[:2]
    rgb = np.zeros((h, w, 3))
    
    for name, (img, color) in images_dict.items():
        norm_img = normalize_image(img)
        # 使用gamma变换增强对比度
        enhanced = np.power(norm_img, gamma)
        for c in range(3):
            rgb[:, :, c] += enhanced * color[c] * alpha
    
    return np.clip(rgb, 0, 1)


def create_alignment_comparison(block, dataset="TMAd", force=False):
    """创建裁剪后(Before) vs 对齐后(After)的对比图

    Args:
        block: 区块ID
        dataset: 数据集名称
        force: 是否强制重新生成

    Returns:
        保存的图像路径列表，失败返回空列表
    """
    return create_alignment_comparison_with_metrics(block, dataset, {}, force=force)


def create_alignment_comparison_with_metrics(block, dataset="TMAd", metrics=None, force=False):
    """Create per-channel DAPI + marker comparisons (Before vs After)."""
    from ..config import BASE_DIR, DATASETS

    if metrics is None:
        metrics = {}

    saved_paths = []
    output_dir = BASE_DIR / "results" / "registered" / block
    output_dir.mkdir(parents=True, exist_ok=True)

    for channel in COMPARISON_CHANNELS:
        out_path = output_dir / f"{block}_registration_comparison_{channel}.png"
        if not force and out_path.exists():
            saved_paths.append(out_path)
            continue

        before_dapi_path, before_marker_path, before_source = resolve_before_paths(block, dataset, channel)
        after_dapi_path, after_marker_path = resolve_after_paths(block, channel)

        if before_dapi_path is None and before_marker_path is None and after_dapi_path is None and after_marker_path is None:
            continue

        before_dapi, before_marker = load_pair_images(before_dapi_path, before_marker_path)
        after_dapi, after_marker = load_pair_images(after_dapi_path, after_marker_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Left: Before (crop-first, no phase correction)
        if before_dapi is not None and before_marker is not None:
            overlay_before = create_dapi_marker_overlay(
                before_dapi,
                before_marker,
                DAPI_COLOR,
                MARKER_COLORS.get(channel, [1.0, 0.0, 0.0]),
                **OVERLAY_DEFAULTS,
            )
            axes[0].imshow(overlay_before)
            src_label = before_source or "before"
            axes[0].set_title(f"Before ({src_label}): DAPI + {channel}", fontsize=10)
        else:
            axes[0].text(0.5, 0.5, f"{channel}\n(before not found)",
                         ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title(f"Before: DAPI + {channel} (N/A)", fontsize=10)
        axes[0].axis('off')

        # Right: After (aligned)
        if after_dapi is not None and after_marker is not None:
            overlay_after = create_dapi_marker_overlay(
                after_dapi,
                after_marker,
                DAPI_COLOR,
                MARKER_COLORS.get(channel, [1.0, 0.0, 0.0]),
                **OVERLAY_DEFAULTS,
            )
            axes[1].imshow(overlay_after)

            title_text = f"After: DAPI + {channel}"
            ch_metrics = metrics.get("channel_metrics", {}).get(channel, {})
            ncc_before = ch_metrics.get("ncc_before")
            ncc_after = ch_metrics.get("ncc_after")
            if ncc_before is not None and ncc_after is not None:
                improvement = (ncc_after - ncc_before) / abs(ncc_before) * 100 if ncc_before != 0 else 0
                title_text += f"\nNCC: {ncc_before:.4f} -> {ncc_after:.4f} (Δ={improvement:+.1f}%)"
            axes[1].set_title(title_text, fontsize=10)
        else:
            axes[1].text(0.5, 0.5, f"{channel}\n(after not found)",
                         ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title(f"After: DAPI + {channel} (N/A)", fontsize=10)
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

        if out_path.exists():
            print(f"[VIS] {block}: Saved {out_path.name}")
            saved_paths.append(out_path)

    return saved_paths


def create_rgb_overlay(red, green, blue):
    """Create RGB overlay from three grayscale images"""
    h, w = red.shape[:2]
    rgb = np.zeros((h, w, 3))
    rgb[:, :, 0] = normalize_image(red)
    rgb[:, :, 1] = normalize_image(green)
    rgb[:, :, 2] = normalize_image(blue)
    return np.clip(rgb, 0, 1)


def create_transparent_marker_overlay(images_dict, intensity_threshold=0.1, alpha=0.6, gamma=0.5):
    """
    使用透明度混合创建多通道Marker叠加图，只在有信号区域显示颜色
    
    Args:
        images_dict: dict, {channel_name: (image, rgb_color)}
            例如: {'ER': (er_img, [1, 0, 0]), 'PR': (pr_img, [0, 1, 0])}
        intensity_threshold: float, 信号检测阈值 (0-1)
        alpha: float, 透明度 (0-1), 值越小颜色越透明
        gamma: float, Gamma校正值，用于增强对比度 (<1 增强亮区, >1 增强暗区)
    
    Returns:
        numpy.ndarray: RGB图像 (H, W, 3), 归一化到 [0, 1]
    """
    if not images_dict:
        return None
    
    # 获取图像尺寸
    first_name = next(iter(images_dict))
    first_img = images_dict[first_name][0]
    h, w = first_img.shape[:2]
    
    # 初始化RGB图像
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    
    for name, (img, color) in images_dict.items():
        # 归一化
        norm_img = normalize_image(img)
        
        # Gamma校正增强对比度
        enhanced = np.power(norm_img, gamma)
        
        # 创建信号掩膜：只在超过阈值的区域显示颜色
        mask = enhanced > intensity_threshold
        
        # 对掩膜区域添加颜色
        for c in range(3):
            # 使用掩膜确保只在有信号区域添加颜色
            rgb[:, :, c] = np.where(
                mask,
                rgb[:, :, c] + enhanced * color[c] * alpha,
                rgb[:, :, c]
            )
    
    # 裁剪到 [0, 1] 范围
    return np.clip(rgb, 0, 1)


def visualize_alignment_for_block(block, dataset="TMAe", force=False):
    """Generate alignment visualization for a block"""

    from ..config import BASE_DIR, DATASETS

    if dataset not in DATASETS:
        print(f"[VIS] Unknown dataset: {dataset}")
        return False

    # Paths
    registered_dir = BASE_DIR / "results" / "registered" / block
    output_dir = BASE_DIR / "results" / "registered" / block
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    overlay_png = output_dir / f"{block}_multi_marker_overlay.png"
    metrics_json = output_dir / f"{block}_alignment_metrics.json"

    expected_comparisons = []
    for channel in COMPARISON_CHANNELS:
        before_dapi_path, before_marker_path, _ = resolve_before_paths(block, dataset, channel)
        after_dapi_path, after_marker_path = resolve_after_paths(block, channel)
        if before_dapi_path is None and before_marker_path is None and after_dapi_path is None and after_marker_path is None:
            continue
        expected_comparisons.append(output_dir / f"{block}_registration_comparison_{channel}.png")

    if not force and expected_comparisons and all(p.exists() for p in expected_comparisons) and overlay_png.exists():
        print(f"[VIS] {block}: Alignment visualization already done")
        return True

    print(f"[VIS] {block}: Creating alignment visualizations...")

    try:
        # 1. Compute quality metrics
        metrics = compute_alignment_quality_metrics(block, dataset)

        # Save metrics to JSON
        with open(metrics_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[VIS] {block}: Saved quality metrics to {metrics_json.name}")

        # 2. Create Before/After comparison with quality metrics
        create_alignment_comparison_with_metrics(block, dataset, metrics, force=force)

        # 3. Load registered images for overlay
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

        # 4. Multi-marker overlay (ER + PR + HER2) using alpha blending with threshold mask
        if er is not None and pr is not None and her2 is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Left: ER(red) + PR(green) + HER2(blue) using alpha blending with threshold mask
            images_dict = {
                'ER': (er, [1.0, 0.0, 0.0]),  # Red
                'PR': (pr, [0.0, 1.0, 0.0]),  # Green
                'HER2': (her2, [0.0, 0.0, 1.0])  # Blue
            }
            rgb = create_transparent_marker_overlay(
                images_dict,
                intensity_threshold=0.15,
                alpha=0.7,
                gamma=0.5
            )
            axes[0].imshow(rgb)
            axes[0].set_title(f'{block}: ER(Red) + PR(Green) + HER2(Blue)\n(Transparent Overlay)')
            axes[0].axis('off')

            # Right: DAPI + Ki67 overlay if Ki67 available
            if ki67 is not None:
                dapi_ki67_dict = {
                    'DAPI': (dapi_c1, [0.3, 0.3, 0.3]),
                    'Ki67': (ki67, [1.0, 0.0, 0.0])
                }
                rgb2 = create_transparent_marker_overlay(
                    dapi_ki67_dict,
                    intensity_threshold=0.15,
                    alpha=0.7,
                    gamma=0.5
                )
                axes[1].imshow(rgb2)
                axes[1].set_title(f'{block}: DAPI + Ki67\n(Transparent Overlay)')
            else:
                axes[1].imshow(normalize_image(dapi_c1), cmap='gray')
                axes[1].set_title(f'{block}: DAPI')
            axes[1].axis('off')

            # Add quality metrics annotation
            add_metrics_annotation(axes[0], metrics, position='lower_right')

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
