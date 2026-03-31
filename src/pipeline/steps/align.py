"""
Step 2: 图像配准 (Alignment / Registration)
封装 alignment 模块的配准逻辑

核心流程:
  1. Cycle1 内部通道 -> DAPI（色差校正）
  2. Cycle2 DAPI -> Cycle1 DAPI（计算旋转变换）
  3. 同一变换应用于 Ki67
  4. 统一裁剪到公共尺寸
  5. 保存单通道 + 合并 5 通道文件
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from scipy.ndimage import shift as ndshift
from skimage.exposure import match_histograms
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate

from pipeline.config import (
    CYCLE1_CHANNELS,
    CYCLE2_CHANNELS,
    REGISTERED_DIR,
    get_registered_path,
    get_stitched_path,
)
from pipeline.utils.logging import get_logger

logger = get_logger("pipeline.align")


# =====================================================================
# 工具函数（从 alignment.py 提取，保持兼容）
# =====================================================================

def load_tiff(path: Path):
    """加载 TIFF 图像"""
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        if img.shape[0] in (3, 4):
            img = img[0]
        elif img.shape[2] in (3, 4):
            img = img[:, :, 0]
    return img.astype(np.float32)


def save_tiff(path: Path, data):
    """保存 TIFF 图像"""
    tifffile.imwrite(str(path), data.astype(np.uint16))


def norm(img):
    """归一化到 [0, 1]"""
    p1, p99 = np.percentile(img, (1, 99))
    if p99 - p1 < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - p1) / (p99 - p1), 0, 1).astype(np.float32)


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


def find_content_bbox(img, threshold_ratio=0.01):
    """找到图像内容边界框"""
    threshold = img.max() * threshold_ratio
    mask = img > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return (0, img.shape[0], 0, img.shape[1])
    y1 = np.argmax(rows)
    y2 = len(rows) - np.argmax(rows[::-1])
    x1 = np.argmax(cols)
    x2 = len(cols) - np.argmax(cols[::-1])
    return (int(y1), int(y2), int(x1), int(x2))


def crop_to_bbox(img, bbox):
    """按边界框裁剪图像，如果图像不够大则用 0 填充以保证输出尺寸完全一致"""
    y1, y2, x1, x2 = bbox
    crop = img[y1:y2, x1:x2]
    
    target_h = y2 - y1
    target_w = x2 - x1
    
    if crop.shape[0] < target_h or crop.shape[1] < target_w:
        pad_h = max(0, target_h - crop.shape[0])
        pad_w = max(0, target_w - crop.shape[1])
        crop = np.pad(crop, ((0, pad_h), (0, pad_w)), mode='constant')
        
    return crop


def _fit_to_ref_shape(img: np.ndarray, ref_shape: tuple[int, int]) -> np.ndarray:
    """Resize by crop/pad to match the reference shape exactly."""
    ref_h, ref_w = ref_shape
    h, w = img.shape[:2]

    fitted = img[: min(h, ref_h), : min(w, ref_w)]

    pad_h = max(0, ref_h - fitted.shape[0])
    pad_w = max(0, ref_w - fitted.shape[1])
    if pad_h or pad_w:
        fitted = np.pad(fitted, ((0, pad_h), (0, pad_w)), mode="constant")

    return fitted


# =====================================================================
# 核心配准逻辑
# =====================================================================

def _align_cycle1_internal(dapi_ref, channel_imgs, logger_ref=None):
    """
    Cycle1 内部通道配准（色差校正）

    Args:
        dapi_ref: DAPI 参考图像
        channel_imgs: dict，键为通道名，值为原始图像

    Returns:
        dict: 配准后的通道图像（包含 DAPI）
    """
    _log = logger_ref or logger
    dapi_n = norm(dapi_ref)
    min_h, min_w = dapi_ref.shape[:2]

    aligned = {"DAPI": dapi_ref}

    for name, img in channel_imgs.items():
        if img.shape != dapi_ref.shape:
            _log.warning(
                f"  {name:5s}: shape {img.shape} differs from DAPI {dapi_ref.shape}, "
                "auto fitting by crop/pad."
            )
            img = _fit_to_ref_shape(img, dapi_ref.shape)

        img_n = norm(img)
        h_q = min(dapi_n.shape[0], img_n.shape[0])
        w_q = min(dapi_n.shape[1], img_n.shape[1])
        
        ncc_before = compute_ncc(dapi_n[:h_q, :w_q], img_n[:h_q, :w_q])

        shift, _, _ = phase_cross_correlation(dapi_n[:h_q, :w_q], img_n[:h_q, :w_q], upsample_factor=100)

        img_aligned = ndshift(img, shift=(-shift[0], -shift[1]), order=1, cval=0)
        
        # update h_q for ncc_after as shape might change or ndshift works? ndshift keeps same shape
        ncc_after = compute_ncc(dapi_n[:h_q, :w_q], norm(img_aligned[:h_q, :w_q]))

        aligned[name] = img_aligned
        _log.debug(
            f"  {name:5s}: shift=({shift[0]:+.2f}, {shift[1]:+.2f})  "
            f"NCC: {ncc_before:.4f} -> {ncc_after:.4f}"
        )

    return aligned


def _align_cycle2_to_cycle1(dapi1_n, dapi2, ki67, logger_ref=None):
    """
    Cycle2 DAPI/Ki67 配准到 Cycle1 DAPI（旋转 + 平移）

    Args:
        dapi1_n: 归一化的 Cycle1 DAPI
        dapi2: 原始 Cycle2 DAPI
        ki67: 原始 Ki67 图像

    Returns:
        dict: {
            "dapi2_aligned": 配准后 DAPI,
            "ki67_aligned": 配准后 Ki67,
            "transform": {"angle": float, "shift_y": float, "shift_x": float},
            "ncc": float  # 最终 NCC 分数
        }
    """
    _log = logger_ref or logger
    dapi2_n = norm(dapi2)

    # 快速检查（无旋转）
    h_q = min(dapi1_n.shape[0], dapi2_n.shape[0])
    w_q = min(dapi1_n.shape[1], dapi2_n.shape[1])
    shift_quick, _, _ = phase_cross_correlation(
        dapi1_n[:h_q, :w_q], dapi2_n[:h_q, :w_q], upsample_factor=100
    )
    shifted_quick = ndshift(dapi2_n[:h_q, :w_q], shift=shift_quick, order=1)
    ncc_quick = compute_ncc(dapi1_n[:h_q, :w_q], shifted_quick)
    _log.debug(
        f"  Quick (no rotation): shift=({shift_quick[0]:.1f}, {shift_quick[1]:.1f}), NCC={ncc_quick:.4f}"
    )

    # 粗搜索（-5° ~ +5°，步长 1°）
    _log.debug("  Phase 1: coarse rotation search (1deg step, -5 to +5) ...")
    coarse_results = []
    for angle in np.linspace(-5, 5, 11):
        dapi2_rot = rotate(dapi2_n, angle, order=1, preserve_range=True)
        h = min(dapi1_n.shape[0], dapi2_rot.shape[0])
        w = min(dapi1_n.shape[1], dapi2_rot.shape[1])
        s, _, _ = phase_cross_correlation(
            dapi1_n[:h, :w], dapi2_rot[:h, :w], upsample_factor=10
        )
        shifted = ndshift(dapi2_rot[:h, :w], shift=s, order=1)
        score = compute_ncc(dapi1_n[:h, :w], shifted)
        coarse_results.append((angle, s[0], s[1], score))

    coarse_results.sort(key=lambda x: x[3], reverse=True)
    coarse_best = coarse_results[0]
    _log.debug(
        f"  Coarse best: angle={coarse_best[0]:.1f}deg, NCC={coarse_best[3]:.4f}"
    )

    # 精搜索（粗最佳角度 ±1°，步长 0.1°）
    _log.debug(
        f"  Phase 2: fine rotation search (0.1deg step, {coarse_best[0]-1:.1f} to {coarse_best[0]+1:.1f}) ..."
    )
    fine_results = []
    for angle in np.arange(coarse_best[0] - 1, coarse_best[0] + 1.01, 0.1):
        dapi2_rot = rotate(dapi2_n, angle, order=1, preserve_range=True)
        h = min(dapi1_n.shape[0], dapi2_rot.shape[0])
        w = min(dapi1_n.shape[1], dapi2_rot.shape[1])
        s, _, _ = phase_cross_correlation(
            dapi1_n[:h, :w], dapi2_rot[:h, :w], upsample_factor=100
        )
        shifted = ndshift(dapi2_rot[:h, :w], shift=s, order=1)
        score = compute_ncc(dapi1_n[:h, :w], shifted)
        fine_results.append((angle, s[0], s[1], score))

    fine_results.sort(key=lambda x: x[3], reverse=True)
    best_angle, best_dy, best_dx, best_ncc = fine_results[0]
    _log.debug(
        f"  Best: angle={best_angle:.2f}deg, shift=({best_dy:.2f}, {best_dx:.2f}), NCC={best_ncc:.4f}"
    )

    # 应用变换到 DAPI2 和 Ki67
    transform = {"angle": float(best_angle), "shift_y": float(best_dy), "shift_x": float(best_dx)}

    dapi2_aligned = ndshift(
        rotate(dapi2, best_angle, order=1, preserve_range=True),
        shift=(best_dy, best_dx), order=1, cval=0
    )
    ki67_aligned = ndshift(
        rotate(ki67, best_angle, order=1, preserve_range=True),
        shift=(best_dy, best_dx), order=1, cval=0
    )

    return {
        "dapi2_aligned": dapi2_aligned,
        "ki67_aligned": ki67_aligned,
        "transform": transform,
        "ncc": float(best_ncc),
    }


def _auto_crop_and_save(block, all_channels, output_dir):
    """
    自动裁剪到公共区域并保存单通道文件 + 合并 5 通道文件

    Args:
        block: Block 名称
        all_channels: dict，键为通道名，值为已配准图像
        output_dir: 输出目录 Path

    Returns:
        dict: {"cropped": dict, "bboxes": dict, "merged_path": Path}
    """
    # 计算各通道内容边界
    bboxes = {name: find_content_bbox(img) for name, img in all_channels.items()}

    y1 = max(b[0] for b in bboxes.values())
    y2 = min(b[1] for b in bboxes.values())
    x1 = max(b[2] for b in bboxes.values())
    x2 = min(b[3] for b in bboxes.values())
    crop_bbox = (y1, y2, x1, x2)

    # 裁剪
    cropped = {name: crop_to_bbox(img, crop_bbox) for name, img in all_channels.items()}

    # 保存单通道
    file_map = {
        "DAPI":  f"{block}_Cycle1_DAPI.tif",
        "HER2":  f"{block}_Cycle1_HER2_aligned.tif",
        "PR":    f"{block}_Cycle1_PR_aligned.tif",
        "ER":    f"{block}_Cycle1_ER_aligned.tif",
        "KI67":  f"{block}_KI67_aligned.tif",
    }
    output_files = {}
    for name, fname in file_map.items():
        out_path = output_dir / fname
        save_tiff(out_path, cropped[name])
        output_files[name] = out_path

    # 保存合并文件
    final_order = ["DAPI", "HER2", "PR", "ER", "KI67"]
    merged = np.stack([cropped[ch] for ch in final_order], axis=0)
    merged_path = output_dir / f"{block}_merged_5channel.tif"
    tifffile.imwrite(str(merged_path), merged.astype(np.uint16), imagej=True)
    output_files["merged"] = merged_path

    return {
        "cropped": cropped,
        "bboxes": bboxes,
        "merged_path": merged_path,
        "final_shape": cropped[final_order[0]].shape,
        "crop_bbox": crop_bbox,
        "output_files": output_files,
    }


# =====================================================================
# TMAd 配准流程（Cycle1 + Cycle2）
# =====================================================================

def _run_alignment_tmad(block, logger_ref=None):
    """TMAd 数据配准：Cycle1 + Cycle2"""
    _log = logger_ref or logger

    # Step 1: 加载所有图像
    cycle1_dir = get_stitched_path(block, "TMAd", "Cycle1", "").parent
    cycle2_dir = get_stitched_path(block, "TMAd", "Cycle2", "").parent

    paths = {
        "dapi1": cycle1_dir / f"{block}_TMAd_Cycle1_DAPI.tif",
        "her2":  cycle1_dir / f"{block}_TMAd_Cycle1_HER2.tif",
        "pr":    cycle1_dir / f"{block}_TMAd_Cycle1_PR.tif",
        "er":    cycle1_dir / f"{block}_TMAd_Cycle1_ER.tif",
        "dapi2": cycle2_dir / f"{block}_TMAd_Cycle2_DAPI.tif",
        "ki67":  cycle2_dir / f"{block}_TMAd_Cycle2_KI67.tif",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")

    dapi1 = load_tiff(paths["dapi1"])
    her2  = load_tiff(paths["her2"])
    pr    = load_tiff(paths["pr"])
    er    = load_tiff(paths["er"])
    dapi2 = load_tiff(paths["dapi2"])
    ki67  = load_tiff(paths["ki67"])

    _log.debug(
        f"  Loaded shapes: DAPI1={dapi1.shape}, HER2={her2.shape}, "
        f"PR={pr.shape}, ER={er.shape}, DAPI2={dapi2.shape}, KI67={ki67.shape}"
    )

    # Step 2: Cycle1 内部通道配准（HER2/PR/ER -> DAPI）
    _log.debug("Step 2: Cycle1 internal alignment (chromatic aberration)")
    cycle1_aligned = _align_cycle1_internal(
        dapi1, {"HER2": her2, "PR": pr, "ER": er}, logger_ref=_log
    )
    cycle1_aligned["DAPI"] = dapi1

    for ch in ("HER2", "PR", "ER"):
        if ch not in cycle1_aligned:
            _log.warning(f"  {ch}: alignment missing, fallback to zero image.")
            cycle1_aligned[ch] = np.zeros_like(dapi1)

    # Step 3: Cycle2 DAPI -> Cycle1 DAPI 配准
    _log.debug("Step 3: Cycle2 DAPI -> Cycle1 DAPI registration")
    dapi1_n = norm(dapi1)
    cycle2_result = _align_cycle2_to_cycle1(dapi1_n, dapi2, ki67, logger_ref=_log)

    _log.info(
        f"  Cycle2 transform: angle={cycle2_result['transform']['angle']:.2f}deg, "
        f"shift=({cycle2_result['transform']['shift_y']:.1f}, "
        f"{cycle2_result['transform']['shift_x']:.1f}), NCC={cycle2_result['ncc']:.4f}"
    )

    # Step 4: 构建所有通道字典
    all_channels = {
        "DAPI":  cycle1_aligned["DAPI"],
        "HER2":  cycle1_aligned["HER2"],
        "PR":    cycle1_aligned["PR"],
        "ER":    cycle1_aligned["ER"],
        "DAPI2": cycle2_result["dapi2_aligned"],
        "KI67":  cycle2_result["ki67_aligned"],
    }

    # Step 5: 自动裁剪并保存
    output_dir = REGISTERED_DIR / block
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_result = _auto_crop_and_save(block, all_channels, output_dir)

    _log.info(
        f"  Final size: {crop_result['final_shape']}, "
        f"Output: {output_dir.name}/"
    )

    return {
        "transform": cycle2_result["transform"],
        "ncc": cycle2_result["ncc"],
        "final_shape": crop_result["final_shape"],
        "merged_path": crop_result["merged_path"],
        "output_files": crop_result["output_files"],
    }


# =====================================================================
# TMAe 配准流程（单 Cycle）
# =====================================================================

def _run_alignment_tmae(block, logger_ref=None):
    """
    TMAe 数据配准：仅 Cycle1 内部通道配准
    TMAe 没有 Cycle2，因此只做色差校正，不涉及旋转
    """
    _log = logger_ref or logger

    # 加载所有通道
    stitch_dir = get_stitched_path(block, "TMAe", None, "").parent

    channel_imgs = {}
    for ch in ["DAPI", "HER2", "PR", "ER"]:
        path = stitch_dir / f"{block}_TMAe_{ch}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
        channel_imgs[ch] = load_tiff(path)

    _log.debug(
        f"  Loaded shapes: DAPI={channel_imgs['DAPI'].shape}, "
        f"HER2={channel_imgs['HER2'].shape}, PR={channel_imgs['PR'].shape}, "
        f"ER={channel_imgs['ER'].shape}"
    )

    # Cycle1 内部配准
    _log.debug("Step 2: Cycle1 internal alignment (chromatic aberration)")
    dapi_ref = channel_imgs["DAPI"]
    non_dapi = {k: v for k, v in channel_imgs.items() if k != "DAPI"}
    aligned = _align_cycle1_internal(dapi_ref, non_dapi, logger_ref=_log)
    aligned["DAPI"] = dapi_ref

    # 保存（单通道文件，无合并 5 通道）
    output_dir = REGISTERED_DIR / block
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}
    file_map = {
        "DAPI":  f"{block}_Cycle1_DAPI.tif",
        "HER2":  f"{block}_Cycle1_HER2_aligned.tif",
        "PR":    f"{block}_Cycle1_PR_aligned.tif",
        "ER":    f"{block}_Cycle1_ER_aligned.tif",
    }

    # 自动裁剪
    bboxes = {name: find_content_bbox(img) for name, img in aligned.items()}
    y1 = max(b[0] for b in bboxes.values())
    y2 = min(b[1] for b in bboxes.values())
    x1 = max(b[2] for b in bboxes.values())
    x2 = min(b[3] for b in bboxes.values())
    crop_bbox = (y1, y2, x1, x2)
    cropped = {name: crop_to_bbox(img, crop_bbox) for name, img in aligned.items()}

    for name, fname in file_map.items():
        out_path = output_dir / fname
        save_tiff(out_path, cropped[name])
        output_files[name] = out_path

    # TMAe 也保存一个 4 通道合并文件（Fiji 兼容）
    merged = np.stack([cropped[ch] for ch in ["DAPI", "HER2", "PR", "ER"]], axis=0)
    merged_path = output_dir / f"{block}_merged_4channel.tif"
    tifffile.imwrite(str(merged_path), merged.astype(np.uint16), imagej=True)
    output_files["merged"] = merged_path

    final_shape = cropped["DAPI"].shape
    _log.info(f"  Final size: {final_shape}, Output: {output_dir.name}/")

    return {
        "transform": None,  # TMAe 无旋转变换
        "ncc": None,
        "final_shape": final_shape,
        "merged_path": merged_path,
        "output_files": output_files,
    }


# =====================================================================
# 公共 API
# =====================================================================

def check_alignment_done(block: str, dataset: str = "TMAd") -> bool:
    """
    检查配准是否已完成

    Args:
        block: Block 名称
        dataset: 数据集名称（"TMAd" 或 "TMAe"）
    """
    if dataset == "TMAd":
        # 检查合并文件是否存在
        path = REGISTERED_DIR / block / f"{block}_merged_5channel.tif"
        return path.exists()

    elif dataset == "TMAe":
        path = REGISTERED_DIR / block / f"{block}_merged_4channel.tif"
        return path.exists()

    return False


def run_alignment(
    block: str,
    dataset: str = "TMAd",
    force: bool = False,
) -> dict:
    """
    对单个 Block 执行图像配准

    Args:
        block: Block 名称
        dataset: 数据集名称 ("TMAd" 或 "TMAe")
        force: 是否强制重新配准（忽略已有结果）

    Returns:
        dict: {
            "status": "success" | "skipped" | "error",
            "block": str,
            "dataset": str,
            "transform": dict | None,   # 仅 TMAd
            "ncc": float | None,        # 最终 NCC 分数，仅 TMAd
            "final_shape": tuple,
            "merged_path": Path,
            "output_files": dict,
            "error": str | None
        }
    """
    logger.info(f"[Align] Starting block={block}, dataset={dataset}")

    # 检查是否已完成
    if not force and check_alignment_done(block, dataset):
        logger.info(f"[Align] {block}: already done, skipping")
        return {
            "status": "skipped",
            "block": block,
            "dataset": dataset,
            "merged_path": (
                REGISTERED_DIR / block / f"{block}_merged_5channel.tif"
                if dataset == "TMAd"
                else REGISTERED_DIR / block / f"{block}_merged_4channel.tif"
            ),
        }

    try:
        if dataset == "TMAd":
            result = _run_alignment_tmad(block, logger_ref=logger)
        elif dataset == "TMAe":
            result = _run_alignment_tmae(block, logger_ref=logger)
        else:
            return {
                "status": "error",
                "block": block,
                "dataset": dataset,
                "error": f"Unknown dataset: {dataset}",
            }

        logger.info(f"[Align] {block}: success")
        return {
            "status": "success",
            "block": block,
            "dataset": dataset,
            **result,
        }

    except FileNotFoundError as e:
        logger.error(f"[Align] {block}: input file missing - {e}")
        return {
            "status": "error",
            "block": block,
            "dataset": dataset,
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"[Align] {block}: {type(e).__name__} - {e}")
        return {
            "status": "error",
            "block": block,
            "dataset": dataset,
            "error": f"{type(e).__name__}: {e}",
        }


def align_tmad_block(block: str, force: bool = False) -> dict:
    """
    配准 TMAd 数据的快捷函数

    Args:
        block: Block 名称（如 "G2"）
        force: 是否强制重新配准

    Returns:
        dict: 配准结果
    """
    return run_alignment(block, dataset="TMAd", force=force)


def align_tmae_block(block: str, force: bool = False) -> dict:
    """
    配准 TMAe 数据的快捷函数

    Args:
        block: Block 名称（如 "D5"）
        force: 是否强制重新配准

    Returns:
        dict: 配准结果
    """
    return run_alignment(block, dataset="TMAe", force=force)
