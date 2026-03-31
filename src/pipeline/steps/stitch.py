"""
Step 1: 图像拼接 (Stitching)
封装 Fiji/ImageJ 的 tile 拼接功能

需要 Fiji 已安装并配置好路径
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from pipeline.config import (
    CELLPOSE_ENV_PYTHON,
    CODE_DIR,
    CYCLE1_CHANNELS,
    CYCLE2_CHANNELS,
    FIJI_PATH,
    STITCHED_DIR,
    STITCH_PARAMS,
    get_block_source_type,
    get_raw_block_path,
    get_stitched_path,
)
from pipeline.utils.logging import get_logger

logger = get_logger("pipeline.stitch")


# =====================================================================
# Flat Tiles 自动通道分类
# =====================================================================

# Cycle1 文件名关键词 → 通道目录名
_CYCLE1_KEYWORD_MAP = {
    "w1dapi": "DAPI",
    "w1dapi": "DAPI",
    "w2gfp":  "HER2",
    "w2gfp":  "HER2",
    "w3cy3":  "PR",
    "w3cy3":  "PR",
    "w4cy5":  "ER",
    "w4cy5":  "ER",
}

# Cycle2 文件名关键词 → 通道目录名（Composite 格式：img_t1_z1_c1/c2）
_CYCLE2_COMPOSITE_KEYWORDS = ["composite", "img_t1_z1_c"]


def _has_channel_subdirs(block_path: Path, channels) -> bool:
    """检查 block 目录下是否有通道子目录"""
    return any((block_path / ch).is_dir() for ch in channels)


def _auto_organize_flat_tiles(block_path: Path, cycle: str) -> dict:
    """
    如果 block_path 下没有通道子目录，自动将 flat TIF 文件分类到对应通道目录。

    Cycle1: 文件含 w1DAPI/w2GFP/w3Cy3/w4Cy5 → DAPI/HER2/PR/ER/
    Cycle2: Composite 文件 → DAPI/ 和 KI67/（通过 tifffile 拆分）

    Returns:
        dict: {"status": "ok"|"skipped"|"error", "moved": int, "error": str}
    """
    import tifffile

    channels = ["DAPI", "HER2", "PR", "ER"] if cycle == "Cycle1" else ["DAPI", "KI67"]

    if _has_channel_subdirs(block_path, channels):
        return {"status": "skipped", "moved": 0, "error": ""}

    tif_files = sorted(block_path.glob("*.tif")) + sorted(block_path.glob("*.tiff"))
    if not tif_files:
        return {"status": "error", "moved": 0, "error": "No TIF files found"}

    total_moved = 0
    errors = []

    for tif_path in tif_files:
        name_lower = tif_path.stem.lower()

        # Cycle1: 关键词匹配
        if cycle == "Cycle1":
            assigned_ch = None
            for kw, ch in _CYCLE1_KEYWORD_MAP.items():
                if kw in name_lower:
                    assigned_ch = ch
                    break
            if assigned_ch is None:
                errors.append(f"无法识别通道: {tif_path.name}")
                continue

        # Cycle2: Composite 文件拆分
        elif cycle == "Cycle2":
            if "composite" in name_lower:
                # 拆分为 DAPI 和 KI67
                try:
                    img = tifffile.imread(str(tif_path))
                except Exception as e:
                    errors.append(f"无法读取 {tif_path.name}: {e}")
                    continue

                # 假设 shape[0] == 2 即为双通道
                if img.ndim != 3 or img.shape[0] != 2:
                    errors.append(f"非双通道图像，跳过: {tif_path.name} shape={img.shape}")
                    continue

                for ch_idx, ch_name in enumerate(["KI67", "DAPI"]):  # 注意顺序：ch0→KI67, ch1→DAPI
                    ch_dir = block_path / ch_name
                    ch_dir.mkdir(exist_ok=True)
                    out_path = ch_dir / f"{tif_path.stem}_{ch_name}.tif"
                    if out_path.exists():
                        continue
                    try:
                        tifffile.imwrite(str(out_path), img[ch_idx].astype(img.dtype))
                        total_moved += 1
                    except Exception as e:
                        errors.append(f"写入失败 {out_path.name}: {e}")
                # 将原始 Composite 移动到 Composite_source/
                src_dir = block_path / "Composite_source"
                src_dir.mkdir(exist_ok=True)
                dest = src_dir / tif_path.name
                if not dest.exists():
                    shutil.move(str(tif_path), str(dest))
                continue
            elif "img_t1_z1_c" in name_lower or any(k in name_lower for k in ("_c1", "_c2")):
                # 已经是单通道格式（img_t1_z1_c1 / img_t1_z1_c2）
                if "_c1" in name_lower or "_c2" not in name_lower:
                    # 假设 c1 → KI67, c2 → DAPI
                    ch_map = {"_c1": "KI67", "_c2": "DAPI"}
                    for suffix, ch_name in ch_map.items():
                        if suffix in name_lower:
                            ch_dir = block_path / ch_name
                            ch_dir.mkdir(exist_ok=True)
                            out_path = ch_dir / f"{tif_path.stem}{tif_path.suffix}"
                            if not out_path.exists():
                                shutil.copy2(str(tif_path), str(out_path))
                                total_moved += 1
                    continue
                else:
                    continue  # c2 已处理或跳过
            else:
                errors.append(f"无法识别 Cycle2 文件: {tif_path.name}")
                continue
        else:
            continue

        # Cycle1: 复制文件到通道目录
        if cycle == "Cycle1":
            ch_dir = block_path / assigned_ch
            ch_dir.mkdir(exist_ok=True)
            dest = ch_dir / tif_path.name
            if not dest.exists():
                shutil.copy2(str(tif_path), str(dest))
                total_moved += 1

    if errors and total_moved == 0:
        return {"status": "error", "moved": 0, "error": "; ".join(errors[:3])}

    logger.info(f"[Stitch] {block_path.name}/{cycle}: auto-organized {total_moved} files")
    return {"status": "ok", "moved": total_moved, "error": "; ".join(errors) if errors else ""}


def check_stitch_done(block: str, dataset: str) -> bool:
    """
    检查拼接是否已完成

    Args:
        block: Block 名称，如 "G2"
        dataset: 数据集名称，如 "TMAd"
    """
    if dataset == "TMAd":
        # TMAd: 检查 Cycle1 所有通道是否存在
        channels = CYCLE1_CHANNELS + CYCLE2_CHANNELS
        for ch in channels:
            if "DAPI" in ch and ch != "DAPI":
                continue
            # Cycle1 channels
            for cycle_ch in CYCLE1_CHANNELS:
                p = get_stitched_path(block, dataset, "Cycle1", cycle_ch)
                if not p.exists():
                    return False
            # Cycle2 channels
            for cycle_ch in CYCLE2_CHANNELS:
                p = get_stitched_path(block, dataset, "Cycle2", cycle_ch)
                if not p.exists():
                    return False
            break
        # 至少检查一个关键通道
        p = get_stitched_path(block, dataset, "Cycle1", "DAPI")
        return p.exists()

    elif dataset == "TMAe":
        p = get_stitched_path(block, dataset, None, "DAPI")
        return p.exists()

    return False


def run_stitching(
    block: str,
    dataset: str,
    force: bool = False,
    interactive: bool = False,
) -> dict:
    """
    对单个 Block 执行 Fiji 拼接

    Args:
        block: Block 名称
        dataset: 数据集名称 ("TMAd" 或 "TMAe")
        force: 是否强制重新拼接（忽略已有结果）
        interactive: 是否使用 Fiji 交互模式

    Returns:
        dict: {"status": "success" | "skipped" | "error", "outputs": [...], "error": str}
    """
    logger.info(f"[Stitch] Starting block={block}, dataset={dataset}")

    # 检查是否已完成
    if not force and check_stitch_done(block, dataset):
        logger.info(f"[Stitch] {block}: already done, skipping")
        return {"status": "skipped", "block": block, "dataset": dataset}

    # 确保 Fiji 路径存在
    fiji_exe = FIJI_PATH / "ImageJ-win64.exe"
    if not fiji_exe.exists():
        logger.error(f"[Stitch] Fiji not found at: {fiji_exe}")
        return {
            "status": "error",
            "block": block,
            "dataset": dataset,
            "error": f"Fiji not found at {fiji_exe}",
        }

    # 确定输入目录（自动适配 direct 或 Calculate_Data 结构）
    if dataset == "TMAd":
        src_type = get_block_source_type(block, dataset)
        if src_type == "none":
            return {
                "status": "error",
                "block": block,
                "error": f"Block '{block}' not found in either direct or Calculate_Data structure",
            }
        input_base = get_raw_block_path(block, dataset, "Cycle1")
        cycles = ["Cycle1", "Cycle2"]
    elif dataset == "TMAe":
        input_base = CODE_DIR.parent / "Raw_Data" / dataset / block
        if not input_base.exists():
            return {
                "status": "error",
                "block": block,
                "error": f"Input directory not found: {input_base}",
            }
        cycles = [None]
    else:
        return {
            "status": "error",
            "block": block,
            "error": f"Unknown dataset: {dataset}",
        }

    # 检查每个 Cycle 的输入是否存在（某些 block 可能只存在于某一个 Cycle）
    available_cycles = []
    for cyc in cycles:
        cyc_input = get_raw_block_path(block, dataset, cyc) if dataset == "TMAd" else input_base
        if cyc_input.exists():
            available_cycles.append(cyc)
    if not available_cycles:
        return {
            "status": "error",
            "block": block,
            "error": f"No cycle data found for block '{block}': checked {cycles}",
        }

    # 自动整理 flat TIF → 通道子目录（仅对 Calculate_Data 结构的 block）
    for cyc in available_cycles:
        cyc_input = get_raw_block_path(block, dataset, cyc) if dataset == "TMAd" else input_base
        if dataset == "TMAd" and cyc_input.exists():
            channels = CYCLE1_CHANNELS if cyc == "Cycle1" else CYCLE2_CHANNELS
            if not _has_channel_subdirs(cyc_input, channels):
                logger.info(f"[Stitch] {block}/{cyc}: flat tiles detected, auto-organizing ...")
                org = _auto_organize_flat_tiles(cyc_input, cyc)
                if org["status"] == "error":
                    logger.error(f"[Stitch] {block}/{cyc} auto-organize failed: {org['error']}")
                    return {
                        "status": "error",
                        "block": block,
                        "error": f"Auto-organize failed for {cyc}: {org['error']}",
                    }

    # 调用 fiji_stitcher 模块
    stitch_script = CODE_DIR / "fiji_stitcher" / "run_stitch.py"
    if not stitch_script.exists():
        return {
            "status": "error",
            "block": block,
            "error": f"Stitch script not found: {stitch_script}",
        }

    cmd = [
        sys.executable,  # 使用当前 Python 执行 Fiji 拼接脚本
        str(stitch_script),
    ]

    if not interactive:
        cmd.append("--batch")

    if force:
        cmd.append("--force-stitch")
    else:
        cmd.append("--skip-existing")

    # 添加数据集过滤，只处理指定的块
    cmd.append(f"--level1={block}")

    logger.info(f"[Stitch] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1小时超时
        )

        if result.returncode != 0:
            logger.error(f"[Stitch] {block} failed: {result.stderr[:500]}")
            return {
                "status": "error",
                "block": block,
                "dataset": dataset,
                "error": result.stderr[:500] if result.stderr else "Unknown error",
            }

        logger.info(f"[Stitch] {block}: success")
        return {
            "status": "success",
            "block": block,
            "dataset": dataset,
        }

    except subprocess.TimeoutExpired:
        logger.error(f"[Stitch] {block} timed out (>1 hour)")
        return {
            "status": "error",
            "block": block,
            "error": "Timeout after 1 hour",
        }
    except Exception as e:
        logger.error(f"[Stitch] {block} exception: {e}")
        return {
            "status": "error",
            "block": block,
            "error": str(e),
        }


def stitch_tmAd_block(block: str, force: bool = False) -> dict:
    """
    拼接 TMAd 数据的快捷函数

    Args:
        block: Block 名称（如 "G2"）
        force: 是否强制重新拼接

    Returns:
        dict: 拼接结果
    """
    return run_stitching(block, dataset="TMAd", force=force)


def stitch_tmae_block(block: str, force: bool = False) -> dict:
    """
    拼接 TMAe 数据的快捷函数

    Args:
        block: Block 名称（如 "D5"）
        force: 是否强制重新拼接

    Returns:
        dict: 拼接结果
    """
    return run_stitching(block, dataset="TMAe", force=force)
