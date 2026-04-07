"""
Step 3: 细胞分割 (Segmentation)
封装 cpsam_cyto_to_nucleus.py 的调用

工作流程:
    Step 1: Cellpose v3 nuclei model -> nuclei_masks
    Step 2: CPSAM -> DAPI + HER2 -> cyto_masks
    Step 3: 细胞/细胞核标签匹配
    Step 4: 提取特征 -> CSV + 可视化

支持断点续传：每个步骤独立保存，中间失败不影响已完成的步骤
"""

import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from ..config import (
    CELLPOSE_ENV_PYTHON,
    CELLPOSE_MODEL_PATH,
    CODE_DIR,
    SEGMENTATION_DIR,
    SEGMENT_PARAMS,
    get_segmentation_path,
    get_registered_path,
)
from ..utils.logging import get_logger

logger = get_logger("pipeline.segment")


# =====================================================================
# 断点续传检查
# =====================================================================

def check_segmentation_done(block: str, dataset: str) -> bool:
    """
    检查分割是否已完成

    Args:
        block: Block 名称
        dataset: 数据集名称

    Returns:
        bool: 特征 CSV 文件存在则视为已完成
    """
    return _resolve_segmentation_feature_path(block, dataset).exists()


def _resolve_segmentation_feature_path(block: str, dataset: str) -> Path:
    """Resolve the current feature CSV path, with legacy fallbacks."""
    base_dir = get_segmentation_path(block, dataset)
    candidates = [
        get_segmentation_path(block, dataset, "features"),
        base_dir / f"{block}_features.csv",
        SEGMENTATION_DIR / block / f"{block}_features.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def _check_inputs_exist(block: str, dataset: str) -> bool:
    """检查配准结果是否存在"""
    reg_dir = get_registered_path(block, dataset, cycle="Cycle1")

    for ch in ["DAPI", "HER2", "PR", "ER"]:
        if _resolve_registered_input_path(reg_dir, block, dataset, ch) is None:
            return False

    if dataset != "TMAe":
        if _resolve_registered_input_path(reg_dir, block, dataset, "KI67") is None:
            return False

    return True


def _resolve_registered_input_path(
    reg_dir: Path,
    block: str,
    dataset: str,
    channel: str,
) -> Optional[Path]:
    """Resolve registered input path across legacy and current filename conventions."""
    ch = channel.upper()

    candidates = [
        reg_dir / f"{block}_{dataset}_Cycle1_{ch}.tif",
    ]

    # TMAe alignment currently writes files without the Cycle1 token.
    if dataset == "TMAe" and ch in {"DAPI", "HER2", "PR", "ER"}:
        candidates.append(reg_dir / f"{block}_{dataset}_{ch}.tif")

    if ch == "KI67":
        candidates.append(reg_dir / f"{block}_{dataset}_KI67.tif")

    for path in candidates:
        if path.exists():
            return path
    return None


# =====================================================================
# 核心分割逻辑
# =====================================================================

def run_segmentation(
    block: str,
    dataset: str = "TMAd",
    force: bool = False,
) -> dict:
    """
    对单个 Block 执行细胞分割

    Args:
        block: Block 名称
        dataset: 数据集名称（"TMAd" 或 "TMAe"）
        force: 是否强制重新分割

    Returns:
        dict: {"status": "success"|"skipped"|"error", "block": str, "error": str,
               "cell_count": int, "output_features": Path}
    """
    logger.info(f"[Segment] Starting block={block}, dataset={dataset}")

    # 检查是否已完成
    if not force and check_segmentation_done(block, dataset):
        logger.info(f"[Segment] {block}: already done, skipping")
        cell_count = _read_cell_count(block, dataset)
        return {
            "status": "skipped",
            "block": block,
            "dataset": dataset,
            "cell_count": cell_count,
            "output_features": get_segmentation_path(block, dataset, "features"),
        }

    # 检查输入
    if not _check_inputs_exist(block, dataset):
        logger.error(f"[Segment] {block}: registered images not found, run alignment first")
        return {
            "status": "error",
            "block": block,
            "dataset": dataset,
            "error": "Registered images not found, run alignment first",
        }

    # 调用 segmentation 脚本
    seg_script = CODE_DIR / "segmentation" / "cpsam_cyto_to_nucleus.py"
    if not seg_script.exists():
        logger.error(f"[Segment] Segmentation script not found: {seg_script}")
        return {
            "status": "error",
            "block": block,
            "error": f"Script not found: {seg_script}",
        }

    reg_dir = get_registered_path(block, dataset, cycle="Cycle1")
    dapi = _resolve_registered_input_path(reg_dir, block, dataset, "DAPI")
    her2 = _resolve_registered_input_path(reg_dir, block, dataset, "HER2")
    pr = _resolve_registered_input_path(reg_dir, block, dataset, "PR")
    er = _resolve_registered_input_path(reg_dir, block, dataset, "ER")
    ki67 = _resolve_registered_input_path(reg_dir, block, dataset, "KI67") if dataset != "TMAe" else None

    if any(p is None for p in (dapi, her2, pr, er)):
        return {
            "status": "error",
            "block": block,
            "dataset": dataset,
            "error": "Registered images not found, run alignment first",
        }

    # 根据数据集选择输出目录
    output_dir = SEGMENTATION_DIR / dataset

    cmd = [
        str(CELLPOSE_ENV_PYTHON),
        str(seg_script),
        "--block-name", block,
        "--dapi", str(dapi),
        "--her2", str(her2),
        "--pr", str(pr),
        "--er", str(er),
        "--ki67", str(ki67) if ki67 else "",
        "--output-dir", str(output_dir),
        "--model", str(CELLPOSE_MODEL_PATH),
        "--diameter", str(SEGMENT_PARAMS["diameter"]),
        "--flow-threshold", str(SEGMENT_PARAMS["flow_threshold"]),
        "--cellprob-threshold", str(SEGMENT_PARAMS["cellprob_threshold"]),
        "--min-nuc-area", str(SEGMENT_PARAMS["min_nuc_area"]),
        "--max-area-ratio", str(SEGMENT_PARAMS["max_area_ratio"]),
    ]

    if not SEGMENT_PARAMS.get("use_cellpose_nuclei", True):
        cmd.append("--no-cellpose-nuclei")

    logger.info(f"[Segment] Running: {' '.join(cmd)}")

    try:
        progress_cm = (
            tqdm(total=1, desc=f"[Step 3] {block}", unit="run", leave=False, dynamic_ncols=True)
            if tqdm is not None
            else nullcontext(None)
        )
        with progress_cm as progress:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2小时超时
            )
            if progress is not None:
                progress.update(1)

        if result.returncode != 0:
            logger.error(f"[Segment] {block} failed: {result.stderr[:500]}")
            return {
                "status": "error",
                "block": block,
                "dataset": dataset,
                "error": result.stderr[:500] if result.stderr else "Unknown error",
            }

        cell_count = _read_cell_count(block, dataset)
        features_path = get_segmentation_path(block, dataset, "features")

        logger.info(f"[Segment] {block}: success (cells={cell_count})")
        return {
            "status": "success",
            "block": block,
            "dataset": dataset,
            "cell_count": cell_count,
            "output_features": features_path,
        }

    except subprocess.TimeoutExpired:
        logger.error(f"[Segment] {block} timed out (>2 hours)")
        return {
            "status": "error",
            "block": block,
            "error": "Timeout after 2 hours",
        }
    except Exception as e:
        logger.error(f"[Segment] {block} exception: {e}")
        return {
            "status": "error",
            "block": block,
            "error": str(e),
        }


# =====================================================================
# 辅助函数
# =====================================================================

def _read_cell_count(block: str, dataset: str) -> int:
    """从特征 CSV 读取细胞数量"""
    features_path = _resolve_segmentation_feature_path(block, dataset)
    if not features_path.exists():
        return 0
    try:
        import pandas as pd
        df = pd.read_csv(features_path)
        return len(df)
    except Exception:
        return 0
