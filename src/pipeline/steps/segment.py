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
from pathlib import Path
from typing import Optional

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
    p = get_segmentation_path(block, dataset, "features")
    return p.exists()


def _check_inputs_exist(block: str, dataset: str) -> bool:
    """检查配准结果是否存在"""
    reg_dir = get_registered_path(block, dataset)
    required = [
        f"{block}_Cycle1_DAPI.tif",
        f"{block}_Cycle1_HER2_aligned.tif",
        f"{block}_Cycle1_PR_aligned.tif",
        f"{block}_Cycle1_ER_aligned.tif",
    ]
    if dataset != "TMAe":
        required.append(f"{block}_KI67_aligned.tif")
    return all((reg_dir / f).exists() for f in required)


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

    reg_dir = get_registered_path(block, dataset)
    dapi = reg_dir / f"{block}_Cycle1_DAPI.tif"
    her2 = reg_dir / f"{block}_Cycle1_HER2_aligned.tif"
    pr = reg_dir / f"{block}_Cycle1_PR_aligned.tif"
    er = reg_dir / f"{block}_Cycle1_ER_aligned.tif"
    ki67 = reg_dir / f"{block}_KI67_aligned.tif" if dataset != "TMAe" else ""

    cmd = [
        str(CELLPOSE_ENV_PYTHON),
        str(seg_script),
        "--block-name", block,
        "--dapi", str(dapi),
        "--her2", str(her2),
        "--pr", str(pr),
        "--er", str(er),
        "--ki67", str(ki67),
        "--output-dir", str(SEGMENTATION_DIR),
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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2小时超时
        )

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
    features_path = get_segmentation_path(block, dataset, "features")
    if not features_path.exists():
        return 0
    try:
        import pandas as pd
        df = pd.read_csv(features_path)
        return len(df)
    except Exception:
        return 0
