"""
校准模块 - 阴性对照阈值校准

Workflow:
    1. 第一步（阴性上限）：对阴性 Block 的细胞强度分布计算 mean + 2*SD
    2. 第二步（阳性分级）：在阳性群体上用迭代 Otsu 自动找分级点

数据依赖：
    阴性 Block 需先经 Step 1 (Fiji Stitching) → Step 2 (Alignment) → Step 3 (Segmentation)
    → results/segmentation/{block}/{block}_features.csv
"""

from .config import (
    BASE_DIR,
    CALIBRATION_DIR,
    CALIBRATION_PLOTS_DIR,
    CALIBRATION_PARAMS,
    HER2_POSITIVE_BLOCKS,
    NEGATIVE_BLOCKS,
    POSITIVE_REFERENCE_BLOCKS,
    SEGMENTATION_DIR,
    get_neg_feature_csv,
    get_neg_segmentation_root,
    get_pos_feature_csv,
    auto_discover_neg_blocks,
    build_neg_channel_values,
)

__all__ = [
    # 路径
    "BASE_DIR",
    "SEGMENTATION_DIR",
    "CALIBRATION_DIR",
    "CALIBRATION_PLOTS_DIR",
    # 阴性 Block 配置
    "NEGATIVE_BLOCKS",
    # 阳性参考 Block 配置
    "HER2_POSITIVE_BLOCKS",
    "POSITIVE_REFERENCE_BLOCKS",
    # 校准参数
    "CALIBRATION_PARAMS",
    # 辅助函数
    "get_neg_segmentation_root",
    "get_neg_feature_csv",
    "get_pos_feature_csv",
    "auto_discover_neg_blocks",
    "build_neg_channel_values",
]
