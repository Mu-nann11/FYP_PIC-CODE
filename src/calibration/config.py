"""
校准配置 - 阴性对照数据路径与通道定义

数据依赖：
    阴性 Block 需要先经过 Step 1 (Fiji Stitching) + Step 2 (Alignment)
    → 输出拼接配准后的图像 → 再跑 Step 3 (Segmentation) 获得细胞掩码和特征。
    分割特征 CSV 输出至 results/segmentation/{block}/{block}_features.csv

分割输出 CSV 列名约定（来自 cpsam_cyto_to_nucleus.py）：
    - DAPI_nuc_mean, HER2_nuc_mean, HER2_cyto_only_mean, HER2_nuc_cyto_ratio
    - ER_nuc_mean, PR_nuc_mean, Ki67_nuc_mean
    - (membrane_ring 列暂未实现，HER2 膜环强度暂用 HER2_cyto_only_mean)

阴性 Block 数据来源：
    - 3ch_Neg: Raw_Data/TMAd/Cycle1/Calculate_Data/3ch_Neg/{block}/
              含 ER/PR/HER2 三通道全阴的 Block
    - Ki67_Neg: Raw_Data/TMAd/Cycle1/Calculate_Data/Ki67_Neg/{block}/
               含 Ki67 阴性的 Block（Cycle1 4 通道格式，同 3ch_Neg）
               Cycle2 Ki67_Neg 瓦片为 Composite 格式，需单独处理

HER2 阳性分级 Block：
    - 从 TMAd Cycle1 主 Block（G2/B8/A10/J10 等）中选取 HER2 阳性 Block
    - 用于在阴性阈值以上做 Otsu 迭代分级（1+/2+/3+）
"""

from pathlib import Path
from typing import Optional

# ==================== 基础路径 ====================
BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")
CODE_DIR = BASE_DIR / "Code"
RAW_DATA_DIR = BASE_DIR / "Raw_Data"

# ==================== 分割结果路径 ====================
# Step 3 (Segmentation) 输出的特征 CSV
# 输出结构：results/segmentation/{block}/{block}_features.csv
SEGMENTATION_DIR = BASE_DIR / "results" / "segmentation"

# ==================== 校准输出路径 ====================
CALIBRATION_DIR = BASE_DIR / "results" / "calibration"
CALIBRATION_PLOTS_DIR = CALIBRATION_DIR / "plots"

# ==================== 阴性对照 Block 定义 ====================
# 用于第一步阈值（mean + 2*SD）
# blocks 留空则自动发现；手动填入可覆盖自动发现逻辑
NEGATIVE_BLOCKS = {
    # 3ch_Neg: ER/PR/HER2 全阴 Block（Cycle1 格式，14 瓦片 × 4 通道）
    # 分割输出 CSV 列：ER_nuc_mean, PR_nuc_mean, HER2_cyto_only_mean
    "3ch_Neg": {
        # 原始瓦片路径（由 Step 1 拼接后配准，再由 Step 3 分割）
        "raw_tile_root": RAW_DATA_DIR / "TMAd" / "Cycle1" / "Calculate_Data" / "3ch_Neg",
        # 分割结果根目录
        "segmentation_root": SEGMENTATION_DIR,
        # 自动发现的 block 名列表（也可手动填入固定 block）
        # 调整：排除 G2（错误数据），保留 H2 等待后续分割
        "blocks": ["A1", "D1", "E10", "H2", "H10", "J10"],
        # Cycle 标识（用于确定图像命名格式）
        "cycle": "Cycle1",
        # 用于校准的通道配置
        # intensity_type:  "nuclear" | "cytoplasmic" | "membrane_ring"
        # column:          CSV 中的列名（严格大小写）
        "channels": {
            "ER": {
                "intensity_type": "nuclear",
                "column": "ER_nuc_mean",
            },
            "PR": {
                "intensity_type": "nuclear",
                "column": "PR_nuc_mean",
            },
            # HER2 膜环强度 - 已实现，使用新的膜环特征列
            "HER2": {
                "intensity_type": "membrane_ring",
                "column": "HER2_membrane_ring_mean",
            },
        },
        "description": "ER/PR/HER2 全阴性对照（Cycle1 格式），用于确定阴性上限阈值",
    },

    # Ki67_Neg: Ki67 阴性 Block（Cycle1 格式，14 瓦片 × 4 通道）
    # 分割输出 CSV 列：Ki67_nuc_mean
    "Ki67_Neg": {
        "raw_tile_root": RAW_DATA_DIR / "TMAd" / "Cycle1" / "Calculate_Data" / "Ki67_Neg",
        "segmentation_root": SEGMENTATION_DIR,
        "blocks": ["A8", "D1", "G1", "H10"],
        # 约束：Ki67 阈值样本要求对应 block 在 Cycle2/Calculate_Data/Ki67_Neg 也存在
        # 用于保证 Ki67_Neg 组来源覆盖 Cycle1 + Cycle2 路径
        "required_companion_raw_roots": [
            RAW_DATA_DIR / "TMAd" / "Cycle2" / "Calculate_Data" / "Ki67_Neg",
        ],
        "cycle": "Cycle1",
        "channels": {
            "KI67": {
                "intensity_type": "nuclear",
                "column": "Ki67_nuc_mean",
            },
        },
        "description": "Ki67 阴性对照（Cycle1 格式），用于确定 Ki67 阴性上限阈值",
    },

    # Ki67_Neg_C2: Ki67 阴性 Block（Cycle2 Composite 格式，14 瓦片 × 2 通道）
    # 注意：Cycle2 瓦片为 Composite 图像（所有通道合在一张图）
    # 目前 Cycle2 格式尚未接入分割流水线，留空待扩展
    "Ki67_Neg_Cycle2": {
        "raw_tile_root": RAW_DATA_DIR / "TMAd" / "Cycle2" / "Calculate_Data" / "Ki67_Neg",
        "segmentation_root": SEGMENTATION_DIR,
        "blocks": ["A8", "D1", "G1", "H10"],
        "cycle": "Cycle2",
        "channels": {
            "KI67": {
                "intensity_type": "nuclear",
                "column": "Ki67_nuc_mean",
            },
        },
        "description": "Ki67 阴性对照（Cycle2 Composite 格式），待分割流水线扩展后使用",
    },
}

# ==================== HER2 阳性 Block 定义 ====================
# 用于第二步 HER2 阳性内部分级（Otsu 分 1+/2+/3+）
# 在 3ch_Neg 阴性阈值以上的 HER2 阳性细胞中，使用迭代 Otsu 找分级点
# 填入 TMAd Cycle1 主 Block（G2/B8/A10/J10 等）
HER2_POSITIVE_BLOCKS = {
    # 手动指定含有 HER2 阳性的 Block
    # 分割结果路径：results/segmentation/{block}/{block}_features.csv
    "blocks": [],
    # HER2 强度来源（与 NEGATIVE_BLOCKS 中保持一致）
    "column": "HER2_cyto_only_mean",
    "description": (
        "HER2 阳性 Block，用于 Otsu 分级 1+/2+/3+。"
        "需在 3ch_Neg 阴性阈值（mean_neg + 2*SD_neg）以上的细胞中进行分级。"
    ),
}

# ==================== 阳性参考 Block 定义（ER/PR/Ki67 分级）====================
# 用于第二步 ER/PR/Ki67 阳性内部分级（Otsu 自动找 2~3 个等级分界）
# 与 HER2 阳性分级逻辑相同：先确定阴性上限，再在阳性群体上用 Otsu 分级
POSITIVE_REFERENCE_BLOCKS = {
    "blocks": [],
    "description": (
        "ER/PR/Ki67 阳性参考 Block，用于在阳性群体上 Otsu 自动分级。"
        "实际分级时也可直接复用 HER2 阳性 Block（G2/B8/A10 等）。"
    ),
}

# ==================== 校准参数 ====================
CALIBRATION_PARAMS = {
    # 第一步：阴性上限阈值（mean + n*SD）
    "neg_threshold_n_sd": 2.0,

    # 第二步：阳性分级（迭代 Otsu 次数）
    # HER2: 0 → 1+ → 2+ → 3+（3 次迭代 → 4 个等级，3 个分界阈值）
    "her2_n_grades": 3,
    # ER/PR: 内部再分 2~3 个等级（留待临床标准确定后调整）
    "erpr_n_grades": 3,
    # Ki67: 默认仅阴/阳两级（Hotspot Otsu 模式由 scoring 模块单独处理）
    "ki67_n_grades": 2,

    # 最小细胞数阈值：block 中少于 N 个细胞时跳过该 block
    "min_cells_per_block": 10,

    # 可视化参数
    "histogram_bins": 80,
    "histogram_alpha": 0.6,
    "histogram_color_neg": "#4C72B0",   # 阴性分布颜色（蓝）
    "histogram_color_pos": "#C44E52",   # 阳性分布颜色（红）
    "threshold_line_color": "#FF9800", # 阈值线颜色（橙）
    "dpi": 150,

    # 输出格式
    "threshold_json_indent": 2,
}

# ==================== 辅助函数 ====================

def get_neg_segmentation_root(neg_key: str) -> Path:
    """获取阴性 Block 分割结果根目录，未指定时自动推断。"""
    cfg = NEGATIVE_BLOCKS.get(neg_key, {})
    seg_root = cfg.get("segmentation_root")
    if seg_root:
        return Path(seg_root)
    # 默认：SEGMENTATION_DIR / {neg_key}（如 results/segmentation/3ch_Neg/）
    return SEGMENTATION_DIR / neg_key


def get_neg_feature_csv(block: str, neg_key: str) -> Optional[Path]:
    """获取单个阴性 Block 的分割特征 CSV 路径。"""
    seg_root = get_neg_segmentation_root(neg_key)
    # 分割输出结构：{seg_root}/{block}/{block}_features.csv
    block_dir = seg_root / block
    csv_path = block_dir / f"{block}_features.csv"
    return csv_path if csv_path.exists() else None


def get_pos_feature_csv(block: str) -> Optional[Path]:
    """获取 HER2 阳性参考 Block 的分割特征 CSV 路径。"""
    block_dir = SEGMENTATION_DIR / block
    csv_path = block_dir / f"{block}_features.csv"
    return csv_path if csv_path.exists() else None


def auto_discover_neg_blocks(neg_key: str) -> list:
    """自动发现阴性 Block 目录中已存在的 block。"""
    seg_root = get_neg_segmentation_root(neg_key)
    if not seg_root.exists():
        return []
    discovered = [
        d.name for d in seg_root.iterdir()
        if d.is_dir() and (d / f"{d.name}_features.csv").exists()
    ]
    return sorted(discovered)


def build_neg_channel_values(block: str, neg_key: str, channel: str) -> Optional["np.ndarray"]:
    """
    加载单个阴性 Block 中指定通道的强度值数组。

    Returns:
        np.ndarray of intensity values, or None if CSV not found.
    """
    csv_path = get_neg_feature_csv(block, neg_key)
    if csv_path is None:
        return None

    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    col = NEGATIVE_BLOCKS.get(neg_key, {}).get("channels", {}).get(channel, {}).get("column")
    if col is None or col not in df.columns:
        return None

    values = df[col].dropna().values
    return values
