"""
TMA 图像分析统一流水线 - 配置文件
集中管理所有硬编码路径和共享常量
"""

from pathlib import Path

# ==================== 基础路径 ====================
BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")
CODE_DIR = BASE_DIR / "Code"
RAW_DATA_DIR = BASE_DIR / "Raw_Data"

# ==================== 结果输出路径 ====================
STITCHED_DIR = BASE_DIR / "results" / "stitched"
REGISTERED_DIR = BASE_DIR / "results" / "registered"
SEGMENTATION_DIR = BASE_DIR / "results" / "segmentation"
CROP_DIR = BASE_DIR / "results" / "crop"

# ==================== 负控数据输出路径（单独分支）====================
NEGCTRL_BASE = BASE_DIR / "results" / "negative_controls"
NEGCTRL_REGISTERED = NEGCTRL_BASE / "registered"
NEGCTRL_SEGMENTATION = NEGCTRL_BASE / "segmentation"
NEGCTRL_CLINICAL_REPORTS = NEGCTRL_BASE / "clinical_reports"
NEGCTRL_PIPELINE_REPORTS = NEGCTRL_BASE / "pipeline_reports"

# ==================== 环境路径 ====================
FIJI_PATH = Path(r"D:\fiji-stable-win64-jdk\Fiji.app")
FIJI_EXE = FIJI_PATH / "ImageJ-win64.exe"
CELLPOSE_ENV_PYTHON = Path(r"D:\Miniconda3\envs\cellpose\python.exe")
CELLPOSE_MODEL_PATH = Path(r"D:\Try_munan\Cellpose_model\model2\models\her2_wholecell_v3")

# ==================== 通道配置 ====================
CYCLE1_CHANNELS = ["DAPI", "HER2", "PR", "ER"]
CYCLE2_CHANNELS = ["DAPI", "KI67"]
ALL_CHANNELS = CYCLE1_CHANNELS + ["KI67"]  # 5通道合并

# ==================== 数据集信息 ====================
# 数据来源类型
#   "direct"   - Cycle1/{block}/ 和 Cycle2/{block}/（直接子目录）
#   "calc"     - Cycle1/Calculate_Data/3ch_Neg/{block}/ 和 Cycle2/Calculate_Data/3ch_Neg/{block}/
BLOCK_SOURCE_TYPE = "direct"   # 默认 "direct"；如需切换可改为 "calc"

DATASETS = {
    "TMAd": {
        "root": RAW_DATA_DIR / "TMAd",
        "stitched_output": STITCHED_DIR / "TMAd",
        "registered_output": REGISTERED_DIR / "TMAd",
        "segmentation_output": SEGMENTATION_DIR,
        "channels_cycle1": CYCLE1_CHANNELS,
        "channels_cycle2": CYCLE2_CHANNELS,
        "has_cycle2": True,
        "channel_source_dir": None,  # 由 Cycle 子目录提供
    },
    "TMAe": {
        "root": RAW_DATA_DIR / "TMAe",
        "stitched_output": STITCHED_DIR / "TMAe",
        "registered_output": REGISTERED_DIR / "TMAe",
        "segmentation_output": SEGMENTATION_DIR,
        "channels": ["DAPI", "HER2", "PR", "ER"],
        "has_cycle2": False,
        "channel_source_dir": None,
    },
}

# ==================== 拼接参数 ====================
STITCH_PARAMS = {
    "fusion_method": "Linear Blending",
    "regression_threshold": "0.30",
    "max_displacement": "2.50",
    "absolute_displacement": "3.50",
    "computation_mode": "Save memory (but be slower)",
    "image_output": "Write to disk",
    "skip_existing": True,
}

# ==================== 配准参数 ====================
ALIGN_PARAMS = {
    "rotation_range": (-5.0, 5.0),
    "rotation_step": 0.25,
    "upsample_factor": 100,
    "size_tolerance": 0.8,
    "cycle1_channel_order": CYCLE1_CHANNELS,
    "cycle2_channel_order": CYCLE2_CHANNELS,
}

# ==================== 分割参数 ====================
SEGMENT_PARAMS = {
    "diameter": 30,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "min_nuc_area": 30,
    "max_area_ratio": 0.8,
    "use_cellpose_nuclei": True,
    "channels": [0, 1],  # [DAPI, HER2] for CPSAM
}

# ==================== 文件名模板 ====================
FILENAME_TEMPLATES = {
    # Stitched 输出文件名
    # TMAd Cycle1: {BLOCK}_TMAd_Cycle1_{CHANNEL}.tif
    # TMAd Cycle2: {BLOCK}_TMAd_Cycle2_{CHANNEL}.tif
    # TMAe: {BLOCK}_TMAe_{CHANNEL}.tif
    "stitched_cycle1": "{BLOCK}_TMAd_Cycle1_{CHANNEL}.tif",
    "stitched_cycle2": "{BLOCK}_TMAd_Cycle2_{CHANNEL}.tif",
    "stitched_tmae": "{BLOCK}_TMAe_{CHANNEL}.tif",

    # Registered 输出文件名（新格式包含 Dataset）
    "registered_cycle1_dapi": "{BLOCK}_{DATASET}_Cycle1_DAPI.tif",
    "registered_cycle1_her2": "{BLOCK}_{DATASET}_Cycle1_HER2.tif",
    "registered_cycle1_pr": "{BLOCK}_{DATASET}_Cycle1_PR.tif",
    "registered_cycle1_er": "{BLOCK}_{DATASET}_Cycle1_ER.tif",
    "registered_ki67": "{BLOCK}_{DATASET}_KI67.tif",
    "registered_merged_5ch": "{BLOCK}_{DATASET}_merged_5channel.tif",
    "registered_merged_4ch": "{BLOCK}_{DATASET}_merged_4channel.tif",

    # Segmentation 输出文件名
    "seg_cyto_masks": "{BLOCK}_cyto_masks.tif",
    "seg_nuclei_masks": "{BLOCK}_nuclei_masks.tif",
    "seg_cell_masks": "{BLOCK}_cell_masks.tif",
    "seg_features": "{BLOCK}_{DATASET}_features.csv",
    "seg_features_core": "{BLOCK}_{DATASET}_features_core.csv",
    "seg_overlay": "{BLOCK}_overlay.png",
}


def get_stitched_path(block: str, dataset: str, cycle: str = None, channel: str = None) -> Path:
    """
    获取拼接后图像的路径

    Args:
        block: Block 名称，如 "G2"
        dataset: 数据集名称，如 "TMAd" 或 "TMAe"
        cycle: 轮次，"Cycle1", "Cycle2" 或 None (TMAe)
        channel: 通道名称，如 "DAPI"
    """
    ds = DATASETS[dataset]
    # Stitched 实际输出结构：{STITCHED_DIR}/{dataset}/{cycle}/{block}/{fname}.tif
    out_dir = STITCHED_DIR / dataset

    if dataset == "TMAd":
        out_dir = out_dir / cycle / block
        fname = FILENAME_TEMPLATES[f"stitched_{cycle.lower()}"].format(
            BLOCK=block, CHANNEL=channel.upper()
        )
    else:
        out_dir = out_dir / block
        fname = FILENAME_TEMPLATES["stitched_tmae"].format(
            BLOCK=block, CHANNEL=channel.upper()
        )

    return out_dir / fname


def _is_negative_control(block: str, dataset: str) -> bool:
    """
    判断 block 是否为负控数据（来自 Calculate_Data）。
    """
    source_type = get_block_source_type(block, dataset)
    return source_type.startswith("calc_")


def get_registered_path(
    block: str,
    dataset: str,
    cycle: str = None,
    channel: str = None,
    file_type: str = None,
) -> Path:
    """
    获取配准后图像的路径（支持普通样本和负控数据分离）

    Args:
        block: Block 名称
        dataset: 数据集名称（如 "TMAd" 或 "TMAe"）
        channel: 通道名称，如果为 None 则返回目录路径
    """
    is_negctrl = _is_negative_control(block, dataset)

    if dataset == "TMAd":
        if cycle is None:
            cycle = "Cycle1"
        if is_negctrl:
            base_dir = NEGCTRL_REGISTERED / dataset / cycle / block
        else:
            base_dir = REGISTERED_DIR / dataset / cycle / block
    else:  # TMAe
        if is_negctrl:
            base_dir = NEGCTRL_REGISTERED / dataset / "Cycle1" / block
        else:
            base_dir = REGISTERED_DIR / dataset / "Cycle1" / block

    if channel is None and file_type is None:
        return base_dir

    if file_type == "merged_5ch":
        fname = FILENAME_TEMPLATES["registered_merged_5ch"].format(
            BLOCK=block,
            DATASET=dataset,
        )
    elif file_type == "merged_4ch":
        fname = FILENAME_TEMPLATES["registered_merged_4ch"].format(
            BLOCK=block,
            DATASET=dataset,
        )
    elif file_type == "params":
        fname = f"{block}_alignment_params.json"
    elif channel:
        if channel.upper() == "DAPI":
            fname = FILENAME_TEMPLATES["registered_cycle1_dapi"].format(
                BLOCK=block,
                DATASET=dataset,
            )
        elif channel.upper() == "HER2":
            fname = FILENAME_TEMPLATES["registered_cycle1_her2"].format(
                BLOCK=block,
                DATASET=dataset,
            )
        elif channel.upper() == "PR":
            fname = FILENAME_TEMPLATES["registered_cycle1_pr"].format(
                BLOCK=block,
                DATASET=dataset,
            )
        elif channel.upper() == "ER":
            fname = FILENAME_TEMPLATES["registered_cycle1_er"].format(
                BLOCK=block,
                DATASET=dataset,
            )
        elif channel.upper() == "KI67":
            fname = FILENAME_TEMPLATES["registered_ki67"].format(
                BLOCK=block,
                DATASET=dataset,
            )
        else:
            fname = f"{block}_{channel.upper()}.tif"
    else:
        fname = None

    return base_dir / fname if fname else base_dir


def get_segmentation_path(block: str, dataset: str, file_type: str = None) -> Path:
    """
    获取分割结果的路径（统一按 dataset/block 组织）

    Args:
        block: Block 名称
        dataset: 数据集名称
        file_type: 文件类型，如 "features", "overlay", "cell_masks" 等

    Returns:
        Path: 分割结果文件的路径
    """
    # Current standard layout: results/segmentation/{dataset}/{block}/
    base_dir = SEGMENTATION_DIR / dataset / block

    if file_type is None:
        return base_dir

    type_map = {
        "features": "seg_features",
        "features_core": "seg_features_core",
        "overlay": "seg_overlay",
        "cyto_masks": "seg_cyto_masks",
        "nuclei_masks": "seg_nuclei_masks",
        "cell_masks": "seg_cell_masks",
    }
    fname = FILENAME_TEMPLATES[type_map[file_type]].format(BLOCK=block, DATASET=dataset)
    return base_dir / fname


def get_block_source_type(block: str, dataset: str) -> str:
    """
    判断某个 block 属于哪种数据来源。

    Returns:
        "direct"  - Raw_Data/{dataset}/{cycle}/{block}/ 存在
        "calc"    - Raw_Data/{dataset}/{cycle}/Calculate_Data/3ch_Neg/{block}/ 存在
        "none"    - 两种来源都不存在
    """
    ds = DATASETS[dataset]
    root = ds["root"]

    if dataset == "TMAd":
        direct_c1 = root / "Cycle1" / block
        calc_c1 = root / "Cycle1" / "Calculate_Data" / "3ch_Neg" / block
        calc_k1 = root / "Cycle1" / "Calculate_Data" / "Ki67_Neg" / block
        # 只要 Cycle1 下存在就算
        if direct_c1.exists():
            return "direct"
        if calc_c1.exists():
            return "calc_3ch"
        if calc_k1.exists():
            return "calc_ki67"
        return "none"
    elif dataset == "TMAe":
        return "direct" if (root / block).exists() else "none"
    return "none"


def get_raw_block_path(block: str, dataset: str, cycle: str) -> Path:
    """
    获取 block 的原始数据根目录（Cycle1 或 Cycle2）。

    自动识别数据来源（direct 或 calc）。

    Args:
        block:   Block 名称，如 "G2" 或 "H2"
        dataset: 数据集名称，如 "TMAd"
        cycle:   "Cycle1" 或 "Cycle2"
    """
    ds = DATASETS[dataset]
    root = ds["root"]

    if dataset == "TMAd":
        direct = root / cycle / block
        calc_3ch = root / cycle / "Calculate_Data" / "3ch_Neg" / block
        calc_ki67 = root / cycle / "Calculate_Data" / "Ki67_Neg" / block
        if direct.exists():
            return direct
        if calc_3ch.exists():
            return calc_3ch
        if calc_ki67.exists():
            return calc_ki67
        return direct  # 返回 direct 路径，调用方自行处理不存在的情况
    elif dataset == "TMAe":
        return root / block
    return root


def discover_blocks(dataset: str) -> list:
    """
    自动发现数据集中的所有 blocks，同时支持两种目录结构：
      - 直接结构：Raw_Data/{dataset}/{cycle}/{block}/
      - Calculate_Data 结构：Raw_Data/{dataset}/{cycle}/Calculate_Data/3ch_Neg/{block}/

    同时扫描 Cycle1 和 Cycle2，以发现只存在于其中一个 Cycle 里的 blocks。

    Returns:
        blocks: Block 名称列表（去重，自动排序）
    """
    ds = DATASETS[dataset]
    root = ds["root"]

    if dataset == "TMAd":
        found = set()

        for cycle in ("Cycle1", "Cycle2"):
            cycle_dir = root / cycle
            if not cycle_dir.exists():
                continue

            for item in sorted(cycle_dir.iterdir()):
                if not item.is_dir():
                    continue
                if item.name in ("logs", "tmp", "Composite_source"):
                    continue

                # 直接 block（如 Cycle1/G2/、Cycle1/A10/）
                if item.name != "Calculate_Data":
                    found.add(item.name)
                    continue

                # Calculate_Data/{sub}/ 下的 blocks（如 Cycle1/Calculate_Data/3ch_Neg/H2/）
                for sub in ("3ch_Neg", "Ki67_Neg"):
                    sub_dir = item / sub
                    if sub_dir.exists():
                        for block_dir in sorted(sub_dir.iterdir()):
                            if block_dir.is_dir() and block_dir.name not in ("logs", "tmp"):
                                found.add(block_dir.name)

        return sorted(found)

    else:
        # TMAe: blocks are direct children
        blocks = []
        for item in root.iterdir():
            if item.is_dir() and item.name not in ["logs", "tmp"]:
                blocks.append(item.name)
        return sorted(blocks)
