"""
TMA Pipeline - 图像拼接、配准、分割统一流水线
"""

from .config import (
    BASE_DIR,
    CODE_DIR,
    RAW_DATA_DIR,
    STITCHED_DIR,
    REGISTERED_DIR,
    SEGMENTATION_DIR,
    CROP_DIR,
    FIJI_PATH,
    FIJI_EXE,
    CELLPOSE_ENV_PYTHON,
    CELLPOSE_MODEL_PATH,
    CYCLE1_CHANNELS,
    CYCLE2_CHANNELS,
    ALL_CHANNELS,
    DATASETS,
    STITCH_PARAMS,
    ALIGN_PARAMS,
    SEGMENT_PARAMS,
    FILENAME_TEMPLATES,
    get_stitched_path,
    get_registered_path,
    get_segmentation_path,
    discover_blocks,
)

__version__ = "1.0.0"
