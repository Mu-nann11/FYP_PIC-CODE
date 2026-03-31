"""
Step 0: Preprocessing
调用 Rename 模块，将扁平的原始数据按通道分类，拆分复合图像等。
"""
import logging
from ..config import get_raw_block_path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from Rename.preprocess import run_preprocess_block as core_run_preprocess_block

logger = logging.getLogger(__name__)

def run_preprocess(block: str, dataset: str, force: bool = False) -> dict:
    """
    执行 Step 0: Preprocessing
    """
    # 由于原始数据的改动在原地进行，通常不用检查是否被跳过，
    # 若已经是归类状态，core_run_preprocess_block 也会自动检测并跳过
    logger.info(f"--- [Step 0] Preprocessing  : block={block} dataset={dataset}")

    try:
        # 获取 Cycle1 和 Cycle2 的路径 (TMAd)
        if dataset == "TMAd":
            c1_path = get_raw_block_path(block, dataset, "Cycle1")
            c2_path = get_raw_block_path(block, dataset, "Cycle2")
        else:
            c1_path = get_raw_block_path(block, dataset, "Cycle1")
            c2_path = None

        results = core_run_preprocess_block(
            block_path_cycle1=c1_path,
            block_path_cycle2=c2_path,
            dataset=dataset,
            logger=logger
        )
        return {
            "status": "success",
            "details": results
        }
    except Exception as e:
        logger.error(f"Preprocessing failed for {block}: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

def check_preprocess_done(block: str, dataset: str) -> bool:
    """
    检查该 block 是否已经完成了 Preprocess。
    由于不同场景要求不同，通常通过检查 DAPI 或者某些目标文件夹存在没
    因为是原地修改数据且 preprocess 有自身的重试机制，此函数可以返回 True。
    或尝试检查子文件夹是否存在。为简单起见，这里假设它如果被设置了 from-step > 0 
    则源数据应该已经准备好了。
    """
    return True
