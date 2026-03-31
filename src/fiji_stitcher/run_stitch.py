"""
Fiji stitching pipeline runner.
Usage: python run_stitch.py
"""
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径，以便 import fiji_stitcher
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from fiji_stitcher import (
    load_config,
    apply_cli_overrides,
    get_logger,
    get_all_level1_directories,
    process_all_level1_dirs,
    init_imagej,
)


def main():
    try:
        # 1. 加载配置（CLI 参数优先）
        cfg = load_config()
        cfg = apply_cli_overrides(cfg)

        # 2. 初始化日志
        logger = get_logger(cfg)
        logger.info("=" * 60)
        logger.info("Fiji stitching pipeline started")
        logger.info("=" * 60)
        print("=" * 60)
        print("Fiji stitching pipeline started")
        print("=" * 60)

        # 3. 初始化 Fiji（使用命令行模式）
        fiji_path = cfg["FIJI_PATH"]
        print(f"\nUsing Fiji at {fiji_path}")
        logger.info("Using Fiji at %s", fiji_path)
        ij = init_imagej(cfg)
        print("Fiji will be called via command line\n")

        # 4. 自动发现需要拼接的目录
        print("Scanning input data directories...")
        level1_dirs = get_all_level1_directories(cfg)
        
        # 如果指定了 --level1，只处理该块
        if cfg.get("ONLY_LEVEL1"):
            target_name = cfg["ONLY_LEVEL1"]
            level1_dirs = [d for d in level1_dirs if Path(d).name == target_name]
            if not level1_dirs:
                print(f"WARNING: specified --level1={target_name} not found")
                logger.warning(f"specified --level1={target_name} not found")
                return
        
        print(f"Discovered {len(level1_dirs)} directories to process:")
        for d in level1_dirs:
            print(f"  - {Path(d).name}")
        logger.info("Found %s level1 directories", len(level1_dirs))

        if not level1_dirs:
            print("WARNING: no input data discovered; check DEFAULT_ROOT_DIR config")
            logger.warning("no input data discovered")
            return

        # 5. 执行拼接流水线
        print(f"\nStarting stitching for {len(level1_dirs)} directories...\n")
        logger.info("Processing enabled, batch mode: %s", not cfg.get("INTERACTIVE", False))
        process_all_level1_dirs(level1_dirs, cfg, ij, logger)

        print("\nAll processing complete.")
        logger.info("Pipeline complete")
    except Exception as e:
        print(f"ERROR: Unexpected error in main: {e}")
        import traceback
        traceback.print_exc()
        logger.exception("Unexpected error in main")


if __name__ == "__main__":
    main()
