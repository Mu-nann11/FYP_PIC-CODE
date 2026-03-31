"""
专用脚本：拼接 Composite_source 目录中的图像（保持为单个 Composite 文件）
用法：python stitch_composite_source.py
"""
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from fiji_stitcher import (
    load_config,
    apply_cli_overrides,
    get_logger,
    init_imagej,
    configure_stitching_parameters,
    build_macro_command,
    execute_stitching_with_retry,
)
from fiji_stitcher.files import get_image_files


def stitch_composite_source(input_dir, output_dir, config, ij, logger):
    """拼接 Composite_source 目录中的所有图像（保持为单个 Composite 文件）"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Stitching composite source images from: %s", input_path)
    print(f"\n{'=' * 60}")
    print(f"拼接 Composite 图像（保持为单个文件）")
    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    print(f"{'=' * 60}\n")

    # 获取所有 tif 文件
    img_files = get_image_files(str(input_path), pattern="G2_TMAd_Composite_*.tif")
    if not img_files:
        logger.error("No image files found in %s", input_path)
        print("❌ 未找到图像文件")
        return False

    logger.info("Found %s image files to stitch", len(img_files))
    print(f"找到 {len(img_files)} 个图像文件")

    # 配置拼接参数
    params = configure_stitching_parameters(config, interactive=False)

    # 构建输出文件名
    fused_name = input_path.name  # "Composite_source"
    tile_cfg_name = f"TileConfiguration_{fused_name}.txt"

    # 检查输出文件是否已存在
    expected_output = output_path / f"{fused_name}.tif"
    if expected_output.exists():
        logger.info("Output file already exists, skipping: %s", expected_output)
        print(f"⏭️  输出文件已存在，跳过拼接: {expected_output.name}")
        return True

    # 构建 Fiji macro 命令
    macro = build_macro_command(
        input_dir=str(input_path),
        output_dir=str(output_path),
        file_pattern="G2_TMAd_Composite_{n}.tif",
        params=params,
        tile_config_name=tile_cfg_name,
    )

    logger.debug("Fiji macro:\n%s", macro)
    print("\n开始执行 Fiji 拼接...\n")

    # 执行拼接
    ok = execute_stitching_with_retry(
        ij, macro, logger, output_dir=output_path, max_retries=3
    )

    if not ok:
        logger.error("Stitching failed for %s", input_path)
        print("❌ 拼接失败")
        return False

    # 验证输出文件 - Fiji 会输出 img_t1_z1_c1, img_t1_z1_c2 等
    stitched_files = sorted(output_path.glob("img_t1_z1_c*"))
    if not stitched_files:
        logger.error("No stitched img_t1_z1_c* files found in %s", output_path)
        print("❌ 拼接宏已执行，但未找到输出文件")
        return False

    logger.info("Found %s stitched channel files: %s", len(stitched_files), [f.name for f in stitched_files])
    print(f"\n找到 {len(stitched_files)} 个拼接后的通道文件")

    # 将通道文件合并为单个 Composite 文件
    composite_output = output_path / f"{fused_name}.tif"
    if merge_channels_to_composite(stitched_files, composite_output, logger):
        logger.info("Composite file created: %s", composite_output)
        print(f"\n✅ Composite 文件已创建: {composite_output.name}")
        
        # 删除单独的通道文件
        for f in stitched_files:
            f.unlink()
            logger.info("Deleted channel file: %s", f.name)
        print("已删除单独的通道文件")
    else:
        logger.error("Failed to create composite file")
        print("❌ 创建 Composite 文件失败")
        return False

    # 清理其他临时文件
    for pattern in ["TileConfiguration_*.txt", "TileConfiguration_*.registered.txt"]:
        for f in output_path.glob(pattern):
            f.unlink(missing_ok=True)

    logger.info("Stitching completed successfully")
    print(f"\n✅ 拼接完成！结果保存于: {output_path}")
    print(f"\n输出文件: {composite_output.name}")

    return True


def merge_channels_to_composite(channel_files, output_path, logger):
    """使用 tifffile 将多个通道合并为单个 Composite TIFF 文件"""
    try:
        import tifffile
        import numpy as np
        
        logger.info("Merging %s channels into composite...", len(channel_files))
        print(f"\n正在合并 {len(channel_files)} 个通道...")
        
        # 读取所有通道
        channels = []
        for f in sorted(channel_files):
            logger.info("Reading channel: %s", f.name)
            img = tifffile.imread(str(f))
            channels.append(img)
            print(f"  - {f.name}: shape={img.shape}, dtype={img.dtype}")
        
        # 检查所有通道的尺寸是否一致
        shapes = [c.shape for c in channels]
        if len(set(shapes)) != 1:
            logger.error("Channel shapes don't match: %s", shapes)
            print(f"❌ 通道尺寸不一致: {shapes}")
            return False
        
        # 沿着第一个维度堆叠通道
        composite = np.stack(channels, axis=0) if len(channels) > 1 else channels[0]
        logger.info("Composite shape: %s", composite.shape)
        print(f"合并后 shape: {composite.shape}")
        
        # 保存为 TIFF
        tifffile.imwrite(str(output_path), composite)
        logger.info("Composite saved to: %s", output_path)
        print(f"\n✅ 已保存: {output_path.name}")
        
        return True
        
    except ImportError as e:
        logger.error("tifffile not available: %s", e)
        print(f"❌ 缺少 tifffile 模块: {e}")
        return False
    except Exception as e:
        logger.exception("Error merging channels: %s", e)
        print(f"❌ 合并通道时出错: {e}")
        return False


def main():
    # 加载配置
    cfg = load_config()
    cfg = apply_cli_overrides(cfg)

    # 初始化日志
    logger = get_logger(cfg)
    logger.info("=" * 60)
    logger.info("Composite 拼接脚本启动")
    logger.info("=" * 60)

    # 目标目录
    input_dir = Path("D:/Try_munan/FYP_LAST/Raw_Data/TMAd/Cycle2/G2/Composite_source")
    output_dir = Path(cfg.get("STITCHED_PARENT_DIR")) / "TMAd" / "Cycle2" / "G2"

    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        logger.error("Input directory not found: %s", input_dir)
        return

    # 初始化 ImageJ/Fiji
    fiji_path = cfg["FIJI_PATH"]
    print(f"\n初始化 Fiji ({fiji_path})，这可能需要几分钟...")
    logger.info("Initializing ImageJ at %s", fiji_path)
    ij = init_imagej(cfg)
    print("✅ Fiji 初始化完成\n")
    logger.info("ImageJ initialized")

    # 执行拼接
    success = stitch_composite_source(input_dir, output_dir, cfg, ij, logger)

    if success:
        print("\n🎉 处理完毕！")
        logger.info("Composite stitching complete")
    else:
        print("\n❌ 处理失败，请检查日志")
        logger.error("Composite stitching failed")


if __name__ == "__main__":
    main()
