from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .files import get_file_pattern, get_image_files
from .stitching import (
    configure_stitching_parameters,
    build_macro_command,
    build_macro_command_from_tile_config,
    execute_stitching_with_retry,
)
from .outputs import (
    validate_and_open_result,
    _snapshot_candidates,
)
from .ui import timeout_input

DEFAULT_CHANNEL_ORDER = ["DAPI", "HER2", "PR", "ER"]


def _channel_order_from_config(config):
    try:
        channels = config.get("LOADER", {}).get("CHANNELS", None)
    except Exception:
        channels = None

    if not channels:
        return DEFAULT_CHANNEL_ORDER

    out = [str(c).strip() for c in channels if str(c).strip()]
    return out if out else DEFAULT_CHANNEL_ORDER


def _channel_order_for_stitch(config, cycle_name):
    """
    根据 cycle 名称返回对应的通道列表。
    Cycle2 使用 LOADER.CYCLE2_CHANNELS，否则使用 LOADER.CHANNELS。
    """
    if cycle_name and "cycle2" in str(cycle_name).lower():
        try:
            ch = config.get("LOADER", {}).get("CYCLE2_CHANNELS", None)
        except Exception:
            ch = None
        if ch:
            out = [str(c).strip() for c in ch if str(c).strip()]
            if out:
                return out
    return _channel_order_from_config(config)


def run_stitch_for_channel(
    level1,
    channel,
    params,
    config,
    ij,
    logger,
    output_dir,
    fused_prefix=None,
    layout_file=None,
):
    ch_dir = level1 / channel
    if not ch_dir.is_dir():
        logger.error("Channel dir not found: %s", ch_dir)
        print("❌ 未找到通道目录: %s" % ch_dir)
        return False, None

    pattern = None
    if layout_file is None:
        pattern = get_file_pattern(str(ch_dir), interactive=config["INTERACTIVE"])
        if not pattern:
            logger.error("No pattern for %s", ch_dir)
            print("❌ %s 下无法推断图像文件模式，跳过 %s" % (ch_dir, channel))
            return False, None

    img_files = get_image_files(str(ch_dir), pattern=pattern)
    if not img_files:
        logger.error("No image files for %s pattern=%s", ch_dir, pattern)
        print("❌ %s 下未找到匹配 %s 的图像文件，跳过 %s" % (ch_dir, pattern, channel))
        return False, None

    logger.info("Channel %s: found %s files (%s)", channel, len(img_files), pattern or "from tile config")
    print("ℹ️ 通道 %s：找到 %s 个匹配文件，开始拼接" % (channel, len(img_files)))
    
    # 使用传入的 fused_prefix 或回退到旧的命名规则
    if fused_prefix is None:
        fused_prefix = level1.name
    
    fused_name = "%s_%s" % (fused_prefix, channel)
    tile_cfg_name = "TileConfiguration_%s.txt" % fused_name
    
    # 检查是否启用跳过已存在文件的配置
    skip_existing = config.get("STITCH_SKIP_EXISTING", True)
    
    # 检查输出文件是否已存在，如果存在则跳过拼接
    expected_output = output_dir / ("%s.tif" % fused_name)
    if skip_existing and expected_output.exists():
        logger.info("Output file already exists, skipping stitching: %s", expected_output)
        print("⏭️  输出文件已存在，跳过拼接: %s" % expected_output.name)
        # 直接返回已存在的文件路径，不再调用 validate_and_open_result
        return True, expected_output
    
    before_candidates = _snapshot_candidates(output_dir)

    if layout_file is None:
        macro = build_macro_command(
            input_dir=str(ch_dir),
            output_dir=str(output_dir),
            file_pattern=pattern,
            params=params,
            tile_config_name=tile_cfg_name,
        )
    else:
        macro = build_macro_command_from_tile_config(
            input_dir=str(ch_dir),
            output_dir=str(output_dir),
            layout_file=str(layout_file),
            params=params,
        )
    logger.debug("Macro for %s:\n%s", ch_dir, macro)

    ok = execute_stitching_with_retry(ij, macro, logger, output_dir=output_dir, max_retries=3)
    if not ok:
        logger.error("Stitching failed for %s %s", level1.name, channel)
        print("❌ %s 通道 %s 拼接失败" % (level1.name, channel))
        return False, None

    result = validate_and_open_result(
        output_dir,
        config,
        fused_name,
        logger,
        before_candidates,
    )
    if result is None:
        logger.error("No output tiff for %s %s", level1.name, channel)
        print("❌ %s 通道 %s 宏已执行，但未找到输出文件" % (level1.name, channel))
        return False, None

    return True, result


def check_channel_sizes(results, logger):
    try:
        import tifffile
    except ImportError:
        logger.warning("tifffile not installed; skip size check")
        return True

    info = {}
    for ch, p in results.items():
        try:
            if not p.exists():
                logger.error("File does not exist for channel size check: %s", p)
                continue
            arr = tifffile.imread(str(p))
            info[ch] = {"shape": arr.shape, "dtype": str(arr.dtype)}
        except Exception as e:
            logger.exception("Read tiff failed for %s: %s", p, e)

    if len(info) <= 1:
        return True

    shapes = dict((ch, d["shape"]) for ch, d in info.items())
    dtypes = dict((ch, d["dtype"]) for ch, d in info.items())

    shapes_match = len(set(shapes.values())) == 1
    dtypes_match = len(set(dtypes.values())) == 1

    if shapes_match and dtypes_match:
        logger.info("All channels match in shape and dtype: %s / %s", list(shapes.values())[0], list(dtypes.values())[0])
        print("✅ 四个通道输出一致：shape=%s, dtype=%s" % (list(shapes.values())[0], list(dtypes.values())[0]))
        return True

    print("⚠️ 注意：四个通道输出格式不一致，请检查该块拼接结果")
    logger.warning("Channel info differs:")
    for ch, i in info.items():
        logger.warning("  %s: %s", ch, i)
        print("  %s: %s" % (ch, i))
    return False


def _build_layout_file_from_reference(ref_registered_path, out_path, from_channel, to_channel):
    src = Path(ref_registered_path)
    dst = Path(out_path)
    text = src.read_text(encoding="utf-8", errors="ignore")

    from_suffix = f"_{from_channel}.tif"
    to_suffix = f"_{to_channel}.tif"

    lines = []
    for line in text.splitlines():
        if from_suffix in line:
            lines.append(line.replace(from_suffix, to_suffix))
        else:
            lines.append(line)

    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _derive_output_structure(raw_level1_path: Path, config: dict) -> Tuple[Path, str]:
    """
    Derive output directory and filename prefix from a raw data level-1 path.

    Input → Output examples:
      Raw_Data/TMAe/F2           → output_dir=TMAe/F2,         prefix=F2_TMAe
      Raw_Data/TMAd/Cycle1/A3    → output_dir=TMAd/Cycle1/A3,   prefix=A3_TMAd_Cycle1
      Raw_Data/TMAd/Cycle2/A3    → output_dir=TMAd/Cycle2/A3,   prefix=A3_TMAd_Cycle2
      Raw_Data/TMAd/F2           → output_dir=TMAd/F2,         prefix=F2_TMAd

    Returns: (output_dir, fused_name_prefix)
    """
    raw_root = Path(config.get("DEFAULT_ROOT_DIR", "/data"))
    raw_data_name = config.get("RAW_DATA_DIR_NAME", "Raw_Data")
    stitched_parent = Path(config["STITCHED_PARENT_DIR"])

    try:
        relative = raw_level1_path.relative_to(raw_root / raw_data_name)
    except ValueError:
        sample_dir = raw_level1_path.name
        return stitched_parent / sample_dir, raw_level1_path.name

    parts = relative.parts  # e.g. ('TMAd', 'F2') or ('TMAd', 'Cycle1', 'A3')

    if len(parts) == 2:
        dataset, sample = parts
        output_dir = stitched_parent / dataset / sample
        if sample.lower() in ("cycle1", "cycle2"):
            fused_prefix = f"{dataset}_{sample}"
        else:
            fused_prefix = f"{sample}_{dataset}"

    elif len(parts) >= 3:
        dataset = parts[0]
        if parts[1].lower().startswith("cycle"):
            # TMAd/Cycle1/A3  → output_dir=TMAd/Cycle1/A3, prefix=A3_TMAd_Cycle1
            cycle = parts[1]
            sample = parts[2]
            output_dir = stitched_parent / dataset / cycle / sample
            fused_prefix = f"{sample}_{dataset}_{cycle}"
        else:
            output_dir = stitched_parent / parts[0] / "_".join(parts[1:])
            fused_prefix = "_".join(parts[1:]) + "_" + parts[0]
    else:
        dataset = parts[0]
        output_dir = stitched_parent / dataset
        fused_prefix = dataset

    return output_dir, fused_prefix


def process_level1_sequential(level1_path, config, ij, logger):
    level1 = Path(level1_path)
    logger.info("Processing level1 (sequential, multi-channel): %s", level1)

    print("\n" + "=" * 60)
    print("开始处理一级目录: %s" % level1.name)
    print("完整路径: %s" % level1)
    print("=" * 60)

    params = configure_stitching_parameters(config, interactive=config["INTERACTIVE"])

    # 根据原始路径推断输出目录和文件名
    output_dir, fused_prefix = _derive_output_structure(level1, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)
    logger.info("Output filename prefix: %s", fused_prefix)

    # 判断是否为 Cycle2：只有通道子目录存在时才拼接
    # 从路径推断 cycle 名称
    raw_data_path = Path(config.get("DEFAULT_ROOT_DIR", "/data")) / config.get("RAW_DATA_DIR_NAME", "Raw_Data")
    try:
        rel_path = level1.relative_to(raw_data_path)
        path_parts = rel_path.parts
    except ValueError:
        path_parts = [level1.name]
    
    # 确定 cycle 名称
    if len(path_parts) >= 2 and path_parts[1].lower().startswith("cycle"):
        cycle_name = path_parts[1]
    else:
        cycle_name = level1.name
    
    channels = _channel_order_for_stitch(config, cycle_name)

    # Cycle2 + 仅 Composite（根目录多 tif、无通道子目录）→ 跳过
    allow_cycle2_composite = bool(config.get("STITCH_ALLOW_CYCLE2_COMPOSITE", False))
    if "cycle2" in str(cycle_name).lower() and not allow_cycle2_composite:
        has_channel_dirs = any((level1 / ch).is_dir() for ch in channels)
        if not has_channel_dirs:
            # 检查根目录是否有 tif 文件（说明是未拆分的 Composite）
            root_tifs = list(level1.glob("*.tif")) + list(level1.glob("*.tiff"))
            if root_tifs:
                logger.warning(
                    "Cycle2 block %s has composite files but no channel subdirs; "
                    "run spit_channel.py first. Skipping.",
                    level1.name,
                )
                print(
                    "⚠️ %s 根目录下有 Composite 文件但无 DAPI/KI67 子目录，"
                    "请先运行 spit_channel.py 拆分通道，已跳过此块。" % level1.name
                )
                return

    results = {}
    ref_channel = str(config.get("STITCH_REFERENCE_CHANNEL", "")).strip()
    ref_registered = None

    if ref_channel and ref_channel in channels:
        ok, result = run_stitch_for_channel(
            level1=level1,
            channel=ref_channel,
            params=params,
            config=config,
            ij=ij,
            logger=logger,
            output_dir=output_dir,
            fused_prefix=fused_prefix,
        )
        if ok and result is not None:
            results[ref_channel] = result
            ref_dir = level1 / ref_channel
            ref_registered = ref_dir / f"TileConfiguration_{fused_prefix}_{ref_channel}.registered.txt"
            if not ref_registered.exists():
                ref_registered = None

    for ch in channels:
        if ch == ref_channel:
            continue

        layout_file = None
        if ref_registered is not None:
            ch_dir = level1 / ch
            if ch_dir.is_dir():
                layout_path = ch_dir / f"TileConfiguration_{fused_prefix}_{ch}.from_{ref_channel}.registered.txt"
                try:
                    _build_layout_file_from_reference(ref_registered, layout_path, ref_channel, ch)
                    layout_file = layout_path.name
                except Exception:
                    layout_file = None

        ok, result = run_stitch_for_channel(
            level1=level1,
            channel=ch,
            params=params,
            config=config,
            ij=ij,
            logger=logger,
            output_dir=output_dir,
            fused_prefix=fused_prefix,
            layout_file=layout_file,
        )
        if ok and result is not None:
            results[ch] = result

    if results:
        check_channel_sizes(results, logger)

    print("✅ %s 处理完成，结果位于: %s" % (level1.name, output_dir))
    logger.info("Level1 done (sequential): %s", level1)


def process_all_level1_dirs(level1_dirs, config, ij, logger):
    total = len(level1_dirs)
    for i, p in enumerate(level1_dirs, 1):
        try:
            print("\n\n" + "#" * 60)
            print("Progress: %s/%s" % (i, total))
            logger.info("Processing %s/%s: %s", i, total, Path(p).name)

            if config.get("INTERACTIVE", False):
                cont = timeout_input(
                    "即将处理 %s。继续? (Y=继续, n=退出, s=跳过，默认Y)" % Path(p).name,
                    "Y",
                    5,
                    True,
                ).strip().lower()
                if cont in ("n", "no"):
                    logger.info("User stopped at %s/%s", i, total)
                    print("User stopped process.")
                    return
                if cont in ("s", "skip"):
                    logger.info("User skipped %s", p)
                    print("Skipping %s" % Path(p).name)
                    continue

            process_level1_sequential(p, config, ij, logger)
        except Exception as e:
            logger.exception("Error processing %s: %s", Path(p).name, e)
            print(f"ERROR processing {Path(p).name}: {e}")
            continue

    print("\n[OK] All level1 directories finished!")
    logger.info("All level1 directories processed")
