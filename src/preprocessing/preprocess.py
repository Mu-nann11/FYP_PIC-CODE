#!/usr/bin/env python3
"""
数据预处理模块 - 整合通道分类和文件重命名功能

功能：
1. 自动检测未分类的原始数据并按通道分类
2. 重命名 Cycle2 中的 Composite 文件
3. 拆分 Cycle2 中的 Composite 文件为 DAPI 和 KI67
4. 检测并报告需要预处理的数据集

使用方式：
  python -m DataManagement.preprocess --dataset TMAd --root /data/Raw_Data
  python -m DataManagement.preprocess --all --root /data/Raw_Data
  python -m DataManagement.preprocess --check --root /data/Raw_Data  # 只检查，不处理

配置项（fiji_config.json）：
  PREPROCESS_AUTO_RUN: 自动运行预处理（默认 True）
  PREPROCESS_DRY_RUN: 只检查不处理（默认 False）
  PREPROCESS_CYCLE1: 处理 Cycle1 数据（默认 True）
  PREPROCESS_CYCLE2: 处理 Cycle2 数据（默认 True）
"""

import os
import shutil
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import tifffile as tiff
except ImportError:
    tiff = None


# Cycle2 归档未拆分的原始 Composite
COMPOSITE_SOURCE_DIR = "Composite_source"


def _ensure_uint16(arr):
    """强制转换为 uint16，保持相对数值范围。"""
    if arr.dtype == np.uint16:
        return arr
    if not np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.uint16)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint16)
    norm = (arr.astype(np.float64) - mn) / (mx - mn)
    return (norm * 65535.0).clip(0, 65535).astype(np.uint16)


@dataclass
class PreprocessResult:
    """预处理结果"""
    dataset: str
    cycle: str
    blocks_processed: int
    files_moved: int
    files_skipped: int
    errors: int


# 通道映射：文件名关键词 → 目标目录名
CHANNEL_MAPPING = {
    "Cycle1": {
        "w1DAPI": "DAPI",
        "w1dapi": "DAPI",
        "w2GFP": "HER2",
        "w2gfp": "HER2",
        "w3Cy3": "PR",
        "w3cy3": "PR",
        "w4Cy5": "ER",
        "w4cy5": "ER",
    },
    "Cycle2": {
        "Composite": "Composite",
    }
}


def get_channel_keyword(filename: str, cycle: str = "Cycle1") -> tuple:
    """从文件名提取通道关键字和通道名称
    
    返回: (通道名称, 通道关键字, 序号)
    """
    filename_upper = str(filename).upper()
    
    for keyword, channel_name in CHANNEL_MAPPING.get(cycle, {}).items():
        if keyword.upper() in filename_upper:
            match = re.search(r'_s(\d+)', filename, re.IGNORECASE)
            seq_num = match.group(1) if match else ""
            return channel_name, keyword, seq_num
    
    return None, None, None


def needs_organize(block_path: Path) -> bool:
    """检查 Block 是否需要通道分类"""
    for item in block_path.iterdir():
        if item.is_dir():
            return False
        if item.is_file() and item.suffix.lower() in ('.tif', '.tiff'):
            channel, _, _ = get_channel_keyword(item.name, "Cycle1")
            if channel:
                return True
    return False


def needs_rename_composite(block_path: Path, dataset: str) -> bool:
    """检查是否需要重命名 Composite 文件"""
    for item in block_path.iterdir():
        if item.is_file() and 'composite' in item.name.lower() and '-' in item.name:
            return True
    return False


def organize_block(block_path: Path, dataset: str, cycle: str = "Cycle1", dry_run: bool = False) -> dict:
    """组织单个 Block 下的文件到通道子目录"""
    block_name = block_path.name
    stats = {"moved": 0, "skipped": 0, "errors": 0}
    
    if not block_path.is_dir():
        return stats
    
    channel_files = {}
    
    for filename in block_path.iterdir():
        if not filename.is_file():
            continue
        
        channel, keyword, seq_num = get_channel_keyword(filename.name, cycle)
        if not channel:
            continue
        
        if channel not in channel_files:
            channel_files[channel] = []
        channel_files[channel].append((filename, keyword, seq_num))
    
    if not channel_files:
        return stats
    
    for channel, files in sorted(channel_files.items()):
        channel_dir = block_path / channel
        if not dry_run:
            channel_dir.mkdir(exist_ok=True)
        
        for file_path, keyword, seq_num in files:
            ext = file_path.suffix
            new_filename = f"{block_name}_{dataset}_{keyword}_s{seq_num}{ext}"
            dest_path = channel_dir / new_filename
            
            if dry_run:
                print(f"  [dry-run] {file_path.name} → {channel}/{new_filename}")
                stats["skipped"] += 1
                continue
            
            if dest_path.exists():
                print(f"  ⏭️ 跳过（已存在）: {channel}/{new_filename}")
                stats["skipped"] += 1
                continue
            
            try:
                shutil.move(str(file_path), str(dest_path))
                print(f"  ✅ {file_path.name} → {channel}/{new_filename}")
                stats["moved"] += 1
            except Exception as e:
                print(f"  ❌ 失败: {file_path.name} - {e}")
                stats["errors"] += 1
    
    return stats


def split_two_channel(img) -> Tuple:
    """拆分双通道图像"""
    if img.ndim != 3:
        raise ValueError("不是多通道图: shape=%s" % (img.shape,))
    if img.shape[0] == 2:
        return img[0], img[1]
    if img.shape[-1] == 2:
        return img[..., 0], img[..., 1]
    raise ValueError("需要恰好 2 通道: shape=%s" % (img.shape,))


def split_composite_block(
    block_path: Path,
    dry_run: bool = False,
    remove_sources: bool = False,
    swap_dapi_ki67: bool = True,
) -> dict:
    """
    拆分 Block 下的 Composite 双通道文件为 DAPI 和 KI67。
    
    返回: {"split": count, "skipped": count, "errors": count}
    """
    if tiff is None:
        print("  ⚠️ tifffile 未安装，无法拆分文件")
        return {"split": 0, "skipped": 0, "errors": 1}

    stats = {"split": 0, "skipped": 0, "errors": 0}
    
    if not block_path.is_dir():
        return stats
    
    dapi_dir = block_path / "DAPI"
    ki67_dir = block_path / "KI67"
    if not dry_run:
        dapi_dir.mkdir(parents=True, exist_ok=True)
        ki67_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集需要拆分的 Composite 文件（排除已拆分的、DAPI、KI67 子目录下的）
    composite_files = []
    for f in sorted(block_path.glob("*.tif")):
        if f.parent != block_path:
            continue
        low = f.name.lower()
        if "_dapi" in low or "_ki67" in low:
            continue  # 已是拆分后的文件
        composite_files.append(f)
    
    for path in sorted(composite_files, key=lambda p: p.name.lower()):
        try:
            img = tiff.imread(str(path))
            ch1, ch2 = split_two_channel(img)
        except Exception as e:
            print("  跳过 %s: %s" % (path.name, e))
            stats["errors"] += 1
            continue
        
        # 默认 swap：ch0→Ki67, ch1→DAPI
        if swap_dapi_ki67:
            dapi_plane, ki67_plane = ch2, ch1
        else:
            dapi_plane, ki67_plane = ch1, ch2
        
        base = path.stem
        dapi_path = dapi_dir / ("%s_DAPI.tif" % base)
        ki67_path = ki67_dir / ("%s_KI67.tif" % base)
        
        if dry_run:
            print("  [dry-run] %s → DAPI/%s , KI67/%s" % (path.name, dapi_path.name, ki67_path.name))
            stats["skipped"] += 1
            continue
        
        try:
            tiff.imwrite(str(dapi_path), _ensure_uint16(dapi_plane))
            tiff.imwrite(str(ki67_path), _ensure_uint16(ki67_plane))
            print("  ✅ %s → DAPI/%s , KI67/%s" % (path.name, dapi_path.name, ki67_path.name))
            stats["split"] += 1
            
            # 处理源文件
            if remove_sources:
                try:
                    path.unlink()
                    print("     (已删除源文件 %s)" % path.name)
                except OSError as e:
                    print("     ⚠️ 未能删除源文件 %s: %s" % (path.name, e))
            else:
                arch = block_path / COMPOSITE_SOURCE_DIR
                arch.mkdir(parents=True, exist_ok=True)
                dest = arch / path.name
                if dest.exists():
                    print("     ⚠️ 归档目标已存在，未移动: %s" % dest)
                else:
                    try:
                        shutil.move(str(path), str(dest))
                        print("     (源文件已移至 %s/)" % COMPOSITE_SOURCE_DIR)
                    except OSError as e:
                        print("     ⚠️ 移至 %s 失败: %s" % (COMPOSITE_SOURCE_DIR, e))
        except Exception as e:
            print("  ❌ 写入失败 %s: %s" % (path.name, e))
            stats["errors"] += 1
    
    return stats


def needs_split_composite(block_path: Path) -> bool:
    """检查是否需要拆分 Composite 文件"""
    for item in block_path.iterdir():
        if item.is_file() and item.suffix.lower() in ('.tif', '.tiff'):
            low = item.name.lower()
            if 'composite' in low and '_dapi' not in low and '_ki67' not in low:
                # 有 Composite 文件且不在 DAPI/KI67 子目录中
                if item.parent == block_path:  # 直接在 block 目录下
                    return True
    return False


def rename_composite_block(block_path: Path, dataset: str, dry_run: bool = False) -> dict:
    """重命名 Block 下的 Composite 文件"""
    block_name = block_path.name
    stats = {"renamed": 0, "skipped": 0, "errors": 0}
    
    if not block_path.is_dir():
        return stats
    
    composite_files = sorted([
        f for f in block_path.iterdir() 
        if f.is_file() and 'composite' in f.name.lower()
    ])
    
    for file_path in composite_files:
        match = re.search(r'composite[\s_-](\d+)', file_path.name, re.IGNORECASE)
        if not match:
            print(f"  ⚠️ 无法从 {file_path.name} 提取序号")
            continue
        
        seq_num = match.group(1)
        ext = file_path.suffix
        new_filename = f"{block_name}_{dataset}_Composite_{seq_num}{ext}"
        dest_path = block_path / new_filename
        
        if dry_run:
            print(f"  [dry-run] {file_path.name} → {new_filename}")
            stats["skipped"] += 1
            continue
        
        if dest_path.exists():
            print(f"  ⏭️ 跳过（已存在）: {new_filename}")
            stats["skipped"] += 1
            continue
        
        try:
            shutil.move(str(file_path), str(dest_path))
            print(f"  ✅ {file_path.name} → {new_filename}")
            stats["renamed"] += 1
        except Exception as e:
            print(f"  ❌ 失败: {file_path.name} - {e}")
            stats["errors"] += 1
    
    return stats


def preprocess_cycle(
    dataset_root: Path,
    dataset: str,
    cycle: str,
    dry_run: bool = False,
    logger=None,
    remove_sources: bool = False,
    swap_dapi_ki67: bool = True,
    flat_blocks_root: Optional[Path] = None,
) -> list:
    """预处理整个 Cycle。

    Args:
        flat_blocks_root: 若不为 None，则直接扫描此目录下的 block 子目录（平铺结构，如 TMAe）。
    """
    if flat_blocks_root is not None:
        blocks_path = flat_blocks_root
    else:
        blocks_path = dataset_root / dataset / cycle

    if not blocks_path.exists():
        if logger:
            logger.warning(f"Blocks path not found: {blocks_path}")
        print(f"WARNING: Path not found: {blocks_path}")
        return []

    print(f"\n{'='*50}")
    print(f"Processing {dataset}/{cycle}/")
    print(f"   Path: {blocks_path}")
    print(f"   Mode: {'dry-run' if dry_run else 'execute'}")
    print(f"{'='*50}\n")

    block_dirs = sorted([d for d in blocks_path.iterdir() if d.is_dir()])

    if not block_dirs:
        print(f"WARNING: No block directories found")
        return []

    # 如果 Cycle1 下有 Calculate_Data/3ch_Neg/ 子结构，也纳入扫描
    # 格式：Cycle1/Calculate_Data/3ch_Neg/{block}/
    calc_extra = []
    if (blocks_path / "Calculate_Data" / "3ch_Neg").exists():
        calc_extra = sorted([
            d for d in (blocks_path / "Calculate_Data" / "3ch_Neg").iterdir()
            if d.is_dir()
        ])
    all_block_dirs = block_dirs + calc_extra
    total_blocks = len(all_block_dirs)

    if not total_blocks:
        print(f"WARNING: No block directories found")
        return []

    results = []

    for i, (block_path, block_name) in enumerate(
        [(p, p.name) for p in all_block_dirs], 1
    ):
        print(f"[{i}/{total_blocks}] Processing Block: {block_name}")

        block_stats = {"block": block_name, "cycle": cycle, "files_processed": 0, "skipped": 0, "errors": 0}

        if flat_blocks_root is not None or cycle == "Cycle1":
            # 平铺结构 或 Cycle1 → 通道分类
            if needs_organize(block_path):
                stats = organize_block(block_path, dataset, cycle, dry_run)
                block_stats["files_processed"] = stats["moved"]
                block_stats["skipped"] = stats["skipped"]
                block_stats["errors"] = stats["errors"]
            else:
                print(f"  Already organized, skipping")
                block_stats["skipped"] = -1

        elif cycle == "Cycle2":
            # Cycle2: Calculate_Data 下可能是 3ch_Neg（H2）或 Ki67_Neg（A8/D1/G1/H10）
            calc_3ch = blocks_path / "Calculate_Data" / "3ch_Neg" / block_name
            calc_ki67 = blocks_path / "Calculate_Data" / "Ki67_Neg" / block_name

            # 优先用实际子目录路径（如果存在）
            actual_path = block_path
            if calc_3ch.exists():
                actual_path = calc_3ch
            elif calc_ki67.exists():
                actual_path = calc_ki67

            # 1. 检查是否需要重命名 Composite
            if needs_rename_composite(actual_path, dataset):
                stats = rename_composite_block(actual_path, dataset, dry_run)
                block_stats["files_processed"] = stats["renamed"]
                block_stats["skipped"] = stats["skipped"]
                block_stats["errors"] = stats["errors"]
            else:
                print(f"  No rename needed, skipping")

            # 2. 检查是否需要拆分 Composite 为 DAPI/KI67
            if needs_split_composite(actual_path):
                print(f"  Composite files detected, will split")
                split_stats = split_composite_block(
                    actual_path,
                    dry_run=dry_run,
                    remove_sources=remove_sources,
                    swap_dapi_ki67=swap_dapi_ki67,
                )
                if not dry_run:
                    block_stats["files_processed"] += split_stats["split"]
                    block_stats["errors"] += split_stats["errors"]
            else:
                print(f"  No split needed or already split, skipping")

        results.append(block_stats)
        print()

    total_processed = sum(r["files_processed"] for r in results if r["files_processed"] > 0)
    total_skipped = sum(1 for r in results if r["skipped"] == -1)

    print(f"Done {dataset}/{cycle}/ - processed {total_processed} files, {total_skipped} blocks skipped")

    return results


def detect_dataset_structure(dataset_dir: Path) -> str:
    """
    检测数据集目录结构类型。
    返回:
      'hierarchical' - 有 Cycle 子目录（TMAd 模式：Dataset/Cycle/Block/）
      'flat'         - 无 Cycle 子目录（TMAe 模式：Dataset/Block/）
    """
    subdirs = [d.name.lower() for d in dataset_dir.iterdir() if d.is_dir()]
    has_cycle = any(d.startswith("cycle") for d in subdirs)
    return "hierarchical" if has_cycle else "flat"


def _iter_all_block_dirs(cycle_path: Path, cycle: str):
    """
    遍历 Cycle 目录下所有 block 子目录，包括：
      - Cycle/{block}/ （直接结构）
      - Cycle/Calculate_Data/3ch_Neg/{block}/ （3ch_Neg 结构）
      - Cycle/Calculate_Data/Ki67_Neg/{block}/ （Ki67_Neg 结构）
    """
    if not cycle_path.exists():
        return

    for item in sorted(cycle_path.iterdir()):
        if not item.is_dir():
            continue
        # 直接 block
        if item.name not in ("logs", "tmp", "Composite_source"):
            yield item, item.name

        # Calculate_Data 子结构
        calc_3ch = item / "3ch_Neg"
        calc_ki67 = item / "Ki67_Neg"
        for calc_parent in (calc_3ch, calc_ki67):
            if calc_parent.exists():
                for sub in sorted(calc_parent.iterdir()):
                    if sub.is_dir() and sub.name not in ("logs", "tmp"):
                        yield sub, sub.name


def check_data_status(dataset_root: Path) -> dict:
    """检查所有数据集的预处理状态（包括 Calculate_Data 结构）"""
    status = {}

    if not dataset_root.exists():
        return status

    for dataset_dir in sorted(dataset_root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        struct = detect_dataset_structure(dataset_dir)

        if struct == "hierarchical":
            status[dataset] = {"Cycle1": [], "Cycle2": []}
            for cycle in ["Cycle1", "Cycle2"]:
                cycle_path = dataset_dir / cycle
                if not cycle_path.exists():
                    continue

                for block_dir, block_name in _iter_all_block_dirs(cycle_path, cycle):
                    needs_preprocess = False
                    reason = ""

                    # 检查 Cycle1
                    if cycle == "Cycle1" and needs_organize(block_dir):
                        needs_preprocess = True
                        reason = "需要通道分类"

                    # 检查 Cycle2
                    if cycle == "Cycle2":
                        reasons = []
                        if needs_rename_composite(block_dir, dataset):
                            reasons.append("重命名 Composite")
                        if needs_split_composite(block_dir):
                            reasons.append("拆分 DAPI/KI67")
                        if reasons:
                            needs_preprocess = True
                            reason = ", ".join(reasons)

                    if needs_preprocess:
                        status[dataset][cycle].append({
                            "block": block_name,
                            "reason": reason
                        })

        else:  # flat 结构（TMAe 模式）
            status[dataset] = {"Flat": []}
            for block_dir in sorted(dataset_dir.iterdir()):
                if not block_dir.is_dir():
                    continue
                if needs_organize(block_dir):
                    status[dataset]["Flat"].append({
                        "block": block_dir.name,
                        "reason": "需要通道分类"
                    })

    return status


def print_status_report(status: dict):
    """打印状态报告"""
    print("\n" + "=" * 60)
    print("📊 数据预处理状态报告")
    print("=" * 60)
    
    has_issues = False
    
    for dataset, cycles in sorted(status.items()):
        dataset_has_issues = False
        
        for cycle, blocks in sorted(cycles.items()):
            if blocks:
                dataset_has_issues = True
                has_issues = True
        
        if dataset_has_issues:
            print(f"\n🔴 {dataset}")
            for cycle, blocks in sorted(cycles.items()):
                if blocks:
                    print(f"   📁 {cycle}:")
                    for block in blocks:
                        print(f"      - {block['block']}: {block['reason']}")
    
    if not has_issues:
        print("\n✅ 所有数据都已完成预处理，无需处理。")
    
    print("\n" + "=" * 60)
    
    return has_issues


def run_preprocess(
    dataset_root: Path,
    dataset: Optional[str] = None,
    dry_run: bool = False,
    logger=None,
    remove_sources: bool = False,
    swap_dapi_ki67: bool = True,
) -> list:
    """运行预处理流程"""
    all_results = []

    if dataset:
        datasets = [dataset]
    else:
        datasets = [d.name for d in dataset_root.iterdir() if d.is_dir()]

    for ds in sorted(datasets):
        dataset_dir = dataset_root / ds
        struct = detect_dataset_structure(dataset_dir)

        if struct == "flat":
            # 平铺结构（如 TMAe）：block 目录直接在 dataset 下
            results = preprocess_cycle(
                dataset_root, ds, "Cycle1", dry_run, logger,
                remove_sources=remove_sources,
                swap_dapi_ki67=swap_dapi_ki67,
                flat_blocks_root=dataset_dir,
            )
            all_results.extend(results)
        else:
            # 层级结构（如 TMAd）：有 Cycle1 / Cycle2 子目录
            for cycle in ["Cycle1", "Cycle2"]:
                results = preprocess_cycle(
                    dataset_root, ds, cycle, dry_run, logger,
                    remove_sources=remove_sources,
                    swap_dapi_ki67=swap_dapi_ki67,
                )
                all_results.extend(results)

    return all_results

def run_preprocess_block(
    block_path_cycle1: Optional[Path],
    block_path_cycle2: Optional[Path],
    dataset: str,
    dry_run: bool = False,
    logger=None,
    remove_sources: bool = False,
    swap_dapi_ki67: bool = True,
) -> dict:
    """
    为流水线提供的单 Block 处理入口。
    
    Args:
        block_path_cycle1: Cycle1 的 block 路径 (e.g. Raw_Data/TMAd/Cycle1/A1)
        block_path_cycle2: Cycle2 的 block 路径 (e.g. Raw_Data/TMAd/Cycle2/A1)
        dataset: 数据集名称 (e.g. TMAd)
        dry_run: 是否仅打印不执行
        logger: 日志器
        remove_sources: 分离后是否删除原复合图像
        swap_dapi_ki67: 是否交换通道
        
    Returns:
        包含了各步处理日志的字典
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    results = {}

    if block_path_cycle1 and block_path_cycle1.exists():
        if needs_organize(block_path_cycle1):
            logger.info(f"开始分类 Cycle1 块: {block_path_cycle1.name}")
            stats = organize_block(block_path_cycle1, dataset, cycle="Cycle1", dry_run=dry_run)
            results["cycle1_organize"] = stats
        else:
            logger.info(f"Cycle1 块 {block_path_cycle1.name} 似乎已经分类, 跳过.")

    if block_path_cycle2 and block_path_cycle2.exists():
        block_name = block_path_cycle2.name
        
        # 1. 重命名复合图像
        if needs_rename_composite(block_path_cycle2, dataset):
             logger.info(f"重命名 Cycle2 块的复合图像: {block_name}")
             results["cycle2_rename"] = rename_composite_block(block_path_cycle2, dataset, dry_run=dry_run)
             
        # 2. 从未分类复合图像、或者 Composite 分离出单通道
        if needs_split_composite(block_path_cycle2):
             logger.info(f"分离 Cycle2 块的复合图像: {block_name}")
             results["cycle2_split"] = split_composite_block(
                 block_path_cycle2, dry_run=dry_run, 
                 remove_sources=remove_sources, swap_dapi_ki67=swap_dapi_ki67
             )
        else:
             logger.info(f"Cycle2 块 {block_name} 无需分离, 跳过.")
             
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据预处理：通道分类、重命名、拆分 Composite")
    parser.add_argument("--root", type=str, default="/data/Raw_Data", help="数据根目录")
    parser.add_argument("--dataset", type=str, default=None, help="指定数据集")
    parser.add_argument("--check", action="store_true", help="只检查状态，不处理")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    parser.add_argument("--remove-sources", action="store_true", help="拆分后删除源 Composite 文件")
    parser.add_argument("--no-swap-dapi-ki67", action="store_true", help="不交换通道顺序")
    
    args = parser.parse_args()
    
    root = Path(args.root).resolve()
    
    if args.check:
        status = check_data_status(root)
        print_status_report(status)
    else:
        run_preprocess(
            root,
            dataset=args.dataset,
            dry_run=args.dry_run,
            logger=None,
            remove_sources=args.remove_sources,
            swap_dapi_ki67=not args.no_swap_dapi_ki67,
        )
