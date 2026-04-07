"""
校准流水线快捷入口脚本。

用法:
    # 默认：使用 config.py 中的配置
    python run_calibration.py

    # 强制重新生成（忽略已有结果）
    python run_calibration.py --force

    # 自定义阴性阈值参数
    python run_calibration.py --n-sd 2.5

    # 指定 HER2 阳性 Block 用于 Otsu 分级
    python run_calibration.py --her2-blocks G2 A2 B8

    # 完整示例
    python run_calibration.py --force --n-sd 2.0 --her2-blocks G2 A2 B8 --verbose

数据依赖:
    运行前请确保已完成以下步骤：
    1. Step 1 (Fiji Stitching) - 瓦片拼接
    2. Step 2 (Alignment)       - Cycle1/Cycle2 配准
    3. Step 3 (Segmentation)    - 细胞分割与特征提取
    → 输出: results/segmentation/{dataset}/{block}/{block}_{dataset}_features.csv

输出文件:
    results/calibration/
    ├── thresholds.json          # 各通道阈值和分级点（JSON 格式）
    ├── calibration_report.txt   # 人类可读的文本报告
    └── plots/
        ├── ER_histogram_3ch_Neg.png
        ├── PR_histogram_3ch_Neg.png
        ├── HER2_histogram_3ch_Neg.png
        ├── KI67_histogram_Ki67_Neg.png
        └── HER2_grades_histogram.png   # HER2 Otsu 分级可视化
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ----- 模块路径兼容 -----
# 支持多种运行方式：
#   1. python run_calibration.py                    （直接运行，calibration 目录内）
#   2. python -m calibration.run_calibration        （模块运行，Code 目录内）
#   3. python -m calibration.run_calibration        （模块运行，项目根目录内）
_script_dir = Path(__file__).resolve().parent
_calib_root = _script_dir.parent  # Code/ 目录

if str(_calib_root) not in sys.path:
    sys.path.insert(0, str(_calib_root))

from calibration.config import (
    BASE_DIR,
    CALIBRATION_DIR,
    NEGATIVE_BLOCKS,
    HER2_POSITIVE_BLOCKS,
    CALIBRATION_PARAMS,
    get_pos_feature_csv,
)
# 延迟导入：numpy/matplotlib/skimage 仅在执行校准时才加载（不由 --list-negative 触发）


# =============================================================================
# 命令行参数
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="阴性对照阈值校准快捷入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python run_calibration.py\n"
            "  python run_calibration.py --force\n"
            "  python run_calibration.py --n-sd 2.5\n"
            "  python run_calibration.py --her2-blocks G2 A2 B8\n"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新生成（忽略已有结果）",
    )

    parser.add_argument(
        "--n-sd",
        type=float,
        default=None,
        help=(
            "阴性阈值标准差倍数（default: 从 config.py 读取，"
            f"当前为 {CALIBRATION_PARAMS.get('neg_threshold_n_sd', 2.0)}）"
        ),
    )

    parser.add_argument(
        "--her2-blocks",
        nargs="+",
        default=None,
        metavar="BLOCK",
        help=(
            "HER2 阳性 Block 列表（用于 Otsu 1+/2+/3+ 分级）。"
            "例如：--her2-blocks G2 A2 B8。"
            "留空则使用 config.py 中的 HER2_POSITIVE_BLOCKS['blocks']"
        ),
    )

    parser.add_argument(
        "--list-negative",
        action="store_true",
        help="列出配置的阴性对照 Block 并退出",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"校准结果输出目录（default: {CALIBRATION_DIR}）",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安静模式（仅输出关键信息）",
    )

    return parser


# =============================================================================
# 预检：数据依赖检查
# =============================================================================

def check_prerequisites(neg_blocks: dict, her2_blocks: list = None, quiet: bool = False):
    """
    检查校准所需的数据依赖是否满足。

    Returns:
        bool: 所有依赖是否满足
    """
    from calibration.analyze_negative_controls import (
        discover_segmentation_files,
        SEGMENTATION_DIR,
    )

    ok = True

    if not quiet:
        print("\n[预检] 检查数据依赖 ...")

    for group_key, group_cfg in neg_blocks.items():
        seg_root = group_cfg["segmentation_root"]
        if seg_root is None:
            config_module = __import__("config", globals(), locals(), ["SEGMENTATION_DIR"])
            seg_root = config_module.SEGMENTATION_DIR # Defaults to SEGMENTATION_DIR for all blocks
        csv_files = discover_segmentation_files(seg_root, group_cfg.get("blocks", []))
        if csv_files:
            status = "✓"
            msg = f"  {status} {group_key}: 发现 {len(csv_files)} 个 Block"
            if not quiet:
                print(msg)
        else:
            status = "✗"
            msg = (
                f"  {status} {group_key}: 未找到分割结果！\n"
                f"    请先运行 Step 3 Segmentation，期望路径：\n"
                f"    {seg_root}/<dataset>/<block>/<block>_<dataset>_features.csv"
            )
            print(msg)
            ok = False

        companion_roots = group_cfg.get("required_companion_raw_roots", [])
        scoped_blocks = group_cfg.get("blocks", []) or []
        for companion_root in companion_roots:
            missing_blocks = []
            for block in scoped_blocks:
                if not (Path(companion_root) / block).exists():
                    missing_blocks.append(block)

            if missing_blocks:
                ok = False
                print(
                    f"  ✗ {group_key}: 伴随原始路径缺失 {len(missing_blocks)} 个 Block\n"
                    f"    路径: {companion_root}\n"
                    f"    缺失: {missing_blocks}"
                )
            elif companion_roots:
                if not quiet:
                    print(
                        f"  ✓ {group_key}: 伴随原始路径完整 ({companion_root})"
                    )

    if her2_blocks:
        for block in her2_blocks:
            csv_path = get_pos_feature_csv(block)
            if csv_path is not None and csv_path.exists():
                status = "✓"
                if not quiet:
                    print(f"  {status} HER2_Otsu[{block}]: 存在")
            else:
                status = "✗"
                print(
                    f"  {status} HER2_Otsu[{block}]: 缺失\n"
                    f"    期望路径：{csv_path}"
                )
                ok = False

    if not quiet:
        print()

    return ok


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ----- 仅列出配置 -----
    if args.list_negative:
        print("=" * 60)
        print("阴性对照 Block 配置")
        print("=" * 60)
        for group_key, group_cfg in NEGATIVE_BLOCKS.items():
            print(f"\n[{group_key}]")
            print(f"  描述: {group_cfg.get('description', 'N/A')}")
            print(f"  分割路径: {group_cfg['segmentation_root']}")
            print(f"  指定 blocks: {group_cfg.get('blocks') or '(自动发现)'}")
            companion_roots = group_cfg.get("required_companion_raw_roots", [])
            if companion_roots:
                print(f"  伴随原始路径: {companion_roots}")
            print(f"  通道配置:")
            for ch, ch_cfg in group_cfg["channels"].items():
                print(f"    - {ch}: {ch_cfg['column']} ({ch_cfg['intensity_type']})")
        print("\n" + "=" * 60)
        print(f"HER2 阳性分级 Blocks: {HER2_POSITIVE_BLOCKS['blocks'] or '(未配置)'}")
        print("=" * 60)
        return

    # ----- 确定参数 -----
    n_sd = (
        args.n_sd
        if args.n_sd is not None
        else CALIBRATION_PARAMS.get("neg_threshold_n_sd", 2.0)
    )

    her2_cfg = dict(HER2_POSITIVE_BLOCKS)
    her2_blocks_arg = args.her2_blocks if args.her2_blocks is not None else her2_cfg.get("blocks", [])

    # ----- 预检 -----
    check_ok = check_prerequisites(
        neg_blocks=NEGATIVE_BLOCKS,
        her2_blocks=her2_blocks_arg,
        quiet=args.quiet,
    )

    if not check_ok:
        print("\n[错误] 数据依赖不满足，请先运行 Step 1-3 流水线生成分割结果。")
        print("       或者使用 --force 跳过预检直接运行。")
        if not args.force:
            sys.exit(1)

    # ----- 打印启动信息 -----
    print("=" * 60)
    print("阴性对照阈值校准")
    print("=" * 60)
    print(f"阴性阈值参数  : mean + {n_sd} * SD")
    print(f"HER2 Otsu Block: {her2_blocks_arg or '(无)'}")
    print(f"输出目录      : {CALIBRATION_DIR}")
    print(f"强制重算      : {args.force}")
    print("=" * 60)

    # 延迟导入 numpy/skimage（仅在实际执行校准时加载）
    from calibration.analyze_negative_controls import run_calibration as _run_calibration

    # ----- 执行校准 -----
    start_time = datetime.now()

    results = _run_calibration(
        neg_blocks=NEGATIVE_BLOCKS,
        her2_pos_blocks=her2_cfg if her2_blocks_arg else None,
        n_sd=n_sd,
        force=args.force,
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    # ----- 汇总输出 -----
    print("\n" + "=" * 60)
    print("校准完成")
    print("=" * 60)

    if results and results.get("channels"):
        print("各通道阴性上限阈值:")
        for ch, ch_results in results["channels"].items():
            for group_key, stats in ch_results.items():
                if isinstance(stats, dict) and "threshold" in stats:
                    print(f"  {ch:8s} [{group_key:10s}]: {stats['threshold']:.4f}")
    else:
        print("  (无结果，请检查数据依赖)")

    print(f"\n耗时: {elapsed:.1f}s")
    print(f"输出路径:")
    print(f"  thresholds.json  : {CALIBRATION_DIR / 'thresholds.json'}")
    print(f"  文本报告         : {CALIBRATION_DIR / 'calibration_report.txt'}")
    print(f"  直方图           : {CALIBRATION_DIR / 'plots'}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
