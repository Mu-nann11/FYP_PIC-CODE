"""
Negative Control Cell Intensity Histograms
- Example: ER, showing intensity distribution and threshold line (μ+2σ)
- Reference: Paper Figure 3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 配置
BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")
RESULTS_DIR = BASE_DIR / "results"
CALIBRATION_DIR = RESULTS_DIR / "calibration"
OUTPUT_DIR = CALIBRATION_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 阴性Block
NEGATIVE_BLOCKS_3CH = ["A1", "D1", "E10", "H2", "H10", "J10"]  # ER/PR/HER2
NEGATIVE_BLOCKS_KI67 = ["A8", "D1", "G1", "H10"]  # Ki67


def load_threshold_data():
    """加载阈值数据"""
    threshold_file = CALIBRATION_DIR / "thresholds_raw_nuclei.json"
    with open(threshold_file, 'r') as f:
        return json.load(f)


def load_negative_nuclei_data(blocks, channels):
    """加载阴性对照块的核强度数据"""
    all_data = []
    for block in blocks:
        csv_path = CALIBRATION_DIR / f"{block}_raw_nuclei_intensity.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # 只保留需要的列
            cols_needed = ['block', 'nuclei_id'] + [f'{ch}_mean' for ch in channels]
            available_cols = [c for c in cols_needed if c in df.columns]
            all_data.append(df[available_cols])
    return pd.concat(all_data, ignore_index=True) if all_data else None


def plot_channel_histogram(ax, intensities, mean, std, threshold, channel_name, color='steelblue'):
    """绘制单个通道的直方图"""
    # 过滤掉0值
    intensities = intensities[intensities > 0].copy()

    # 确定x轴范围：确保包含阈值区域
    p99 = np.percentile(intensities, 99.5)
    x_max = max(p99, threshold * 1.3)  # 至少显示到阈值的1.3倍
    x_max = min(x_max, intensities.max() * 1.05)  # 但不超过实际最大值

    # 绘制直方图
    n, bins, patches = ax.hist(intensities, bins=80, color=color, alpha=0.7,
                                edgecolor='white', linewidth=0.5, density=True)

    # 添加均值线
    ax.axvline(mean, color='#E74C3C', linestyle='-', linewidth=2.5,
               label=f'Mean (μ) = {mean:.1f}')

    # 添加阈值线 (μ + 2σ)
    ax.axvline(threshold, color='#27AE60', linestyle='--', linewidth=2.5,
               label=f'Threshold (μ+2σ) = {threshold:.1f}')

    # 填充阈值右侧区域（阳性区域）
    ax.axvspan(threshold, x_max, alpha=0.18, color='#27AE60')

    # 设置标签和标题
    ax.set_xlabel('Nuclear Intensity (a.u.)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(f'{channel_name}', fontsize=14, fontweight='bold', pad=10)

    # 添加图例
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # 计算阳性比例
    positive_count = np.sum(intensities >= threshold)
    positive_pct = positive_count / len(intensities) * 100

    # 添加统计信息文本框
    textstr = f'n = {len(intensities):,}\nσ = {std:.1f}\nMedian = {np.median(intensities):.1f}\nPositive% = {positive_pct:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, x_max)


def main():
    print("=" * 70)
    print("Negative Control Cell Intensity Histogram Generator")
    print("=" * 70)

    # Load threshold data
    print("\n[1] Loading threshold data...")
    thresholds = load_threshold_data()
    print(f"    3ch_Neg: ER, PR, HER2")
    print(f"    Ki67_Neg: Ki67")

    # Load negative control nuclei data
    print("\n[2] Loading negative control nuclei intensity data...")

    # ER/PR/HER2 通道
    df_3ch = load_negative_nuclei_data(NEGATIVE_BLOCKS_3CH, ['ER', 'PR', 'HER2'])
    if df_3ch is not None:
        print(f"    3ch_Neg: {len(df_3ch)} 个核")

    # Ki67 通道
    df_ki67 = load_negative_nuclei_data(NEGATIVE_BLOCKS_KI67, ['Ki67'])
    if df_ki67 is not None:
        print(f"    Ki67_Neg: {len(df_ki67)} 个核")

    # Create comprehensive figure
    print("\n[3] Generating histograms...")

    # Figure 1: Main channels histogram (ER, PR, HER2, Ki67)
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    fig1.suptitle('Negative Control Cells - Nuclear Intensity Distribution\n'
                  'Threshold = Mean + 2×Standard Deviation',
                  fontsize=16, fontweight='bold', y=0.98)

    # ER
    if df_3ch is not None:
        er_intensities = df_3ch['ER_mean'].dropna()
        er_stats = thresholds['3ch_Neg']['ER']
        plot_channel_histogram(axes1[0, 0], er_intensities,
                               er_stats['mean'], er_stats['std'], er_stats['threshold'],
                               'ER (Estrogen Receptor)', color='#3498DB')

    # PR
    if df_3ch is not None:
        pr_intensities = df_3ch['PR_mean'].dropna()
        pr_stats = thresholds['3ch_Neg']['PR']
        plot_channel_histogram(axes1[0, 1], pr_intensities,
                               pr_stats['mean'], pr_stats['std'], pr_stats['threshold'],
                               'PR (Progesterone Receptor)', color='#9B59B6')

    # HER2
    if df_3ch is not None:
        her2_intensities = df_3ch['HER2_mean'].dropna()
        her2_stats = thresholds['3ch_Neg']['HER2']
        plot_channel_histogram(axes1[1, 0], her2_intensities,
                               her2_stats['mean'], her2_stats['std'], her2_stats['threshold'],
                               'HER2', color='#E67E22')

    # Ki67
    if df_ki67 is not None:
        ki67_intensities = df_ki67['Ki67_mean'].dropna()
        ki67_stats = thresholds['Ki67_Neg']['Ki67']
        plot_channel_histogram(axes1[1, 1], ki67_intensities,
                               ki67_stats['mean'], ki67_stats['std'], ki67_stats['threshold'],
                               'Ki67 (Proliferation Marker)', color='#27AE60')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save Figure 1
    output_path1 = OUTPUT_DIR / "negative_control_intensity_histograms.png"
    plt.savefig(output_path1, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"    [OK] Saved: {output_path1}")
    plt.close()

    # ========== Figure 2: ER Detail Figure (Reference Paper Figure 3) ==========
    print("\n[4] Generating ER detail histogram...")
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    if df_3ch is not None:
        er_intensities = df_3ch['ER_mean'].dropna()
        er_intensities = er_intensities[er_intensities > 0]
        er_intensities = er_intensities[er_intensities < np.percentile(er_intensities, 99.5)]

        er_stats = thresholds['3ch_Neg']['ER']
        mean = er_stats['mean']
        std = er_stats['std']
        threshold = er_stats['threshold']

        # 直方图
        n, bins, patches = ax2.hist(er_intensities, bins=100, color='#3498DB', alpha=0.75,
                                     edgecolor='white', linewidth=0.5, density=True)

        # 标注样式
        ax2.axvline(mean, color='#C0392B', linestyle='-', linewidth=3,
                   label=f'Mean (μ) = {mean:.1f}')
        ax2.axvline(threshold, color='#27AE60', linestyle='--', linewidth=3,
                   label=f'Threshold (μ + 2σ) = {threshold:.1f}')

        # 填充
        ax2.axvspan(threshold, er_intensities.max(), alpha=0.12, color='#27AE60',
                   label='Positive Region')

        # 标注 μ+σ, μ+2σ, μ+3σ
        for i, (label, val, color) in enumerate([
            ('', mean + std, '#F39C12'),
            ('', threshold, '#27AE60'),
            ('', mean + 3*std, '#8E44AD')
        ], 0):
            if val < er_intensities.max():
                ax2.axvline(val, color=color, linestyle=':', linewidth=1.5, alpha=0.7)

        ax2.set_xlabel('Nuclear Intensity (a.u.)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax2.set_title('ER Intensity Distribution in Negative Control Cells\n'
                     f'(n = {len(er_intensities):,} nuclei from {len(NEGATIVE_BLOCKS_3CH)} blocks)',
                     fontsize=14, fontweight='bold')

        # 图例
        ax2.legend(loc='upper right', fontsize=11, framealpha=0.95)

        # 统计信息
        stats_text = (f'Statistics:\n'
                      f'  Mean (μ) = {mean:.2f}\n'
                      f'  Std (σ) = {std:.2f}\n'
                      f'  Threshold = {threshold:.2f}\n'
                      f'  Median = {np.median(er_intensities):.2f}\n'
                      f'  Min = {er_intensities.min():.2f}\n'
                      f'  Max = {er_intensities.max():.2f}')
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
        ax2.text(0.02, 0.97, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')

        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(0, np.percentile(er_intensities, 99.5))

    plt.tight_layout()

    # Save Figure 2
    output_path2 = OUTPUT_DIR / "negative_control_ER_histogram.png"
    plt.savefig(output_path2, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"    [OK] Saved: {output_path2}")
    plt.close()

    # ========== Output Threshold Summary Table ==========
    print("\n[5] Generating threshold summary table...")
    summary = []
    for neg_type, channels in thresholds.items():
        for channel, stats in channels.items():
            summary.append({
                'Group': neg_type,
                'Channel': channel,
                'Mean (μ)': f"{stats['mean']:.2f}",
                'Std (σ)': f"{stats['std']:.2f}",
                'Threshold (μ+2σ)': f"{stats['threshold']:.2f}",
                'Median': f"{stats['median']:.2f}",
                'n_Nuclei': stats['n_nuclei']
            })

    df_summary = pd.DataFrame(summary)
    summary_path = OUTPUT_DIR / "negative_control_thresholds_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"    [OK] Saved: {summary_path}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  1. {output_path1}")
    print(f"  2. {output_path2}")
    print(f"  3. {summary_path}")
    print("\nThreshold Summary:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
