"""
生成校准直方图对比图 - 叠加显示所有通道

用法：
    python generate_calibration_comparison.py

输出：
    results/calibration/calibration_comparison.png
    results/calibration/calibration_comparison.pdf
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ==================== 配置 ====================
BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")
SEGMENTATION_DIR = BASE_DIR / "results" / "segmentation"
CALIBRATION_DIR = BASE_DIR / "results" / "calibration"

# 负对照 Block 定义
NEG_3CH_BLOCKS = ["A1", "D1", "E10", "H2", "H10", "J10"]
NEG_KI67_BLOCKS = ["A8", "D1", "G1", "H10"]

# 特征列名映射（根据 config.py）
FEATURE_COLUMNS = {
    "ER": ("3ch_Neg", "ER_nuc_mean"),
    "PR": ("3ch_Neg", "PR_nuc_mean"),
    "HER2": ("3ch_Neg", "HER2_cyto_only_mean"),
    "KI67": ("Ki67_Neg", "Ki67_nuc_mean"),
}

# 颜色方案（报告级别的专业色）
COLORS = {
    "ER": "#E74C3C",      # 红色
    "PR": "#3498DB",      # 蓝色
    "HER2": "#2ECC71",    # 绿色
    "KI67": "#F39C12",    # 橙色
}

# 阈值数据（从 thresholds.json 读取）
THRESHOLDS_FILE = CALIBRATION_DIR / "thresholds.json"

# ==================== 函数 ====================

def load_thresholds() -> Dict:
    """加载校准阈值 JSON"""
    with open(THRESHOLDS_FILE, "r") as f:
        return json.load(f)


def load_channel_data(channel: str, neg_blocks: list, column: str) -> np.ndarray:
    """
    加载指定通道的所有细胞强度数据
    
    Args:
        channel: 通道名 (ER/PR/HER2/Ki67)
        neg_blocks: 负对照 Block 列表
        column: CSV 中的列名
    
    Returns:
        all_values: 所有负对照 Block 的细胞强度值
    """
    all_values = []
    
    for block in neg_blocks:
        csv_file = SEGMENTATION_DIR / block / f"{block}_features.csv"
        if not csv_file.exists():
            print(f"  ⚠ 文件不存在: {csv_file}")
            continue
        
        df = pd.read_csv(csv_file)
        if column not in df.columns:
            print(f"  ⚠ 列不存在: {column} in {block}")
            continue
        
        # 过滤无效数据（NaN, 0 等）
        values = df[column].dropna()
        values = values[values > 0]  # 只保留正值
        all_values.extend(values.tolist())
        print(f"  ✓ {block}: {len(values)} 个细胞")
    
    return np.array(all_values)


def plot_overlaid_histograms(thresholds: Dict) -> Tuple[plt.Figure, Dict]:
    """
    生成叠加直方图
    
    Args:
        thresholds: 从 thresholds.json 读取的阈值数据
    
    Returns:
        fig, axes: matplotlib figure 和统计数据字典
    """
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    
    stats = {}
    
    # 为每个通道绘制直方图
    for channel, (neg_group, column) in FEATURE_COLUMNS.items():
        
        # 确定负对照 Block 列表
        if neg_group == "3ch_Neg":
            neg_blocks = NEG_3CH_BLOCKS
        else:  # Ki67_Neg
            neg_blocks = NEG_KI67_BLOCKS
        
        print(f"\n{'='*60}")
        print(f"处理通道: {channel}")
        print(f"负对照 Block: {neg_blocks}")
        print(f"特征列: {column}")
        print(f"{'='*60}")
        
        # 加载数据
        values = load_channel_data(channel, neg_blocks, column)
        if len(values) == 0:
            print(f"⚠ {channel} 没有有效数据，跳过")
            continue
        
        print(f"总计: {len(values)} 个细胞")
        
        # 从 thresholds.json 获取统计信息
        threshold_data = thresholds["channels"][channel][neg_group]
        mean = threshold_data["mean"]
        std = threshold_data["std"]
        threshold = threshold_data["threshold"]
        n_cells = threshold_data["n_cells"]
        
        stats[channel] = {
            "values": values,
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "n_cells": n_cells,
            "color": COLORS[channel],
        }
        
        # 绘制直方图
        bins = np.linspace(np.min(values) * 0.9, np.max(values) * 1.1, 50)
        ax.hist(values, bins=bins, alpha=0.4, label=f"{channel} (n={n_cells})",
                color=COLORS[channel], edgecolor="black", linewidth=0.5)
        
        # 绘制阈值线（粗红线）
        ax.axvline(threshold, color=COLORS[channel], linestyle="--", 
                   linewidth=2.5, label=f"{channel} threshold: {threshold:.1f}")
        
        # 绘制平均值 ± 标准差区域
        ax.axvline(mean, color=COLORS[channel], linestyle=":", 
                   linewidth=2, alpha=0.7)
        ax.axvline(mean - std, color=COLORS[channel], linestyle=":", 
                   linewidth=1, alpha=0.5)
        ax.axvline(mean + std, color=COLORS[channel], linestyle=":", 
                   linewidth=1, alpha=0.5)
        
        # 在顶部添加样本数标注
        y_pos = ax.get_ylim()[1] * 0.95
        ax.text(threshold, y_pos, f"n={n_cells}", 
               color=COLORS[channel], fontsize=10, fontweight="bold",
               ha="center", bbox=dict(boxstyle="round,pad=0.3", 
               facecolor="white", alpha=0.7, edgecolor=COLORS[channel]))
    
    # 美化图表
    ax.set_xlabel("Cell Intensity", fontsize=14, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")
    ax.set_title("Calibration Thresholds - Overlaid Histograms\n" +
                 "Negative Controls (3ch_Neg & Ki67_Neg)", 
                 fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#F8F9FA")
    
    plt.tight_layout()
    
    return fig, stats


def create_summary_stats_panel(stats: Dict) -> str:
    """生成统计信息面板"""
    panel = "\n统计信息面板\n"
    panel += "=" * 70 + "\n"
    
    for channel, data in stats.items():
        panel += f"\n{channel}:\n"
        panel += f"  样本数: {data['n_cells']}\n"
        panel += f"  平均值: {data['mean']:.2f}\n"
        panel += f"  标准差: {data['std']:.2f}\n"
        panel += f"  阈值 (mean+2σ): {data['threshold']:.2f}\n"
        panel += f"  数据范围: [{data['values'].min():.2f}, {data['values'].max():.2f}]\n"
    
    return panel


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("生成校准直方图对比图")
    print("="*70)
    
    # 加载阈值数据
    print("\n➤ 加载阈值数据...")
    thresholds = load_thresholds()
    print(f"✓ 已加载，生成时间: {thresholds['generated_at']}")
    
    # 生成图表
    print("\n➤ 生成叠加直方图...")
    fig, stats = plot_overlaid_histograms(thresholds)
    
    # 保存文件
    output_png = CALIBRATION_DIR / "calibration_comparison.png"
    output_pdf = CALIBRATION_DIR / "calibration_comparison.pdf"
    
    print(f"\n➤ 保存文件...")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"✓ PNG 已保存: {output_png}")
    
    fig.savefig(output_pdf, bbox_inches="tight")
    print(f"✓ PDF 已保存: {output_pdf}")
    
    # 打印统计信息
    print(create_summary_stats_panel(stats))
    
    print("\n" + "="*70)
    print("✓ 完成！")
    print("="*70)
