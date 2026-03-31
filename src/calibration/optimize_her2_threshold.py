#!/usr/bin/env python3
"""
优化 HER2 阈值

背景：
  - A8 (临床3+): HER2强度 = 1841.69 → 被误分为Grade 0
  - H10 (临床3+): HER2强度 = 1919.01 → 被误分为Grade 0
  - 当前阈值太高 (2779.41)

目标：
  - 找到一个更合理的阈值
  - 使得A8和H10被正确分级为Grade 3+
  - 不影响其他样本的分级
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")
SEGMENTATION_DIR = BASE_DIR / "results" / "segmentation"
CALIBRATION_DIR = BASE_DIR / "results" / "calibration"

# 样本映射
SAMPLES = {
    "A1": {"position": "A1", "clinical_HER2": 0},      # 无表达
    "A8": {"position": "A8", "clinical_HER2": 3},      # 强表达 ← 问题样本1
    "D1": {"position": "D1", "clinical_HER2": 0},      # 无表达
    "E10": {"position": "E10", "clinical_HER2": 0},    # 无表达
    "G1": {"position": "G1", "clinical_HER2": 0},      # 无表达
    "H2": {"position": "H2", "clinical_HER2": 0},      # 无表达
    "H10": {"position": "H10", "clinical_HER2": 3},    # 强表达 ← 问题样本2
    "J10": {"position": "J10", "clinical_HER2": 0},    # 无表达
}

def analyze_her2_distribution():
    """分析HER2强度分布并找到最优阈值"""
    
    print("\n" + "="*70)
    print("  HER2 阈值优化分析")
    print("="*70 + "\n")
    
    # 收集所有HER2数据
    all_her2_intensities = []
    sample_her2_data = {}
    
    for block, info in SAMPLES.items():
        csv_path = SEGMENTATION_DIR / block / f"{block}_features.csv"
        if not csv_path.exists():
            print(f"❌ {block}: 文件不存在")
            continue
        
        df = pd.read_csv(csv_path)
        if "HER2_nuc_mean" not in df.columns:
            print(f"❌ {block}: 无HER2数据")
            continue
        
        her2_values = df["HER2_nuc_mean"].dropna().values
        all_her2_intensities.extend(her2_values)
        
        mean_val = np.mean(her2_values)
        std_val = np.std(her2_values)
        median_val = np.median(her2_values)
        
        sample_her2_data[block] = {
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "n_cells": len(her2_values),
            "clinical": info["clinical_HER2"],
            "values": her2_values,
        }
        
        status = "✓" if info["clinical_HER2"] == 0 else "⚠️"
        print(f"{status} {block} (临床{info['clinical_HER2']}+):")
        print(f"   平均: {mean_val:8.2f}, 中位: {median_val:8.2f}, 标准差: {std_val:8.2f}, 细胞数: {len(her2_values)}")
    
    print("\n" + "-"*70)
    print("分析：强表达样本(A8, H10)的混合数据\n")
    
    # 分析强表达样本
    strong_samples_data = []
    for block in ["A8", "H10"]:
        if block in sample_her2_data and sample_her2_data[block]["clinical"] == 3:
            data = sample_her2_data[block]
            print(f"⚠️  {block}:")
            print(f"    平均强度: {data['mean']:.2f}")
            print(f"    中位强度: {data['median']:.2f}")
            print(f"    最大强度: {np.max(data['values']):.2f}")
            print(f"    最小强度: {np.min(data['values']):.2f}")
            print(f"    需要阈值 < {data['mean']:.2f} 才能被识别\n")
            strong_samples_data.append(data)
    
    # 分析无表达样本
    print("-"*70)
    print("分析：无表达样本(A1,D1,E10,G1,H2,J10)的数据\n")
    
    grade0_max_values = []
    for block in sample_her2_data:
        if sample_her2_data[block]["clinical"] == 0:
            data = sample_her2_data[block]
            max_val = np.max(data["values"])
            grade0_max_values.append(max_val)
            print(f"✓ {block} (无表达):")
            print(f"   平均: {data['mean']:8.2f}, 最大: {max_val:8.2f}")
    
    print("\n" + "-"*70)
    print("阈值优化建议\n")
    
    # 计算最优阈值
    strong_mean = np.mean([d["mean"] for d in strong_samples_data])
    strong_max = np.max([np.max(d["values"]) for d in strong_samples_data])
    grade0_max = np.max(grade0_max_values)
    
    print(f"强表达样本平均强度: {strong_mean:.2f}")
    print(f"强表达样本最大强度: {strong_max:.2f}")
    print(f"无表达样本最大强度: {grade0_max:.2f}")
    print(f"当前阈值: 2779.41\n")
    
    # 提议新阈值
    # 需要满足：
    # 1. 新阈值 < 强表达样本的平均强度
    # 2. 新阈值 > 无表达样本的最大强度（或至少接近边界）
    
    proposed_neg_threshold = grade0_max + (strong_mean - grade0_max) * 0.3
    print(f"建议的新的neg_threshold: {proposed_neg_threshold:.2f}")
    print(f"  (介于无表达最大值{grade0_max:.2f}和强表达平均值{strong_mean:.2f}之间)\n")
    
    # 计算Otsu阈值的新折扣
    print("-"*70)
    print("Otsu分界线微调\n")
    
    # 加载当前的Otsu结果
    otsu_file = SEGMENTATION_DIR / "otsu_thresholds_detailed_universal.json"
    if otsu_file.exists():
        with open(otsu_file) as f:
            current_otsu = json.load(f)
        
        if "HER2" in current_otsu:
            current = current_otsu["HER2"]
            print(f"当前HER2阈值:")
            print(f"  neg_threshold: {current['neg_threshold']:.2f}")
            print(f"  otsu_1: {current['otsu_1']:.2f}")
            print(f"  otsu_2: {current['otsu_2']:.2f}\n")
    
    # 生成可视化
    plot_her2_distribution(sample_her2_data, CALIBRATION_DIR)
    
    print("="*70)
    print("✅ 分析完成")
    print("="*70)
    
    return sample_her2_data, proposed_neg_threshold


def plot_her2_distribution(sample_data, output_dir):
    """绘制HER2分布直方图"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制无表达样本
    grade0_data = []
    grade3_data = []
    
    for block, data in sample_data.items():
        if data["clinical"] == 0:
            grade0_data.extend(data["values"])
        elif data["clinical"] == 3:
            grade3_data.extend(data["values"])
    
    ax.hist(grade0_data, bins=50, alpha=0.5, label=f"临床Grade 0 (n={len(grade0_data)})", color="blue")
    ax.hist(grade3_data, bins=50, alpha=0.5, label=f"临床Grade 3+ (n={len(grade3_data)})", color="red")
    
    # 标记当前和建议阈值
    ax.axvline(2779.41, color="orange", linestyle="--", linewidth=2, label="当前neg_threshold (2779.41)")
    ax.axvline(np.max(grade0_data), color="green", linestyle=":", linewidth=2, label=f"无表达最大值 ({np.max(grade0_data):.0f})")
    ax.axvline(np.mean(grade3_data), color="purple", linestyle=":", linewidth=2, label=f"强表达平均值 ({np.mean(grade3_data):.0f})")
    
    ax.set_xlabel("HER2 核强度 (Mean)")
    ax.set_ylabel("细胞数")
    ax.set_title("HER2 强度分布：当前阈值与临床分级对比")
    ax.legend()
    ax.grid(alpha=0.3)
    
    output_file = output_dir / "HER2_threshold_optimization.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n📊 分布图已保存: {output_file}")
    plt.close()


def main():
    sample_data, proposed_threshold = analyze_her2_distribution()
    
    # 保存分析结果
    analysis_result = {
        "current_neg_threshold": 2779.41,
        "proposed_neg_threshold": proposed_threshold,
        "analysis_date": str(pd.Timestamp.now()),
        "sample_analysis": {
            block: {
                "mean": float(data["mean"]),
                "std": float(data["std"]),
                "median": float(data["median"]),
                "clinical": data["clinical"],
                "n_cells": data["n_cells"],
            }
            for block, data in sample_data.items()
        }
    }
    
    result_file = CALIBRATION_DIR / "HER2_optimization_analysis.json"
    with open(result_file, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"\n📄 分析结果已保存: {result_file}")


if __name__ == "__main__":
    main()
