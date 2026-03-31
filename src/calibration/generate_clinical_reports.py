#!/usr/bin/env python3
"""
生成临床诊断报告 - 基于Otsu分级的完整和简化版本

功能：
1. 计算每个通道的Otsu分界线
2. 对所有样本中的细胞进行四级分类（0/1+/2+/3+）
3. 生成两种格式的临床报告：
   - 选项A：完整临床诊断报告（TXT）
   - 选项B：简化统计版本（CSV+TXT）
4. 生成可视化（分级分布直方图）
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from skimage import filters

# ==================== 路径配置 ====================
BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")
SEGMENTATION_DIR = BASE_DIR / "results" / "segmentation"
CALIBRATION_DIR = BASE_DIR / "results" / "calibration"
CLINICAL_REPORT_DIR = BASE_DIR / "results" / "clinical_reports"
CLINICAL_REPORT_DIR.mkdir(exist_ok=True)

# 阈值文件
THRESHOLD_FILE = CALIBRATION_DIR / "thresholds_raw_nuclei.json"

# Block分组配置
BLOCK_GROUPS = {
    "3ch_Neg": ["A1", "D1", "E10", "H2", "H10", "J10"],
    "Ki67_Neg": ["A8", "D1", "G1", "H10"],
}

BLOCK_TO_GROUP = {}
for group, blocks in BLOCK_GROUPS.items():
    for block in blocks:
        if block not in BLOCK_TO_GROUP:
            BLOCK_TO_GROUP[block] = []
        BLOCK_TO_GROUP[block].append(group)


def calculate_otsu_thresholds():
    """
    为每个通道计算两个Otsu阈值（区分弱/中/强表达）
    返回: {channel: {"neg_threshold": float, "otsu_1": float, "otsu_2": float}}
    """
    thresholds = json.load(open(THRESHOLD_FILE))
    otsu_results = {}
    
    print("\n📊 计算Otsu分界线...\n")
    
    # 收集所有通道的强度数据
    channel_data = {
        "ER": [],
        "PR": [],
        "HER2": [],
        "Ki67": [],
    }
    
    # 遍历所有block，收集强度数据
    for block_name in sorted(BLOCK_TO_GROUP.keys()):
        features_file = SEGMENTATION_DIR / block_name / f"{block_name}_features.csv"
        if not features_file.exists():
            continue
        
        df = pd.read_csv(features_file)
        groups = BLOCK_TO_GROUP.get(block_name, [])
        
        for group in groups:
            if group == "3ch_Neg":
                for channel in ["ER", "PR", "HER2"]:
                    col = f"{channel}_nuc_mean"
                    if col in df.columns:
                        channel_data[channel].extend(df[col].dropna().values)
            elif group == "Ki67_Neg":
                col = "Ki67_nuc_mean"
                if col in df.columns:
                    channel_data["Ki67"].extend(df[col].dropna().values)
    
    # 为每个通道计算Otsu阈值
    for channel, intensities in channel_data.items():
        if len(intensities) == 0:
            continue
        
        intensities = np.array(intensities)
        threshold_data = None
        
        # 从阈值文件中获取阴性阈值
        if channel in ["ER", "PR", "HER2"]:
            threshold_data = thresholds["3ch_Neg"][channel]
        elif channel == "Ki67":
            threshold_data = thresholds["Ki67_Neg"]["Ki67"]
        
        if threshold_data is None:
            continue
        
        neg_threshold = threshold_data["threshold"]
        std_dev = threshold_data["std"]
        
        # 分为无表达和有表达两组
        positive_cells = intensities[intensities >= neg_threshold]
        
        print(f"  {channel}: {len(positive_cells)} 个阳性细胞", end="")
        
        # === 混合方法：根据细胞数量选择分级方式 ===
        if len(positive_cells) >= 10:
            # 细胞足够：使用Otsu自适应
            print(" → 使用 Otsu 自适应分级")
            
            # 第一次Otsu：分界弱阳性和中阳性
            otsu_1 = filters.threshold_otsu(positive_cells.astype(np.uint16))
            
            # 第二次Otsu：在强阳性细胞中再分一次
            strong_cells = positive_cells[positive_cells >= otsu_1]
            if len(strong_cells) >= 10:
                otsu_2 = filters.threshold_otsu(strong_cells.astype(np.uint16))
            else:
                # 如果强阳性细胞过少，使用强阳性的平均值
                otsu_2 = np.mean(strong_cells) if len(strong_cells) > 0 else otsu_1 + 100
            
            otsu_results[channel] = {
                "method": "Otsu",
                "neg_threshold": float(neg_threshold),
                "otsu_1": float(otsu_1),
                "otsu_2": float(otsu_2),
                "n_positive": len(positive_cells),
                "mean_positive": float(np.mean(positive_cells)),
                "std_positive": float(np.std(positive_cells)),
            }
            
            print(f"    - 无表达阈值: {neg_threshold:.2f}")
            print(f"    - 弱/中分界: {otsu_1:.2f}")
            print(f"    - 中/强分界: {otsu_2:.2f}\n")
        
        else:
            # 细胞不足：使用标准差方法
            print(" → 细胞不足，使用 标准差 固定阈值")
            
            # 使用标准差来定义分界线
            # Grade 1: threshold ~ threshold + std
            # Grade 2: threshold + std ~ threshold + 2*std
            # Grade 3: >= threshold + 2*std
            threshold_1 = neg_threshold + std_dev
            threshold_2 = neg_threshold + 2 * std_dev
            
            otsu_results[channel] = {
                "method": "StdDev",
                "neg_threshold": float(neg_threshold),
                "threshold_1": float(threshold_1),
                "threshold_2": float(threshold_2),
                "n_positive": len(positive_cells),
                "mean_positive": float(np.mean(positive_cells)),
                "std_positive": float(np.std(positive_cells)),
                "std_dev": float(std_dev),
            }
            
            print(f"    - 无表达阈值: {neg_threshold:.2f}")
            print(f"    - 弱/中分界: {threshold_1:.2f}")
            print(f"    - 中/强分界: {threshold_2:.2f}\n")
    
    # 保存Otsu阈值
    otsu_file = SEGMENTATION_DIR / "otsu_thresholds_detailed.json"
    with open(otsu_file, 'w') as f:
        json.dump(otsu_results, f, indent=2)
    
    return otsu_results


def classify_intensity(intensity, threshold_config):
    """
    根据阈值配置分级
    支持两种方法：Otsu 自适应 或 标准差固定阈值
    """
    if pd.isna(intensity) or intensity == 0:
        return 0
    
    method = threshold_config.get("method", "Otsu")
    neg_threshold = threshold_config["neg_threshold"]
    
    if intensity < neg_threshold:
        return 0
    
    if method == "Otsu":
        # 使用Otsu分界线
        otsu_1 = threshold_config["otsu_1"]
        otsu_2 = threshold_config["otsu_2"]
        
        if intensity < otsu_1:
            return 1
        elif intensity < otsu_2:
            return 2
        else:
            return 3
    
    elif method == "StdDev":
        # 使用标准差分界线
        threshold_1 = threshold_config["threshold_1"]
        threshold_2 = threshold_config["threshold_2"]
        
        if intensity < threshold_1:
            return 1
        elif intensity < threshold_2:
            return 2
        else:
            return 3
    
    return 0


def apply_otsu_grading(otsu_results):
    """为所有样本应用混合分级（Otsu + 标准差）"""
    print("\n🔄 应用混合分级到所有样本...\n")
    
    all_results = []
    
    for block_name in sorted(BLOCK_TO_GROUP.keys()):
        features_file = SEGMENTATION_DIR / block_name / f"{block_name}_features.csv"
        if not features_file.exists():
            print(f"  ⚠️ {block_name}: 文件不存在")
            continue
        
        df = pd.read_csv(features_file)
        print(f"  📄 {block_name}: 处理 {len(df)} 个细胞")
        
        groups = BLOCK_TO_GROUP.get(block_name, [])
        
        for group in groups:
            if group == "3ch_Neg":
                for channel in ["ER", "PR", "HER2"]:
                    if channel in otsu_results:
                        col = f"{channel}_nuc_mean"
                        grade_col = f"{channel}_nuc_grade_otsu"
                        
                        if col in df.columns:
                            df[grade_col] = df[col].apply(
                                lambda x: classify_intensity(x, otsu_results[channel])
                            )
            
            elif group == "Ki67_Neg":
                if "Ki67" in otsu_results:
                    col = "Ki67_nuc_mean"
                    grade_col = "Ki67_nuc_grade_otsu"
                    
                    if col in df.columns:
                        df[grade_col] = df[col].apply(
                            lambda x: classify_intensity(x, otsu_results["Ki67"])
                        )
        
        # 保存分级后的CSV
        output_file = SEGMENTATION_DIR / block_name / f"{block_name}_features_otsu_graded.csv"
        df.to_csv(output_file, index=False)
        
        all_results.append(df)
    
    return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()


def generate_clinical_report_option_a(otsu_results, all_results_df):
    """
    选项A：完整临床诊断报告
    包含：样本概述、详细分析、分子分型、综合诊断
    """
    report_text = ""
    
    # ==================== 报告头部 ====================
    report_text += "="*80 + "\n"
    report_text += "  乳腺癌TMA病理诊断报告 - 完整版\n"
    report_text += "="*80 + "\n"
    report_text += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_text += f"分析方法: Otsu自适应阈值分级\n"
    report_text += f"总样本数: {len(set(BLOCK_TO_GROUP.keys()))}\n"
    report_text += f"总核数: {len(all_results_df)}\n\n"
    
    # ==================== 分级标准说明 ====================
    report_text += "-"*80 + "\n"
    report_text += "1. 分级标准说明\n"
    report_text += "-"*80 + "\n"
    report_text += "Grade 0 (0)     : 无表达      (强度 < 阴性阈值)\n"
    report_text += "Grade 1+ (1)    : 弱阳性      (阴性阈值 ≤ 强度 < Otsu_1)\n"
    report_text += "Grade 2+ (2)    : 中阳性      (Otsu_1 ≤ 强度 < Otsu_2)\n"
    report_text += "Grade 3+ (3)    : 强阳性      (强度 ≥ Otsu_2)\n\n"
    
    # ==================== 阈值详情 ====================
    report_text += "-"*80 + "\n"
    report_text += "2. 每个标志物的分级阈值\n"
    report_text += "-"*80 + "\n\n"
    
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        if channel not in otsu_results:
            continue
        
        otsu = otsu_results[channel]
        method = otsu.get("method", "Otsu")
        
        report_text += f"{channel} 标志物 [{method} 方法]:\n"
        report_text += f"  无表达上限 (Neg Threshold)  : {otsu['neg_threshold']:.2f}\n"
        
        if method == "Otsu":
            report_text += f"  弱/中分界 (Otsu_1)         : {otsu['otsu_1']:.2f}\n"
            report_text += f"  中/强分界 (Otsu_2)         : {otsu['otsu_2']:.2f}\n"
        elif method == "StdDev":
            report_text += f"  弱/中分界 (threshold+1σ)  : {otsu['threshold_1']:.2f}\n"
            report_text += f"  中/强分界 (threshold+2σ)  : {otsu['threshold_2']:.2f}\n"
        
        report_text += f"  阳性细胞数                : {otsu['n_positive']}\n"
        report_text += f"  阳性细胞平均强度          : {otsu['mean_positive']:.2f} ± {otsu['std_positive']:.2f}\n\n"
    
    # ==================== 全局统计 ====================
    report_text += "-"*80 + "\n"
    report_text += "3. 全局表达分析\n"
    report_text += "-"*80 + "\n\n"
    
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        grade_col = f"{channel}_nuc_grade_otsu"
        if grade_col not in all_results_df.columns:
            continue
        
        grade_counts = all_results_df[grade_col].value_counts().sort_index()
        total = len(all_results_df)
        
        report_text += f"{channel} 表达分析:\n"
        report_text += f"  ├─ Grade 0 (无表达) : {grade_counts.get(0, 0):6d} 核 ({grade_counts.get(0, 0)/total*100:5.1f}%)\n"
        report_text += f"  ├─ Grade 1+ (弱)    : {grade_counts.get(1, 0):6d} 核 ({grade_counts.get(1, 0)/total*100:5.1f}%)\n"
        report_text += f"  ├─ Grade 2+ (中)    : {grade_counts.get(2, 0):6d} 核 ({grade_counts.get(2, 0)/total*100:5.1f}%)\n"
        report_text += f"  └─ Grade 3+ (强)    : {grade_counts.get(3, 0):6d} 核 ({grade_counts.get(3, 0)/total*100:5.1f}%)\n"
        
        positive_rate = (total - grade_counts.get(0, 0)) / total * 100 if total > 0 else 0
        report_text += f"  → 整体表达阳性率: {positive_rate:.1f}%\n\n"
    
    # ==================== 分子分型 ====================
    report_text += "-"*80 + "\n"
    report_text += "4. 初步分子分型评估\n"
    report_text += "-"*80 + "\n\n"
    
    # ER/PR阳性率
    er_positive = all_results_df["ER_nuc_grade_otsu"].gt(0).sum() if "ER_nuc_grade_otsu" in all_results_df.columns else 0
    pr_positive = all_results_df["PR_nuc_grade_otsu"].gt(0).sum() if "PR_nuc_grade_otsu" in all_results_df.columns else 0
    her2_positive = all_results_df["HER2_nuc_grade_otsu"].gt(0).sum() if "HER2_nuc_grade_otsu" in all_results_df.columns else 0
    
    report_text += f"激素受体状态:\n"
    report_text += f"  ER 阳性 : {'是' if er_positive/len(all_results_df)*100 >= 1 else '否'} ({er_positive/len(all_results_df)*100:.1f}% 核表达)\n"
    report_text += f"  PR 阳性 : {'是' if pr_positive/len(all_results_df)*100 >= 1 else '否'} ({pr_positive/len(all_results_df)*100:.1f}% 核表达)\n\n"
    
    report_text += f"HER2 状态:\n"
    report_text += f"  HER2 阳性: {'是' if her2_positive/len(all_results_df)*100 >= 1 else '否'} ({her2_positive/len(all_results_df)*100:.1f}% 核表达)\n\n"
    
    # Ki67增殖指数
    ki67_positive = all_results_df["Ki67_nuc_grade_otsu"].gt(0).sum() if "Ki67_nuc_grade_otsu" in all_results_df.columns else 0
    ki67_index = ki67_positive/len(all_results_df)*100 if len(all_results_df) > 0 else 0
    
    report_text += f"增殖活性:\n"
    report_text += f"  Ki67 增殖指数: {ki67_index:.1f}%\n"
    report_text += f"  评估: "
    if ki67_index < 10:
        report_text += "低增殖活性 (预后良好)\n\n"
    elif ki67_index < 30:
        report_text += "中等增殖活性\n\n"
    else:
        report_text += "高增殖活性 (预后相对较差)\n\n"
    
    # ==================== 分子亚型预测 ====================
    report_text += "-"*80 + "\n"
    report_text += "5. 分子亚型初步预测\n"
    report_text += "-"*80 + "\n\n"
    
    hr_positive = er_positive > 0 or pr_positive > 0
    her2_amp = her2_positive > len(all_results_df) * 0.1  # >10%表达为HER2阳性
    
    if hr_positive and her2_amp:
        subtype = "Luminal B (HR+/HER2+)"
    elif hr_positive and not her2_amp:
        subtype = "Luminal A (HR+/HER2-)"
    elif not hr_positive and her2_amp:
        subtype = "HER2富集型 (HR-/HER2+)"
    else:
        subtype = "三阴性 (HR-/HER2-)"
    
    report_text += f"初步分子亚型: {subtype}\n"
    report_text += f"（需结合临床和其他检验结果进一步确认）\n\n"
    
    report_text += "="*80 + "\n"
    report_text += "报告结束\n"
    report_text += "="*80 + "\n"
    
    return report_text


def generate_clinical_report_option_b(otsu_results, all_results_df):
    """
    选项B：简化统计版本
    包含：基本统计、简化诊断
    """
    report_text = ""
    
    report_text += "="*60 + "\n"
    report_text += "  乳腺癌TMA病理诊断报告 - 简化版\n"
    report_text += "="*60 + "\n"
    report_text += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_text += f"总核数: {len(all_results_df)}\n\n"
    
    report_text += "-"*60 + "\n"
    report_text += "表达统计\n"
    report_text += "-"*60 + "\n\n"
    
    summary_data = []
    
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        grade_col = f"{channel}_nuc_grade_otsu"
        if grade_col not in all_results_df.columns:
            continue
        
        grade_counts = all_results_df[grade_col].value_counts().sort_index()
        total = len(all_results_df)
        positive_rate = (total - grade_counts.get(0, 0)) / total * 100 if total > 0 else 0
        
        report_text += f"{channel}: "
        if positive_rate >= 1:
            report_text += f"阳性 ({positive_rate:.1f}%)"
        else:
            report_text += "阴性 (0%)"
        report_text += "\n"
        
        summary_data.append({
            "Channel": channel,
            "Grade_0": grade_counts.get(0, 0),
            "Grade_1": grade_counts.get(1, 0),
            "Grade_2": grade_counts.get(2, 0),
            "Grade_3": grade_counts.get(3, 0),
            "Positive_Rate_%": positive_rate,
        })
    
    report_text += "\n"
    
    # 简化诊断
    report_text += "-"*60 + "\n"
    report_text += "初步诊断\n"
    report_text += "-"*60 + "\n\n"
    
    er_positive = all_results_df["ER_nuc_grade_otsu"].gt(0).sum() if "ER_nuc_grade_otsu" in all_results_df.columns else 0
    pr_positive = all_results_df["PR_nuc_grade_otsu"].gt(0).sum() if "PR_nuc_grade_otsu" in all_results_df.columns else 0
    her2_positive = all_results_df["HER2_nuc_grade_otsu"].gt(0).sum() if "HER2_nuc_grade_otsu" in all_results_df.columns else 0
    ki67_positive = all_results_df["Ki67_nuc_grade_otsu"].gt(0).sum() if "Ki67_nuc_grade_otsu" in all_results_df.columns else 0
    
    if er_positive > 0 or pr_positive > 0:
        report_text += "□ 激素受体阳性 (HR+)\n"
    else:
        report_text += "□ 激素受体阴性 (HR-)\n"
    
    if her2_positive > len(all_results_df) * 0.1:
        report_text += "□ HER2 阳性\n"
    else:
        report_text += "□ HER2 阴性\n"
    
    ki67_index = ki67_positive/len(all_results_df)*100 if len(all_results_df) > 0 else 0
    if ki67_index >= 30:
        report_text += "□ 高增殖活性\n"
    elif ki67_index >= 10:
        report_text += "□ 中等增殖活性\n"
    else:
        report_text += "□ 低增殖活性\n"
    
    report_text += "\n" + "="*60 + "\n"
    
    return report_text, summary_data


def generate_visualizations(otsu_results, all_results_df):
    """生成可视化图表"""
    print("\n📊 生成可视化...\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('核强度表达分级分布 (Otsu方法)', fontsize=14, fontweight='bold')
    
    channels = ["ER", "PR", "HER2", "Ki67"]
    for idx, channel in enumerate(channels):
        ax = axes[idx // 2, idx % 2]
        grade_col = f"{channel}_nuc_grade_otsu"
        
        if grade_col in all_results_df.columns:
            grade_counts = all_results_df[grade_col].value_counts().sort_index()
            
            # 绘制柱状图
            grades = ['0\n(无表达)', '1+\n(弱)', '2+\n(中)', '3+\n(强)']
            counts = [grade_counts.get(i, 0) for i in range(4)]
            colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
            
            bars = ax.bar(range(4), counts, color=colors, alpha=0.7, edgecolor='black')
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                if height > 0:
                    percentage = count / len(all_results_df) * 100
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}\n({percentage:.1f}%)',
                           ha='center', va='bottom', fontsize=9)
            
            ax.set_ylabel('细胞数')
            ax.set_title(f'{channel} 表达分级')
            ax.set_xticks(range(4))
            ax.set_xticklabels(grades)
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CLINICAL_REPORT_DIR / 'expression_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ 可视化已保存: {CLINICAL_REPORT_DIR / 'expression_distribution.png'}")
    plt.close()


def main():
    print("\n" + "="*60)
    print("  临床诊断报告生成系统")
    print("="*60)
    
    # Step 1: 计算Otsu阈值
    otsu_results = calculate_otsu_thresholds()
    
    if not otsu_results:
        print("\n❌ 无法计算Otsu阈值，退出")
        return
    
    # Step 2: 应用Otsu分级
    all_results_df = apply_otsu_grading(otsu_results)
    
    if all_results_df.empty:
        print("\n❌ 无分级结果，退出")
        return
    
    # Step 3: 生成选项A（完整版）
    print("\n📝 生成选项A（完整临床诊断报告）...\n")
    report_a = generate_clinical_report_option_a(otsu_results, all_results_df)
    
    report_a_file = CLINICAL_REPORT_DIR / "clinical_report_complete.txt"
    with open(report_a_file, 'w', encoding='utf-8') as f:
        f.write(report_a)
    print(f"✓ 完整版已保存: {report_a_file}")
    
    # Step 4: 生成选项B（简化版）
    print("\n📝 生成选项B（简化统计版本）...\n")
    report_b, summary_data = generate_clinical_report_option_b(otsu_results, all_results_df)
    
    report_b_file = CLINICAL_REPORT_DIR / "clinical_report_summary.txt"
    with open(report_b_file, 'w', encoding='utf-8') as f:
        f.write(report_b)
    print(f"✓ 简化版已保存: {report_b_file}")
    
    # 保存统计CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv_file = CLINICAL_REPORT_DIR / "expression_statistics.csv"
    summary_df.to_csv(summary_csv_file, index=False)
    print(f"✓ 统计数据已保存: {summary_csv_file}")
    
    # Step 5: 生成可视化
    try:
        generate_visualizations(otsu_results, all_results_df)
    except Exception as e:
        print(f"⚠️ 可视化生成失败: {e}")
    
    print("\n" + "="*60)
    print("✅ 报告生成完成！")
    print("="*60)
    print(f"\n输出位置: {CLINICAL_REPORT_DIR}")
    print(f"  - 完整版: clinical_report_complete.txt")
    print(f"  - 简化版: clinical_report_summary.txt")
    print(f"  - 统计表: expression_statistics.csv")
    print(f"  - 可视化: expression_distribution.png")


if __name__ == "__main__":
    main()
