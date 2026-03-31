#!/usr/bin/env python3
"""
为每个block单独生成临床诊断报告 (混合方法: Otsu + StdDev)

功能：
1. 逐个block计算该block内的阳性细胞Otsu阈值
2. 对该block内的细胞进行四级分类
3. 为该block生成独立的临床诊断报告
4. 输出结构: results/clinical_reports/{block}/{block}_clinical_report.txt
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


def classify_by_threshold(intensity, neg_threshold, std_dev):
    """使用标准差方法分级"""
    if pd.isna(intensity) or intensity == 0:
        return 0
    if intensity < neg_threshold:
        return 0
    elif intensity < neg_threshold + std_dev:
        return 1
    elif intensity < neg_threshold + 2 * std_dev:
        return 2
    else:
        return 3


def classify_by_otsu(intensity, neg_threshold, otsu_1, otsu_2):
    """使用Otsu方法分级"""
    if pd.isna(intensity) or intensity == 0:
        return 0
    if intensity < neg_threshold:
        return 0
    elif intensity < otsu_1:
        return 1
    elif intensity < otsu_2:
        return 2
    else:
        return 3


def process_single_block(block_name):
    """
    处理单个block：计算Otsu阈值、分级、生成报告
    """
    print(f"\n{'='*70}")
    print(f"  处理 Block: {block_name}")
    print(f"{'='*70}")
    
    # 加载阈值文件
    thresholds = json.load(open(THRESHOLD_FILE))
    
    # 读取该block的特征文件
    features_file = SEGMENTATION_DIR / block_name / f"{block_name}_features.csv"
    if not features_file.exists():
        print(f"⚠️ 文件不存在: {features_file}")
        return None
    
    df = pd.read_csv(features_file)
    total_cells = len(df)
    print(f"📄 加载 {total_cells} 个细胞\n")
    
    groups = BLOCK_TO_GROUP.get(block_name, [])
    otsu_results = {}
    
    # ==================== Step 1: 计算Otsu阈值 ====================
    print("📊 计算每个通道的Otsu阈值...\n")
    
    for group in groups:
        if group == "3ch_Neg":
            for channel in ["ER", "PR", "HER2"]:
                col = f"{channel}_nuc_mean"
                if col not in df.columns:
                    continue
                
                threshold_data = thresholds["3ch_Neg"][channel]
                neg_threshold = threshold_data["threshold"]
                std_dev = threshold_data["std"]
                
                # 获取该通道的所有强度值
                intensities = df[col].dropna().values
                positive_intensities = intensities[intensities >= neg_threshold]
                
                print(f"{channel}:")
                print(f"  - 总细胞数: {len(intensities)}")
                print(f"  - 阳性细胞数: {len(positive_intensities)}")
                
                if len(positive_intensities) < 10:
                    # 阳性细胞太少，使用标准差方法
                    print(f"  - 选用方法: StdDev (阳性细胞不足)")
                    print(f"    • 无表达上限: {neg_threshold:.2f}")
                    print(f"    • 弱/中分界: {neg_threshold + std_dev:.2f}")
                    print(f"    • 中/强分界: {neg_threshold + 2*std_dev:.2f}\n")
                    
                    otsu_results[channel] = {
                        "method": "StdDev",
                        "neg_threshold": float(neg_threshold),
                        "otsu_1": float(neg_threshold + std_dev),
                        "otsu_2": float(neg_threshold + 2 * std_dev),
                        "std_dev": float(std_dev),
                        "n_positive": len(positive_intensities),
                    }
                else:
                    # 阳性细胞足够，使用Otsu方法
                    # 第一次Otsu：分界弱阳性和中阳性
                    otsu_1 = filters.threshold_otsu(positive_intensities.astype(np.uint16))
                    
                    # 第二次Otsu：在强阳性细胞中再分一次
                    strong_cells = positive_intensities[positive_intensities >= otsu_1]
                    if len(strong_cells) >= 10:
                        otsu_2 = filters.threshold_otsu(strong_cells.astype(np.uint16))
                    else:
                        otsu_2 = np.mean(strong_cells) if len(strong_cells) > 0 else otsu_1 + 100
                    
                    print(f"  - 选用方法: Otsu自适应")
                    print(f"    • 无表达上限: {neg_threshold:.2f}")
                    print(f"    • 弱/中分界: {otsu_1:.2f}")
                    print(f"    • 中/强分界: {otsu_2:.2f}\n")
                    
                    otsu_results[channel] = {
                        "method": "Otsu",
                        "neg_threshold": float(neg_threshold),
                        "otsu_1": float(otsu_1),
                        "otsu_2": float(otsu_2),
                        "std_dev": float(std_dev),
                        "n_positive": len(positive_intensities),
                    }
        
        elif group == "Ki67_Neg":
            col = "Ki67_nuc_mean"
            if col in df.columns:
                threshold_data = thresholds["Ki67_Neg"]["Ki67"]
                neg_threshold = threshold_data["threshold"]
                std_dev = threshold_data["std"]
                
                intensities = df[col].dropna().values
                positive_intensities = intensities[intensities >= neg_threshold]
                
                print(f"Ki67:")
                print(f"  - 总细胞数: {len(intensities)}")
                print(f"  - 阳性细胞数: {len(positive_intensities)}")
                
                if len(positive_intensities) < 10:
                    print(f"  - 选用方法: StdDev (阳性细胞不足)")
                    print(f"    • 无表达上限: {neg_threshold:.2f}")
                    print(f"    • 弱/中分界: {neg_threshold + std_dev:.2f}")
                    print(f"    • 中/强分界: {neg_threshold + 2*std_dev:.2f}\n")
                    
                    otsu_results["Ki67"] = {
                        "method": "StdDev",
                        "neg_threshold": float(neg_threshold),
                        "otsu_1": float(neg_threshold + std_dev),
                        "otsu_2": float(neg_threshold + 2 * std_dev),
                        "std_dev": float(std_dev),
                        "n_positive": len(positive_intensities),
                    }
                else:
                    otsu_1 = filters.threshold_otsu(positive_intensities.astype(np.uint16))
                    strong_cells = positive_intensities[positive_intensities >= otsu_1]
                    if len(strong_cells) >= 10:
                        otsu_2 = filters.threshold_otsu(strong_cells.astype(np.uint16))
                    else:
                        otsu_2 = np.mean(strong_cells) if len(strong_cells) > 0 else otsu_1 + 100
                    
                    print(f"  - 选用方法: Otsu自适应")
                    print(f"    • 无表达上限: {neg_threshold:.2f}")
                    print(f"    • 弱/中分界: {otsu_1:.2f}")
                    print(f"    • 中/强分界: {otsu_2:.2f}\n")
                    
                    otsu_results["Ki67"] = {
                        "method": "Otsu",
                        "neg_threshold": float(neg_threshold),
                        "otsu_1": float(otsu_1),
                        "otsu_2": float(otsu_2),
                        "std_dev": float(std_dev),
                        "n_positive": len(positive_intensities),
                    }
    
    # ==================== Step 2: 应用分级 ====================
    print("🔄 应用分级到所有细胞...\n")
    
    for group in groups:
        if group == "3ch_Neg":
            for channel in ["ER", "PR", "HER2"]:
                if channel in otsu_results:
                    col = f"{channel}_nuc_mean"
                    grade_col = f"{channel}_nuc_grade"
                    otsu = otsu_results[channel]
                    
                    if otsu["method"] == "StdDev":
                        df[grade_col] = df[col].apply(
                            lambda x: classify_by_threshold(
                                x, otsu["neg_threshold"], otsu["std_dev"]
                            )
                        )
                    else:  # Otsu
                        df[grade_col] = df[col].apply(
                            lambda x: classify_by_otsu(
                                x, otsu["neg_threshold"], otsu["otsu_1"], otsu["otsu_2"]
                            )
                        )
        
        elif group == "Ki67_Neg":
            if "Ki67" in otsu_results:
                col = "Ki67_nuc_mean"
                grade_col = "Ki67_nuc_grade"
                otsu = otsu_results["Ki67"]
                
                if otsu["method"] == "StdDev":
                    df[grade_col] = df[col].apply(
                        lambda x: classify_by_threshold(
                            x, otsu["neg_threshold"], otsu["std_dev"]
                        )
                    )
                else:  # Otsu
                    df[grade_col] = df[col].apply(
                        lambda x: classify_by_otsu(
                            x, otsu["neg_threshold"], otsu["otsu_1"], otsu["otsu_2"]
                        )
                    )
    
    # 保存分级后的CSV
    output_csv = SEGMENTATION_DIR / block_name / f"{block_name}_features_graded.csv"
    df.to_csv(output_csv, index=False)
    print(f"✓ 分级数据已保存: {output_csv.name}\n")
    
    # ==================== Step 3: 生成临床报告 ====================
    print("📝 生成临床诊断报告...\n")
    
    # 建立输出目录
    report_dir = CLINICAL_REPORT_DIR / block_name
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成完整版报告
    report_text = generate_report(block_name, total_cells, otsu_results, df)
    
    report_file = report_dir / f"{block_name}_clinical_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"✓ 完整版报告已保存: {report_file}")
    
    # 生成统计表
    summary_data = []
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        grade_col = f"{channel}_nuc_grade"
        if grade_col not in df.columns:
            continue
        
        grade_counts = df[grade_col].value_counts().sort_index()
        positive_rate = (total_cells - grade_counts.get(0, 0)) / total_cells * 100
        
        summary_data.append({
            "Channel": channel,
            "Method": otsu_results.get(channel, {}).get("method", "N/A"),
            "Grade_0": grade_counts.get(0, 0),
            "Grade_1": grade_counts.get(1, 0),
            "Grade_2": grade_counts.get(2, 0),
            "Grade_3": grade_counts.get(3, 0),
            "Positive_Rate_%": positive_rate,
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = report_dir / f"{block_name}_expression_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ 统计表已保存: {summary_csv.name}\n")
    
    # 生成分级分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Block {block_name} - 核强度分级分布", fontsize=16, fontweight='bold')
    
    channels = ["ER", "PR", "HER2", "Ki67"]
    for idx, channel in enumerate(channels):
        ax = axes[idx // 2, idx % 2]
        grade_col = f"{channel}_nuc_grade"
        
        if grade_col in df.columns:
            grade_counts = df[grade_col].value_counts().sort_index().to_dict()
            grades = [0, 1, 2, 3]
            counts = [grade_counts.get(g, 0) for g in grades]
            
            colors = ['lightgray', 'yellow', 'orange', 'red']
            bars = ax.bar(grades, counts, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_xlabel('分级', fontsize=11)
            ax.set_ylabel('细胞数', fontsize=11)
            ax.set_title(f"{channel} ({otsu_results.get(channel, {}).get('method', 'N/A')})", 
                        fontsize=12, fontweight='bold')
            ax.set_xticks(grades)
            ax.set_xticklabels(['0\n无表达', '1+\n弱', '2+\n中', '3+\n强'])
            
            # 添加柱状图数值
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    hist_file = report_dir / f"{block_name}_grading_distribution.png"
    plt.savefig(hist_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 分布直方图已保存: {hist_file.name}\n")
    
    return df, otsu_results


def generate_report(block_name, total_cells, otsu_results, df):
    """生成临床诊断报告"""
    report = ""
    
    # 报告头
    report += "="*80 + "\n"
    report += f"  组织微阵列 (TMA) Block {block_name} 病理诊断报告\n"
    report += "="*80 + "\n"
    report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"总核数: {total_cells}\n"
    report += f"分析方法: 混合方法 (Otsu自适应 + 标准差)\n\n"
    
    # 分级标准说明
    report += "-"*80 + "\n"
    report += "分级标准说明:\n"
    report += "-"*80 + "\n"
    report += "Grade 0     : 无表达\n"
    report += "Grade 1+    : 弱阳性\n"
    report += "Grade 2+    : 中阳性\n"
    report += "Grade 3+    : 强阳性\n\n"
    
    # 每个标志物的详情
    report += "-"*80 + "\n"
    report += "标志物分级详情:\n"
    report += "-"*80 + "\n\n"
    
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        grade_col = f"{channel}_nuc_grade"
        if grade_col not in df.columns:
            continue
        
        if channel not in otsu_results:
            continue
        
        otsu = otsu_results[channel]
        grade_counts = df[grade_col].value_counts().sort_index()
        positive_count = total_cells - grade_counts.get(0, 0)
        positive_rate = positive_count / total_cells * 100 if total_cells > 0 else 0
        
        report += f"{channel} (方法: {otsu['method']}):\n"
        report += f"  阈值配置:\n"
        report += f"    • 无表达上限: {otsu['neg_threshold']:.2f}\n"
        report += f"    • 弱/中分界: {otsu['otsu_1']:.2f}\n"
        report += f"    • 中/强分界: {otsu['otsu_2']:.2f}\n"
        report += f"\n  分级分布:\n"
        report += f"    • Grade 0 (无表达): {grade_counts.get(0, 0):6d} 核 ({grade_counts.get(0, 0)/total_cells*100:5.1f}%)\n"
        report += f"    • Grade 1+ (弱): {grade_counts.get(1, 0):6d} 核 ({grade_counts.get(1, 0)/total_cells*100:5.1f}%)\n"
        report += f"    • Grade 2+ (中): {grade_counts.get(2, 0):6d} 核 ({grade_counts.get(2, 0)/total_cells*100:5.1f}%)\n"
        report += f"    • Grade 3+ (强): {grade_counts.get(3, 0):6d} 核 ({grade_counts.get(3, 0)/total_cells*100:5.1f}%)\n"
        report += f"\n  诊断: {channel} {'阳性' if positive_rate >= 1 else '阴性'} ({positive_rate:.1f}% 阳性表达)\n\n"
    
    # 综合诊断
    report += "-"*80 + "\n"
    report += "初步综合诊断:\n"
    report += "-"*80 + "\n\n"
    
    er_positive = df["ER_nuc_grade"].gt(0).sum() if "ER_nuc_grade" in df.columns else 0
    pr_positive = df["PR_nuc_grade"].gt(0).sum() if "PR_nuc_grade" in df.columns else 0
    her2_positive = df["HER2_nuc_grade"].gt(0).sum() if "HER2_nuc_grade" in df.columns else 0
    ki67_positive = df["Ki67_nuc_grade"].gt(0).sum() if "Ki67_nuc_grade" in df.columns else 0
    
    hr_positive = er_positive > 0 or pr_positive > 0
    her2_positive_rate = her2_positive / total_cells * 100 if total_cells > 0 else 0
    ki67_index = ki67_positive / total_cells * 100 if total_cells > 0 else 0
    
    report += "□ " + ("激素受体阳性 (HR+)" if hr_positive else "激素受体阴性 (HR-)") + "\n"
    report += "□ " + ("HER2 阳性" if her2_positive_rate >= 10 else "HER2 阴性") + "\n"
    report += f"□ Ki67 增殖指数: {ki67_index:.1f}%\n"
    
    if ki67_index < 10:
        report += "  (低增殖活性)\n"
    elif ki67_index < 30:
        report += "  (中等增殖活性)\n"
    else:
        report += "  (高增殖活性)\n"
    
    report += "\n" + "="*80 + "\n"
    
    return report


def main():
    print("\n" + "="*70)
    print("  为每个Block生成独立的临床诊断报告")
    print("="*70)
    
    all_blocks = sorted(set(BLOCK_TO_GROUP.keys()))
    
    for block_name in all_blocks:
        try:
            process_single_block(block_name)
        except Exception as e:
            print(f"❌ 处理 {block_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ 所有Block的报告已生成！")
    print("="*70)
    print(f"\n输出位置: {CLINICAL_REPORT_DIR}")
    print("输出结构:")
    for block in all_blocks:
        print(f"  {block}/")
        print(f"    ├─ {block}_clinical_report.txt")
        print(f"    ├─ {block}_expression_statistics.csv")
        print(f"    └─ {block}_grading_distribution.png")


if __name__ == "__main__":
    main()
