#!/usr/bin/env python3
"""
为每个block单独生成临床诊断报告 (混合方法: Otsu + StdDev)

功能：
1. 逐个block计算该block内的阳性细胞Otsu阈值
2. 对该block内的细胞进行四级分类
3. 为该block生成独立的临床诊断报告
4. Output structure: results/clinical_reports/{dataset}/{block}/{block}_{dataset}_clinical_report.txt
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from skimage import filters

from calibration.config import REPORT_SCHEMA

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
    """使用标准差方法Grade"""
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


def infer_dataset_for_block(block_name):
    from calibration.config import BASE_DIR, SEGMENTATION_DIR
    for dataset in ['TMAd', 'TMAe']:
        if (SEGMENTATION_DIR / dataset / block_name / f'{block_name}_{dataset}_features.csv').exists():
            return dataset
        if (BASE_DIR / 'results' / 'negative_controls' / 'segmentation' / block_name / f'{block_name}_{dataset}_features.csv').exists():
            return dataset
        if (SEGMENTATION_DIR / block_name / f'{block_name}_{dataset}_features.csv').exists():
            return dataset
    return None


def classify_by_otsu(intensity, neg_threshold, otsu_1, otsu_2):
    """使用Otsu方法Grade"""
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
    处理单个block：计算Otsu阈值、Grade、生成报告
    """
    print(f"\n{'='*70}")
    print(f"  Processing Block: {block_name}")
    print(f"{'='*70}")
    
    # 加载阈值文件
    thresholds = json.load(open(THRESHOLD_FILE))
    
    # 读取该block的特征文件
    dataset = infer_dataset_for_block(block_name)
    if dataset is None:
        print(f"⚠️ Cannot find segmentation results for {block_name} (TMAd/TMAe)")
        return None

    features_file = SEGMENTATION_DIR / dataset / block_name / f"{block_name}_{dataset}_features.csv"
    if not features_file.exists():
        features_file = BASE_DIR / "results" / "negative_controls" / "segmentation" / block_name / f"{block_name}_{dataset}_features.csv"
    if not features_file.exists():
        features_file = SEGMENTATION_DIR / block_name / f"{block_name}_{dataset}_features.csv"
    if not features_file.exists():
        return None
    
    df = pd.read_csv(features_file)
    total_cells = len(df)
    print(f"📄 Loaded {total_cells} cells\n")
    
    groups = BLOCK_TO_GROUP.get(block_name, [])
    otsu_results = {}
    
    # ==================== Step 1: 计算Otsu阈值 ====================
    print("📊 Calculating Otsu thresholds for each channel...\n")
    
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        if True:
            if True:
                col = f"{channel}_nuc_mean"
                if col not in df.columns:
                    continue
                
                group_key = "Ki67_Neg" if channel == "Ki67" else "3ch_Neg"
                threshold_data = thresholds[group_key][channel]
                neg_threshold = threshold_data["threshold"]
                std_dev = threshold_data["std"]
                
                # 获取该通道的所有Strong度值
                intensities = df[col].dropna().values
                positive_intensities = intensities[intensities >= neg_threshold]
                
                print(f"{channel}:")
                print(f"  - Total cells: {len(intensities)}")
                print(f"  - Positive cells: {len(positive_intensities)}")
                
                if len(positive_intensities) < 10:
                    # 阳性细胞太少，使用标准差方法
                    print(f"  - Selected method: StdDev (insufficient positive cells)")
                    print(f"    • No Expression Upper Limit: {neg_threshold:.2f}")
                    print(f"    • Weak/Moderate Boundary: {neg_threshold + std_dev:.2f}")
                    print(f"    • Moderate/Strong Boundary: {neg_threshold + 2*std_dev:.2f}\n")
                    
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
                    # 第一次Otsu：分界Weak阳性和Mod阳性
                    otsu_1 = filters.threshold_otsu(positive_intensities.astype(np.uint16))
                    
                    # 第二次Otsu：在Strong阳性细胞Mod再分一次
                    strong_cells = positive_intensities[positive_intensities >= otsu_1]
                    if len(strong_cells) >= 10:
                        otsu_2 = filters.threshold_otsu(strong_cells.astype(np.uint16))
                    else:
                        otsu_2 = np.mean(strong_cells) if len(strong_cells) > 0 else otsu_1 + 100
                    
                    print(f"  - Selected method: Otsu adaptive")
                    print(f"    • No Expression Upper Limit: {neg_threshold:.2f}")
                    print(f"    • Weak/Moderate Boundary: {otsu_1:.2f}")
                    print(f"    • Moderate/Strong Boundary: {otsu_2:.2f}\n")
                    
                    otsu_results[channel] = {
                        "method": "Otsu",
                        "neg_threshold": float(neg_threshold),
                        "otsu_1": float(otsu_1),
                        "otsu_2": float(otsu_2),
                        "std_dev": float(std_dev),
                        "n_positive": len(positive_intensities),
                    }
        

    # ==================== Step 2: 应用Grade ====================
    print("🔄 Applying grading to all cells...\n")
    
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        if True:
            if True:
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
        

    # 保存Grade后的CSV
    output_csv = SEGMENTATION_DIR / dataset / block_name / f"{block_name}_{dataset}_features_graded.csv"
    df.to_csv(output_csv, index=False)
    print(f"✓ Grade data saved: {output_csv.name}\n")
    
    # ==================== Step 3: 生成临床报告 ====================
    print("📝 Generating clinical diagnostic report...\n")
    
    # 建立输出目录
    report_dir = CLINICAL_REPORT_DIR / dataset / block_name
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成完整版报告
    report_text = generate_report(block_name, total_cells, otsu_results, df)
    
    report_file = report_dir / f"{block_name}_{dataset}_clinical_report.txt"
    with open(report_file, 'w', encoding='utf-8-sig') as f:
        f.write(report_text)
    print(f"✓ Complete report saved: {report_file}")
    
    # 生成统计表
    summary_data = []
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        grade_col = f"{channel}_nuc_grade"
        if grade_col not in df.columns:
            continue
        
        grade_counts = df[grade_col].value_counts().sort_index()
        positive_rate = (total_cells - grade_counts.get(0, 0)) / total_cells * 100
        
        summary_data.append({
            REPORT_SCHEMA["channel"]: channel,
            REPORT_SCHEMA["method"]: otsu_results.get(channel, {}).get("method", "N/A"),
            REPORT_SCHEMA["grade_0"]: grade_counts.get(0, 0),
            REPORT_SCHEMA["grade_1"]: grade_counts.get(1, 0),
            REPORT_SCHEMA["grade_2"]: grade_counts.get(2, 0),
            REPORT_SCHEMA["grade_3"]: grade_counts.get(3, 0),
            REPORT_SCHEMA["positive_rate"]: positive_rate,
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = report_dir / f"{block_name}_{dataset}_expression_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Statistics table saved: {summary_csv.name}\n")
    
    # 生成Grade分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Block {block_name} - Nuclear Intensity Grading Distribution", fontsize=16, fontweight='bold')
    
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
            ax.set_xlabel('Grade', fontsize=11)
            ax.set_ylabel('Cell Count', fontsize=11)
            ax.set_title(f"{channel} ({otsu_results.get(channel, {}).get('method', 'N/A')})", 
                        fontsize=12, fontweight='bold')
            ax.set_xticks(grades)
            ax.set_xticklabels(['0\nNo Expr', '1+\nWeak', '2+\nMod', '3+\nStrong'])
            
            # 添加柱状图数值
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    hist_file = report_dir / f"{block_name}_{dataset}_grading_distribution.png"
    plt.savefig(hist_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Distribution histogram saved: {hist_file.name}\n")
    
    return df, otsu_results


def generate_report(block_name, total_cells, otsu_results, df):
    """生成临床诊断报告"""
    report = ""
    
    # 报告头
    report += "="*80 + "\n"
    report += f"  Tissue Microarray (TMA) Block {block_name} Pathology Diagnostic Report\n"
    report += "="*80 + "\n"
    report += f"Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"Total Nuclei: {total_cells}\n"
    report += f"Analysis Method: Hybrid Method (Otsu Adaptive + Standard Deviation)\n\n"
    
    # Grade标准说明
    report += "-"*80 + "\n"
    report += "Grading Criteria Description:\n"
    report += "-"*80 + "\n"
    report += "Grade 0     : No Expression\n"
    report += "Grade 1+    : Weak Positive\n"
    report += "Grade 2+    : Moderate Positive\n"
    report += "Grade 3+    : Strong Positive\n\n"
    
    # 每个标志物的详情
    report += "-"*80 + "\n"
    report += "Marker Grading Details:\n"
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
        
        report += f"{channel} (Method: {otsu['method']}):\n"
        report += f"  Threshold Configuration:\n"
        report += f"    • No Expression Upper Limit: {otsu['neg_threshold']:.2f}\n"
        report += f"    • Weak/Moderate Boundary: {otsu['otsu_1']:.2f}\n"
        report += f"    • Moderate/Strong Boundary: {otsu['otsu_2']:.2f}\n"
        report += f"\n  Grading Distribution:\n"
        report += f"    • Grade 0 (No Expr): {grade_counts.get(0, 0):6d} nuclei ({grade_counts.get(0, 0)/total_cells*100:5.1f}%)\n"
        report += f"    • Grade 1+ (Weak)  : {grade_counts.get(1, 0):6d} nuclei ({grade_counts.get(1, 0)/total_cells*100:5.1f}%)\n"
        report += f"    • Grade 2+ (Mod)   : {grade_counts.get(2, 0):6d} nuclei ({grade_counts.get(2, 0)/total_cells*100:5.1f}%)\n"
        report += f"    • Grade 3+ (Strong): {grade_counts.get(3, 0):6d} nuclei ({grade_counts.get(3, 0)/total_cells*100:5.1f}%)\n"
        report += f"\n  Diagnosis: {channel} {'Positive' if positive_rate >= 1 else 'Negative'} ({positive_rate:.1f}% Positive Expression)\n\n"
    
    # 综合诊断
    report += "-"*80 + "\n"
    report += "Preliminary Comprehensive Diagnosis:\n"
    report += "-"*80 + "\n\n"

    er_tested = 'ER_nuc_grade' in df.columns
    pr_tested = 'PR_nuc_grade' in df.columns
    her2_tested = 'HER2_nuc_grade' in df.columns
    ki67_tested = 'Ki67_nuc_grade' in df.columns
    
    er_positive = df['ER_nuc_grade'].gt(0).sum() if er_tested else 0
    pr_positive = df['PR_nuc_grade'].gt(0).sum() if pr_tested else 0
    her2_positive = df['HER2_nuc_grade'].gt(0).sum() if her2_tested else 0
    ki67_positive = df['Ki67_nuc_grade'].gt(0).sum() if ki67_tested else 0
    
    total = total_cells if total_cells > 0 else 1
    
    er_rate = er_positive / total * 100
    pr_rate = pr_positive / total * 100
    her2_rate = her2_positive / total * 100
    ki67_rate = ki67_positive / total * 100
    
    hr_positive = er_rate >= 1 or pr_rate >= 1
    her2_positive_status = her2_rate >= 10
    
    report += "□ Hormone Receptor (HR) Status: "
    if not er_tested and not pr_tested:
        report += "Not Tested\n"
    else:
        report += ("Positive (HR+)" if hr_positive else "Negative (HR-)") + "\n"
        
    report += "□ HER2 Status: "
    if not her2_tested:
        report += "Not Tested\n"
    else:
        report += ("Positive" if her2_positive_status else "Negative") + "\n"
        
    report += "□ Ki67 Proliferation: "
    if not ki67_tested:
        report += "Not Tested\n"
    else:
        report += f"{ki67_rate:.1f}%\n"
        if ki67_rate < 10:
            report += "  (Low Proliferation Activity)\n"
        elif ki67_rate < 30:
            report += "  (Moderate Proliferation Activity)\n"
        else:
            report += "  (High Proliferation Activity)\n"
            
    report += "\n□ Inferred Molecular Subtype: "
    if er_tested and pr_tested and her2_tested:
        if hr_positive:
            if her2_positive_status:
                report += "Luminal B (HER2-positive)\n"
            else:
                if ki67_tested and ki67_rate >= 20:
                    report += "Luminal B (HER2-negative)\n"
                elif ki67_tested:
                    report += "Luminal A\n"
                else:
                    report += "Luminal (A/B indeterminate, Ki67 missing)\n"
        else:
            if her2_positive_status:
                report += "HER2-enriched\n"
            else:
                report += "Triple Negative Breast Cancer (TNBC)\n"
    else:
        report += "Indeterminate (Missing core markers: ER, PR, or HER2)\n"
        
    report += "\n" + "="*80 + "\n"
    
    return report



def main():
    print("\n" + "="*70)
    print("  Generating independent clinical reports for each block")
    print("="*70)
    
    all_blocks = set()
    for d in ["TMAd", "TMAe"]:
        p1 = SEGMENTATION_DIR / d
        if p1.exists():
            for p in p1.iterdir():
                if p.is_dir(): all_blocks.add(p.name)
    pneg = BASE_DIR / "results" / "negative_controls" / "segmentation"
    if pneg.exists():
        for p in pneg.iterdir():
            if p.is_dir(): all_blocks.add(p.name)
    all_blocks.update(["A10", "B8", "G2", "J10"])
    all_blocks = sorted(all_blocks)
    
    for block_name in all_blocks:
        try:
            process_single_block(block_name)
        except Exception as e:
            print(f"❌ 处理 {block_name} errored out: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ All block reports have been generated!")
    print("="*70)
    print(f"\nOutput location: {CLINICAL_REPORT_DIR}")
    print("Output structure:")
    for block in all_blocks:
        dataset = infer_dataset_for_block(block)
        if dataset is None:
            continue
        print(f"  {dataset}/{block}/")
        print(f"    ├─ {block}_{dataset}_clinical_report.txt")
        print(f"    ├─ {block}_{dataset}_expression_statistics.csv")
        print(f"    └─ {block}_{dataset}_grading_distribution.png")


if __name__ == "__main__":
    main()
