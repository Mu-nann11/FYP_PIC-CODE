#!/usr/bin/env python3
"""
通用通道分级脚本 - 对所有blocks应用所有可用通道的阈值

改进点：
1. 移除分组限制（3ch_Neg vs Ki67_Neg）
2. 自动检测每个block的可用通道
3. 对所有有数据的通道应用相应的校准阈值
4. 生成包含所有通道的完整报告

分级策略（混合方法）：
  - 阳性细胞 >= 10: 使用 Otsu 自适应分界
  - 阳性细胞 < 10: 使用 StdDev 固定阈值
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")
SEGMENTATION_DIR = BASE_DIR / "results" / "segmentation"
CALIBRATION_DIR = BASE_DIR / "results" / "calibration"
CLINICAL_REPORT_DIR = BASE_DIR / "results" / "clinical_reports"
CLINICAL_REPORT_DIR.mkdir(exist_ok=True)

# 阈值文件
THRESHOLD_FILE = CALIBRATION_DIR / "thresholds_raw_nuclei.json"

# All blocks are dynamically discovered from TMAd / TMAe directories
# (see get_all_available_blocks() below)


def get_all_available_blocks():
    """Discover all available blocks by scanning TMAd and TMAe directories."""
    blocks = []
    for dataset in ["TMAd", "TMAe"]:
        dataset_dir = SEGMENTATION_DIR / dataset
        if not dataset_dir.exists():
            continue
        for block_dir in dataset_dir.iterdir():
            if block_dir.is_dir():
                features_file = block_dir / f"{block_dir.name}_{dataset}_features.csv"
                if features_file.exists():
                    blocks.append(block_dir.name)
    return sorted(set(blocks))


def infer_dataset_for_block(block_name):
    """Infer dataset by checking segmentation file locations."""
    for dataset in ["TMAd", "TMAe"]:
        candidate = SEGMENTATION_DIR / dataset / block_name / f"{block_name}_{dataset}_features.csv"
        if candidate.exists():
            return dataset
    return None


def load_thresholds():
    """加载校准后的阈值"""
    with open(THRESHOLD_FILE) as f:
        return json.load(f)


def calculate_otsu_threshold(intensities):
    """
    计算Otsu阈值
    使用直方图法自动选择分界线
    """
    from skimage import filters
    
    intensities = np.array(intensities)
    intensities_uint16 = (intensities / intensities.max() * 65535).astype(np.uint16)
    
    if len(np.unique(intensities_uint16)) < 2:
        return None
    
    try:
        threshold = filters.threshold_otsu(intensities_uint16)
        # 还原到原始尺度
        return threshold / 65535 * intensities.max()
    except:
        return None


def calculate_channel_otsu_thresholds(all_data_df, channel):
    """
    为特定通道计算两个Otsu分界线
    
    Args:
        all_data_df: 包含所有样本数据的DataFrame
        channel: 通道名称 (ER/PR/HER2/Ki67)
    
    Returns:
        {neg_threshold, otsu_1, otsu_2, n_positive, method} 或 None
    """
    thresholds_dict = load_thresholds()
    
    # 获取阴性阈值
    if channel in ["ER", "PR", "HER2"]:
        if "3ch_Neg" not in thresholds_dict or channel not in thresholds_dict["3ch_Neg"]:
            return None
        threshold_data = thresholds_dict["3ch_Neg"][channel]
    elif channel == "Ki67":
        if "Ki67_Neg" not in thresholds_dict or "Ki67" not in thresholds_dict["Ki67_Neg"]:
            return None
        threshold_data = thresholds_dict["Ki67_Neg"]["Ki67"]
    else:
        return None
    
    neg_threshold = threshold_data["threshold"]
    std_dev = threshold_data["std"]
    
    # 获取该通道的所有强度值
    col = f"{channel}_nuc_mean"
    if col not in all_data_df.columns:
        return None
    
    intensities = all_data_df[col].dropna().values
    if len(intensities) == 0:
        return None
    
    # 分为无表达和有表达
    positive_intensities = intensities[intensities >= neg_threshold]
    
    print(f"  {channel}:")
    print(f"    - 总核数: {len(intensities)}")
    print(f"    - 阳性核数: {len(positive_intensities)} ({len(positive_intensities)/len(intensities)*100:.1f}%)")
    
    # 根据阳性细胞数决定方法
    if len(positive_intensities) >= 10:
        # 使用Otsu方法
        method = "Otsu"
        otsu_1 = calculate_otsu_threshold(positive_intensities)
        
        if otsu_1 is None:
            print(f"    ⚠️ Otsu计算失败，降级为StdDev")
            method = "StdDev"
            otsu_1 = neg_threshold + std_dev
            otsu_2 = neg_threshold + 2 * std_dev
        else:
            # 第二层Otsu：对强阳性细胞
            strong_intensities = positive_intensities[positive_intensities >= otsu_1]
            if len(strong_intensities) >= 5:
                otsu_2 = calculate_otsu_threshold(strong_intensities)
                if otsu_2 is None:
                    otsu_2 = otsu_1 + (otsu_1 - neg_threshold) * 0.5
            else:
                otsu_2 = otsu_1 + (otsu_1 - neg_threshold) * 0.5
    else:
        # 细胞太少，使用StdDev方法
        method = "StdDev"
        otsu_1 = neg_threshold + std_dev
        otsu_2 = neg_threshold + 2 * std_dev
    
    print(f"    - 方法: {method}")
    print(f"    - 无表达上限: {neg_threshold:.2f}")
    print(f"    - 弱/中分界(Otsu_1): {otsu_1:.2f}")
    print(f"    - 中/强分界(Otsu_2): {otsu_2:.2f}\n")
    
    return {
        "neg_threshold": float(neg_threshold),
        "otsu_1": float(otsu_1),
        "otsu_2": float(otsu_2),
        "n_positive": len(positive_intensities),
        "method": method,
        "std_dev": float(std_dev),
    }


def classify_by_otsu(intensity, neg_threshold, otsu_1, otsu_2):
    """根据Otsu分界线对细胞进行分级"""
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


def detect_available_channels(block_df):
    """检测该block有哪些通道的数据"""
    available = {}
    
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        col = f"{channel}_nuc_mean"
        if col in block_df.columns and block_df[col].notna().sum() > 0:
            available[channel] = col
    
    return available


def apply_universal_grading(all_results_dict):
    """为所有blocks应用通用通道分级"""
    all_blocks = get_all_available_blocks()
    print("\n" + "="*80)
    print(f"🔄 第二步：为 {len(all_blocks)} 个blocks应用通道分级")
    print("="*80 + "\n")

    graded_blocks = {}

    for block_name in sorted(all_blocks):
        print(f"📊 处理Block: {block_name}")

        dataset = infer_dataset_for_block(block_name)
        if dataset is None:
            print(f"  ⚠️ 未找到分割结果（TMAd/TMAe）\n")
            continue

        features_file = SEGMENTATION_DIR / dataset / block_name / f"{block_name}_{dataset}_features.csv"
        if not features_file.exists():
            print(f"  ⚠️ 文件不存在\n")
            continue
        
        df = pd.read_csv(features_file)
        print(f"  ✓ 加载 {len(df)} 个细胞")
        
        # 检测可用通道
        available_channels = detect_available_channels(df)
        print(f"  ✓ 可用通道: {', '.join(available_channels.keys())}")
        
        # 对每个可用通道应用分级
        for channel, col in available_channels.items():
            if channel not in all_results_dict:
                print(f"  ⚠️ {channel}: 无校准阈值，跳过")
                continue
            
            thresholds = all_results_dict[channel]
            grade_col = f"{channel}_nuc_grade"
            
            df[grade_col] = df[col].apply(
                lambda x: classify_by_otsu(
                    x,
                    thresholds["neg_threshold"],
                    thresholds["otsu_1"],
                    thresholds["otsu_2"]
                )
            )
        
        # 保存分级后的CSV
        output_file = SEGMENTATION_DIR / dataset / block_name / f"{block_name}_{dataset}_features_graded_universal.csv"
        df.to_csv(output_file, index=False)
        graded_blocks[block_name] = {"dataset": dataset, "df": df}
        print(f"  ✓ 已保存: {output_file.name}\n")
    
    return graded_blocks


def generate_block_report(block_name, block_df, all_thresholds):
    """为单个block生成临床报告"""
    
    available_channels = detect_available_channels(block_df)
    if not available_channels:
        return None, None
    
    # ==================== 报告头部 ====================
    report_text = ""
    report_text += "="*80 + "\n"
    report_text += f"  样本诊断报告: Block {block_name}\n"
    report_text += "="*80 + "\n"
    report_text += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_text += f"分析方法: 混合通道分级 (Otsu + StdDev)\n"
    report_text += f"可用通道: {', '.join(available_channels.keys())}\n"
    report_text += f"总核数: {len(block_df)}\n\n"
    
    # ==================== 分级标准说明 ====================
    report_text += "-"*80 + "\n"
    report_text += "1. 分级标准\n"
    report_text += "-"*80 + "\n"
    report_text += "Grade 0 (0)     : 无表达  (强度 < 阴性阈值)\n"
    report_text += "Grade 1+ (1)    : 弱阳性  (阴性阈值 ≤ 强度 < Otsu_1)\n"
    report_text += "Grade 2+ (2)    : 中阳性  (Otsu_1 ≤ 强度 < Otsu_2)\n"
    report_text += "Grade 3+ (3)    : 强阳性  (强度 ≥ Otsu_2)\n\n"
    
    # ==================== 每个通道的分析 ====================
    report_text += "-"*80 + "\n"
    report_text += "2. 分通道表达分析\n"
    report_text += "-"*80 + "\n\n"
    
    summary_data = []
    
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        if channel not in available_channels:
            report_text += f"{channel}: 无数据\n"
            continue
        
        col = f"{channel}_nuc_mean"
        grade_col = f"{channel}_nuc_grade"
        
        if grade_col not in block_df.columns:
            report_text += f"{channel}: 未分级\n"
            continue
        
        thresholds = all_thresholds[channel]
        grade_counts = block_df[grade_col].value_counts().sort_index()
        total = len(block_df)
        positive_count = total - grade_counts.get(0, 0)
        positive_rate = positive_count / total * 100 if total > 0 else 0
        
        report_text += f"{channel} 表达:\n"
        report_text += f"  └─ 方法: {thresholds['method']} (n_positive={thresholds['n_positive']})\n"
        report_text += f"  ├─ Grade 0 (无)     : {grade_counts.get(0, 0):6d} 核 ({grade_counts.get(0, 0)/total*100:5.1f}%)\n"
        report_text += f"  ├─ Grade 1+ (弱)    : {grade_counts.get(1, 0):6d} 核 ({grade_counts.get(1, 0)/total*100:5.1f}%)\n"
        report_text += f"  ├─ Grade 2+ (中)    : {grade_counts.get(2, 0):6d} 核 ({grade_counts.get(2, 0)/total*100:5.1f}%)\n"
        report_text += f"  └─ Grade 3+ (强)    : {grade_counts.get(3, 0):6d} 核 ({grade_counts.get(3, 0)/total*100:5.1f}%)\n"
        report_text += f"  → 阳性率: {positive_rate:.1f}%\n\n"
        
        summary_data.append({
            "Channel": channel,
            "Method": thresholds["method"],
            "Grade_0": grade_counts.get(0, 0),
            "Grade_1": grade_counts.get(1, 0),
            "Grade_2": grade_counts.get(2, 0),
            "Grade_3": grade_counts.get(3, 0),
            "Positive_Rate_%": positive_rate,
            "Threshold_Neg": thresholds["neg_threshold"],
            "Threshold_Otsu_1": thresholds["otsu_1"],
            "Threshold_Otsu_2": thresholds["otsu_2"],
        })
    
    # ==================== 初步诊断 ====================
    report_text += "-"*80 + "\n"
    report_text += "3. 初步诊断\n"
    report_text += "-"*80 + "\n\n"
    
    # 激素受体
    if "ER" in available_channels:
        er_grade_col = "ER_nuc_grade"
        er_positive = len(block_df[block_df[er_grade_col] > 0]) if er_grade_col in block_df.columns else 0
        report_text += f"ER: {'阳性' if er_positive > 0 else '阴性'} ({er_positive/len(block_df)*100:.1f}%)\n"
    
    if "PR" in available_channels:
        pr_grade_col = "PR_nuc_grade"
        pr_positive = len(block_df[block_df[pr_grade_col] > 0]) if pr_grade_col in block_df.columns else 0
        report_text += f"PR: {'阳性' if pr_positive > 0 else '阴性'} ({pr_positive/len(block_df)*100:.1f}%)\n"
    
    if "HER2" in available_channels:
        her2_grade_col = "HER2_nuc_grade"
        her2_positive = len(block_df[block_df[her2_grade_col] > 0]) if her2_grade_col in block_df.columns else 0
        report_text += f"HER2: {'阳性' if her2_positive > 0 else '阴性'} ({her2_positive/len(block_df)*100:.1f}%)\n"
    
    if "Ki67" in available_channels:
        ki67_grade_col = "Ki67_nuc_grade"
        ki67_positive = len(block_df[block_df[ki67_grade_col] > 0]) if ki67_grade_col in block_df.columns else 0
        ki67_index = ki67_positive/len(block_df)*100 if len(block_df) > 0 else 0
        report_text += f"Ki67: {ki67_index:.1f}% (增殖指数)\n"
    
    report_text += "\n" + "="*80 + "\n"
    report_text += "报告结束\n"
    report_text += "="*80 + "\n"
    
    return report_text, summary_data


def save_block_report(block_name, dataset, report_text, summary_data):
    """保存block的报告"""
    block_report_dir = CLINICAL_REPORT_DIR / dataset / block_name
    block_report_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存文本报告
    report_file = block_report_dir / f"{block_name}_{dataset}_clinical_report_universal.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 保存CSV统计
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        csv_file = block_report_dir / f"{block_name}_{dataset}_expression_statistics_universal.csv"
        summary_df.to_csv(csv_file, index=False)
    
    return report_file


def generate_reports_for_all_blocks(graded_blocks, all_thresholds):
    """为所有blocks生成报告"""
    print("\n" + "="*80)
    print("📝 第三步：生成所有Blocks的临床报告")
    print("="*80 + "\n")
    
    for block_name in sorted(graded_blocks.keys()):
        print(f"📄 生成 {block_name} 的报告...")

        block_df = graded_blocks[block_name]["df"]
        dataset = graded_blocks[block_name]["dataset"]
        report_text, summary_data = generate_block_report(block_name, block_df, all_thresholds)
        
        if report_text is None:
            print(f"  ⚠️ 无法生成报告（无可用通道）\n")
            continue
        
        save_block_report(block_name, dataset, report_text, summary_data)
        print(f"  ✓ 已保存\n")


def main():
    print("\n" + "="*80)
    print("  通用通道分级系统 v2.0")
    print("  移除分组限制 | 对所有blocks应用所有通道")
    print("="*80)
    
    # Step 1: 加载所有数据并计算Otsu阈值
    all_blocks = get_all_available_blocks()
    print("\n" + "="*80)
    print(f"第一步：从 {len(all_blocks)} 个blocks收集数据并计算各通道Otsu分界线")
    print("="*80 + "\n")

    all_data_df = []
    for block_name in sorted(all_blocks):
        dataset = infer_dataset_for_block(block_name)
        if dataset is None:
            continue
        features_file = SEGMENTATION_DIR / dataset / block_name / f"{block_name}_{dataset}_features.csv"
        if features_file.exists():
            df = pd.read_csv(features_file)
            all_data_df.append(df)
    
    if not all_data_df:
        print("❌ 没有找到分割结果文件！")
        return
    
    all_data_df = pd.concat(all_data_df, ignore_index=True)
    print(f"✓ 已加载 {len(all_data_df)} 个细胞\n")
    
    # 计算每个通道的分界线
    all_thresholds = {}
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        print(f"计算 {channel} 的Otsu分界线:")
        thresholds = calculate_channel_otsu_thresholds(all_data_df, channel)
        if thresholds:
            all_thresholds[channel] = thresholds
    
    if not all_thresholds:
        print("❌ 没有成功计算任何通道的阈值！")
        return
    
    # Step 2: 应用分级
    graded_blocks = apply_universal_grading(all_thresholds)
    
    # Step 3: 生成报告
    generate_reports_for_all_blocks(graded_blocks, all_thresholds)
    
    # 总结
    print("\n" + "="*80)
    print("✅ 完成！")
    print("="*80)
    print(f"\n输出位置: {CLINICAL_REPORT_DIR}")
    print("\n每个block的文件:")
    for block_name in sorted(all_blocks):
        dataset = infer_dataset_for_block(block_name)
        if dataset is None:
            continue
        block_dir = CLINICAL_REPORT_DIR / dataset / block_name
        if block_dir.exists():
            print(f"\n{dataset}/{block_name}:")
            for file in sorted(block_dir.glob("*universal*")):
                print(f"  - {file.name}")


if __name__ == "__main__":
    main()
