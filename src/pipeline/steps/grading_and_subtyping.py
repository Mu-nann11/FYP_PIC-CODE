"""
Pipeline Step 4: Grading and Molecular Subtype Classification
应用已有阈值进行细胞分级，然后进行分子亚型分类

功能：
1. 使用已有的 thresholds.json 应用阈值
2. 计算 Otsu 分界线（自适应）
3. 对所有细胞进行分级（0/1+/2+/3+）
4. 进行分子亚型分类（St. Gallen consensus）
5. 生成临床报告

输入：
    - {block}_{dataset}_features.csv（分割后的特征）
  - thresholds.json（已有的校准阈值）
  
输出：
        - {block}_{dataset}_features_graded_universal.csv（带分级和亚型标签）
    - results/clinical_reports/{dataset}/{block}/{block}_{dataset}_clinical_report.txt（临床报告）
    - results/clinical_reports/{dataset}/{block}/{block}_{dataset}_expression_statistics.csv（表达统计）
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).resolve().parents[2]  # Code/pipeline -> Code
SEGMENTATION_DIR = BASE_DIR.parent / "results" / "segmentation"
CALIBRATION_DIR = BASE_DIR.parent / "results" / "calibration"
CLINICAL_REPORT_DIR = BASE_DIR.parent / "results" / "clinical_reports"
CLINICAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Threshold files - try both possible names
THRESHOLD_FILE_OPTIONS = [
    CALIBRATION_DIR / "thresholds.json",
    CALIBRATION_DIR / "thresholds_raw_nuclei.json",
]


def _resolve_feature_csv_path(block: str, dataset: str) -> Path:
    """Resolve the segmentation feature CSV, with legacy fallbacks."""
    candidates = [
        SEGMENTATION_DIR / dataset / block / f"{block}_{dataset}_features.csv",
        SEGMENTATION_DIR / dataset / block / f"{block}_features.csv",
        SEGMENTATION_DIR / block / f"{block}_{dataset}_features.csv",
        SEGMENTATION_DIR / block / f"{block}_features.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def _resolve_graded_csv_path(block: str, dataset: str) -> Path:
    """Resolve the graded output CSV, with legacy fallbacks."""
    candidates = [
        SEGMENTATION_DIR / dataset / block / f"{block}_{dataset}_features_graded_universal.csv",
        SEGMENTATION_DIR / dataset / block / f"{block}_features_graded_universal.csv",
        SEGMENTATION_DIR / block / f"{block}_{dataset}_features_graded_universal.csv",
        SEGMENTATION_DIR / block / f"{block}_features_graded_universal.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def _iter_feature_csv_paths():
    """Iterate all available segmentation feature CSVs once."""
    seen_paths = set()
    seen_block_names = set()

    for dataset in ("TMAd", "TMAe"):
        dataset_dir = SEGMENTATION_DIR / dataset
        if not dataset_dir.exists():
            continue

        for block_dir in dataset_dir.iterdir():
            if not block_dir.is_dir():
                continue

            block_name = block_dir.name
            candidates = [
                block_dir / f"{block_name}_{dataset}_features.csv",
                block_dir / f"{block_name}_features.csv",
            ]

            for path in candidates:
                if path.exists():
                    if path not in seen_paths:
                        seen_paths.add(path)
                        seen_block_names.add(f"{dataset}/{block_name}")
                        yield path
                    break

    for block_dir in SEGMENTATION_DIR.iterdir():
        if not block_dir.is_dir() or block_dir.name in ("TMAd", "TMAe"):
            continue

        if block_dir.name in {item.split("/", 1)[1] for item in seen_block_names if "/" in item}:
            continue

        candidates = [
            block_dir / f"{block_dir.name}_features.csv",
            block_dir / f"{block_dir.name}_TMAd_features.csv",
            block_dir / f"{block_dir.name}_TMAe_features.csv",
        ]

        for path in candidates:
            if path.exists() and path not in seen_paths:
                seen_paths.add(path)
                yield path
                break


def load_thresholds():
    """Load calibration thresholds from JSON file"""
    for threshold_file in THRESHOLD_FILE_OPTIONS:
        if threshold_file.exists():
            try:
                with open(threshold_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {threshold_file}: {e}")
    
    raise FileNotFoundError(f"No threshold file found in {THRESHOLD_FILE_OPTIONS}")


def calculate_otsu_threshold(intensities):
    """
    Calculate Otsu threshold using histogram method
    
    Args:
        intensities: array of intensity values
    
    Returns:
        float: Otsu threshold value
    """
    try:
        from skimage import filters
    except ImportError:
        print("Warning: skimage not available, using percentile fallback")
        return np.percentile(intensities, 75)
    
    intensities = np.array(intensities)
    if len(intensities) == 0:
        return None
    
    # Normalize to 0-65535 for Otsu calculation
    intensities_uint16 = (intensities / intensities.max() * 65535).astype(np.uint16)
    
    if len(np.unique(intensities_uint16)) < 2:
        return None
    
    try:
        threshold = filters.threshold_otsu(intensities_uint16)
        # Convert back to original scale
        return float(threshold / 65535 * intensities.max())
    except:
        # Fallback to percentile
        return float(np.percentile(intensities, 75))


def calculate_channel_otsu_thresholds(all_data_df, channel):
    """
    Calculate Otsu thresholds for a specific channel
    
    Args:
        all_data_df: DataFrame with all cells from all blocks
        channel: Channel name (ER/PR/HER2/Ki67)
    
    Returns:
        dict with neg_threshold, otsu_1, otsu_2, and metadata
    """
    thresholds_dict = load_thresholds()
    
    # Get negative threshold, handle both JSON formats
    if channel in ["ER", "PR", "HER2"]:
        # Try new format first
        if "channels" in thresholds_dict:
            if channel in thresholds_dict["channels"]:
                if "3ch_Neg" in thresholds_dict["channels"][channel]:
                    threshold_data = thresholds_dict["channels"][channel]["3ch_Neg"]
                    neg_threshold = threshold_data.get("threshold")
                    std_dev = threshold_data.get("std")
                else:
                    return None
        else:
            # Try old format
            if "3ch_Neg" in thresholds_dict and channel in thresholds_dict["3ch_Neg"]:
                threshold_data = thresholds_dict["3ch_Neg"][channel]
                neg_threshold = threshold_data.get("threshold")
                std_dev = threshold_data.get("std")
            else:
                return None
    elif channel == "Ki67":
        # Try new format first
        if "channels" in thresholds_dict and "KI67" in thresholds_dict["channels"]:
            if "Ki67_Neg" in thresholds_dict["channels"]["KI67"]:
                threshold_data = thresholds_dict["channels"]["KI67"]["Ki67_Neg"]
                neg_threshold = threshold_data.get("threshold")
                std_dev = threshold_data.get("std")
            else:
                return None
        else:
            # Try old format
            if "Ki67_Neg" in thresholds_dict and "Ki67" in thresholds_dict["Ki67_Neg"]:
                threshold_data = thresholds_dict["Ki67_Neg"]["Ki67"]
                neg_threshold = threshold_data.get("threshold")
                std_dev = threshold_data.get("std")
            else:
                return None
    else:
        return None
    
    if neg_threshold is None:
        return None
    
    # Get intensity values for this channel
    col = f"{channel}_nuc_mean"
    if col not in all_data_df.columns:
        return None
    
    intensities = all_data_df[col].dropna().values
    if len(intensities) == 0:
        return None
    
    # Separate negative and positive cells
    positive_intensities = intensities[intensities >= neg_threshold]
    
    # Decide method based on number of positive cells
    if len(positive_intensities) >= 10:
        # Use Otsu method
        method = "Otsu"
        otsu_1 = calculate_otsu_threshold(positive_intensities)
        
        if otsu_1 is None:
            # Fallback to StdDev
            method = "StdDev"
            otsu_1 = neg_threshold + std_dev
            otsu_2 = neg_threshold + 2 * std_dev
        else:
            # Calculate second Otsu threshold for strong positive cells
            strong_intensities = positive_intensities[positive_intensities >= otsu_1]
            if len(strong_intensities) >= 5:
                otsu_2 = calculate_otsu_threshold(strong_intensities)
                if otsu_2 is None or otsu_2 <= otsu_1:
                    otsu_2 = otsu_1 + (otsu_1 - neg_threshold) * 0.5
            else:
                otsu_2 = otsu_1 + (otsu_1 - neg_threshold) * 0.5
    else:
        # Not enough positive cells, use StdDev method
        method = "StdDev"
        otsu_1 = neg_threshold + std_dev
        otsu_2 = neg_threshold + 2 * std_dev
    
    return {
        "neg_threshold": float(neg_threshold),
        "otsu_1": float(otsu_1),
        "otsu_2": float(otsu_2),
        "n_positive": len(positive_intensities),
        "method": method,
        "std_dev": float(std_dev),
    }


def classify_by_otsu(intensity, neg_threshold, otsu_1, otsu_2):
    """Classify cell grade based on Otsu thresholds"""
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


def detect_available_channels(df):
    """Detect which channels have data in this block"""
    available = {}
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        col = f"{channel}_nuc_mean"
        if col in df.columns and df[col].notna().sum() > 0:
            available[channel] = col
    return available


def classify_subtype(row):
    """
    Classify cell into molecular subtype based on St. Gallen consensus
    
    Uses grade columns where 0=negative, 1=low positive, 2+ = high positive
    """
    er_grade = row.get('ER_nuc_grade', 0)
    pr_grade = row.get('PR_nuc_grade', 0)
    her2_grade = row.get('HER2_nuc_grade', 0)
    ki67_grade = row.get('Ki67_nuc_grade', 0)
    
    # Handle NaN values
    er_grade = 0 if pd.isna(er_grade) else er_grade
    pr_grade = 0 if pd.isna(pr_grade) else pr_grade
    her2_grade = 0 if pd.isna(her2_grade) else her2_grade
    ki67_grade = 0 if pd.isna(ki67_grade) else ki67_grade
    
    # Positive if grade >= 1
    er_positive = er_grade >= 1
    pr_positive = pr_grade >= 1
    her2_positive = her2_grade >= 1
    ki67_high = ki67_grade >= 1  # Ki67 high if positive
    
    hormone_positive = er_positive or pr_positive
    
    # Classification logic (St. Gallen consensus)
    if her2_positive and not hormone_positive:
        return 'HER2 Enriched'
    elif not er_positive and not pr_positive and not her2_positive:
        return 'Triple Negative'
    elif hormone_positive and not her2_positive and not ki67_high:
        return 'Luminal A'
    elif hormone_positive and not her2_positive and ki67_high:
        return 'Luminal B'
    elif hormone_positive and her2_positive:
        if ki67_high:
            return 'Luminal B (HER2+)'
        else:
            return 'Luminal A (HER2+)'
    else:
        return 'Unknown'


def run_grading_and_subtyping(block: str, dataset: str, force: bool = False) -> dict:
    """
    Main function for Step 4: Apply grading and molecular subtype classification
    
    Args:
        block: Block name (e.g., "A1", "D5")
        dataset: Dataset name (e.g., "TMAd", "TMAe")
        force: Whether to force re-execution
    
    Returns:
        dict with status, error (if any), and metadata
    """
    try:
        logger = setup_logging("grading_and_subtyping")
        logger.info(f"Step 4: Grading and Molecular Subtype Classification for {block}")
        features_file = _resolve_feature_csv_path(block, dataset)
        graded_file = _resolve_graded_csv_path(block, dataset)

        if graded_file.exists() and not force:
            logger.info("Graded file already exists, using cached version")
            df = pd.read_csv(graded_file)
            logger.info(f"Loaded {len(df)} cells from cached graded file")
            all_thresholds = {}
        else:
            if not features_file.exists():
                return {
                    "status": "error",
                    "error": f"Segmentation file not found: {features_file}",
                }

            # Load features
            logger.info(f"Loading features from {features_file}")
            df = pd.read_csv(features_file)
            logger.info(f"Loaded {len(df)} cells")

            # Load all data to compute Otsu thresholds
            logger.info("Loading all block data to compute Otsu thresholds...")
            all_data_df = []

            for block_features in _iter_feature_csv_paths():
                try:
                    all_data_df.append(pd.read_csv(block_features))
                except Exception as exc:
                    logger.warning(f"Skipping unreadable feature file {block_features}: {exc}")

            if not all_data_df:
                all_data_df = [df]

            all_data_df = pd.concat(all_data_df, ignore_index=True)
            logger.info(f"Loaded {len(all_data_df)} total cells for threshold calculation")

            # Calculate Otsu thresholds
            logger.info("Computing Otsu thresholds for each channel...")
            all_thresholds = {}
            for channel in ["ER", "PR", "HER2", "Ki67"]:
                thresholds = calculate_channel_otsu_thresholds(all_data_df, channel)
                if thresholds:
                    all_thresholds[channel] = thresholds
                    logger.info(
                        f"  {channel}: neg={thresholds['neg_threshold']:.2f}, "
                        f"otsu_1={thresholds['otsu_1']:.2f}, "
                        f"otsu_2={thresholds['otsu_2']:.2f} ({thresholds['method']})"
                    )

            if not all_thresholds:
                return {
                    "status": "error",
                    "error": "Could not compute any channel thresholds",
                }

            # Apply grading
            logger.info("Applying grades to cells...")
            available_channels = detect_available_channels(df)

            for channel in available_channels.keys():
                if channel not in all_thresholds:
                    logger.warning(f"No threshold for {channel}, skipping")
                    continue

                thresholds = all_thresholds[channel]
                col = f"{channel}_nuc_mean"
                grade_col = f"{channel}_nuc_grade"

                df[grade_col] = df[col].apply(
                    lambda x: classify_by_otsu(
                        x,
                        thresholds["neg_threshold"],
                        thresholds["otsu_1"],
                        thresholds["otsu_2"]
                    )
                )

            # Save graded features
            df.to_csv(graded_file, index=False)
            logger.info(f"Saved graded features to {graded_file}")
        
        # Molecular subtype classification
        logger.info("Classifying molecular subtypes...")
        if all(col in df.columns for col in ['ER_nuc_grade', 'PR_nuc_grade', 'HER2_nuc_grade', 'Ki67_nuc_grade']):
            df['subtype'] = df.apply(classify_subtype, axis=1)
            
            # Calculate subtype distribution
            subtype_counts = df['subtype'].value_counts()
            logger.info("Molecular subtype distribution:")
            for subtype, count in subtype_counts.items():
                pct = count / len(df) * 100
                logger.info(f"  {subtype}: {count} cells ({pct:.1f}%)")
        
        # Save the final graded file with subtype
        output_features = graded_file
        df.to_csv(output_features, index=False)
        logger.info(f"Saved final features with subtype to {output_features}")
        
        # Generate clinical report
        logger.info("Generating clinical report...")
        report_text = generate_clinical_report(block, df, all_thresholds if 'all_thresholds' in locals() else {})
        
        block_report_dir = CLINICAL_REPORT_DIR / dataset / block
        block_report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = block_report_dir / f"{block}_{dataset}_clinical_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"Saved clinical report to {report_file}")
        
        # Generate statistics CSV
        stats_df = calculate_expression_statistics(block, df)
        stats_file = block_report_dir / f"{block}_{dataset}_expression_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        logger.info(f"Saved expression statistics to {stats_file}")
        
        return {
            "status": "success",
            "output_features": str(output_features),
            "cell_count": len(df),
            "channels": list(detect_available_channels(df).keys()),
        }
    
    except Exception as e:
        logger = setup_logging("grading_and_subtyping")
        logger.error(f"Error during grading and subtyping: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
        }


def generate_clinical_report(block_name, df, all_thresholds) -> str:
    """Generate a clinical report for a block"""
    available_channels = detect_available_channels(df)
    
    report = ""
    report += "="*80 + "\n"
    report += f"Clinical Report: Block {block_name}\n"
    report += "="*80 + "\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"Total cells: {len(df)}\n"
    report += f"Available channels: {', '.join(available_channels.keys())}\n\n"
    
    # Expression analysis by channel
    report += "-"*80 + "\n"
    report += "Expression Analysis\n"
    report += "-"*80 + "\n\n"
    
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        grade_col = f"{channel}_nuc_grade"
        
        if grade_col not in df.columns:
            report += f"{channel}: No data\n"
            continue
        
        grade_counts = df[grade_col].value_counts().sort_index()
        total = len(df)
        positive_count = total - grade_counts.get(0, 0)
        positive_rate = positive_count / total * 100 if total > 0 else 0
        
        report += f"{channel} Expression:\n"
        report += f"  Grade 0 (negative):   {grade_counts.get(0, 0):6d} cells ({grade_counts.get(0, 0)/total*100:5.1f}%)\n"
        report += f"  Grade 1+ (weak):      {grade_counts.get(1, 0):6d} cells ({grade_counts.get(1, 0)/total*100:5.1f}%)\n"
        report += f"  Grade 2+ (moderate):  {grade_counts.get(2, 0):6d} cells ({grade_counts.get(2, 0)/total*100:5.1f}%)\n"
        report += f"  Grade 3+ (strong):    {grade_counts.get(3, 0):6d} cells ({grade_counts.get(3, 0)/total*100:5.1f}%)\n"
        report += f"  Positive rate: {positive_rate:.1f}%\n\n"
    
    # Molecular subtype
    if 'subtype' in df.columns:
        report += "-"*80 + "\n"
        report += "Molecular Subtype Classification (St. Gallen Consensus)\n"
        report += "-"*80 + "\n\n"
        
        subtype_counts = df['subtype'].value_counts()
        for subtype, count in subtype_counts.items():
            pct = count / len(df) * 100
            report += f"  {subtype}: {count} cells ({pct:.1f}%)\n"
        
        report += "\n"
    
    report += "="*80 + "\n"
    report += "End of Report\n"
    report += "="*80 + "\n"
    
    return report


def calculate_expression_statistics(block_name, df) -> pd.DataFrame:
    """Calculate per-channel expression statistics"""
    stats = []
    
    for channel in ["ER", "PR", "HER2", "Ki67"]:
        grade_col = f"{channel}_nuc_grade"
        
        if grade_col not in df.columns:
            continue
        
        grade_counts = df[grade_col].value_counts().sort_index()
        total = len(df)
        
        stats.append({
            "Channel": channel,
            "Grade_0": grade_counts.get(0, 0),
            "Grade_1": grade_counts.get(1, 0),
            "Grade_2": grade_counts.get(2, 0),
            "Grade_3": grade_counts.get(3, 0),
            "Positive_Rate_%": (total - grade_counts.get(0, 0)) / total * 100,
        })
    
    return pd.DataFrame(stats)


def check_grading_done(block: str, dataset: str) -> bool:
    """Check if grading is already done for a block"""
    return _resolve_graded_csv_path(block, dataset).exists()


def setup_logging(module_name: str):
    """Setup logging for this module"""
    import logging
    logger = logging.getLogger(module_name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(name)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
