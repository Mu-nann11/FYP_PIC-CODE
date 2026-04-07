"""
Step 6: Summary Visualization
生成跨块汇总可视化：分子亚型分布柱状图、表达热力图等
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def classify_subtype(row):
    """Classify cell into molecular subtype based on marker grades"""
    er_grade = row.get('ER_nuc_grade', 0)
    pr_grade = row.get('PR_nuc_grade', 0)
    her2_grade = row.get('HER2_nuc_grade', 0)
    ki67_grade = row.get('Ki67_nuc_grade', 0)
    
    # Handle NaN
    er_grade = 0 if pd.isna(er_grade) else int(er_grade)
    pr_grade = 0 if pd.isna(pr_grade) else int(pr_grade)
    her2_grade = 0 if pd.isna(her2_grade) else int(her2_grade)
    ki67_grade = 0 if pd.isna(ki67_grade) else int(ki67_grade)
    
    # Positive if grade >= 1
    er_positive = er_grade >= 1
    pr_positive = pr_grade >= 1
    her2_positive = her2_grade >= 1
    ki67_high = ki67_grade >= 1
    
    hormone_positive = er_positive or pr_positive
    
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


def _resolve_graded_features_path(block, base_dir):
    for dataset in ["TMAd", "TMAe"]:
        candidate = (
            base_dir
            / "results"
            / "segmentation"
            / dataset
            / block
            / f"{block}_{dataset}_features_graded_universal.csv"
        )
        if candidate.exists():
            return candidate
    legacy = base_dir / "results" / "segmentation" / block / f"{block}_features_graded_universal.csv"
    return legacy


def load_graded_features(block, base_dir):
    """Load graded features for a block"""
    csv_path = _resolve_graded_features_path(block, base_dir)
    
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    df['block'] = block
    return df


def create_subtype_barplot(blocks, base_dir, output_path):
    """Create molecular subtype distribution barplot"""
    
    all_data = []
    
    for block in blocks:
        df = load_graded_features(block, base_dir)
        if df is None:
            continue
        
        # Classify cells
        df['subtype'] = df.apply(classify_subtype, axis=1)
        
        # Count subtypes
        subtype_counts = df['subtype'].value_counts()
        total_cells = len(df)
        
        for subtype, count in subtype_counts.items():
            pct = 100 * count / total_cells
            all_data.append({
                'Block': block,
                'Subtype': subtype,
                'Count': count,
                'Percentage': pct
            })
    
    if not all_data:
        print("[VIS] No data for subtype barplot")
        return False
    
    df_stats = pd.DataFrame(all_data)
    
    # Create barplot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    subtypes = df_stats['Subtype'].unique()
    blocks_list = df_stats['Block'].unique()
    x = np.arange(len(blocks_list))
    width = 0.15
    
    colors = {
        'Luminal A': '#1f77b4',
        'Luminal B': '#ff7f0e',
        'Luminal A (HER2+)': '#2ca02c',
        'Luminal B (HER2+)': '#d62728',
        'HER2 Enriched': '#9467bd',
        'Triple Negative': '#8c564b',
    }
    
    for i, subtype in enumerate(sorted(subtypes)):
        data = df_stats[df_stats['Subtype'] == subtype]
        values = []
        for block in blocks_list:
            block_data = data[data['Block'] == block]
            if len(block_data) > 0:
                values.append(block_data['Percentage'].values[0])
            else:
                values.append(0)
        
        ax.bar(x + i * width, values, width, label=subtype, 
               color=colors.get(subtype, '#999999'), alpha=0.8)
    
    ax.set_xlabel('Block', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cell Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Molecular Subtype Distribution Across Blocks', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(blocks_list, rotation=0)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[VIS] Saved subtype distribution to {output_path.name}")
    return True


def create_marker_expression_heatmap(blocks, base_dir, output_path):
    """Create heatmap of marker expression by block"""
    
    block_stats = []
    
    for block in blocks:
        df = load_graded_features(block, base_dir)
        if df is None:
            continue
        
        # Calculate mean expression (using grades: 0=negative, 1=low, 2=high)
        er_mean = df['ER_nuc_grade'].mean()
        pr_mean = df['PR_nuc_grade'].mean()
        her2_mean = df['HER2_nuc_grade'].mean()
        ki67_mean = df['Ki67_nuc_grade'].mean()
        
        # Calculate positive percentage (grade >= 1)
        er_pos_pct = 100 * (df['ER_nuc_grade'] >= 1).sum() / len(df)
        pr_pos_pct = 100 * (df['PR_nuc_grade'] >= 1).sum() / len(df)
        her2_pos_pct = 100 * (df['HER2_nuc_grade'] >= 1).sum() / len(df)
        ki67_pos_pct = 100 * (df['Ki67_nuc_grade'] >= 1).sum() / len(df)
        
        block_stats.append({
            'Block': block,
            'ER+ %': er_pos_pct,
            'PR+ %': pr_pos_pct,
            'HER2+ %': her2_pos_pct,
            'Ki67+ %': ki67_pos_pct,
        })
    
    if not block_stats:
        print("[VIS] No data for expression heatmap")
        return False
    
    df_heatmap = pd.DataFrame(block_stats).set_index('Block')
    
    fig, ax = plt.subplots(figsize=(10, len(blocks) * 0.4 + 2))
    
    sns.heatmap(df_heatmap, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Positive %'},
                ax=ax, vmin=0, vmax=100, linewidths=0.5)
    
    ax.set_title('Marker Expression by Block', fontsize=14, fontweight='bold')
    ax.set_xlabel('Marker', fontsize=12, fontweight='bold')
    ax.set_ylabel('Block', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[VIS] Saved expression heatmap to {output_path.name}")
    return True


def create_summary_statistics_table(blocks, base_dir, output_path):
    """Create summary statistics table CSV"""
    
    all_stats = []
    
    for block in blocks:
        df = load_graded_features(block, base_dir)
        if df is None:
            continue
        
        df['subtype'] = df.apply(classify_subtype, axis=1)
        
        stats = {
            'Block': block,
            'Total Cells': len(df),
            'ER+ Cells': int((df['ER_nuc_grade'] >= 1).sum()),
            'PR+ Cells': int((df['PR_nuc_grade'] >= 1).sum()),
            'HER2+ Cells': int((df['HER2_nuc_grade'] >= 1).sum()),
            'Ki67+ Cells': int((df['Ki67_nuc_grade'] >= 1).sum()),
            'Luminal A %': 100 * (df['subtype'] == 'Luminal A').sum() / len(df),
            'Luminal B %': 100 * (df['subtype'] == 'Luminal B').sum() / len(df),
            'HER2+ Luminal A %': 100 * (df['subtype'] == 'Luminal A (HER2+)').sum() / len(df),
            'HER2+ Luminal B %': 100 * (df['subtype'] == 'Luminal B (HER2+)').sum() / len(df),
            'HER2 Enriched %': 100 * (df['subtype'] == 'HER2 Enriched').sum() / len(df),
            'Triple Negative %': 100 * (df['subtype'] == 'Triple Negative').sum() / len(df),
        }
        all_stats.append(stats)
    
    if not all_stats:
        print("[VIS] No data for summary statistics")
        return False
    
    df_summary = pd.DataFrame(all_stats)
    df_summary.to_csv(output_path, index=False)
    
    print(f"[VIS] Saved summary statistics to {output_path.name}")
    return True


def run_visualization_summary(blocks, base_dir=None, force=False):
    """Run summary visualization for all blocks"""
    
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
    else:
        base_dir = Path(base_dir)
    
    output_dir = base_dir / "results" / "pipeline_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[VIS] Creating summary visualizations for {len(blocks)} blocks...")
    
    # Create visualizations
    subtype_plot = output_dir / "subtype_distribution.png"
    expression_heatmap = output_dir / "marker_expression_heatmap.png"
    summary_stats = output_dir / "summary_statistics.csv"
    
    success = True
    
    if force or not subtype_plot.exists():
        success &= create_subtype_barplot(blocks, base_dir, subtype_plot)
    else:
        print(f"[VIS] Subtype distribution already exists")
    
    if force or not expression_heatmap.exists():
        success &= create_marker_expression_heatmap(blocks, base_dir, expression_heatmap)
    else:
        print(f"[VIS] Expression heatmap already exists")
    
    if force or not summary_stats.exists():
        success &= create_summary_statistics_table(blocks, base_dir, summary_stats)
    else:
        print(f"[VIS] Summary statistics already exists")
    
    if success:
        print(f"\n[VIS] Summary visualization complete!")
        print(f"[VIS] Output saved to: {output_dir}")
    
    return success


if __name__ == "__main__":
    # Quick test
    blocks = ["D5", "E9"]
    run_visualization_summary(blocks, force=True)
