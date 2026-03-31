"""
Figure 7: Feature Table Preview
Display key columns from the output CSV with cell ID, area, and marker intensities
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = Path(r"d:\Try_munan\FYP_LAST\results")
OUTPUT_DIR = RESULTS_DIR / "figures"

# Select a representative block with good data
SAMPLE_BLOCK = "H2"
CSV_PATH = RESULTS_DIR / "segmentation" / SAMPLE_BLOCK / f"{SAMPLE_BLOCK}_features_graded_universal.csv"


def create_feature_preview():
    """Create feature table preview figure"""
    print("Creating Figure 7: Feature Table Preview...")

    # Load data
    df = pd.read_csv(CSV_PATH)
    print(f"Total cells in {SAMPLE_BLOCK}: {len(df)}")

    # Select key columns for preview
    preview_columns = [
        'cell_label',
        'nuc_area',
        'cyto_area',
        'cell_area',
        'DAPI_nuc_mean',
        'ER_nuc_mean',
        'ER_nuc_grade',
        'PR_nuc_mean',
        'PR_nuc_grade',
        'HER2_nuc_mean',
        'HER2_nuc_grade',
        'Ki67_nuc_mean',
        'Ki67_nuc_grade'
    ]

    # Create subset with available columns
    available_cols = [col for col in preview_columns if col in df.columns]
    df_preview = df[available_cols].head(20).copy()

    # Rename columns for display
    rename_map = {
        'cell_label': 'Cell ID',
        'nuc_area': 'Nucleus\nArea',
        'cyto_area': 'Cyto\nArea',
        'cell_area': 'Cell\nArea',
        'DAPI_nuc_mean': 'DAPI\nMean',
        'ER_nuc_mean': 'ER\nMean',
        'ER_nuc_grade': 'ER\nGrade',
        'PR_nuc_mean': 'PR\nMean',
        'PR_nuc_grade': 'PR\nGrade',
        'HER2_nuc_mean': 'HER2\nMean',
        'HER2_nuc_grade': 'HER2\nGrade',
        'Ki67_nuc_mean': 'Ki67\nMean',
        'Ki67_nuc_grade': 'Ki67\nGrade'
    }
    df_preview.rename(columns=rename_map, inplace=True)

    # Format numeric columns
    for col in df_preview.columns:
        if 'Mean' in col or 'Area' in col:
            df_preview[col] = df_preview[col].apply(lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A')

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=df_preview.values,
        colLabels=df_preview.columns,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(df_preview.columns)
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # Header style
    for j, col in enumerate(df_preview.columns):
        cell = table[(0, j)]
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_facecolor('#4472C4')

    # Alternate row colors
    for i in range(len(df_preview)):
        for j in range(len(df_preview.columns)):
            cell = table[(i+1, j)]
            if i % 2 == 0:
                cell.set_facecolor('#D6DCE5')
            else:
                cell.set_facecolor('#FFFFFF')

            # Highlight grade columns
            if 'Grade' in df_preview.columns[j]:
                cell.set_facecolor('#FFF2CC')

    # Add title and info
    fig.suptitle('[Figure 7] Feature Table Preview', fontsize=16, fontweight='bold', y=0.95)

    # Add summary statistics
    total_cells = len(df)
    er_pos = (df['ER_nuc_grade'] >= 1).sum() if 'ER_nuc_grade' in df.columns else 0
    pr_pos = (df['PR_nuc_grade'] >= 1).sum() if 'PR_nuc_grade' in df.columns else 0
    her2_pos = (df['HER2_nuc_grade'] >= 1).sum() if 'HER2_nuc_grade' in df.columns else 0
    ki67_pos = (df['Ki67_nuc_grade'] >= 1).sum() if 'Ki67_nuc_grade' in df.columns else 0

    summary_text = (
        f"Block: {SAMPLE_BLOCK} | Total Cells: {total_cells:,} | "
        f"ER+: {er_pos:,} ({er_pos/total_cells*100:.1f}%) | "
        f"PR+: {pr_pos:,} ({pr_pos/total_cells*100:.1f}%) | "
        f"HER2+: {her2_pos:,} ({her2_pos/total_cells*100:.1f}%) | "
        f"Ki67+: {ki67_pos:,} ({ki67_pos/total_cells*100:.1f}%)"
    )

    fig.text(0.5, 0.88, summary_text, ha='center', fontsize=10, style='italic', color='gray')
    fig.text(0.5, 0.83, 'Showing first 20 rows. Yellow cells indicate grade columns (0=negative, 1=low, 2=high positive)',
             ha='center', fontsize=9, color='gray')

    # Add column descriptions
    col_descriptions = {
        'Cell ID': 'Unique cell identifier',
        'Nucleus\nArea': 'Nucleus pixel area',
        'Cyto\nArea': 'Cytoplasm pixel area',
        'Cell\nArea': 'Total cell pixel area',
        'DAPI\nMean': 'Nuclear DAPI intensity (mean)',
        'ER\nMean': 'Nuclear ER intensity (mean)',
        'ER\nGrade': 'ER positive grade (0/1/2)',
        'PR\nMean': 'Nuclear PR intensity (mean)',
        'PR\nGrade': 'PR positive grade (0/1/2)',
        'HER2\nMean': 'Nuclear HER2 intensity (mean)',
        'HER2\nGrade': 'HER2 positive grade (0/1/2)',
        'Ki67\nMean': 'Nuclear Ki67 intensity (mean)',
        'Ki67\nGrade': 'Ki67 positive grade (0/1/2)'
    }

    # Save outputs
    output_png = OUTPUT_DIR / "figure7_feature_table_preview.png"
    output_pdf = OUTPUT_DIR / "figure7_feature_table_preview.pdf"

    print(f"Saving to {output_png}...")
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"Saving to {output_pdf}...")
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')

    plt.close()

    return output_png, output_pdf


def create_summary_statistics():
    """Create summary statistics box"""
    print("\nCreating summary statistics...")

    # Load all blocks
    blocks = ["A1", "A8", "D1", "E10", "G1", "H10", "H2", "J10"]

    all_stats = []
    for block in blocks:
        csv_path = RESULTS_DIR / "segmentation" / block / f"{block}_features_graded_universal.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            total = len(df)
            er_pos = (df['ER_nuc_grade'] >= 1).sum() if 'ER_nuc_grade' in df.columns else 0
            pr_pos = (df['PR_nuc_grade'] >= 1).sum() if 'PR_nuc_grade' in df.columns else 0
            her2_pos = (df['HER2_nuc_grade'] >= 1).sum() if 'HER2_nuc_grade' in df.columns else 0
            ki67_pos = (df['Ki67_nuc_grade'] >= 1).sum() if 'Ki67_nuc_grade' in df.columns else 0

            all_stats.append({
                'Block': block,
                'Total': total,
                'ER+': er_pos,
                'ER%': er_pos/total*100 if total > 0 else 0,
                'PR+': pr_pos,
                'PR%': pr_pos/total*100 if total > 0 else 0,
                'HER2+': her2_pos,
                'HER2%': her2_pos/total*100 if total > 0 else 0,
                'Ki67+': ki67_pos,
                'Ki67%': ki67_pos/total*100 if total > 0 else 0
            })

    stats_df = pd.DataFrame(all_stats)

    # Create summary figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Summary table
    summary_data = []
    for _, row in stats_df.iterrows():
        summary_data.append([
            row['Block'],
            f"{row['Total']:,}",
            f"{row['ER+']:,} ({row['ER%']:.1f}%)",
            f"{row['PR+']:,} ({row['PR%']:.1f}%)",
            f"{row['HER2+']:,} ({row['HER2%']:.1f}%)",
            f"{row['Ki67+']:,} ({row['Ki67%']:.1f}%)"
        ])

    # Add total row
    total_cells = stats_df['Total'].sum()
    total_er = stats_df['ER+'].sum()
    total_pr = stats_df['PR+'].sum()
    total_her2 = stats_df['HER2+'].sum()
    total_ki67 = stats_df['Ki67+'].sum()

    summary_data.append([
        'TOTAL',
        f"{int(total_cells):,}",
        f"{int(total_er):,} ({total_er/total_cells*100:.1f}%)",
        f"{int(total_pr):,} ({total_pr/total_cells*100:.1f}%)",
        f"{int(total_her2):,} ({total_her2/total_cells*100:.1f}%)",
        f"{int(total_ki67):,} ({total_ki67/total_cells*100:.1f}%)"
    ])

    headers = ['Block', 'Total Cells', 'ER+ (%)', 'PR+ (%)', 'HER2+ (%)', 'Ki67+ (%)']

    table = ax.table(
        cellText=summary_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#2E75B6'] * len(headers)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)

    # Style
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_text_props(color='white', fontweight='bold')

    # Last row (total) highlighted
    for j in range(len(headers)):
        cell = table[(len(summary_data), j)]
        cell.set_facecolor('#FFC000')
        cell.set_text_props(fontweight='bold')

    fig.suptitle('[Figure 7b] Marker Expression Summary by Block', fontsize=14, fontweight='bold', y=0.95)
    fig.text(0.5, 0.88, 'Positive cells determined by Otsu thresholding (graded: 0=negative, 1=low, 2=high)',
             ha='center', fontsize=10, style='italic', color='gray')

    output_png = OUTPUT_DIR / "figure7b_marker_summary.png"
    output_pdf = OUTPUT_DIR / "figure7b_marker_summary.pdf"

    print(f"Saving summary to {output_png}...")
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')

    plt.close()

    return output_png, output_pdf


if __name__ == "__main__":
    print("=" * 60)
    print("Figure 7: Feature Table Preview")
    print("=" * 60)

    output_png, output_pdf = create_feature_preview()
    print(f"\nOutput files:")
    print(f"  PNG: {output_png}")
    print(f"  PDF: {output_pdf}")

    print("\n" + "=" * 60)
    print("Creating marker summary...")
    output_png2, output_pdf2 = create_summary_statistics()
    print(f"\nSummary output files:")
    print(f"  PNG: {output_png2}")
    print(f"  PDF: {output_pdf2}")
    print("=" * 60)
