"""
Figure 6: Molecular Subtype Distribution by Block
Bar chart showing cell percentages for each subtype across blocks
Based on St. Gallen consensus classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = Path(r"d:\Try_munan\FYP_LAST\results")
OUTPUT_DIR = RESULTS_DIR / "figures"
GRADING_DIR = RESULTS_DIR / "grading"


def get_all_available_blocks():
    """Dynamically discover all blocks from TMAd and TMAe segmentation directories."""
    blocks = []
    for dataset in ["TMAd", "TMAe"]:
        dataset_dir = RESULTS_DIR / "segmentation" / dataset
        if not dataset_dir.exists():
            continue
        for block_dir in dataset_dir.iterdir():
            if block_dir.is_dir():
                blocks.append(block_dir.name)
    return sorted(set(blocks))


# Blocks to process
BLOCKS = get_all_available_blocks()

# Subtype definitions (St. Gallen consensus)
# Using grade columns: 0=negative, 1=low positive, 2=high positive
SUBTYPES = {
    'Luminal A': 'ER+ or PR+, HER2-, Ki67 low',
    'Luminal B': 'ER+ or PR+, HER2-, Ki67 high',
    'HER2 Enriched': 'HER2+, ER- & PR-',
    'Triple Negative': 'ER-, PR-, HER2-'
}


def classify_subtype(row):
    """
    Classify cell into molecular subtype based on marker expression.
    Using grade columns where 0=negative, 1=low, 2=high
    """
    er_grade = row.get('ER_nuc_grade', 0)
    pr_grade = row.get('PR_nuc_grade', 0)
    her2_grade = row.get('HER2_nuc_grade', 0)
    ki67_grade = row.get('Ki67_nuc_grade', 0)

    # Handle potential missing or NaN values
    er_grade = 0 if pd.isna(er_grade) else er_grade
    pr_grade = 0 if pd.isna(pr_grade) else pr_grade
    her2_grade = 0 if pd.isna(her2_grade) else her2_grade
    ki67_grade = 0 if pd.isna(ki67_grade) else ki67_grade

    # Positive if grade >= 1 (low or high positive)
    er_positive = er_grade >= 1
    pr_positive = pr_grade >= 1
    her2_positive = her2_grade >= 1
    ki67_high = ki67_grade >= 1  # Ki67 high if positive

    # Classification logic
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
        # HER2+ type if HER2 is positive regardless of hormone status
        if ki67_high:
            return 'Luminal B (HER2+)'
        else:
            return 'Luminal A (HER2+)'
    else:
        return 'Unknown'


def load_and_classify(block):
    """Load features for a block and classify cells"""
    csv_path = None
    for dataset in ["TMAd", "TMAe"]:
        candidate = RESULTS_DIR / "segmentation" / dataset / block / f"{block}_{dataset}_features_graded_universal.csv"
        if candidate.exists():
            csv_path = candidate
            break
    if csv_path is None:
        csv_path = RESULTS_DIR / "segmentation" / block / f"{block}_features_graded_universal.csv"

    if not csv_path.exists():
        print(f"  Warning: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)

    # Classify each cell
    df['subtype'] = df.apply(classify_subtype, axis=1)

    return df


def create_subtype_statistics(blocks):
    """Calculate subtype statistics for all blocks"""
    all_stats = []

    for block in blocks:
        print(f"Processing {block}...")
        df = load_and_classify(block)

        if df is None:
            continue

        total = len(df)

        # Count subtypes
        subtype_counts = df['subtype'].value_counts()

        stats = {
            'Block': block,
            'Total Cells': total
        }

        # Calculate percentages for main subtypes
        main_subtypes = ['Luminal A', 'Luminal B', 'HER2 Enriched', 'Triple Negative',
                         'Luminal A (HER2+)', 'Luminal B (HER2+)']

        for st in main_subtypes:
            count = subtype_counts.get(st, 0)
            stats[f'{st}_count'] = count
            stats[f'{st}_pct'] = (count / total * 100) if total > 0 else 0

        # Also track unknown
        unknown_count = subtype_counts.get('Unknown', 0)
        stats['Unknown_count'] = unknown_count
        stats['Unknown_pct'] = (unknown_count / total * 100) if total > 0 else 0

        all_stats.append(stats)

        print(f"  Total cells: {total}")
        print(f"  Subtypes: {dict(subtype_counts)}")

    return pd.DataFrame(all_stats)


def create_figure6(stats_df):
    """Create Figure 6: Molecular Subtype Distribution Bar Chart"""
    print("\nCreating Figure 6...")

    # Define colors for subtypes
    colors = {
        'Luminal A': '#3498DB',          # Blue
        'Luminal B': '#2980B9',          # Darker blue
        'HER2 Enriched': '#E74C3C',       # Red
        'Triple Negative': '#95A5A6',    # Gray
        'Luminal A (HER2+)': '#9B59B6',  # Purple
        'Luminal B (HER2+)': '#8E44AD',   # Dark purple
        'Unknown': '#BDC3C7'              # Light gray
    }

    # Subtypes to plot (main 4 + HER2+ variants)
    subtypes_to_plot = ['Luminal A', 'Luminal B', 'HER2 Enriched', 'Triple Negative']

    blocks = stats_df['Block'].tolist()
    n_blocks = len(blocks)

    # Prepare data for stacked bar chart
    x = np.arange(n_blocks)
    bar_width = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Stacked bar chart
    ax1 = axes[0]
    bottom = np.zeros(n_blocks)

    for subtype in subtypes_to_plot:
        values = stats_df[f'{subtype}_pct'].values
        ax1.bar(x, values, bar_width, bottom=bottom, label=subtype,
                color=colors[subtype], edgecolor='white', linewidth=0.5)
        bottom += values

    ax1.set_xlabel('Block', fontsize=14)
    ax1.set_ylabel('Cell Percentage (%)', fontsize=14)
    ax1.set_title('[Figure 6] Molecular Subtype Distribution by Block', fontsize=15, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(blocks, fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)

    # Right: Grouped bar chart for detailed comparison
    ax2 = axes[1]

    # All subtypes including HER2+ variants
    all_subtypes = ['Luminal A', 'Luminal B', 'HER2 Enriched', 'Triple Negative',
                    'Luminal A (HER2+)', 'Luminal B (HER2+)']

    bar_width2 = 0.12
    x2 = np.arange(len(blocks))

    for i, subtype in enumerate(all_subtypes):
        values = stats_df[f'{subtype}_pct'].values
        offset = (i - len(all_subtypes)/2 + 0.5) * bar_width2
        ax2.bar(x2 + offset, values, bar_width2, label=subtype,
                color=colors[subtype], edgecolor='black', linewidth=0.3)

    ax2.set_xlabel('Block', fontsize=14)
    ax2.set_ylabel('Cell Percentage (%)', fontsize=14)
    ax2.set_title('Detailed Subtype Comparison', fontsize=15, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(blocks, fontsize=11)
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.set_ylim(0, 80)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save outputs
    output_png = OUTPUT_DIR / "figure6_molecular_subtype_distribution.png"
    output_pdf = OUTPUT_DIR / "figure6_molecular_subtype_distribution.pdf"

    print(f"Saving to {output_png}...")
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"Saving to {output_pdf}...")
    fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')

    plt.close()

    return output_png, output_pdf


def create_summary_table(stats_df):
    """Create summary statistics table"""
    print("\n" + "=" * 80)
    print("Molecular Subtype Summary Statistics")
    print("=" * 80)

    subtypes = ['Luminal A', 'Luminal B', 'HER2 Enriched', 'Triple Negative',
               'Luminal A (HER2+)', 'Luminal B (HER2+)']

    # Print header
    print(f"{'Block':<8} {'Total':<8}", end='')
    for st in subtypes:
        print(f" {st[:15]:<16}", end='')
    print()
    print("-" * 120)

    # Print data
    for _, row in stats_df.iterrows():
        print(f"{row['Block']:<8} {row['Total Cells']:<8}", end='')
        for st in subtypes:
            count = row[f'{st}_count']
            pct = row[f'{st}_pct']
            print(f" {count:>4} ({pct:>5.1f}%)", end='')
        print()

    # Print totals
    print("-" * 120)
    totals = stats_df[['Total Cells'] + [f'{st}_count' for st in subtypes]].sum()
    print(f"{'Total':<8} {int(totals['Total Cells']):<8}", end='')
    for st in subtypes:
        count = int(totals[f'{st}_count'])
        pct = count / totals['Total Cells'] * 100 if totals['Total Cells'] > 0 else 0
        print(f" {count:>4} ({pct:>5.1f}%)", end='')
    print()

    return stats_df


if __name__ == "__main__":
    print("=" * 60)
    print("Figure 6: Molecular Subtype Distribution by Block")
    print("=" * 60)

    # Calculate statistics
    stats_df = create_subtype_statistics(BLOCKS)

    # Save detailed statistics
    stats_output = OUTPUT_DIR / "figure6_subtype_statistics.csv"
    stats_df.to_csv(stats_output, index=False)
    print(f"\nStatistics saved to: {stats_output}")

    # Create summary table
    stats_df = create_summary_table(stats_df)

    # Create figure
    output_png, output_pdf = create_figure6(stats_df)

    print("\n" + "=" * 60)
    print("Output files:")
    print(f"  PNG: {output_png}")
    print(f"  PDF: {output_pdf}")
    print(f"  CSV: {stats_output}")
    print("=" * 60)
