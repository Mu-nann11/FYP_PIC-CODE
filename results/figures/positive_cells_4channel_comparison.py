"""
Positive Cells - 4-Channel Intensity Comparison
Similar layout to negative_control_intensity_histograms.png but for positive cells (grade >= 1)
Using Otsu-based universal thresholds across all blocks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ==================== Paths ====================
BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")
RESULTS_DIR = BASE_DIR / "results"
SEG_DIR = RESULTS_DIR / "segmentation"
OUTPUT_DIR = RESULTS_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# All blocks with graded data
ALL_BLOCKS = ["A1", "A8", "D1", "E10", "G1", "H2", "H10", "J10"]

# Universal Otsu Thresholds (from A1 expression_statistics_universal.csv)
# These are computed from positive blocks (blocks with known high expression)
THRESHOLDS = {
    'ER':   {'neg': 459.09,   'otsu1': 490.31,  'otsu2': 520.85},
    'PR':   {'neg': 2888.38, 'otsu1': 3838.67,  'otsu2': 4906.54},
    'HER2': {'neg': 2779.41, 'otsu1': 3236.58,  'otsu2': 3560.20},
    'Ki67': {'neg': 977.92,  'otsu1': 2491.89,  'otsu2': 4351.70},
}

# Color scheme (matching HER2_otsu_grading_illustration_v2.py)
COLORS = {
    'ER':   '#3498DB',   # Blue
    'PR':   '#9B59B6',   # Purple
    'HER2': '#E67E22',   # Orange
    'Ki67': '#27AE60',   # Green
}
COLOR_NEG_LINE = '#E74C3C'   # Red - negative threshold
COLOR_OTSU_LINE = '#34495E'  # Dark gray - Otsu thresholds


def load_positive_cells():
    """Load all positive cells (grade >= 1) from universal graded files"""
    all_data = []
    
    for block in ALL_BLOCKS:
        csv_path = SEG_DIR / block / f"{block}_features_graded_universal.csv"
        if not csv_path.exists():
            continue
        
        df = pd.read_csv(csv_path)
        
        # Select relevant columns
        cols = ['cell_label', 
                'ER_nuc_mean', 'ER_nuc_grade',
                'PR_nuc_mean', 'PR_nuc_grade',
                'HER2_nuc_mean', 'HER2_nuc_grade',
                'Ki67_nuc_mean', 'Ki67_nuc_grade']
        
        available_cols = [c for c in cols if c in df.columns]
        block_df = df[available_cols].copy()
        block_df['block'] = block  # Add block identifier
        all_data.append(block_df)
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)


def plot_channel(ax, data, channel, thresholds, color):
    """Plot intensity histogram for one channel with Otsu threshold zones"""
    
    # Filter positive cells (grade >= 1)
    intensities = data[data[f'{channel}_nuc_grade'] >= 1][f'{channel}_nuc_mean'].dropna()
    intensities = intensities[intensities > 0].copy()
    
    if len(intensities) == 0:
        ax.text(0.5, 0.5, f'No positive cells\nfor {channel}',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        return
    
    neg_thresh = thresholds['neg']
    otsu1 = thresholds['otsu1']
    otsu2 = thresholds['otsu2']
    
    # X-axis range: start from negative threshold, not 0
    x_min = neg_thresh * 0.95  # Start slightly before threshold
    x_max = max(intensities.max() * 1.05, otsu2 * 1.2)
    x_max = min(x_max, intensities.max() * 1.1)
    
    # Draw histogram - filter to show only >= neg_thresh
    filtered_intensities = intensities[intensities >= neg_thresh].copy()
    if len(filtered_intensities) == 0:
        ax.text(0.5, 0.5, f'No cells >= threshold\nfor {channel}',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        return
    
    bins = np.linspace(x_min, x_max, 80)
    counts, bin_edges, patches = ax.hist(filtered_intensities, bins=bins, alpha=0.0,
                                         edgecolor='white', linewidth=0.5)
    
    # Color each bin by grade zone (grade 1+, 2+, 3+ only)
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if count == 0:
            continue
        bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
        
        if bin_center < otsu1:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        elif bin_center < otsu2:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    # Background color zones (for grade 1+, 2+, 3+ regions only)
    ax.axvspan(neg_thresh, otsu1, alpha=0.06, color=color, zorder=0)
    ax.axvspan(otsu1, otsu2, alpha=0.06, color=color, zorder=0)
    ax.axvspan(otsu2, x_max, alpha=0.06, color=color, zorder=0)
    
    # Threshold lines - with labels
    ax.axvline(neg_thresh, color=COLOR_NEG_LINE, linestyle='--', linewidth=2.5, zorder=10)
    ax.axvline(otsu1, color=COLOR_OTSU_LINE, linestyle='-.', linewidth=2.0, zorder=10)
    ax.axvline(otsu2, color=COLOR_OTSU_LINE, linestyle='-.', linewidth=2.0, zorder=10)
    
    # Grade labels on top (foreground layer - high zorder to avoid being covered)
    y_max = max(counts) * 1.15 if len(counts) > 0 else 1
    
    # Calculate vertical positions
    grade_y = y_max * 0.90
    
    # Grade 1+ label (left of otsu1)
    ax.text((neg_thresh + otsu1) / 2, grade_y, '1+\n(Weak)', ha='center', va='center',
            fontsize=11, fontweight='bold', color=color, zorder=15)
    
    # Grade 2+ label (between otsu1 and otsu2)
    ax.text((otsu1 + otsu2) / 2, grade_y, '2+\n(Moderate)', ha='center', va='center',
            fontsize=11, fontweight='bold', color=color, zorder=15)
    
    # Grade 3+ label (right of otsu2)
    ax.text((otsu2 + x_max * 0.85) / 2, grade_y, '3+\n(Strong)', ha='center', va='center',
            fontsize=11, fontweight='bold', color=color, zorder=15)
    
    # Add threshold value annotations on top of threshold lines
    # 1+/2+ boundary (Otsu 1)
    ax.annotate(f'Otsu₁ = {otsu1:.1f}', 
                xy=(otsu1, y_max * 0.15), 
                xytext=(otsu1, y_max * 0.05),
                ha='center', va='top',
                fontsize=8, color=COLOR_OTSU_LINE,
                fontweight='bold', zorder=20,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # 2+/3+ boundary (Otsu 2) - positioned to avoid being covered
    ax.annotate(f'Otsu₂ = {otsu2:.1f}', 
                xy=(otsu2, y_max * 0.15), 
                xytext=(otsu2, y_max * 0.05),
                ha='center', va='top',
                fontsize=8, color=COLOR_OTSU_LINE,
                fontweight='bold', zorder=20,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Negative threshold annotation
    ax.annotate(f'Neg = {neg_thresh:.1f}', 
                xy=(neg_thresh, y_max * 0.15), 
                xytext=(neg_thresh, y_max * 0.05),
                ha='center', va='top',
                fontsize=8, color=COLOR_NEG_LINE,
                fontweight='bold', zorder=20,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Axis labels
    channel_full = {
        'ER': 'ER (Estrogen Receptor)',
        'PR': 'PR (Progesterone Receptor)',
        'HER2': 'HER2',
        'Ki67': 'Ki67 (Proliferation Marker)'
    }
    ax.set_xlabel('Nuclear Intensity (a.u.)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
    ax.set_title(f'{channel_full[channel]}', fontsize=14, fontweight='bold', pad=10, color=color)
    
    # Statistics box (moved to bottom-right to avoid overlap)
    n_total = len(data) if f'{channel}_nuc_grade' in data.columns else 0
    n_pos = len(intensities)
    pct = n_pos / n_total * 100 if n_total > 0 else 0
    textstr = (f'n = {n_pos:,}\n'
               f'Positive% = {pct:.1f}%\n'
               f'Median = {intensities.median():.0f}\n'
               f'Max = {intensities.max():.0f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    print("=" * 70)
    print("Positive Cells 4-Channel Intensity Comparison")
    print("=" * 70)
    
    print("\n[1] Loading positive cell data from universal graded files...")
    df = load_positive_cells()
    if df is None:
        print("  [ERROR] No data found!")
        return
    print(f"    Total blocks loaded: {df['block'].nunique()}")
    print(f"    Total cells: {len(df):,}")
    
    print("\n[2] Generating figure...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Positive Cell Nuclear Intensity Distribution — 4 Channels\n'
                 'Otsu-Based Grading Thresholds (Grade 1+, 2+, 3+ cells from all blocks)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ER (top-left)
    plot_channel(axes[0, 0], df, 'ER', THRESHOLDS['ER'], COLORS['ER'])
    
    # PR (top-right)
    plot_channel(axes[0, 1], df, 'PR', THRESHOLDS['PR'], COLORS['PR'])
    
    # HER2 (bottom-left)
    plot_channel(axes[1, 0], df, 'HER2', THRESHOLDS['HER2'], COLORS['HER2'])
    
    # Ki67 (bottom-right)
    plot_channel(axes[1, 1], df, 'Ki67', THRESHOLDS['Ki67'], COLORS['Ki67'])
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    # Save
    output_path = OUTPUT_DIR / "positive_cells_4channel_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"    [OK] Saved: {output_path}")
    
    pdf_path = OUTPUT_DIR / "positive_cells_4channel_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"    [OK] Saved: {pdf_path}")
    
    plt.close()
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
