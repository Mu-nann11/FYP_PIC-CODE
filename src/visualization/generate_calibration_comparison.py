"""
Generate an overlaid calibration histogram comparison for all channels.

Usage:
    python generate_calibration_comparison.py

Outputs:
    results/calibration/calibration_comparison.png
    results/calibration/calibration_comparison.pdf
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ==================== Configuration ====================
BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")
SEGMENTATION_DIR = BASE_DIR / "results" / "segmentation"
CALIBRATION_DIR = BASE_DIR / "results" / "calibration"

# Negative control block definitions
NEG_3CH_BLOCKS = ["A1", "D1", "E10", "H2", "H10", "J10"]
NEG_KI67_BLOCKS = ["A8", "D1", "G1", "H10"]

# Feature column mapping (per config.py)
FEATURE_COLUMNS = {
    "ER": ("3ch_Neg", "ER_nuc_mean"),
    "PR": ("3ch_Neg", "PR_nuc_mean"),
    "HER2": ("3ch_Neg", "HER2_cyto_only_mean"),
    "KI67": ("Ki67_Neg", "Ki67_nuc_mean"),
}

# Color palette (report-friendly colors)
COLORS = {
    "ER": "#E74C3C",      # Red
    "PR": "#3498DB",      # Blue
    "HER2": "#2ECC71",    # Green
    "KI67": "#F39C12",    # Orange
}

# Threshold data (loaded from thresholds.json)
THRESHOLDS_FILE = CALIBRATION_DIR / "thresholds.json"

# ==================== Functions ====================

def load_thresholds() -> Dict:
    """Load the calibration threshold JSON."""
    with open(THRESHOLDS_FILE, "r") as f:
        return json.load(f)


def load_channel_data(channel: str, neg_blocks: list, column: str) -> np.ndarray:
    """
    Load all cell-intensity values for a specific channel.
    
    Args:
        channel: Channel name (ER/PR/HER2/Ki67)
        neg_blocks: Negative-control block list
        column: CSV column name
    
    Returns:
        all_values: Intensity values across all negative-control blocks
    """
    all_values = []
    
    for block in neg_blocks:
        csv_file = SEGMENTATION_DIR / "TMAd" / block / f"{block}_TMAd_features.csv"
        if not csv_file.exists():
            print(f"  [WARN] Missing file: {csv_file}")
            continue
        
        df = pd.read_csv(csv_file)
        if column not in df.columns:
            print(f"  [WARN] Missing column: {column} in {block}")
            continue
        
        # Filter invalid data (NaN, 0, etc.)
        values = df[column].dropna()
        values = values[values > 0]  # Keep positive values only
        all_values.extend(values.tolist())
        print(f"  [OK] {block}: {len(values)} cells")
    
    return np.array(all_values)


def plot_overlaid_histograms(thresholds: Dict) -> Tuple[plt.Figure, Dict]:
    """
    Generate overlaid histograms.
    
    Args:
        thresholds: Threshold data loaded from thresholds.json
    
    Returns:
        fig, stats: Matplotlib figure and statistics dictionary
    """
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    
    stats = {}
    
    # Plot a histogram for each channel
    for channel, (neg_group, column) in FEATURE_COLUMNS.items():
        
        # Determine the negative-control block list
        if neg_group == "3ch_Neg":
            neg_blocks = NEG_3CH_BLOCKS
        else:  # Ki67_Neg
            neg_blocks = NEG_KI67_BLOCKS
        
        print(f"\n{'='*60}")
        print(f"Processing channel: {channel}")
        print(f"Negative-control blocks: {neg_blocks}")
        print(f"Feature column: {column}")
        print(f"{'='*60}")
        
        # Load data
        values = load_channel_data(channel, neg_blocks, column)
        if len(values) == 0:
            print(f"[WARN] No valid data for {channel}; skipping")
            continue
        
        print(f"Total: {len(values)} cells")
        
        # Pull statistics from thresholds.json
        threshold_data = thresholds["channels"][channel][neg_group]
        mean = threshold_data["mean"]
        std = threshold_data["std"]
        threshold = threshold_data["threshold"]
        n_cells = threshold_data["n_cells"]
        
        stats[channel] = {
            "values": values,
            "mean": mean,
            "std": std,
            "threshold": threshold,
            "n_cells": n_cells,
            "color": COLORS[channel],
        }
        
        # Plot histogram
        bins = np.linspace(np.min(values) * 0.9, np.max(values) * 1.1, 50)
        ax.hist(values, bins=bins, alpha=0.4, label=f"{channel} (n={n_cells})",
                color=COLORS[channel], edgecolor="black", linewidth=0.5)
        
        # Plot threshold line
        ax.axvline(threshold, color=COLORS[channel], linestyle="--", 
                   linewidth=2.5, label=f"{channel} threshold: {threshold:.1f}")
        
        # Plot mean +/- standard deviation markers
        ax.axvline(mean, color=COLORS[channel], linestyle=":", 
                   linewidth=2, alpha=0.7)
        ax.axvline(mean - std, color=COLORS[channel], linestyle=":", 
                   linewidth=1, alpha=0.5)
        ax.axvline(mean + std, color=COLORS[channel], linestyle=":", 
                   linewidth=1, alpha=0.5)
        
        # Add sample count label at the top
        y_pos = ax.get_ylim()[1] * 0.95
        ax.text(threshold, y_pos, f"n={n_cells}", 
               color=COLORS[channel], fontsize=10, fontweight="bold",
               ha="center", bbox=dict(boxstyle="round,pad=0.3", 
               facecolor="white", alpha=0.7, edgecolor=COLORS[channel]))
    
        # Style the chart
    ax.set_xlabel("Cell Intensity", fontsize=14, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")
    ax.set_title("Calibration Thresholds - Overlaid Histograms\n" +
                 "Negative Controls (3ch_Neg & Ki67_Neg)", 
                 fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#F8F9FA")
    
    plt.tight_layout()
    
    return fig, stats


def create_summary_stats_panel(stats: Dict) -> str:
    """Generate a statistics summary panel."""
    panel = "\nStatistics Panel\n"
    panel += "=" * 70 + "\n"
    
    for channel, data in stats.items():
        panel += f"\n{channel}:\n"
        panel += f"  Sample count: {data['n_cells']}\n"
        panel += f"  Mean: {data['mean']:.2f}\n"
        panel += f"  Standard deviation: {data['std']:.2f}\n"
        panel += f"  Threshold (mean + 2*std): {data['threshold']:.2f}\n"
        panel += f"  Data range: [{data['values'].min():.2f}, {data['values'].max():.2f}]\n"
    
    return panel


# ==================== Main Program ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("Generating calibration histogram comparison")
    print("="*70)
    
    # Load threshold data
    print("\n➤ Loading threshold data...")
    thresholds = load_thresholds()
    print(f"[OK] Loaded; generated at: {thresholds['generated_at']}")
    
    # Generate the plot
    print("\n➤ Generating overlaid histograms...")
    fig, stats = plot_overlaid_histograms(thresholds)
    
    # Save files
    output_png = CALIBRATION_DIR / "calibration_comparison.png"
    output_pdf = CALIBRATION_DIR / "calibration_comparison.pdf"
    
    print("\n➤ Saving files...")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"[OK] PNG saved: {output_png}")
    
    fig.savefig(output_pdf, bbox_inches="tight")
    print(f"[OK] PDF saved: {output_pdf}")
    
    # Print statistics
    print(create_summary_stats_panel(stats))
    
    print("\n" + "="*70)
    print("[OK] Done!")
    print("="*70)
