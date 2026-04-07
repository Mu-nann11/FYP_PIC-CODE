#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Data path
BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")

# Collect HER2 data from all blocks
all_her2_data = []
blocks = ["A1", "A8", "D1", "E10", "G1", "H10", "H2", "J10"]

for block in blocks:
    csv_path = BASE_DIR / "results" / "segmentation" / "TMAd" / block / f"{block}_TMAd_features.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "HER2_nuc_mean" in df.columns:
            her2_values = df["HER2_nuc_mean"].dropna().values
            all_her2_data.extend(her2_values)

all_her2_data = np.array(all_her2_data)

# HER2 threshold parameters
neg_threshold = 2779.41
otsu_1 = 3237.00
otsu_2 = 3560.00

# Create figure
fig, ax = plt.subplots(figsize=(14, 7))

# Plot histogram
counts, bins, patches = ax.hist(all_her2_data, bins=100, color="#3498db", alpha=0.7, edgecolor="black", linewidth=0.5)

# Color histogram bins by grade range
for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i + 1]) / 2
    if bin_center < neg_threshold:
        patch.set_facecolor("#e74c3c")  # Red - Grade 0
    elif bin_center < otsu_1:
        patch.set_facecolor("#f39c12")  # Orange - Grade 1+
    elif bin_center < otsu_2:
        patch.set_facecolor("#2ecc71")  # Green - Grade 2+
    else:
        patch.set_facecolor("#9b59b6")  # Purple - Grade 3+

# Add threshold lines
ax.axvline(neg_threshold, color="red", linestyle="--", linewidth=3, label=f"neg_threshold = {neg_threshold:.0f}", zorder=5)
ax.axvline(otsu_1, color="green", linestyle="--", linewidth=3, label=f"Otsu_1 (weak/medium) = {otsu_1:.0f}", zorder=5)
ax.axvline(otsu_2, color="orange", linestyle="--", linewidth=3, label=f"Otsu_2 (medium/strong) = {otsu_2:.0f}", zorder=5)

# Add grading labels
y_max = max(counts) * 0.85
ax.text((0 + neg_threshold) / 2, y_max, "Grade 0\n(No expression)", ha="center", fontsize=11, weight="bold", color="#e74c3c", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.text((neg_threshold + otsu_1) / 2, y_max, "Grade 1+\n(Weak positive)", ha="center", fontsize=11, weight="bold", color="#f39c12", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.text((otsu_1 + otsu_2) / 2, y_max, "Grade 2+\n(Moderate positive)", ha="center", fontsize=11, weight="bold", color="#2ecc71", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.text((otsu_2 + max(all_her2_data)) / 2, y_max, "Grade 3+\n(Strong positive)", ha="center", fontsize=11, weight="bold", color="#9b59b6", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Statistics
grade_0_count = (all_her2_data < neg_threshold).sum()
grade_1_count = ((all_her2_data >= neg_threshold) & (all_her2_data < otsu_1)).sum()
grade_2_count = ((all_her2_data >= otsu_1) & (all_her2_data < otsu_2)).sum()
grade_3_count = (all_her2_data >= otsu_2).sum()

# Add summary text
textstr = f'''HER2 nuclear intensity grading statistics

Grade 0 (No expression): {grade_0_count:,} nuclei ({grade_0_count / len(all_her2_data) * 100:.1f}%)
Grade 1+ (Weak positive): {grade_1_count:,} nuclei ({grade_1_count / len(all_her2_data) * 100:.2f}%)
Grade 2+ (Moderate positive): {grade_2_count:,} nuclei ({grade_2_count / len(all_her2_data) * 100:.2f}%)
Grade 3+ (Strong positive): {grade_3_count:,} nuclei ({grade_3_count / len(all_her2_data) * 100:.2f}%)

Total nuclei: {len(all_her2_data):,}
Grading method: iterative Otsu (68 positive cells >= 10)
Data source: 8 TMA samples'''

ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9), family="monospace")

# Labels and title
ax.set_xlabel("HER2 nuclear mean intensity (Mean Intensity)", fontsize=13, weight="bold")
ax.set_ylabel("Cell count (Number of Cells)", fontsize=13, weight="bold")
ax.set_title("HER2 positive cell intensity grading illustration (iterative Otsu method)", fontsize=15, weight="bold", pad=20)
ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()

# Save figure
fig_dir = BASE_DIR / "results" / "figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "HER2_otsu_grading_illustration.png", dpi=300, bbox_inches="tight")
print(f"Figure saved: {fig_dir / 'HER2_otsu_grading_illustration.png'}")

plt.close()
