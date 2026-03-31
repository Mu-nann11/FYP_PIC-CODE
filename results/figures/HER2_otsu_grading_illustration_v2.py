#!/usr/bin/env python3
"""
HER2 阳性细胞强度分级示意图 - 迭代 Otsu 方法
基于 Code/calibration/analyze_negative_controls.py 中的实际算法实现

功能：
1. 展示阴性细胞分布和阴性阈值（mean + 2*SD）
2. 展示阳性细胞分布
3. 展示迭代 Otsu 分级点（1+/2+ 和 2+/3+）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ==================== 路径设置 ====================
BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")
OUTPUT_DIR = BASE_DIR / "results" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ==================== 实际阈值数据（来自 thresholds.json） ====================
# 阴性阈值
NEG_THRESHOLD = 2345.72  # HER2 膜环特征，mean + 2*SD

# Otsu 分级阈值（基于阳性 Block 迭代 Otsu）
OTSU_1 = 3500.0  # 1+/2+ 分界
OTSU_2 = 4500.0  # 2+/3+ 分界

# ==================== 生成模拟数据 ====================
np.random.seed(42)

# 阴性细胞分布（基于实际统计数据）
n_neg = 15417
neg_mean = 1398.20
neg_std = 473.76
neg_data = np.random.normal(neg_mean, neg_std, n_neg)
neg_data = neg_data[neg_data > 0]

# 阳性细胞分布（模拟 1+, 2+, 3+ 三个等级）
n_pos_1plus = 3000
n_pos_2plus = 1500
n_pos_3plus = 800

# 1+ 阳性：强度略高于阴性阈值
pos_1plus = np.random.normal(3000, 300, n_pos_1plus)
pos_1plus = pos_1plus[(pos_1plus > NEG_THRESHOLD) & (pos_1plus < OTSU_1)]

# 2+ 阳性：中等强度
pos_2plus = np.random.normal(4000, 300, n_pos_2plus)
pos_2plus = pos_2plus[(pos_2plus >= OTSU_1) & (pos_2plus < OTSU_2)]

# 3+ 阳性：高强度
pos_3plus = np.random.normal(5500, 500, n_pos_3plus)
pos_3plus = pos_3plus[pos_3plus >= OTSU_2]

# 合并所有数据
all_data = np.concatenate([neg_data, pos_1plus, pos_2plus, pos_3plus])

# ==================== 创建图形 ====================
fig, ax = plt.subplots(figsize=(16, 9))

# 颜色方案
COLOR_NEG = '#3498db'      # 蓝色 - 阴性
COLOR_1PLUS = '#f39c12'    # 橙色 - 1+
COLOR_2PLUS = '#2ecc71'     # 绿色 - 2+
COLOR_3PLUS = '#9b59b6'     # 紫色 - 3+
COLOR_NEG_LINE = '#e74c3c'  # 红色 - 阴性阈值线
COLOR_OTSU_LINE = '#34495e' # 深灰 - Otsu阈值线

# 绘制直方图
bins = np.linspace(0, max(all_data) * 1.1, 120)
counts, bin_edges, patches = ax.hist(all_data, bins=bins, alpha=0.0, 
                                       edgecolor='white', linewidth=0.5)

# 对每个 bin着色
for i, (count, patch) in enumerate(zip(counts, patches)):
    if count == 0:
        continue
    bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
    
    if bin_center < NEG_THRESHOLD:
        patch.set_facecolor(COLOR_NEG)
    elif bin_center < OTSU_1:
        patch.set_facecolor(COLOR_1PLUS)
    elif bin_center < OTSU_2:
        patch.set_facecolor(COLOR_2PLUS)
    else:
        patch.set_facecolor(COLOR_3PLUS)
    patch.set_alpha(0.75)

# ==================== 添加阈值线和标注 ====================
# 阴性阈值线 (mean + 2*SD)
ax.axvline(NEG_THRESHOLD, color=COLOR_NEG_LINE, linestyle='--', 
           linewidth=3, zorder=10)
ax.annotate(f'Negative Threshold\n(mean + 2×SD)\n{NEG_THRESHOLD:.0f}',
            xy=(NEG_THRESHOLD, ax.get_ylim()[1] * 0.85),
            xytext=(NEG_THRESHOLD + 400, ax.get_ylim()[1] * 0.9),
            fontsize=11, fontweight='bold', color=COLOR_NEG_LINE,
            ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color=COLOR_NEG_LINE, lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                      edgecolor=COLOR_NEG_LINE, alpha=0.9))

# Otsu 阈值线 1 (1+/2+ 分界)
ax.axvline(OTSU_1, color=COLOR_OTSU_LINE, linestyle='-.', 
           linewidth=2.5, zorder=10)
ax.annotate(f'1+/2+ Boundary\n(Otsu Threshold)\n{OTSU_1:.0f}',
            xy=(OTSU_1, ax.get_ylim()[1] * 0.65),
            xytext=(OTSU_1 + 350, ax.get_ylim()[1] * 0.72),
            fontsize=11, fontweight='bold', color=COLOR_OTSU_LINE,
            ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color=COLOR_OTSU_LINE, lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                      edgecolor=COLOR_OTSU_LINE, alpha=0.9))

# Otsu 阈值线 2 (2+/3+ 分界)
ax.axvline(OTSU_2, color=COLOR_OTSU_LINE, linestyle='-.', 
           linewidth=2.5, zorder=10)
ax.annotate(f'2+/3+ Boundary\n(Otsu Threshold)\n{OTSU_2:.0f}',
            xy=(OTSU_2, ax.get_ylim()[1] * 0.5),
            xytext=(OTSU_2 + 500, ax.get_ylim()[1] * 0.58),
            fontsize=11, fontweight='bold', color=COLOR_OTSU_LINE,
            ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color=COLOR_OTSU_LINE, lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                      edgecolor=COLOR_OTSU_LINE, alpha=0.9))

# ==================== 添加分级区域背景 ====================
y_max = max(counts) * 1.15

# 阴性区域
ax.axvspan(0, NEG_THRESHOLD, alpha=0.08, color=COLOR_NEG, zorder=0)
ax.text(NEG_THRESHOLD / 2, y_max * 0.92, 'Grade 0\n(Negative)', 
        ha='center', va='center', fontsize=14, fontweight='bold', 
        color=COLOR_NEG)

# 1+ 区域
ax.axvspan(NEG_THRESHOLD, OTSU_1, alpha=0.08, color=COLOR_1PLUS, zorder=0)
ax.text((NEG_THRESHOLD + OTSU_1) / 2, y_max * 0.92, 'Grade 1+\n(Weak)', 
        ha='center', va='center', fontsize=14, fontweight='bold', 
        color=COLOR_1PLUS)

# 2+ 区域
ax.axvspan(OTSU_1, OTSU_2, alpha=0.08, color=COLOR_2PLUS, zorder=0)
ax.text((OTSU_1 + OTSU_2) / 2, y_max * 0.92, 'Grade 2+\n(Moderate)', 
        ha='center', va='center', fontsize=14, fontweight='bold', 
        color=COLOR_2PLUS)

# 3+ 区域
ax.axvspan(OTSU_2, max(all_data) * 1.1, alpha=0.08, color=COLOR_3PLUS, zorder=0)
ax.text((OTSU_2 + max(all_data) * 0.9) / 2, y_max * 0.92, 'Grade 3+\n(Strong)', 
        ha='center', va='center', fontsize=14, fontweight='bold', 
        color=COLOR_3PLUS)

# ==================== 添加流程图（右侧） ====================
# 流程框位置（往右移，避免挡住 2+/3+ 标注）
box_x = max(all_data) * 0.72
box_width = max(all_data) * 0.25
box_height = y_max * 0.65

# 标题
ax.text(box_x + box_width/2, box_height * 1.08, 'Grading Algorithm', 
        ha='center', va='bottom', fontsize=13, fontweight='bold',
        color='#2c3e50')

# 背景框
fancy_box = FancyBboxPatch((box_x, box_height * 0.25), box_width, box_height * 0.75,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             facecolor='#ecf0f1', edgecolor='#bdc3c7', 
                             linewidth=2, alpha=0.95, zorder=3)
ax.add_patch(fancy_box)

# 步骤文本（2+/3+ 放在第一位，作为第一图层）
steps = [
    ('Step 1', 'Iterative Otsu', f'2+/3+ = {OTSU_2:.0f}', COLOR_OTSU_LINE),
    ('Step 2', 'Iterative Otsu', f'1+/2+ = {OTSU_1:.0f}', COLOR_OTSU_LINE),
    ('Step 3', 'Negative Threshold', f'mean + 2×SD = {NEG_THRESHOLD:.0f}', COLOR_NEG_LINE),
]

for i, (step, title, value, color) in enumerate(steps):
    y_pos = box_height * (0.75 - i * 0.25)
    
    # 步骤标签
    ax.text(box_x + 10, y_pos, step, fontsize=10, fontweight='bold',
            color=color, va='center')
    
    # 标题
    ax.text(box_x + box_width * 0.35, y_pos, title, fontsize=10, 
            va='center', color='#2c3e50')
    
    # 值
    ax.text(box_x + box_width * 0.75, y_pos, value, fontsize=10, 
            fontweight='bold', va='center', color=color)

# ==================== 统计信息框 ====================
# 计算实际数量
n_total = len(all_data)
n_grade0 = (all_data < NEG_THRESHOLD).sum()
n_grade1 = ((all_data >= NEG_THRESHOLD) & (all_data < OTSU_1)).sum()
n_grade2 = ((all_data >= OTSU_1) & (all_data < OTSU_2)).sum()
n_grade3 = (all_data >= OTSU_2).sum()

stats_text = f"""Statistical Summary

Total cells: {n_total:,}
─────────────────────
Grade 0 (Negative): {n_grade0:,} ({n_grade0/n_total*100:.1f}%)
Grade 1+ (Weak):     {n_grade1:,} ({n_grade1/n_total*100:.1f}%)
Grade 2+ (Moderate): {n_grade2:,} ({n_grade2/n_total*100:.1f}%)
Grade 3+ (Strong):  {n_grade3:,} ({n_grade3/n_total*100:.1f}%)
─────────────────────
Method: Iterative Otsu
Data: HER2 Membrane Ring Intensity"""

# 统计框位置
stats_box = FancyBboxPatch((20, box_height * 0.15), max(all_data) * 0.28, box_height * 0.7,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            facecolor='#fff9e6', edgecolor='#f39c12', 
                            linewidth=2, alpha=0.95, zorder=3)
ax.add_patch(stats_box)

ax.text(max(all_data) * 0.15 + 20, box_height * 0.78, stats_text,
        fontsize=10, va='top', ha='center', color='#2c3e50',
        family='monospace', linespacing=1.3)

# ==================== 图例 ====================
legend_patches = [
    mpatches.Patch(color=COLOR_NEG, alpha=0.75, label='Grade 0 (Negative)'),
    mpatches.Patch(color=COLOR_1PLUS, alpha=0.75, label='Grade 1+ (Weak Positive)'),
    mpatches.Patch(color=COLOR_2PLUS, alpha=0.75, label='Grade 2+ (Moderate Positive)'),
    mpatches.Patch(color=COLOR_3PLUS, alpha=0.75, label='Grade 3+ (Strong Positive)'),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=11, 
          framealpha=0.95, edgecolor='#bdc3c7')

# ==================== 标签和标题 ====================
ax.set_xlabel('HER2 Membrane Ring Intensity (Pixel Value)', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('Number of Cells', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('HER2 Positive Cell Intensity Grading — Iterative Otsu Method', 
             fontsize=16, fontweight='bold', pad=20, color='#2c3e50')

# 网格
ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.8)
ax.set_axisbelow(True)

# 设置 x 轴范围
ax.set_xlim(0, max(all_data) * 1.1)
ax.set_ylim(0, y_max)

# 移除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# ==================== 保存 ====================
output_path = OUTPUT_DIR / "HER2_otsu_grading_illustration_v2.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"[OK] Figure saved: {output_path}")

# 也保存 PDF 版本
pdf_path = OUTPUT_DIR / "HER2_otsu_grading_illustration_v2.pdf"
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
print(f"[OK] PDF saved: {pdf_path}")

plt.close()
