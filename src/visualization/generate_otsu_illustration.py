#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 数据路径
BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")

# 收集所有块的HER2数据
all_her2_data = []
blocks = ['A1', 'A8', 'D1', 'E10', 'G1', 'H10', 'H2', 'J10']

for block in blocks:
    csv_path = BASE_DIR / "results" / "segmentation" / block / f"{block}_features.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if 'HER2_nuc_mean' in df.columns:
            her2_values = df['HER2_nuc_mean'].dropna().values
            all_her2_data.extend(her2_values)

all_her2_data = np.array(all_her2_data)

# HER2 阈值参数
neg_threshold = 2779.41
otsu_1 = 3237.00
otsu_2 = 3560.00

# 创建图
fig, ax = plt.subplots(figsize=(14, 7))

# 绘制直方图
counts, bins, patches = ax.hist(all_her2_data, bins=100, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.5)

# 对直方图着色（按分级区间）
for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    if bin_center < neg_threshold:
        patch.set_facecolor('#e74c3c')  # 红色 - Grade 0
    elif bin_center < otsu_1:
        patch.set_facecolor('#f39c12')  # 橙色 - Grade 1+
    elif bin_center < otsu_2:
        patch.set_facecolor('#2ecc71')  # 绿色 - Grade 2+
    else:
        patch.set_facecolor('#9b59b6')  # 紫色 - Grade 3+

# 添加分界线
ax.axvline(neg_threshold, color='red', linestyle='--', linewidth=3, label=f'neg_threshold = {neg_threshold:.0f}', zorder=5)
ax.axvline(otsu_1, color='green', linestyle='--', linewidth=3, label=f'Otsu_1 (弱/中) = {otsu_1:.0f}', zorder=5)
ax.axvline(otsu_2, color='orange', linestyle='--', linewidth=3, label=f'Otsu_2 (中/强) = {otsu_2:.0f}', zorder=5)

# 添加分级标签
y_max = max(counts) * 0.85
ax.text((0 + neg_threshold)/2, y_max, 'Grade 0\n(无表达)', ha='center', fontsize=11, weight='bold', color='#e74c3c', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text((neg_threshold + otsu_1)/2, y_max, 'Grade 1+\n(弱阳性)', ha='center', fontsize=11, weight='bold', color='#f39c12', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text((otsu_1 + otsu_2)/2, y_max, 'Grade 2+\n(中阳性)', ha='center', fontsize=11, weight='bold', color='#2ecc71', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.text((otsu_2 + max(all_her2_data))/2, y_max, 'Grade 3+\n(强阳性)', ha='center', fontsize=11, weight='bold', color='#9b59b6', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 统计信息
grade_0_count = (all_her2_data < neg_threshold).sum()
grade_1_count = ((all_her2_data >= neg_threshold) & (all_her2_data < otsu_1)).sum()
grade_2_count = ((all_her2_data >= otsu_1) & (all_her2_data < otsu_2)).sum()
grade_3_count = (all_her2_data >= otsu_2).sum()

# 添加文本框
textstr = f'''HER2 核强度分级统计

Grade 0 (无表达): {grade_0_count:,} 核 ({grade_0_count/len(all_her2_data)*100:.1f}%)
Grade 1+ (弱阳性): {grade_1_count:,} 核 ({grade_1_count/len(all_her2_data)*100:.2f}%)
Grade 2+ (中阳性): {grade_2_count:,} 核 ({grade_2_count/len(all_her2_data)*100:.2f}%)
Grade 3+ (强阳性): {grade_3_count:,} 核 ({grade_3_count/len(all_her2_data)*100:.2f}%)

总核数: {len(all_her2_data):,}
分级方法: 迭代 Otsu (阳性细胞68个 ≥ 10)
数据源: 8个TMA样本'''

ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), family='monospace')

# 标签和标题
ax.set_xlabel('HER2 核平均强度 (Mean Intensity)', fontsize=13, weight='bold')
ax.set_ylabel('细胞数量 (Number of Cells)', fontsize=13, weight='bold')
ax.set_title('HER2 阳性细胞强度分级示意图 (迭代 Otsu 方法)', fontsize=15, weight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()

# 保存图
fig_dir = BASE_DIR / "results" / "figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "HER2_otsu_grading_illustration.png", dpi=300, bbox_inches='tight')
print(f"✓ 图已保存: {fig_dir / 'HER2_otsu_grading_illustration.png'}")

plt.close()
