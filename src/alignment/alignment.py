"""
统一配准流程：
  1. Cycle1 内部 HER2/PR/ER -> DAPI（色差校正）
  2. Cycle2 DAPI -> Cycle1 DAPI（计算一次变换）
  3. 同一个变换应用到 Ki67
  4. 自动统一尺寸
  5. 合并 5 通道
"""

import sys
import numpy as np
import tifffile
from pathlib import Path
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate
from scipy.ndimage import shift as ndshift
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# =====================================================================
# 配置
# =====================================================================

if len(sys.argv) > 1:
    BLOCK = sys.argv[1]
else:
    BLOCK = "G2"

# ====== Cycle1 拼接图路径 ======
CYCLE1_DIR = Path(rf"d:\Try_munan\FYP_LAST\results\stitched\TMAd\Cycle1\{BLOCK}")
CYCLE1_DAPI_PATH = CYCLE1_DIR / f"{BLOCK}_TMAd_Cycle1_DAPI.tif"
CYCLE1_HER2_PATH = CYCLE1_DIR / f"{BLOCK}_TMAd_Cycle1_HER2.tif"
CYCLE1_PR_PATH   = CYCLE1_DIR / f"{BLOCK}_TMAd_Cycle1_PR.tif"
CYCLE1_ER_PATH   = CYCLE1_DIR / f"{BLOCK}_TMAd_Cycle1_ER.tif"

# ====== Cycle2 拼接图路径 ======
CYCLE2_DIR = Path(rf"d:\Try_munan\FYP_LAST\results\stitched\TMAd\Cycle2\{BLOCK}")
CYCLE2_DAPI_PATH = CYCLE2_DIR / f"{BLOCK}_TMAd_Cycle2_DAPI.tif"
CYCLE2_KI67_PATH = CYCLE2_DIR / f"{BLOCK}_TMAd_Cycle2_KI67.tif"

# ====== 输出目录 ======
OUTPUT_DIR = Path(rf"d:\Try_munan\FYP_LAST\results\registered\{BLOCK}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 工具函数
# =====================================================================

def load_tiff(path):
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        if img.shape[0] in (3, 4):
            img = img[0]
        elif img.shape[2] in (3, 4):
            img = img[:, :, 0]
    return img.astype(np.float32)


def save_tiff(path, data):
    tifffile.imwrite(str(path), data.astype(np.uint16))


def norm(img):
    p1, p99 = np.percentile(img, (1, 99))
    if p99 - p1 < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - p1) / (p99 - p1), 0, 1).astype(np.float32)


def compute_ncc(a, b):
    min_h = min(a.shape[0], b.shape[0])
    min_w = min(a.shape[1], b.shape[1])
    a, b = a[:min_h, :min_w], b[:min_h, :min_w]

    x, y = a.ravel(), b.ravel()
    mask = (x > 0) | (y > 0)
    if mask.sum() < 100:
        return -1.0
    x, y = x[mask], y[mask]
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float(np.corrcoef(x, y)[0, 1])


def find_content_bbox(img, threshold_ratio=0.01):
    threshold = img.max() * threshold_ratio
    mask = img > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return (0, img.shape[0], 0, img.shape[1])
    y1 = np.argmax(rows)
    y2 = len(rows) - np.argmax(rows[::-1])
    x1 = np.argmax(cols)
    x2 = len(cols) - np.argmax(cols[::-1])
    return (int(y1), int(y2), int(x1), int(x2))


def crop_to_bbox(img, bbox):
    y1, y2, x1, x2 = bbox
    return img[y1:y2, x1:x2]


# =====================================================================
# Step 1: 加载所有图像
# =====================================================================

print("=" * 60)
print(f"Block: {BLOCK}")
print("=" * 60)

dapi1 = load_tiff(CYCLE1_DAPI_PATH)
her2  = load_tiff(CYCLE1_HER2_PATH)
pr    = load_tiff(CYCLE1_PR_PATH)
er    = load_tiff(CYCLE1_ER_PATH)
dapi2 = load_tiff(CYCLE2_DAPI_PATH)
ki67  = load_tiff(CYCLE2_KI67_PATH)

print(f"  Cycle1 DAPI: {dapi1.shape}")
print(f"  Cycle1 HER2: {her2.shape}")
print(f"  Cycle1 PR:   {pr.shape}")
print(f"  Cycle1 ER:   {er.shape}")
print(f"  Cycle2 DAPI: {dapi2.shape}")
print(f"  Cycle2 KI67: {ki67.shape}")


# =====================================================================
# Step 2: Cycle1 内部通道 -> DAPI 色差校正
# =====================================================================

print("\n" + "=" * 60)
print("Step 2: Cycle1 internal alignment (chromatic aberration)")
print("=" * 60)

dapi1_n = norm(dapi1)

cycle1_channels = {"HER2": her2, "PR": pr, "ER": er}
cycle1_aligned = {"DAPI": dapi1}

min_h = dapi1.shape[0]
min_w = dapi1.shape[1]

for name, img in cycle1_channels.items():
    img_n = norm(img)
    ncc_before = compute_ncc(dapi1_n, img_n)

    img_n_crop = img_n[:min_h, :min_w]
    shift, _, _ = phase_cross_correlation(
        dapi1_n, img_n_crop, upsample_factor=100
    )

    img_aligned = ndshift(img, shift=(-shift[0], -shift[1]), order=1, cval=0)
    ncc_after = compute_ncc(dapi1_n, norm(img_aligned[:min_h, :min_w]))

    cycle1_aligned[name] = img_aligned
    print(f"  {name:5s}: shift=({shift[0]:+.2f}, {shift[1]:+.2f})  "
          f"NCC: {ncc_before:.4f} -> {ncc_after:.4f}")


# =====================================================================
# Step 3: Cycle2 DAPI -> Cycle1 DAPI（计算变换，只算一次）
# =====================================================================

print("\n" + "=" * 60)
print("Step 3: Cycle2 DAPI -> Cycle1 DAPI registration")
print("=" * 60)

dapi2_n = norm(dapi2)

# 3a: 快速检查（无旋转）
h_quick = min(dapi1_n.shape[0], dapi2_n.shape[0])
w_quick = min(dapi1_n.shape[1], dapi2_n.shape[1])
shift_quick, _, _ = phase_cross_correlation(
    dapi1_n[:h_quick, :w_quick], dapi2_n[:h_quick, :w_quick], upsample_factor=100
)
shifted_quick = ndshift(dapi2_n[:h_quick, :w_quick], shift=shift_quick, order=1)
ncc_quick = compute_ncc(dapi1_n[:h_quick, :w_quick], shifted_quick)
print(f"  Quick (no rotation): shift=({shift_quick[0]:.1f}, {shift_quick[1]:.1f}), NCC={ncc_quick:.4f}")

# 3b: 两阶段旋转搜索（粗搜 + 精搜）
print(f"  Phase 1: coarse search (1deg step, -5 to +5) ...")
coarse_results = []
for angle in np.linspace(-5, 5, 11):
    dapi2_rot = rotate(dapi2_n, angle, order=1, preserve_range=True)
    h = min(dapi1_n.shape[0], dapi2_rot.shape[0])
    w = min(dapi1_n.shape[1], dapi2_rot.shape[1])
    s, _, _ = phase_cross_correlation(
        dapi1_n[:h, :w], dapi2_rot[:h, :w], upsample_factor=10
    )
    shifted = ndshift(dapi2_rot[:h, :w], shift=s, order=1)
    score = compute_ncc(dapi1_n[:h, :w], shifted)
    coarse_results.append((angle, s[0], s[1], score))

coarse_results.sort(key=lambda x: x[3], reverse=True)
coarse_best = coarse_results[0]
print(f"  Coarse best: angle={coarse_best[0]:.1f}deg, NCC={coarse_best[3]:.4f}")

print(f"  Phase 2: fine search (0.1deg step, {coarse_best[0]-1:.1f} to {coarse_best[0]+1:.1f}) ...")
fine_results = []
for angle in np.arange(coarse_best[0] - 1, coarse_best[0] + 1.01, 0.1):
    dapi2_rot = rotate(dapi2_n, angle, order=1, preserve_range=True)
    h = min(dapi1_n.shape[0], dapi2_rot.shape[0])
    w = min(dapi1_n.shape[1], dapi2_rot.shape[1])
    s, _, _ = phase_cross_correlation(
        dapi1_n[:h, :w], dapi2_rot[:h, :w], upsample_factor=100
    )
    shifted = ndshift(dapi2_rot[:h, :w], shift=s, order=1)
    score = compute_ncc(dapi1_n[:h, :w], shifted)
    fine_results.append((angle, s[0], s[1], score))

fine_results.sort(key=lambda x: x[3], reverse=True)
best_angle, best_dy, best_dx, best_ncc = fine_results[0]

print(f"\n  Best: angle={best_angle:.2f}deg, shift=({best_dy:.2f}, {best_dx:.2f}), NCC={best_ncc:.4f}")

TRANSFORM = {
    "angle": best_angle,
    "shift_y": best_dy,
    "shift_x": best_dx,
}
print(f"\n  Transform: rotate({TRANSFORM['angle']:.2f} deg) + "
      f"shift({TRANSFORM['shift_y']:.2f}, {TRANSFORM['shift_x']:.2f})")


# =====================================================================
# Step 4: 同一个变换应用到 Ki67 和 Cycle2 DAPI
# =====================================================================

print("\n" + "=" * 60)
print("Step 4: Apply SAME transform to Ki67 and Cycle2 DAPI")
print("=" * 60)

angle = TRANSFORM["angle"]
shift = (TRANSFORM["shift_y"], TRANSFORM["shift_x"])

dapi2_aligned = ndshift(
    rotate(dapi2, angle, order=1, preserve_range=True),
    shift=shift, order=1, cval=0
)
ki67_aligned = ndshift(
    rotate(ki67, angle, order=1, preserve_range=True),
    shift=shift, order=1, cval=0
)

h = min(dapi1.shape[0], dapi2_aligned.shape[0])
w = min(dapi1.shape[1], dapi2_aligned.shape[1])
ncc_final = compute_ncc(norm(dapi1[:h, :w]), norm(dapi2_aligned[:h, :w]))
print(f"  DAPI aligned NCC: {ncc_final:.4f}")
print(f"  Ki67 aligned: same transform applied")


# =====================================================================
# Step 5: 自动统一尺寸
# =====================================================================

print("\n" + "=" * 60)
print("Step 5: Auto-size (common crop)")
print("=" * 60)

all_channels = {
    "DAPI":     dapi1,
    "HER2":     cycle1_aligned["HER2"],
    "PR":       cycle1_aligned["PR"],
    "ER":       cycle1_aligned["ER"],
    "DAPI2":    dapi2_aligned,
    "KI67":     ki67_aligned,
}

print(f"\n  Content bboxes:")
bboxes = {}
for name, img in all_channels.items():
    bbox = find_content_bbox(img)
    bboxes[name] = bbox
    print(f"    {name:5s}: y=[{bbox[0]}, {bbox[1]}] x=[{bbox[2]}, {bbox[3]}]  "
          f"size=({bbox[1]-bbox[0]}, {bbox[3]-bbox[2]})")

y1 = max(b[0] for b in bboxes.values())
y2 = min(b[1] for b in bboxes.values())
x1 = max(b[2] for b in bboxes.values())
x2 = min(b[3] for b in bboxes.values())
print(f"\n  Intersection: y=[{y1}, {y2}] x=[{x1}, {x2}]  size=({y2-y1}, {x2-x1})")

cropped = {}
for name, img in all_channels.items():
    cropped[name] = crop_to_bbox(img, (y1, y2, x1, x2))

print(f"\n  Cropped sizes:")
for name, img in cropped.items():
    print(f"    {name:5s}: {img.shape}")


# =====================================================================
# Step 6: 保存
# =====================================================================

print("\n" + "=" * 60)
print("Step 6: Saving")
print("=" * 60)

final_channels = ["DAPI", "HER2", "PR", "ER", "KI67"]
file_map = {
    "DAPI":  f"{BLOCK}_Cycle1_DAPI.tif",
    "HER2":  f"{BLOCK}_Cycle1_HER2_aligned.tif",
    "PR":    f"{BLOCK}_Cycle1_PR_aligned.tif",
    "ER":    f"{BLOCK}_Cycle1_ER_aligned.tif",
    "KI67":  f"{BLOCK}_KI67_aligned.tif",
}

for name in final_channels:
    out_path = OUTPUT_DIR / file_map[name]
    save_tiff(out_path, cropped[name])
    print(f"  {name:5s} -> {out_path.name}  {cropped[name].shape}")

merged = np.stack([cropped[ch] for ch in final_channels], axis=0)
merged_path = OUTPUT_DIR / f"{BLOCK}_merged_5channel.tif"
tifffile.imwrite(str(merged_path), merged.astype(np.uint16), imagej=True)
print(f"\n  Merged -> {merged_path.name}  shape={merged.shape}")
print(f"  Channel order: {', '.join(final_channels)}")


# =====================================================================
# Step 7: 可视化
# =====================================================================

print("\n" + "=" * 60)
print("Step 7: Visualization")
print("=" * 60)

dapi_crop = cropped["DAPI"]
dapi_n = norm(dapi_crop)
h, w = dapi_n.shape

# --- 7a: DAPI 配准对比 ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"{BLOCK} Cycle1 vs Cycle2 DAPI Registration", fontsize=14)

dapi2_n_crop = norm(dapi2[:h, :w])
overlay_before = np.zeros((h, w, 3))
overlay_before[:, :, 0] = dapi_n
overlay_before[:, :, 1] = dapi2_n_crop
axes[0].imshow(overlay_before)
axes[0].set_title("BEFORE (unregistered)")

dapi2_aligned_n = norm(cropped["DAPI2"])
overlay_after = np.zeros((h, w, 3))
overlay_after[:, :, 0] = dapi_n
overlay_after[:, :, 1] = dapi2_aligned_n
axes[1].imshow(overlay_after)
axes[1].set_title(f"AFTER (angle={angle:.2f}deg, shift=({shift[0]:.1f},{shift[1]:.1f}))")

axes[2].imshow(dapi_n, cmap='gray')
axes[2].set_title("Cycle1 DAPI (reference)")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"{BLOCK}_dapi_registration.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {BLOCK}_dapi_registration.png")

# --- 7b: 多标记叠加 ---
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(f"{BLOCK} Multi-marker Overlay", fontsize=14)

o1 = np.zeros((h, w, 3))
o1[:, :, 2] = dapi_n * 0.6
o1[:, :, 1] = norm(cropped["HER2"])
axes[0].imshow(o1)
axes[0].set_title("DAPI (blue) + HER2 (green)")

o2 = np.zeros((h, w, 3))
o2[:, :, 2] = dapi_n * 0.6
o2[:, :, 1] = norm(cropped["ER"])
o2[:, :, 0] = norm(cropped["PR"])
axes[1].imshow(o2)
axes[1].set_title("DAPI (blue) + ER (green) + PR (red)")

o3 = np.zeros((h, w, 3))
o3[:, :, 2] = dapi_n * 0.6
o3[:, :, 0] = norm(cropped["KI67"])
axes[2].imshow(o3)
axes[2].set_title("DAPI (blue) + Ki67 (red)")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"{BLOCK}_multi_marker_overlay.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {BLOCK}_multi_marker_overlay.png")

# --- 7c: 每通道 Before/After 对比 ---
fig, axes = plt.subplots(5, 2, figsize=(12, 24))
fig.suptitle(f"{BLOCK} Before vs After Registration", fontsize=16)

row = 0

# Row 0: DAPI (Cycle1 vs Cycle2)
dapi2_n_raw = norm(dapi2[:h, :w])
dapi2_n_aligned = norm(cropped["DAPI2"])

ov = np.zeros((h, w, 3))
ov[:, :, 0] = dapi_n
ov[:, :, 1] = dapi2_n_raw
axes[row, 0].imshow(ov)
axes[row, 0].set_title(f"DAPI Before\nNCC={compute_ncc(dapi_n, dapi2_n_raw):.4f}", fontsize=11)

ov = np.zeros((h, w, 3))
ov[:, :, 0] = dapi_n
ov[:, :, 1] = dapi2_n_aligned
axes[row, 1].imshow(ov)
axes[row, 1].set_title(f"DAPI After\nNCC={compute_ncc(dapi_n, dapi2_n_aligned):.4f}", fontsize=11)

row += 1

# Row 1-3: HER2, PR, ER (Cycle1 internal)
cycle1_before = {"HER2": her2, "PR": pr, "ER": er}

for name in ["HER2", "PR", "ER"]:
    img_before_raw = cycle1_before[name]
    img_after_raw = cropped[name]

    h_ch = min(dapi1.shape[0], img_before_raw.shape[0], img_after_raw.shape[0])
    w_ch = min(dapi1.shape[1], img_before_raw.shape[1], img_after_raw.shape[1])

    d_ref = norm(dapi1[:h_ch, :w_ch])
    n_before = norm(img_before_raw[:h_ch, :w_ch])
    n_after = norm(img_after_raw[:h_ch, :w_ch])

    ncc_before = compute_ncc(d_ref, n_before)
    ncc_after = compute_ncc(d_ref, n_after)

    ov = np.zeros((h_ch, w_ch, 3))
    ov[:, :, 0] = d_ref
    ov[:, :, 1] = n_before
    axes[row, 0].imshow(ov)
    axes[row, 0].set_title(f"{name} Before\nNCC={ncc_before:.4f}", fontsize=11)

    ov = np.zeros((h_ch, w_ch, 3))
    ov[:, :, 0] = d_ref
    ov[:, :, 1] = n_after
    axes[row, 1].imshow(ov)
    axes[row, 1].set_title(f"{name} After\nNCC={ncc_after:.4f}", fontsize=11)

    row += 1

# Row 4: Ki67 (Cycle2 -> Cycle1)
ki67_before_raw = ki67[:h, :w]
ki67_after_raw = cropped["KI67"]

h_k = min(h, ki67_before_raw.shape[0], ki67_after_raw.shape[0])
w_k = min(w, ki67_before_raw.shape[1], ki67_after_raw.shape[1])

d_ref = norm(dapi1[:h_k, :w_k])
n_before = norm(ki67_before_raw[:h_k, :w_k])
n_after = norm(ki67_after_raw[:h_k, :w_k])

ncc_before = compute_ncc(d_ref, n_before)
ncc_after = compute_ncc(d_ref, n_after)

ov = np.zeros((h_k, w_k, 3))
ov[:, :, 0] = d_ref
ov[:, :, 1] = n_before
axes[row, 0].imshow(ov)
axes[row, 0].set_title(f"Ki67 Before\nNCC={ncc_before:.4f}", fontsize=11)

ov = np.zeros((h_k, w_k, 3))
ov[:, :, 0] = d_ref
ov[:, :, 1] = n_after
axes[row, 1].imshow(ov)
axes[row, 1].set_title(f"Ki67 After\nNCC={ncc_after:.4f}", fontsize=11)

# ====== 美化 ======
row_labels = ["DAPI", "HER2", "PR", "ER", "Ki67"]
for i, label in enumerate(row_labels):
    axes[i, 0].text(-0.05, 0.5, label, transform=axes[i, 0].transAxes,
                     fontsize=14, fontweight='bold',
                     va='center', ha='right', color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

fig.text(0.28, 0.98, "BEFORE", ha='center', fontsize=14, fontweight='bold', color='red')
fig.text(0.72, 0.98, "AFTER", ha='center', fontsize=14, fontweight='bold', color='green')

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout(rect=[0.08, 0, 1, 0.96])
plt.savefig(str(OUTPUT_DIR / f"{BLOCK}_before_after_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {BLOCK}_before_after_comparison.png")


# =====================================================================
# 汇总
# =====================================================================

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"  Block: {BLOCK}")
print(f"  Final size: ({y2-y1}, {x2-x1})")
print(f"  Cycle2 transform: rotate({angle:.2f}deg) + shift({shift[0]:.1f}, {shift[1]:.1f})")
print(f"  DAPI registration NCC: {ncc_final:.4f}")
print(f"  Output: {OUTPUT_DIR}")
print()
print(f"  Single-channel files:")
for name in final_channels:
    print(f"    {name:5s} -> {file_map[name]}")
print()
print(f"  Merged file:")
print(f"    {BLOCK}_merged_5channel.tif  (5 channels, Fiji ready)")
print()
print(f"  Visualization:")
print(f"    {BLOCK}_dapi_registration.png")
print(f"    {BLOCK}_multi_marker_overlay.png")
print(f"    {BLOCK}_before_after_comparison.png")
print()
print("Done!")
