"""
批量配准流程
  - TMAd: Cycle1 通道对齐 + Cycle2 DAPI/Ki67 配准
  - TMAe: Cycle1 通道对齐（无 Cycle2）
"""

import numpy as np
import tifffile
from pathlib import Path
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate
from scipy.ndimage import shift as ndshift
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
import warnings
import traceback

warnings.filterwarnings("ignore")


# =====================================================================
# 配置
# =====================================================================

BASE_DIR = Path(r"d:\Try_munan\FYP_LAST")
CROP_DIR = BASE_DIR / "results" / "crop"
OUTPUT_DIR = BASE_DIR / "results" / "registered"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE_TOLERANCE = 0.8


# =====================================================================
# 数据集定义
# =====================================================================

def discover_blocks():
    """自动发现所有数据集和 block"""

    datasets = {}

    # ----- TMAd -----
    tmad_dir = CROP_DIR / "TMAd"
    if tmad_dir.exists():
        for block_dir in sorted(tmad_dir.iterdir()):
            if not block_dir.is_dir():
                continue
            block = block_dir.name

            cycle1_dir = block_dir / f"{block}_TMAd_Cycle1"
            cycle2_dir = block_dir / f"{block}_TMAd_Cycle2"

            if not cycle1_dir.exists():
                continue

            # 检查 Cycle1 文件
            c1_files = {}
            for ch in ["DAPI", "HER2", "PR", "ER"]:
                p = cycle1_dir / f"{block}_TMAd_Cycle1_{ch}_crop.tif"
                if p.exists():
                    c1_files[ch] = p

            if "DAPI" not in c1_files:
                continue

            # 检查 Cycle2 文件
            c2_files = {}
            if cycle2_dir.exists():
                for ch in ["DAPI", "KI67"]:
                    p = cycle2_dir / f"{block}_TMAd_Cycle2_{ch}_crop.tif"
                    if p.exists():
                        c2_files[ch] = p

            datasets[f"TMAd/{block}"] = {
                "dataset": "TMAd",
                "block": block,
                "cycle1": c1_files,
                "cycle2": c2_files,
                "has_cycle2": len(c2_files) > 0,
            }

    # ----- TMAe -----
    tmae_dir = CROP_DIR / "TMAe"
    if tmae_dir.exists():
        for block_dir in sorted(tmae_dir.iterdir()):
            if not block_dir.is_dir():
                continue
            block = block_dir.name

            data_dir = block_dir / f"{block}_TMAe"
            if not data_dir.exists():
                continue

            c1_files = {}
            for ch in ["DAPI", "HER2", "ER", "PR"]:
                p = data_dir / f"{ch}_crop.tif"
                if p.exists():
                    c1_files[ch] = p

            if "DAPI" not in c1_files:
                continue

            datasets[f"TMAe/{block}"] = {
                "dataset": "TMAe",
                "block": block,
                "cycle1": c1_files,
                "cycle2": {},
                "has_cycle2": False,
            }

    return datasets


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
    x, y = a.ravel(), b.ravel()
    mask = (x > 0) | (y > 0)
    if mask.sum() < 100:
        return -1.0
    x, y = x[mask], y[mask]
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float(np.corrcoef(x, y)[0, 1])


def ncc_safe(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return compute_ncc(a[:h, :w], b[:h, :w])


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


def check_size_consistency(images_dict, tolerance=0.8):
    shapes = {name: img.shape for name, img in images_dict.items()}
    heights = [s[0] for s in shapes.values()]
    widths = [s[1] for s in shapes.values()]
    h_max = max(heights)
    w_max = max(widths)

    print(f"    Size check (threshold: {tolerance*100:.0f}%):")
    print(f"    {'Ch':>6s} {'H':>8s} {'W':>8s} {'H%':>6s} {'W%':>6s} {'Status':>8s}")
    print(f"    {'-'*44}")

    valid = {}
    skipped = {}

    for name, img in images_dict.items():
        h, w = img.shape
        h_ratio = h / h_max
        w_ratio = w / w_max

        if h_ratio < tolerance or w_ratio < tolerance:
            status = "SKIP"
            skipped[name] = {"shape": img.shape, "reason": f"H={h_ratio:.2f}, W={w_ratio:.2f}"}
        else:
            status = "OK"
            valid[name] = img

        print(f"    {name:>6s} {h:>8d} {w:>8d} {h_ratio:>6.2f} {w_ratio:>6.2f} {status:>8s}")

    if skipped:
        print(f"    [SKIP] Skipped: {list(skipped.keys())}")

    return valid, skipped


# =====================================================================
# 处理单个 block
# =====================================================================

def process_block(key, config):
    dataset = config["dataset"]
    block = config["block"]

    print("\n" + "=" * 60)
    print(f"Processing: {key}")
    print(f"  Dataset: {dataset}")
    print(f"  Cycle1 channels: {list(config['cycle1'].keys())}")
    print(f"  Cycle2 channels: {list(config['cycle2'].keys()) if config['has_cycle2'] else 'N/A'}")
    print("=" * 60)

    out_dir = OUTPUT_DIR / dataset / block
    out_dir.mkdir(parents=True, exist_ok=True)

    # 跳过已处理
    summary_file = out_dir / f"{block}_summary.txt"
    if summary_file.exists():
        print(f"  [SKIP] Already processed, skipping")
        return {"status": "skipped"}

    try:
        # ====== 加载 Cycle1 ======
        print(f"\n  [Step 1] Loading Cycle1 images...")
        cycle1_raw = {}
        for ch, path in config["cycle1"].items():
            cycle1_raw[ch] = load_tiff(path)
            print(f"    {ch}: {cycle1_raw[ch].shape}")

        # 尺寸检查
        valid_c1, skipped_c1 = check_size_consistency(cycle1_raw, tolerance=SIZE_TOLERANCE)

        if "DAPI" not in valid_c1:
            raise ValueError("DAPI not in valid channels, cannot continue")

        dapi = valid_c1["DAPI"]
        dapi_n = norm(dapi)

        # ====== Cycle1 内部对齐 ======
        print(f"\n  [Step 2] Cycle1 internal alignment...")
        aligned = {"DAPI": dapi}
        align_info = {}

        for ch in ["HER2", "PR", "ER"]:
            if ch not in valid_c1:
                continue
            img = valid_c1[ch]
            img_n = match_histograms(norm(img), dapi_n)
            ncc_before = ncc_safe(dapi_n, img_n)

            h = min(dapi_n.shape[0], img_n.shape[0])
            w = min(dapi_n.shape[1], img_n.shape[1])
            shift, _, _ = phase_cross_correlation(
                dapi_n[:h, :w], img_n[:h, :w], upsample_factor=100
            )

            img_aligned = ndshift(img, shift=(shift[0], shift[1]), order=1, cval=0)
            ncc_after = ncc_safe(dapi_n, norm(img_aligned))

            aligned[ch] = img_aligned
            align_info[ch] = {
                "shift": (float(shift[0]), float(shift[1])),
                "ncc_before": float(ncc_before),
                "ncc_after": float(ncc_after),
            }
            print(f"    {ch}: shift=({shift[0]:+.2f}, {shift[1]:+.2f})  NCC: {ncc_before:.4f} -> {ncc_after:.4f}")

        # ====== Cycle2 配准 ======
        cycle2_info = {}
        if config["has_cycle2"] and config["cycle2"]:
            print(f"\n  [Step 3] Cycle2 registration...")

            # 加载 Cycle2
            cycle2_raw = {}
            for ch, path in config["cycle2"].items():
                cycle2_raw[ch] = load_tiff(path)
                print(f"    {ch}: {cycle2_raw[ch].shape}")

            valid_c2, skipped_c2 = check_size_consistency(cycle2_raw, tolerance=SIZE_TOLERANCE)

            if "DAPI" in valid_c2:
                dapi2 = valid_c2["DAPI"]
                dapi2_n = match_histograms(norm(dapi2), dapi_n)

                # 旋转搜索
                print(f"    Searching rotation...")
                rotation_results = []
                for angle in np.arange(-5, 5.1, 0.25):
                    dapi2_rot = rotate(dapi2_n, angle, order=1, preserve_range=True)
                    h = min(dapi_n.shape[0], dapi2_rot.shape[0])
                    w = min(dapi_n.shape[1], dapi2_rot.shape[1])
                    s, _, _ = phase_cross_correlation(
                        dapi_n[:h, :w], dapi2_rot[:h, :w], upsample_factor=100
                    )
                    shifted = ndshift(dapi2_rot[:h, :w], shift=s, order=1)
                    score = compute_ncc(dapi_n[:h, :w], shifted)
                    rotation_results.append((angle, s[0], s[1], score))

                rotation_results.sort(key=lambda x: x[3], reverse=True)
                best_angle, best_dy, best_dx, best_ncc = rotation_results[0]

                # 应用变换
                dapi2_aligned = ndshift(
                    rotate(dapi2, best_angle, order=1, preserve_range=True),
                    shift=(best_dy, best_dx), order=1, cval=0
                )
                ncc_final = ncc_safe(norm(dapi), norm(dapi2_aligned))

                aligned["DAPI2"] = dapi2_aligned
                cycle2_info = {
                    "angle": float(best_angle),
                    "shift": (float(best_dy), float(best_dx)),
                    "ncc": float(ncc_final),
                }
                print(f"    Best: angle={best_angle:.2f}, shift=({best_dy:.2f}, {best_dx:.2f}), NCC={ncc_final:.4f}")

                # 同一个变换应用到其他 Cycle2 通道
                for ch in ["KI67"]:
                    if ch in valid_c2:
                        img = valid_c2[ch]
                        img_aligned = ndshift(
                            rotate(img, best_angle, order=1, preserve_range=True),
                            shift=(best_dy, best_dx), order=1, cval=0
                        )
                        aligned[ch] = img_aligned
                        print(f"    {ch}: same transform applied")
            else:
                print(f"    [WARN] Cycle2 DAPI not valid, skipping Cycle2")
        else:
            print(f"\n  [Step 3] No Cycle2 data, skipping")

        # ====== 统一尺寸 ======
        print(f"\n  [Step 4] Auto-size...")
        bboxes = {}
        for name, img in aligned.items():
            bboxes[name] = find_content_bbox(img)

        y1 = max(b[0] for b in bboxes.values())
        y2 = min(b[1] for b in bboxes.values())
        x1 = max(b[2] for b in bboxes.values())
        x2 = min(b[3] for b in bboxes.values())

        if y1 >= y2 or x1 >= x2:
            raise ValueError("No intersection region")

        cropped = {}
        for name, img in aligned.items():
            cropped[name] = crop_to_bbox(img, (y1, y2, x1, x2))

        print(f"    Final size: ({y2-y1}, {x2-x1})")

        # ====== 保存 ======
        print(f"\n  [Step 5] Saving...")

        # 确定输出通道
        output_channels = [ch for ch in ["DAPI", "HER2", "PR", "ER", "KI67"] if ch in cropped]

        file_map = {
            "DAPI": f"{block}_DAPI.tif",
            "HER2": f"{block}_HER2_aligned.tif",
            "PR":   f"{block}_PR_aligned.tif",
            "ER":   f"{block}_ER_aligned.tif",
            "KI67": f"{block}_KI67_aligned.tif",
        }

        for ch in output_channels:
            out_path = out_dir / file_map[ch]
            save_tiff(out_path, cropped[ch])
            print(f"    {ch:5s} -> {out_path.name}")

        # 合并 TIFF
        if len(output_channels) >= 2:
            merged = np.stack([cropped[ch] for ch in output_channels], axis=0)
            merged_name = f"{block}_merged_{len(output_channels)}channel.tif"
            merged_path = out_dir / merged_name
            tifffile.imwrite(str(merged_path), merged.astype(np.uint16), imagej=True)
            print(f"    Merged -> {merged_name}  channels={output_channels}")

        # ====== 可视化 ======
        print(f"\n  [Step 6] Visualization...")
        dapi_n = norm(cropped["DAPI"])
        h, w = dapi_n.shape

        # 多标记叠加
        markers = [ch for ch in output_channels if ch != "DAPI"]
        if markers:
            n = len(markers)
            fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
            if n == 1:
                axes = [axes]
            fig.suptitle(f"{key}", fontsize=14)

            for i, ch in enumerate(markers):
                overlay = np.zeros((h, w, 3))
                overlay[:, :, 2] = dapi_n * 0.6
                if ch == "HER2":
                    overlay[:, :, 1] = norm(cropped[ch])
                elif ch == "PR":
                    overlay[:, :, 0] = norm(cropped[ch])
                elif ch == "ER":
                    overlay[:, :, 1] = norm(cropped[ch])
                elif ch == "KI67":
                    overlay[:, :, 0] = norm(cropped[ch])

                axes[i].imshow(overlay)
                axes[i].set_title(f"DAPI + {ch}")
                axes[i].axis("off")

            plt.tight_layout()
            plt.savefig(str(out_dir / f"{block}_overlay.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    Saved: {block}_overlay.png")

        # ====== 保存摘要 ======
        with open(summary_file, "w") as f:
            f.write(f"Block: {key}\n")
            f.write(f"Final size: ({y2-y1}, {x2-x1})\n")
            f.write(f"Output channels: {output_channels}\n")
            f.write(f"\nCycle1 alignment:\n")
            for ch, info in align_info.items():
                f.write(f"  {ch}: shift=({info['shift'][0]:.2f}, {info['shift'][1]:.2f}) "
                        f"NCC: {info['ncc_before']:.4f} -> {info['ncc_after']:.4f}\n")
            if cycle2_info:
                f.write(f"\nCycle2 registration:\n")
                f.write(f"  angle={cycle2_info['angle']:.2f}, "
                        f"shift=({cycle2_info['shift'][0]:.2f}, {cycle2_info['shift'][1]:.2f}), "
                        f"NCC={cycle2_info['ncc']:.4f}\n")
            if skipped_c1:
                f.write(f"\nSkipped Cycle1 channels: {list(skipped_c1.keys())}\n")

        print(f"\n  [OK] Done: {key}")
        return {
            "status": "success",
            "output_channels": output_channels,
            "final_size": (y2 - y1, x2 - x1),
        }

    except Exception as e:
        print(f"\n  [ERROR] Error: {e}")
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


# =====================================================================
# 主流程
# =====================================================================

def main():
    print("=" * 60)
    print("Batch Registration Pipeline")
    print("=" * 60)

    # 发现所有 block
    datasets = discover_blocks()

    print(f"\nFound {len(datasets)} blocks:")
    for key, config in datasets.items():
        c2_status = "with Cycle2" if config["has_cycle2"] else "no Cycle2"
        print(f"  {key:20s} Cycle1: {list(config['cycle1'].keys())}  {c2_status}")

    if not datasets:
        print("No blocks found!")
        return

    # 逐个处理
    results = {}
    for key, config in datasets.items():
        result = process_block(key, config)
        results[key] = result

    # 汇总
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)

    success = [k for k, v in results.items() if v.get("status") == "success"]
    skipped = [k for k, v in results.items() if v.get("status") == "skipped"]
    failed  = [k for k, v in results.items() if v.get("status") == "error"]

    print(f"  [OK] Success: {len(success)}")
    for k in success:
        r = results[k]
        print(f"      {k}: {r['output_channels']}  size={r['final_size']}")

    if skipped:
        print(f"  [SKIP] Skipped: {len(skipped)}")
        for k in skipped:
            print(f"      {k}")

    if failed:
        print(f"  [ERROR] Failed: {len(failed)}")
        for k in failed:
            print(f"      {k}: {results[k].get('error', 'Unknown')}")

    print(f"\n  Output: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
