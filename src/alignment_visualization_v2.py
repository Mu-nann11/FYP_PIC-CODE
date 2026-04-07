import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tifffile import imread
import argparse


# ============================================================
# Core: normalization with background subtraction (replaces the earlier normalize_channel)
# ============================================================

def estimate_background(img, method='mode'):
    """
    Estimate the image background value automatically.
    """
    img_flat = img.ravel()

    if method == 'low':
        return np.percentile(img_flat, 1.0)

    elif method == 'mode':
        hist, edges = np.histogram(img_flat, bins=512)
        peak_idx = np.argmax(hist)
        return (edges[peak_idx] + edges[peak_idx + 1]) / 2

    elif method == 'corner':
        h, w = img.shape
        s = min(50, h // 10, w // 10)
        corners = np.concatenate([
            img[:s, :s].ravel(),
            img[:s, -s:].ravel(),
            img[-s:, :s].ravel(),
            img[-s:, -s:].ravel(),
        ])
        return np.median(corners)
    else:
        raise ValueError(f"Unknown method: {method}")


def normalize_channel(img, low_pct=1.0, high_pct=99.8, subtract_bg=True, bg_method='percentile'):
    """
    Simulate QuPath/Fiji brightness/contrast adjustment.
    """
    img = img.astype(np.float64)

    if subtract_bg:
        if bg_method == 'mode':
            bg = estimate_background(img, 'mode')
        else:
            bg = np.percentile(img, low_pct)
            
        signal_max = np.percentile(img, high_pct)
        # Key: subtract the background baseline so the film disappears.
        img = img - bg
        if signal_max - bg < 1:
            return np.zeros_like(img)
        img = img / (signal_max - bg)
    else:
        vmin = np.percentile(img, low_pct)
        vmax = np.percentile(img, high_pct)
        if vmax - vmin < 1:
            return np.zeros_like(img)
        img = (img - vmin) / (vmax - vmin)

    return np.clip(img, 0, 1)


# ============================================================
# Channel compositing
# ============================================================

def composite_dapi_ki67(dapi, ki67,
                        dapi_low=1.0, dapi_high=99.8,
                        ki67_low=2.0, ki67_high=99.5,
                        ki67_gamma=1.0,
                        ki67_bg_method='percentile'):
    """
    DAPI (blue) + Ki67 (magenta) composite.
    """
    d = normalize_channel(dapi, low_pct=dapi_low, high_pct=dapi_high)
    k = normalize_channel(ki67, low_pct=ki67_low, high_pct=ki67_high, bg_method=ki67_bg_method)

    if ki67_gamma != 1.0:
        k = np.power(k, ki67_gamma)

    rgb = np.zeros((*d.shape, 3), dtype=np.float64)
    rgb[..., 0] = k                 # R = Ki67
    rgb[..., 1] = 0.0               # G = empty
    rgb[..., 2] = d * 0.85 + k * 0.3  # B = DAPI-dominant

    return np.clip(rgb, 0, 1)


def composite_dapi_er(dapi, er,
                      dapi_low=1.0, dapi_high=99.8,
                      er_low=2.0, er_high=99.5):
    """
    DAPI (blue) + ER (green) composite.
    """
    d = normalize_channel(dapi, low_pct=dapi_low, high_pct=dapi_high)
    e = normalize_channel(er, low_pct=er_low, high_pct=er_high)

    rgb = np.zeros((*d.shape, 3), dtype=np.float64)
    rgb[..., 0] = 0.0
    rgb[..., 1] = e
    rgb[..., 2] = d * 0.85 + e * 0.2

    return np.clip(rgb, 0, 1)


# ============================================================
# Visualization
# ============================================================

def make_comparison(
    dapi_before, ki67_before,
    dapi_after, ki67_after,
    save_path=None,
    title="Ki67 × DAPI Alignment",
    crop=None,
    dapi_low=1.0, dapi_high=99.8,
    ki67_low=2.0, ki67_high=99.5,
    ki67_gamma=1.0,
    ki67_bg_method='percentile',
    dpi=200,
):
    """
    Before / After comparison figure.
    """
    # Avoid mismatched before/after channel sizes (Cycle 1/2 can differ slightly before alignment).
    min_h_b = min(dapi_before.shape[0], ki67_before.shape[0])
    min_w_b = min(dapi_before.shape[1], ki67_before.shape[1])
    dapi_before = dapi_before[:min_h_b, :min_w_b]
    ki67_before = ki67_before[:min_h_b, :min_w_b]

    min_h_a = min(dapi_after.shape[0], ki67_after.shape[0])
    min_w_a = min(dapi_after.shape[1], ki67_after.shape[1])
    dapi_after = dapi_after[:min_h_a, :min_w_a]
    ki67_after = ki67_after[:min_h_a, :min_w_a]

    if crop is not None:
        y, x, h, w = crop
        dapi_before = dapi_before[y:y+h, x:x+w]
        ki67_before = ki67_before[y:y+h, x:x+w]
        dapi_after  = dapi_after[y:y+h, x:x+w]
        ki67_after  = ki67_after[y:y+h, x:x+w]

    comp_before = composite_dapi_ki67(
        dapi_before, ki67_before,
        dapi_low=dapi_low, dapi_high=dapi_high,
        ki67_low=ki67_low, ki67_high=ki67_high,
        ki67_gamma=ki67_gamma,
        ki67_bg_method=ki67_bg_method,
    )
    comp_after = composite_dapi_ki67(
        dapi_after, ki67_after,
        dapi_low=dapi_low, dapi_high=dapi_high,
        ki67_low=ki67_low, ki67_high=ki67_high,
        ki67_gamma=ki67_gamma,
        ki67_bg_method=ki67_bg_method,
    )

    # ---- Plotting ----
    fig = plt.figure(figsize=(18, 8.5), facecolor='#0a0a0a')
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.03,
                  left=0.03, right=0.97, top=0.90, bottom=0.06)

    ax_b = fig.add_subplot(gs[0])
    ax_a = fig.add_subplot(gs[1])
    cax  = fig.add_subplot(gs[2])

    for ax, img, label in [
        (ax_b, comp_before, "Before (raw, no alignment)"),
        (ax_a, comp_after,  "After (aligned)"),
    ]:
        ax.imshow(img, interpolation='nearest')
        ax.set_title(label, color='white', fontsize=14,
                     fontweight='bold', pad=10)
        ax.axis('off')

    # Legend
    _draw_legend(cax)

    # Parameter info
    fig.suptitle(title, color='white', fontsize=16, fontweight='bold', y=0.96)

    # Bottom parameter label
    param_text = (f"DAPI: pct=[{dapi_low}, {dapi_high}]  |  "
                  f"Ki67: bg={ki67_bg_method}, high_pct={ki67_high}  |  "
                  f"Ki67 γ={ki67_gamma}")
    fig.text(0.5, 0.01, param_text, color='#666666', fontsize=9,
             ha='center', va='bottom', family='monospace')

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
        plt.close(fig)


def _draw_legend(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    items = [
        (3.5, (0, 0, 1),       'DAPI (nuclei)'),
        (2.5, (1, 0, 0),       'Ki67 (proliferation)'),
        (1.5, (1, 0, 0.45),    'Overlap (Ki67⁺ nucleus)'),
        (0.5, (0.03, 0, 0.06), 'Background'),
    ]
    for y, color, label in items:
        ax.add_patch(plt.Rectangle((0.05, y), 0.35, 0.6,
                     facecolor=color, edgecolor='#444', linewidth=0.5))
        ax.text(0.55, y + 0.3, label, color='white', fontsize=10,
                va='center', ha='left')


# ============================================================
# Debug tool: help find the best percentile parameters
# ============================================================

def debug_channel(img, channel_name="channel", save_path=None):
    """
    Plot a single-channel histogram and previews with different percentile cuts.
    """
    bg_mode = estimate_background(img, 'mode')
    bg_low  = estimate_background(img, 'low')
    bg_corner = estimate_background(img, 'corner')

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='#0a0a0a')
    fig.suptitle(f"Debug: {channel_name}", color='white',
                 fontsize=14, fontweight='bold')

    # 直方图
    ax = axes[0, 0]
    ax.hist(img.ravel(), bins=512, color='#4a9eff', alpha=0.8,
            edgecolor='none')
    ax.axvline(bg_mode, color='#ff4444', linewidth=2, label=f'mode={bg_mode:.0f}')
    ax.axvline(bg_low, color='#44ff44', linewidth=2, label=f'p1={bg_low:.0f}')
    ax.axvline(bg_corner, color='#ffaa00', linewidth=2,
               label=f'corner={bg_corner:.0f}')
    ax.set_facecolor('#1a1a1a')
    ax.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333',
              labelcolor='white')
    ax.set_title('Histogram + BG estimates', color='white', fontsize=11)
    ax.tick_params(colors='white')

    configs = [
        ('pct [1, 99.8] (default)', 1.0, 99.8),
        ('pct [2, 99.5]',           2.0, 99.5),
        ('pct [5, 99.0]',           5.0, 99.0),
        ('pct [0.5, 99.9]',         0.5, 99.9),
        ('subtract mode BG',        None, None),
    ]

    for idx, (label, low, high) in enumerate(configs):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        ax = axes[row, col]

        if low is None:
            adjusted = img.astype(np.float64) - bg_mode
            adjusted = np.clip(adjusted / np.percentile(
                np.maximum(adjusted, 0), 99.8), 0, 1)
        else:
            adjusted = normalize_channel(img, low_pct=low, high_pct=high)

        ax.imshow(adjusted, cmap='gray', interpolation='nearest')
        ax.set_title(label, color='white', fontsize=10)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  Debug saved: {save_path}")
    else:
        plt.show()
        plt.close(fig)


# ============================================================
# 主入口
# ============================================================

def generate_for_block(dataset, block,
                       raw_root="Raw_Data",
                       aligned_root="Aligned_Results",
                       output_dir="alignment_vis",
                       dapi_low=1.0, dapi_high=99.8,
                       ki67_low=2.0, ki67_high=99.5,
                       ki67_gamma=1.0,
                       ki67_bg_method='percentile',
                       debug=False):
    """
    根据你的实际文件结构修改文件匹配逻辑
    """
    raw_dir_base = Path(raw_root) / dataset
    aligned_dir_base = Path(aligned_root) / dataset
    out_dir = Path(output_dir) / dataset / block

    # 全局搜索 block 的相关文件
    dapi_b_files = list(raw_dir_base.rglob(f"**/*{block}*dapi*.tif")) + list(raw_dir_base.rglob(f"**/*{block}*DAPI*.tif"))
    ki67_b_files = list(raw_dir_base.rglob(f"**/*{block}*ki67*.tif")) + list(raw_dir_base.rglob(f"**/*{block}*KI67*.tif"))
    dapi_a_files = list(aligned_dir_base.rglob(f"**/*{block}*dapi*.tif")) + list(aligned_dir_base.rglob(f"**/*{block}*DAPI*.tif"))
    ki67_a_files = list(aligned_dir_base.rglob(f"**/*{block}*ki67*.tif")) + list(aligned_dir_base.rglob(f"**/*{block}*KI67*.tif"))

    if not all([dapi_b_files, ki67_b_files, dapi_a_files, ki67_a_files]):
        print(f"  [SKIP] {dataset}/{block} - Missing required TIFF files")
        return

    dapi_before = imread(str(dapi_b_files[0]))
    ki67_before = imread(str(ki67_b_files[0]))
    dapi_after  = imread(str(dapi_a_files[0]))
    ki67_after  = imread(str(ki67_a_files[0]))

    if debug:
        debug_channel(dapi_before, "DAPI (before)",
                      save_path=out_dir / "debug_dapi.png")
        debug_channel(ki67_before, "Ki67 (before)",
                      save_path=out_dir / "debug_ki67.png")

    make_comparison(
        dapi_before, ki67_before, dapi_after, ki67_after,
        save_path=out_dir / f"{block}_full.png",
        title=f"{dataset} / {block} — Ki67 × DAPI",
        dapi_low=dapi_low, dapi_high=dapi_high,
        ki67_low=ki67_low, ki67_high=ki67_high,
        ki67_gamma=ki67_gamma,
        ki67_bg_method=ki67_bg_method,
    )

    h, w = dapi_before.shape
    ch, cw = min(512, h), min(512, w)
    cy, cx = h // 2 - ch // 2, w // 2 - cw // 2
    make_comparison(
        dapi_before, ki67_before, dapi_after, ki67_after,
        save_path=out_dir / f"{block}_crop.png",
        title=f"{dataset} / {block} — Ki67 × DAPI (Crop)",
        crop=(cy, cx, ch, cw),
        dapi_low=dapi_low, dapi_high=dapi_high,
        ki67_low=ki67_low, ki67_high=ki67_high,
        ki67_gamma=ki67_gamma,
        ki67_bg_method=ki67_bg_method,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--blocks", nargs="+", required=True)
    parser.add_argument("--raw-root", default="Raw_Data")
    if Path("results/registered").exists():
        parser.add_argument("--aligned-root", default="results/registered")
    else:
        parser.add_argument("--aligned-root", default="Aligned_Results")
    parser.add_argument("--output-dir", default="alignment_vis")
    parser.add_argument("--dapi-low", type=float, default=1.0)
    parser.add_argument("--dapi-high", type=float, default=99.8)
    parser.add_argument("--ki67-low", type=float, default=2.0)
    parser.add_argument("--ki67-high", type=float, default=99.5)
    parser.add_argument("--ki67-gamma", type=float, default=1.0)
    parser.add_argument("--ki67-bg-method", type=str, default="percentile", choices=["percentile", "mode"], help="Ki67 背景减除方法: percentile(默认) 或 mode(众数)")
    parser.add_argument("--debug", action="store_true", help="生成直方图+预览帮你选参数")
    args = parser.parse_args()

    for block in args.blocks:
        print(f"\n[{args.dataset}/{block}]")
        generate_for_block(
            dataset=args.dataset, block=block,
            raw_root=args.raw_root, aligned_root=args.aligned_root,
            output_dir=args.output_dir,
            dapi_low=args.dapi_low, dapi_high=args.dapi_high,
            ki67_low=args.ki67_low, ki67_high=args.ki67_high,
            ki67_gamma=args.ki67_gamma,
            ki67_bg_method=args.ki67_bg_method,
            debug=args.debug,
        )
    print("\nDone.")
