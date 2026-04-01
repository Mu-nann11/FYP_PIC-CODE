import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from tifffile import imread
import argparse
import cv2
from skimage.registration import phase_cross_correlation

def compute_ncc(a, b):
    min_h = min(a.shape[0], b.shape[0])
    min_w = min(a.shape[1], b.shape[1])
    a, b = a[:min_h, :min_w], b[:min_h, :min_w]
    mask = (a > 0) | (b > 0)
    if mask.sum() < 100: return 0.0
    va, vb = a[mask].astype(np.float64), b[mask].astype(np.float64)
    va = (va - va.mean()) / (va.std() + 1e-8)
    vb = (vb - vb.mean()) / (vb.std() + 1e-8)
    return float(np.corrcoef(va, vb)[0, 1])

def estimate_background(img, method='mode'):
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
            img[:s, :s].ravel(), img[:s, -s:].ravel(),
            img[-s:, :s].ravel(), img[-s:, -s:].ravel(),
        ])
        return np.median(corners)
    else:
        raise ValueError(f"Unknown method: {method}")

def normalize_channel(img, low_pct=1.0, high_pct=99.8, subtract_bg=True, bg_method='percentile'):
    img = img.astype(np.float64)
    if subtract_bg:
        if bg_method == 'mode':
            bg = estimate_background(img, 'mode')
        else:
            bg = np.percentile(img, low_pct)
        signal_max = np.percentile(img, high_pct)
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

def compute_ncc(a, b):
    a_m = a.astype(np.float32)
    b_m = b.astype(np.float32)
    mask = (a_m > 0) | (b_m > 0)
    if mask.sum() < 100: return 0.0
    a_m = a_m[mask]
    b_m = b_m[mask]
    std_a, std_b = a_m.std(), b_m.std()
    if std_a < 1e-8 or std_b < 1e-8: return 0.0
    a_m = (a_m - a_m.mean()) / std_a
    b_m = (b_m - b_m.mean()) / std_b
    return float(np.corrcoef(a_m, b_m)[0, 1])

def get_transformation_details(before_img, after_img):
    min_h = min(before_img.shape[0], after_img.shape[0])
    min_w = min(before_img.shape[1], after_img.shape[1])
    
    scale = 0.5
    b_small = cv2.resize(before_img[:min_h, :min_w], (0,0), fx=scale, fy=scale)
    a_small = cv2.resize(after_img[:min_h, :min_w], (0,0), fx=scale, fy=scale)
    
    b_f = np.float32(b_small)
    a_f = np.float32(a_small)
    shift, _ = cv2.phaseCorrelate(b_f, a_f)
    
    dx = -shift[0] / scale
    dy = -shift[1] / scale
    
    return f"dy={dy:.1f}, dx={dx:.1f}, rot~0.0\u00B0"

def composite_dapi_marker(dapi, marker_img,
                          dapi_low=1.0, dapi_high=99.8,
                          marker_low=2.0, marker_high=99.5,
                          marker_gamma=1.0,
                          marker_bg_method='percentile',
                          color_style='magenta'):
    d = normalize_channel(dapi, low_pct=dapi_low, high_pct=dapi_high)
    m = normalize_channel(marker_img, low_pct=marker_low, high_pct=marker_high, bg_method=marker_bg_method)

    if marker_gamma != 1.0:
        m = np.power(m, marker_gamma)

    rgb = np.zeros((*d.shape, 3), dtype=np.float64)
    if color_style == 'magenta':
        rgb[..., 0] = m                 
        rgb[..., 1] = 0.0               
        rgb[..., 2] = d * 0.85 + m * 0.3  
    elif color_style == 'green':
        rgb[..., 0] = 0.0
        rgb[..., 1] = m
        rgb[..., 2] = d * 0.85 + m * 0.2
    else: 
        rgb[..., 0] = m
        rgb[..., 1] = 0.0
        rgb[..., 2] = d * 0.85
        
    return np.clip(rgb, 0, 1)

def find_crop_in_before(dapi_before, dapi_after, crop_h, crop_w):
    h_a, w_a = dapi_after.shape
    ya = h_a // 2 - crop_h // 2
    xa = w_a // 2 - crop_w // 2
    crop_after = dapi_after[ya:ya+crop_h, xa:xa+crop_w]

    norm_before = cv2.normalize(dapi_before, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    norm_after_crop = cv2.normalize(crop_after, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    res = cv2.matchTemplate(norm_before, norm_after_crop, cv2.TM_CCOEFF_NORMED) 
    _, _, _, max_loc = cv2.minMaxLoc(res)
    xb, yb = max_loc

    return [(yb, xb, crop_h, crop_w), (ya, xa, crop_h, crop_w)]

def make_comparison(
    dapi_before, marker_before,
    dapi_after, marker_after,
    marker_name="Ki67",
    save_path=None,
    title="Alignment Comparison",
    crop_info=None, 
    dapi_low=1.0, dapi_high=99.8,
    marker_low=2.0, marker_high=99.5,
    marker_gamma=1.0,
    marker_bg_method='percentile',
    color_style='magenta',
    transform_text="",
    ncc_before=0.0,
    ncc_after=0.0,
    dpi=300,
):
    if crop_info is not None:
        (yb, xb, hb, wb), (ya, xa, ha, wa) = crop_info
        d_before = dapi_before[yb:yb+hb, xb:xb+wb]
        m_before = marker_before[yb:yb+hb, xb:xb+wb]
        d_after  = dapi_after[ya:ya+ha, xa:xa+wa]
        m_after  = marker_after[ya:ya+ha, xa:xa+wa]
    else:
        min_h_b = min(dapi_before.shape[0], marker_before.shape[0])
        min_w_b = min(dapi_before.shape[1], marker_before.shape[1])
        d_before = dapi_before[:min_h_b, :min_w_b]
        m_before = marker_before[:min_h_b, :min_w_b]

        min_h_a = min(dapi_after.shape[0], marker_after.shape[0])
        min_w_a = min(dapi_after.shape[1], marker_after.shape[1])
        d_after = dapi_after[:min_h_a, :min_w_a]
        m_after = marker_after[:min_h_a, :min_w_a]

    comp_before = composite_dapi_marker(
        d_before, m_before,
        dapi_low=dapi_low, dapi_high=dapi_high,
        marker_low=marker_low, marker_high=marker_high,
        marker_gamma=marker_gamma,
        marker_bg_method=marker_bg_method,
        color_style=color_style
    )
    comp_after = composite_dapi_marker(
        d_after, m_after,
        dapi_low=dapi_low, dapi_high=dapi_high,
        marker_low=marker_low, marker_high=marker_high,
        marker_gamma=marker_gamma,
        marker_bg_method=marker_bg_method,
        color_style=color_style
    )

    fig = plt.figure(figsize=(18, 8.5), facecolor='white')
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.08], wspace=0.05,
                  left=0.03, right=0.97, top=0.88, bottom=0.10)

    ax_b = fig.add_subplot(gs[0])
    ax_a = fig.add_subplot(gs[1])
    cax  = fig.add_subplot(gs[2])

    for ax, img, label in [
        (ax_b, comp_before, "Before (Raw, Pre-alignment)"),
        (ax_a, comp_after,  "After (Registered)"),
    ]:
        ax.imshow(img, interpolation='nearest')
        ax.set_title(label, color='black', fontsize=16,
                     fontweight='bold', fontfamily='Arial', pad=12)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        ax.set_xticks([])
        ax.set_yticks([])

    _draw_legend(cax, marker_name, color_style)

    fig.suptitle(title, color='black', fontsize=20, fontweight='bold', fontfamily='Arial', y=0.97)

    param_text = (f"Processing: DAPI pct=[{dapi_low}, {dapi_high}] | "
                  f"{marker_name} bg={marker_bg_method}, high_pct={marker_high} | "
                  f"{marker_name} \u03b3={marker_gamma}\n"
                  f"Metrics: Before NCC = {ncc_before:.4f} | After NCC = {ncc_after:.4f} | "
                  f"Best Transform = [{transform_text}]")
    fig.text(0.5, 0.02, param_text, color='#444444', fontsize=12,
             ha='center', va='bottom', fontfamily='monospace', fontweight='bold')

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, facecolor='white',
                    bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
        plt.close(fig)

def _draw_legend(ax, marker_name, color_style):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    marker_color = (1, 0, 0)
    overlap_color = (1, 0, 0.45)
    if color_style == 'green':
        marker_color = (0, 1, 0)
        overlap_color = (0.2, 0.8, 0.5)

    items = [
        (3.5, (0, 0, 1),       'DAPI\n(Nuclei)'),
        (2.5, marker_color,    f'{marker_name}\n(Signal)'),
        (1.5, overlap_color,   f'Overlap\n(Coloc+)'),
        (0.5, (0.1, 0.1, 0.1), 'Background'),
    ]
    for y, color, label in items:
        ax.add_patch(plt.Rectangle((0.05, y), 0.25, 0.6,
                     facecolor=color, edgecolor='black', linewidth=1.5))
        ax.text(0.40, y + 0.3, label, color='black', fontsize=11,
                fontfamily='Arial', fontweight='bold', va='center', ha='left')

def generate_for_block(dataset, block,
                       raw_root="Raw_Data",
                       aligned_root="Aligned_Results",
                       output_dir="results/alignment_vis_paper",
                       dapi_low=1.0, dapi_high=99.8,
                       marker_low=2.0, marker_high=99.5,
                       marker_gamma=1.0,
                       marker_bg_method='mode'):
    
    raw_dir_base = Path(raw_root) / dataset
    aligned_dir_base = Path(aligned_root) / dataset
    out_dir = Path(output_dir) / dataset / block

    markers = ['KI67', 'ER', 'PR', 'HER2']
    color_map = {'KI67': 'magenta', 'ER': 'green', 'PR': 'magenta', 'HER2': 'green'}

    dapi_b_files = list(raw_dir_base.rglob(f"**/*{block}*dapi*.tif")) + list(raw_dir_base.rglob(f"**/*{block}*DAPI*.tif"))
    dapi_a_files = list(aligned_dir_base.rglob(f"**/*{block}*dapi*.tif")) + list(aligned_dir_base.rglob(f"**/*{block}*DAPI*.tif"))

    if not dapi_b_files or not dapi_a_files:
        print(f"  [SKIP] {dataset}/{block} - Missing DAPI")
        return

    dapi_before = imread(str(dapi_b_files[0]))
    dapi_after  = imread(str(dapi_a_files[0]))

    for marker in markers:
        m_b_files = list(raw_dir_base.rglob(f"**/*{block}*{marker.lower()}*.tif")) + list(raw_dir_base.rglob(f"**/*{block}*{marker.upper()}*.tif"))
        m_a_files = list(aligned_dir_base.rglob(f"**/*{block}*{marker.lower()}*.tif")) + list(aligned_dir_base.rglob(f"**/*{block}*{marker.upper()}*.tif"))
        
        if not m_b_files or not m_a_files:
            continue
            
        print(f"  Processing {marker}...")
        marker_before = imread(str(m_b_files[0]))
        marker_after  = imread(str(m_a_files[0]))
        
        c_style = color_map.get(marker, 'magenta')

        # 1. NCCO
        min_h_b = min(dapi_before.shape[0], marker_before.shape[0])
        min_w_b = min(dapi_before.shape[1], marker_before.shape[1])
        nc_b = compute_ncc(dapi_before[:min_h_b, :min_w_b], marker_before[:min_h_b, :min_w_b])
        
        min_h_a = min(dapi_after.shape[0], marker_after.shape[0])
        min_w_a = min(dapi_after.shape[1], marker_after.shape[1])
        nc_a = compute_ncc(dapi_after[:min_h_a, :min_w_a], marker_after[:min_h_a, :min_w_a])

        # 2. Transform details
        transform_text = get_transformation_details(marker_before, marker_after)

        make_comparison(
            dapi_before, marker_before, dapi_after, marker_after,
            marker_name=marker,
            save_path=out_dir / f"{block}_{marker}_full.png",
            title=f"{dataset} / {block} \u2014 {marker} \u00D7 DAPI Alignment",
            dapi_low=dapi_low, dapi_high=dapi_high,
            marker_low=marker_low, marker_high=marker_high,
            marker_gamma=marker_gamma,
            marker_bg_method=marker_bg_method,
            color_style=c_style,
            transform_text=transform_text,
            ncc_before=nc_b, ncc_after=nc_a
        )

        h, w = dapi_before.shape
        ch, cw = min(512, h), min(512, w)
        crop_info = find_crop_in_before(dapi_before, dapi_after, ch, cw)

        make_comparison(
            dapi_before, marker_before, dapi_after, marker_after,
            marker_name=marker,
            save_path=out_dir / f"{block}_{marker}_crop.png",
            title=f"{dataset} / {block} \u2014 {marker} \u00D7 DAPI (Crop Detailed View)",
            crop_info=crop_info,
            dapi_low=dapi_low, dapi_high=dapi_high,
            marker_low=marker_low, marker_high=marker_high,
            marker_gamma=marker_gamma,
            marker_bg_method=marker_bg_method,
            color_style=c_style,
            transform_text=transform_text,
            ncc_before=nc_b, ncc_after=nc_a
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--blocks", nargs="+", required=True)
    parser.add_argument("--raw-root", default="results/stitched")
    parser.add_argument("--aligned-root", default="results/registered")
    parser.add_argument("--output-dir", default="results/alignment_vis_paper")
    parser.add_argument("--marker-bg-method", type=str, default="mode", choices=["percentile", "mode"])
    args = parser.parse_args()

    for block in args.blocks:
        print(f"\n[{args.dataset}/{block}] Paper Mode")
        generate_for_block(
            dataset=args.dataset, block=block,
            raw_root=args.raw_root, aligned_root=args.aligned_root,
            output_dir=args.output_dir,
            marker_bg_method=args.marker_bg_method
        )
    print("\nDone.")