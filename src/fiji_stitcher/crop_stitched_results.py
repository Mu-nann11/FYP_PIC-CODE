from pathlib import Path
import re
import tifffile
import numpy as np
import cv2

CHANNELS = ["DAPI", "HER2", "PR", "ER", "KI67"]


def _ensure_uint16(arr: np.ndarray) -> np.ndarray:
    """将图像强制转换为 uint16，保持数值范围。"""
    if arr.dtype == np.uint16:
        return arr
    if not np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.uint16)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= 0:
        return np.zeros(arr.shape, dtype=np.uint16)
    norm = (arr.astype(np.float64) - mn) / (mx - mn)
    return (norm * 65535.0).clip(0, 65535).astype(np.uint16)


def find_channel_files(block_dir: Path):
    """根据目录结构智能匹配通道文件"""
    paths = {}
    all_tifs = list(block_dir.glob("*.tif")) + list(block_dir.glob("*.tiff"))
    
    for ch in CHANNELS:
        matched = []
        for p in all_tifs:
            name = p.name.upper()
            if ch.upper() in name:
                matched.append(p)
        
        if matched:
            paths[ch] = sorted(matched, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    
    return paths


def crop_one_block(block_dir: Path, crop_root: Path, block_name: str, cycle_suffix: str, margin: int = 20):
    """
    Crop one block's channels.
    
    输出目录: {crop_root}/{block_name}/{block_name}_{dataset}_{cycle_suffix}/
    输出文件: {block_name}_{dataset}_{cycle_suffix}_{channel}_crop.tif
    """
    # 输出目录: TMAd/A3/A3_TMAd_Cycle1/
    out_dir = crop_root / block_name / f"{block_name}_{cycle_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    paths = find_channel_files(block_dir)
    
    if not paths:
        print(f"[{block_name}] 未找到任何通道文件，跳过该块")
        return None

    for ch in CHANNELS:
        if ch not in paths:
            print(f"[{block_name}] 通道 {ch} 未找到")

    sizes = {}
    arrays = {}
    for ch, p in paths.items():
        arr = tifffile.imread(str(p))
        arrays[ch] = arr
        h, w = arr.shape[:2]
        sizes[ch] = (h, w, str(arr.dtype))

    min_h = min(v[0] for v in sizes.values())
    min_w = min(v[1] for v in sizes.values())

    crop_h = max(min_h - 2 * margin, 1)
    crop_w = max(min_w - 2 * margin, 1)

    y0, x0 = margin, margin
    y1, x1 = y0 + crop_h, x0 + crop_w

    print(f"[{block_name}] 原始尺寸: {sizes}")
    print(f"[{block_name}] 裁剪窗口: x={x0}:{x1}, y={y0}:{y1} -> {crop_w}×{crop_h}")

    for ch, arr in arrays.items():
        h, w = arr.shape[:2]
        if h < y1 or w < x1:
            print(f"[{block_name}] 通道 {ch} 尺寸过小 ({h}x{w})，无法按统一窗口裁剪，跳过该块")
            return None

    for ch, arr in arrays.items():
        cropped = arr[y0:y1, x0:x1]
        # 文件名: A3_TMAd_Cycle1_DAPI_crop.tif
        out_path = out_dir / f"{block_name}_{cycle_suffix}_{ch}_crop.tif"
        tifffile.imwrite(str(out_path), _ensure_uint16(cropped))
        print(f"[{block_name}] 已保存: {out_path}")

    return (y0, x0, y1, x1)


def crop_cycle2(cycle2_dir: Path, crop_root: Path, block_name: str, cycle_suffix: str, window):
    """Crop Cycle2 所有通道文件，使用与 Cycle1 相同的裁剪窗口。"""
    y0, x0, y1, x1 = window
    
    # 查找所有通道文件
    paths = find_channel_files(cycle2_dir)
    
    if not paths:
        # 如果没有找到通道文件（命名不符合预期），尝试找所有 tif
        all_tifs = list(cycle2_dir.glob("*.tif")) + list(cycle2_dir.glob("*.tiff"))
        for tif_path in all_tifs:
            if tif_path.is_file():
                ch = None
                for c in CHANNELS:
                    if c in tif_path.name.upper():
                        ch = c
                        break
                if ch:
                    paths[ch] = tif_path
    
    if not paths:
        print(f"[{cycle2_dir.name}] 未找到 Cycle2 tif 文件，跳过")
        return
    
    # 输出目录: TMAd/A3/A3_TMAd_Cycle2/
    out_dir = crop_root / block_name / f"{block_name}_{cycle_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for ch, tif_path in paths.items():
        arr = tifffile.imread(str(tif_path))
        h, w = arr.shape[-2], arr.shape[-1]
        
        if h < y1 or w < x1:
            print(f"[{cycle2_dir.name}] 通道 {ch} 尺寸过小 ({h}x{w})，使用 resize 对齐")
            target_h, target_w = y1 - y0, x1 - x0
            if arr.ndim == 2:
                arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                cropped = arr
            else:
                cropped = np.stack([
                    cv2.resize(arr[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    for i in range(arr.shape[0])
                ])
        else:
            cropped = arr[..., y0:y1, x0:x1] if arr.ndim > 2 else arr[y0:y1, x0:x1]
        
        out_path = out_dir / f"{block_name}_{cycle_suffix}_{ch}_crop.tif"
        tifffile.imwrite(str(out_path), _ensure_uint16(cropped))
        print(f"[{cycle2_dir.name}] 通道 {ch} 已保存: {out_path}")


def crop_all_blocks(config):
    stitched_root = Path(config["CROP_INPUT_DIR"])
    crop_root = Path(config["CROP_OUTPUT_DIR"])
    margin = int(config.get("CROP_MARGIN", 20))

    crop_root.mkdir(parents=True, exist_ok=True)
    skip_names = {"logs", "log", "crop", "crop_result", "stitched_result", "stitched_results", 
                  "Cycle1", "Cycle2", "cycle1", "cycle2", "DAPI", "HER2", "PR", "ER", "KI67"}

    print(f"[裁剪] 输入目录: {stitched_root}")
    print(f"[裁剪] 输出目录: {crop_root}")
    print(f"[裁剪] 边距: {margin}")

    blocks_to_process = []  # [(dataset_name, block_name, has_cycle1_cycle2)]
    
    for dataset_dir in sorted(stitched_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        cycle_dirs = {}
        for sub in sorted(dataset_dir.iterdir()):
            if not sub.is_dir():
                continue
            sub_name_lower = sub.name.lower()
            if 'cycle1' in sub_name_lower:
                cycle_dirs['cycle1'] = sub
            elif 'cycle2' in sub_name_lower:
                cycle_dirs['cycle2'] = sub
        
        if cycle_dirs:
            sample_cycle_dir = cycle_dirs.get('cycle1') or cycle_dirs.get('cycle2')
            for block_dir in sorted(sample_cycle_dir.iterdir()):
                if not block_dir.is_dir():
                    continue
                if block_dir.name.lower() in skip_names:
                    continue
                
                block_tifs = list(block_dir.glob("*.tif")) + list(block_dir.glob("*.tiff"))
                if block_tifs:
                    blocks_to_process.append((dataset_dir.name, block_dir.name, True))
                    print(f"[发现] 数据集 {dataset_dir.name}, 块 {block_dir.name} (有 Cycle 结构)")
        else:
            for block_dir in sorted(dataset_dir.iterdir()):
                if not block_dir.is_dir():
                    continue
                if block_dir.name.lower() in skip_names:
                    continue
                
                block_tifs = list(block_dir.glob("*.tif")) + list(block_dir.glob("*.tiff"))
                if block_tifs:
                    blocks_to_process.append((dataset_dir.name, block_dir.name, False))
                    print(f"[发现] 数据集 {dataset_dir.name}, 块 {block_dir.name} (普通结构)")

    print(f"[裁剪] 发现 {len(blocks_to_process)} 个数据块")
    
    for dataset_name, block_name, has_cycle in blocks_to_process:
        dataset_dir = stitched_root / dataset_name
        
        print(f"\n[处理] {dataset_name}/{block_name}")
        
        if has_cycle:
            cycle1_dir = dataset_dir / "Cycle1" / block_name
            cycle2_dir = dataset_dir / "Cycle2" / block_name
            
            if cycle1_dir.exists():
                cycle_suffix = f"{dataset_name}_Cycle1"
                window = crop_one_block(cycle1_dir, crop_root / dataset_name, block_name, cycle_suffix, margin=margin)
                
                if cycle2_dir.exists() and window is not None:
                    cycle_suffix = f"{dataset_name}_Cycle2"
                    crop_cycle2(cycle2_dir, crop_root / dataset_name, block_name, cycle_suffix, window)
        else:
            cycle_suffix = dataset_name
            crop_one_block(dataset_dir / block_name, crop_root / dataset_name, block_name, cycle_suffix, margin=margin)

    print(f"\n✅ 裁剪完成，输出目录: {crop_root}")


if __name__ == "__main__":
    import sys
    _project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_project_root))
    
    from fiji_stitcher.config import load_config, apply_cli_overrides
    
    cfg = load_config()
    cfg = apply_cli_overrides(cfg)
    
    print("=" * 60)
    print("裁剪流水线启动")
    print("=" * 60)
    
    crop_all_blocks(cfg)
    
    print("\n🎉 裁剪完成!")
