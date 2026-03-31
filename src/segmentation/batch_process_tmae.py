#!/usr/bin/env python3
"""
Batch processing script for TMAe CPSAM cytoplasm-to-nucleus segmentation.
Processes all samples: B7, C4, D5, G10, K1
"""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")

# Configuration
CONDA_ENV_PYTHON = r"D:\Miniconda3\envs\cellpose\python.exe"
MODEL_PATH = r"C:\Users\24037101d\.cellpose\models\her2_wholecell_v3"
SCRIPT_PATH = Path(__file__).parent / "cpsam_cyto_to_nucleus.py"
RAW_DATA_DIR = Path(r"D:\Try_munan\FYP_LAST\Raw_Data\TMAe")
OUTPUT_DIR = Path(r"D:\Try_munan\FYP_LAST\results\cpsam_segmentation")


def parse_tile_config(config_path: Path) -> dict:
    """Parse TileConfiguration file to get tile positions."""
    positions = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(';')
            if len(parts) >= 3:
                filename = parts[0].strip()
                coords = parts[2].strip().strip('()')
                x, y = map(float, coords.split(','))
                positions[filename] = (x, y)
    return positions


def stitch_tiles(sample_dir: Path, channel: str, tile_dir: Path) -> tuple:
    """
    Stitch tiles for a given channel using TileConfiguration.registered.txt
    Returns (merged_image, offset_x, offset_y) where offset is the minimum x,y
    """
    import numpy as np
    import tifffile

    config_file = tile_dir / f"TileConfiguration_{tile_dir.parent.name}_TMAe_{channel}.registered.txt"
    
    if not config_file.exists():
        print(f"    Warning: Config file not found: {config_file}")
        # Try alternative naming
        sample_name = tile_dir.parent.name
        config_file = tile_dir / f"TileConfiguration_{sample_name}_TMAe_{channel}.registered.txt"
        if not config_file.exists():
            raise FileNotFoundError(f"Cannot find config file for {sample_name}/{channel}")

    positions = parse_tile_config(config_file)
    
    # Load all tiles
    tiles = {}
    for tif_file in tile_dir.glob("*.TIF"):
        if tif_file.name in positions:
            img = tifffile.imread(str(tif_file))
            tiles[tif_file.name] = img
    
    if not tiles:
        raise ValueError(f"No tiles found in {tile_dir}")
    
    # Find tile size (assume all same size)
    first_tile = list(tiles.values())[0]
    tile_h, tile_w = first_tile.shape[:2]
    
    # Calculate output canvas size
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    
    min_x, max_x = min(xs), max(xs) + tile_w
    min_y, max_y = min(ys), max(ys) + tile_h
    
    canvas_h = int(max_y - min_y)
    canvas_w = int(max_x - min_x)
    
    # Create output canvas
    merged = np.zeros((canvas_h, canvas_w), dtype=first_tile.dtype)
    
    # Place tiles
    for filename, img in tiles.items():
        x, y = positions[filename]
        x, y = int(x - min_x), int(y - min_y)
        # Handle different tile sizes
        h, w = img.shape[:2]
        merged[y:y+h, x:x+w] = img
    
    return merged, -min_x, -min_y


def prepare_sample(sample_name: str) -> dict:
    """Prepare merged DAPI and HER2 images for a sample."""
    import numpy as np
    import tifffile
    
    sample_dir = RAW_DATA_DIR / sample_name
    dapi_dir = sample_dir / "DAPI"
    her2_dir = sample_dir / "HER2"
    
    # Create temp directory for merged images
    temp_dir = OUTPUT_DIR / "temp_merges" / sample_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Stitching tiles for {sample_name}...")
    dapi_merged, _, _ = stitch_tiles(sample_dir, "DAPI", dapi_dir)
    her2_merged, _, _ = stitch_tiles(sample_dir, "HER2", her2_dir)
    
    dapi_path = temp_dir / f"{sample_name}_DAPI_merged.tif"
    her2_path = temp_dir / f"{sample_name}_HER2_merged.tif"
    
    tifffile.imwrite(str(dapi_path), dapi_merged.astype(np.float32))
    tifffile.imwrite(str(her2_path), her2_merged.astype(np.float32))
    
    return {
        "sample_name": sample_name,
        "dapi": str(dapi_path),
        "her2": str(her2_path),
        "temp_dir": temp_dir
    }


def run_cpsam_segmentation(dapi_path: str, her2_path: str, sample_name: str) -> subprocess.CompletedProcess:
    """Run the CPSAM segmentation script."""
    cmd = [
        CONDA_ENV_PYTHON,
        str(SCRIPT_PATH),
        "--dapi", dapi_path,
        "--her2", her2_path,
        "--model", MODEL_PATH,
        "--block-name", sample_name,
        "--output-dir", str(OUTPUT_DIR),
        "--diameter", "30",
        "--flow-threshold", "0.4",
        "--cellprob-threshold", "0.0",
        "--use-cellpose-nuclei",
    ]
    
    print(f"  Running CPSAM segmentation for {sample_name}...")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return result


def process_sample(sample_name: str) -> dict:
    """Process a single sample: stitch tiles and run segmentation."""
    import shutil
    
    try:
        # Step 1: Prepare merged images
        sample_info = prepare_sample(sample_name)
        
        # Step 2: Run CPSAM segmentation
        result = run_cpsam_segmentation(
            sample_info["dapi"],
            sample_info["her2"],
            sample_name
        )
        
        success = result.returncode == 0
        
        if success:
            print(f"  [OK] {sample_name} completed successfully")
            # Cleanup temp files
            shutil.rmtree(sample_info["temp_dir"], ignore_errors=True)
        else:
            print(f"  [FAIL] {sample_name} failed")
            print(f"    Error: {result.stderr[-500:] if result.stderr else 'Unknown error'}")
        
        return {
            "sample": sample_name,
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except Exception as e:
        print(f"  [FAIL] {sample_name} failed with exception: {e}")
        return {
            "sample": sample_name,
            "success": False,
            "error": str(e)
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch CPSAM segmentation for TMAe")
    parser.add_argument("--samples", nargs="+", default=["D5", "G10", "K1"],
                       help="Samples to process (default: D5, G10, K1 - B7 and C4 already completed)")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers (default: 1)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("TMAe CPSAM Batch Processing")
    print("=" * 70)
    print(f"Samples: {args.samples}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)
    
    results = []
    
    if args.workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_sample, s): s for s in args.samples}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    else:
        # Sequential processing
        for sample in args.samples:
            print(f"\n[{args.samples.index(sample) + 1}/{len(args.samples)}] Processing {sample}...")
            result = process_sample(sample)
            results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    print(f"Successfully processed: {len(successful)}/{len(results)}")
    if successful:
        print(f"  Completed: {', '.join([r['sample'] for r in successful])}")
    if failed:
        print(f"  Failed: {', '.join([r['sample'] for r in failed])}")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
