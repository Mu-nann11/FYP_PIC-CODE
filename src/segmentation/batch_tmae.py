"""
Run CPSAM cytoplasm-to-nucleus segmentation for all TMAe blocks in batch.
Usage: python batch_tmae.py
"""
import sys
from pathlib import Path

# Ensure the module path for the current script directory is available
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
from cpsam_cyto_to_nucleus import (
    load_channel, merge_channels, segment_cytoplasm,
    find_nucleus_in_cytoplasm, align_labels,
    extract_features, save_results, parse_args,
)

MODEL = r"D:\Try_munan\Cellpose_model\model2\models\her2_wholecell_v3"
STITCHED = Path(r"d:\Try_munan\FYP_LAST\results\stitched\TMAe")
OUTPUT_DIR = r"d:\Try_munan\FYP_LAST\results\segmentation\TMAe"

blocks = sorted([d.name for d in STITCHED.iterdir() if d.is_dir()])
print(f"Found {len(blocks)} TMAe blocks: {blocks}")
print()

results = {}
for block in blocks:
    dapi_path = STITCHED / block / f"{block}_TMAe_DAPI.tif"
    her2_path = STITCHED / block / f"{block}_TMAe_HER2.tif"
    out_block = Path(OUTPUT_DIR) / block

    # Skip if already processed
    if (out_block / f"{block}_TMAe_features.csv").exists():
        print(f"[{block}] already completed, skipping\n")
        results[block] = 0
        continue

    print("=" * 60)
    print(f"Processing block: {block}")
    print("=" * 60)

    try:
        # 1. Load
        print("\n[1/7] Loading images ...")
        dapi_img = load_channel(str(dapi_path))
        her2_img = load_channel(str(her2_path))
        print(f"    DAPI shape: {dapi_img.shape}")
        print(f"    HER2 shape: {her2_img.shape}")

        # 2. Merge
        print("\n[2/7] Merging channels ...")
        img_2ch = merge_channels(dapi_img, her2_img)
        her2_img = her2_img[:img_2ch.shape[1], :img_2ch.shape[2]]
        print(f"    Merged: {img_2ch.shape}")

        # 3. Segment cytoplasm
        print("\n[3/7] Running CPSAM ...")
        cyto_masks = segment_cytoplasm(img_2ch, MODEL, diameter=30,
                                        flow_threshold=0.4, cellprob_threshold=0.0,
                                        channels=(0, 1))
        print(f"    {len([l for l in set(cyto_masks.flat) if l > 0])} cytoplasm regions")

        # 4. Find nuclei
        print("\n[4/7] Finding nuclei ...")
        nuclei_masks = find_nucleus_in_cytoplasm(cyto_masks, dapi_img,
                                                  min_nuc_area=30, max_area_ratio=0.8)
        n_nuc = len(set(nuclei_masks.flat) - {0})
        print(f"    {n_nuc} nuclei")

        # 5. Align
        print("\n[5/7] Aligning labels ...")
        cell_masks = align_labels(cyto_masks, nuclei_masks)
        print(f"    {len(set(cell_masks.flat) - {0})} complete cells")

        # 6. Extract features
        print("\n[6/7] Extracting features ...")
        features_df = extract_features(cyto_masks, nuclei_masks, cell_masks,
                                       dapi_img, her2_img)
        no_nuc = (~features_df["has_nucleus"]).sum()
        print(f"    {len(features_df)} cells, {int(no_nuc)} without nucleus")

        # 7. Save
        print("\n[7/7] Saving ...")
        save_results(OUTPUT_DIR, block, cyto_masks, nuclei_masks, cell_masks,
                     features_df, dapi_img, her2_img)

        results[block] = 0
        print(f"\n✅ {block} completed\n")

    except Exception as e:
        results[block] = -1
        print(f"\n❌ {block} failed: {e}\n")
        import traceback
        traceback.print_exc()

print()
print("=" * 60)
print("Summary:")
for block, code in results.items():
    status = "✅ Success" if code == 0 else "❌ Failed"
    print(f"  {block}: {status}")
