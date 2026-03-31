#!/usr/bin/env python3
"""
Cellpose v3 Nuclei Detection – serial version (fast enough: ~0.07s/cell).
Uses the v3 environment where the official nuclei model is available.
"""
import argparse
import time
import numpy as np
import tifffile
from tqdm import tqdm

from cellpose import models


def extract_roi(dapi_img, cyto_masks, label, margin=10):
    """Extract ROI patch. Returns (sub_img, y0, x0, cyto_area)."""
    region = cyto_masks == label
    rows = np.any(region, axis=1)
    cols = np.any(region, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0 = max(0, y0 - margin)
    y1 = min(dapi_img.shape[0], y1 + margin + 1)
    x0 = max(0, x0 - margin)
    x1 = min(dapi_img.shape[1], x1 + margin + 1)
    return dapi_img[y0:y1, x0:x1], y0, x0, int(np.sum(region))


def main():
    p = argparse.ArgumentParser(description="Cellpose v3 Nuclei Detection")
    p.add_argument("--cyto-masks", required=True)
    p.add_argument("--dapi-img", required=True)
    p.add_argument("--output-npy", required=True)
    p.add_argument("--min-nuc-area", type=int, default=30)
    p.add_argument("--max-area-ratio", type=float, default=0.8)
    p.add_argument("--nuclei-diameter", type=int, default=None)
    args = p.parse_args()

    print("Loading cytoplasm masks …", flush=True)
    cyto_masks = tifffile.imread(args.cyto_masks).astype(np.int32)
    print("Loading DAPI image …", flush=True)
    dapi_img = tifffile.imread(args.dapi_img).astype(np.float32)

    min_h = min(dapi_img.shape[0], cyto_masks.shape[0])
    min_w = min(dapi_img.shape[1], cyto_masks.shape[1])
    dapi_img = dapi_img[:min_h, :min_w]
    cyto_masks = cyto_masks[:min_h, :min_w]

    print("Loading Cellpose v3 nuclei model …", flush=True)
    t0 = time.time()
    nuc_model = models.Cellpose(gpu=True, model_type='nuclei')
    print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

    unique_labels = np.unique(cyto_masks)
    unique_labels = unique_labels[unique_labels > 0]
    total = len(unique_labels)
    print(f"  Total cytoplasm regions: {total}", flush=True)

    nuclei_masks = np.zeros_like(cyto_masks, dtype=np.int32)
    found = 0
    t_start = time.time()

    est_diameter = args.nuclei_diameter

    for idx, label in enumerate(tqdm(unique_labels, desc="Detecting nuclei", unit="cell")):
        roi, y0, x0, cyto_area = extract_roi(dapi_img, cyto_masks, label)

        if est_diameter is None:
            diameter = max(10, int(2 * np.sqrt(cyto_area / np.pi / 6)))
        else:
            diameter = est_diameter

        sub_masks, _, _, _ = nuc_model.eval(
            roi,
            diameter=diameter,
            channels=[0, 0],
            flow_threshold=0.4,
            cellprob_threshold=0.0,
        )

        if sub_masks.max() == 0:
            continue

        best_score = -1
        best_mask = None
        cyto_region = (cyto_masks == label)

        for sub_label in range(1, sub_masks.max() + 1):
            sub_nuc = sub_masks == sub_label
            nuc_area = int(np.sum(sub_nuc))
            if nuc_area < args.min_nuc_area:
                continue
            if nuc_area > cyto_area * args.max_area_ratio:
                continue

            candidate = np.zeros(cyto_masks.shape, dtype=bool)
            candidate[y0:y0 + sub_masks.shape[0],
                      x0:x0 + sub_masks.shape[1]] = sub_nuc > 0
            overlap = int(np.sum(candidate & cyto_region))
            score = overlap / (nuc_area + 1e-6)
            if score > best_score:
                best_score = score
                best_mask = candidate

        if best_mask is not None:
            nuclei_masks[best_mask] = label
            found += 1

    total_time = time.time() - t_start
    print(f"\nDone: {found}/{total} nuclei found in {total_time:.1f}s",
          flush=True)

    np.save(args.output_npy, nuclei_masks)
    print(f"Nuclei masks saved: {args.output_npy}", flush=True)


if __name__ == "__main__":
    main()
