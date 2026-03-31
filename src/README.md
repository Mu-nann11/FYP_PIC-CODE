# TMA Image Analysis Pipeline

Automated breast cancer tissue microarray (TMA) analysis using deep learning.

## Project Structure

```
src/
├── pipeline/           # Main pipeline (stitch -> align -> segment -> grade)
│   ├── main.py        # Entry point
│   ├── config.py      # Configuration
│   ├── steps/         # Pipeline steps
│   │   ├── preprocess.py
│   │   ├── stitch.py
│   │   ├── align.py
│   │   ├── segment.py
│   │   ├── grading_and_subtyping.py
│   │   ├── visualization_alignment.py
│   │   └── visualization_summary.py
│   └── utils/         # Utilities
│       ├── logging.py
│       └── report.py
├── segmentation/       # Cell segmentation modules
│   ├── cpsam_cyto_to_nucleus.py   # Core CPSAM + CellPose
│   ├── detect_nuclei_v3.py        # CellPose v3 nuclei
│   └── ...
├── alignment/         # Image registration
│   ├── alignment.py   # Two-stage rotation search
│   └── batch_alignment.py
├── calibration/       # Threshold calibration
│   ├── config.py
│   └── run_calibration.py
├── visualization/     # Figure generation
│   ├── figure5_cell_segmentation_results.py
│   ├── figure6_molecular_subtype_distribution.py
│   ├── figure7_feature_table_preview.py
│   └── figure7b_marker_summary.py
├── fiji_stitcher/    # Fiji tile stitching
│   └── ...
└── preprocessing/   # Data preprocessing
    └── ...
```

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n tma python=3.10
conda activate tma

# Install dependencies
pip install -r requirements.txt

# Install CellPose models
python -m cellpose --download
```

### 2. Run Pipeline

```bash
# Activate environment
conda activate tma

# Run specific blocks
python -m src.pipeline.main --dataset TMAd --blocks G2 A2 B8

# Run all blocks
python -m src.pipeline.main --dataset TMAd --all-blocks

# Resume from specific step
python -m src.pipeline.main --dataset TMAd --blocks G2 --from-step 2

# Force rerun
python -m src.pipeline.main --dataset TMAd --blocks G2 --force
```

### 3. Generate Figures

```bash
# Figure 5: Cell segmentation results
python -m src.visualization.figure5_cell_segmentation_results

# Figure 6: Molecular subtype distribution
python -m src.visualization.figure6_molecular_subtype_distribution

# Figure 7: Feature table preview
python -m src.visualization.figure7_feature_table_preview
```

## Pipeline Steps

| Step | Description |
|------|-------------|
| 0 | Preprocessing (tile to channel) |
| 1 | Stitching (Fiji tile assembly) |
| 2 | Alignment (multi-channel registration) |
| 3 | Segmentation (CellPose + CPSAM) |
| 4 | Grading & Molecular Subtyping |

## Scientific Background

This pipeline analyzes breast cancer TMA for biomarker scoring:

- **HER2**: Membrane protein (scored 0-3+)
- **ER/PR**: Nuclear receptors
- **Ki67**: Proliferation marker

Molecular subtypes:
- Luminal A/B
- HER2 Enriched
- Triple Negative

## Citation

If you use this code, please cite:

```
TMA Image Analysis Pipeline
Final Year Project (FYP)
```

## License

MIT License
