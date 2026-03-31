# FYP_LAST - TMA Image Analysis

Automated breast cancer tissue microarray (TMA) analysis using deep learning.

## Repository Structure

```
FYP_PIC-CODE/
├── src/                      # Source code (organized by module)
├── results/figures/          # Generated figures (Figure 5, 6, 7)
├── Raw_Data/                 # Input data (not included, see below)
└── README.md
```

## Figures

| Figure | Description |
|--------|-------------|
| Figure 5 | Cell Segmentation Results (A1, H2, J10 blocks) |
| Figure 6 | Molecular Subtype Distribution |
| Figure 7 | Feature Table Preview |
| Figure 7b | Marker Summary Statistics |

## Source Code

The source code is organized in the `src/` directory:

```
src/
├── pipeline/           # Main analysis pipeline
├── segmentation/       # Cell segmentation (CPSAM + CellPose)
├── alignment/          # Multi-channel image registration
├── calibration/        # Threshold calibration
├── visualization/     # Figure generation
├── fiji_stitcher/      # Fiji tile stitching integration
└── preprocessing/      # Data preprocessing
```

See [src/README.md](src/README.md) for detailed documentation.

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Mu-nann11/FYP_PIC-CODE.git
cd FYP_PIC-CODE

# 2. Setup environment
conda create -n tma python=3.10
conda activate tma
pip install -r src/requirements.txt

# 3. Run pipeline
python -m src.pipeline.main --dataset TMAd --blocks G2

# 4. Generate figures
python -m src.visualization.figure5_cell_segmentation_results
```

## Scientific Context

This project implements automated analysis of breast cancer tissue microarrays for:

1. **Cell Segmentation**: CPSAM cytoplasm + CellPose nuclei
2. **Biomarker Scoring**: HER2, ER, PR, Ki67
3. **Molecular Subtyping**: Luminal A/B, HER2+, Triple Negative

## License

MIT License

## Author

ZENG QINAN (Mu-nann11)
