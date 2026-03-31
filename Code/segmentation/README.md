# CPSAM Cytoplasm-to-Nucleus Segmentation

## 环境配置

- **Python 环境**: `D:\Miniconda3\envs\cellpose\python.exe`
- **Conda 环境名**: `cellpose`

## 模型路径

- **CPSAM 细胞质分割模型**: `D:\Try_munan\Cellpose_model\model2\models\her2_wholecell_v3`

## 输入数据

- **路径**: `D:\Try_munan\FYP_LAST\results\stitched\TMAe\<BLOCK_NAME>\`
- **必需文件**:
  - `<BLOCK>_TMAe_DAPI.tif` - DAPI 通道图像
  - `<BLOCK>_TMAe_HER2.tif` - HER2 通道图像

## 输出结果

- **路径**: `D:\Try_munan\FYP_LAST\results\cpsam_segmentation\<BLOCK_NAME>\`
- **生成文件**:
  - `<BLOCK>_cyto_masks.tif` - 细胞质分割掩码
  - `<BLOCK>_nuclei_masks.tif` - 细胞核分割掩码
  - `<BLOCK>_cell_masks.tif` - 完整细胞分割掩码
  - `<BLOCK>_features.csv` - 特征表
  - `<BLOCK>_overlay.png` - 可视化叠加图

## 使用方法

### 单个 Block 运行

```bash
D:\Miniconda3\envs\cellpose\python.exe D:\Try_munan\FYP_LAST\Code\segmentation\cpsam_cyto_to_nucleus.py \
    --dapi "D:\Try_munan\FYP_LAST\results\stitched\TMAe\<BLOCK>\XXX_TMAe_DAPI.tif" \
    --her2 "D:\Try_munan\FYP_LAST\results\stitched\TMAe\<BLOCK>\XXX_TMAe_HER2.tif" \
    --model "D:\Try_munan\Cellpose_model\model2\models\her2_wholecell_v3" \
    --block-name <BLOCK> \
    --output-dir "D:\Try_munan\FYP_LAST\results\cpsam_segmentation"
```

### 可选参数

- `--diameter`: CellPose 直径参数 (默认: 30)
- `--flow-threshold`: Flow threshold (默认: 0.4)
- `--cellprob-threshold`: Cell probability threshold (默认: 0.0)
- `--min-nuc-area`: 最小细胞核面积 (默认: 30)
- `--max-area-ratio`: 最大细胞核/细胞质面积比 (默认: 0.8)
- `--use-cellpose-nuclei`: 使用 CellPose 检测细胞核 (默认: True)
- `--no-cellpose-nuclei`: 使用 Otsu 方法检测细胞核
- `--nuclei-diameter`: 细胞核直径 (默认: 自动)

### 批量运行

使用 `batch_tmae.py` 脚本可以批量处理多个 blocks:

```bash
D:\Miniconda3\envs\cellpose\python.exe D:\Try_munan\FYP_LAST\Code\segmentation\batch_tmae.py
```

## 已处理 Blocks

| Block | 细胞数 | 细胞核数 | 无核细胞数 |
|-------|--------|----------|-------------|
| D5    | 1279   | 1273     | 6           |
| G10   | 2690   | 2640     | 50          |
| K1    | 3      | 3        | 0           |

## 注意事项

1. **环境选择**: 必须使用 `cellpose` 环境，而非 `cellpose_nuclei`
2. **DAPI + HER2 尺寸匹配**: 脚本会自动裁剪不匹配的尺寸
3. **K1 特殊情况**: K1 检测到的细胞数较少，可能需要调整参数
