# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Sunspot/magnetogram segmentation in Solar Dynamics Observatory (SDO) images using Masked Autoencoders (MAE) with transfer learning. Two-phase pipeline:
1. **mae_project**: MAE pre-training on multi-wavelength SDO images (spatial + channel masking)
2. **seg_project**: Downstream binary segmentation using frozen/fine-tuned MAE encoder + custom decoder

## Environment

```bash
conda env create -f mae_project/environment.yml
conda activate SDOenv
```

Key packages: PyTorch 2.9.1, timm 1.0.19, MONAI 1.5.0, WandB 0.19.8, AstroPy/SunPy, Zarr 2.18.4.

## Running Training

Active entry point: `seg_project/prova_script_rapido.py`. Training utilities live in `seg_project/utils_2.py`.

```bash
cd seg_project

# From scratch
python prova_script_rapido.py --mode train --model MAE_2Channel \
  --batch_size 3 --epochs 200 --lr 1e-4 --device cuda:0

# Frozen encoder (transfer learning)
python prova_script_rapido.py --mode train --model MAE_Seg_Deformer \
  --freeze_encoder --load_pretrained \
  --mae_checkpoint /path/to/mae_weights.pth \
  --batch_size 3 --epochs 200 --lr 1e-4

# Fine-tuning (end-to-end with pre-trained encoder)
python prova_script_rapido.py --mode train --model MAE_Seg_Deformer \
  --load_pretrained --mae_checkpoint /path/to/mae_weights.pth \
  --batch_size 2 --epochs 150 --lr 5e-5

# Resume
python prova_script_rapido.py --mode resume --model MAE_2Channel \
  --checkpoint_path /path/to/seg_checkpoint.pth --epochs 200

# Test/inference
python prova_script_rapido.py --mode test --model MAE_2Channel \
  --checkpoint_path /path/to/best_model.pth --device cuda:0
```

SLURM scripts in `seg_project/`: `train.sbatch`, `train_frozen.sbatch`, `train_finetuning.sbatch`, `resume*.sbatch`.

```bash
sbatch seg_project/train_frozen.sbatch
squeue -u gpatane
tail -f seg_project/prova_train-<job_id>.out
```

## WandB Sweep

```bash
wandb sweep seg_project/sweep_config.yaml   # prints sweep_id
wandb agent <sweep_id>
```

Sweep config: `sweep_config.yaml` — Bayesian optimization over LR (log-uniform 1e-5–1e-3), batch size [1,2,4], loss (DiceCELoss vs. TwerskyLoss). Metric: `val/dice`.

## Architecture

### MAE Encoder (`seg_project/mae/models_mae_2.py`)

`MaskedAutoencoderViT`: ViT-based, `img_size=1024, patch_size=16, in_chans=9, embed_dim=768, depth=12`. Supports:
- Spatial masking (75% patches)
- Channel masking (`n_img_mask` param)
- CrossChannelAttentionBlock for temporal attention

The same model class is duplicated in `mae_project/mae/models_mae_2.py` for pre-training.

### Segmentation Decoders (`seg_project/models.py`)

| Class | Description |
|-------|-------------|
| `MAEFeatureExtractor` | Extracts features at encoder layers 3, 6, 9, 12 (4 scales) |
| `MAE_Seg_Deformer` | **Recommended** — deformable conv refinement + pixel-shuffle upsampling |
| `MAE_UNet_Segmentation` | Classic U-Net decoder on MAE features |
| `MAE_Seg_Advanced` | ASPP + SCSE attention blocks |
| `MAE_FrozenEncoderSeg` | Minimal linear decoder (frozen encoder baseline) |

`MAE_Seg_Deformer` uses `SegDeformerUNetDecoder`: projects 4-scale features to (256, 128, 64, 32) channels, `DeformableRefinementBlock` for spatial adaptation, skip connections.

### Data (`seg_project/dataset.py`)

- **Zarr source**: `/home/gpatane/Dataset/zarr_file_magnetogram_1024_definitivo.zarr`
- **Channels**: 9 AIA wavelengths + Magnetogram + `Ic_noLimbDark` (mask generation)
- **`SDO_9Channel_Dataset`**: log-normalization `log1p(0.01 * img)`, optional percentile clipping, `scipy.ndimage.zoom` resize, binary masks from `Ic_noLimbDark`
- **`SDOMosaicZarrDataset`**: grid-based access for large images, with retry on failed loads

### Checkpoint Loading

Channel adaptation when MAE weights have different `in_chans` than the current model:

```python
# In utils_2.py
model = load_checkpoint_with_channel_adaptation(model, path, in_chans=9, out_chans=2, device=device)
```

Checkpoint format: `{"model_state_dict", "optimizer_state_dict", "epoch", "best_dice", "scheduler_state_dict"}`.

## Key Files

- `seg_project/prova_script_rapido.py` — active main script (argparse entry point)
- `seg_project/segmentation.py` — alternative script (less active)
- `seg_project/utils_2.py` — training loop (`train_model`, `train_one_epoch`, `validate_one_epoch`)
- `seg_project/utils.py` — deprecated, prefer `utils_2.py`
- `seg_project/losses.py` — DiceBCELoss, TwerskyLoss, custom combinations
- `mae_project/train.py` — MAE pre-training entry point
- `AIA_to_zarr_file.py` — FITS → Zarr conversion
