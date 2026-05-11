# SDO Downstream — Solar Image Analysis with Masked Autoencoders

Pipeline completa per l'analisi di immagini solari multispettrali provenienti dal **Solar Dynamics Observatory (SDO/AIA)**. Il progetto è organizzato in tre fasi: pre-training self-supervised con MAE, segmentazione di macchie solari, e previsione temporale dell'attività solare.

---

## Struttura del repository

```
SDO_downstream/
├── mae_project/          # Fase 1 — Pre-training MAE
├── seg_project/          # Fase 2 — Segmentazione downstream
├── forecast_project/     # Fase 3 — Previsione temporale downstream
├── AIA_to_zarr_file.py   # Conversione FITS → Zarr
└── allineatore.py        # Allineamento temporale dei canali
```

---

## Dati

**Sorgente:** file Zarr da immagini FITS SDO/AIA acquisite dal satellite NASA.

| Campo | Valore |
|---|---|
| Path principale | `/home/gpatane/Dataset/zarr_file_magnetogram_1024_definitivo.zarr` |
| Path rechunked (forecast) | `/home/gpatane/Dataset/zarr_file_magnetogram_1024_rechunked.zarr` |
| Risoluzione nativa | 1024 × 1024 px |
| Copertura temporale | 2011 – 2025 |
| Frequenza | ~2 immagini/giorno per canale |

**Canali AIA disponibili:**

| Canale | Lunghezza d'onda | Strato solare |
|---|---|---|
| AIA 94 Å | EUV | Corona calda (~6.3 MK) |
| AIA 131 Å | EUV | Flare e regioni attive |
| AIA 171 Å | EUV | Corona tranquilla (~0.6 MK) |
| AIA 193 Å | EUV | Corona + plasma flare |
| AIA 211 Å | EUV | Regioni attive coronali |
| AIA 304 Å | EUV+UV | Cromosfera / regione di transizione |
| AIA 335 Å | EUV | Regioni attive corona alta |
| AIA 1600 Å | UV | Zona di transizione / fotosfera alta |
| AIA 1700 Å | UV | Fotosfera continua |
| Magnetogramma HMI | — | Campo magnetico linea di vista |
| Ic_noLimbDark | — | Intensità continua senza oscuramento al bordo (usata per maschere) |

**Normalizzazione:** `sign(x) * log1p(0.01 * |x|)` applicata pixel per pixel su ogni canale AIA.

---

## Fase 1 — Pre-training MAE (`mae_project/`)

### Obiettivo

Addestrare in modo self-supervised un Masked Autoencoder (MAE) a ricostruire patch e canali mascherati di immagini SDO multispettrali. L'encoder risultante viene poi riutilizzato come backbone congelato o fine-tuned nelle fasi downstream.

### Architettura: `MaskedAutoencoderViT`

Definita in `mae_project/mae/models_mae_2.py`.

#### Encoder — ViT-Base

| Iperparametro | Valore |
|---|---|
| `img_size` | 1024 |
| `patch_size` | 16 |
| `in_chans` | 9 (canali AIA) |
| `embed_dim` | 768 |
| `depth` | 12 blocchi Transformer |
| `num_heads` | 12 |
| `mlp_ratio` | 4.0 |
| Patch totali | 4096 (64 × 64 griglia) |
| Parametri encoder | ~86 M |

#### Decoder

| Iperparametro | Valore |
|---|---|
| `decoder_embed_dim` | 512 |
| `decoder_depth` | 8 blocchi Transformer |
| `decoder_num_heads` | 16 |
| Output | patch ricostruite (pixel space) |

#### Strategia di masking

Il modello supporta due modalità combinabili:

**1. Spatial masking**
Rimuove casualmente il 75% delle patch spaziali prima dell'encoder (standard MAE). L'encoder processa solo le patch visibili (25%), riducendo drasticamente il costo computazionale. Il decoder riceve anche i token mascherati (come learned `mask_token`) e ricostruisce tutte le patch.

**2. Channel masking**
Seleziona casualmente `n_img_mask` canali AIA da mascherare prima del patch embedding. Per i canali mascherati, il contributo al token di patch viene sostituito con un `channel_mask_value` apprendibile per canale. Il modello impara quindi a ricostruire canali interi a partire dagli altri, analogamente a un problema di image inpainting multispettrale.

```
Input [B, 9, 1024, 1024]
    ↓  Channel masking: k canali sostituiti con learnable value
    ↓  PatchEmbed + PositionalEmbed + ChannelEmbed
    ↓  Spatial masking (75% patch rimosse)
    ↓  12× ViT Block (encoder)
    ↓  LayerNorm → latent [B, N_vis+1, 768]
    ↓  decoder_embed → mask_token insert → decoder_pos_embed
    ↓  8× ViT Block (decoder)
    ↓  LayerNorm → linear head
Ricostruzione [B, N_patch, patch_size² × 9]
```

#### `CrossChannelAttentionBlock`

Blocco di self-attention aggiuntivo nel decoder che opera tra le rappresentazioni dei canali. Permette ai canali visibili di comunicare con quelli mascherati durante la ricostruzione, migliorando la qualità della previsione cross-channel.

#### Loss

MSE pixel-per-pixel sulle sole patch mascherate spazialmente. Opzionalmente normalizzata per media/varianza di ogni patch (`norm_pix_loss`).

#### Factory function

```python
from mae.models_mae_2 import mae_model_channel_masking_9ch_with_temporal_attn

model = mae_model_channel_masking_9ch_with_temporal_attn(
    img_size=1024,
    patch_size=16,
    in_chans=9,
    mask_ratio=0.75,
    norm_pix_loss=False,
)
```

#### Addestramento

```bash
cd mae_project
python train.py \
    --zarr_path /home/gpatane/Dataset/zarr_file_magnetogram_1024_ORDINATO.zarr \
    --epochs 100 --batch_size 8 --lr 1.5e-3 \
    --mask_ratio 0.3 --accum_steps 8 \
    --mixed_precision --wandb_enabled
```

| Parametro chiave | Valore usato |
|---|---|
| Optimizer | AdamW |
| LR | 1.5e-3 (cosine decay) |
| Warmup | 5 epoche lineari |
| Weight decay | 0.05 |
| Gradient clip | 1.0 |
| Batch effettivo | `batch_size × accum_steps` |
| Precisione | bf16 mixed precision |

---

## Fase 2 — Segmentazione downstream (`seg_project/`)

### Obiettivo

Segmentazione binaria di macchie solari (sunspot) e regioni magneticamente attive su immagini SDO a 9 canali, riutilizzando l'encoder MAE pre-addestrato. La maschera ground truth è derivata dal canale `Ic_noLimbDark` con soglia adattiva.

### Dataset: `SDO_9Channel_Dataset`

Definito in `seg_project/dataset.py`.

- Input: stack `[9, H, W]` di canali AIA normalizzati con `log1p(0.01 · x)`
- Target: maschera binaria `[H, W]` derivata da `Ic_noLimbDark` (pixel sotto soglia = macchia solare)
- Split temporale: train 2011–2020, val 2021–2022, test 2023–2025

### Architettura: `MAEFeatureExtractor` + decoder

#### `MAEFeatureExtractor`

Wrap dell'encoder MAE che estrae feature intermedie ai layer **3, 6, 9, 12** (quarti del Transformer), ottenendo 4 mappe a risoluzione identica (64 × 64) ma con progressiva astrazione semantica. Le feature sono restituite come tensori 2D `[B, 768, 64, 64]`.

```
Input [B, 9, 1024, 1024]
    ↓  PatchEmbed → [B, 4096, 768]
    ↓  PositionalEmbed
    ↓  Block 0–2   → feature f1 [B, 768, 64, 64]   (layer 3  — texture / bordi)
    ↓  Block 3–5   → feature f2 [B, 768, 64, 64]   (layer 6  — strutture locali)
    ↓  Block 6–8   → feature f3 [B, 768, 64, 64]   (layer 9  — semantica)
    ↓  Block 9–11  → feature f4 [B, 768, 64, 64]   (layer 12 — contesto globale)
```

### Decoder disponibili

#### `SegDeformerUNetDecoder` — V1 (2.5 M parametri trainabili)

Decoder leggero ispirato a SegDeformer con pixel-shuffle upsampling.

| Stage | Operazione | Canali |
|---|---|---|
| Proiezioni 1×1 | `proj4/3/2/1` | 768 → 256 / 128 / 64 / 32 |
| Bottleneck | `up1` (PixelShuffle ×2) | 256 → 128 |
| Stage 3 | `DeformableRefinementBlock` + cat + `up2` | 256 → 64 |
| Stage 2 | `DeformableRefinementBlock` + cat + `up3` | 128 → 32 |
| Stage 1 | cat skip + `final_head` | 64 → 2 classi |
| Upsample finale | bilinear interpolation | → 1024 × 1024 |

#### `SegDeformerUNetDecoderV2` — V2 (7.5 M parametri trainabili) ⭐ Raccomandato

Decoder potenziato con canali doppi, contesto multi-scala e attenzione.

| Componente | Descrizione | Parametri |
|---|---|---|
| `FeatureAdapter` ×4 | Bottleneck residuale 768→192→768 per adattare feature congelate al task | ~1.2 M |
| Proiezioni 1×1 | 768 → 512 / 256 / 128 / 64 | ~0.7 M |
| `ASPPLite` | Atrous Spatial Pyramid Pooling (rate 1, 6, 12) sul bottleneck 512 ch | ~1.6 M |
| Upsample ×3 | Bilinear ×2 + Conv 3×3 + BN + ReLU | ~1.5 M |
| `DoubleConv` ×3 | 2 × (Conv 3×3 + BN + ReLU) dopo ogni fusione skip | ~2.3 M |
| `SCSEBlock` ×3 | Squeeze-and-Excitation canale + spaziale dopo ogni stage | ~0.1 M |
| Final head | Dropout2d(0.1) + Conv 3×3 + Conv 1×1 | ~0.02 M |

**Flusso forward V2:**

```
f4 → FeatureAdapter → proj4(512) → ASPP → Upsample×2
                                              ↓ cat con proj3(f3)
                                         DoubleConv(512→256) → scSE → Upsample×2
                                              ↓ cat con proj2(f2)
                                         DoubleConv(256→128) → scSE → Upsample×2
                                              ↓ cat con proj1(f1)
                                         DoubleConv(128→64) → scSE → head → [B,2,512,512]
                                              ↓ interpolate bilinear
                                         [B, 2, 1024, 1024]
```

#### Altri decoder disponibili

| Classe | Descrizione |
|---|---|
| `MAE_UNet_Segmentation` | Decoder U-Net classico su feature MAE (`UNetViTDecoder`) |
| `MAE_Seg_Advanced` | ASPP + scSE + CoordConv + FeatureAdapter (`AdvancedUNetViTDecoder`) |
| `MAE_FrozenEncoderSeg` | Decoder MAE originale riutilizzato per segmentazione (baseline lineare) |

### Training (`seg_project/prova_script_rapido.py`)

```bash
cd seg_project

# Encoder congelato — V2 (raccomandato)
python prova_script_rapido.py --mode train --model MAE_Seg_DeformerV2 \
  --freeze_encoder --load_pretrained \
  --mae_checkpoint /path/to/mae_weights.pth \
  --batch_size 2 --epochs 200 --lr 1e-4 --device cuda:0

# Fine-tuning end-to-end — V2
python prova_script_rapido.py --mode train --model MAE_Seg_DeformerV2 \
  --load_pretrained --mae_checkpoint /path/to/mae_weights.pth \
  --batch_size 1 --epochs 150 --lr 5e-5

# Encoder congelato — V1 (leggero, bassa VRAM)
python prova_script_rapido.py --mode train --model MAE_Seg_Deformer \
  --freeze_encoder --load_pretrained \
  --mae_checkpoint /path/to/mae_weights.pth \
  --batch_size 3 --epochs 200 --lr 1e-4

# Resume
python prova_script_rapido.py --mode resume --model MAE_Seg_DeformerV2 \
  --checkpoint_path /path/to/seg_checkpoint.pth --epochs 200

# Test / Inference
python prova_script_rapido.py --mode test --model MAE_Seg_DeformerV2 \
  --checkpoint_path /path/to/best_model.pth --device cuda:0
```

| Parametro | V1 frozen | V2 frozen | V2 fine-tune |
|---|---|---|---|
| Parametri trainabili | 2.5 M | 7.5 M | ~130 M |
| Batch size consigliato | 3 | 2 | 1 |
| VRAM stimata (1024²) | ~6 GB | ~10 GB | ~18 GB |
| LR iniziale | 1e-4 | 1e-4 | 5e-5 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |

### WandB Sweep

```bash
wandb sweep seg_project/sweep_config.yaml   # stampa sweep_id
wandb agent <sweep_id>
```

Ottimizzazione Bayesiana su: LR ∈ [1e-5, 1e-3], batch size ∈ {1, 2, 4}, loss ∈ {DiceCELoss, TwerskyLoss}. Metrica target: `val/dice`.

### SLURM

```bash
sbatch seg_project/train_frozen_normpix.sbatch
sbatch seg_project/train_finetuning_normpix.sbatch
sbatch seg_project/resume_finetuning_normpix.sbatch
squeue -u gpatane
tail -f seg_project/prova_train-<job_id>.out
```

---

## Fase 3 — Previsione temporale (`forecast_project/`)

### Obiettivo

Predire l'immagine solare multispettrale (9 canali AIA) a un orizzonte temporale `Δt` a partire dall'immagine corrente. Il task è formulato come regressione immagine→immagine condizionata sul Δt.

Orizzonti supportati: **12 h, 24 h, 36 h, 48 h, 168 h (1 settimana)**.

### Dataset: `SDO_TemporalDataset`

Definito in `forecast_project/dataset.py`.

- Costruisce una timeline globale ordinata per timestamp `T_OBS` (attributo Zarr)
- Per ogni campione: `(immagine_t, immagine_{t+Δt}, indice_Δt)`
- Tolerance massima sul gap temporale: 3 h (configurabile con `--max_gap_hours`)
- Normalizzazione: identica al pre-training (`log1p(0.01 · x)`)
- Split: train 2011–2020 / val 2021–2022 / test 2023–2025

### Architettura: `MAE_TemporalForecaster`

Definita in `forecast_project/models.py`.

```
Input [B, 9, 1024, 1024]
    ↓  Encoder MAE congelato (ViT-Base, 12 blocchi, embed=768)
    ↓  + Delta-t Embedding(num_horizons, 768) sommato a ogni token
    ↓  num_temporal_blocks blocchi ViT trainabili (adattamento al task)
    ↓  LayerNorm
    ↓  decoder_embed (768 → 512)
    ↓  decoder_depth blocchi ViT decoder
    ↓  LayerNorm → linear head (patch_size² × 9)
    ↓  Unpatchify
Output [B, 9, 1024, 1024]  — immagine predetta a t + Δt
```

| Componente | Dettaglio |
|---|---|
| Encoder (congelato) | ViT-Base: 12 blocchi, embed=768, heads=12, ~86 M param |
| Delta-t conditioning | `nn.Embedding(len(delta_t_values), 768)` sommato a ogni token |
| Temporal blocks | 4–6 blocchi ViT trainabili post-encoder |
| Decoder | 6 blocchi ViT, embed=512, heads=16 |
| Loss | MSE pixel-level sulle patch (opz. `norm_pix_loss`) |
| Precisione | bf16 mixed precision + gradient checkpointing |

### Training

```bash
cd forecast_project

# Encoder congelato (primo run consigliato)
python train.py --mode train --freeze_encoder \
    --mae_checkpoint /path/to/best_model.pth \
    --epochs 100 --lr 3e-4 --batch_size 2 \
    --num_temporal_blocks 4 --decoder_depth 6

# Fine-tuning (dopo convergenza frozen)
python train.py --mode train \
    --mae_checkpoint /path/to/best_model.pth \
    --checkpoint_path /path/to/frozen_best.pth \
    --no_freeze_encoder \
    --epochs 50 --lr 5e-5 --batch_size 1

# Resume
python train.py --mode resume \
    --checkpoint_path /path/to/last.pth --epochs 150

# SLURM
sbatch forecast_project/train_normpix.sbatch
```

---

## Confronto parametri — tutti i modelli decoder

| Modello | Parametri trainabili | Encoder frozen | Uso |
|---|---|---|---|
| `MAE_UNet_Segmentation` | ~1.5 M | 122 M | U-Net classico (baseline) |
| `MAE_Seg_Deformer` (V1) | 2.5 M | 122 M | Segmentazione leggera |
| `MAE_Seg_Advanced` | ~4 M | 122 M | ASPP + CoordConv + Adapter |
| `MAE_Seg_DeformerV2` | **7.5 M** | 122 M | **Segmentazione — raccomandato** |
| `MAE_TemporalForecaster` | ~25 M | 86 M | Previsione temporale |

---

## Ambiente e dipendenze

```bash
conda env create -f mae_project/environment.yml
conda activate SDOenv
```

Dipendenze principali: `PyTorch 2.9.1`, `timm 1.0.19`, `MONAI 1.5.0`, `WandB 0.19.8`, `AstroPy`, `SunPy`, `Zarr 2.18.4`, `scikit-learn`, `scipy`.

---

## Checkpoint

I checkpoint vengono salvati in `<progetto>/checkpoints/` nel formato:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": int,
    "best_dice": float,           # seg_project
    "scheduler_state_dict": ...,
}
```

Caricamento con adattamento automatico dei canali (quando `in_chans` del checkpoint differisce dal modello corrente):

```python
from utils_2 import load_checkpoint_with_channel_adaptation
model = load_checkpoint_with_channel_adaptation(
    model, path, in_chans=9, out_chans=2, device=device
)
```
