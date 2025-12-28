import sys


import torch.nn as nn
import torch
from torch import optim 

from torch.utils.data import DataLoader

torch.manual_seed(5)
import numpy as np

from monai.transforms import (
    Resized,
    
    Compose,
    EnsureTyped,
    
    RandScaleIntensityd,
    RandShiftIntensityd,
    AsDiscrete,
    ToTensord
)
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
import random

from models import  MAESegmentationModel
from mae.models_mae_2 import mae_for_segmentation_7 ,mae_for_segmentation_14

from dataset import SDOMosaicZarrDataset_2, PhotosphereDataset
from utils_2 import visualize_batch,train_model, dice_score_wt_bg, dice_score_bg


import matplotlib.pyplot as plt
import wandb

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Inserisci il modello che vuoi allenare: 0 per patch size 7, 1 per patch size 14")
#model_choice = int(input().strip())
model_choice = 0
if model_choice == 0:
    mae_model = mae_for_segmentation_7().to(device)
    mae_model.load_state_dict(torch.load("/home/gpatane/checkpoints/mae_project/7_patch_anni_equamente.pth", map_location=device))
elif model_choice == 1:
    mae_model = mae_for_segmentation_14().to(device)
    mae_model.load_state_dict(torch.load("/home/gpatane/checkpoints/mae_project/new_mae_magnetogram.pth", map_location=device))
#checkpoint funzionante pathc_size 14

#model = MAESegmentationModel(mae_model, num_classes=2, freeze_encoder=False, decoder_type='deep', dropout=0.1)
#for param in mae_model.parameters():
#    param.requires_grad = False
#print(f"✓ MAE encoder frozen: {sum(p.numel() for p in mae_model.parameters()):,} parameters")

# Ora puoi usare freeze_encoder=False perché hai già congelato manualmente
model = MAESegmentationModel(mae_model, num_classes=2, freeze_encoder=True,
                            decoder_type='deep', dropout=0.1)

# checkpoint_path = '/home/gpatane/checkpoints/seg_project/NUOVO_RUN_deep_decoder_model_improved.pth'
# model.load_state_dict(torch.load(checkpoint_path, weights_only=False, map_location=device)["model_state_dict"])
# print("✓ Checkpoint loaded successfully!")
start_period = 2010
end_period = 2026
random.seed(5)
all_years = list(range(start_period, end_period))
random.shuffle(all_years)

train_split = int(0.7 * len(all_years))
val_split = int(0.85 * len(all_years))

train_years = list(range(2010,2021))
val_years   = list(range(2021,2023))
test_years  = list(range(2023,2026))

print(f"Train years: {train_years}")
print(f"Validation years: {val_years}")
print(f"Test years: {test_years}")
#tolgo 2018 da train


transform = Compose([
    EnsureTyped(keys=["image", "mask"]),
    ToTensord(keys=["image", "mask"]),
    Resized(keys="image", spatial_size=[672, 672], mode="area"),
    Resized(keys="mask", spatial_size=[224, 224], mode="nearest"),

])
zarr_path = "/home/gpatane/Dataset/zarr_file_magnetogram.zarr"

wavelengths = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram', "Ic_noLimbDark"]
tr_dataset = SDOMosaicZarrDataset_2(zarr_path, train_years, wavelengths, target_size=224, transform=transform)
vl_dataset = SDOMosaicZarrDataset_2(zarr_path, val_years, wavelengths, target_size=224, transform=transform)
ts_dataset = SDOMosaicZarrDataset_2(zarr_path, test_years, wavelengths, target_size=224, transform=transform)
t_loader = DataLoader(tr_dataset, batch_size=32, shuffle=True, num_workers=8)
v_loader = DataLoader(vl_dataset, batch_size=32, shuffle=False, num_workers=8)
ts_loader = DataLoader(ts_dataset, batch_size=4, shuffle=False, num_workers=4)

from torch.nn import BCEWithLogitsLoss
import torch.nn as nn

from utils import train_model
def get_class_weights(dataloader, minimum =10):
    total_pixels = 0
    positive_pixels = 0
    
    for batch in dataloader:
        masks = batch["mask"]
        total_pixels += masks.numel()
        positive_pixels += masks.sum().item()
    
    pos_weight = (total_pixels - positive_pixels) / positive_pixels if positive_pixels > 0 else 1.0
    return min(pos_weight, minimum)


from monai.losses import DiceCELoss

pos_weight = get_class_weights(t_loader, minimum=100)
class_weights = torch.tensor([1.0, pos_weight], device=device)

criterion = DiceCELoss(
    to_onehot_y=True,        
    sigmoid=False,            
    softmax=True,             
    include_background=False, 
    weight=class_weights,     
    lambda_dice=0.5,          
    lambda_ce=0.5             
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR
scheduler = StepLR(optimizer, step_size=40, gamma=0.3)
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)
#scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])


from monai.transforms import AsDiscrete, Compose

post_pred = Compose([
    AsDiscrete(argmax=True, to_onehot=2)  # [B,2,H,W] logits → [B,2,H,W] one-hot
])

post_label = Compose([
    AsDiscrete(to_onehot=2)  # [B,1,H,W] indices {0,1} → [B,2,H,W] one-hot
])

from monai.metrics import DiceMetric

dice_metric = DiceMetric(
    include_background=False,  # ← Escludi background (calcola solo foreground)
    reduction="mean",
    get_not_nans=False,
    num_classes=2              # ← Specifica 2 classi
)

print("\n" + "="*60)
print("NUOVA CONFIGURAZIONE:")
print("="*60)
print(f"Loss: Combined (Dice + BCE with pos_weight={pos_weight:.2f})")
print(f"Optimizer: Adam with lr=1e-4")
print(f"Scheduler: Warmup (5 epochs) + CosineAnnealing")
print(f"Gradient Clipping: max_norm=1.0")
print("="*60)

# INIZIALIZZA WANDB
wandb.init(
    project="solar-segmentation-deep-decoder",
    name="deep_decoder_patch14_unfrozen_encoder",
    config={
        "architecture": "MAE + DeepDecoder",
        "encoder": "MAE (unfrozen)",
        "decoder": "DeepDecoder with Attention",
        "patch_size": 14,
        "learning_rate": 1e-2,
        "optimizer": "Adam",
        "weight_decay": 1e-5,
        "scheduler": "StepLR (step_size=40, gamma=0.3)",
        "loss": "CombinedLoss (Dice + BCE)",
        "pos_weight": pos_weight,
        "dropout": 0.1,
        "batch_size": t_loader.batch_size,
        "epochs": 200,
        "gradient_clipping": 1.0,
        "train_samples": len(t_loader.dataset), 
        "val_samples": len(v_loader.dataset),
        "wavelengths": wavelengths,
        "image_size": 672,
        "mask_size": 224
    },
    tags=["deep_decoder", "mae", "segmentation", "solar", "attention"]
)

# Log model architecture
wandb.watch(model, log="all", log_freq=100)

print("\n✓ WandB initialized successfully!")
print(f"  Project: solar-segmentation-deep-decoder")
print(f"  Run: {wandb.run.name}")
print(f"  URL: {wandb.run.url}\n")


#
from utils_2 import train_model

# Assicurati che il modello sia sul device corretto
model = model.to(device)

print("\n" + "="*70)
print("STARTING TRAINING WITH IMPROVED CONFIGURATION")
print("="*70)
print(f"Device: {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
print(f"\nTraining samples: {len(t_loader.dataset)}")
print(f"Validation samples: {len(v_loader.dataset)}")
print(f"Batch size: {t_loader.batch_size}")
print(f"Training batches: {len(t_loader)}")
print(f"Validation batches: {len(v_loader)}")
print("="*70 + "\n")

# Training
train_losses, val_losses, val_dice_scores = train_model(
    model=model,
    num_epochs=200,
    train_loader=t_loader,
    test_loader=v_loader,  # Usa validation loader invece di test
    optimizer=optimizer,
    device=device,
    criterion=criterion,
    scheduler=scheduler,
    dice_metric=dice_metric,  # Funzione di dice corretta
    post_pred=post_pred,
    post_label=post_label,
    model_save_path="/home/gpatane/checkpoints/seg_project/anotherNew_checkpoint.pth",
    wandb_run=wandb.run,  # Passa wandb run per logging
    max_grad_norm=1.0  # Gradient clipping
)

print("\n" + "="*70)
print("TRAINING COMPLETED!")
print("="*70)
print(f"Best validation Dice: {max(val_dice_scores):.4f}")
print(f"Best epoch: {val_dice_scores.index(max(val_dice_scores)) + 1}")
print(f"Final train loss: {train_losses[-1]:.4f}")
print(f"Final val loss: {val_losses[-1]:.4f}")
print(f"Final val Dice: {val_dice_scores[-1]:.4f}")
print("="*70)

# Log risultati finali su wandb
wandb.run.summary["best_val_dice"] = max(val_dice_scores)
wandb.run.summary["best_epoch"] = val_dice_scores.index(max(val_dice_scores)) + 1
wandb.run.summary["final_train_loss"] = train_losses[-1]
wandb.run.summary["final_val_loss"] = val_losses[-1]
wandb.run.summary["final_val_dice"] = val_dice_scores[-1]

# Crea e logga plot finale
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot delle loss
axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=4)
axes[0].plot(val_losses, label='Val Loss', marker='s', markersize=4)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Plot del Dice Score
axes[1].plot(val_dice_scores, label='Val Dice Score', marker='o', color='green', markersize=4)
axes[1].axhline(y=max(val_dice_scores), color='r', linestyle='--', alpha=0.5, label=f'Best: {max(val_dice_scores):.4f}')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Dice Score', fontsize=12)
axes[1].set_title('Validation Dice Score', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
wandb.log({"training_summary": wandb.Image(fig)})
plt.close()

# Chiudi wandb run
wandb.finish()
print("\n✓ WandB run finished successfully!")
