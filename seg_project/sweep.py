import wandb
import torch
import monai
import os

from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from mae.models_mae_2 import mae_model_channel_masking_9ch_with_temporal_attn
from dataset import SDO_9Channel_Dataset
from utils_2 import train_model, load_checkpoint_with_channel_adaptation, freeze_encoder
from monai.losses import DiceCELoss

# Configurazione sweep
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/dice", "goal": "maximize"},
    "parameters": {
        "lr":         {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "batch_size": {"values": [1, 2, 4]},
        "loss":       {"values": ["DiceCELoss", "TwerskyLoss"]},
        "freeze_encoder": {"values": [True, False]},
    }
}

# Costanti fisse
FIXED = {
    "zarr_path":      "/home/gpatane/Dataset/zarr_file_magnetogram_1024_definitivo.zarr",
    "mae_checkpoint": "/home/gpatane/SDO_downstream/mae_project/checkpoints/checkpoint_epoch55.pth",
    "model_path":     "/home/gpatane/checkpoints/sweep/",
    "image_size": 1024,
    "epochs":         5,    # meno epoche durante sweep
    "device":         "cuda:0",
}

WAVELENGTHS = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram']
TRAIN_YEARS = list(range(2011, 2021))
VAL_YEARS   = list(range(2021, 2023))


def build_model(cfg, device):
    model = mae_model_channel_masking_9ch_with_temporal_attn(out_chans=2)

    if os.path.exists(FIXED["mae_checkpoint"]):
        model = load_checkpoint_with_channel_adaptation(
            model, FIXED["mae_checkpoint"], in_chans=9, out_chans=2, device=device
        )
        print(f"✅ MAE pretrained loaded")

    if cfg.freeze_encoder:
        model = freeze_encoder(model)
        print("❄️  Encoder frozen")

    return model.to(device)


def train_one_run():
    run = wandb.init()
    cfg = wandb.config
    device = torch.device(FIXED["device"] if torch.cuda.is_available() else "cpu")

    # Dataset
    train_ds = SDO_9Channel_Dataset(FIXED["zarr_path"], TRAIN_YEARS, WAVELENGTHS,
                                    target_size=FIXED["image_size"])
    val_ds   = SDO_9Channel_Dataset(FIXED["zarr_path"], VAL_YEARS,   WAVELENGTHS,
                                    target_size=FIXED["image_size"])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, num_workers=4)

    # Model
    model = build_model(cfg, device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)

    # Loss
    if cfg.loss == "DiceCELoss":
        criterion = DiceCELoss(to_onehot_y=True, softmax=True,
                               lambda_dice=1.5, lambda_ce=1.0, include_background=False)
    else:
        criterion = monai.losses.TverskyLoss(to_onehot_y=True, softmax=True,
                                             alpha=0.5, beta=0.5, include_background=False)

    # Scheduler
    warmup_epochs = min(10, max(1, FIXED["epochs"] // 10))
    cosine_epochs = max(1, FIXED["epochs"] - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6),
        ],
        milestones=[warmup_epochs]
    )

    # Metrics & post-processing
    dice_metric   = DiceMetric(include_background=False, reduction="mean")
    dice_metric_T = DiceMetric(include_background=True,  reduction="mean")
    post_pred     = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label    = Compose([AsDiscrete(to_onehot=2)])

    # Save path univoco per ogni run dello sweep
    os.makedirs(FIXED["model_path"], exist_ok=True)
    save_path = os.path.join(FIXED["model_path"], f"sweep_{run.id}.pth")

    train_model(
        model=model,
        num_epochs=FIXED["epochs"],
        train_loader=train_loader,
        test_loader=val_loader,
        optimizer=optimizer,
        device=device,
        criterion=criterion,
        scheduler=scheduler,
        dice_metric=dice_metric,
        post_pred=post_pred,
        post_label=post_label,
        model_save_path=save_path,
        wandb_run=run,
        max_grad_norm=1.0,
        dice_metric_T=dice_metric_T,
    )


if __name__ == "__main__":
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="seg-sdo")
    wandb.agent(sweep_id, function=train_one_run, count=20)