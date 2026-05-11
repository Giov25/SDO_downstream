import sys
import argparse
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb
import monai

from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete, Compose
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from unet_pytorch.model import UNet
from models import MAE_UNet_Segmentation, MAE_Seg_Advanced, MAE_Seg_Deformer, MAE_Seg_DeformerV2, MAE_FrozenEncoderSeg
from mae.models_mae_2 import mae_model_channel_masking_9ch_with_temporal_attn
from dataset import SDO_9Channel_Dataset
from utils_2 import train_model, load_checkpoint_with_channel_adaptation, freeze_encoder
from sunpy.map import Map


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="SDO Segmentation Training and Inference")

    parser.add_argument('--mode', type=str, choices=['train', 'test', 'resume'], required=True,
                        help="Run mode: 'train' to start training, 'test' to run inference, 'resume' to continue.")
    parser.add_argument('--model', type=str, default='MAE_2Channel',
                        help="Model architecture to use.")

    parser.add_argument('--zarr_path', type=str,
                        default="/home/gpatane/Dataset/zarr_file_magnetogram_1024_definitivo.zarr")
    parser.add_argument('--resume_checkpoints', type=str, default=None,
                        help="Path to checkpoint for resuming training (deprecated, use --checkpoint_path).")
    parser.add_argument('--mae_checkpoint', type=str,
                        default='/home/gpatane/SDO_downstream/mae_project/checkpoints/checkpoint_epoch55.pth')
    parser.add_argument('--model_path', type=str, default="/home/gpatane/checkpoints/")
    parser.add_argument('--save_plot', type=str,
                        default="/home/gpatane/checkpoints/predictions/pred.png")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="Path to model checkpoint for testing or resuming.")

    parser.add_argument('--load_pretrained', action='store_true',
                        help="Load MAE weights from checkpoint. If absent, train from scratch.")
    parser.add_argument('--freeze_encoder', action='store_true',
                        help="If set, freeze the encoder during training.")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--loss', type=str, choices=['DiceCELoss', 'TwerskyLoss'],
                        default='TwerskyLoss', help="Loss function to use.")
    parser.add_argument('--device', type=str, default="cuda:0")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_save_path(model_path, model_name, load_pretrained, freeze_encoder):
    """Return the checkpoint save/load path based on experiment type."""
    if load_pretrained and freeze_encoder:
        prefix = "Frozen_"
    elif load_pretrained:
        prefix = "Finetuning_"
    else:
        prefix = "Scratch_"
    return os.path.join(model_path, prefix + model_name + ".pth")


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_model(args, device):
    """Build and (optionally) initialise the model requested by --model."""

    model_name = args.model

    if model_name == 'MAE_UNet_Segmentation':
        mae_backbone = mae_model_channel_masking_9ch_with_temporal_attn().to(device)
        model = MAE_UNet_Segmentation(mae_backbone, num_classes=2).to(device)
        for param in model.encoder.parameters():
            param.requires_grad = False

    elif model_name == 'MAE_Seg_Advanced':
        mae_backbone = mae_model_channel_masking_9ch_with_temporal_attn().to(device)
        model = MAE_Seg_Advanced(mae_backbone, num_classes=2).to(device)
        for param in model.encoder.parameters():
            param.requires_grad = False

    elif model_name == 'MAE_2Channel':
        model = mae_model_channel_masking_9ch_with_temporal_attn(out_chans=2)

        # STEP 1: carica SEMPRE i pesi MAE pretrained (encoder backbone)
        if os.path.exists(args.mae_checkpoint):
            model = load_checkpoint_with_channel_adaptation(
                model, args.mae_checkpoint, in_chans=9, out_chans=2, device=device
            )
            print(f"✅ MAE pretrained weights loaded from: {args.mae_checkpoint}")
        else:
            print(f"⚠️  MAE checkpoint not found at {args.mae_checkpoint}, starting from scratch")

        # STEP 2: se resume o test, carica il checkpoint di segmentazione
        # (sovrascrive i pesi MAE con quelli aggiornati dal training)
        if args.mode in ('resume', 'test') and args.checkpoint_path:
            if os.path.exists(args.checkpoint_path):
                ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict)
                print(f"✅ Segmentation checkpoint loaded from: {args.checkpoint_path}")
            else:
                print(f"⚠️  Segmentation checkpoint not found at {args.checkpoint_path}")

        # STEP 3: freeze encoder se richiesto
        if args.freeze_encoder:
            model = freeze_encoder(model)
            print("❄️  Encoder frozen")

        model = model.to(device)
        
    elif model_name == 'MAE_Seg_Deformer':
        mae_backbone = mae_model_channel_masking_9ch_with_temporal_attn().to(device)

        if args.load_pretrained:
            checkpoint = torch.load(args.mae_checkpoint, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            mae_backbone.load_state_dict(state_dict, strict=False)
            print("✅ Loaded MAE pretrained weights from:", args.mae_checkpoint)
            model = MAE_Seg_Deformer(mae_backbone, num_classes=2).to(device)

            if args.freeze_encoder:
                for param in model.encoder.parameters():
                    param.requires_grad = False
                print("❄️  Encoder FROZEN — feature extraction mode")
            else:
                for param in model.encoder.parameters():
                    param.requires_grad = True
                print("🔥 Encoder TRAINABLE — fine-tuning mode")
        else:
            print("🆕 Training MAE backbone from scratch (no pretrained weights)")
            model = MAE_Seg_Deformer(mae_backbone, num_classes=2).to(device)
            for param in model.encoder.parameters():
                param.requires_grad = True

    elif model_name == 'MAE_Seg_DeformerV2':
        mae_backbone = mae_model_channel_masking_9ch_with_temporal_attn().to(device)

        if args.load_pretrained:
            checkpoint = torch.load(args.mae_checkpoint, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            mae_backbone.load_state_dict(state_dict, strict=False)
            print("✅ Loaded MAE pretrained weights from:", args.mae_checkpoint)

        model = MAE_Seg_DeformerV2(mae_backbone, num_classes=2).to(device)

        if args.freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("❄️  Encoder FROZEN — feature extraction mode")
        else:
            for param in model.encoder.parameters():
                param.requires_grad = True
            print("🔥 Encoder TRAINABLE — fine-tuning mode")

    elif model_name == 'MAE_FrozenEncoder':
        # Crea il backbone MAE con 2 canali di output (segmentazione)
        mae_backbone = mae_model_channel_masking_9ch_with_temporal_attn(out_chans=2)

        # Carica i pesi pretrained dalla ricostruzione e adatta l'head finale
        if os.path.exists(args.mae_checkpoint):
            mae_backbone = load_checkpoint_with_channel_adaptation(
                mae_backbone, args.mae_checkpoint, in_chans=9, out_chans=2, device=device
            )
            print(f"✅ MAE pretrained weights loaded from: {args.mae_checkpoint}")
        else:
            print(f"⚠️  MAE checkpoint not found at {args.mae_checkpoint}, decoder initialized randomly")

        # Costruisce il wrapper: encoder congelato (pretrained), decoder trainabile
        model = MAE_FrozenEncoderSeg(mae_backbone, freeze_encoder=args.freeze_encoder).to(device)

        # In resume/test carica il checkpoint di segmentazione (sovrascrive i pesi MAE
        # con quelli aggiornati dal training downstream)
        if args.mode in ('resume', 'test') and args.checkpoint_path:
            if os.path.exists(args.checkpoint_path):
                ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict)
                print(f"✅ Segmentation checkpoint loaded from: {args.checkpoint_path}")
            else:
                print(f"⚠️  Segmentation checkpoint not found at {args.checkpoint_path}")

    elif model_name == 'Unet':
        model = UNet(in_channels=9, out_channels=2).to(device)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    trainable_params, total_params = count_parameters(model)
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Total parameters     : {total_params:,}")
    print(f"Frozen parameters    : {total_params - trainable_params:,}")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    wavelengths = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram']
    train_years = list(range(2011, 2021))
    val_years   = list(range(2021, 2023))
    test_years  = list(range(2023, 2026))

    # ------------------------------------------------------------------
    # TRAIN / RESUME
    # ------------------------------------------------------------------
    if args.mode in ('train', 'resume'):
        print(f"Starting {'resume' if args.mode == 'resume' else 'training'} mode...")

        start_epoch  = 0
        best_dice    = 0.0
        wandb_run_id = None

        # Determine save/load path before touching WandB so we can recover run_id
        save_path = get_save_path(args.model_path, args.model, args.load_pretrained, args.freeze_encoder)
        print(f"💾 Checkpoint path: {save_path}")

        # ---- Resume: extract metadata from existing checkpoint ----
        if args.mode == 'resume':
            ckpt_path = args.checkpoint_path or save_path
            if os.path.exists(ckpt_path):
                print(f"Loading checkpoint metadata for resume: {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                start_epoch  = ckpt.get('epoch', 0)
                best_dice    = ckpt.get('val_dice', 0.0)
                wandb_run_id = ckpt.get('wandb_run_id', None)
                print(f"  Resuming from epoch {start_epoch}, best dice {best_dice:.4f}")
                if wandb_run_id:
                    print(f"  WandB run ID: {wandb_run_id}")
                # Training già completato: esce senza fare nulla
                if start_epoch >= args.epochs:
                    print(f"  Training already complete ({start_epoch}/{args.epochs} epochs). Exiting.")
                    return
            else:
                print(f"Warning: checkpoint not found at {ckpt_path}. Starting fresh.")
                args.mode = 'train'

        # ---- WandB init ----
        if wandb_run_id:
            run = wandb.init(project="seg-sdo", id=wandb_run_id, resume="allow",
                             config=vars(args))
            print(f"Resuming WandB run: {run.url}")
        else:
            run = wandb.init(project="seg-sdo", config=vars(args))
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*",   step_metric="epoch")
            wandb.define_metric("learning_rate", step_metric="epoch")
            print(f"Starting new WandB run: {run.url}")

        # NOTE: We read hyperparams from wandb.config so sweeps can override them,
        # but we keep the original args object for non-hyperparameter fields
        # (mode, paths, etc.) to avoid AttributeError later.
        wcfg = wandb.config

        # ---- Datasets ----
        train_ds = SDO_9Channel_Dataset(wcfg.zarr_path, train_years, wavelengths,
                                        target_size=wcfg.image_size)
        val_ds   = SDO_9Channel_Dataset(wcfg.zarr_path, val_years,   wavelengths,
                                        target_size=wcfg.image_size)
        train_loader = DataLoader(train_ds, batch_size=wcfg.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=wcfg.batch_size, shuffle=False,
                                  num_workers=4)

        # ---- Model ----
        # Build a simple namespace with the fields setup_model needs
        model_args = argparse.Namespace(
            model=args.model,
            mode=args.mode,
            load_pretrained=args.load_pretrained,
            freeze_encoder=args.freeze_encoder,
            mae_checkpoint=args.mae_checkpoint,
            checkpoint_path=args.checkpoint_path,
        )
        model = setup_model(model_args, device)

        # ---- Optimizer ----
        optimizer = torch.optim.AdamW(model.parameters(), lr=wcfg.lr, weight_decay=1e-2)

        # ---- Loss ----
        if wcfg.loss == 'DiceCELoss':
            criterion = DiceCELoss(to_onehot_y=True, softmax=True,
                                   lambda_dice=1.5, lambda_ce=1.0, include_background=False)
        else:  # TwerskyLoss
            criterion = monai.losses.TverskyLoss(to_onehot_y=True, softmax=True,
                                                 alpha=0.5, beta=0.5, include_background=False)

        dice_metric   = DiceMetric(include_background=False, reduction="mean")
        dice_metric_T = DiceMetric(include_background=True,  reduction="mean")

        # ---- Schedulers ----
        warmup_epochs  = min(20, max(1, wcfg.epochs // 10))
        cosine_epochs  = max(1, wcfg.epochs - warmup_epochs)
        warmup_sched   = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_sched   = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)
        scheduler      = SequentialLR(optimizer,
                                      schedulers=[warmup_sched, cosine_sched],
                                      milestones=[warmup_epochs])

        # ---- Post-processing ----
        post_pred  = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([AsDiscrete(to_onehot=2)])

        # ---- Restore full checkpoint state if resuming ----
        if args.mode == 'resume':
            ckpt_path = args.checkpoint_path or save_path
            if os.path.exists(ckpt_path):
                print("Restoring model / optimizer / scheduler state...")
                ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict)

                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    print("  Optimizer state restored")
                if ckpt.get('scheduler_state_dict') is not None:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    print("  Scheduler state restored")

        # ---- Train ----
        train_model(
            model=model,
            num_epochs=wcfg.epochs,
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
            start_epoch=start_epoch,
            best_dice=best_dice,
        )

    # ------------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------------
    elif args.mode == 'test':
        print("Starting inference mode...")

        test_ds     = SDO_9Channel_Dataset(args.zarr_path, test_years, wavelengths,
                                           target_size=args.image_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)

        model = setup_model(args, device)

        # Determine checkpoint to load
        if args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
        else:
            checkpoint_path = os.path.join(
                args.model_path,
                f"{args.image_size}_{args.model}.pth"
            )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Test checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt       = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.eval()

        print(f"Running inference with model: {args.model}")

        dice_metric   = DiceMetric(include_background=False, reduction="mean")
        dice_metric_T = DiceMetric(include_background=True,  reduction="mean")

        from utils_2 import run_and_plot_predictions_all_channels
        run_and_plot_predictions_all_channels(
            model, test_loader, device,
            dice_metric=dice_metric,
            dice_metric_T=dice_metric_T,
            n_images=5,
            threshold=0.5,
            use_wandb=False,
            save_path=args.save_plot,
        )


if __name__ == "__main__":
    main()