import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete, Compose
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from unet_pytorch.model import UNet
# Import your custom modules
from models import MAE_UNet_Segmentation, MAE_Seg_Advanced
from mae.models_mae_2 import mae_model_channel_masking_9ch_with_temporal_attn
from dataset import SDO_9Channel_Dataset
from utils_2 import train_model  # Ensure this matches your project structure

def get_args():
    parser = argparse.ArgumentParser(description="SDO Segmentation Training and Inference")
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'resume'], required=True, 
                        help="Run mode: 'train' to start training, 'test' to run inference.")
    parser.add_argument('--model' , type=str, default='MAE_UNet_Segmentation', help="Model architecture to use.")
    # Paths
    parser.add_argument('--zarr_path', type=str, default="/home/gpatane/Dataset/zarr_file_magnetogram.zarr")
    parser.add_argument('--mae_checkpoint', type=str, default='/home/gpatane/solar_project/SDO_downstream/mae_project/checkpoints/512/checkpoint_epoch_100.pth')
    parser.add_argument('--model_path', type=str, default="/home/gpatane/checkpoints/seg_project/checkpoints/")
    parser.add_argument('--save_plot', type=str, default="inference_results.png")
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--device', type=str, default="cuda:2")
    
    return parser.parse_args()

def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def setup_model(args, device):
    # Initialize MAE backbone
    if args.model == 'MAE_UNet_Segmentation':
        mae_backbone = mae_model_channel_masking_9ch_with_temporal_attn().to(device)
        model = MAE_UNet_Segmentation(mae_backbone, num_classes=2).to(device)
    
        # OPZIONALE: Se vuoi freezare l'encoder all'inizio per stabilizzare il decoder
        for param in model.encoder.parameters():
            param.requires_grad = False

        
    elif args.model == 'MAE_Seg_Advanced':
        mae_backbone = mae_model_channel_masking_9ch_with_temporal_attn().to(device)
        model = MAE_Seg_Advanced(mae_backbone, num_classes=2).to(device)
        
        # OPZIONALE: Se vuoi freezare l'encoder all'inizio per stabilizzare il decoder
        for param in model.encoder.parameters():
            param.requires_grad = False
            
    elif args.model == 'Unet':
        model = UNet(
            in_channels=9,
            out_channels=2,
        ).to(device)
        
    trainable_params, total_params = count_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    if args.mode == 'test':
        print(f"Loading weights for inference from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        # Handle cases where checkpoint is a dict or a direct state_dict
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    
    return model

def main():
    args = get_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # --- Data Setup ---
    wavelengths = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram']
    train_years = list(range(2010, 2021))
    val_years   = list(range(2021, 2023))
    test_years  = list(range(2023, 2026))

    if args.mode != 'test':
        print("Starting raining/Sweep mode... ")
        run = wandb.init(project="seg-sdo", config=args)
        args = wandb.config
        
        train_ds = SDO_9Channel_Dataset(args.zarr_path, train_years, wavelengths, target_size=args.image_size)
        val_ds = SDO_9Channel_Dataset(args.zarr_path, val_years, wavelengths, target_size=args.image_size)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        model = setup_model(args, device)
        # 5. Optimizer & Criterion (Usano args.lr aggiornato)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        criterion = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.5, lambda_ce=1.0, include_background=False)
        
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        dice_metric_T = DiceMetric(include_background=True, reduction="mean")

        # 6. Schedulers
        warmup_epochs = 20
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        
        # 7. Post-processing
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([AsDiscrete(to_onehot=2)]) 

        # 8. Training Loop
        if args.mode == 'resume':
            print(f"Resuming training from checkpoint: {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)        
        
        # Chiamata unica a train_model (gestisce sia resume che train normale)
        save_path = args.model_path if args.mode == 'resume' else os.path.join(args.model_path, (args.model+".pth"))
        
        train_model(
            model=model,
            num_epochs=args.epochs,
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

    elif args.mode == 'test':
        print("Starting Inference Mode...")
        # (La logica del test rimane fuori da wandb.init se non vuoi loggare il test)
        test_ds = SDO_9Channel_Dataset(args.zarr_path, test_years, wavelengths, target_size=args.image_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        model = setup_model(args, device)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        dice_metric_T = DiceMetric(include_background=True, reduction="mean")
        
        from utils_2 import testing
        val_metric, val_metric_T = testing(model, test_loader, device, dice_metric, dice_metric_T)
        print(f"Test Dice (no bg): {val_metric:.4f}, Test Dice (with bg): {val_metric_T:.4f}")

if __name__ == "__main__":
    main()
