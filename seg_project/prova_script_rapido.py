import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete, Compose
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Import your custom modules
from models import MAESegmentationModel, MAE_UNet_Segmentation
from mae.models_mae_2 import mae_model_channel_masking_9ch_with_temporal_attn
from dataset import SDO_9Channel_Dataset
from utils_2 import train_model  # Ensure this matches your project structure

def get_args():
    parser = argparse.ArgumentParser(description="SDO Segmentation Training and Inference")
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'resume'], required=True, 
                        help="Run mode: 'train' to start training, 'test' to run inference.")
    
    # Paths
    parser.add_argument('--zarr_path', type=str, default="/home/gpatane/Dataset/zarr_file_magnetogram.zarr")
    parser.add_argument('--mae_checkpoint', type=str, default='/home/gpatane/solar_project/SDO_downstream/mae_project/checkpoints/512/checkpoint_epoch_100.pth')
    parser.add_argument('--model_path', type=str, default="/home/gpatane/checkpoints/seg_project/checkpoints/prova_allenamento.pth")
    parser.add_argument('--save_plot', type=str, default="inference_results.png")
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--device', type=str, default="cuda:2")
    
    return parser.parse_args()

def setup_model(args, device):
    # Initialize MAE backbone
    mae_backbone = mae_model_channel_masking_9ch_with_temporal_attn().to(device)
    

    model = MAE_UNet_Segmentation(mae_backbone, num_classes=2).to(device)

    # OPZIONALE: Se vuoi freezare l'encoder all'inizio per stabilizzare il decoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
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
        train_ds = SDO_9Channel_Dataset(args.zarr_path, train_years, wavelengths, target_size=args.image_size)
        val_ds = SDO_9Channel_Dataset(args.zarr_path, val_years, wavelengths, target_size=args.image_size)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        train_ds = SDO_9Channel_Dataset(args.zarr_path, train_years, wavelengths, target_size=args.image_size)
        val_ds = SDO_9Channel_Dataset(args.zarr_path, val_years, wavelengths, target_size=args.image_size)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_ds = SDO_9Channel_Dataset(args.zarr_path, test_years, wavelengths, target_size=args.image_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Model Setup ---
    model = setup_model(args, device)

    # --- Execution ---
    if args.mode != 'test':
        print("Starting Training Mode...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        criterion = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.5, lambda_ce=1.0, include_background=False)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        warmup_epochs = 5
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            total_iters=warmup_epochs
        )

        # Scheduler 2: Cosine Decay
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs - warmup_epochs, 
            eta_min=1e-6 # LR minimo alla fine del training
        )

        # Unione dei due
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_epochs]
        )
        
        post_pred = Compose([
        AsDiscrete(argmax=True, to_onehot=2) # [B,2,H,W] logits → [B,2,H,W] one-hot
        ])
        
        post_label = Compose([
        AsDiscrete(to_onehot=2) # [B,1,H,W] indices {0,1} → [B,2,H,W] one-hot
        ]) 
        run = wandb.init(
            project="seg-sdo",           # Nome del progetto
            name="esperimento-2",               # Nome del run (opzionale)
            config={                            # Configurazione/hyperparameters
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "model": "MAE_UNet_Segmentation",
                
                
                # ... altri parametri
            }
        )
        if args.mode == 'resume':
            print(f"Resuming training from checkpoint: {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
            # Handle cases where checkpoint is a dict or a direct state_dict
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)        
            train_model(
                model=model,
                num_epochs=args.epochs,
                train_loader=train_loader,
                test_loader=val_loader, # Usa validation loader invece di test
                optimizer=optimizer,
                device=device,
                criterion=criterion,
                scheduler=scheduler,
                dice_metric=dice_metric, # Funzione di dice corretta
                post_pred=post_pred,
                post_label=post_label,
                model_save_path=args.model_path,
                wandb_run=run, # Passa wandb run per logging
                max_grad_norm=1.0 # Gradient clipping
                )
            
        else:
            train_model(
                model=model,
                num_epochs=args.epochs,
                train_loader=train_loader,
                test_loader=val_loader, # Usa validation loader invece di test
                optimizer=optimizer,
                device=device,
                criterion=criterion,
                scheduler=scheduler,
                dice_metric=dice_metric, # Funzione di dice corretta
                post_pred=post_pred,
                post_label=post_label,
                model_save_path=args.model_path,
                wandb_run=run, # Passa wandb run per logging
                max_grad_norm=1.0 # Gradient clipping
                )



    elif args.mode == 'test':
        print("Starting Inference Mode...")
        from utils_2 import test_and_plot # Assumes the function is in this file
        dice_list, mean_dice = test_and_plot(
            model, 
            train_loader, 
            device, 
            n_images=args.batch_size, 
            save_path=args.save_plot
        )

if __name__ == "__main__":
    main()
