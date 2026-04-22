import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
import monai
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete, Compose
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from unet_pytorch.model import UNet
# Import your custom modules
from models import MAE_UNet_Segmentation, MAE_Seg_Advanced, MAE_Seg_Deformer
from mae.models_mae_2 import mae_model_channel_masking_9ch_with_temporal_attn
from dataset import SDO_9Channel_Dataset
from utils_2 import train_model, load_checkpoint_with_channel_adaptation  # Ensure this matches your project structure
from sunpy.map import Map
def get_args():
    parser = argparse.ArgumentParser(description="SDO Segmentation Training and Inference")
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'resume'], required=True, 
                        help="Run mode: 'train' to start training, 'test' to run inference.")
    parser.add_argument('--model' , type=str, default='MAE_2Channel', help="Model architecture to use.")
    # Paths
    
    parser.add_argument('--zarr_path', type=str, default="/home/gpatane/Dataset/zarr_file_magnetogram_1024_definitivo.zarr")
    parser.add_argument('--resume_checkpoints', type=str, default=None, help="Path to checkpoint for resuming training.")
    parser.add_argument('--mae_checkpoint', type=str, default='/home/gpatane/checkpoints/mae_ckp/best_model.pth')
    parser.add_argument('--model_path', type=str, default="/home/gpatane/checkpoints/")
    parser.add_argument('--save_plot', type=str, default="/home/gpatane/checkpoints/predictions/pred.png")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to model checkpoint for testing or resuming.")
    parser.add_argument('--load_pretrained', action='store_true', 
                            help="Se presente, carica i pesi MAE dal checkpoint. Se assente, allena da zero (Scratch).")
    parser.add_argument('--freeze_encoder', action='store_true', help="If set, freeze the encoder during training.")    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--loss', type=str, choices=['DiceCELoss', 'TwerskyLoss'], default='TwerskyLoss', help="Loss function to use.")
    parser.add_argument('--device', type=str, default="cuda:0")
    
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
        for param in model.encoder.parameters():
            param.requires_grad = False

        
    elif args.model == 'MAE_Seg_Advanced':
        mae_backbone = mae_model_channel_masking_9ch_with_temporal_attn().to(device)
        model = MAE_Seg_Advanced(mae_backbone, num_classes=2).to(device)
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    elif args.model == 'MAE_2Channel':
        from utils_2 import load_checkpoint_with_channel_adaptation, freeze_encoder
        import sys
        sys.path.append('/home/gpatane/SDO_downstream/mae_project')
        from mae.models_mae_2 import mae_model_channel_masking_9ch_with_temporal_attn
        
        # Crea il modello con 2 canali di output
        model = mae_model_channel_masking_9ch_with_temporal_attn(out_chans=2)
        
        # Determina quale checkpoint caricare
        checkpoint_to_load = None
        if args.mode == 'test' and args.checkpoint_path:
            # Per il test, usa il checkpoint salvato del modello
            checkpoint_to_load = args.checkpoint_path
            print(f"[MAE_2Channel] Test mode: carico da {checkpoint_to_load}")
        elif args.load_pretrained:
            # Per il training, usa il checkpoint MAE pretrained
            checkpoint_to_load = args.mae_checkpoint
            print(f"[MAE_2Channel] Train mode con pretrained: carico da {checkpoint_to_load}")
        else:
            # Training from scratch - nessun checkpoint
            print(f"[MAE_2Channel] Training da zero (no pretrained)")
        
        # Se c'è un checkpoint da caricare
        if checkpoint_to_load and os.path.exists(checkpoint_to_load):
            model = load_checkpoint_with_channel_adaptation(
                model, 
                checkpoint_to_load, 
                in_chans=9, 
                out_chans=2, 
                device=device
            )
        
        # Applica freeze se richiesto
        if args.freeze_encoder:
            model = freeze_encoder(model)
            print("[MAE_2Channel] ❄️  Encoder congelato")
        
        model = model.to(device)

    
    elif args.model == 'MAE_Seg_Deformer':
        mae_backbone = mae_model_channel_masking_9ch_with_temporal_attn().to(device)
        
        # CASO 1: Pretrained weights caricati
        if args.load_pretrained:
            checkpoint = torch.load(args.mae_checkpoint, map_location=device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            mae_backbone.load_state_dict(state_dict, strict=False)
            print("✅ Loaded MAE pretrained weights from:", args.mae_checkpoint)
            
            model = MAE_Seg_Deformer(mae_backbone, num_classes=2).to(device)
            
            # CASO 1a: Encoder FROZEN (Feature Extraction)
            if args.freeze_encoder:
                for param in model.encoder.parameters():
                    param.requires_grad = False
                print("❄️  Encoder FROZEN - Feature Extraction mode")
            # CASO 1b: Encoder TRAINABLE (Fine-tuning)
            else:
                for param in model.encoder.parameters():
                    param.requires_grad = True
                print("🔥 Encoder TRAINABLE - Fine-tuning mode")
        
        # CASO 2: Training FROM SCRATCH
        else:
            print("🆕 Training MAE backbone FROM SCRATCH (no pretrained weights)")
            model = MAE_Seg_Deformer(mae_backbone, num_classes=2).to(device)
            for param in model.encoder.parameters():
                param.requires_grad = True
        
            
    elif args.model == 'Unet':
        model = UNet(
            in_channels=9,
            out_channels=2,
        ).to(device)
        
    trainable_params, total_params = count_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    
    return model

def main():
    args = get_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # --- Data Setup ---
    wavelengths = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram']
    train_years = list(range(2011, 2021,3))
    val_years   = list(range(2021, 2023,3))
    #test_years  = list(range(2023, 2026))   #test_years  = list(range(2023, 2024))   

    if args.mode != 'test':
        print("Starting training/Sweep mode... ")
        
        # Variabili per riprendere il training
        start_epoch = 0
        best_dice = 0.0
        wandb_run_id = None
        checkpoint_to_load = None
        
        # Se è in modalità resume, carica prima il checkpoint per recuperare run_id
        if args.mode == 'resume':
            # Determina quale checkpoint caricare in base alla configurazione
            if args.load_pretrained and args.freeze_encoder:
                checkpoint_to_load = os.path.join(args.model_path + "Frozen_" + args.model + ".pth")
            elif args.load_pretrained:
                checkpoint_to_load = os.path.join(args.model_path + "Finetuning_" + args.model + ".pth")
            else:
                checkpoint_to_load = os.path.join(args.model_path + "Scratch_" + args.model + ".pth")
            
            if os.path.exists(checkpoint_to_load):
                print(f"Loading checkpoint for resume: {checkpoint_to_load}")
                checkpoint = torch.load(checkpoint_to_load, map_location=device, weights_only=False)
                start_epoch = checkpoint.get('epoch', 0)
                best_dice = checkpoint.get('val_dice', 0.0)
                wandb_run_id = checkpoint.get('wandb_run_id', None)
                print(f"  - Resuming from epoch {start_epoch}")
                print(f"  - Best dice score: {best_dice:.4f}")
                if wandb_run_id:
                    print(f"  - WandB run ID: {wandb_run_id}")
            else:
                print(f"Warning: Checkpoint not found at {checkpoint_to_load}. Starting from scratch.")
                args.mode = 'train'
        
        # Inizializza WandB (con resume se necessario)
        if wandb_run_id:
            run = wandb.init(
                project="seg-sdo",
                id=wandb_run_id,
                resume="allow",
                config=args
            )
            
            print(f"Resuming WandB run: {run.url}")
        else:
            run = wandb.init(project="seg-sdo", config=args)
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("learning_rate", step_metric="epoch")
            print(f"Starting new WandB run: {run.url}")
        
        args = wandb.config
        
        train_ds = SDO_9Channel_Dataset(args.zarr_path, train_years, wavelengths, target_size=args.image_size)
        val_ds = SDO_9Channel_Dataset(args.zarr_path, val_years, wavelengths, target_size=args.image_size)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        model = setup_model(args, device)
        # 5. Optimizer & Criterion (Usano args.lr aggiornato)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        if args.loss == 'DiceCELoss':
            criterion = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.5, lambda_ce=1.0, include_background=False)
        elif args.loss == 'TwerskyLoss':
            criterion = monai.losses.TverskyLoss(to_onehot_y=True, softmax=True, alpha=0.5, beta=0.5, include_background=False)
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

        # 8. Carica checkpoint se in modalità resume
        if args.mode == 'resume' and checkpoint_to_load and os.path.exists(checkpoint_to_load):
            print(f"Loading model weights and optimizer state from checkpoint...")
            checkpoint = torch.load(checkpoint_to_load, map_location=device, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            
            # Ripristina optimizer e scheduler
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("  - Optimizer state restored")
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("  - Scheduler state restored")        
        
        # Determina il path di salvataggio in base al tipo di esperimento
        if args.load_pretrained and args.freeze_encoder:
            save_path = os.path.join(args.model_path + "Frozen_" + args.model + ".pth")
            print(f"💾 Checkpoint will be saved as: Frozen_{args.model}.pth")
        elif args.load_pretrained:
            save_path = os.path.join(args.model_path + "Finetuning_" + args.model + ".pth")
            print(f"💾 Checkpoint will be saved as: Finetuning_{args.model}.pth")
        else:
            save_path = os.path.join(args.model_path + "Scratch_" + args.model + ".pth")
            print(f"💾 Checkpoint will be saved as: Scratch_{args.model}.pth")
        
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
            start_epoch=start_epoch,
            best_dice=best_dice,
        )

    elif args.mode == 'test':
        print("Starting Inference Mode...")
        # (La logica del test rimane fuori da wandb.init se non vuoi loggare il test)
        test_ds = SDO_9Channel_Dataset(args.zarr_path, test_years, wavelengths, target_size=args.image_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        
        model = setup_model(args, device)
        if args.checkpoint_path is None:
            checkpoint_path = os.path.join(args.model_path, str(args.image_size) + "_"+args.model + ".pth")
        else:
            checkpoint_path = args.checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)      
        print(f"Run inference using model {args.model}")
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        dice_metric_T = DiceMetric(include_background=True, reduction="mean")
        
        from utils_2 import testing, test_and_plot, run_and_plot_predictions_all_channels
        # metric_dice, metric_dice_T, metric_iou, metric_iou_T = testing(model, test_loader, device, dice_metric, dice_metric_T)
        # print(f"Test Dice Metric (without background): {metric_dice:.4f}")
        # print(f"Test Dice Metric (with background): {metric_dice_T:.4f}")
        # print(f"Test IoU Metric (without background): {metric_iou:.4f}")
        # print(f"Test IoU Metric (with background): {metric_iou_T:.4f}")
        run_and_plot_predictions_all_channels(model, test_loader, device, dice_metric=dice_metric, dice_metric_T=dice_metric_T, n_images=5, threshold=0.5, use_wandb=False, save_path=args.save_plot)
        

if __name__ == "__main__":
    main()
