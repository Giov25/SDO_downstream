import sys
import torch.nn as nn
import re
import torch
from torch import optim 
from torch.utils.data import DataLoader
import numpy as np

from monai.transforms import (
    Resized,
    Compose,
    EnsureTyped,
    AsDiscrete,
    ToTensord
)

from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
import random

from models import  MAESegmentationModel
sys.path.append('./mae')
sys.path.append('..')
from mae.models_mae_2 import mae_for_segmentation_2, mae_model_for_pretraining

from dataset import SDOMosaicZarrDataset_2, PhotosphereDataset
from utils import train_model, get_class_weights, train_one_epoch

import matplotlib.pyplot as plt
import argparse
import wandb
import time
import os
import torch.nn.functional as F

class DicePosWeightLoss(nn.Module):
    def __init__(self, dice_loss, bce_loss, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_loss = dice_loss
        self.bce_loss = bce_loss
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return self.dice_weight * dice + self.bce_weight * bce

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        
    def forward(self, logits, target):
        # BCE con logits (più stabile numericamente)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        # Sigmoid per dice
        pred = torch.sigmoid(logits)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + 1e-8) / (pred_flat.sum() + target_flat.sum() + 1e-8)
        dice_loss = 1 - dice
        
        total_loss = bce + self.dice_weight * dice_loss
        # Debug
        print(f"BCE: {bce:.4f}, Dice: {dice_loss:.4f}, Total: {total_loss:.4f}")
        
        return total_loss
def Focal_loss(inputs, targets, alpha=0.5, gamma=3.0):
    BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets)
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate Segmentation task from MAE task on SDO data with WandB logging.')
    parser.add_argument('--folder_path', type=str, default="/home/gpatane/Dataset/zarr_file_magnetogram.zarr",
                        help='Path to the Zarr dataset file.')
    parser.add_argument('--wavelengths', type=list, default = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram', "Ic_noLimbDark"],
                        help='List of wavelengths to use for training')

    parser.add_argument('--model_save_path', type=str, default="/home/gpatane/checkpoints/seg_project",
                        help='Path to save the trained model.')
    parser.add_argument('--load_model_path', type=str, default="/home/gpatane/checkpoints/mae_project/mae_sdo_patch_14.pth",
                        help='Path to load a pre-trained model for evaluation or resuming training.')
    parser.add_argument('--start_period', type=int, required=True, help='First year (e.g., 2010 2011 2012).')
    parser.add_argument('--end_period', type=int, required=True, help='Last year (e.g., 2013 2014).')
    parser.add_argument('--mae_weight_path', type=str, default="/home/gpatane/checkpoints/mae_project/train_magnetogram.pth",
                        help='Path to the pre-trained MAE weights for segmentation task.')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze the MAE encoder during training.')
    # --- Training Params ---
    parser.add_argument('--seed', type=int, default=5, help='Random seed.')
    parser.add_argument('--train_test_split_ratio', type=float, default=0.7,
                        help='Ratio of years for training.')
    parser.add_argument('--batch_size', type=int, default=3, help='Training and validation batch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Optimizer learning rate.') # Example LR
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='Optimizer type.') # Example optimizer choice
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Optimizer weight decay.') # Example weight decay
    parser.add_argument('--device', type=str, default="cuda:1",
                        help='Device (e.g., "cpu", "cuda", "cuda:0").')
    parser.add_argument('--loss', type=str, default="dice_bce", choices=['bce', 'focal', 'dice', 'dice_bce', 'pos_weight', "dice_posweight", "dice_ce"], help='Loss function to use for training.')
    # --- Action Flags ---
    parser.add_argument('--train', action='store_true', help='Run training.')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on one image after training or loading.')
    parser.add_argument('--disable_scheduler', action='store_true', help='Disable the CosineAnnealingLR scheduler.')
    parser.add_argument('--min', type=float, default=10, help='Minimum value for class weights in BCE with pos_weight loss.')


    parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging.')
    parser.add_argument('--wandb_project', type=str, default='sdo-seg-training', help='WandB project name.')
    parser.add_argument('--wandb_entity', type=str, default='patane-giovanni-universit-catania', help='WandB entity (username or team).')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Custom WandB run name.')


    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # --- Initialize WandB ---
    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            print("WandB is requested but not installed. Disabling WandB.")
            args.use_wandb = False
        else:
            try:
                wandb_run = wandb.init(
                    project="sdo-segmentation",
                    name=args.wandb_run_name,
                    config={
                        "learning_rate": args.learning_rate,
                        "epochs": args.num_epochs,
                        "batch_size": args.batch_size,
                        "model": "MAE-Segmentation"
                    },
                    settings=wandb.Settings(
                        _disable_stats=True,  
                        _disable_meta=True,
                    ))
                print(f"WandB logging enabled. Run: {wandb_run.url}")
            except Exception as e:
                print(f"Error initializing WandB: {e}. Disabling WandB.")
                args.use_wandb = False
                wandb_run = None
                
    # Set Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if "cuda" in args.device and not torch.cuda.is_available():
        print(f"Warning: CUDA device '{args.device}' requested but unavailable. Using CPU.")
        device = torch.device("cpu")
        args.device = "cpu"
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    if "cuda" in args.device:
        torch.cuda.manual_seed_all(args.seed)
    
    # Define Model
    
    mae_model = mae_for_segmentation_2().to(device)
    #mae_model = mae_model_for_pretraining().to(device)
    mae_model.load_state_dict(torch.load(args.mae_weight_path, map_location=device))
    segmentation_model = MAESegmentationModel(mae_model, num_classes=2, freeze_encoder=args.freeze_encoder).to(device)

    
    if wandb_run:
         num_params = sum(p.numel() for p in segmentation_model.parameters() if p.requires_grad)
         print(f"Model has {num_params:,} trainable parameters.")
         wandb_run.summary["trainable_params"] = num_params


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    transform = Compose([
    EnsureTyped(keys=["image", "mask"]),
    ToTensord(keys=["image", "mask"]),
    Resized(keys="image", spatial_size=[672, 672], mode="area"),
    Resized(keys="mask", spatial_size=[224, 224], mode="area")
])

    all_years = list(range(args.start_period, args.end_period))
    random.shuffle(all_years)

    train_split = int(0.7 * len(all_years))
    val_split = int(0.85 * len(all_years))

    train_years = sorted(all_years[:train_split])
    val_years = sorted(all_years[train_split:val_split])
    test_years = sorted(all_years[val_split:])
    
    tr_dataset = SDOMosaicZarrDataset_2(args.folder_path, train_years, args.wavelengths, target_size=224, transform=transform)
    vl_dataset = SDOMosaicZarrDataset_2(args.folder_path, val_years, args.wavelengths, target_size=224, transform=transform)
    ts_dataset = SDOMosaicZarrDataset_2(args.folder_path, test_years, args.wavelengths, target_size=224, transform=transform)
    t_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    v_loader = DataLoader(vl_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    ts_loader = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    
    # Training
    if args.train:
        
        segmentation_model = segmentation_model.to(device)

        if args.loss == "focal":
            criterion = Focal_loss
            print("Using Focal Loss")
        elif args.loss == "dice":
            criterion = DiceLoss(sigmoid=True, include_background=False, reduction="mean")
            print("Using Dice Loss")
        elif args.loss == "dice_bce":
            criterion = DiceBCELoss()
            print("Using Dice + BCE Loss")
        elif args.loss == "pos_weight":
            pos_weight = get_class_weights(t_loader, minimum = args.min)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
            print(f"Using BCE Loss with pos_weight: {pos_weight:.2f}")
        elif args.loss == "dice_posweight":
            pos_weight = get_class_weights(t_loader, minimum = args.min)
            dice_loss = DiceLoss(sigmoid=False, include_background=True, reduction="mean")
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
            criterion = DicePosWeightLoss(dice_loss, bce_loss, dice_weight=0.5, bce_weight=0.5)
            print(f"Using Dice + BCE Loss with pos_weight: {pos_weight:.2f}")
        elif args.loss == "dice_ce":
                criterion = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
                #criterion = DiceCELoss(include_background=True, to_onehot_y=True, sigmoid=True, squared_pred=False, lambda_dice=1.0, lambda_ce=1.0 )
                
                print("Using Dice + Cross Entropy Loss")
        else:
            criterion = nn.BCEWithLogitsLoss()  # Per segmentazione binaria
            print("Using BCE Loss")
        optimizer = torch.optim.SGD(segmentation_model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=args.num_epochs * len(t_loader))
        #add scheduler steplr
        

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) if not args.disable_scheduler else None
        # Post-processing per le metriche
        post_pred = Compose([AsDiscrete(argmax=False, threshold=0.9)])
        post_label = Compose([AsDiscrete(to_onehot=None)])
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)

        train_model(
            model=segmentation_model,
            num_epochs=args.num_epochs,
            train_loader=t_loader,
            test_loader=ts_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            dice_score=dice_metric,
            post_pred=post_pred,
            post_label=post_label,
            model_save_path=os.path.join(args.model_save_path,(args.wandb_run_name+".pth")),  
            save_best_only=False,
            wandb_run=wandb_run,
            #patience=100  
        )
        
        input = next(iter(t_loader))
        out = segmentation_model(input["image"].to(device))
        out = post_pred(out)
        # plot result on wandb
        if wandb_run:
            wandb_run.finish()
        else:
            print("WandB logging is disabled. Not logging training metrics.")
if __name__ == "__main__":
    main()