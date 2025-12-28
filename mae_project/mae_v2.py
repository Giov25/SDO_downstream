import sys
import torch.nn as nn
import torch
import numpy as np
import random
from utils import (train_model, 
                    run_one_image)
from dataset import SDOMosaicZarrDataset, MC_SolarDataset
import wandb
from torch.utils.data import DataLoader
sys.path.append('./mae')
sys.path.append('..')
torch.manual_seed(1)
from functools import partial


from mae.models_mae_2 import MaskedAutoencoderViT, mae_model_for_pretraining, mae_model_for_pretraining_2x2
import argparse
import time
import os
from torch import optim
from monai.transforms import (
    Resized,
    Compose,
    EnsureTyped,
    ToTensord
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate Masked Autoencoder on SDO data with WandB logging.')

    # --- Data/Model Paths ---
    parser.add_argument('--zarr_path', type=str, default="/home/gpatane/Dataset/zarr_file_magnetogram.zarr",
                        help='Path to the Zarr dataset file.')
    parser.add_argument('--model_save_path', type=str, default="/home/gpatane/checkpoints/mae_project",
                        help='Path to save the trained model.')
    parser.add_argument('--load_model_path', type=str, default="/home/gpatane/checkpoints/mae_project/mae_sdo_patchsize7.pth",
                        help='Path to load a pre-trained model for evaluation or resuming training.')
    
    parser.add_argument('--start_period', type=int, default=2010, help='First year (e.g., 2010 2011 2012).')
    parser.add_argument('--end_period', type=int, default=2026, help='Last year (e.g., 2013 2014).')
    parser.add_argument('--dataset', type=str, default='grid', choices=['grid', 'multi_channel_solar'],
                        help='Dataset type: grid 3x3 for SDO data or multi channel dataset for MC Solar dataset.')

    # --- Data Params ---
    parser.add_argument('--wavelengths', nargs='+', default=['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram'],
                        help='List of wavelengths to use.')
    parser.add_argument('--target_size', type=int, default=224,
                        help='Target size for individual images in the mosaic.')
    parser.add_argument('--img_size', type=int,
                        help='Model input image size (target_size * grid_size).')
    parser.add_argument('--patch_size', type=int, default=7,
                        help='Size of image patches for ViT.')
    parser.add_argument('--grid_size', type=int, default=3,
                        help='Grid size for mosaic (e.g., 3x3).')
    parser.add_argument('--in_chans', type=int, default=1,
                        help='Number of input channels.')

    # --- Model Architecture Params ---
    parser.add_argument('--embed_dim', type=int, default=768, help='ViT embedding dimension.')
    parser.add_argument('--depth', type=int, default=12, help='ViT depth.')
    parser.add_argument('--num_heads', type=int, default=12, help='ViT number of attention heads.')
    parser.add_argument('--decoder_embed_dim', type=int, default=512, help='MAE decoder embedding dimension.')
    parser.add_argument('--decoder_depth', type=int, default=8, help='MAE decoder depth.')
    parser.add_argument('--decoder_num_heads', type=int, default=16, help='MAE decoder number of attention heads.')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio in transformer blocks.')

    # --- Training Params ---
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--train_test_split_ratio', type=float, default=0.7,
                        help='Ratio of years for training.')
    parser.add_argument('--batch_size', type=int, default=2, help='Training and validation batch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers.')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Optimizer learning rate.') # Example LR
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'], help='Optimizer type.') # Example optimizer choice
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Optimizer weight decay.') # Example weight decay
    parser.add_argument('--device', type=str, default="cuda:1",
                        help='Device (e.g., "cpu", "cuda", "cuda:0").')
    # --- Action Flags ---
    parser.add_argument('--train', action='store_true', help='Run training.')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation on one image after training or loading.')
    parser.add_argument('--disable_scheduler', action='store_true', help='Disable the CosineAnnealingLR scheduler.')

    # --- WandB Params ---
    parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging.')
    parser.add_argument('--wandb_project', type=str, default='sdo-mae-training', help='WandB project name.')
    parser.add_argument('--wandb_entity', type=str, default='patane-giovanni-universit-catania', help='WandB entity (username or team).')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Custom WandB run name.')

    # --- Derived/Internal ---
    # Calculate img_size based on target_size and grid_size after parsing
    args = parser.parse_args()
    #args.img_size = args.target_size * args.grid_size
    # Add MAE repo path to sys.path AFTER parsing args
    #sys.path.insert(0, args.mae_repo_path)

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
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name or f"mae_sdo_{time.strftime('%Y%m%d_%H%M%S')}", # Default run name
                    config=vars(args)
                )
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


    #model = mae_model_for_pretraining(img_size=args.img_size, in_channel=3).to(device)
    

         # wandb.watch(model, log_freq=100) # Watch gradients, potentially slow

    # Prepare Data
    
    all_years = list(range(args.start_period, args.end_period))
    #all_years = list(range(2010, 2025))

    random.shuffle(all_years)
    
    
    train_split = int(0.7 * len(all_years))
    val_split = int(0.85 * len(all_years))

    # train_years = (all_years[:train_split])
    # val_years = (all_years[train_split:val_split])
    # test_years = (all_years[val_split:])
    train_years = list(range(2010,2021))
    val_years   = list(range(2021,2023))
    test_years  = list(range(2023,2026))
    
    print(f"Training years: {train_years}")
    print(f"Validation years: {val_years}")
    print(f"Test years: {test_years}")
    print(f"Loading data from: {args.zarr_path}")
    print(f"Wavelengths: {args.wavelengths}")
    print(f"Target size: {args.target_size}, Image size: {args.img_size}, Patch size: {args.patch_size}, Grid size: {args.grid_size}")

    if args.dataset=='multi_channel_solar':
        print("Using multi-channel solar dataset.")
        train_dataset = MC_SolarDataset(args.zarr_path, train_years, transform=None, target_size=args.img_size , wavelengths=args.wavelengths)
        validation_dataset = MC_SolarDataset(args.zarr_path, val_years, transform=None, target_size=args.img_size, wavelengths=args.wavelengths)
        test_dataset = MC_SolarDataset(args.zarr_path, test_years, transform=None, target_size=args.img_size, wavelengths=args.wavelengths)
        # Define Model
        print(f"Initializing MAE model with img_size={args.img_size}, patch_size={args.patch_size}")
        model = mae_model_for_pretraining(img_size=args.img_size, in_channel=1).to(device)
        
    elif args.dataset=="grid":
        train_dataset = SDOMosaicZarrDataset(args.zarr_path, train_years, args.wavelengths, target_size=args.target_size, grid_size=args.grid_size, n_channels=args.in_chans)
        validation_dataset = SDOMosaicZarrDataset(args.zarr_path, val_years, args.wavelengths, target_size=args.target_size, grid_size=args.grid_size, n_channels=args.in_chans)
        test_dataset = SDOMosaicZarrDataset(args.zarr_path, test_years, args.wavelengths, target_size=args.target_size, grid_size=args.grid_size, n_channels=args.in_chans)
        # Define Model
        model = mae_model_for_pretraining(img_size=args.img_size, patch_size=args.patch_size).to(device)

        print(f"Initializing MAE model with img_size={model.img_size}, patch_size={model.patch_size}")

    if wandb_run:
         num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
         print(f"Model has {num_params:,} trainable parameters.")
         wandb_run.summary["trainable_params"] = num_params


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory="cuda" in args.device, drop_last=True)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory="cuda" in args.device)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory="cuda" in args.device)
    print(f"Train dataset size: {len(train_dataset)}, Train loader batches: {len(train_loader)}")
    print(f"Validation dataset size: {len(validation_dataset)}, Val loader batches: {len(val_loader)}")


    # Training
    if args.train:
        # --- Define Optimizer ---
        if args.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            print(f"Using AdamW optimizer (LR={args.learning_rate}, WD={args.weight_decay})")
        elif args.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.sgd_momentum, weight_decay=args.weight_decay)
            print(f"Using SGD optimizer (LR={args.learning_rate}, Momentum={args.sgd_momentum}, WD={args.weight_decay})")
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

        # --- Start Training ---
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            wandb_run=wandb_run,
            model_save_path=os.path.join(args.model_save_path, args.wandb_run_name)+".pth",
            use_scheduler=not args.disable_scheduler
        )

    # Evaluation
    if args.evaluate:
        print("\n--- Running Evaluation ---")

        # Determine model path: load_model arg > saved best model > error
        model_path_to_load = args.load_model_path
        if not model_path_to_load and args.train: # If trained, use the saved best model path
            model_path_to_load = args.model_save_path
        elif not model_path_to_load and not args.train:
            print("Error: Evaluation requested (--evaluate) but no model specified via --load_model_path and training was not run (--train).")
            if wandb_run: wandb.finish(exit_code=1)
            sys.exit(1)

        if model_path_to_load:
            print(f"Loading model weights from: {model_path_to_load}")
            try:
                state_dict = torch.load(model_path_to_load, map_location=device)
                model.load_state_dict(state_dict)
                print("Model weights loaded successfully.")
            except FileNotFoundError:
                print(f"Error: Model file not found at {model_path_to_load}")
                if wandb_run: wandb.finish(exit_code=1)
                sys.exit(1)
            except Exception as e:
                print(f"Error loading model weights: {e}")
                if wandb_run: wandb.finish(exit_code=1)
                sys.exit(1)
        else:
            print("Warning: Evaluating model state directly after training without reloading from file.")

        model.eval()

        if len(val_loader) == 0:
             print("Validation loader is empty. Cannot perform evaluation.")
        else:
            print("Running 'run_one_image' on the first validation batch...")
            with torch.no_grad():
                try:
                    batch = next(iter(val_loader))
                    if isinstance(batch, (list, tuple)):
                         input_data = batch[0]
                    else:
                         input_data = batch

                    # Expecting B, C, H, W from loader
                    if input_data.dim() == 4:
                         input_data = input_data[0] # Take the first image in the batch: (C, H, W)
                    elif input_data.dim() != 3:
                         raise ValueError(f"Unexpected input data dimensions: {input_data.shape}")

                    input_data = input_data.to(device)

                    # Verify dimensions
                    if input_data.shape[1] != args.img_size or input_data.shape[2] != args.img_size:
                         print(f"Error: Model expects H, W = {args.img_size}, but got {input_data.shape[1:]}")
                    else:
                        # Assuming run_one_image takes (C, H, W) and handles device internally
                        # Pass wandb_run if run_one_image uses it for logging images
                        run_one_image(input_data, model, device, wandb_run=wandb_run)
                        print("Evaluation image processed.")

                except StopIteration:
                    print("Validation loader became empty unexpectedly.")
                except Exception as e:
                    print(f"An error occurred during evaluation image run: {e}")
                    import traceback
                    traceback.print_exc() # Print detailed traceback

    if not args.train and not args.evaluate:
        print("Neither --train nor --evaluate specified. Nothing to do.")

    # --- Finish WandB Run ---
    if wandb_run:
        print("Finishing WandB run...")
        wandb.finish()

if __name__ == "__main__":
    main()