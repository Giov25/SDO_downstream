import argparse
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
import wandb

# Import dal tuo progetto
from mae.models_mae_2 import mae_model_channel_masking_9ch_with_temporal_attn
from dataset import SDO_Dataset_channels_FAST

def get_args():
    parser = argparse.ArgumentParser(description="Training script for SDO MAE")
    
    # Path e Dataset
    parser.add_argument("--zarr_path", type=str, default="/home/gpatane/Dataset/zarr_file_magnetogram.zarr", help="Path al file zarr")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory dove salvare i modelli")
    parser.add_argument("--image_size", type=int, default=1024, help="Risoluzione immagine")
    
    # Hyperparameters Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--save_every", type=int, default=5, help="Salva checkpoint ogni X epoche")
    parser.add_argument("--seed", type=int, default=1)
    
    # Configurazione Modello
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="mae-sdo")
    parser.add_argument("--wandb_run_name", type=str, default="mae_9channels_masking_DGX")

    return parser.parse_args()

def train():
    args = get_args()
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Configurazione canali e anni
    wavelengths = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram']
    train_years = list(range(2011, 2021))
    val_years = list(range(2021, 2023))

    # Datasets
    train_dataset = SDO_Dataset_channels_FAST(args.zarr_path, train_years, wavelengths, target_size=args.image_size)
    val_dataset = SDO_Dataset_channels_FAST(args.zarr_path, val_years, wavelengths, target_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Modello
# Nel main del file creato precedentemente:
    model = mae_model_channel_masking_9ch_with_temporal_attn(
        img_size=args.image_size, 
        patch_size=args.patch_size,
        in_chans=len(wavelengths)
    ).to(args.device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # WandB Initialization
    wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_run_name}_{args.image_size}",
        config=vars(args)
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # --- TRAIN ---
        model.train()
        total_train_loss = 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar_train:
            batch = batch.to(args.device)
            loss, _, _ = model(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar_train.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)

        # --- VAL ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in pbar_val:
                batch = batch.to(args.device)
                loss, _, _ = model(batch)
                total_val_loss += loss.item()
                pbar_val.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = total_val_loss / len(val_loader)
        
        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']

        # Logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": curr_lr
        })

        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

        # Save Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))

        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"checkpoint_{epoch+1}.pth"))

    wandb.finish()

if __name__ == "__main__":
    train()