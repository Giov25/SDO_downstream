import matplotlib.pyplot as plt
from IPython.display import clear_output
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as npmenom
import matplotlib.pyplot as plt
import sunpy
from sunpy.map import Map
import sys
import os
import requests
import torch.nn as nn
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch import optim 
#from utils import validate_one_epoch, run_one_image, show_image

from torch.utils.data import DataLoader


torch.manual_seed(1)
from functools import partial
from mae.models_mae_2 import mae_model_channel_masking_9ch_with_temporal_attn
from mae import models_mae_2
from  torchvision.transforms import transforms


from mae.MAE import new_mae_trial_small_patches
from dataset import SDOMosaicZarrDataset, SDO_Dataset_channels

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

from dataset import SDOMosaicZarrDataset, SDO_Dataset_channels, SDO_Dataset_channels_FAST
import random
random.seed(1)


train_years = list(range(2011,2021))
val_years   = list(range(2021,2023))
#test_years  = list(range(2023,2026))
image_size = 1024

zarr_path = "/home/gpatane/Dataset/zarr_file_magnetogram.zarr"

wavelengths = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A',  'Magnetogram']
#['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', '94A', 'Ic_noLimbDark', 'Magnetogram']


train_dataset = SDO_Dataset_channels_FAST(zarr_path, train_years, wavelengths, target_size=image_size)
validation_dataset = SDO_Dataset_channels_FAST(zarr_path, val_years, wavelengths, target_size=image_size)
#test_dataset = SDO_Dataset_channels_FAST(zarr_path, test_years, wavelengths, target_size=image_size)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=4)
#test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)


import wandb
import os
from tqdm import tqdm

# Configurazione training
num_epochs = 100
save_every = 5
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
model = mae_model_channel_masking_9ch_with_temporal_attn().to(device)
lr = 1.5e-3
scheduler_step_size = 20
#steplr = 0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim.AdamW(model.parameters(), lr=lr), step_size=scheduler_step_size, gamma=0.1)
# Optimizer e scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)


# Inizializza wandb
wandb.init(
    project="mae-sdo",
    name=f"mae_9channels_masking_{image_size}H100",
    config={
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": lr,
        "image_size": image_size,
        "patch_size": model.patch_embed.patch_size[0],
        "model": "MAE_ViT",
        "channels": len(wavelengths),
        "wavelengths": wavelengths
    }
)

print(f"Starting training for {num_epochs} epochs...")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")
print(f"Checkpoint directory: {checkpoint_dir}")


# Training loop
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # ============ Training Phase ============
    model.train()
    train_loss = 0.0
    train_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for batch in pbar:
        batch = batch.to(device)
        
        # Forward pass
        loss, pred, mask = model(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        train_loss += loss.item()
        train_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = train_loss / train_batches
    
    # ============ Validation Phase ============
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        for batch in pbar:
            batch = batch.to(device)
            
            # Forward pass
            loss, pred, mask = model(batch)
            
            # Accumulate loss
            val_loss += loss.item()
            val_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_val_loss = val_loss / val_batches
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Log to wandb
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'learning_rate': current_lr
    })
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{num_epochs}:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    print(f"  LR:         {current_lr:.6f}")
    
    # Save checkpoint
    if (epoch + 1) % save_every == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, best_model_path)
        print(f"  Best model saved! Val Loss: {best_val_loss:.4f}")
    
    print("-" * 60)

# Save final model
final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': avg_train_loss,
    'val_loss': avg_val_loss,
}, final_model_path)

print(f"\n{'='*60}")
print(f"Training completed!")
print(f"Final model saved: {final_model_path}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"{'='*60}")

wandb.finish()