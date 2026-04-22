from IPython.display import clear_output
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as npmenom
import sunpy
from sunpy.map import Map
import sys
import os
import requests
import torch.nn as nn
import torch
import numpy as np
import random
from PIL import Image
from torch import optim 

from torch.utils.data import DataLoader


import argparse



from dataset import SDO_Dataset_channels_FAST
import random
from mae.models_mae_2 import mae_model_fixed_channel_masking
from utils import validate_one_epoch



def get_args():
    parser = argparse.ArgumentParser(description="Training script for SDO MAE")

    # Path e Dataset
    parser.add_argument("--zarr_path", type=str, default="/home/gpatane/Dataset/zarr_file_magnetogram.zarr")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--image_size", type=int, default=1024)

    # Hyperparameters Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)

    # Configurazione Modello
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--channels_to_mask", type=int, default=5)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Abilita determinismo sulla GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    all_years = list(range(2011, 2026))
    random.shuffle(all_years)

    train_years = list(range(2011, 2021))
    val_years   = list(range(2021, 2023))
    test_years  = list(range(2023, 2025))


    wavelengths = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram']
    #['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', '94A', 'Ic_noLimbDark', 'Magnetogram']


    train_dataset = SDO_Dataset_channels_FAST(args.zarr_path, train_years, wavelengths, target_size=args.image_size)
    validation_dataset = SDO_Dataset_channels_FAST(args.zarr_path, val_years, wavelengths, target_size=args.image_size)
    test_dataset = SDO_Dataset_channels_FAST(args.zarr_path, test_years, wavelengths, target_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)




    model = mae_model_fixed_channel_masking(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_chans=9,
        n_channels_to_mask=args.channels_to_mask
    ).to(args.device)
    if args.patch_size == 16:
        checkpoint_path = '/home/gpatane/SDO_downstream/mae_project/checkpoints/checkpoint_epoch55.pth'
    elif args.patch_size == 8:
        checkpoint_path = '/home/gpatane/SDO_downstream/mae_project/checkpoints/best_8_patch/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    avg_epoch_loss, avg_epoch_ssim = validate_one_epoch(model, test_loader, args.device)
    print(f"results for {args.channels_to_mask}, seed: {args.seed}")
    print(f' average loss: {avg_epoch_loss:.4f}, average ssim: {avg_epoch_ssim:.4f} ')
    
if __name__ == "__main__":
    main()
