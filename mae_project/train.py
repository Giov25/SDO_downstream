import argparse
import os
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Import dal tuo progetto
from mae.models_mae_2 import mae_model_channel_masking_9ch_with_temporal_attn
from dataset import SDO_Dataset_channels_FAST

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_args():
    parser = argparse.ArgumentParser(description="Training script for SDO MAE")

    # Path e Dataset
    parser.add_argument("--zarr_path", type=str, default="/home/gpatane/Dataset/zarr_file_magnetogram.zarr")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--image_size", type=int, default=1024)

    # Hyperparameters Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6, help="LR minima per cosine scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Epoche di warmup lineare")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max norm per gradient clipping (0 = disabilitato)")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    # Configurazione Modello
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true", help="Abilita mixed precision (fp16)")
    parser.add_argument("--compile", action="store_true", help="Usa torch.compile per velocizzare il training")

    # Resume
    parser.add_argument("--resume_from", type=str, default=None, help="Path a un checkpoint .pth per riprendere il training")

    # WandB
    parser.add_argument("--wandb_enabled", action="store_true", help="Abilita il logging su Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="mae-sdo")
    parser.add_argument("--wandb_run_name", type=str, default="mae_9channels_masking_DGX")

    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """Garantisce riproducibilità nei DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def get_lr(epoch, args):
    """
    Cosine decay con warmup lineare.
    - Epoche [0, warmup_epochs): LR cresce linearmente da 0 a args.lr
    - Epoche [warmup_epochs, epochs): cosine decay da args.lr a args.min_lr
    """
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / args.warmup_epochs
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + (args.lr - args.min_lr) * cosine_factor


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Carica un checkpoint completo (modello + optimizer + stato training)."""
    print(f"[Resume] Carico checkpoint da: {path}")
    ckpt = torch.load(path, map_location=device)

    # Supporta sia checkpoint 'full' che semplici state_dict
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[Resume] Riparto dall'epoca {start_epoch + 1}, best val loss = {best_val_loss:.4f}")
    else:
        # Checkpoint legacy (solo state_dict)
        model.load_state_dict(ckpt)
        start_epoch = 0
        best_val_loss = float("inf")
        print("[Resume] Checkpoint legacy (solo pesi), riparto dall'epoca 1")

    return start_epoch, best_val_loss


def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def train():
    args = get_args()
    set_seeds(args.seed)

    # Validazione wandb
    use_wandb = args.wandb_enabled
    if use_wandb and not WANDB_AVAILABLE:
        print("[Warning] wandb non installato, logging disabilitato.")
        use_wandb = False

    # Canali e anni
    wavelengths = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram']
    train_years = list(range(2011, 2021))
    val_years   = list(range(2021, 2023))

    # Datasets e DataLoader
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset = SDO_Dataset_channels_FAST(args.zarr_path, train_years, wavelengths, target_size=args.image_size)
    val_dataset   = SDO_Dataset_channels_FAST(args.zarr_path, val_years,   wavelengths, target_size=args.image_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn, generator=g,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Modello
    model = mae_model_channel_masking_9ch_with_temporal_attn(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_chans=len(wavelengths),
    ).to(args.device)
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], capture_output=True, text=True)
    print('Tutte le GPU fisiche:')
    print(result.stdout)
    print(torch.cuda.get_device_properties(0).uuid)
    if args.compile:
        print("[Info] Compilo il modello con torch.compile...")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] Parametri trainabili: {n_params / 1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.mixed_precision)

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume_from is not None:
        start_epoch, best_val_loss = load_checkpoint(
            args.resume_from, model, optimizer, None, args.device
        )

    # WandB init
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_run_name}_{args.image_size}",
            config=vars(args),
            resume="allow",
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, args.epochs):

        # Aggiorna LR manualmente (cosine + warmup)
        current_lr = get_lr(epoch, args)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # ---- TRAIN ----
        model.train()
        total_train_loss = 0.0
        total_grad_norm  = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for step, batch in enumerate(pbar):
            batch = batch.to(args.device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=args.mixed_precision):
                loss, _, _ = model(batch)

            scaler.scale(loss).backward()

            # Gradient clipping
            grad_norm = 0.0
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()

            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            total_grad_norm  += grad_norm

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "grad": f"{grad_norm:.2f}", "lr": f"{current_lr:.2e}"})

        avg_train_loss = total_train_loss / len(train_loader)
        avg_grad_norm  = total_grad_norm  / len(train_loader)

        # ---- VAL ----
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in pbar_val:
                batch = batch.to(args.device, non_blocking=True)
                with autocast(enabled=args.mixed_precision):
                    loss, _, _ = model(batch)
                total_val_loss += loss.item()
                pbar_val.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1:3d} | Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | "
              f"LR {current_lr:.2e} | GradNorm {avg_grad_norm:.2f}")

        # Log su wandb (asse x = epoca)
        if use_wandb:
            wandb.log({
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "train/grad_norm": avg_grad_norm,
                "train/lr": current_lr,
            }, step=epoch + 1)

        # ---- Salvataggio ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "best_model.pth"),
                model, optimizer, epoch, best_val_loss,
            )
            print(f"  → Nuovo best model salvato (val loss: {best_val_loss:.4f})")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pth"),
                model, optimizer, epoch, best_val_loss,
            )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()