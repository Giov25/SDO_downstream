import argparse
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from seg_model import build_seg_model
from dataset_seg import SDO_BinarySeg_Dataset, JointAugment

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_args():
    parser = argparse.ArgumentParser(description="Binary segmentation on top of MAE encoder")

    # Path e Dataset
    parser.add_argument("--zarr_path", type=str,
                        default="/home/gpatane/Dataset/zarr_file_magnetogram_1024_ORDINATO.zarr")
    parser.add_argument("--json_stats", type=str,
                        default="/home/gpatane/Dataset/statistiche_globali.json")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_seg")
    parser.add_argument("--image_size", type=int, default=1024)

    # Hyperparameters Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max norm per gradient clipping (0 = disabilitato)")
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Gradient accumulation (batch effettivo = batch_size * accum_steps)")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    # Modello
    parser.add_argument("--mae_checkpoint", type=str, required=True,
                        help="Checkpoint MAE pre-addestrato")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--decoder_embed_dim", type=int, default=512)
    parser.add_argument("--decoder_depth", type=int, default=8)
    parser.add_argument("--freeze_encoder", action="store_true", default=True)
    parser.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false")

    # Loss
    parser.add_argument("--pixel_loss", type=str, default="focal", choices=["bce", "focal"])
    parser.add_argument("--pixel_weight",       type=float, default=1.0)
    parser.add_argument("--focal_gamma_pixel",  type=float, default=2.0)
    parser.add_argument("--focal_alpha_pixel",  type=float, default=0.25)
    parser.add_argument("--pos_weight",         type=float, default=None)
    parser.add_argument("--region_loss", type=str, default="focal_tversky",
                        choices=["dice", "tversky", "focal_tversky", "dice_tversky"])
    parser.add_argument("--region_weight",      type=float, default=1.0)
    parser.add_argument("--tversky_alpha",      type=float, default=0.3)
    parser.add_argument("--tversky_beta",       type=float, default=0.7)
    parser.add_argument("--dice_tversky_weight", type=float, default=0.5,
                        help="Weight per combinare Dice (1-w) e Tversky (w) quando region_loss='dice_tversky'")
    parser.add_argument("--focal_gamma_region", type=float, default=4.0/3.0)
    parser.add_argument("--no_valid_mask", action="store_true",
                        help="Disabilita il mascheramento off-disk nella loss e nelle metriche")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_augment", action="store_true")

    # Resume
    parser.add_argument("--resume_from", type=str, default=None)

    # WandB
    parser.add_argument("--wandb_enabled", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mae-sdo-seg")
    parser.add_argument("--wandb_run_name", type=str, default="seg_binary_9ch")
    parser.add_argument("--wandb_run_id", type=str, default=None)

    return parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):  # noqa: ARG001
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_lr(epoch, args):
    if epoch < args.warmup_epochs:
        return args.lr * (epoch + 1) / args.warmup_epochs
    progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
    return args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_checkpoint(path, model, optimizer, device):
    print(f"[Resume] Carico checkpoint da: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch    = ckpt.get("epoch", 0)
        best_dice_pos  = ckpt.get("best_dice_pos", 0.0)
        wandb_run_id   = ckpt.get("wandb_run_id", None)
        print(f"[Resume] Riparto dall'epoca {start_epoch + 1}, best dice_pos = {best_dice_pos:.4f}")
    else:
        model.load_state_dict(ckpt)
        start_epoch, best_dice_pos, wandb_run_id = 0, 0.0, None
    return start_epoch, best_dice_pos, wandb_run_id


def save_checkpoint(path, model, optimizer, epoch, best_dice_pos, wandb_run_id=None):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_dice_pos": best_dice_pos,
        "wandb_run_id": wandb_run_id,
    }, path)


@torch.no_grad()
def batch_metrics(logits, targets, valid_mask=None, threshold=0.5):
    """
    Calcola IoU, Dice foreground (dice_pos) e Dice background (dice_neg).

    dice_pos: 2·TP / (2·TP + FP + FN)  → quanto bene rileviamo le macchie solari.
    dice_neg: 2·TN / (2·TN + FP + FN)  → quanto bene rileviamo il background.
    Per maschere piccole (sunspot), dice_pos è la metrica principale.
    """
    probs = torch.sigmoid(logits.float())
    preds = (probs > threshold).float()

    if valid_mask is not None:
        m = valid_mask
        preds_m, targets_m = preds * m, targets * m
        neg_t = (1 - targets) * m
        neg_p = (1 - preds)   * m
    else:
        preds_m, targets_m = preds, targets
        neg_t, neg_p = 1 - targets, 1 - preds

    eps  = 1e-7
    dims = (1, 2, 3)
    tp = (preds_m   * targets_m).sum(dim=dims)
    fp = (preds_m   * neg_t   ).sum(dim=dims)
    fn = (neg_p     * targets_m).sum(dim=dims)
    tn = (neg_p     * neg_t   ).sum(dim=dims)

    iou      = (tp / (tp + fp + fn + eps)).mean().item()
    dice_pos = (2 * tp / (2 * tp + fp + fn + eps)).mean().item()
    dice_neg = (2 * tn / (2 * tn + fp + fn + eps)).mean().item()
    return iou, dice_pos, dice_neg


def _unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def train():
    args = get_args()
    set_seeds(args.seed)

    use_wandb = args.wandb_enabled and WANDB_AVAILABLE
    if args.wandb_enabled and not WANDB_AVAILABLE:
        print("[Warning] wandb non installato, logging disabilitato.")

    wavelengths = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram']
    train_years = list(range(2011, 2021,4))
    val_years   = list(range(2021, 2023,1))

    g = torch.Generator()
    g.manual_seed(args.seed)
    use_valid_mask = not args.no_valid_mask

    train_dataset = SDO_BinarySeg_Dataset(
        zarr_path=args.zarr_path, stats_json_path=args.json_stats,
        list_year=train_years, wavelengths=wavelengths, target_size=args.image_size,
        transform=None if args.no_augment else JointAugment(),
        return_valid_mask=use_valid_mask,
    )
    val_dataset = SDO_BinarySeg_Dataset(
        zarr_path=args.zarr_path, stats_json_path=args.json_stats,
        list_year=val_years, wavelengths=wavelengths, target_size=args.image_size,
        transform=None, return_valid_mask=use_valid_mask,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn, generator=g,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"[Data] train: {len(train_dataset):,}  |  val: {len(val_dataset):,}")

    # pos_weight (solo se pixel_loss=bce)
    pos_weight = args.pos_weight
    if args.pixel_loss == 'bce' and pos_weight is None:
        print("[Data] stima pos_weight da sottocampione...")
        rng = np.random.default_rng(args.seed)
        idx_sample = rng.choice(len(train_dataset), size=min(50, len(train_dataset)), replace=False)
        pos, neg = 0.0, 0.0
        for i in idx_sample:
            s = train_dataset[int(i)]
            m = s['mask']
            v = s.get('valid_mask', None)
            if v is not None:
                pos += (m * v).sum().item(); neg += ((1 - m) * v).sum().item()
            else:
                pos += m.sum().item(); neg += m.numel() - m.sum().item()
        pos_weight = max(neg / max(pos, 1.0), 1.0)
        print(f"[Data] pos_weight stimato = {pos_weight:.2f}")
    elif args.pixel_loss == 'focal':
        pos_weight = None

    # Modello
    model = build_seg_model(
        pretrained_ckpt=args.mae_checkpoint,
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_chans=len(wavelengths),
        embed_dim=args.embed_dim,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        freeze_encoder=args.freeze_encoder,
        pixel_loss=args.pixel_loss,
        pixel_weight=args.pixel_weight,
        focal_gamma_pixel=args.focal_gamma_pixel,
        focal_alpha_pixel=args.focal_alpha_pixel,
        pos_weight=pos_weight,
        region_loss=args.region_loss,
        region_weight=args.region_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        dice_tversky_weight=args.dice_tversky_weight,
        focal_gamma_region=args.focal_gamma_region,
        use_gradient_checkpointing=True,
    ).to(args.device)

    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True,
        )
        print('GPU:', result.stdout.strip())
    except Exception:
        pass

    if args.compile:
        print("[Info] Compilo con torch.compile...")
        model = torch.compile(model)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Info] Parametri trainabili: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    scaler = GradScaler("cuda", enabled=args.mixed_precision)

    start_epoch   = 0
    best_dice_pos = 0.0
    wandb_run_id  = args.wandb_run_id
    if args.resume_from is not None:
        start_epoch, best_dice_pos, ckpt_run_id = load_checkpoint(
            args.resume_from, model, optimizer, args.device)
        if wandb_run_id is None:
            wandb_run_id = ckpt_run_id

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.wandb_run_name}_{args.image_size}",
            id=wandb_run_id, config=vars(args), resume="allow",
        )
        wandb_run_id = wandb.run.id

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    effective_batch = args.batch_size * args.accum_steps
    n_patches = (args.image_size // args.patch_size) ** 2
    print(f"[Info] Patch per immagine: {n_patches}  |  Batch effettivo: {effective_batch}")

    # ------------------------------------------------------------------ #
    # Training loop                                                       #
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, args.epochs):

        current_lr = get_lr(epoch, args)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # ---- TRAIN ----
        model.train()
        if args.freeze_encoder:
            _unwrap(model).keep_encoder_eval()

        total_loss      = 0.0
        total_grad_norm = 0.0
        total_iou       = 0.0
        total_dice_pos  = 0.0
        total_dice_neg  = 0.0
        n_metric        = 0
        n_opt_steps     = 0
        grad_norm       = 0.0
        iou_b = dp_b = dn_b = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            img   = batch['image'].to(args.device, non_blocking=True)
            mask  = batch['mask'].to(args.device,  non_blocking=True)
            vmask = batch['valid_mask'].to(args.device, non_blocking=True) \
                if 'valid_mask' in batch else None
            is_last = (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader)

            with autocast("cuda", enabled=args.mixed_precision):
                logits = model(img)
                loss   = _unwrap(model).compute_loss(logits, mask, valid_mask=vmask)
                loss   = loss / args.accum_steps

            scaler.scale(loss).backward()

            if is_last:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.grad_clip,
                    ).item()
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scaler.get_scale() >= scale_before and math.isfinite(grad_norm):
                    total_grad_norm += grad_norm
                    n_opt_steps += 1

            loss_val = loss.item() * args.accum_steps
            if math.isfinite(loss_val):
                total_loss += loss_val
                iou_b, dp_b, dn_b = batch_metrics(logits.detach(), mask, valid_mask=vmask)
                total_iou      += iou_b
                total_dice_pos += dp_b
                total_dice_neg += dn_b
                n_metric       += 1

            pbar.set_postfix({
                "loss":      f"{loss_val:.4f}",
                "dice_pos":  f"{dp_b:.3f}",
                "dice_neg":  f"{dn_b:.3f}",
                "grad":      f"{grad_norm:.2f}",
                "lr":        f"{current_lr:.2e}",
            })

        n_tr = max(1, len(train_loader))
        n_m  = max(1, n_metric)
        avg_train_loss      = total_loss      / n_tr
        avg_train_grad_norm = total_grad_norm / max(1, n_opt_steps)
        avg_train_iou       = total_iou       / n_m
        avg_train_dice_pos  = total_dice_pos  / n_m
        avg_train_dice_neg  = total_dice_neg  / n_m

        # ---- VAL ----
        model.eval()
        total_val_loss     = 0.0
        total_val_iou      = 0.0
        total_val_dice_pos = 0.0
        total_val_dice_neg = 0.0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in pbar_val:
                img   = batch['image'].to(args.device, non_blocking=True)
                mask  = batch['mask'].to(args.device,  non_blocking=True)
                vmask = batch['valid_mask'].to(args.device, non_blocking=True) \
                    if 'valid_mask' in batch else None
                with autocast("cuda", enabled=args.mixed_precision):
                    logits = model(img)
                    loss   = _unwrap(model).compute_loss(logits, mask, valid_mask=vmask)
                iou_b, dp_b, dn_b = batch_metrics(logits, mask, valid_mask=vmask)
                total_val_loss     += loss.item()
                total_val_iou      += iou_b
                total_val_dice_pos += dp_b
                total_val_dice_neg += dn_b
                pbar_val.set_postfix({
                    "loss":     f"{loss.item():.4f}",
                    "dice_pos": f"{dp_b:.3f}",
                    "dice_neg": f"{dn_b:.3f}",
                })

        n_v = len(val_loader)
        avg_val_loss     = total_val_loss     / n_v
        avg_val_iou      = total_val_iou      / n_v
        avg_val_dice_pos = total_val_dice_pos / n_v
        avg_val_dice_neg = total_val_dice_neg / n_v

        print(
            f"Epoch {epoch+1:3d} | "
            f"Train  loss {avg_train_loss:.4f}  dice_pos {avg_train_dice_pos:.3f}  dice_neg {avg_train_dice_neg:.3f} | "
            f"Val    loss {avg_val_loss:.4f}  dice_pos {avg_val_dice_pos:.3f}  dice_neg {avg_val_dice_neg:.3f} | "
            f"LR {current_lr:.2e}  GradNorm {avg_train_grad_norm:.2f}"
        )

        if use_wandb:
            wandb.log({
                "train/loss":      avg_train_loss,
                "train/iou":       avg_train_iou,
                "train/dice_pos":  avg_train_dice_pos,
                "train/dice_neg":  avg_train_dice_neg,
                "train/grad_norm": avg_train_grad_norm,
                "train/lr":        current_lr,
                "val/loss":        avg_val_loss,
                "val/iou":         avg_val_iou,
                "val/dice_pos":    avg_val_dice_pos,
                "val/dice_neg":    avg_val_dice_neg,
            }, step=epoch + 1)

        # Salva best model su dice_pos (foreground): più rilevante per macchie piccole
        if avg_val_dice_pos > best_dice_pos:
            best_dice_pos = avg_val_dice_pos
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "best_model.pth"),
                model, optimizer, epoch, best_dice_pos, wandb_run_id,
            )
            print(
                f"  → Nuovo best model: dice_pos={best_dice_pos:.4f} "
                f"dice_neg={avg_val_dice_neg:.4f} loss={avg_val_loss:.4f}"
            )

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pth"),
                model, optimizer, epoch, best_dice_pos, wandb_run_id,
            )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
