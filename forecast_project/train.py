"""
Temporal forecasting downstream task for SDO solar images.

Usage
-----
# Frozen encoder (recommended first run)
python train.py --mode train --freeze_encoder \
    --mae_checkpoint /path/to/best_model.pth \
    --epochs 100 --lr 3e-4 --batch_size 2

# Fine-tuning (after frozen convergence)
python train.py --mode train \
    --mae_checkpoint /path/to/best_model.pth \
    --checkpoint_path /path/to/frozen_best.pth \
    --epochs 50 --lr 5e-5 --batch_size 1

# Resume
python train.py --mode resume \
    --checkpoint_path /path/to/last.pth \
    --epochs 150

# Test
python train.py --mode test \
    --checkpoint_path /path/to/best.pth
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

import wandb

sys.path.insert(0, os.path.dirname(__file__))
from dataset import SDO_TemporalDataset, WAVELENGTHS_9, ZARR_PATH
from models import MAE_TemporalForecaster, MAE_FullForecaster

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_YEARS  = list(range(2011, 2021))
VAL_YEAR     = list(range(2021, 2023))
TEST_YEARS   = list(range(2023, 2026))
#DELTA_T_H    = [12, 24, 36, 48, 168]
DELTA_T_H    = [12]
CKPT_DIR     = os.path.join(os.path.dirname(__file__), 'checkpoints')
DEFAULT_MAE  = '/home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p16/best_model.pth'

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train', 'resume', 'test'], default='train')
    p.add_argument('--mae_checkpoint', default=DEFAULT_MAE)
    p.add_argument('--checkpoint_path', default=None,
                   help='Path to resume from or test with')
    p.add_argument('--freeze_encoder', action='store_true', default=True)
    p.add_argument('--no_freeze_encoder', dest='freeze_encoder', action='store_false')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--target_size', type=int, default=1024)
    p.add_argument('--model_type', choices=['temporal_blocks', 'full_mae_decoder'],
                   default='temporal_blocks',
                   help="'temporal_blocks': nuovi temporal blocks + nuovo decoder (default). "
                        "'full_mae_decoder': riusa encoder + decoder pre-addestrati del MAE.")
    p.add_argument('--num_temporal_blocks', type=int, default=4)
    p.add_argument('--decoder_depth', type=int, default=6)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--wandb_project', default='SDO_forecast')
    p.add_argument('--run_name', default=None)
    p.add_argument('--wandb_id', default=None, help='Wandb run ID to resume (e.g. 1a2b3c4d)')
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--mixed_precision', action='store_true', default=True,
                   help='Enable bf16 mixed precision (disable with --no_mixed_precision)')
    p.add_argument('--no_mixed_precision', dest='mixed_precision', action='store_false')
    p.add_argument('--max_gap_hours', type=float, default=3.0,
                   help='Max allowed gap between requested and actual target timestamp')
    p.add_argument('--val_every_steps', type=int, default=2000,
                   help='Run validation every N training steps (within an epoch)')
    p.add_argument('--log_every_steps', type=int, default=50,
                   help='Print progress and log to WandB every N steps')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_datasets(args):
    common = dict(
        zarr_path=ZARR_PATH,
        wavelengths=WAVELENGTHS_9,
        target_size=args.target_size,
        delta_t_hours=DELTA_T_H,
        max_gap_hours=args.max_gap_hours,
    )
    train_ds = SDO_TemporalDataset(list_year=TRAIN_YEARS, **common)
    val_ds   = SDO_TemporalDataset(list_year=VAL_YEAR,   **common)
    test_ds  = SDO_TemporalDataset(list_year=TEST_YEARS, **common)
    return train_ds, val_ds, test_ds

# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimiser, device, epoch, args,
                    use_wandb, global_step, val_loader, best_val_loss,
                    run_name, scaler):
    """
    Train one epoch.  Returns (avg_train_loss, global_step, best_val_loss).
    Prints progress every args.log_every_steps steps (flush=True so the
    SLURM .out file updates in real time).
    Runs validation every args.val_every_steps steps.
    All wandb.log calls use step=global_step for a consistent x-axis.
    """
    model.train()
    total_loss = 0.0
    steps_in_epoch = len(loader)
    t0 = time.time()

    for local_step, batch in enumerate(loader):
        x      = batch['input'].to(device)
        target = batch['target'].to(device)
        dt_idx = batch['delta_t_idx'].to(device)

        # Skip batch if input already contains NaN (corrupt zarr sample)
        if torch.isnan(x).any() or torch.isnan(target).any():
            print(f'  [WARN] NaN in input at step {global_step}, skipping batch', flush=True)
            global_step += 1
            continue

        with autocast('cuda', dtype=torch.bfloat16, enabled=scaler.is_enabled()):
            pred = model(x, dt_idx)
            loss = model.compute_loss(pred, target)

        # Skip step if loss exploded — don't let NaN poison the weights
        if not torch.isfinite(loss):
            print(f'  [WARN] Non-finite loss={loss.item()} at step {global_step}, skipping backward', flush=True)
            optimiser.zero_grad()
            global_step += 1
            continue

        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        grad_norm = nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 0.5
        )
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item()
        global_step += 1

        # ---- periodic console print + wandb log -------------------------
        if local_step % args.log_every_steps == 0:
            elapsed = time.time() - t0
            lr_now  = optimiser.param_groups[0]['lr']
            avg     = total_loss / (local_step + 1)
            print(
                f'[E{epoch+1:03d} S{local_step:05d}/{steps_in_epoch}] '
                f'loss={loss.item():.4f}  avg={avg:.4f}  '
                f'gnorm={grad_norm:.3f}  lr={lr_now:.2e}  elapsed={elapsed:.0f}s',
                flush=True,
            )
            if use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/loss_avg': avg,
                    'train/grad_norm': grad_norm.item(),
                    'train/epoch': epoch + local_step / steps_in_epoch,
                    'lr': lr_now,
                }, step=global_step)

        # ---- periodic validation ----------------------------------------
        if local_step > 0 and local_step % args.val_every_steps == 0:
            val_loss, per_dt = validate(model, val_loader, device, DELTA_T_H)
            per_dt_str = '  '.join(f'{dt}h={v:.4f}' for dt, v in sorted(per_dt.items()))
            print(
                f'  [VAL @ step {global_step}]  val={val_loss:.4f}  {per_dt_str}',
                flush=True,
            )
            if use_wandb:
                log = {'val/loss': val_loss}
                log.update({f'val/loss_{dt}h': v for dt, v in per_dt.items()})
                wandb.log(log, step=global_step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    os.path.join(CKPT_DIR, f'{run_name}_best.pth'),
                    model, optimiser, None, epoch, best_val_loss, args,
                )
                print(f'  *** New best val loss: {best_val_loss:.4f} ***', flush=True)

            model.train()  # back to train mode after validate()

    return total_loss / steps_in_epoch, global_step, best_val_loss


@torch.no_grad()
def validate(model, loader, device, delta_t_values):
    model.eval()
    per_dt_loss = {dt: [] for dt in delta_t_values}
    dt_idx_map  = {i: dt for i, dt in enumerate(delta_t_values)}

    for batch in loader:
        x      = batch['input'].to(device)
        target = batch['target'].to(device)
        dt_idx = batch['delta_t_idx'].to(device)

        with autocast('cuda', dtype=torch.bfloat16):
            pred = model(x, dt_idx)
            loss = model.compute_loss(pred, target)

        for i, dt_i in enumerate(dt_idx.tolist()):
            dt = dt_idx_map[dt_i]
            per_dt_loss[dt].append(loss.item())

    results = {dt: (sum(v) / len(v) if v else float('nan'))
               for dt, v in per_dt_loss.items()}
    overall = sum(results.values()) / len(results)
    return overall, results

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimiser, scheduler, epoch, best_loss, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_loss,
        'args': vars(args),
    }, path)


def load_checkpoint(path, model, optimiser=None, scheduler=None, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimiser and 'optimizer_state_dict' in ckpt:
        optimiser.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt.get('epoch', 0), ckpt.get('best_val_loss', float('inf'))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(CKPT_DIR, exist_ok=True)

    run_name = args.run_name or (
        f"{'frozen' if args.freeze_encoder else 'finetuned'}"
        + (f"_tb{args.num_temporal_blocks}_dd{args.decoder_depth}"
           if args.model_type == 'temporal_blocks'
           else '_full_mae_dec')
    )

    use_wandb = not args.no_wandb
    wandb_id_file = (args.checkpoint_path.replace('.pth', '.wandb_id')
                     if args.checkpoint_path else None)
    if use_wandb:
        # Try to recover run_id from sidecar file if not passed explicitly
        if not args.wandb_id and wandb_id_file and os.path.exists(wandb_id_file):
            with open(wandb_id_file) as f:
                args.wandb_id = f.read().strip()
        resume_mode = "must" if args.wandb_id else None
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args),
                   id=args.wandb_id, resume=resume_mode)
        # Save run_id for future jobs in the chain
        if wandb_id_file and wandb.run:
            os.makedirs(os.path.dirname(wandb_id_file), exist_ok=True)
            with open(wandb_id_file, 'w') as f:
                f.write(wandb.run.id)

    # ---- Build datasets --------------------------------------------------
    print('Building datasets...')
    train_ds, val_ds, test_ds = build_datasets(args)
    print(f'  train pairs: {len(train_ds):,}  |  val: {len(val_ds):,}  |  test: {len(test_ds):,}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.num_workers > 0, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0, prefetch_factor=2)

    # ---- Build model -----------------------------------------------------
    print(f'Building model (model_type={args.model_type})...')

    common_kwargs = dict(
        mae_checkpoint=args.mae_checkpoint,
        img_size=args.target_size,
        patch_size=16,
        in_chans=9,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        delta_t_values=DELTA_T_H,
        freeze_encoder=args.freeze_encoder,
        use_gradient_checkpointing=True,
        device=str(device),
    )

    if args.model_type == 'full_mae_decoder':
        model = MAE_FullForecaster(
            **common_kwargs,
            decoder_depth=8,   # deve corrispondere al MAE pre-addestrato
        ).to(device)
    else:
        model = MAE_TemporalForecaster(
            **common_kwargs,
            decoder_depth=args.decoder_depth,
            num_temporal_blocks=args.num_temporal_blocks,
        ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f'  Trainable params: {trainable:,} / {total:,}')

    # ---- Optimiser & scheduler ------------------------------------------
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # bf16 doesn't overflow so GradScaler is disabled even with mixed precision
    scaler = GradScaler('cuda', enabled=False)

    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0

    # ---- Resume ----------------------------------------------------------
    if args.mode == 'resume' and args.checkpoint_path:
        print(f'Resuming from {args.checkpoint_path}', flush=True)
        start_epoch, best_val_loss = load_checkpoint(
            args.checkpoint_path, model, optimiser, scheduler, str(device)
        )
        global_step = start_epoch * len(train_loader)
        start_epoch += 1

    # ---- Test only -------------------------------------------------------
    if args.mode == 'test':
        ckpt_path = args.checkpoint_path
        assert ckpt_path, '--checkpoint_path required for test mode'
        load_checkpoint(ckpt_path, model, device=str(device))
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
        overall, per_dt = validate(model, test_loader, device, DELTA_T_H)
        print(f'\n=== TEST RESULTS ===')
        print(f'  Overall loss : {overall:.4f}')
        for dt, v in sorted(per_dt.items()):
            print(f'  Δt={dt:4d}h  loss={v:.4f}')
        return

    # ---- Training loop ---------------------------------------------------
    steps_per_epoch = len(train_loader)
    print(f'Steps per epoch: {steps_per_epoch:,}  |  '
          f'Val every {args.val_every_steps} steps  |  '
          f'Log every {args.log_every_steps} steps', flush=True)

    for epoch in range(start_epoch, args.epochs):
        print(f'\n=== Epoch {epoch+1}/{args.epochs} ===', flush=True)

        train_loss, global_step, best_val_loss = train_one_epoch(
            model, train_loader, optimiser, device, epoch, args,
            use_wandb, global_step, val_loader, best_val_loss, run_name, scaler,
        )
        scheduler.step()

        # End-of-epoch validation
        val_loss, per_dt = validate(model, val_loader, device, DELTA_T_H)
        lr_now = scheduler.get_last_lr()[0]

        per_dt_str = '  '.join(f'{dt}h={v:.4f}' for dt, v in sorted(per_dt.items()))
        print(
            f'[END E{epoch+1:03d}] train={train_loss:.4f}  val={val_loss:.4f}  '
            f'lr={lr_now:.2e}\n  per-Δt: {per_dt_str}',
            flush=True,
        )

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss_epoch': train_loss,
                'val/loss_epoch': val_loss,
                'lr': lr_now,
                **{f'val/loss_epoch_{dt}h': v for dt, v in per_dt.items()},
            }, step=global_step)

        # Save last checkpoint (includes scheduler state for resume)
        save_checkpoint(
            os.path.join(CKPT_DIR, f'{run_name}_last.pth'),
            model, optimiser, scheduler, epoch, best_val_loss, args,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(CKPT_DIR, f'{run_name}_best.pth'),
                model, optimiser, scheduler, epoch, best_val_loss, args,
            )
            print(f'  *** New best val loss: {best_val_loss:.4f} ***', flush=True)

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
