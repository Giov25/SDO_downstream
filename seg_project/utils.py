import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import wandb
from tqdm import tqdm
import numpy as np
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from monai.losses import DiceLoss, DiceCELoss

def get_class_weights(dataloader, minimum =10):
    total_pixels = 0
    positive_pixels = 0
    
    for batch in dataloader:
        masks = batch["mask"]
        total_pixels += masks.numel()
        positive_pixels += masks.sum().item()
    
    pos_weight = (total_pixels - positive_pixels) / positive_pixels if positive_pixels > 0 else 1.0
    return min(pos_weight, minimum)



def train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """
    Training per una singola epoca
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    #criterion = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        data = batch['image'].to(device, non_blocking=True)
        labels = batch['mask'].to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/num_batches:.4f}'
        })
    if scheduler is not None:
        scheduler.step()

    
    return total_loss / num_batches

def validate_one_epoch(model, val_loader, criterion, device, dice_metric=None, 
                      post_pred=None, post_label=None):
    """
    Validazione per una singola epoca
    """
    model.eval()
    total_loss_T = 0.0
    total_loss_F = 0.0
    num_batches = 0
    dice_scores = []
    
    progress_bar = tqdm(val_loader, desc="Validation")
    criterion_T = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    criterion_F = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            data = batch['image'].to(device, non_blocking=True)
            labels = batch['mask'].to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(data)
            
            # Calculate loss
            loss_T = criterion_T(outputs, labels)
            loss_F = criterion_F(outputs, labels)
            total_loss_T += loss_T.item()
            total_loss_F += loss_F.item()
            num_batches += 1
            
            # Calculate Dice metric if provided
            if dice_metric is not None:
                try:
                    # Apply post-processing if provided
                    if post_pred is not None:
                        pred_processed = post_pred(outputs)
                    else:
                        # Default: sigmoid + threshold at 0.5
                        probs = torch.sigmoid(outputs)
                        pred_processed = (probs > 0.5).float()
                    
                    if post_label is not None:
                        targets_processed = post_label(labels)
                    else:
                        targets_processed = labels.float()
                    
                    # Reset and calculate Dice metric
                    dice_metric.reset()
                    dice_metric(y_pred=pred_processed, y=targets_processed)
                    batch_dice = dice_metric.aggregate()
                    
                    # Extract dice score from tuple if needed
                    if isinstance(batch_dice, tuple):
                        batch_dice = batch_dice[0]
                    
                    dice_score = batch_dice.item()
                    
                    # Filter out NaN values
                    if not np.isnan(dice_score):
                        dice_scores.append(dice_score)
                        
                except Exception as e:
                    print(f"Warning: Error computing dice for batch {batch_idx}: {e}")
                    continue
            
            # Update progress bar
            avg_loss_T = total_loss_T / num_batches
            avg_loss_F = total_loss_F / num_batches
            current_dice = np.mean(dice_scores) if dice_scores else 0.0
            progress_bar.set_postfix({
                'Val Loss True': f'{avg_loss_T:.4f}',
                'Val Loss False': f'{avg_loss_F:.4f}',
                'Dice': f'{current_dice:.4f}'
            })
    
    avg_loss_T = total_loss_T / num_batches
    avg_loss_F = total_loss_F / num_batches
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    
    return avg_loss_T, avg_loss_F, avg_dice

def train_model(model, 
                num_epochs, 
                train_loader, 
                test_loader, 
                criterion,
                optimizer, 
                device, 
                scheduler=None,
                dice_metric=None, 
                post_pred=None, 
                post_label=None,
                model_save_path=None,
                wandb_run=None,
                save_best_only=True,
                patience=None):

    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize tracking variables
    train_losses = []
    val_losses_T = []
    val_losses_F = []
    val_dice_scores = []
    
    best_dice = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time()
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler if hasattr(scheduler, 'step_update') else None
        )
        
        # Validation phase
        val_loss_T, val_loss_F, val_dice = validate_one_epoch(
            model=model,
            val_loader=test_loader,
            criterion=criterion,
            device=device,
            dice_metric=dice_metric,
            post_pred=post_pred,
            post_label=post_label
        )
        
        
        # Scheduler step (se è per epoca)
        if scheduler is not None and not hasattr(scheduler, 'step_update'):
            scheduler.step()
        
        # Update tracking
        train_losses.append(train_loss)
        val_losses_T.append(val_loss_T)
        val_losses_F.append(val_loss_F)
        val_dice_scores.append(val_dice)
        
        # Calculate epoch time
        epoch_time = time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss background True:   {val_loss_T:.4f}")
        print(f"  Val Loss background False:   {val_loss_F:.4f}")
        print(f"  Val Dice:   {val_dice:.4f}")
        print(f"  Time:       {epoch_time:.2f}s")
        
        # Check for best model
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            print(f"  *** New best Dice score: {best_dice:.4f} ***")
        else:
            epochs_without_improvement += 1
        
        #Save model
        if model_save_path is not None:
            if save_best_only:
                if is_best:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'train_loss': train_loss,
                        'val_loss True': val_loss_T,
                        'val_loss False': val_loss_F,
                        'val_dice': val_dice,
                        'best_dice': best_dice
                    }, model_save_path)
                    print(f"  Model saved: {model_save_path}")
            else:
                # Save every epoch
                checkpoint_path = model_save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_loss': train_loss,
                    'val_loss True': val_loss_T,
                    'val_loss False': val_loss_F,
                    'val_dice': val_dice
                }, checkpoint_path)
        
        # Wandb logging
        if wandb_run is not None:
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                'val_loss True': val_loss_T,
                'val_loss False': val_loss_F,
                "val_dice_score": val_dice,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time
            }
            
            if is_best:

                wandb_run.summary["best_dice"] = best_dice
                wandb_run.summary["best_epoch"] = best_epoch

            wandb_run.log(log_dict)
        
        # Early stopping
        if patience is not None and epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            print(f"Best Dice score: {best_dice:.4f} at epoch {best_epoch}")
            break
    
    # Training summary
    total_time = time() - start_time
    print(f"\nTraining completed!")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Best Dice score: {best_dice:.4f} at epoch {best_epoch}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss True: {val_losses_T[-1]:.4f}")
    print(f"Final val loss False: {val_losses_F[-1]:.4f}")
    print(f"Final val dice: {val_dice_scores[-1]:.4f}")
    
    
    #save checkpoint at the end of training
    if model_save_path is not None:
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_losses[-1],
            'val_loss True': val_losses_T[-1],
            'val_loss False': val_losses_F[-1],
            'val_dice': val_dice_scores[-1],
            'best_dice': best_dice
        }, model_save_path)
        print(f"Final model saved: {model_save_path}")
    
    return train_losses, val_losses_T, val_losses_F, val_dice_scores

# Utility function for loading checkpoint
def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    Carica un checkpoint salvato
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_dice = checkpoint.get('best_dice', 0.0)
    
    print(f"Checkpoint loaded: epoch {epoch}, best dice {best_dice:.4f}")
    
    return epoch, best_dice

from monai.data import decollate_batch
def validate(model, loader, criterion, device, post_pred, post_label, dice_metric):
    model.eval()
    epoch_loss = 0
    step = 0
    metric_sum = 0.0
    metric_count = 0
    num_empty_masks = 0
    num_nan_predictions = 0
    
    with torch.no_grad():
        for batch in loader:
            data = batch['image'].to(device)
            labels = batch['mask'].to(device)

            outputs = model(data)
            
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            
            outputs = [post_pred(i) for i in decollate_batch(outputs)]
            labels = [post_label(i) for i in decollate_batch(labels)]
            
            # Filtra le maschere nere
            valid_outputs = []
            valid_labels = []
            for pred, label in zip(outputs, labels):
                if label.sum() != 0:
                    valid_outputs.append(pred)
                    valid_labels.append(label)
                else:
                    num_empty_masks += 1
            
            # Verifica se le predizioni contengono NaN
            for pred in valid_outputs:
                if torch.isnan(pred).any():
                    num_nan_predictions += 1
            
            # Calcola la metrica solo per le maschere non vuote
            if valid_outputs:
                metric = dice_metric(y_pred=valid_outputs, y=valid_labels)
                # Debug: stampa i valori intermedi delle metriche
                print(f"Intermediate Dice Scores: {metric}")
                valid_metric_values = metric[~torch.isnan(metric)]
                if len(valid_metric_values) > 0:
                    metric_sum += valid_metric_values.mean().item() * len(valid_metric_values)
                    metric_count += len(valid_metric_values)
            step += 1
    
    epoch_loss /= step
    metric = metric_sum / metric_count if metric_count > 0 else float('nan')
    return epoch_loss, metric

from time import time
import torch
import wandb

# Assumendo che train_one_epoch sia definita prima...

def validate_one_epoch_prova(model, val_loader, criterion, device, dice_metric, post_pred, post_label):
    model.eval()
    total_val_loss = 0
    
    # --- MODIFICA: Inizializza contatori per il calcolo custom del Dice ---
    true_negative_count = 0
    total_samples = 0
    
    dice_metric.reset()

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            total_samples += images.shape[0]

            # Forward pass
            logits = model(images)
            loss = criterion(logits, masks)
            total_val_loss += loss.item()

            # Applica post-processing per ottenere maschere binarie
            pred_binary = post_pred(logits)
            
            # --- MODIFICA: Logica per separare i casi "entrambi neri" ---
            
            # Somma i pixel per ogni maschera nel batch
            gt_sum_per_sample = masks.view(masks.shape[0], -1).sum(dim=1)
            pred_sum_per_sample = pred_binary.view(pred_binary.shape[0], -1).sum(dim=1)
            
            # Trova gli indici dove sia GT che predizione sono nere (somma dei pixel = 0)
            is_true_negative = (gt_sum_per_sample == 0) & (pred_sum_per_sample == 0)
            
            # Conta quanti ce ne sono in questo batch
            true_negative_count += is_true_negative.sum().item()
            
            # Seleziona solo i casi "normali" da passare alla metrica di MONAI
            is_normal_case = ~is_true_negative
            
            if is_normal_case.sum() > 0:
                normal_preds = pred_binary[is_normal_case]
                normal_gts = masks[is_normal_case]
                
                # Applica il post-labeling solo ai GT normali (se necessario)
                normal_gts_processed = post_label(normal_gts)
                
                dice_metric(y_pred=normal_preds, y=normal_gts_processed)

    # --- MODIFICA: Calcolo finale del Dice Score ---
    
    # Calcola il numero di campioni processati dalla metrica MONAI
    num_metric_samples = total_samples - true_negative_count
    
    if num_metric_samples > 0:
        # Ottieni il Dice score solo per i casi normali
        dice_from_metric = dice_metric.aggregate().item()
        # Calcola la media pesata
        final_dice = (dice_from_metric * num_metric_samples + 1.0 * true_negative_count) / total_samples
    elif total_samples > 0:
        # Caso in cui TUTTI i campioni erano "entrambi neri"
        final_dice = 1.0
    else:
        # Il validation set è vuoto
        final_dice = 0.0
        
    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
    

    
    return avg_val_loss, final_dice

import matplotlib.pyplot as plt
import os
def wb_mask(self, img, mask, gt):
    return wandb.Image(img, masks={"ground_truth" : {"mask_data" : gt, 
                                                    "class_labels" : {0: "background", 1: "mask"}},
                                    "predictions": { "mask_data": mask+2, 
                                                    "class_labels": {2: "background", 3: "mask"}}
    })
        
@torch.no_grad()
def log_slices(self, slices: torch.Tensor, masks: torch.Tensor, ground_truth: torch.tensor, step: int, name: str, phase: str):

    assert len(slices.shape)==5, 'Missing dimension.'
    assert slices.shape[1] == masks.shape[1] == ground_truth.shape[1]
    assert slices.shape[-1] == masks.shape[-1] == ground_truth.shape[-1]
    num_samples = slices.shape[0]
    num_slices = slices.shape[-1]
    s = slices.squeeze(1)
    msk = masks.squeeze(1)
    gt_in = ground_truth.squeeze(1)
    for b in range(num_samples):
        wandb_mask_logs = []
        for idx in range(num_slices):
            img = s[b, :, :, idx]
            m = msk[b, :, : , idx]
            gt = gt_in[b, :, :, idx]

            wandb_mask_logs.append(self.wb_mask(img.cpu().numpy(), m.cpu().numpy(), gt.cpu().numpy()))
        wandb.log({f"Segmentation/{phase}/{name}": wandb_mask_logs, 'step': step})

@torch.no_grad()

def save_images(self, images: torch.Tensor,masks: torch.Tensor, name: str, step: int):
    '''
    Save images to disk
    '''
    num_images = images.shape[-1]
    assert num_images == masks.shape[-1]
    images = images.cpu().detach().numpy()
    masks = masks.cpu().detach().numpy()
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 2, rows * 2))

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < num_images:
            ax.imshow(images[...,i], cmap='gray')
            ax.imshow(masks[...,i], cmap='hot', alpha=0.3)
        ax.axis('off')
    for i in range(num_images, rows * cols):
        fig.delaxes(axes.flatten()[i])
    plt.tight_layout()

    plt.savefig(os.path.join(self.vis_path, name+'.png'))

    plt.close()