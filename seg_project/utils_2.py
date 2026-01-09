import torch
from tqdm import tqdm
from monai.transforms import (Compose,AsDiscrete)
from monai.data import decollate_batch
from time import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb


def log_predictions_to_wandb(model, test_loader, device, num_images=6, epoch=0, phase="validation", dice_metric=None):
    """
    Log predictions to wandb with image, mask, and prediction
    """
    model.eval()
    images_logged = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            if images_logged >= num_images:
                break
                
            inputs = batch_data["image"].to(device)
            labels = batch_data["mask"].to(device)
            
            outputs = model(inputs)  # [B, 2, H, W] logits
            
            # Converti logits in probabilità con softmax
            probs = torch.softmax(outputs, dim=1)  # [B, 2, H, W]
            
            # Prendi solo il canale 1 (foreground/sunspot)
            preds = probs[:, 1, :, :]  # [B, H, W]
            
            # Converti labels da [B, 1, H, W] a [B, H, W]
            labels_vis = labels.squeeze(1)  # [B, H, W]
            
            for i in range(min(inputs.shape[0], num_images - images_logged)):
                img = inputs[i].cpu().numpy()
                label = labels_vis[i].cpu().numpy()  # [H, W]
                pred = preds[i].cpu().numpy()  # [H, W] con valori [0, 1]
                
                # NORMALIZZA L'IMMAGINE NEL RANGE [0, 1]
                if img.shape[0] == 3:
                    img_vis = np.transpose(img, (1, 2, 0))  # [H, W, 3]
                else:
                    img_vis = img[0]  # [H, W]
                
                # Normalizza img_vis nel range [0, 1]
                img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)
                
                # Crea figura
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Immagine originale
                if len(img_vis.shape) == 3:
                    axes[0].imshow(img_vis)
                else:
                    axes[0].imshow(img_vis, cmap='gray')
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                # Ground truth mask
                axes[1].imshow(label, cmap='Reds', vmin=0, vmax=1)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # Predizione (probabilità del foreground)
                axes[2].imshow(pred, cmap='Blues', alpha=0.8, vmin=0, vmax=1)
                axes[2].set_title(f'Prediction (Epoch {epoch})')
                axes[2].axis('off')
                
                plt.tight_layout()
                
                # Log su wandb
                wandb.log({
                    f"{phase}_prediction_{images_logged}": wandb.Image(fig),
                }, step=epoch)
                
                plt.close(fig)
                images_logged += 1
                
                if images_logged >= num_images:
                    break
    
    model.train()


# def visualize_batch(loader):
#     batch = next(iter(loader))
#     image = batch["image"]
#     mask = batch["mask"]
#     no_limb = batch["ic_no_limb_dark"]
#     images = image[0].numpy()[0,:,:]
#     img = plot_grid_to_image(images, colormaps)
#     print(img.shape)
#     #images=image.squeeze(0).squeeze(0).cpu().numpy()[0,:,:]
#     label = mask.squeeze(0).squeeze(0).cpu().numpy()[0,:,:]
#     no_limb = no_limb.squeeze(0).squeeze(0).cpu().numpy()[0,:,:]
#     plt.figure(figsize=(10, 10))
#     plt.subplot(1, 3, 1)
#     #plt.imshow(images[0], cmap='gray')
#     plt.imshow(img, cmap='gray')
#     plt.axis('off')
#     plt.title('Data')

#     plt.subplot(1, 3, 2)
#     plt.imshow(label[0], cmap='gray')
#     plt.axis('off')
#     plt.title('Label')
    
#     plt.subplot(1, 3, 3)
#     plt.imshow(no_limb[0], cmap='gray')
#     plt.axis('off')
#     plt.title('No Limb Darkening')
#     plt.tight_layout()
#     plt.show()
    
#     return label[0], images[0], no_limb[0]
    
# def dice_score_wt_bg(pred, target, epsilon=1e-6):
#     if not isinstance(pred, np.ndarray):
#         pred = np.array(pred)
#     if not isinstance(target, np.ndarray):
#         target = np.array(target)

#     pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
#     target_flat = target.reshape(target.shape[0], target.shape[1], -1)

#     intersection = (pred_flat * target_flat).sum(axis=2)

#     dice = (2. * intersection + epsilon) / (pred_flat.sum(axis=2) + target_flat.sum(axis=2) + epsilon)
#     return dice.mean()

# def dice_score_bg(pred, target, epsilon=1e-6, include_background=False):
#     """
#     Calcola il Dice score tra pred e target.
#     Può escludere il background (primo canale) se include_background=False.
#     """
#     # Assicurati che pred e target siano tensori PyTorch
#     if not isinstance(pred, torch.Tensor):
#         pred = torch.tensor(pred)
#     if not isinstance(target, torch.Tensor):
#         target = torch.tensor(target)

#     # Escludi il background (primo canale) se richiesto
#     if not include_background:
#         pred = pred[:, 1:]  # Escludi il primo canale
#         target = target[:, 1:]  # Escludi il primo canale

#     # Cambia la forma dei tensori
#     pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
#     target_flat = target.view(target.shape[0], target.shape[1], -1)

#     # Calcola l'intersezione
#     intersection = (pred_flat * target_flat).sum(dim=2)

#     # Calcola il Dice score
#     dice = (2. * intersection + epsilon) / (pred_flat.sum(dim=2) + target_flat.sum(dim=2) + epsilon)
#     return dice.mean()

def _dice_score_numpy(pred, gt, eps=1e-8):
    """Compute Dice score between two binary numpy arrays."""
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0  # both empty -> perfect
    return 2.0 * inter / (denom + eps)

def train_one_epoch(model, loader, criterion, optimizer, device, scheduler, max_grad_norm=1.0):
    model.train()
    epoch_loss = 0
    step = 0

    for batch in tqdm(loader, desc="Training"):
        data = batch['image'].to(device)
        labels = batch['mask'].to(device)
        optimizer.zero_grad()
        outputs = model(data)

        loss = criterion(outputs, labels)                                
        loss.backward()
        
        # Gradient clipping per stabilità
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        step += 1
    
    if scheduler:
        scheduler.step()
    
    epoch_loss /= step
    return epoch_loss
'''
        outputs = model(inputs.to(device))

        if outputs.dim() >= 4 and outputs.shape[1] > 1:
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1, keepdim=True).float()
 
        for i in range(inputs.shape[0]):

            gt_mask = labels[i, 0].cpu().numpy().astype(np.uint8)
            pred_mask = preds[i].squeeze(0).cpu().numpy().astype(np.uint8)
            dice_i = _dice_score_numpy(pred_mask, gt_mask)
            dice_list.append(dice_i)
        mean_dice = float(np.mean(dice_list)) if len(dice_list) > 0 else 0.0
            
'''
def testing(model, loader, device, dice_score, dice_score_T):
    model.eval()
    epoch_loss = 0
    step = 0

    dice_scores = []
    dice_scores_T = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            data = batch['image'].to(device)
            labels = batch['mask'].to(device)
            outputs = model(data)
            if outputs.dim() >= 4 and outputs.shape[1] > 1:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True).float()
    
            for i in range(data.shape[0]):
                # Get masks: [H, W]
                gt_mask = labels[i, 0].long()  # [H, W]
                pred_mask = preds[i, 0].long()  # [H, W]
                
                # Convert to one-hot: [H, W] -> [H, W, C] -> [C, H, W] -> [1, C, H, W]
                gt_one_hot = torch.nn.functional.one_hot(gt_mask, num_classes=2).permute(2, 0, 1).unsqueeze(0).float()  # [1, 2, H, W]
                pred_one_hot = torch.nn.functional.one_hot(pred_mask, num_classes=2).permute(2, 0, 1).unsqueeze(0).float()  # [1, 2, H, W]
                
                # dice_score: without background (only foreground class)
                dice_i = dice_score(pred_one_hot, gt_one_hot)  # include_background=False
                # Replace NaN with 1.0 (perfect score when class is absent in both GT and pred)
                dice_i = torch.nan_to_num(dice_i, nan=1.0)
                dice_scores.append(dice_i.mean().item())
                
                # dice_score_T: with background (both classes)
                if dice_score_T is not None:
                    dice_T = dice_score_T(pred_one_hot, gt_one_hot)  # include_background=True
                    dice_T = torch.nan_to_num(dice_T, nan=1.0)
                    dice_scores_T.append(dice_T.mean().item())
    metric = np.mean(dice_scores) if dice_scores else 0.0
    metric_T = np.mean(dice_scores_T) if dice_scores_T else 0.0
    return metric, metric_T
                    
            
def validate_one_epoch(model, loader, criterion, device, dice_score, post_pred, post_label, dice_score_T=None):
    model.eval()
    epoch_loss = 0
    step = 0

    dice_scores = []
    dice_scores_T = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            data = batch['image'].to(device)
            labels = batch['mask'].to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            if outputs.dim() >= 4 and outputs.shape[1] > 1:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True).float()
    
            for i in range(data.shape[0]):
                # Get masks: [H, W]
                gt_mask = labels[i, 0].long()  # [H, W]
                pred_mask = preds[i, 0].long()  # [H, W]
                
                # Convert to one-hot: [H, W] -> [H, W, C] -> [C, H, W] -> [1, C, H, W]
                gt_one_hot = torch.nn.functional.one_hot(gt_mask, num_classes=2).permute(2, 0, 1).unsqueeze(0).float()  # [1, 2, H, W]
                pred_one_hot = torch.nn.functional.one_hot(pred_mask, num_classes=2).permute(2, 0, 1).unsqueeze(0).float()  # [1, 2, H, W]
                
                # dice_score: without background (only foreground class)
                dice_i = dice_score(pred_one_hot, gt_one_hot)  # include_background=False
                # Replace NaN with 1.0 (perfect score when class is absent in both GT and pred)
                dice_i = torch.nan_to_num(dice_i, nan=1.0)
                dice_scores.append(dice_i.mean().item())
                
                # dice_score_T: with background (both classes)
                if dice_score_T is not None:
                    dice_T = dice_score_T(pred_one_hot, gt_one_hot)  # include_background=True
                    dice_T = torch.nan_to_num(dice_T, nan=1.0)
                    dice_scores_T.append(dice_T.mean().item())
            
            step += 1
    epoch_loss /= step
    metric = np.mean(dice_scores) if dice_scores else 0.0
    metric_T = np.mean(dice_scores_T) if dice_scores_T else 0.0
    return epoch_loss, metric, metric_T


def predict_and_plot(model, loader, device, post_pred, post_label, dice_metric):
    metric_sum = 0.0
    metric_count = 0
    num_empty_masks = 0
    num_nan_predictions = 0
    model.eval()
    batch = next(iter(loader))
    data = batch['image'].to(device)
    labels = batch['mask'].to(device)
    with torch.no_grad():
        outputs = model(data)
        # Apply post-processing transforms
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
        
        for pred in valid_outputs:
            if torch.isnan(pred).any():
                num_nan_predictions += 1
        
        if valid_outputs:
            metric = dice_metric(y_pred=valid_outputs, y=valid_labels)
            # Debug: stampa i valori intermedi delle metriche
            print(f"Intermediate Dice Scores: {metric}")
            valid_metric_values = metric[~torch.isnan(metric)]
            if len(valid_metric_values) > 0:
                metric_sum += valid_metric_values.mean().item() * len(valid_metric_values)
                metric_count += len(valid_metric_values)
        metric = metric_sum / metric_count
        print(f"Dice score: {np.mean(metric):.4f}")

    return outputs, labels
        
import os
def train_model(model, 
                num_epochs, 
                train_loader, 
                test_loader, 
                criterion, 
                optimizer, 
                device, 
                scheduler, 
                dice_metric, 
                dice_metric_T,
                post_pred, 
                post_label,
                model_save_path=None,
                wandb_run=None,
                max_grad_norm=1.0,
                log_images_every=5  # Log predictions ogni N epochs
                
                ):
    train_losses, val_losses, val_dice_scores, val_dice_scores_T = [], [], [], []
    max_validation = 0.0
    since = time()
    
    for epoch in range(num_epochs):
        epoch_start = time()
        
        #train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler, max_grad_norm=max_grad_norm)
        val_loss, val_metric, val_metric_T = validate_one_epoch(model, test_loader, criterion, device, dice_metric, post_pred, post_label, dice_score_T=dice_metric_T)
        epoch_time = time() - epoch_start
        train_loss=0
        
        if scheduler:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to wandb: log the current epoch's losses (not cumulative mean)
        if wandb_run:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": float(train_loss),      # valore per epoca
                "val/loss": float(val_loss),
                "val/dice_score": float(val_metric),
                "val/dice_score_T": float(val_metric_T),
                "learning_rate": float(current_lr),
                "epoch_time": float(epoch_time)
            }
            # Usa epoch come step per avere le epoche sull'asse X
            wandb.log(log_dict, step=epoch)
            
            # Log predictions every N epochs
            if (epoch + 1) % log_images_every == 0 or epoch == 0:
                print(f"  Logging predictions to wandb...")
                log_predictions_to_wandb(model, test_loader, device, num_images=6, epoch=epoch+1, phase="validation", dice_metric=dice_metric)
        
        # Save best model
        if val_metric > max_validation:
            max_validation = val_metric
            if model_save_path:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_dice': val_metric,
                    'val_dice_T': val_metric_T,
                }, model_save_path)
                
                print(f"  ✓ Model saved at epoch {epoch+1} with validation dice score: {val_metric:.4f}")
                
                if wandb_run:
                    wandb.log({
                        "best_dice": float(val_metric),
                        "best_epoch": epoch + 1,
                    }, step=epoch)
                    
                    # Save model to wandb
                    wandb.save(model_save_path)
        
        print(f"Epoch {epoch+1}/{num_epochs} completed.")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice_scores.append(val_metric)
        val_dice_scores_T.append(val_metric_T)

        #print(f'  Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f} | Val Dice: {np.mean(val_dice_scores):.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s')
        print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_metric:.4f} | Val Dice (with background): {val_metric_T:.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s')
        print(f'  Best Dice so far: {max_validation:.4f}')
        print("-" * 80)
    # salva l'ultimo checkpoint
    if model_save_path:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_metric,
        }, model_save_path)
    return train_losses, val_losses, val_dice_scores

def test_and_plot(model, dataloader, device, dice_metric=None, dice_metric_T=None, n_images=6, threshold=0.5, use_wandb=False, save_path=None):
    """
    Esegue il modello e plotta:
      - colonna 1: input del modello (primo canale)
      - colonna 2: ground truth (overlay rosso) su gt_image
      - colonna 3: predizione (overlay blu) su gt_image

    Args:
        model: modello PyTorch
        dataloader: DataLoader (deve fornire un dizionario con 'image', 'mask', e 'ic_no_limb_dark')
        device: torch device
        dice_metric: DiceMetric (without background) per calcolare Dice solo sul foreground
        dice_metric_T: DiceMetric (with background) per calcolare Dice su tutte le classi
        n_images: numero di esempi da mostrare
        threshold: soglia per binarizzare la predizione (binary)
        use_wandb: se True, logga le immagini su wandb
        save_path: se fornito, salva la figura in questo path
    Returns:
        dice_list: lista dei Dice score per immagine (foreground only)
        dice_list_T: lista dei Dice score per immagine (with background)
        mean_dice: Dice score medio (foreground only)
        mean_dice_T: Dice score medio (with background)
    """
    model.eval()
    batch = next(iter(dataloader))
    inputs = batch['image']      # Immagine di input per il modello
    labels = batch['mask']       # Maschera di ground truth
    gt_images = batch['ic_no_limb_dark'] # Immagine di sfondo per visualizzazione

    batch_size = inputs.shape[0]
    n_show = min(n_images, batch_size)

    dice_list = []
    dice_list_T = []

    with torch.no_grad():
        outputs = model(inputs.to(device))

        if outputs.dim() >= 4 and outputs.shape[1] > 1:
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1, keepdim=True).float()
        else:
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

        fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
        if n_show == 1:
            axes = axes.reshape(1, 3)

        for i in range(n_show):
            # Immagine di input (primo canale)
            img_input = inputs[i, 0].cpu().numpy()
            # Normalizza per visualizzazione
            img_input = (img_input - img_input.min()) / (img_input.max() - img_input.min() + 1e-8)
            
            # Maschera di ground truth
            gt_mask = labels[i, 0].cpu().numpy().astype(np.uint8)
            # Predizione
            pred_mask = preds[i].squeeze(0).cpu().numpy().astype(np.uint8)

            # Immagine di sfondo per overlay
            background_img = gt_images[i, 0].cpu().numpy()

            # Calcola Dice scores usando MONAI metrics se disponibili
            if dice_metric is not None and dice_metric_T is not None:
                gt_mask_tensor = labels[i, 0].long().to(device)  # [H, W]
                pred_mask_tensor = preds[i, 0].long().to(device)  # [H, W]
                
                # Convert to one-hot
                gt_one_hot = torch.nn.functional.one_hot(gt_mask_tensor, num_classes=2).permute(2, 0, 1).unsqueeze(0).float()
                pred_one_hot = torch.nn.functional.one_hot(pred_mask_tensor, num_classes=2).permute(2, 0, 1).unsqueeze(0).float()
                
                # Dice without background
                dice_i = dice_metric(pred_one_hot, gt_one_hot)
                dice_i = torch.nan_to_num(dice_i, nan=1.0).mean().item()
                
                # Dice with background
                dice_i_T = dice_metric_T(pred_one_hot, gt_one_hot)
                dice_i_T = torch.nan_to_num(dice_i_T, nan=1.0).mean().item()
            else:
                # Fallback a numpy dice
                dice_i = _dice_score_numpy(pred_mask, gt_mask)
                dice_i_T = dice_i  # Stesso valore se non abbiamo le metriche
            
            dice_list.append(dice_i)
            dice_list_T.append(dice_i_T)

            # Column 1: input image (primo canale normalizzato)
            axes[i, 0].imshow(img_input, cmap='gray')
            axes[i, 0].set_title(f'Input #{i+1}')
            axes[i, 0].axis('off')

            # Column 2: GT overlay on the background image
            axes[i, 1].imshow(background_img, cmap='gray', alpha=0.8)
            axes[i, 1].imshow(gt_mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            # Column 3: Prediction overlay on the background image
            axes[i, 2].imshow(background_img, cmap='gray', alpha=0.8)
            axes[i, 2].imshow(pred_mask, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
            axes[i, 2].set_title(f'Prediction\nDice (FG): {dice_i:.3f} | Dice (All): {dice_i_T:.3f}')
            axes[i, 2].axis('off')

    plt.tight_layout()
    mean_dice = float(np.mean(dice_list)) if len(dice_list) > 0 else 0.0
    mean_dice_T = float(np.mean(dice_list_T)) if len(dice_list_T) > 0 else 0.0

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

    try:
        if use_wandb and wandb.run is not None:
            wandb.log({
                "test/predictions": wandb.Image(fig),
                "test/dice_per_image_fg": dice_list,
                "test/dice_per_image_all": dice_list_T,
                "test/dice_mean_fg": mean_dice,
                "test/dice_mean_all": mean_dice_T
            })
    except Exception as e:
        print(f"Warning: failed to log to wandb: {e}")

    print("\nPer-image Dice Scores:")
    print(f"{'Image':<8} {'Foreground':<12} {'With BG':<12}")
    print("-" * 35)
    for idx, (d_fg, d_all) in enumerate(zip(dice_list, dice_list_T), 1):
        print(f"{idx:<8} {d_fg:<12.4f} {d_all:<12.4f}")
    print("-" * 35)
    print(f"{'Mean':<8} {mean_dice:<12.4f} {mean_dice_T:<12.4f}")

    plt.show()
    plt.close(fig)

    return dice_list, dice_list_T, mean_dice, mean_dice_T

