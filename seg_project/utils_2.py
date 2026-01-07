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
    
    
import matplotlib.pyplot as plt
import numpy as np

def plot_grid_to_image(img_batch, colormaps, title='', n_images=3, image_size=224):
    """
    Crea una griglia di immagini con diverse colormap e restituisce
    il risultato come un singolo array NumPy (immagine RGBA).
    """
    
    # 1. Creazione della figura (come prima)
    fig, axes = plt.subplots(n_images, n_images, figsize=(6, 6))
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=0.95)

    for i in range(n_images):
        for j in range(n_images):
            sub_img = img_batch[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size]
            idx = i * n_images + j
            axes[i, j].imshow(sub_img, cmap=colormaps[idx])
            axes[i, j].axis('off')
            
            for spine in axes[i, j].spines.values():
                spine.set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    # Adattamento finale dei margini
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95 if title else 1)
    
    # --- Modifiche per restituire un'immagine ---
    
    # 2. Forza il rendering della canvas
    fig.canvas.draw()
    
    # 3. Ottieni le dimensioni della canvas (in pixel)
    width, height = fig.canvas.get_width_height()
    
    # 4. Estrai i pixel come buffer RGBA e convertili in array NumPy
    #    Forma risultante: (height, width, 4)
    buffer_rgba = fig.canvas.buffer_rgba()
    image_array = np.frombuffer(buffer_rgba, dtype=np.uint8).reshape((height, width, 4))
    
    # 5. Chiudi la figura per liberare memoria
    plt.close(fig)
    
    # 6. Restituisci l'array dell'immagine
    return image_array
wavelengths = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram']
#wavelengths = ['1600A', '304A', '171A', 'Magnetogram']

colormaps=[]
for wl in wavelengths:
    if wl == 'Magnetogram':
        colormaps.append('gray')
    else:
        colormaps.append('sdoaia' + wl.replace('A', ''))
        


def visualize_batch(loader):
    batch = next(iter(loader))
    image = batch["image"]
    mask = batch["mask"]
    no_limb = batch["ic_no_limb_dark"]
    images = image[0].numpy()[0,:,:]
    img = plot_grid_to_image(images, colormaps)
    print(img.shape)
    #images=image.squeeze(0).squeeze(0).cpu().numpy()[0,:,:]
    label = mask.squeeze(0).squeeze(0).cpu().numpy()[0,:,:]
    no_limb = no_limb.squeeze(0).squeeze(0).cpu().numpy()[0,:,:]
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    #plt.imshow(images[0], cmap='gray')
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Data')

    plt.subplot(1, 3, 2)
    plt.imshow(label[0], cmap='gray')
    plt.axis('off')
    plt.title('Label')
    
    plt.subplot(1, 3, 3)
    plt.imshow(no_limb[0], cmap='gray')
    plt.axis('off')
    plt.title('No Limb Darkening')
    plt.tight_layout()
    plt.show()
    
    return label[0], images[0], no_limb[0]
    
def dice_score_wt_bg(pred, target, epsilon=1e-6):
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
    target_flat = target.reshape(target.shape[0], target.shape[1], -1)

    intersection = (pred_flat * target_flat).sum(axis=2)

    dice = (2. * intersection + epsilon) / (pred_flat.sum(axis=2) + target_flat.sum(axis=2) + epsilon)
    return dice.mean()

def dice_score_bg(pred, target, epsilon=1e-6, include_background=False):
    """
    Calcola il Dice score tra pred e target.
    Può escludere il background (primo canale) se include_background=False.
    """
    # Assicurati che pred e target siano tensori PyTorch
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)

    # Escludi il background (primo canale) se richiesto
    if not include_background:
        pred = pred[:, 1:]  # Escludi il primo canale
        target = target[:, 1:]  # Escludi il primo canale

    # Cambia la forma dei tensori
    pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
    target_flat = target.view(target.shape[0], target.shape[1], -1)

    # Calcola l'intersezione
    intersection = (pred_flat * target_flat).sum(dim=2)

    # Calcola il Dice score
    dice = (2. * intersection + epsilon) / (pred_flat.sum(dim=2) + target_flat.sum(dim=2) + epsilon)
    return dice.mean()

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
def validate_one_epoch(model, loader, criterion, device, dice_score, post_pred, post_label):
    model.eval()
    epoch_loss = 0
    step = 0

    dice_scores = []
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

                gt_mask = labels[i, 0].cpu().numpy().astype(np.uint8)
                pred_mask = preds[i].squeeze(0).cpu().numpy().astype(np.uint8)
                dice_i = _dice_score_numpy(pred_mask, gt_mask)
                dice_scores.append(dice_i)
            
            step += 1
    epoch_loss /= step
    metric = np.mean(dice_scores) if dice_scores else 0.0
    return epoch_loss, metric


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
                post_pred, 
                post_label,
                model_save_path=None,
                wandb_run=None,
                max_grad_norm=1.0,
                log_images_every=5  # Log predictions ogni N epochs
                ):
    train_losses, val_losses, val_dice_scores = [], [], []
    max_validation = 0.0
    since = time()
    
    for epoch in range(num_epochs):
        epoch_start = time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler, max_grad_norm=max_grad_norm)
        val_loss, val_metric = validate_one_epoch(model, test_loader, criterion, device, dice_metric, post_pred, post_label)
        epoch_time = time() - epoch_start
        
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

        print(f'  Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f} | Val Dice: {np.mean(val_dice_scores):.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s')
        print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_metric:.4f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s')
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

def test_and_plot(model, dataloader, device, n_images=6, threshold=0.5, use_wandb=False, save_path=None):
    """
    Esegue il modello e plotta:
      - riga 1: input del modello
      - riga 2: ground truth (overlay rosso) su gt_image
      - riga 3: predizione (overlay blu) su gt_image

    Args:
        model: modello PyTorch
        dataloader: DataLoader (deve fornire un dizionario con 'image', 'mask', e 'gt_image')
        device: torch device
        n_images: numero di esempi da mostrare
        threshold: soglia per binarizzare la predizione (binary)
        use_wandb: se True, logga le immagini su wandb
        save_path: se fornito, salva la figura in questo path
    Returns:
        dice_list: lista dei Dice score per immagine
        mean_dice: Dice score medio
    """
    model.eval()
    batch = next(iter(dataloader))
    inputs = batch['image']      # Immagine di input per il modello
    labels = batch['mask']       # Maschera di ground truth
    # --- MODIFICA 1: Estrai l'immagine di sfondo per gli overlay ---
    gt_images = batch['ic_no_limb_dark'] # Immagine di sfondo per visualizzazione

    batch_size = inputs.shape[0]
    n_show = min(n_images, batch_size)

    dice_list = []

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
            # Immagine di input (per la prima riga)
            img_input = inputs[i, 0].cpu().numpy()
            # Maschera di ground truth
            gt_mask = labels[i, 0].cpu().numpy().astype(np.uint8)
            # Predizione
            pred_mask = preds[i].squeeze(0).cpu().numpy().astype(np.uint8)

            # Immagine di sfondo per questo campione
            background_img = gt_images[i, 0].cpu().numpy()

            # Calcola il Dice score
            dice_i = _dice_score_numpy(pred_mask, gt_mask)
            dice_list.append(dice_i)

            # Column 1: original input image (quello che vede il modello)
            axes[i, 0].imshow(gt_mask, cmap='gray')
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
            axes[i, 2].set_title(f'Prediction (Dice={dice_i:.3f})')
            axes[i, 2].axis('off')

    plt.tight_layout()
    mean_dice = float(np.mean(dice_list)) if len(dice_list) > 0 else 0.0

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

    try:
        if use_wandb and 'wandb' in globals() and wandb.run is not None:
            wandb.log({"test/predictions": wandb.Image(fig)})
            wandb.log({"test/dice_per_image": dice_list, "test/dice_mean": mean_dice})
    except Exception as e:
        print(f"Warning: failed to log to wandb: {e}")

    print("Per-image Dice:")
    for idx, d in enumerate(dice_list, 1):
        print(f"  Image {idx}: {d:.4f}")
    print(f"Mean Dice (shown images): {mean_dice:.4f}")

    plt.show()
    plt.close(fig)


    return dice_list, mean_dice

