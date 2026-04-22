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
import os

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
def freeze_encoder(model):
    """
    Freezes the encoder components of MaskedAutoencoderViT.
    
    Frozen components:
    - patch_embed: patch embedding layer
    - blocks: transformer encoder blocks
    - pos_embed: positional embeddings
    - cls_token: class token
    - norm: normalization layer
    """
    # Freeze patch embedding
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    
    # Freeze encoder blocks
    for param in model.blocks.parameters():
        param.requires_grad = False
    
    # Freeze positional embeddings
    model.pos_embed.requires_grad = False
    
    # Freeze class token
    model.cls_token.requires_grad = False
    
    # Freeze normalization layer
    for param in model.norm.parameters():
        param.requires_grad = False
    
    print("✓ Encoder congelato (frozen)")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"  Parametri totali: {total_params:,}")
    print(f"  Parametri trainabili: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Parametri congelati: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    
    return model
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
            
            #outputs = model(inputs)  # [B, 2, H, W] logits
            _,pred,_ = model(inputs)
            outputs = model.unpatchify(pred)
            
            
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

def _dice_score_numpy(pred, gt, eps=1e-8):
    """Compute Dice score between two binary numpy arrays."""
    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0  # both empty -> perfect
    return 2.0 * inter / (denom + eps)

def load_checkpoint_with_channel_adaptation(model, checkpoint_path, in_chans=9, out_chans=2, device='cuda'):
    """
    Carica un checkpoint da un modello pre-addestrato e adatta i canali se necessario.
    Rileva automaticamente quanti canali ha il checkpoint.
    
    Args:
        model: il modello target
        checkpoint_path: path al checkpoint salvato
        in_chans: numero di canali input (informativo, non usato)
        out_chans: numero di canali output del nuovo modello
        device: dispositivo torch
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    
    # Determina i canali dal checkpoint in base alla forma di decoder_pred.weight
    original_weight = state_dict['decoder_pred.weight']
    original_bias = state_dict['decoder_pred.bias']
    
    patch_size = model.patch_size
    weights_per_channel = patch_size ** 2  # 256 per patch_size=16
    
    # Calcola quanti canali ha il checkpoint
    checkpoint_out_chans = original_weight.shape[0] // weights_per_channel
    
    print(f"[Load] Checkpoint ha {checkpoint_out_chans} canali, modello target ne aspetta {out_chans}")
    
    if checkpoint_out_chans != out_chans:
        if checkpoint_out_chans > out_chans:
            # Checkpoint ha PIU' canali: prendi i primi out_chans
            new_weight = original_weight[:out_chans * weights_per_channel, :]
            new_bias = original_bias[:out_chans * weights_per_channel]
            print(f"[Resize] Ridotto decoder_pred: {original_weight.shape} → {new_weight.shape}")
        else:
            # Checkpoint ha MENO canali: replica i pesi per i canali mancanti
            repeats = out_chans // checkpoint_out_chans
            remainder = out_chans % checkpoint_out_chans
            
            new_weight = original_weight.repeat(repeats, 1)
            new_bias = original_bias.repeat(repeats)
            
            if remainder > 0:
                new_weight = torch.cat([new_weight, original_weight[:remainder * weights_per_channel, :]], dim=0)
                new_bias = torch.cat([new_bias, original_bias[:remainder * weights_per_channel]])
            
            print(f"[Resize] Espanso decoder_pred: {original_weight.shape} → {new_weight.shape}")
        
        state_dict['decoder_pred.weight'] = new_weight
        state_dict['decoder_pred.bias'] = new_bias
    
    # Carica lo state_dict adattato
    model.load_state_dict(state_dict, strict=False)
    print(f"[Load] ✓ Checkpoint caricato correttamente")
    
    return model


def train_one_epoch_mod(model, loader, criterion, optimizer, device, scheduler, max_grad_norm=1.0):
    model.train()
    epoch_loss = 0
    step = 0

    for batch in tqdm(loader, desc="Training"):
        data = batch['image'].to(device)
        labels = batch['mask'].to(device)
        optimizer.zero_grad()
        _,pred,_ = model(data)
        recon = model.unpatchify(pred)
        loss = criterion(recon, labels)                                
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

def testing(model, loader, device, dice_score, dice_score_T):
    model.eval()
    
    # Liste per salvare i risultati
    dice_scores = []
    dice_scores_T = []
    iou_scores = []      # <--- NUOVO: IoU solo Foreground
    iou_scores_T = []    # <--- NUOVO: Mean IoU (Background + Foreground)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            data = batch['image'].to(device)
            labels = batch['mask'].to(device)
            _,pred,_ = model(data)
            outputs = model.unpatchify(pred)

            
            # Gestione output
            if outputs.dim() >= 4 and outputs.shape[1] > 1:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True).float()
            else:
                # Fallback nel caso output sia [B, 1, H, W] (attivazione sigmoide)
                preds = (torch.sigmoid(outputs) > 0.5).float()

            for i in range(data.shape[0]):
                # Get masks: [H, W]
                gt_mask = labels[i, 0].long()    # [H, W] (valori 0 o 1)
                pred_mask = preds[i, 0].long()   # [H, W] (valori 0 o 1)
                
                # --- CALCOLO DICE (Codice Originale) ---
                # Convert to one-hot: [1, 2, H, W]
                gt_one_hot = torch.nn.functional.one_hot(gt_mask, num_classes=2).permute(2, 0, 1).unsqueeze(0).float()
                pred_one_hot = torch.nn.functional.one_hot(pred_mask, num_classes=2).permute(2, 0, 1).unsqueeze(0).float()
                
                # dice_score: only foreground
                dice_i = dice_score(pred_one_hot, gt_one_hot)
                dice_i = torch.nan_to_num(dice_i, nan=1.0)
                dice_scores.append(dice_i.mean().item())
                
                # dice_score_T: with background
                if dice_score_T is not None:
                    dice_T = dice_score_T(pred_one_hot, gt_one_hot)
                    dice_T = torch.nan_to_num(dice_T, nan=1.0)
                    dice_scores_T.append(dice_T.mean().item())

                # --- CALCOLO IOU (NUOVO) ---
                # L'IoU si calcola come: Intersezione / Unione
                
                # 1. IoU Classe 1 (Foreground)
                inter_1 = ((pred_mask == 1) & (gt_mask == 1)).sum().item()
                union_1 = ((pred_mask == 1) | (gt_mask == 1)).sum().item()
                
                # Gestione divisione per zero: se union è 0, significa che non c'è oggetto né in GT né in Pred -> IoU = 1.0
                iou_1 = inter_1 / union_1 if union_1 > 0 else 1.0
                iou_scores.append(iou_1)

                # 2. IoU Totale (Mean IoU: media tra bg e fg)
                if dice_score_T is not None:
                    # Calcolo anche per Classe 0 (Background)
                    inter_0 = ((pred_mask == 0) & (gt_mask == 0)).sum().item()
                    union_0 = ((pred_mask == 0) | (gt_mask == 0)).sum().item()
                    iou_0 = inter_0 / union_0 if union_0 > 0 else 1.0
                    
                    # Mean IoU
                    mean_iou = (iou_0 + iou_1) / 2
                    iou_scores_T.append(mean_iou)

    # Calcolo medie finali
    metric_dice = np.mean(dice_scores) if dice_scores else 0.0
    metric_dice_T = np.mean(dice_scores_T) if dice_scores_T else 0.0
    
    metric_iou = np.mean(iou_scores) if iou_scores else 0.0
    metric_iou_T = np.mean(iou_scores_T) if iou_scores_T else 0.0

    # Ritorna 4 valori: Dice FG, Dice Tot, IoU FG, IoU Tot
    return metric_dice, metric_dice_T, metric_iou, metric_iou_T            
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

def validate_one_epoch_mod(model, loader, criterion, device, dice_score, post_pred, post_label, dice_score_T=None):
    model.eval()
    epoch_loss = 0
    step = 0

    dice_scores = []
    dice_scores_T = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            data = batch['image'].to(device)
            labels = batch['mask'].to(device)
            _,pred,_ = model(data)
            outputs = model.unpatchify(pred)
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
                log_images_every=5,  # Log predictions ogni N epochs
                start_epoch=0,  # Epoca da cui riprendere
                best_dice=0.0  # Miglior dice score precedente
                ):
    train_losses, val_losses, val_dice_scores, val_dice_scores_T = [], [], [], []
    max_validation = best_dice
    since = time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time()
        
        #train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler, max_grad_norm=max_grad_norm)
        #val_loss, val_metric, val_metric_T = validate_one_epoch(model, test_loader, criterion, device, dice_metric, post_pred, post_label, dice_score_T=dice_metric_T)
        train_loss = train_one_epoch_mod(model, train_loader, criterion, optimizer, device, scheduler, max_grad_norm=max_grad_norm)
        val_loss, val_metric, val_metric_T = validate_one_epoch_mod(model, test_loader, criterion, device, dice_metric, post_pred, post_label, dice_score_T=dice_metric_T)
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
                "val/dice_score_T": float(val_metric_T),
                "learning_rate": float(current_lr),
                "epoch_time": float(epoch_time)
            }
            # Usa epoch come step per avere le epoche sull'asse X
            wandb.log(log_dict)
            
            # Log predictions every N epochs
            if (epoch + 1) % log_images_every == 0 or epoch == 0:
                print(f"  Logging predictions to wandb...")
                log_predictions_to_wandb(model, test_loader, device, num_images=6, epoch=epoch+1, phase="validation", dice_metric=dice_metric)
        
        # Save best model
        if val_metric > max_validation:
            max_validation = val_metric
            if model_save_path:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                print(f"\n  New best model found! Saving model to {model_save_path} ...")
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_dice': val_metric,
                    'val_dice_T': val_metric_T,
                }
                # Salva wandb run id se disponibile
                if wandb_run:
                    checkpoint_data['wandb_run_id'] = wandb_run.id
                    checkpoint_data['wandb_project'] = wandb_run.project
                    checkpoint_data['wandb_entity'] = wandb_run.entity
                
                torch.save(checkpoint_data, model_save_path)
                
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

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sunpy.map import Map
def run_and_plot_predictions_all_channels(model, dataloader, device, dice_metric=None, dice_metric_T=None, 
                                          n_images=5, threshold=0.5, use_wandb=False, save_path=None):
    """
    Esegue il modello e plotta:
      - Col 1-9: I 9 canali di input
      - Col 10:  Ground Truth (overlay rosso)
      - Col 11:  Predizione (overlay blu)
    
    Salva sia l'immagine riassuntiva che le singole righe separatamente.
    """
    model.eval()
    batch = next(iter(dataloader))
    inputs = batch['image']              # Shape: [B, 9, H, W]
    labels = batch['mask']               # Shape: [B, 1, H, W]
    gt_images = batch['ic_no_limb_dark'] # Background image [B, 1, H, W]

    batch_size = inputs.shape[0]
    n_show = min(n_images, batch_size)
    n_input_channels = inputs.shape[1]   # Dovrebbe essere 9

    # Totale colonne: canali input + GT + Pred
    total_cols = n_input_channels + 2 

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

        # Figura principale riassuntiva: molto larga per far stare 11 colonne
        # Larghezza stimata: 2.5 pollici per colonna -> ~28 pollici
        fig, axes = plt.subplots(n_show, total_cols, figsize=(2.5 * total_cols, 3.5 * n_show))
        
        # Gestione caso singola immagine (axes deve essere sempre 2D [row, col])
        if n_show == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_show):
            # --- MASCHERE E BACKGROUND ---
            gt_mask = labels[i, 0].cpu().numpy().astype(np.uint8)
            pred_mask = preds[i].squeeze(0).cpu().numpy().astype(np.uint8)
            background_img = gt_images[i, 0].cpu().numpy() # Usato per overlay GT/Pred

            # --- CALCOLO DICE ---
            if dice_metric is not None and dice_metric_T is not None:
                gt_mask_tensor = labels[i, 0].long().to(device)
                pred_mask_tensor = preds[i, 0].long().to(device)
                
                gt_one_hot = torch.nn.functional.one_hot(gt_mask_tensor, num_classes=2).permute(2, 0, 1).unsqueeze(0).float()
                pred_one_hot = torch.nn.functional.one_hot(pred_mask_tensor, num_classes=2).permute(2, 0, 1).unsqueeze(0).float()
                
                dice_i = dice_metric(pred_one_hot, gt_one_hot)
                dice_i = torch.nan_to_num(dice_i, nan=1.0).mean().item()
                dice_i_T = dice_metric_T(pred_one_hot, gt_one_hot)
                dice_i_T = torch.nan_to_num(dice_i_T, nan=1.0).mean().item()
            else:
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = pred_mask.sum() + gt_mask.sum()
                dice_i = (2. * intersection / (union + 1e-8))
                dice_i_T = dice_i 
            
            dice_list.append(dice_i)
            dice_list_T.append(dice_i_T)

            # --- PLOT CANALI INPUT (Col 0 a 8) ---
            for c in range(n_input_channels):
                img_chan = inputs[i, c].cpu().numpy()
                # Normalizza ogni canale singolarmente per vederne i dettagli
                img_chan = (img_chan - img_chan.min()) / (img_chan.max() - img_chan.min() + 1e-8)
                
                axes[i, c].imshow(img_chan, cmap='gray')
                axes[i, c].set_title(f'Ch {c+1}') # Es. Ch 1, Ch 2...
                axes[i, c].axis('off')

            # --- PLOT GT (Col 9) ---
            idx_gt = n_input_channels
            #axes[i, idx_gt].imshow(background_img, cmap='gray', alpha=0.8)
            axes[i, idx_gt].imshow(gt_mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
            axes[i, idx_gt].set_title('Ground Truth')
            axes[i, idx_gt].axis('off')

            # --- PLOT PRED (Col 10) ---
            idx_pred = n_input_channels + 1
            #axes[i, idx_pred].imshow(background_img, cmap='gray', alpha=0.8)
            axes[i, idx_pred].imshow(pred_mask, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
            axes[i, idx_pred].set_title(f'Pred\nDice: {dice_i:.2f}')
            axes[i, idx_pred].axis('off')

            # --- SALVATAGGIO SINGOLA RIGA ---
            if save_path:
                # Creiamo una figura larga per la singola riga
                fig_single, ax_single = plt.subplots(1, total_cols, figsize=(6.5 * total_cols, 9.5))
                wl = ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram']
                # 1. Input Channels
                for c in range(n_input_channels):
                    img_chan = inputs[i, c].cpu().numpy()
                    img_chan = (img_chan - img_chan.min()) / (img_chan.max() - img_chan.min() + 1e-8)
                    if c<n_input_channels-1:
                        cmap ='sdoaia' + wl[c].replace('A', '')
                    else:
                        cmap = 'gray'
                    ax_single[c].imshow(img_chan, cmap=cmap)
                    ax_single[c].set_title(f'Ch {c+1}')
                    ax_single[c].axis('off')

                # 2. GT
                #ax_single[idx_gt].imshow(background_img, cmap='gray', alpha=0.8)
                ax_single[idx_gt].imshow(gt_mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
                ax_single[idx_gt].set_title('Ground Truth')
                ax_single[idx_gt].axis('off')

                # 3. Pred
                #ax_single[idx_pred].imshow(background_img, cmap='gray', alpha=0.8)
                ax_single[idx_pred].imshow(pred_mask, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
                ax_single[idx_pred].set_title(f'Dice (FG): {dice_i:.3f}')
                ax_single[idx_pred].axis('off')

                root, ext = os.path.splitext(save_path)
                single_row_path = f"{root}_sample_{i}{ext}"
                
                plt.tight_layout()
                fig_single.savefig(single_row_path, dpi=150, bbox_inches='tight')
                plt.close(fig_single)

    # --- OUTPUT FINALE ---
    plt.tight_layout()
    mean_dice = float(np.mean(dice_list)) if len(dice_list) > 0 else 0.0
    mean_dice_T = float(np.mean(dice_list_T)) if len(dice_list_T) > 0 else 0.0

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

    try:
        if use_wandb and wandb.run is not None:
            wandb.log({
                "test/predictions_all_channels": wandb.Image(fig),
                "test/dice_mean_fg": mean_dice
            })
    except Exception as e:
        print(f"Warning: wandb log failed: {e}")

    print(f"\nMean Dice: {mean_dice_T:.4f}")
    plt.show()
    plt.close(fig)

    return dice_list, dice_list_T, mean_dice, mean_dice_T

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

