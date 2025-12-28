import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

import time
from dataset import SDOMosaicZarrDataset

def visualize_batch(loader):
    batch = next(iter(loader))
    image = batch["image"]
    mask = batch["mask"]
    images=image.squeeze(0).squeeze(0).cpu().numpy()[0,:,:]
    label = mask.squeeze(0).squeeze(0).cpu().numpy()[0,:,:]
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(images[0], cmap='gray')
    plt.axis('off')
    plt.title('Data')

    plt.subplot(1, 2, 2)
    plt.imshow(label[0], cmap='gray')
    plt.axis('off')
    plt.title('Label')

def calculate_ssim_pytorch(img1_tensor: torch.Tensor, img2_tensor: torch.Tensor, data_range: float = 1.0, multichannel: bool = True, channel_axis: int = -1) -> float:

    img1_np = img1_tensor.detach().cpu().numpy()
    img2_np = img2_tensor.detach().cpu().numpy()
    try:
        ssim_value = ssim(img1_np, img2_np, data_range=data_range, channel_axis=channel_axis, win_size=7)
    except TypeError:
        ssim_value = ssim(img1_np, img2_np, data_range=data_range, multichannel=multichannel, win_size=7)
    return ssim_value

def train_one_epoch(model, dataloader, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    # Wrap dataloader with tqdm for progress bar
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Assuming batch structure is appropriate (e.g., just images or images first)
        if isinstance(batch, (list, tuple)):
             inputs = batch.to(device) # Adjust index if data is not first element
        else:
             inputs = batch.to(device)

        optimizer.zero_grad()
        # Ensure model forward returns loss first if mask_ratio is provided
        loss, _, _ = model(inputs)
        # Handle potential loss tensors (e.g., from DataParallel)
        if isinstance(loss, torch.Tensor) and loss.numel() > 1:
             loss = loss.mean() # Aggregate loss if needed
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
        # if num_batches >= 2158:
        #     tqdm.write(f"Processed {num_batches} batches, current loss: {loss.item():.4f}")

    return epoch_loss / num_batches if num_batches > 0 else 0.0


def validate_one_epoch(model, dataloader, device):

    model.eval()
    epoch_loss = 0.0
    epoch_ssim = 0.0
    num_batches = 0
    total_images = 0 

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)
            if inputs.ndim == 3: 
                inputs = inputs.unsqueeze(0)
            batch_size = inputs.shape[0]
            if batch_size == 0: continue 
            loss, pred, mask = model(inputs)

            if isinstance(loss, torch.Tensor) and loss.numel() > 1:
                loss = loss.mean()
            epoch_loss += loss.item() * batch_size # Pondera per batch size

            y = model.unpatchify(pred) 
            y = torch.einsum('nchw->nhwc', y)
            original_imgs = torch.einsum('nchw->nhwc', inputs)
            batch_ssim_sum = 0.0
            for i in range(batch_size):
                data_range_val = 2.0
                current_ssim = calculate_ssim_pytorch(
                    y[i],
                    original_imgs[i],
                    data_range=data_range_val,
                    channel_axis=-1
                )
                batch_ssim_sum += current_ssim

            epoch_ssim += batch_ssim_sum
            num_batches += 1
            total_images += batch_size

    avg_epoch_loss = epoch_loss / total_images if total_images > 0 else 0.0
    avg_epoch_ssim = epoch_ssim / total_images if total_images > 0 else 0.0

    return avg_epoch_loss, avg_epoch_ssim


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=100, wandb_run=None, model_save_path=None, use_scheduler=False):
    model = model.to(device)
    # Scheduler: CosineAnnealing for gradual learning rate reduction
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None
    best_val_loss = float('inf')
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, avg_epoch_ssim = validate_one_epoch(model, val_loader, device)
        if scheduler:
            scheduler.step()
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} Summary | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        print(f"Epoch {epoch+1} Summary | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.4f}") #| Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        if wandb_run:
            wandb_run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "avg_epoch_ssim": avg_epoch_ssim,
                "learning_rate": current_lr,
                "epoch_time_seconds": epoch_time
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if model_save_path:
                print(f"Validation loss improved to {best_val_loss:.4f}. Saving model to {model_save_path}")
                try:
                    torch.save(model.state_dict(), model_save_path)
                    # Optional: Log best loss to wandb summary
                    if wandb_run:
                        wandb_run.summary["best_val_loss"] = best_val_loss
                        wandb_run.summary["best_epoch"] = epoch + 1
                except Exception as e:
                    print(f"Error saving model: {e}")
            else:
                 print(f"Validation loss improved to {best_val_loss:.4f} (model saving disabled).")

    print(f"\nTraining finished after {num_epochs} epochs.")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    
def show_image(image, colormaps, title=''):
    _, axes = plt.subplots(3, 3, figsize=(12, 12))
    print(f"Image shape: {image.shape}")
    for i in range(3):
        for j in range(3):
            sub_img = image[i*224:(i+1)*224, j*224:(j+1)*224]
            idx = i * 3 + j
            
            axes[i, j].imshow(sub_img, cmap=colormaps[idx])
            axes[i, j].axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    return

def run_one_image(img, model, dev):
    colormaps = ['sdoaia1700', 'sdoaia1600', 'sdoaia335', 'sdoaia304', 'sdoaia211', 'sdoaia193', 'sdoaia171', 'sdoaia131', 'sdoaia94']
    if dev is not None:
         x = torch.tensor(img).to(dev)
         model = model.to(dev)
    else:
         x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)


    # run MAE
    loss, y, mask = model(x.float())
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    only_reconstruct = y * mask - x * (1 - mask) 
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [20, 20]
    x, im_masked, y, im_paste, only_reconstruct= x.cpu(), im_masked.cpu(), y.cpu(), im_paste.cpu(), only_reconstruct.cpu() 
    only_masked = x - im_masked
    
    img_x =x[0].detach().cpu().numpy()
    img_im_masked = im_masked[0].detach().cpu().numpy()
    img_im_paste = im_paste[0].detach().cpu().numpy()
    plt.subplot(1, 3, 1)
    show_image(img_x, colormaps, "original")

    plt.subplot(1, 3, 2)
    show_image(img_im_masked, colormaps, "masked")

    plt.subplot(1, 3, 3)
    show_image(img_im_paste, colormaps, "reconstruction + visible")

    # plt.subplot(1, 5, 4)
    # show_image(only_masked[0], "original - masked",)
    
    # plt.subplot(1, 5, 5)
    # show_image(only_reconstruct[0], "recontruction - original")
    plt.show()
    
    return only_reconstruct[0], only_masked[0]
