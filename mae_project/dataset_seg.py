from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from astropy.io import fits
import zarr
import dask.array as da

import zarr
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from scipy.ndimage import zoom
import torch
import dask.array as da
import random
from sklearn.model_selection import train_test_split


class SDO_9Channel_Dataset(Dataset):
    def __init__(self, zarr_path, list_year, wavelengths, target_size=512, transform=None, num_classes=1):
        """
        Args:
            zarr_path (str): Path al file Zarr.
            list_year (list): Anni da includere (es. ['2014', '2015']).
            wavelengths (list): Lista dei 9 canali AIA.
            target_size (int): Dimensione finale dell'immagine (H, W).
            transform (callable, optional): Trasformazioni da applicare al dizionario batch.
            num_classes (int): 1 per regressione/binary mask, 2 per CrossEntropy.
        """
        self.z = zarr.open(zarr_path, mode='r')
        self.list_year = [str(y) for y in list_year]
        self.wavelengths = wavelengths # Assicurati siano i 9 canali AIA
        self.target_size = target_size
        self.transform = transform
        self.num_classes = num_classes
        
        # Pre-calcolo degli indici per accesso immediato O(1)
        self.indices = []
        for year in self.list_year:
            # Calcoliamo il numero di campioni sincronizzati per quell'anno
            # Consideriamo sia i canali AIA che Ic_noLimbDark
            available_channels = self.wavelengths + ["Ic_noLimbDark"]
            count = min(self.z[year][wl].shape[0] for wl in available_channels)
            
            for local_idx in range(count):
                self.indices.append((year, local_idx))
        
        self.N = len(self.indices)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        year, local_idx = self.indices[idx]
        
        # 1. Caricamento e Normalizzazione dei 9 canali AIA
        aia_imgs = []
        for i, wl in enumerate(self.wavelengths):
            img = self.z[year][wl][local_idx].astype(np.float32)
            img_scaled = 0.01 * img
            img_transformed = np.sign(img_scaled) * np.log1p(np.abs(img_scaled))
            if hasattr(self, 'means') and self.means is not None:
                img_transformed = (img_transformed - self.means[i]) / (self.stds[i] + 1e-8)        
            aia_imgs.append(img_transformed)
            # Normalizzazione Robust Percentile (2.5% - 99.5%)
            # p_low, p_high = np.percentile(img, [2.5, 99.5])
            # img = np.clip(img, p_low, p_high)
            # img = (img - p_low) / (p_high - p_low + 1e-6)
            
            # aia_imgs.append(img)
            
        # Stack dei 9 canali: [9, H, W]
        aia_stack = np.stack(aia_imgs, axis=0)
        
        # 2. Gestione Ic_noLimbDark e Maschera
        ic_img = self.z[year]["Ic_noLimbDark"][local_idx].astype(np.float32)
        
        # Generazione maschera (assumendo che clv_correction sia definita globalmente)
        # Se clv_correction non è disponibile, sostituisci con la tua logica di soglia
        mask = clv_correction(ic_img) 
        
        # 3. Resize se l'immagine nel Zarr non è già del target_size
        # Usiamo zoom solo se necessario (operazione costosa)
        current_h, current_w = aia_stack.shape[1], aia_stack.shape[2]
        if current_h != self.target_size or current_w != self.target_size:
            scale = self.target_size / current_h
            # Resize AIA (Order 1 o 3 per qualità)
            aia_stack = zoom(aia_stack, (1, scale, scale), order=1)
            # Resize Maschera (Order 0 per preservare labels)
            mask = zoom(mask.astype(np.float32), (scale, scale), order=0)
            # Resize IC originale
            ic_img = zoom(ic_img, (scale, scale), order=1)

        # 4. Conversione in Tensori
        image_tensor = torch.from_numpy(aia_stack).float()
        
        if self.num_classes == 1:
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) # [1, H, W]
        else:
            mask_tensor = torch.from_numpy(mask).long().unsqueeze(0)  # [1, H, W]
            
        ic_tensor = torch.from_numpy(ic_img).float().unsqueeze(0) # [1, H, W]

        batch = {
            'image': image_tensor,           # [9, H, W]
            'mask': mask_tensor,             # [1, H, W]
            'ic_no_limb_dark': ic_tensor     # [1, H, W]
        }

        if self.transform:
            batch = self.transform(batch)

        return batch


# def compute_limb_darkening(image, a=0.3, b=0.1, solar_radius_arcsec=960, pixel_scale=0.5):
#     ny, nx = image.shape
#     center_x, center_y = nx // 2, ny // 2 
#     solar_radius_pixels = solar_radius_arcsec / pixel_scale

#     y, x = np.indices((ny, nx))
#     r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

#     grid = r / solar_radius_pixels

#     mask = grid <= 1.0

#     mu = np.zeros_like(grid)
#     mu[mask] = np.sqrt(1 - grid[mask]**2)
#     limb_darkening_factor = (1 - a * (1 - mu) - b * (1 - mu)**2)
#     limb_darkening_factor[~mask] = 1.0


#     corrected_image = image / limb_darkening_factor
#     norm_corrected_image=corrected_image/np.max(corrected_image)
#     return norm_corrected_image

def limb_darkening_model(radius, *coeffs):
    return np.polyval(coeffs, radius)

TARGET_RADIUS = 1583.0

def clv_correction(image, center=None, poly_degree=5):
    # Always define x and y coordinates
    y, x = np.indices(image.shape)
    
    if center is None:
        mask = image > (np.mean(image) * 0.5)
        center_y, center_x = center_of_mass(mask)
        center = (center_x, center_y)

    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    r_flat = r.flatten()
    image_flat = image.flatten()
    radial_bins = np.arange(0, r.max(), 1)
    radial_sums, _ = np.histogram(r_flat, bins=radial_bins, weights=image_flat)
    radial_counts, _ = np.histogram(r_flat, bins=radial_bins)
    radial_profile = radial_sums / np.maximum(radial_counts, 1)  # Avoid division by zero
    radial_centers = (radial_bins[:-1] + radial_bins[1:]) / 2

    valid_mask = radial_counts > 0

    if not np.any(valid_mask):
        raise ValueError("No valid data in the radial profile for fitting.")

    popt, _ = curve_fit(
            limb_darkening_model, 
            radial_centers[valid_mask], 
            radial_profile[valid_mask],
            p0=np.zeros(poly_degree + 1)  # Initial guess for coefficients
        )

    clv_model = limb_darkening_model(r, *popt)
    clv_model = np.maximum(clv_model, 1)
    corrected_image = image / clv_model  
    sunspot_mask = (corrected_image < 0.78).astype(int)                                   #ombra e penombra
    # sunspot_mask = ((corrected_image >= 0.7) & (corrected_image <= 0.85)).astype(int)     #penombra
    #sunspot_mask = (corrected_image <= 0.50).astype(int)                                  #ombra
    
    background = r > 197
    sunspot_mask[background] = 0

    return sunspot_mask
"""
Dataset di segmentazione binaria SDO.

Riusa la pipeline di caricamento/normalizzazione di SDO_Dataset_channels_FAST
(normalizzazione fisica con statistiche globali dal JSON) e aggiunge la
generazione della maschera binaria via clv_correction(Ic_noLimbDark).

Restituisce un dict:
    {
        'image': torch.FloatTensor  shape [9, H, W]   # 9 canali AIA/HMI normalizzati
        'mask':  torch.FloatTensor  shape [1, H, W]   # maschera binaria target
    }
"""
import json

import numpy as np
import torch
import zarr
from scipy.ndimage import zoom
from torch.utils.data import Dataset

# clv_correction è definita nel tuo progetto. Importa dal tuo modulo.
# Lasciamo un import "soft" con fallback per non rompere l'avvio.



class SDO_BinarySeg_Dataset(Dataset):
    """
    Dataset di segmentazione binaria.

    Args:
        zarr_path (str): percorso al file .zarr ordinato.
        stats_json_path (str): JSON con statistiche globali (stesso del pretrain).
        list_year (list): anni da caricare (es. [2014, 2015]).
        wavelengths (list): 9 canali AIA/HMI nello stesso ordine del pretrain.
        target_size (int): dimensione finale H=W (es. 1024).
        transform (callable, optional): applicato al dict di output.
        return_ic (bool): se True, include 'ic_no_limb_dark' nel dict (utile per debug).
    """

    def __init__(self, zarr_path, stats_json_path, list_year, wavelengths,
                 target_size, transform=None, return_ic=False,
                 valid_mask_threshold: float = 0.0,
                 return_valid_mask: bool = True):
        """
        Args aggiuntivi:
            valid_mask_threshold: soglia su Ic_noLimbDark raw per definire il
                disco solare. Pixel con ic_raw > threshold sono 'on-disk'.
            return_valid_mask: se True aggiunge 'valid_mask' al dict (necessaria
                per loss off-disk-aware).
        """


        self.z = zarr.open(zarr_path, mode='r')
        self.list_year = list_year
        self.wavelengths = wavelengths
        self.target_size = target_size
        self.transform = transform
        self.return_ic = return_ic
        self.valid_mask_threshold = valid_mask_threshold
        self.return_valid_mask = return_valid_mask

        with open(stats_json_path, 'r') as f:
            self.stats = json.load(f)

        # Costruisci indice globale (anno, indice locale)
        self.indices = []
        for year in self.list_year:
            available = self.wavelengths + ["Ic_noLimbDark"]
            count = min(self.z[year][wl].shape[0] for wl in available)
            for local_idx in range(count):
                self.indices.append((year, local_idx))

        self.N = len(self.indices)
        self.scale_factor = None  # calcolato pigramente alla prima chiamata

    def __len__(self):
        return self.N

    # ------------------------------------------------------------------ #
    # Normalizzazione fisica per canale (identica a SDO_Dataset_channels_FAST)
    # ------------------------------------------------------------------ #
    def _normalize_channel(self, img_data, wl):
        img_data = np.nan_to_num(img_data, nan=0.0)

        if wl == 'Magnetogram':
            limite = self.stats[wl]["clip_max"]
            img_clipped = np.clip(img_data, -limite, limite)
            img_scaled = 0.01 * img_clipped
            img_norm = np.sign(img_scaled) * np.log1p(np.abs(img_scaled))

        elif wl == 'Ic_noLimbDark':
            p_min = self.stats[wl]["p_min"]
            p_max = self.stats[wl]["p_max"]
            img_clipped = np.clip(img_data, p_min, p_max)
            den = (p_max - p_min) if (p_max - p_min) > 0 else 1e-8
            img_norm = (img_clipped - p_min) / den

        else:  # EUV/UV channels
            p_max_log = self.stats[wl]["p_max_log"]
            img_zeroed = np.clip(img_data, 0, None)
            img_log = np.log1p(img_zeroed * 0.01)
            img_clipped_log = np.clip(img_log, 0, p_max_log)
            den = p_max_log if p_max_log > 0 else 1e-8
            img_norm = img_clipped_log / den

        return img_norm.astype(np.float32)

    def __getitem__(self, idx):
        year, local_idx = self.indices[idx]

        # 1. Carica e normalizza i 9 canali AIA/HMI
        imgs = []
        for wl in self.wavelengths:
            img_raw = self.z[year][wl][local_idx].astype(np.float32)
            imgs.append(self._normalize_channel(img_raw, wl))
        stacked = np.stack(imgs, axis=0)  # [9, H, W]

        # 2. Carica Ic_noLimbDark (NON normalizzato) e genera maschera + valid_mask
        ic_raw = self.z[year]["Ic_noLimbDark"][local_idx].astype(np.float32)
        ic_raw = np.nan_to_num(ic_raw, nan=0.0)
        mask = clv_correction(ic_raw)         # maschera binaria target {0, 1}
        mask = mask.astype(np.float32)

        # valid_mask: dentro il disco solare = 1, off-disk = 0
        # Off-disk → ic_raw ≈ 0; on-disk → ic_raw >> 0 (intensità del continuo)
        valid_mask = (ic_raw > self.valid_mask_threshold).astype(np.float32)

        # 3. Resize a target_size
        if stacked.shape[-1] != self.target_size:
            if self.scale_factor is None:
                self.scale_factor = self.target_size / stacked.shape[-1]
            stacked = zoom(stacked, (1, self.scale_factor, self.scale_factor), order=1)
            # order=0 sia per la maschera target che per la valid_mask (preserva discrete)
            mask = zoom(mask, (self.scale_factor, self.scale_factor), order=0)
            valid_mask = zoom(valid_mask, (self.scale_factor, self.scale_factor), order=0)
            if self.return_ic:
                ic_raw = zoom(ic_raw, (self.scale_factor, self.scale_factor), order=1)

        # 4. Binarizza dopo il resize
        mask = (mask > 0.5).astype(np.float32)
        valid_mask = (valid_mask > 0.5).astype(np.float32)

        # 5. Tensorizza
        image_tensor = torch.from_numpy(stacked).float()                  # [9, H, W]
        mask_tensor  = torch.from_numpy(mask).float().unsqueeze(0)        # [1, H, W]

        batch = {'image': image_tensor, 'mask': mask_tensor}
        if self.return_valid_mask:
            batch['valid_mask'] = torch.from_numpy(valid_mask).float().unsqueeze(0)
        if self.return_ic:
            batch['ic_no_limb_dark'] = torch.from_numpy(ic_raw).float().unsqueeze(0)

        if self.transform:
            batch = self.transform(batch)

        return batch


# ---------------------------------------------------------------------- #
# Augmentations per dict
# ---------------------------------------------------------------------- #
class JointAugment:
    """
    Augmentations sicure su immagini multi-canale: flip H/V e rotazioni di 90°.
    Applicate identicamente a 'image' e 'mask' (e 'ic_no_limb_dark' se presente).
    """
    def __init__(self, p_hflip=0.5, p_vflip=0.5, p_rot90=0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90

    def _apply(self, batch, fn):
        for k in ('image', 'mask', 'valid_mask', 'ic_no_limb_dark'):
            if k in batch:
                batch[k] = fn(batch[k])
        return batch

    def __call__(self, batch):
        if torch.rand(1).item() < self.p_hflip:
            batch = self._apply(batch, lambda t: torch.flip(t, dims=[-1]))
        if torch.rand(1).item() < self.p_vflip:
            batch = self._apply(batch, lambda t: torch.flip(t, dims=[-2]))
        if torch.rand(1).item() < self.p_rot90:
            k = int(torch.randint(1, 4, (1,)).item())
            batch = self._apply(batch, lambda t: torch.rot90(t, k, dims=[-2, -1]))
        return batch