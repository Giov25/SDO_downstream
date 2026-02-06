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

# class SunDataset(Dataset):
#     def __init__(self, folder, transform=None):
#         """
#         Args:
#             folder (str): Path to the folder containing FITS files.
#             threshold (float): Threshold for mask generation.
#             diff (float): Difference for mask generation.
#             transform (callable, optional): Transform to be applied to images and masks.
#         """
#         self.folder = folder
#         self.file_list = sorted(os.listdir(folder))
#         self.transform = transform

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         file_name = self.file_list[idx]
#         file_path = os.path.join(self.folder, file_name)
        
#         file = fits.open(file_path)
#         data = file[1].data
#         data = np.nan_to_num(data, nan=0)
#         file.close()
#         norm_corrected_image = compute_limb_darkening(data)
#         mask = clv_correction(data)
#         image_3_channels_np = np.stack([norm_corrected_image] * 3, axis=0)
#         image = torch.tensor(image_3_channels_np/3, dtype=torch.float32)
#         mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
#         batch = {'image': image, 'mask': mask}
        
#         if self.transform:
#             batch = self.transform(batch)

#         return batch
    

class PhotosphereDataset(Dataset):
    def __init__(self, folder_path, list_year, transform=None, wavelength = 'Ic_noLimbDark'):
        """
        Args:
            folder_path (str): Path to the Zarr file.
            list_year (list): Lista degli anni da includere.
            transform (callable, optional): Trasformazioni da applicare.
        """
        self.z = zarr.open(folder_path, mode='r')
        self.list_year = list_year
        self.transform = transform
        self.wavelength = wavelength
        
        # Calcola le lunghezze cumulative per indicizzazione globale
        self.cumulative_lengths = []
        self.year_lengths = {}
        cumulative = 0
        for year in self.list_year:
            year_data = self.z[str(year)][self.wavelength]
            year_length = year_data.shape[0]
            self.year_lengths[year] = year_length
            cumulative += year_length
            self.cumulative_lengths.append(cumulative)
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        
        year_idx = 0
        while idx >= self.cumulative_lengths[year_idx]:
            year_idx += 1
        local_idx = idx if year_idx == 0 else idx - self.cumulative_lengths[year_idx - 1]
        year = self.list_year[year_idx]
        
        data = self.z[str(year)]['Ic_noLimbDark']
        image = da.from_array(data[local_idx]).compute()

        # Apply clv_correction to the original image before any modifications
        mask = torch.tensor(clv_correction(image), dtype=torch.float32).unsqueeze(0)
        
        data = self.z[str(year)][self.wavelength]
        image = da.from_array(data[local_idx]).compute()
        # Create 3-channel image without dividing by 3
        image_3_channels_np = np.stack([image] * 3, axis=0)
        img = torch.tensor(image_3_channels_np, dtype=torch.float32)
        
        batch = {'image': img, 'mask': mask}
        
        if self.transform:
            batch = self.transform(batch)
        
        return batch
    
    

class SDOMosaicZarrDataset_2(Dataset):
    def __init__(self, zarr_path, list_year, wavelengths, target_size=512, transform=None, num_classes=1):
        """
        Args:
            zarr_path (str): zarr path
            list_year (list): list of years to take in to account.
            wavelengths (list): wavelenght list.
            target_size (int): image shape.
            transform (callable, optional).
        """
        self.z = zarr.open(zarr_path, mode='r')
        self.list_year = list_year
        self.wavelengths = wavelengths
        self.target_size = target_size
        self.transform = transform
        self.num_classes = num_classes
        self.cum_counts = []
        cum = 0
        for year in self.list_year:
            count = min(self.z[year][wl].shape[0] for wl in self.wavelengths)
            cum += count
            self.cum_counts.append(cum)
        self.N = cum
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        for i, cum_count in enumerate(self.cum_counts):
            if idx < cum_count:
                year = self.list_year[i]
                local_idx = idx if i == 0 else idx - self.cum_counts[i - 1]
                break
        else:
            raise IndexError("Indice fuori range")
        
        imgs = []
        mask = None
        
        for wl in self.wavelengths:
            if wl == "Ic_noLimbDark":
                img = self.z[year][wl][local_idx]
                mask = clv_correction(img)
                
                # Store the original Ic_noLimbDark image
                ic_no_limb_dark = img.copy()
                
                # Ridimensiona anche la maschera se necessario
                # if mask.shape[0] != self.target_size or mask.shape[1] != self.target_size:
                #     h, w = mask.shape[:2]
                #     crop_h, crop_w = int(h * 0.83), int(w * 0.83)
                #     start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
                #     scale_factor = (self.target_size / mask.shape[0], self.target_size / mask.shape[1])
                #     mask = mask[start_h:start_h + crop_h, start_w:start_w + crop_w]
                #     mask = zoom(mask.astype(np.float32), scale_factor, order=0)  # order=0 per mantenere valori binari
                #     mask = mask.astype(int)
                
                # #Resize Ic_noLimbDark to target size
                # if ic_no_limb_dark.shape[0] != self.target_size or ic_no_limb_dark.shape[1] != self.target_size:
                #     h, w = ic_no_limb_dark.shape[:2]
                #     crop_h, crop_w = int(h * 0.83), int(w * 0.83)
                #     start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
                #     ic_no_limb_dark = ic_no_limb_dark[start_h:start_h + crop_h, start_w:start_w + crop_w]
                    
                #     scale_factor = (self.target_size / ic_no_limb_dark.shape[0], self.target_size / ic_no_limb_dark.shape[1])
                #     ic_no_limb_dark = zoom(ic_no_limb_dark, scale_factor, order=3)                  
                
                ic_no_limb_dark = ic_no_limb_dark.astype(np.float32)
                
                continue
            
            img = self.z[year][wl][local_idx]

            # ✅ CONTROLLO NaN E VALORI INVALIDI
            if np.any(np.isnan(img)):
                raise ValueError(f"NaN trovato in immagine {year}/{wl}/{local_idx}")
            
            if np.any(np.isinf(img)):
                raise ValueError(f"Infinito trovato in immagine {year}/{wl}/{local_idx}")
            
            if img.size == 0:
                raise ValueError(f"Immagine vuota {year}/{wl}/{local_idx}")
            
            if wl == "Magnetogram":
                p_low, p_high = np.min(img), np.max(img)
            else:
                p_low, p_high = np.percentile(img, [2.5, 99.5])
            img = np.clip(img, p_low, p_high)
            img = (img - p_low) / (p_high - p_low + 1e-6)
            
            # # Crop to 83% then resize to target_size
            # if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
            #     # First crop to 83% of original size (center crop)
            #     h, w = img.shape[:2]
            #     crop_h, crop_w = int(h * 0.83), int(w * 0.83)
            #     start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
            #     img = img[start_h:start_h + crop_h, start_w:start_w + crop_w]
                
            #     # Then resize to target_size
            #     scale_factor = (self.target_size / img.shape[0], self.target_size / img.shape[1])
            #     img = zoom(img, scale_factor, order=3)
            
            if img.ndim == 2:
                img = img.astype(np.float32)
            else:
                img = img[..., 0].astype(np.float32)
            
            imgs.append(img)
        
        row1 = np.hstack(imgs[0:3])
        row2 = np.hstack(imgs[3:6])
        row3 = np.hstack(imgs[6:9])
        mosaic = np.vstack((row1, row2, row3))
        
        mosaic = mosaic[np.newaxis, :, :]
        mosaic = np.repeat(mosaic, 3, axis=0)

        img = torch.from_numpy(mosaic).float()
        if self.num_classes == 2:
            mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)  # [1, H, W]
        if self.num_classes == 1:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, H, W]    
        #mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
        
        ic_no_limb_dark_tensor = torch.tensor(ic_no_limb_dark, dtype=torch.float32).unsqueeze(0)
        
        batch = {'image': img, 'mask': mask, 'ic_no_limb_dark': ic_no_limb_dark_tensor}
        
        if self.transform:
            batch = self.transform(batch)
        
        return batch
    




class MC_SolarDataset(Dataset):
    def __init__(self, zarr_path, list_year, wavelengths, target_size=512, transform=None):
        """
        Args:
            zarr_path (str): zarr path
            list_year (list): list of years to take in to account.
            wavelengths (list): wavelenght list.
            target_size (int): image shape.
            transform (callable, optional).
        """
        self.z = zarr.open(zarr_path, mode='r')
        self.list_year = list_year
        self.wavelengths = wavelengths
        self.target_size = target_size
        self.transform = transform

        self.cum_counts = []
        cum = 0
        for year in self.list_year:
            count = min(self.z[year][wl].shape[0] for wl in self.wavelengths)
            cum += count
            self.cum_counts.append(cum)
        self.N = cum
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, idx):
        for i, cum_count in enumerate(self.cum_counts):
            if idx < cum_count:
                year = self.list_year[i]
                local_idx = idx if i == 0 else idx - self.cum_counts[i - 1]
                break
        else:
            raise IndexError("Indice fuori range")
        
        imgs = []
        mask = None
        ic_no_limb_dark = None
        
        for wl in self.wavelengths:
            if wl == "Ic_noLimbDark":
                img = self.z[year][wl][local_idx]
                mask = clv_correction(img)
                
                # Store the original Ic_noLimbDark image
                ic_no_limb_dark = img.copy()
                
                # # Ridimensiona anche la maschera se necessario
                # if mask.shape[0] != self.target_size or mask.shape[1] != self.target_size:
                #     scale_factor = (self.target_size / mask.shape[0], self.target_size / mask.shape[1])
                #     mask = zoom(mask.astype(np.float32), scale_factor, order=0)  # order=0 per mantenere valori binari
                #     mask = mask.astype(int)
                
                # # Resize Ic_noLimbDark to target size
                # if ic_no_limb_dark.shape[0] != self.target_size or ic_no_limb_dark.shape[1] != self.target_size:
                #     scale_factor = (self.target_size / ic_no_limb_dark.shape[0], self.target_size / ic_no_limb_dark.shape[1])
                #     ic_no_limb_dark = zoom(ic_no_limb_dark, scale_factor, order=3)
                
                ic_no_limb_dark = ic_no_limb_dark.astype(np.float32)
                continue
            
            img = self.z[year][wl][local_idx]

            # ✅ CONTROLLO NaN E VALORI INVALIDI
            if np.any(np.isnan(img)):
                raise ValueError(f"NaN trovato in immagine {year}/{wl}/{local_idx}")
            
            if np.any(np.isinf(img)):
                raise ValueError(f"Infinito trovato in immagine {year}/{wl}/{local_idx}")
            
            if img.size == 0:
                raise ValueError(f"Immagine vuota {year}/{wl}/{local_idx}")
            
            if wl == "Magnetogram":
                p_low, p_high = np.min(img), np.max(img)
            else:
                p_low, p_high = np.percentile(img, [2.5, 99.5])
            img = np.clip(img, p_low, p_high)
            img = (img - p_low) / (p_high - p_low)
            
            if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                scale_factor = (self.target_size / img.shape[0], self.target_size / img.shape[1])
                img = zoom(img, scale_factor, order=3)
            
            if img.ndim == 2:
                img = img.astype(np.float32)
            else:
                img = img[..., 0].astype(np.float32)
            
            imgs.append(img)
        
        # Stack come canali separati invece di creare mosaico
        multi_channel_img = np.stack(imgs, axis=0)  # Shape: (9, target_size, target_size)
        
        img = torch.from_numpy(multi_channel_img).float()
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        ic_no_limb_dark_tensor = torch.tensor(ic_no_limb_dark, dtype=torch.float32).unsqueeze(0)
        
        batch = {'image': img, 'mask': mask, 'ic_no_limb_dark': ic_no_limb_dark_tensor}
        
        if self.transform:
            batch = self.transform(batch)
        
        return batch