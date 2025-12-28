from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass, zoom
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
from astropy.io import fits
import zarr
import dask.array as da
import random
from sklearn.model_selection import train_test_split
import logging

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
    sunspot_mask = (corrected_image < 0.75).astype(int)                                   #ombra e penombra
    # sunspot_mask = ((corrected_image >= 0.7) & (corrected_image <= 0.85)).astype(int)     #penombra
    # sunspot_mask = (corrected_image <= 0.50).astype(int)                                  #ombra
    
    background = r > 197
    sunspot_mask[background] = 0

    return sunspot_mask



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
        
        data = self.z[str(year)][self.wavelength]
        image = da.from_array(data[local_idx]).compute()

        # Apply clv_correction to the original image before any modifications
        mask = torch.tensor(clv_correction(image), dtype=torch.float32).unsqueeze(0)
        
        # Create 3-channel image without dividing by 3
        image_3_channels_np = np.stack([image] * 3, axis=0)
        img = torch.tensor(image_3_channels_np, dtype=torch.float32)
        
        batch = {'image': img, 'mask': mask}
        
        if self.transform:
            batch = self.transform(batch)
        
        return batch
    
class SDOMosaicZarrDataset_2(Dataset):
    def __init__(self, zarr_path, list_year, wavelengths, target_size=224, transform=None):
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
        
        for wl in self.wavelengths:
            if wl == "Ic_noLimbDark":
                img = self.z[year][wl][local_idx]
                mask = clv_correction(img)
                
                # Ridimensiona anche la maschera se necessario
                if mask.shape[0] != self.target_size or mask.shape[1] != self.target_size:
                    scale_factor = (self.target_size / mask.shape[0], self.target_size / mask.shape[1])
                    mask = zoom(mask.astype(np.float32), scale_factor, order=0)  # order=0 per mantenere valori binari
                    mask = mask.astype(int)
                
                continue

            img = self.z[year][wl][local_idx]

            # ✅ CONTROLLO NaN E VALORI INVALIDI
            if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                return None  # DataLoader can filter these out
            if img.size == 0:
                
                raise ValueError(f"Immagine vuota {year}/{wl}/{local_idx}")
            
            
            p_low, p_high = np.percentile(img, [2.5, 99.5])
            img = np.clip(img, p_low, p_high)
            img = (img - p_low) / (p_high - p_low + 1e-6)
            
            if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                scale_factor = (self.target_size / img.shape[0], self.target_size / img.shape[1])
                img = zoom(img, scale_factor, order=3)
            
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
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        batch = {'image': img, 'mask': mask}
        
        if self.transform:
            batch = self.transform(batch)
        
        return batch
    
    


class SDOMosaicZarrDataset(Dataset):
    def __init__(self, zarr_path, list_year, wavelengths, target_size=224, transform=None, skip_invalid=True):
        """
        Args:
            zarr_path (str): zarr path
            list_year (list): list of years to take in to account.
            wavelengths (list): wavelenght list.
            target_size (int): image shape.
            transform (callable, optional).
            skip_invalid (bool): Se True, salta immagini invalide; se False, solleva errore.
        """
        self.z = zarr.open(zarr_path, mode='r')
        self.list_year = list_year
        self.wavelengths = wavelengths
        self.target_size = target_size
        self.transform = transform
        self.skip_invalid = skip_invalid
        
        # Contatori per debug
        self.nan_count = 0
        self.error_count = 0

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
        max_retries = 5  # Numero massimo di tentativi
        
        for attempt in range(max_retries):
            try:
                # Trova anno e indice locale
                for i, cum_count in enumerate(self.cum_counts):
                    if idx < cum_count:
                        year = self.list_year[i]
                        local_idx = idx if i == 0 else idx - self.cum_counts[i - 1]
                        break
                else:
                    raise IndexError(f"Indice fuori range: {idx}")
                
                imgs = []
                
                # Carica le immagini per ogni wavelength
                for wl_idx, wl in enumerate(self.wavelengths):
                    try:
                        # Verifica che il dato esista
                        if year not in self.z:
                            raise KeyError(f"Anno {year} non trovato in zarr")
                        if wl not in self.z[year]:
                            raise KeyError(f"Wavelength {wl} non trovata per anno {year}")
                        if local_idx >= self.z[year][wl].shape[0]:
                            raise IndexError(f"local_idx {local_idx} fuori range per {year}/{wl}")
                        
                        img = self.z[year][wl][local_idx]
                        
                        # ✅ CONTROLLO NaN E VALORI INVALIDI
                        if np.any(np.isnan(img)):
                            raise ValueError(f"NaN trovato in immagine {year}/{wl}/{local_idx}")
                        
                        if np.any(np.isinf(img)):
                            raise ValueError(f"Infinito trovato in immagine {year}/{wl}/{local_idx}")
                        
                        if img.size == 0:
                            raise ValueError(f"Immagine vuota {year}/{wl}/{local_idx}")
                        
                        # Preprocessing
                        p_low, p_high = np.percentile(img, [2.5, 99.5])
                        
                        # Verifica che i percentili siano validi
                        if np.isnan(p_low) or np.isnan(p_high) or p_low == p_high:
                            raise ValueError(f"Percentili invalidi: p_low={p_low}, p_high={p_high}")
                        
                        img = np.clip(img, p_low, p_high)
                        img = (img - p_low) / (p_high - p_low + 1e-6)
                        
                        # Verifica che la normalizzazione sia riuscita
                        if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                            raise ValueError(f"NaN/Inf dopo normalizzazione in {year}/{wl}/{local_idx}")
                        
                        # Resize se necessario
                        if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
                            scale_factor = (self.target_size / img.shape[0], self.target_size / img.shape[1])
                            img = zoom(img, scale_factor, order=3)
                            
                            # Verifica dopo il resize
                            if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                                raise ValueError(f"NaN/Inf dopo resize in {year}/{wl}/{local_idx}")
                        
                        # Conversione formato
                        if img.ndim == 2:
                            img = img.astype(np.float32)
                        else:
                            img = img[..., 0].astype(np.float32)
                        
                        imgs.append(img)
                        
                    except Exception as e:
                        print(f"⚠️  Errore nel caricamento {year}/{wl}/{local_idx}: {e}")
                        if not self.skip_invalid:
                            raise
                        # Prova il prossimo indice
                        idx = (idx + 1) % self.N
                        break
                else:
                    # Se arriviamo qui, tutte le wavelength sono state caricate con successo
                    break
                    
            except Exception as e:
                print(f"⚠️  Errore generale per idx {idx}: {e}")
                self.error_count += 1
                
                if not self.skip_invalid:
                    raise
                
                # Prova con un indice diverso
                idx = (idx + 1) % self.N
                
        else:
            # Se arriviamo qui, abbiamo esaurito i tentativi
            raise RuntimeError(f"Impossibile caricare un'immagine valida dopo {max_retries} tentativi")
        
        try:
            # Crea il mosaico
            row1 = np.hstack(imgs[0:3])
            row2 = np.hstack(imgs[3:6])
            row3 = np.hstack(imgs[6:9])
            mosaic = np.vstack((row1, row2, row3))
            
            # ✅ CONTROLLO FINALE SUL MOSAICO
            if np.any(np.isnan(mosaic)) or np.any(np.isinf(mosaic)):
                self.nan_count += 1
                raise ValueError(f"NaN/Inf nel mosaico finale per idx {idx}")
            
            mosaic = mosaic[np.newaxis, :, :]
            mosaic = np.repeat(mosaic, 3, axis=0)
            
            tensor = torch.from_numpy(mosaic).float()
            
            # ✅ CONTROLLO FINALE SUL TENSOR
            if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                self.nan_count += 1
                raise ValueError(f"NaN/Inf nel tensor finale per idx {idx}")
            
            if self.transform:
                tensor = self.transform(tensor)
                
                # Controllo dopo transform
                if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                    raise ValueError(f"NaN/Inf dopo transform per idx {idx}")
            
            return tensor
            
        except Exception as e:
            print(f"⚠️  Errore nella creazione del mosaico per idx {idx}: {e}")
            if not self.skip_invalid:
                raise
            # Ritorna un tensor di default se tutto fallisce
            return torch.zeros(3, self.target_size * 3, self.target_size * 3, dtype=torch.float32)
    
    def get_stats(self):
        """Restituisce statistiche sui dati invalidi"""
        return {
            "total_samples": self.N,
            "nan_count": self.nan_count,
            "error_count": self.error_count
        }
        
        
        
