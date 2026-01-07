
import zarr
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from scipy.ndimage import zoom
import torch
import dask.array as da
import random
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import logging




class SDOMosaicZarrDataset(Dataset):
    def __init__(self, zarr_path, list_year, wavelengths, target_size=224, transform=None, skip_invalid=True, grid_size=3, n_channels=3):
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
        self.grid_size = grid_size                      #number of images per row/column in the mosaic
        self.list_year = list_year                     
        self.wavelengths = wavelengths                  #list of wavelengths
        self.target_size = target_size
        self.transform = transform
        self.skip_invalid = skip_invalid
        self.n_channels = n_channels
        
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
                        if wl == "Magnetogram":
                            p_low, p_high = np.min(img), np.max(img)
                        else:
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
            if self.grid_size == 3:
                row1 = np.hstack(imgs[0:3])
                row2 = np.hstack(imgs[3:6])
                row3 = np.hstack(imgs[6:9])
                mosaic = np.vstack((row1, row2, row3))
            elif self.grid_size == 2:
                row1 = np.hstack(imgs[0:2])
                row2 = np.hstack(imgs[2:4])
                mosaic = np.vstack((row1, row2))
            
            # ✅ CONTROLLO FINALE SUL MOSAICO
            if np.any(np.isnan(mosaic)) or np.any(np.isinf(mosaic)):
                self.nan_count += 1
                raise ValueError(f"NaN/Inf nel mosaico finale per idx {idx}")
            
            mosaic = mosaic[np.newaxis, :, :]
            if self.n_channels == 3:
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
        
class MC_SolarDataset(Dataset):
    def __init__(self, zarr_path, list_year, wavelengths, target_size=256, transform=None):
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
        
        
        for wl in self.wavelengths:
            
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
        
        
        
        if self.transform:
            img = self.transform(img)
        
        return img
    
class SDO_Dataset_channels(Dataset):
    def __init__(self, zarr_path, list_year, wavelengths, target_size=512, transform=None, skip_invalid=True):
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
                        if np.isnan(p_low) or np.isnan(p_high) :#or p_low == p_high:
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
        
        # Concatena le immagini lungo un nuovo asse (canali)
        stacked_imgs = np.stack(imgs, axis=0)  # Shape: (9, target_size, target_size)

        # CONTROLLO FINALE SULLE IMMAGINI IMPILATE
        if np.any(np.isnan(stacked_imgs)) or np.any(np.isinf(stacked_imgs)):
            self.nan_count += 1
            raise ValueError(f"NaN/Inf nelle immagini impilate per idx {idx}")

        # Aggiunge la dimensione temporale (T=1) per la compatibilità con modelli 3D
        # La forma diventa (C, T, H, W) -> (9, 1, target_size, target_size)
        stacked_imgs = stacked_imgs[:, np.newaxis, :, :]
        
        tensor = torch.from_numpy(stacked_imgs).float()
        
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

class SDO_Dataset_channels_FAST(Dataset):
    
    #img_data = np.nan_to_num(img_data, nan=-5000)
    def __init__(self, zarr_path, list_year, wavelengths, target_size, transform=None):
        self.z = zarr.open(zarr_path, mode='r')
        self.list_year = list_year
        self.wavelengths = wavelengths
        self.target_size = target_size
        self.transform = transform
        
        # Indici pre-calcolati
        self.indices = []
        for year in self.list_year:
            count = min(self.z[year][wl].shape[0] for wl in self.wavelengths)
            for local_idx in range(count):
                self.indices.append((year, local_idx))
        
        self.N = len(self.indices)
        self.scale_factor = None
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, idx):
        year, local_idx = self.indices[idx]
        
        imgs = []
        for wl in self.wavelengths:
            img_data = self.z[year][wl][local_idx]
            
            # --- MODIFICA RICHIESTA ---
            # Se la lunghezza d'onda è il magnetogramma, gestisci i NaN
            if str(wl) == 'Magnetogram': 
                img_data = np.nan_to_num(img_data, nan=-5000)
            # ---------------------------
            
            imgs.append(img_data)
        
        # Normalizzazione semplice 0-1
        for i, img in enumerate(imgs):
            p_low, p_high = np.percentile(img, [2.5, 99.5])
            #img_min, img_max = img.min(), img.max()
            imgs[i] = (img - p_low) / (p_high - p_low)
            
        
        # Stack e resize
        stacked = np.stack(imgs, axis=0)
        #return #stacked
        if stacked.shape[-1] != self.target_size:
            if self.scale_factor is None:
                self.scale_factor = self.target_size / stacked.shape[-1]
            stacked = zoom(stacked, (1, self.scale_factor, self.scale_factor), order=0)
        
        tensor = torch.from_numpy(stacked).float()
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor