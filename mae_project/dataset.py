
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
    
import json
class SDO_Dataset_channels_FAST(Dataset):
    """
    Dataset ottimizzato per immagini SDO (AIA/HMI) da formato Zarr ordinato.
    Utilizza normalizzazione a livello di dataset (globale) pre-calcolata
    per preservare la dinamica fisica degli eventi solari e massimizzare la velocità.
    """
    
    def __init__(self, zarr_path, stats_json_path, list_year, wavelengths, target_size, transform=None):
        """
        Args:
            zarr_path (str): Percorso al file .zarr contenente i dati ordinati.
            stats_json_path (str): Percorso al file .json con le statistiche globali precalcolate.
            list_year (list): Lista di anni da caricare (es. ['2011', '2012']).
            wavelengths (list): Lista dei canali (es. ['131A', 'Magnetogram']).
            target_size (int): Dimensione finale desiderata (es. 256 per [C, 256, 256]).
            transform (callable, optional): Trasformazioni PyTorch aggiuntive.
        """
        self.z = zarr.open(zarr_path, mode='r')
        self.list_year = list_year
        self.wavelengths = wavelengths
        self.target_size = target_size
        self.transform = transform
        
        # 1. Carica le statistiche globali
        with open(stats_json_path, 'r') as f:
            self.stats = json.load(f)
            
        # 2. Crea l'indice globale (Mappa un singolo intero [0...N] alla tupla (Anno, Indice_Locale))
        self.indices = []
        for year in self.list_year:
            # Assicuriamoci che tutti i canali abbiano lo stesso numero di frame per quell'anno
            # Prendiamo il minimo per evitare errori di out-of-bounds se un canale è monco
            count = min(self.z[year][wl].shape[0] for wl in self.wavelengths)
            for local_idx in range(count):
                self.indices.append((year, local_idx))
        
        self.N = len(self.indices)
        self.scale_factor = None # Verrà calcolato alla prima chiamata
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, idx):
        year, local_idx = self.indices[idx]
        year_str = str(year)
        imgs = []
        for wl in self.wavelengths:
            # Estrazione immagine grezza
            img_data = self.z[year][wl][local_idx].astype(np.float32)
            
            # Gestione sicura e universale dei NaN (Fuori dal disco o errori del sensore)
            img_data = np.nan_to_num(img_data, nan=0.0)

            # ====================================================
            # NORMALIZZAZIONE FISICA BASATA SU STATISTICHE GLOBALI
            # ====================================================
            
            # 1. MAGNETOGRAMMA (Campo Vettoriale: Valori Negativi e Positivi)
            if wl == 'Magnetogram':
                limite = self.stats[wl]["clip_max"]
                img_clipped = np.clip(img_data, -limite, limite)
                
                # Signum-Log: Comprime mantenendo la polarità Nord/Sud
                img_scaled = 0.01 * img_clipped 
                img_norm = np.sign(img_scaled) * np.log1p(np.abs(img_scaled))

            # 2. FOTOSFERA (Intensità del continuo: Valori lineari positivi)
            elif wl == 'Ic_noLimbDark':
                p_min = self.stats[wl]["p_min"]
                p_max = self.stats[wl]["p_max"]
                
                img_clipped = np.clip(img_data, p_min, p_max)
                
                # Min-Max Scaling Lineare. Prevenzione divisione per zero.
                den = (p_max - p_min) if (p_max - p_min) > 0 else 1e-8
                img_norm = (img_clipped - p_min) / den # Range finale: [0, 1]

            # 3. CANALI EUV / UV (131A, 171A, 304A...: Variazioni esponenziali, flares)
            else:
                p_max_log = self.stats[wl]["p_max_log"]

                # Taglia i negativi fisicamente invalidi (rumore CCD)
                img_zeroed = np.clip(img_data, 0, None)
                
                # Log1p compresso
                img_log = np.log1p(img_zeroed * 0.01) 
                img_clipped_log = np.clip(img_log, 0, p_max_log)
                
                # Normalizzazione. Prevenzione divisione per zero.
                den = p_max_log if p_max_log > 0 else 1e-8
                img_norm = img_clipped_log / den # Range finale: [0, 1]

            # Aggiungiamo l'immagine processata alla lista dei canali
            imgs.append(img_norm.astype(np.float32))

        # ====================================================
        # CREAZIONE DEL TENSORE E RESIZE
        # ====================================================
        
        # Stack per creare l'array [Canali, H, W]
        stacked = np.stack(imgs, axis=0)
        
        # Ridimensionamento spaziale (Se target_size è diverso dalla risoluzione nativa)
        # scipy.ndimage.zoom è veloce, order=1 equivale a bilinear interpolation
        if stacked.shape[-1] != self.target_size:
            if self.scale_factor is None:
                # Calcola il fattore di scala solo la prima volta (es. 256 / 1024 = 0.25)
                self.scale_factor = self.target_size / stacked.shape[-1]
                
            # zoom agisce su [C, H, W]. Vogliamo scalare H e W, ma lasciare invariati i Canali (zoom=1)
            stacked = zoom(stacked, (1, self.scale_factor, self.scale_factor), order=1)

        # Conversione in tensore PyTorch
        tensor = torch.from_numpy(stacked)

        # Applicazione di eventuali trasformazioni aggiuntive (es. Data Augmentation)
        if self.transform:
            tensor = self.transform(tensor)
        canale_rif = self.wavelengths[0]
        date_str = self.z[year_str][canale_rif].attrs["DATE-OBS"][local_idx]
        return tensor, date_str