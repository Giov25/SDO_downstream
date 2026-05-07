import numpy as np
import zarr
from tqdm import tqdm

path_originale = "/home/gpatane/Dataset/zarr_file_magnetogram_1024_definitivo.zarr"
path_nuovo = "/home/gpatane/Dataset/zarr_file_magnetogram_1024_ORDINATO.zarr"

zarr_orig = zarr.open(path_originale, mode='r')
zarr_nuovo = zarr.open(path_nuovo, mode='w')

# Canale di riferimento affidabile da cui "rubare" le date
CANALE_RIFERIMENTO = '131A'

for anno in zarr_orig.group_keys():
    gruppo_anno_orig = zarr_orig[anno]
    gruppo_anno_nuovo = zarr_nuovo.require_group(anno)
    
    print(f"\n{'='*40}\nInizio elaborazione ANNO: {anno}\n{'='*40}")
    
    # 1. CALCOLA IL "MASTER INDEX" PER QUESTO ANNO
    if CANALE_RIFERIMENTO in gruppo_anno_orig:
        header_rif = dict(gruppo_anno_orig[CANALE_RIFERIMENTO].attrs)
        date_rif = header_rif.get("DATE-OBS")
        
        # Calcoliamo l'ordine corretto una volta sola per tutto l'anno
        indici_ordinati_master = np.argsort(date_rif)
        num_frames_master = len(indici_ordinati_master)
        print(f"Master index calcolato usando {CANALE_RIFERIMENTO}. ({num_frames_master} frame)")
    else:
        print(f"ERRORE: Canale di riferimento {CANALE_RIFERIMENTO} assente nell'anno {anno}. Salto l'anno.")
        continue

    # 2. APPLICA IL MASTER INDEX A TUTTI I CANALI
    for canale in gruppo_anno_orig.array_keys():
        data_orig = gruppo_anno_orig[canale]
        
        # Controllo di sicurezza: il numero di immagini coincide?
        if data_orig.shape[0] != num_frames_master:
            print(f"   [!] ATTENZIONE: Il canale {canale} ha {data_orig.shape[0]} frame, diverso dai {num_frames_master} di riferimento. Salto l'ordinamento.")
            zarr.copy(data_orig, gruppo_anno_nuovo, name=canale)
            continue
            
        print(f"-> Ordinamento Canale: {canale} (usando il Master Index)")
        
        data_nuovo = gruppo_anno_nuovo.create_dataset(
            canale, shape=data_orig.shape, chunks=data_orig.chunks, dtype=data_orig.dtype
        )
        
        header_orig = dict(data_orig.attrs)
        header_nuovo = {}
        
        # Riordina i metadati (se sono liste lunghe quanto il numero di frame)
        for chiave, valore in header_orig.items():
            if isinstance(valore, list) and len(valore) == num_frames_master:
                header_nuovo[chiave] = [valore[i] for i in indici_ordinati_master]
            else:
                header_nuovo[chiave] = valore
                
        # [OPZIONALE] Se vuoi, aggiungi le date "rubate" al Magnetogramma per comodità futura
        if canale == 'Magnetogram' or canale == 'Ic_noLimbDark' and 'DATE-OBS' not in header_nuovo:
             header_nuovo['DATE-OBS'] = [date_rif[i] for i in indici_ordinati_master]
             
        data_nuovo.attrs.update(header_nuovo)
        
        # Copia i pixel riordinati
        for nuovo_indice, vecchio_indice in enumerate(tqdm(indici_ordinati_master, desc=f"   Copia {canale}", leave=False)):
            data_nuovo[nuovo_indice] = data_orig[vecchio_indice]

print("\n\nELABORAZIONE COMPLETATA!")