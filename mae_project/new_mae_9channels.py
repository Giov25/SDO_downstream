import os
import torch
import numpy as np
import random
import warnings
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

# Importa wandb per il logging
import wandb # <<< CAMBIAMENTO

# I tuoi import personalizzati (assicurati che i path siano corretti)
from dataset import SDO_Dataset_channels_FAST
from mae.MAE import new_mae_trial_small_patches

# Ignora avvisi non critici
warnings.filterwarnings("ignore")

# --- 1. CONFIGURAZIONE ---
# Centralizza tutti gli iperparametri in un dizionario per una facile gestione e logging
config = {
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "zarr_path": "/home/gpatane/Dataset/zarr_file_magnetogram.zarr",
    "wavelengths": ['1700A', '1600A', '335A', '304A', '211A', '193A', '171A', '131A', 'Magnetogram'],
    "target_size": 512,
    "train_years": sorted([2011, 2012, 2013, 2015, 2016]), # Esempio, usa i tuoi anni
    "val_years": sorted([2017, 2018]),
    "test_years": sorted([2014]),
    "batch_size": 64,
    "num_workers": 8, # Aumentato per un caricamento dati potenzialmente più veloce
    "pin_memory": True, # Ottimizzazione per il trasferimento dati CPU -> GPU
    "learning_rate": 1e-6, # Un lr più comune per AdamW, da aggiustare
    "weight_decay": 0.05,
    "num_epochs": 100,
    "save_every": 10, # Salva un checkpoint ogni 5 epoche
    "checkpoint_dir": "./checkpoints", # Directory per salvare i modelli
    "use_amp": True, # Abilita/disabilita Automatic Mixed Precision
}

# --- 2. PREPARAZIONE E RIPRODUCIBILITÀ ---

# Imposta seed per la riproducibilità
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])
random.seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])

# Ottimizzazione per GPU moderne (Ampere+)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# Crea la directory per i checkpoint se non esiste
os.makedirs(config["checkpoint_dir"], exist_ok=True)


# --- 3. DATASET E DATALOADER ---

# Nota: la divisione degli anni è stata semplificata. Adatta con la tua logica originale se necessario.
train_dataset = SDO_Dataset_channels_FAST(config["zarr_path"], config["train_years"], config["wavelengths"], target_size=config["target_size"])
validation_dataset = SDO_Dataset_channels_FAST(config["zarr_path"], config["val_years"], config["wavelengths"], target_size=config["target_size"])

# Ottimizzazione dei DataLoader
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=config["pin_memory"])
val_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=config["pin_memory"])


# --- 4. MODELLO E OTTIMIZZAZIONE ---

device = torch.device(config["device"])
model = new_mae_trial_small_patches().to(device)


try:
    model = torch.compile(model)
    print("Modello compilato con torch.compile() per performance migliori! ⚡️")
except Exception:
    print("torch.compile() non disponibile. Eseguo il modello in modalità standard.")


optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
scaler = GradScaler(enabled=config["use_amp"]) # Per Automatic Mixed Precision (AMP)

# --- 5. TRAINING LOOP ---

def training_loop(model, train_loader, val_loader, optimizer, scaler, config):
    """
    Loop di training ottimizzato con logging wandb e mixed precision.
    """
    # Inizializza il progetto wandb
    wandb.init(project="sdo-mae-pretraining", config=config)
    wandb.watch(model, log="all", log_freq=100) # Monitora gradienti e topologia del modello
    
    print("Inizio del training...")
    for epoch in range(config["num_epochs"]):
        # --- Fase di Training ---
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True) # non_blocking con pin_memory=True
            
            optimizer.zero_grad()
            
            # Automatic Mixed Precision (AMP)
            with autocast(dtype=torch.float16, enabled=config["use_amp"]):
                # Assumiamo che model.training_step() restituisca la loss
                loss = model.training_step(batch, batch_idx) 
            
            # Scalatura della loss e backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Fase di Validazione ---
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = batch.to(device, non_blocking=True)
                
                with autocast(dtype=torch.float16, enabled=config["use_amp"]):
                    # <<< CORREZIONE CRITICA: usa la funzione di validazione per calcolare la loss
                    # ed evita la chiamata ridondante. Assumo che validation_step restituisca la loss.
                    # Se non lo fa, modifica il metodo nel tuo modello.
                    val_loss = model.validation_step(batch, batch_idx)
                    # La chiamata originale: loss, _, _ = model(batch) era ridondante e inefficiente.
                
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # <<< CAMBIAMENTO: Logga le metriche su wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "validation_loss": avg_val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # <<< CAMBIAMENTO: Logica di checkpointing corretta e più robusta
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint_path = os.path.join(config["checkpoint_dir"], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint salvato in: {checkpoint_path}")
            # wandb.save(checkpoint_path) # Opzionale: salva il checkpoint anche su wandb

    wandb.finish() # Chiudi il run di wandb
    print("Training completato.")


# --- 6. ESEGUI IL TRAINING ---
if __name__ == "__main__":
    training_loop(model, train_loader, val_loader, optimizer, scaler, config)