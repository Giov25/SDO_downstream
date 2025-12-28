# WandB Integration Guide

## 🚀 Setup Rapido

### Per lo Script Python (`prova_script_rapido.py`)
```bash
# Esegui lo script - wandb è già configurato
python prova_script_rapido.py
```

### Per il Notebook (`new_segmentation.ipynb`)
Esegui le celle nell'ordine:
1. **Cella imports** (cella 1)
2. **Cella configurazione modello** (cella 3)
3. **Cella dataset** (cella 5)
4. **Cella configurazione loss** (cella 13)
5. **Cella inizializzazione wandb** (NUOVA - prima del training)
6. **Cella training** (cella con train_model)
7. **Cella plot** (visualizzazione risultati)
8. **Cella chiusura wandb** (NUOVA - alla fine)

## 📊 Cosa Viene Loggato

### Durante Ogni Epoch
- ✅ **train/loss**: Loss sul training set
- ✅ **val/loss**: Loss sul validation set
- ✅ **val/dice_score**: Dice score sul validation set
- ✅ **learning_rate**: Learning rate corrente
- ✅ **epoch_time**: Tempo impiegato per l'epoch

### Ogni 5 Epochs
- 🖼️ **validation/predictions**: 6 esempi di predizioni con overlay
  - Input image
  - Ground truth overlay (rosso)
  - Prediction overlay (blu)

### Al Salvataggio del Best Model
- 🏆 **best_dice**: Miglior Dice score raggiunto
- 🏆 **best_epoch**: Epoch del miglior modello
- 💾 **Model file**: Checkpoint salvato su wandb

### Al Termine del Training
- 📈 **training_summary**: Plot finale di loss e Dice score
- 📊 Summary metrics:
  - `best_val_dice`
  - `best_epoch`
  - `final_train_loss`
  - `final_val_loss`
  - `final_val_dice`

## 🎯 Configurazione Wandb

### Parametri Loggati
```python
{
    "architecture": "MAE + DeepDecoder",
    "encoder": "MAE (frozen/unfrozen)",
    "decoder": "DeepDecoder with Attention",
    "patch_size": 14,
    "learning_rate": 1e-4,
    "optimizer": "Adam",
    "weight_decay": 1e-5,
    "scheduler": "Warmup(5) + CosineAnnealing",
    "loss": "CombinedLoss (Dice + BCE)",
    "pos_weight": <calculated>,
    "dropout": 0.1,
    "batch_size": 4,
    "epochs": 50,
    "gradient_clipping": 1.0,
    "train_samples": <dataset_size>,
    "val_samples": <dataset_size>,
    "wavelengths": [...],
    "image_size": 672,
    "mask_size": 224
}
```

## 🔧 Personalizzazione

### Modificare Frequenza Log Immagini
```python
train_model(
    ...
    log_images_every=10  # Default: 5 (ogni 5 epochs)
)
```

### Modificare Numero di Immagini Loggrate
Modifica in `utils_2.py`, funzione `log_predictions_to_wandb`:
```python
log_predictions_to_wandb(model, test_loader, device, num_images=10, ...)  # Default: 6
```

### Cambiare Nome Progetto
```python
wandb.init(
    project="il-tuo-progetto",  # Cambia questo
    name="esperimento-1",       # E questo
    ...
)
```

## 🎨 Visualizzazioni in WandB

### 1. Grafici Automatici
- Loss curves (train vs validation)
- Dice score progression
- Learning rate schedule

### 2. Predizioni Visive
Ogni N epochs vedrai:
- Grid di 6 immagini
- Ogni immagine mostra: Input | Ground Truth | Prediction
- Overlay con colori diversi per GT (rosso) e Pred (blu)

### 3. Model Tracking
- Gradients histogram
- Parameters histogram
- Model graph (se supportato)

## 📝 Best Practices

### ✅ DO
- Usa nomi descrittivi per i run
- Aggiungi tag rilevanti per filtrare esperimenti
- Commenta i run con note importanti
- Salva i checkpoint migliori

### ❌ DON'T
- Non cambiare il nome del progetto durante esperimenti correlati
- Non loggare troppo frequentemente (può rallentare il training)
- Non dimenticare di chiamare `wandb.finish()` alla fine

## 🔍 Accesso ai Risultati

1. **Durante il Training**: Controlla l'URL stampato all'inizio
2. **Dopo il Training**: Vai su [wandb.ai](https://wandb.ai)
3. **Dashboard**: Tutti i tuoi progetti e run

## 🐛 Troubleshooting

### Problema: "wandb not logged in"
```bash
wandb login
# Poi incolla la tua API key
```

### Problema: "Too many images logged"
Riduci `log_images_every` o `num_images`

### Problema: "WandB rallenta il training"
```python
# Disabilita temporaneamente
wandb_run = None  # invece di wandb.run
```

### Problema: "Voglio usare offline mode"
```python
import os
os.environ["WANDB_MODE"] = "offline"
wandb.init(...)
```

## 🎓 Risorse Utili

- [WandB Documentation](https://docs.wandb.ai)
- [WandB Examples](https://github.com/wandb/examples)
- [Your Project Dashboard](https://wandb.ai/<username>/solar-segmentation-deep-decoder)

## 📧 Support

Per problemi con wandb:
- WandB Support: [support@wandb.com](mailto:support@wandb.com)
- Community Forum: [community.wandb.ai](https://community.wandb.ai)
