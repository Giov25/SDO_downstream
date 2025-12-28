# Fixes Applicate per il Problema delle Maschere Nere

## Problema Iniziale
- **Loss bloccata** a ~0.9993 senza miglioramenti
- **Dice Score quasi zero** (0.0003)
- **Il modello predice solo background** (maschere completamente nere)

## Cause Identificate
1. **DiceLoss non efficace** con classi fortemente sbilanciate
2. **Learning rate troppo basso** (1e-3 con SGD)
3. **Configurazione loss incorretta** (num_classes=1 ma to_onehot_y=True)
4. **Decoder troppo profondo** potenziale gradient vanishing
5. **Mancanza di gradient clipping** per stabilità

## Soluzioni Implementate

### 1. **Nuova Loss Function: CombinedLoss**
```python
CombinedLoss = 0.5 * DiceLoss + 0.5 * BCEWithLogitsLoss
```
- Combina Dice Loss per la sovrapposizione e BCE per pixel-wise accuracy
- BCE include `pos_weight` calcolato automaticamente per gestire lo sbilanciamento
- Configurazione corretta: `sigmoid=True` per single class

### 2. **Optimizer Migliorato**
- **Prima**: SGD con lr=1e-3
- **Dopo**: Adam con lr=1e-4
- Adam è più robusto per questo tipo di task

### 3. **Learning Rate Scheduling**
- **Warm-up**: 5 epochs con lr crescente (da 1e-5 a 1e-4)
- **Cosine Annealing**: riduzione graduale fino a 1e-6
- Migliora la convergenza iniziale e la stabilità finale

### 4. **Gradient Clipping**
- `max_grad_norm=1.0` per evitare gradient explosion
- Particolarmente importante con decoder profondi

### 5. **Decoder Migliorato: DeepDecoder**
Caratteristiche:
- **4 blocchi decoder** con 3-4 convoluzioni ciascuno
- **Channel Attention**: focalizza sui canali importanti
- **Spatial Attention**: focalizza sulle regioni rilevanti
- **Residual connections**: migliora il flusso del gradiente
- **Dropout configurabile**: default 0.1 per regolarizzazione

### 6. **Opzione Decoder Base**
Se DeepDecoder è troppo complesso:
- Usa `decoder_type='basic'` per Decoder5 originale
- Riduci/rimuovi dropout

## Come Usare

### Setup del Modello
```python
# Decoder Profondo (raccomandato)
model = MAESegmentationModel(
    mae_model, 
    num_classes=1, 
    freeze_encoder=True, 
    decoder_type='deep',
    dropout=0.1
)

# OPPURE Decoder Base (se problemi con deep)
model = MAESegmentationModel(
    mae_model, 
    num_classes=1, 
    freeze_encoder=True, 
    decoder_type='basic',
    dropout=0.0
)
```

### Training
Esegui le celle nel notebook nell'ordine:
1. Cella configurazione loss e optimizer
2. Cella training con train_model
3. Cella plot risultati

## Risultati Attesi
- **Loss dovrebbe scendere** progressivamente (non rimanere bloccata)
- **Dice Score dovrebbe crescere** da ~0.0 verso 0.3-0.7+
- **Le predizioni non saranno più nere** dopo poche epochs

## Troubleshooting

### Se la loss non scende ancora:
1. Riduci il dropout a 0.05 o 0.0
2. Usa decoder 'basic' invece di 'deep'
3. Aumenta il learning rate a 2e-4
4. Aumenta pos_weight_minimum a 100

### Se overfitting (val loss sale):
1. Aumenta dropout a 0.2
2. Aggiungi weight_decay a 1e-4
3. Usa data augmentation più forte

### Se out of memory:
1. Riduci batch_size
2. Usa decoder 'basic'
3. Riduci num_workers

## File Modificati
- `/seg_project/models.py`: Aggiunto DeepDecoder, ChannelAttention, SpatialAttention, DeepDecoderBlock
- `/seg_project/utils_2.py`: Aggiunto gradient clipping, learning rate logging, wandb logging, e funzione log_predictions_to_wandb
- `/seg_project/new_segmentation.ipynb`: Aggiunte celle di configurazione, training e wandb logging
- `/seg_project/prova_script_rapido.py`: Script completo con wandb integration

## WandB Integration

### Logging Automatico
Durante il training vengono loggati:
- **Loss**: train e validation loss per ogni epoch
- **Dice Score**: validation dice score per ogni epoch
- **Learning Rate**: valore corrente del learning rate
- **Tempo**: tempo per epoch
- **Predizioni**: esempi di predizioni visive ogni 5 epochs
- **Best Model**: salvataggio automatico del miglior modello

### Metriche nel Summary
Al termine del training:
- Best validation Dice score
- Best epoch
- Final train/val loss
- Final validation Dice

### Visualizzazioni
- Plot di loss durante il training
- Plot di Dice score
- Overlay di predizioni vs ground truth
- Input images con predizioni

### Come Usare
```python
# Nel notebook: esegui la cella di inizializzazione wandb PRIMA del training
# Nello script: wandb è già configurato, basta eseguire

# Per disabilitare wandb:
# - Nel notebook: non eseguire la cella wandb init
# - Nello script: commenta la sezione wandb.init()
```

## Note Importanti
⚠️ **L'encoder resta FROZEN** - solo il decoder viene addestrato
✅ **Usa validation set** (v_loader) non test set durante training
✅ **Monitora il learning rate** - deve partire basso e aumentare nel warmup
