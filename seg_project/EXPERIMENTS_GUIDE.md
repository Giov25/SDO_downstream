# 🧪 Guida ai 3 Esperimenti di Segmentazione

## Panoramica

Sono stati configurati **3 tipi di esperimenti** per confrontare diverse strategie di training:

| Esperimento | Pretrained Weights | Encoder | Decoder | Checkpoint Nome |
|------------|-------------------|---------|---------|-----------------|
| **1. Frozen** ❄️ | ✅ Sì (MAE) | 🔒 Frozen | ✅ Trainable | `Frozen_MAE_Seg_Deformer.pth` |
| **2. Fine-tuning** 🔥 | ✅ Sì (MAE) | ✅ Trainable | ✅ Trainable | `Finetuning_MAE_Seg_Deformer.pth` |
| **3. From Scratch** 🆕 | ❌ No | ✅ Trainable | ✅ Trainable | `Scratch_MAE_Seg_Deformer.pth` |

---

## 🚀 Come Lanciare gli Esperimenti

### Esperimento 1: Encoder FROZEN (Feature Extraction)
```bash
sbatch train_frozen.sbatch
```
**Cosa fa:**
- Carica i pesi pretrained dal task MAE di ricostruzione
- **Freeza completamente l'encoder** (no backprop)
- Allena solo il decoder di segmentazione
- Utile per: vedere quanto sono buone le features estratte dal MAE

**Quando usare:**
- Quando hai pochi dati di training
- Quando vuoi un training veloce
- Per fare transfer learning puro

---

### Esperimento 2: FINE-TUNING (Full Training)
```bash
sbatch train_finetuning.sbatch
```
**Cosa fa:**
- Carica i pesi pretrained dal task MAE di ricostruzione
- **Allena tutto**: encoder + decoder
- L'encoder parte da pesi pretrained ma si adatta al task
- Utile per: sfruttare il pretrain ma adattarlo alla segmentazione

**Quando usare:**
- Quando hai abbastanza dati per il fine-tuning
- Quando vuoi ottenere le migliori performance
- Standard per transfer learning completo

---

### Esperimento 3: FROM SCRATCH (Random Init)
```bash
sbatch train_scratch.sbatch
```
**Cosa fa:**
- **NON carica pesi pretrained**
- Inizializza tutto random
- Allena encoder + decoder da zero
- Utile per: baseline senza pretrain, vedere il valore del pretrain

**Quando usare:**
- Per avere un baseline di confronto
- Per verificare l'utilità del pretrain
- Quando il pretrain non è disponibile

---

## 🔄 Riprendere il Training (dopo 24h)

Ogni esperimento ha il suo script di resume che continua **la stessa run WandB**:

### Resume Frozen:
```bash
sbatch resume_frozen.sbatch
```

### Resume Fine-tuning:
```bash
sbatch resume_finetuning.sbatch
```

### Resume Scratch:
```bash
sbatch resume_scratch.sbatch
```

⚠️ **IMPORTANTE:** Usa lo script di resume che corrisponde all'esperimento originale!

---

## 📊 Monitoraggio su WandB

Ogni esperimento crea una **run separata** su WandB nel progetto `seg-sdo`:
- Run frozen: `run_frozen_xxx`
- Run finetuning: `run_finetuning_xxx` 
- Run scratch: `run_scratch_xxx`

I grafici da monitorare:
- `val/dice_score`: metrica principale (senza background)
- `val/dice_score_T`: con background
- `train/loss` vs `val/loss`: per vedere overfitting
- `learning_rate`: andamento dello scheduler

---

## 📁 Checkpoint Salvati

I checkpoint vengono salvati in:
```
/home/gpatane/checkpoints/seg_project/checkpoints/
├── Frozen_MAE_Seg_Deformer.pth       # Exp 1: Frozen encoder
├── Finetuning_MAE_Seg_Deformer.pth   # Exp 2: Fine-tuning
└── Scratch_MAE_Seg_Deformer.pth      # Exp 3: From scratch
```

Ogni checkpoint contiene:
- Weights del modello
- Stato optimizer e scheduler
- Epoca corrente
- Best dice score
- WandB run ID (per resume)

---

## 🔍 Controllare lo Stato di un Checkpoint

Usa lo script helper per vedere le info:

```bash
# Checkpoint frozen
python check_checkpoint.py --pretrained --freeze

# Checkpoint finetuning
python check_checkpoint.py --pretrained

# Checkpoint scratch
python check_checkpoint.py
```

Oppure specifica direttamente il path:
```bash
python check_checkpoint.py /path/to/checkpoint.pth
```

---

## 🎯 Workflow Tipico

### Setup Iniziale (Giorno 1)
```bash
# Lancia i 3 esperimenti in parallelo (se hai 3 GPU)
sbatch train_frozen.sbatch
sbatch train_finetuning.sbatch  
sbatch train_scratch.sbatch

# Oppure uno alla volta
sbatch train_frozen.sbatch
```

### Dopo 24 ore (Giorno 2+)
```bash
# Controlla lo stato
python check_checkpoint.py --pretrained --freeze

# Riprendi l'esperimento
sbatch resume_frozen.sbatch
```

### Ripeti fino a completamento
Continua a fare submit degli script `resume_*.sbatch` finché il training non raggiunge le 200 epoche (o la convergenza).

---

## 📈 Confronto dei Risultati

Dopo che tutti gli esperimenti sono completati, confronta:

1. **Performance finale**: quale ottiene il miglior `val/dice_score`?
2. **Velocità di convergenza**: quale converge prima?
3. **Stabilità**: quale ha meno oscillazioni?
4. **Overfitting**: quale generalizza meglio (gap train-val)?

### Aspettative Tipiche:
- **Frozen**: Converge velocemente ma performance limitata
- **Fine-tuning**: Migliori performance, converge bene
- **Scratch**: Richiede più epoche, potrebbe underperform se dati limitati

---

## 🛠️ Parametri Configurabili

Puoi modificare nei file `.sbatch`:
- `--batch_size`: default 3 (aumenta se hai più VRAM)
- `--epochs`: default 200
- `--lr`: learning rate, default 1e-4
- `--model`: architettura (default MAE_Seg_Deformer)

---

## ❓ FAQ

**Q: Posso cambiare il learning rate durante il resume?**  
A: Sì, basta modificare `--lr` nello script resume. L'optimizer verrà aggiornato.

**Q: Come faccio a sapere quale esperimento sta andando meglio?**  
A: Guarda su WandB, vai nella vista "Compare runs" e confronta i 3 esperimenti.

**Q: Cosa succede se lancio train invece di resume?**  
A: Sovrascrive il checkpoint esistente e ricomincia da capo.

**Q: Posso cambiare da frozen a fine-tuning a metà training?**  
A: No, cambierebbero i pesi trainabili. Devi ricominciare da capo.

**Q: Quanto tempo ci vuole per 200 epoche?**  
A: Dipende dal dataset. Con batch_size=3, calcola ~3-4 giorni (3-4 submit da 24h).

---

## 📝 Note Tecniche

### Differenze Implementative:

**Frozen Encoder:**
```python
for param in model.encoder.parameters():
    param.requires_grad = False
```

**Fine-tuning:**
```python
mae_backbone.load_state_dict(pretrained_weights)
for param in model.encoder.parameters():
    param.requires_grad = True
```

**Scratch:**
```python
# No pretrained weights loaded
for param in model.encoder.parameters():
    param.requires_grad = True
```

---

## 🎓 Consigli

1. **Lancia prima frozen**: è il più veloce, ti dà subito un'idea
2. **Monitora overfitting**: se `val_loss` aumenta mentre `train_loss` scende, riduci learning rate
3. **Salva i log**: gli output `.out` e `.err` sono utili per debug
4. **Backup checkpoint**: fai copia dei `.pth` prima di sperimentare
5. **Confronta tutto su WandB**: usa "Compare runs" per visualizzare side-by-side

Buon training! 🚀
