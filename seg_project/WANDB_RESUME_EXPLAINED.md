# 🔄 Come Funziona il Resume di WandB

## Il Meccanismo Completo

WandB usa un **Run ID univoco** per identificare ogni esperimento. Il sistema salva questo ID nel checkpoint e lo usa per continuare la stessa run.

---

## 📝 Passo per Passo

### FASE 1: Primo Training (Giorno 1)

#### 1.1 Inizializzazione WandB
```python
# In prova_script_rapido.py linea ~165
run = wandb.init(project="seg-sdo", config=args)
```

**Cosa succede:**
- WandB crea una **nuova run** con un ID univoco (es: `abc123xyz`)
- Puoi vederlo nell'URL: `https://wandb.ai/user/seg-sdo/runs/abc123xyz`
- `run.id` contiene questo ID

#### 1.2 Salvataggio Checkpoint
```python
# In utils_2.py linea ~436
checkpoint_data = {
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    # ... altre metriche ...
    'wandb_run_id': wandb_run.id,        # ← QUESTO È LA CHIAVE!
    'wandb_project': wandb_run.project,   # 'seg-sdo'
    'wandb_entity': wandb_run.entity      # il tuo username
}
torch.save(checkpoint_data, model_save_path)
```

**Nel checkpoint viene salvato:**
```
Frozen_MAE_Seg_Deformer.pth
├─ epoch: 85
├─ model_state_dict: {...}
├─ optimizer_state_dict: {...}
├─ val_dice: 0.8234
├─ wandb_run_id: "abc123xyz"  ← ID UNIVOCO DELLA RUN
├─ wandb_project: "seg-sdo"
└─ wandb_entity: "tuo-username"
```

---

### FASE 2: Resume (Giorno 2)

#### 2.1 Caricamento Checkpoint
```python
# In prova_script_rapido.py linea ~135
checkpoint = torch.load(checkpoint_to_load, map_location=device)
start_epoch = checkpoint.get('epoch', 0)              # 85
best_dice = checkpoint.get('val_dice', 0.0)           # 0.8234
wandb_run_id = checkpoint.get('wandb_run_id', None)  # "abc123xyz"
```

**Il codice recupera:**
- ✅ Epoca dove si era fermato (85)
- ✅ Miglior dice score (0.8234)
- ✅ **WandB Run ID** ("abc123xyz")

#### 2.2 Riconnessione a WandB
```python
# In prova_script_rapido.py linea ~158
if wandb_run_id:  # Se trovato nel checkpoint
    run = wandb.init(
        project="seg-sdo",
        id=wandb_run_id,        # ← USA LO STESSO ID!
        resume="allow",         # ← DICE A WANDB DI CONTINUARE
        config=args
    )
    print(f"Resuming WandB run: {run.url}")
```

**Cosa fa `resume="allow"`:**
- WandB cerca una run esistente con quell'ID
- Se la trova, **continua quella run** (stesso grafico, stessa pagina)
- I nuovi log vengono **aggiunti** ai vecchi
- Le epoche continuano dal punto giusto (86, 87, 88...)

---

## 🎯 Il Trucco: l'ID Univoco

```python
# PRIMA ESECUZIONE
wandb.init(project="seg-sdo")  
# → Crea run con ID = "abc123xyz"
# → Salva nel checkpoint: 'wandb_run_id': "abc123xyz"

# RESUME
wandb.init(project="seg-sdo", id="abc123xyz", resume="allow")
# → WandB: "Conosco questo ID! Continuo quella run"
# → Stessa pagina, stessi grafici, tutto continua
```

---

## 📊 Cosa Vede WandB

### Senza Resume (Sbagliato ❌)
```
Run 1: abc123xyz
├─ Epoch 1-85
├─ Train Loss: [...]
└─ Val Dice: [...] 

Run 2: xyz456abc  ← NUOVA RUN!
├─ Epoch 1-115    ← Riparte da 1!
└─ Val Dice: [...] ← Grafici separati
```
**Problema:** 2 run separate, grafici non continui

### Con Resume (Corretto ✅)
```
Run 1: abc123xyz
├─ Epoch 1-85      ← Dal primo training
├─ Epoch 86-200    ← Dal resume!
├─ Train Loss: [.........................]  ← Continuo!
└─ Val Dice: [...........................]  ← Continuo!
```
**Risultato:** 1 run unica, grafici continui

---

## 🔍 Come Verifica WandB Cosa Riprendere

1. **Controlla l'ID**: `id="abc123xyz"`
2. **Cerca nel progetto**: "Esiste una run con questo ID in seg-sdo?"
3. **Se SÌ**: 
   - Riprende quella run
   - I nuovi log si aggiungono ai vecchi
   - Epoch counter continua (il codice lo gestisce con `start_epoch`)
4. **Se NO e resume="allow"**: 
   - Crea una nuova run con quell'ID
   - (Utile se il checkpoint è stato spostato)

---

## 💾 Informazioni nel Checkpoint

```python
checkpoint = {
    # Training state
    'epoch': 85,                          # Da dove ripartire
    'model_state_dict': {...},           # Pesi del modello
    'optimizer_state_dict': {...},       # Stato optimizer (momentum, ecc)
    'scheduler_state_dict': {...},       # Stato scheduler (warmup, ecc)
    
    # Metrics
    'val_dice': 0.8234,                  # Best score finora
    'train_loss': 0.1234,
    'val_loss': 0.2345,
    
    # WandB info (per il resume)
    'wandb_run_id': 'abc123xyz',         # ← ID univoco della run
    'wandb_project': 'seg-sdo',          # ← Nome progetto
    'wandb_entity': 'tuo-username'       # ← Username/team
}
```

---

## 🎓 Perché Funziona

### Il Flow Completo:

```
TRAINING INIZIALE
┌─────────────────┐
│ wandb.init()    │
│ → crea run      │
│ → id="abc123"   │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Allena 85 epoch │
└────────┬────────┘
         │
         ↓
┌─────────────────────────┐
│ torch.save(checkpoint)  │
│ + wandb_run_id="abc123" │
└────────┬────────────────┘
         │
    [STOP - 24h]
         │
         ↓
┌──────────────────────────┐
│ torch.load(checkpoint)   │
│ → legge id="abc123"      │
└────────┬─────────────────┘
         │
         ↓
┌────────────────────────────┐
│ wandb.init(id="abc123",    │
│             resume="allow")│
│ → riprende stessa run!     │
└────────┬───────────────────┘
         │
         ↓
┌─────────────────────────┐
│ Allena epoch 86-200     │
│ → log si aggiungono     │
│ → grafici continuano    │
└─────────────────────────┘
```

---

## 🔑 Punti Chiave

1. **ID Univoco**: Ogni run WandB ha un ID unico (`run.id`)
2. **Salvato nel Checkpoint**: L'ID viene salvato nel file `.pth`
3. **Ricaricato al Resume**: Il codice legge l'ID dal checkpoint
4. **Passato a WandB**: `wandb.init(id=...)` riconnette alla run originale
5. **Resume Allow**: `resume="allow"` dice a WandB di continuare

---

## ❓ FAQ

**Q: Cosa succede se perdo il checkpoint?**  
A: Perdi l'ID della run. WandB crea una nuova run separata.

**Q: Posso cambiare il run_id manualmente?**  
A: Sì, ma collegherai a una run diversa. Meglio non farlo.

**Q: E se il run_id non è nel checkpoint (vecchi checkpoint)?**  
A: `wandb_run_id = None` → il codice crea una nuova run.

**Q: I grafici ripartono da 0 o continuano?**  
A: Continuano! WandB usa l'epoch come `step` nei log:
```python
wandb.log({"val_dice": 0.85}, step=86)  # ← Continua da 86
```

**Q: WandB sa quale epoch è l'ultima?**  
A: No, il TUO codice lo gestisce con `start_epoch`. WandB registra solo ciò che gli mandi.

---

## 🧪 Test Pratico

Prova questo per capire:

```bash
# 1. Lancia training
sbatch train_frozen.sbatch

# 2. Dopo alcune epoche, annulla il job
scancel <job_id>

# 3. Controlla il checkpoint
python check_checkpoint.py --pretrained --freeze
# Vedrai: wandb_run_id: abc123xyz

# 4. Fai resume
sbatch resume_frozen.sbatch

# 5. Su WandB vedrai che continua la STESSA pagina/run!
```

---

## 📌 Codice Rilevante

### Salvataggio (utils_2.py)
```python
if wandb_run:
    checkpoint_data['wandb_run_id'] = wandb_run.id  # ← Salva ID
```

### Caricamento (prova_script_rapido.py)
```python
wandb_run_id = checkpoint.get('wandb_run_id', None)  # ← Legge ID
```

### Riconnessione (prova_script_rapido.py)
```python
if wandb_run_id:
    run = wandb.init(id=wandb_run_id, resume="allow")  # ← Riconnette
```

---

## 🎉 Conclusione

WandB sa cosa riprendere grazie a:
1. **Run ID salvato nel checkpoint**
2. **Resume durante init** con quell'ID
3. **Start epoch gestito dal tuo codice**

È come un "bookmark" che dice a WandB: "Continua da questa pagina, non crearne una nuova!" 📖
