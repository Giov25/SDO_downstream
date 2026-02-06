# Guida: Riprendere il Training dopo 24 ore

## Problema
Gli esperimenti sul cluster non possono durare più di 24 ore, ma il training richiede più tempo. Questa guida spiega come riprendere il training continuando a loggare su WandB nella stessa run.

## Come Funziona

Il sistema ora salva automaticamente nei checkpoint:
- **Epoca corrente**
- **Stato del modello** 
- **Stato dell'optimizer**
- **Stato dello scheduler**
- **Miglior Dice score**
- **WandB Run ID** (per continuare la stessa run)

## Uso

### 1. Primo Training (fino a 24 ore)

```bash
conda run -n SDOenv python prova_script_rapido.py \
  --mode train \
  --load_pretrained \
  --model MAE_Seg_Deformer \
  --batch_size 3 \
  --epochs 200
```

Questo:
- Crea una nuova run su WandB
- Salva checkpoint automaticamente quando trova un miglior modello
- Il checkpoint include il run ID di WandB

### 2. Riprendere il Training (dopo 24 ore)

**IMPORTANTE**: Cambia solo `--mode train` in `--mode resume`

```bash
conda run -n SDOenv python prova_script_rapido.py \
  --mode resume \
  --load_pretrained \
  --model MAE_Seg_Deformer \
  --batch_size 3 \
  --epochs 200
```

Questo:
- Carica il checkpoint salvato
- **Riprende la stessa run su WandB** (i grafici continuano)
- Riparte dall'epoca dove si era interrotto
- Mantiene lo stato dell'optimizer e scheduler
- Conserva il miglior Dice score precedente

### 3. Script SBATCH

Modifica il tuo [train.sbatch](train.sbatch):

**Per il primo training:**
```bash
#!/bin/bash
#SBATCH --job-name=Seg_train
#SBATCH --time=23:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

conda run -n SDOenv python prova_script_rapido.py \
  --mode train \
  --load_pretrained \
  --model MAE_Seg_Deformer \
  --batch_size 3 \
  --epochs 200
```

**Per riprendere:**
```bash
#!/bin/bash
#SBATCH --job-name=Seg_resume
#SBATCH --time=23:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

conda run -n SDOenv python prova_script_rapido.py \
  --mode resume \
  --load_pretrained \
  --model MAE_Seg_Deformer \
  --batch_size 3 \
  --epochs 200
```

## Percorsi dei Checkpoint

I checkpoint vengono salvati in base alla configurazione:

- **Training da pretrained**: `/home/gpatane/checkpoints/seg_project/checkpoints/Finetuning_MAE_Seg_Deformer.pth`
- **Training from scratch**: `/home/gpatane/checkpoints/seg_project/checkpoints/Scratch_MAE_Seg_Deformer.pth`
- **Training con encoder frozen**: `/home/gpatane/checkpoints/seg_project/checkpoints/Finetuning_vero_MAE_Seg_Deformer.pth`

## Vantaggi

✅ **Grafici continui su WandB**: I grafici di loss, dice score, learning rate continuano senza interruzioni  
✅ **Nessuna perdita di progresso**: Riprendi esattamente dove ti eri fermato  
✅ **Stesso Run ID**: Tutto rimane organizzato in una singola run su WandB  
✅ **Stato completo**: Optimizer e scheduler mantengono il loro stato  

## Esempio Pratico

**Giorno 1 (23h):**
```bash
sbatch train.sbatch  # mode=train, epochs=200
# Training raggiunge epoch 85 prima di essere interrotto
```

**Giorno 2 (23h):**
```bash
sbatch resume.sbatch  # mode=resume, epochs=200
# Riprende da epoch 86, stessa run WandB
# Training raggiunge epoch 170
```

**Giorno 3 (rimangono ~30 epoche):**
```bash
sbatch resume.sbatch  # mode=resume, epochs=200
# Riprende da epoch 171, stessa run WandB
# Training completa epoch 200
```

## Verifica

Per verificare che funzioni:
1. Controlla il log all'avvio: dovrebbe stampare "Resuming from epoch X"
2. Su WandB: i grafici dovrebbero continuare senza salti
3. L'URL della run WandB dovrebbe essere lo stesso

## Note

- Il sistema salva solo il **best model**, quindi assicurati di fare progressi
- Se il checkpoint non esiste e usi `--mode resume`, il training ripartirà da zero
- Puoi modificare `--epochs` quando fai resume se vuoi trainare più a lungo
