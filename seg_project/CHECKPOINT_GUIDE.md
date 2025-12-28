# Guida: Come Caricare i Checkpoint

## 🔧 Formati di Checkpoint Supportati

### Formato Nuovo (Raccomandato)
Salva tutto: model, optimizer, scheduler, metriche
```python
checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_dice': val_metric,
}
torch.save(checkpoint, 'model.pth')
```

### Formato Vecchio (Solo Pesi)
Salva solo i pesi del modello
```python
torch.save(model.state_dict(), 'model.pth')
```

## 📥 Come Caricare

### Scenario 1: Continuare Training (Resume)
```python
# Il codice nello script gestisce automaticamente entrambi i formati
# Se trova 'model_state_dict' → Nuovo formato (carica tutto)
# Altrimenti → Vecchio formato (carica solo pesi)

checkpoint_path = "deep_decoder_model_improved.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        # Nuovo formato
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        # Vecchio formato
        model.load_state_dict(checkpoint)
```

### Scenario 2: Solo Inference (Evaluation)
```python
# Carica solo i pesi del modello per fare predizioni
checkpoint = torch.load('model.pth', map_location=device)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
```

### Scenario 3: Fine-tuning da Checkpoint Pre-addestrato
```python
# Carica pesi pre-addestrati ma riparti con nuovo optimizer
checkpoint = torch.load('pretrained_model.pth', map_location=device)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

# Crea nuovo optimizer (non caricare quello vecchio)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

## ⚠️ Errori Comuni e Soluzioni

### Errore: "weights_only=True incompatibile"
❌ **Sbagliato:**
```python
model.load_state_dict(torch.load(..., weights_only=True))
```

✅ **Corretto:**
```python
checkpoint = torch.load(..., map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
```

### Errore: "Unexpected key in state_dict"
Stai provando a caricare un checkpoint completo direttamente nel modello.

❌ **Sbagliato:**
```python
model.load_state_dict(checkpoint)  # checkpoint contiene optimizer, scheduler, etc.
```

✅ **Corretto:**
```python
model.load_state_dict(checkpoint['model_state_dict'])
```

### Errore: Optimizer non riprende da dove aveva lasciato
Devi caricare anche lo stato dell'optimizer.

❌ **Sbagliato:**
```python
model.load_state_dict(checkpoint['model_state_dict'])
# Optimizer non caricato → riparte da learning rate iniziale
```

✅ **Corretto:**
```python
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

## 🎯 Best Practices

### 1. Salva Checkpoint Regolarmente
```python
# Durante training, salva ogni N epochs
if (epoch + 1) % 5 == 0:
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, f'checkpoint_epoch_{epoch+1}.pth')
```

### 2. Salva Anche il Miglior Modello
```python
if val_dice > best_dice:
    best_dice = val_dice
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'val_dice': val_dice,
    }, 'best_model.pth')
```

### 3. Usa Nomi Descrittivi
```python
# ✅ Buono
'deep_decoder_epoch50_dice0.72.pth'

# ❌ Non buono
'model.pth'
```

### 4. Backup Prima di Sovrascrivere
```python
import shutil
if os.path.exists('model.pth'):
    shutil.copy('model.pth', 'model_backup.pth')
torch.save(checkpoint, 'model.pth')
```

## 🔍 Debug: Ispeziona un Checkpoint

```python
checkpoint = torch.load('model.pth', map_location='cpu')

print("Checkpoint keys:", checkpoint.keys())

if 'model_state_dict' in checkpoint:
    print("\nNuovo formato - Contenuto:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Val Dice: {checkpoint.get('val_dice', 'N/A')}")
    print(f"  Model params: {len(checkpoint['model_state_dict'])} layers")
else:
    print("\nVecchio formato - Solo pesi del modello")
    print(f"  Model params: {len(checkpoint)} layers")
```

## 📝 Template Completo

```python
import os
import torch

# 1. Crea modello, optimizer, scheduler
model = ...
optimizer = ...
scheduler = ...

# 2. Carica checkpoint se esiste
checkpoint_path = "model.pth"
start_epoch = 0

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        # Nuovo formato - carica tutto
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✓ Resumed from epoch {start_epoch}")
    else:
        # Vecchio formato - solo pesi
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded model weights (training from scratch)")
else:
    print("No checkpoint found, starting fresh")

# 3. Training loop
for epoch in range(start_epoch, num_epochs):
    # ... training code ...
    
    # Salva checkpoint
    if should_save:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_dice': val_dice,
        }, checkpoint_path)
```
