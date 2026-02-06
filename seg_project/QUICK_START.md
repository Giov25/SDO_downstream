# 🎯 Quick Reference - 3 Esperimenti

## Lancio Rapido

```bash
# Esperimento 1: Frozen Encoder ❄️
sbatch train_frozen.sbatch

# Esperimento 2: Fine-tuning 🔥  
sbatch train_finetuning.sbatch

# Esperimento 3: From Scratch 🆕
sbatch train_scratch.sbatch
```

## Resume (dopo 24h)

```bash
sbatch resume_frozen.sbatch      # Resume exp 1
sbatch resume_finetuning.sbatch  # Resume exp 2
sbatch resume_scratch.sbatch     # Resume exp 3
```

## Checkpoint Prodotti

- `Frozen_MAE_Seg_Deformer.pth` - Exp 1
- `Finetuning_MAE_Seg_Deformer.pth` - Exp 2
- `Scratch_MAE_Seg_Deformer.pth` - Exp 3

## Verifica Stato

```bash
python check_checkpoint.py --pretrained --freeze  # Exp 1
python check_checkpoint.py --pretrained           # Exp 2
python check_checkpoint.py                        # Exp 3
```

## Differenze

| Exp | Pretrained | Encoder Frozen | Best For |
|-----|-----------|----------------|----------|
| 1   | ✅        | ✅             | Pochi dati, veloce |
| 2   | ✅        | ❌             | Best performance |
| 3   | ❌        | ❌             | Baseline comparison |

Vedi [EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md) per dettagli completi.
