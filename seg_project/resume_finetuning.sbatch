#!/bin/bash
#SBATCH --job-name=Resume_Finetune
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# 🔥 RESUME: FINE-TUNING
# Riprende il training dell'esperimento di fine-tuning

echo "🔄 Resuming FINE-TUNING experiment..."

conda run -n SDOenv python prova_script_rapido.py \
  --mode resume \
  --load_pretrained \
  --model MAE_Seg_Deformer \
  --batch_size 3 \
  --epochs 200 \
  --lr 1e-4

echo "✅ Resume completed!"
