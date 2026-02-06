#!/bin/bash
#SBATCH --job-name=Resume_Scratch
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# 🆕 RESUME: FROM SCRATCH
# Riprende il training dell'esperimento from scratch

echo "🔄 Resuming FROM SCRATCH experiment..."

conda run -n SDOenv python prova_script_rapido.py \
  --mode resume \
  --model MAE_Seg_Deformer \
  --batch_size 3 \
  --epochs 200 \
  --lr 1e-4

echo "✅ Resume completed!"
