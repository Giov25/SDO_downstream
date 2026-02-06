#!/bin/bash
#SBATCH --job-name=Resume_Frozen
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ❄️  RESUME: ENCODER FROZEN
# Riprende il training dell'esperimento con encoder freezato

echo "🔄 Resuming FROZEN ENCODER experiment..."

conda run -n SDOenv python prova_script_rapido.py \
  --mode resume \
  --load_pretrained \
  --freeze_encoder \
  --model MAE_Seg_Deformer \
  --batch_size 3 \
  --epochs 200 \
  --lr 1e-4

echo "✅ Resume completed!"
