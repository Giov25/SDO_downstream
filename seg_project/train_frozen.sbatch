#!/bin/bash
#SBATCH --job-name=Seg_Frozen
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ❄️  ESPERIMENTO 1: ENCODER FROZEN (Feature Extraction)
# Carica i pesi pretrained dal task di ricostruzione MAE e FREEZA l'encoder
# Solo il decoder viene allenato

echo "🚀 Starting FROZEN ENCODER experiment..."

conda run -n SDOenv python prova_script_rapido.py \
  --mode train \
  --load_pretrained \
  --freeze_encoder \
  --model MAE_Seg_Deformer \
  --batch_size 3 \
  --epochs 200 \
  --lr 1e-4

echo "✅ Frozen encoder training completed!"

#python prova_script_rapido.py --mode train --load_pretrained --freeze_encoder --model MAE_Seg_Deformer --batch_size 3 --epochs 200 --lr 1e-4