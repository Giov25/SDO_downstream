#!/bin/bash
#SBATCH --job-name=Seg_Finetune
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# 🔥 ESPERIMENTO 2: FINE-TUNING
# Carica i pesi pretrained dal task di ricostruzione MAE e allena TUTTO
# Sia encoder che decoder vengono allenati

echo "🚀 Starting FINE-TUNING experiment..."

conda run -n SDOenv python prova_script_rapido.py \
  --mode train \
  --load_pretrained \
  --model MAE_Seg_Deformer \
  --batch_size 3 \
  --epochs 200 \
  --lr 1e-4

echo "✅ Fine-tuning training completed!"
