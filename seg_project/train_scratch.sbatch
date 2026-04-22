#!/bin/bash
#SBATCH --job-name=Seg_Scratch
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# 🆕 ESPERIMENTO 3: FROM SCRATCH
# NON carica pesi pretrained, allena tutto da zero
# Sia encoder che decoder vengono inizializzati random e allenati

echo "🚀 Starting FROM SCRATCH experiment..."

conda run -n SDOenv python prova_script_rapido.py \
  --mode train \
  --model MAE_Seg_Deformer \
  --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/checkpoint_epoch55.pth \
  --batch_size 3 \
  --epochs 200 \
  --lr 1e-4

echo "✅ Scratch training completed!"

#python prova_script_rapido.py --mode train --model MAE_Seg_Deformer --batch_size 3 --epochs 200 --lr 1e-4
