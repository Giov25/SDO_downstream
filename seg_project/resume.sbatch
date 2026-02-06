#!/bin/bash
#SBATCH --job-name=Seg_resume
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Script per RIPRENDERE il training dopo un'interruzione
# Questo continuerà la stessa run su WandB

conda run -n SDOenv python prova_script_rapido.py \
  --mode resume \
  --load_pretrained \
  --model MAE_Seg_Deformer \
  --batch_size 3 \
  --epochs 200 \
  --lr 1e-4
