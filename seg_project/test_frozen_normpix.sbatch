#!/bin/bash
#SBATCH --job-name=Test_Frozen_NormPix
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Test/forecast encoder FROZEN con pesi MAE normpix
# Richiede che train_frozen_normpix.sbatch sia gia' stato eseguito

conda run -n SDOenv python prova_script_rapido.py \
  --mode test \
  --model MAE_Seg_Deformer \
  --checkpoint_path /home/gpatane/checkpoints/normpix/Frozen_MAE_Seg_Deformer.pth \
  --batch_size 1 \
  --device cuda
