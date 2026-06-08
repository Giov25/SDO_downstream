#!/bin/bash
#SBATCH --job-name=Test_Finetune_NormPix
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Test/forecast fine-tuning end-to-end con pesi MAE normpix
# Richiede che train_finetuning_normpix.sbatch sia gia' stato eseguito

conda run -n SDOenv python prova_script_rapido.py \
  --mode test \
  --model MAE_Seg_Deformer \
  --checkpoint_path /home/gpatane/checkpoints/normpix/Finetuning_MAE_Seg_Deformer.pth \
  --batch_size 1 \
  --device cuda
