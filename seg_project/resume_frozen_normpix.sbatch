#!/bin/bash
#SBATCH --job-name=Resume_Frozen_NormPix
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

conda run -n SDOenv python prova_script_rapido.py \
  --mode resume \
  --load_pretrained \
  --freeze_encoder \
  --model MAE_Seg_Deformer \
  --model_path /home/gpatane/checkpoints/normpix/ \
  --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p16_normpix/best_model.pth \
  --batch_size 3 \
  --epochs 200 \
  --lr 1e-4
