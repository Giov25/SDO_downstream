#!/bin/bash
#SBATCH --job-name=Seg_Frozen_NormPix
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Encoder FROZEN con pesi MAE pretrained con norm_pix_loss=True

CKPT=/home/gpatane/checkpoints/normpix_p8/Frozen_MAE_Seg_DeformerV3.pth
[ -f "$CKPT" ] && MODE=resume || MODE=train

conda run -n SDOenv python prova_script_rapido.py \
  --mode $MODE \
  --load_pretrained \
  --freeze_encoder \
  --model MAE_Seg_DeformerV3 \
  --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p8_normpix/best_model.pth \
  --patch_size 8 \
  --model_path /home/gpatane/checkpoints/normpix_p8/ \
  --batch_size 2 \
  --epochs 150 \
  --lr 5e-4 \
  --loss TwerskyLoss
