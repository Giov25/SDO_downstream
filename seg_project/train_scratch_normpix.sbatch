#!/bin/bash
#SBATCH --job-name=Seg_Scratch
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Encoder FROZEN con pesi MAE pretrained con norm_pix_loss=True

LATEST=/home/gpatane/checkpoints/normpix_p8/Scratch_MAE_Seg_DeformerV3_latest.pth
if [ -f "$LATEST" ]; then
    MODE=resume
    CKPT_ARG="--checkpoint_path $LATEST"
else
    MODE=train
    CKPT_ARG=""
fi

conda run -n SDOenv python -u prova_script_rapido.py \
  --mode $MODE \
  $CKPT_ARG \
  --model MAE_Seg_DeformerV3 \
  --patch_size 8 \
  --model_path /home/gpatane/checkpoints/normpix_p8/ \
  --batch_size 2 \
  --epochs 150 \
  --lr 2e-4 \
  --loss TwerskyLoss