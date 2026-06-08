#!/bin/bash
#SBATCH --job-name=Seg_encoder_freeze_encoder
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

BEST_CKPT=/home/gpatane/checkpoints/Scratch_MAE_2Channel.pth
LATEST_CKPT=/home/gpatane/checkpoints/Scratch_MAE_2Channel_latest.pth

if [ -f "$LATEST_CKPT" ]; then
    MODE=resume
    CHECKPOINT=$LATEST_CKPT
else
    MODE=train
    CHECKPOINT=$BEST_CKPT
fi

echo "🚀 Starting experiment — mode: $MODE — checkpoint: $CHECKPOINT"

conda run -n SDOenv python -u prova_script_rapido.py \
    --mode $MODE \
    --model MAE_2Channel \
    --batch_size 8 \
    --freeze_encoder \
    --epochs 100 \
    --lr 1e-4 \
    --checkpoint_path $CHECKPOINT \
    --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_log1p/best_model.pth

echo "✅ Done"