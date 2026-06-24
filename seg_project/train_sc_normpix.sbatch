#!/bin/bash
#SBATCH --job-name=Seg_Scratch_NormPix
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
echo "[$(date)] Job $SLURM_JOB_ID — MODE=$MODE — LATEST exists: $([ -f "$LATEST" ] && echo yes || echo no)"

conda run -n SDOenv python -u prova_script_rapido.py \
    --mode $MODE \
    $CKPT_ARG \
    --model MAE_Seg_DeformerV3 \
    --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p8_normpix/best_model.pth \
    --patch_size 8 \
    --model_path /home/gpatane/checkpoints/normpix_p8/ \
    --batch_size 2 \
    --epochs 150 \
    --lr 5e-4 \
    --loss TwerskyLoss