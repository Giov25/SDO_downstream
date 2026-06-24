#!/bin/bash
#SBATCH --job-name=Resume_Scratch_NormPix
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

LATEST=/home/gpatane/checkpoints/normpix_p8/Scratch_MAE_Seg_DeformerV3_latest.pth

if [ ! -f "$LATEST" ]; then
    echo "Errore: checkpoint non trovato in $LATEST"
    exit 1
fi

echo "🔄 Resuming Scratch NormPix experiment from $LATEST..."

conda run -n SDOenv python prova_script_rapido.py \
  --mode resume \
  --checkpoint_path $LATEST \
  --model MAE_Seg_DeformerV3 \
  --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p8_normpix/best_model.pth \
  --patch_size 8 \
  --model_path /home/gpatane/checkpoints/normpix_p8/ \
  --batch_size 2 \
  --epochs 150 \
  --lr 2e-4 \
  --loss TwerskyLoss

echo "✅ Resume completed!"
