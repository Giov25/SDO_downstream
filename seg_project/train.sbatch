#!/bin/bash
#SBATCH --job-name=prova_train
#SBATCH --partition=gpu
#SBATCH --qos=debug
#SBATCH --gres=shard:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ❄️  ESPERIMENTO 1: ENCODER FROZEN (Feature Extraction)
# Carica i pesi pretrained dal task di ricostruzione MAE e FREEZA l'encoder
# Solo il decoder viene allenato

echo "🚀 Starting experiment..."
#--mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/checkpoint_epoch55.pth \
conda run -n SDOenv WANDB_MODE=disabled python prova_script_rapido.py \
  --mode train \
  --model MAE_2Channel \
  --batch_size 8 \
  --epochs 20 \
  --lr 1e-4
echo "Modello originale in segmentazione"

#python prova_script_rapido.py --mode train --load_pretrained --freeze_encoder --model MAE_2Channel --batch_size 3 --epochs 200 --lr 1e-4
