#!/bin/bash
#SBATCH --job-name=Seg_Finetune_NormPix
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Fine-tuning end-to-end con pesi MAE pretrained con norm_pix_loss=True

conda run -n SDOenv python prova_script_rapido.py \
  --mode train \
  --load_pretrained \
  --model MAE_Seg_Deformer \
  --mae_checkpoint /home/gpatane/SDO_downstream/mae_project/checkpoints/ch9_1024_p8_normpix/best_model.pth \
  --patch_size 8 \
  --model_path /home/gpatane/checkpoints/normpix_p8/ \
  --batch_size 2 \
  --epochs 200 \
  --lr 5e-5
