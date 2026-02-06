#!/bin/bash
#SBATCH --job-name=Seg_train
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# 🧪 Script Generico per Training
# Per esperimenti specifici, usa invece:
#   - train_frozen.sbatch      (encoder frozen)
#   - train_finetuning.sbatch  (fine-tuning completo)  
#   - train_scratch.sbatch     (training da zero)

# Questo script è mantenuto per retrocompatibilità
# Di default fa FINE-TUNING

echo "🚀 Starting training..."

conda run -n SDOenv python prova_script_rapido.py \
  --mode train \
  --load_pretrained \
  --model MAE_Seg_Deformer \
  --batch_size 3 \
  --epochs 200 \
  --lr 1e-4

echo "✅ Training completed!"
