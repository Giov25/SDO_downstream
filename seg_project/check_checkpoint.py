#!/usr/bin/env python3
"""
Utility per controllare lo stato di un checkpoint di training.
Utile per capire da che epoca riprendere e quale run WandB usare.
"""

import torch
import argparse
import os

def check_checkpoint(checkpoint_path):
    """Mostra le informazioni contenute in un checkpoint."""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint non trovato: {checkpoint_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"📦 Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"{'='*60}\n")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Informazioni di base
        print("📊 Informazioni Training:")
        print(f"  Epoca: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A'):.6f}" if 'train_loss' in checkpoint else "  Train Loss: N/A")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}" if 'val_loss' in checkpoint else "  Val Loss: N/A")
        print(f"  Val Dice (no bg): {checkpoint.get('val_dice', 'N/A'):.4f}" if 'val_dice' in checkpoint else "  Val Dice: N/A")
        print(f"  Val Dice (con bg): {checkpoint.get('val_dice_T', 'N/A'):.4f}" if 'val_dice_T' in checkpoint else "  Val Dice (con bg): N/A")
        
        # Informazioni WandB
        print(f"\n🔗 WandB Info:")
        if 'wandb_run_id' in checkpoint:
            print(f"  Run ID: {checkpoint['wandb_run_id']}")
            print(f"  Project: {checkpoint.get('wandb_project', 'N/A')}")
            print(f"  Entity: {checkpoint.get('wandb_entity', 'N/A')}")
            
            # Costruisce l'URL
            if 'wandb_entity' in checkpoint and 'wandb_project' in checkpoint:
                url = f"https://wandb.ai/{checkpoint['wandb_entity']}/{checkpoint['wandb_project']}/runs/{checkpoint['wandb_run_id']}"
                print(f"  URL: {url}")
        else:
            print("  ⚠️  Nessuna informazione WandB salvata (vecchio checkpoint)")
        
        # Stati salvati
        print(f"\n💾 Stati Salvati:")
        print(f"  Model State: {'✅' if 'model_state_dict' in checkpoint else '❌'}")
        print(f"  Optimizer State: {'✅' if 'optimizer_state_dict' in checkpoint else '❌'}")
        print(f"  Scheduler State: {'✅' if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None else '❌'}")
        
        # Dimensione file
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"\n📁 Dimensione File: {file_size_mb:.2f} MB")
        
        print(f"\n{'='*60}")
        print("✅ Checkpoint valido per resume")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Errore nel caricamento del checkpoint: {e}")

def main():
    parser = argparse.ArgumentParser(description="Controlla lo stato di un checkpoint")
    parser.add_argument('checkpoint', type=str, nargs='?', 
                      help='Path al checkpoint da controllare')
    parser.add_argument('--model', type=str, default='MAE_Seg_Deformer',
                      help='Nome del modello (default: MAE_Seg_Deformer)')
    parser.add_argument('--pretrained', action='store_true',
                      help='Controlla checkpoint finetuning')
    parser.add_argument('--freeze', action='store_true',
                      help='Controlla checkpoint frozen encoder (usare con --pretrained)')
    parser.add_argument('--base-path', type=str, 
                      default='/home/gpatane/checkpoints/seg_project/checkpoints/',
                      help='Path base per i checkpoint')
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # Usa il path fornito
        checkpoint_path = args.checkpoint
    else:
        # Costruisce il path automaticamente
        if args.pretrained and args.freeze:
            checkpoint_name = f"Frozen_{args.model}.pth"
        elif args.pretrained:
            checkpoint_name = f"Finetuning_{args.model}.pth"
        else:
            checkpoint_name = f"Scratch_{args.model}.pth"
        
        checkpoint_path = os.path.join(args.base_path, checkpoint_name)
    
    check_checkpoint(checkpoint_path)

if __name__ == "__main__":
    main()
