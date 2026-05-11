#!/bin/bash
# Lancia una catena di job per il downstream seg con encoder normpix.
#
# Uso:
#   bash submit_chain_seg_normpix.sh frozen [N]      # encoder frozen
#   bash submit_chain_seg_normpix.sh finetuning [N]  # fine-tuning end-to-end
#
# N = numero totale di job nella catena (default: 5)
# Il primo job parte con 'train', i successivi con 'resume'.

MODE=${1:-frozen}
N=${2:-5}
DIR="$(dirname "$0")"

if [ "$MODE" = "frozen" ]; then
    TRAIN_SBATCH="$DIR/train_frozen_normpix.sbatch"
    RESUME_SBATCH="$DIR/resume_frozen_normpix.sbatch"
elif [ "$MODE" = "finetuning" ]; then
    TRAIN_SBATCH="$DIR/train_finetuning_normpix.sbatch"
    RESUME_SBATCH="$DIR/resume_finetuning_normpix.sbatch"
else
    echo "Errore: modalita' non riconosciuta '$MODE'. Usa 'frozen' o 'finetuning'."
    exit 1
fi

echo "Catena di $N job | modalita': $MODE"

JID=$(sbatch --parsable "$TRAIN_SBATCH")
echo "  Job 1 (train): $JID"

for i in $(seq 2 $N); do
    JID=$(sbatch --parsable --dependency=afterany:$JID "$RESUME_SBATCH")
    echo "  Job $i (resume): $JID"
done

echo "Catena sottomessa. Ultimo job ID: $JID"
echo "Monitora con: squeue -u $USER"
