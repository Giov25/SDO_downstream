#!/bin/bash
# Lancia N job in catena per il pre-training norm_pix_loss=True.
# Uso: bash submit_chain_normpix.sh [N]   (default: 10)

N=${1:-10}
SBATCH_FILE="$(dirname "$0")/train_sc_normpix.sbatch"

echo "Submitting chain of $N jobs from $SBATCH_FILE"

JID=$(sbatch --parsable "$SBATCH_FILE")
echo "  Job 1: $JID"

for i in $(seq 2 $N); do
    JID=$(sbatch --parsable --dependency=afterany:$JID "$SBATCH_FILE")
    echo "  Job $i: $JID"
done

echo "Chain submitted. Last job ID: $JID"
echo "Monitor with: squeue -u $USER"
