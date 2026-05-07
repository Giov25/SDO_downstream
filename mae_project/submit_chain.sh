#!/bin/bash
# Lancia N job in catena: ognuno aspetta che il precedente finisca
# (sia per completamento normale che per time limit SLURM).
# Uso: bash submit_chain.sh [N]   (default: 10)

N=${1:-3}
SBATCH_FILE="$(dirname "$0")/train.sbatch"

echo "Submitting chain of $N jobs from $SBATCH_FILE"

JID=$(sbatch --parsable "$SBATCH_FILE")
echo "  Job 1: $JID"

for i in $(seq 2 $N); do
    JID=$(sbatch --parsable --dependency=afterany:$JID "$SBATCH_FILE")
    echo "  Job $i: $JID"
done

echo "Chain submitted. Last job ID: $JID"
echo "Monitor with: squeue -u $USER"
