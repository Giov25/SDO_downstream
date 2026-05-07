#!/bin/bash
#SBATCH --job-name=rechunk_zarr
#SBATCH --partition=cpu
#SBATCH --qos=train
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=rechunk_zarr-%j.out
#SBATCH --error=rechunk_zarr-%j.err

cd /home/gpatane/SDO_downstream
conda run -n SDOenv python rechunk_zarr.py --workers 32
