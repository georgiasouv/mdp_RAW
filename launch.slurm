#!/bin/bash
#SBATCH --job-name=raw_ens
#SBATCH --array=0-49%16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out

# 10 regimes x 5 datasets = 50 cells. %16 = at most 16 run concurrently.
# Edit the concurrency cap to match your queue allowance.

set -euo pipefail

REGIMES=(solo_T solo_1 solo_2 pair_T1 pair_T2 pair_12 triple homo_T homo_1 homo_2)
DATASETS=(pascalraw lod nod rod aodraw)

# REAL per-dataset class counts -- EDIT these.
declare -A NC=( [pascalraw]=3 [lod]=8 [nod]=10 [rod]=6 [aodraw]=12 )

NR=${#REGIMES[@]}
ri=$(( SLURM_ARRAY_TASK_ID % NR ))
di=$(( SLURM_ARRAY_TASK_ID / NR ))
R=${REGIMES[$ri]}
D=${DATASETS[$di]}

python train.py \
  --regime "$R" \
  --dataset "$D" \
  --num-classes "${NC[$D]}" \
  --out "runs/${D}__${R}" \
  --ckpt-dir ./detectors \
  --epochs 20 --bs 8 --lr 1e-3 --size 512 \
  --log-conflict-every 50
