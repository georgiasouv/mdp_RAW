#!/bin/bash
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --job-name=pair_12_pcgrad
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs/pair_12_pcgrad_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs/pair_12_pcgrad_%j.err
# ============================================================
#  pair_12_pcgrad  (regime=pair_12, combiner=pcgrad)  FRESH 0->80
# ============================================================
# -- Environment --------------------------------------------------
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate rawdet
# -- W&B (headless node: log offline, sync after) -----------------
export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_PROJECT=mdp-raw-preprocessing
export WANDB_MODE=offline
# -- Memory + data path -------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PASCALRAW_ROOT=/scratch/INC1526354/pascalraw
# -- Training (FRESH -- no --resume, starts at epoch 0) -----------
echo "=== Starting pair_12_pcgrad ==="
cd /networkhome/WMGDS/souval_g/raw-mdp
export PYTHONPATH="$(pwd):${PYTHONPATH}"
OUT=runs/pascalraw__pair_12_pcgrad
python -m core.train \
    --regime pair_12 \
    --dataset pascalraw \
    --num-classes 3 \
    --combine pcgrad \
    --epochs 80 \
    --bs 4 \
    --val-every 1 \
    --val-max-batches 50 \
    --out "$OUT" \
    --wandb \
    --wandb-entity georgiasouval-university-of-warwick
# -- Sync the offline W&B run -------------------------------------
echo "=== syncing W&B offline run ==="
wandb sync "$OUT"/wandb/offline-run-* 2>/dev/null
echo "=== pair_12_pcgrad finished ==="
