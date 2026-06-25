#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --job-name=solo_T_normgrad
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs/solo_T_normgrad_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs/solo_T_normgrad_%j.err
# ============================================================
#  solo_T_normgrad  (regime=solo_T, combiner=normgrad)
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

# -- Training -----------------------------------------------------
echo "=== Starting solo_T_normgrad ==="
cd /networkhome/WMGDS/souval_g/raw-mdp
export PYTHONPATH="$(pwd):${PYTHONPATH}"

OUT=runs/pascalraw__solo_T_normgrad

python -m core.train \
    --regime solo_T \
    --dataset pascalraw \
    --num-classes 3 \
    --combine normgrad \
    --resume "$OUT/preprocessor.pth" \
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

echo "=== solo_T_normgrad finished ==="
