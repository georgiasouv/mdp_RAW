#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=rod_homo_T_mgda
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs_rod/homo_T_mgda_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs_rod/homo_T_mgda_%j.err
# ============================================================
#  ROD homo_T_mgda  (regime=homo_T, combiner=mgda)
# ============================================================
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate rawdet

export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_PROJECT=mdp-raw-preprocessing
export WANDB_MODE=offline

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROD_ROOT=/scratch/INC1526354/rod

echo "=== Starting ROD homo_T_mgda ==="
cd /networkhome/WMGDS/souval_g/raw-mdp
export PYTHONPATH="$(pwd):${PYTHONPATH}"

OUT=runs_rod/rod__homo_T_mgda

python -m core.train \
    --regime homo_T \
    --dataset rod \
    --num-classes 5 \
    --combine mgda \
    --resume "$OUT/preprocessor.pth" \
    --epochs 80 \
    --bs 4 \
    --val-every 1 \
    --val-max-batches 50 \
    --out "$OUT" \
    --wandb \
    --wandb-entity georgiasouval-university-of-warwick

echo "=== syncing W&B offline run ==="
wandb sync "$OUT"/wandb/offline-run-* 2>/dev/null
echo "=== ROD homo_T_mgda finished ==="
