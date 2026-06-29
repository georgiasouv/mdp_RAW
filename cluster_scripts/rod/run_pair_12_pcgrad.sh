#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=rod_pair_12_pcgrad
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs_rod/pair_12_pcgrad_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs_rod/pair_12_pcgrad_%j.err
# ============================================================
#  ROD pair_12_pcgrad  (regime=pair_12, combiner=pcgrad)
# ============================================================
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate rawdet

export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_PROJECT=mdp-raw-preprocessing
export WANDB_MODE=offline

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROD_ROOT=/scratch/INC1526354/rod

echo "=== Starting ROD pair_12_pcgrad ==="
cd /networkhome/WMGDS/souval_g/raw-mdp
export PYTHONPATH="$(pwd):${PYTHONPATH}"

OUT=runs_rod/rod__pair_12_pcgrad

python -m core.train \
    --regime pair_12 \
    --dataset rod \
    --num-classes 5 \
    --combine pcgrad \
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
echo "=== ROD pair_12_pcgrad finished ==="
