#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_pair_12_sum
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs/eval_pair_12_sum_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs/eval_pair_12_sum_%j.err
# ============================================================
#  eval pair_12_sum  ->  vs fcos, fasterrcnn, detr, retinanet
# ============================================================

# -- Environment --------------------------------------------------
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate rawdet

# -- W&B: group all 4 detector evals under ONE group per regime ---
export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_PROJECT=mdp-raw-preprocessing
export WANDB_RUN_GROUP=eval__pair_12_sum
export WANDB_MODE=offline

# -- Memory + data path -------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PASCALRAW_ROOT=/scratch/INC1526354/pascalraw

cd /networkhome/WMGDS/souval_g/raw-mdp
export PYTHONPATH="$(pwd):${PYTHONPATH}"

PREP=runs/pascalraw__pair_12_sum/preprocessor.pth
mkdir -p results/eval_sum

# Guard: skip cleanly if the checkpoint doesn't exist yet
if [ ! -f "$PREP" ]; then
    echo "ERROR: checkpoint $PREP not found — did training for pair_12_sum finish?"
    exit 1
fi

echo "=== Evaluating pair_12_sum against all detectors ==="
for DET in fcos fasterrcnn detr retinanet; do
    echo "--- pair_12_sum vs $DET ---"
    python -m core.evaluate \
        --prep "$PREP" \
        --detector "$DET" \
        --dataset pascalraw \
        --num-classes 3 \
        --out results/eval_sum/pair_12_sum__${DET}.json \
        --wandb \
        --wandb-name pair_12_sum__eval_${DET}
done

# -- Sync all offline W&B runs from this job ----------------------
echo "=== syncing W&B offline runs ==="
wandb sync --sync-all 2>/dev/null

echo "=== eval pair_12_sum finished ==="
