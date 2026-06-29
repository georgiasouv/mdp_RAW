#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_rod_homo_T_pcgrad
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs_rod/eval_homo_T_pcgrad_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs_rod/eval_homo_T_pcgrad_%j.err
# ============================================================
#  ROD eval homo_T_pcgrad  ->  vs fcos, fasterrcnn, detr, retinanet
# ============================================================
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate rawdet

export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_PROJECT=mdp-raw-preprocessing
export WANDB_RUN_GROUP=eval__rod__homo_T_pcgrad
export WANDB_MODE=offline

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROD_ROOT=/scratch/INC1526354/rod

cd /networkhome/WMGDS/souval_g/raw-mdp
export PYTHONPATH="$(pwd):${PYTHONPATH}"

PREP=runs_rod/rod__homo_T_pcgrad/preprocessor.pth
mkdir -p results_rod/eval_pcgrad

if [ ! -f "$PREP" ]; then
    echo "ERROR: checkpoint $PREP not found — did training for homo_T_pcgrad finish?"
    exit 1
fi

echo "=== Evaluating ROD homo_T_pcgrad against all detectors ==="
for DET in fcos fasterrcnn detr retinanet; do
    echo "--- ROD homo_T_pcgrad vs $DET ---"
    python -m core.evaluate \
        --prep "$PREP" \
        --detector "$DET" \
        --dataset rod \
        --num-classes 5 \
        --out results_rod/eval_pcgrad/homo_T_pcgrad__${DET}.json \
        --wandb \
        --wandb-name rod__homo_T_pcgrad__eval_${DET}
done

echo "=== syncing W&B offline runs ==="
wandb sync --sync-all 2>/dev/null
echo "=== ROD eval homo_T_pcgrad finished ==="
