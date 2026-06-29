#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_rod_solo_T_normgrad
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs_rod/eval_solo_T_normgrad_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs_rod/eval_solo_T_normgrad_%j.err
# ============================================================
#  ROD eval solo_T_normgrad  ->  vs fcos, fasterrcnn, detr, retinanet
# ============================================================
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate rawdet

export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_PROJECT=mdp-raw-preprocessing
export WANDB_RUN_GROUP=eval__rod__solo_T_normgrad
export WANDB_MODE=offline

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROD_ROOT=/scratch/INC1526354/rod

cd /networkhome/WMGDS/souval_g/raw-mdp
export PYTHONPATH="$(pwd):${PYTHONPATH}"

PREP=runs_rod/rod__solo_T_normgrad/preprocessor.pth
mkdir -p results_rod/eval_normgrad

if [ ! -f "$PREP" ]; then
    echo "ERROR: checkpoint $PREP not found — did training for solo_T_normgrad finish?"
    exit 1
fi

echo "=== Evaluating ROD solo_T_normgrad against all detectors ==="
for DET in fcos fasterrcnn detr retinanet; do
    echo "--- ROD solo_T_normgrad vs $DET ---"
    python -m core.evaluate \
        --prep "$PREP" \
        --detector "$DET" \
        --dataset rod \
        --num-classes 5 \
        --out results_rod/eval_normgrad/solo_T_normgrad__${DET}.json \
        --wandb \
        --wandb-name rod__solo_T_normgrad__eval_${DET}
done

echo "=== syncing W&B offline runs ==="
wandb sync --sync-all 2>/dev/null
echo "=== ROD eval solo_T_normgrad finished ==="
