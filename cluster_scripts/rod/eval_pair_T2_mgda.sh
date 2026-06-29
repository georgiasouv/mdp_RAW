#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_rod_pair_T2_mgda
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs_rod/eval_pair_T2_mgda_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs_rod/eval_pair_T2_mgda_%j.err
# ============================================================
#  ROD eval pair_T2_mgda  ->  vs fcos, fasterrcnn, detr, retinanet
# ============================================================
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate rawdet

export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_PROJECT=mdp-raw-preprocessing
export WANDB_RUN_GROUP=eval__rod__pair_T2_mgda
export WANDB_MODE=offline

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROD_ROOT=/scratch/INC1526354/rod

cd /networkhome/WMGDS/souval_g/raw-mdp
export PYTHONPATH="$(pwd):${PYTHONPATH}"

PREP=runs_rod/rod__pair_T2_mgda/preprocessor.pth
mkdir -p results_rod/eval_mgda

if [ ! -f "$PREP" ]; then
    echo "ERROR: checkpoint $PREP not found — did training for pair_T2_mgda finish?"
    exit 1
fi

echo "=== Evaluating ROD pair_T2_mgda against all detectors ==="
for DET in fcos fasterrcnn detr retinanet; do
    echo "--- ROD pair_T2_mgda vs $DET ---"
    python -m core.evaluate \
        --prep "$PREP" \
        --detector "$DET" \
        --dataset rod \
        --num-classes 5 \
        --out results_rod/eval_mgda/pair_T2_mgda__${DET}.json \
        --wandb \
        --wandb-name rod__pair_T2_mgda__eval_${DET}
done

echo "=== syncing W&B offline runs ==="
wandb sync --sync-all 2>/dev/null
echo "=== ROD eval pair_T2_mgda finished ==="
