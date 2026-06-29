#!/bin/bash
# Generate all 28 ROD training + 28 eval scripts, mirroring the PascalRAW grid.
# Only dataset/num-classes/paths differ from PascalRAW; all SBATCH/env identical.
set -euo pipefail
ROOT=/networkhome/WMGDS/souval_g/raw-mdp
SDIR=$ROOT/cluster_scripts/rod
mkdir -p "$SDIR" "$ROOT/cluster_scripts/logs_rod" "$ROOT/runs_rod"

# regime -> combiner assignments
# Batch 1: headline grid (normgrad on all 10 regimes + sum on 3)
# Batch 2: mgda/pcgrad/cagrad on the 5 sweep regimes
declare -a JOBS=(
  # --- Batch 1: normgrad (10) ---
  "solo_1 normgrad" "solo_2 normgrad" "solo_T normgrad"
  "pair_12 normgrad" "pair_T1 normgrad" "pair_T2 normgrad" "triple normgrad"
  "homo_1 normgrad" "homo_2 normgrad" "homo_T normgrad"
  # --- Batch 1: sum (3) ---
  "pair_12 sum" "pair_T1 sum" "triple sum"
  # --- Batch 2: combiner sweep (15) ---
  "pair_12 mgda" "pair_T1 mgda" "pair_T2 mgda" "triple mgda" "homo_T mgda"
  "pair_12 pcgrad" "pair_T1 pcgrad" "pair_T2 pcgrad" "triple pcgrad" "homo_T pcgrad"
  "pair_12 cagrad" "pair_T1 cagrad" "pair_T2 cagrad" "triple cagrad" "homo_T cagrad"
)

for J in "${JOBS[@]}"; do
  REGIME=$(echo "$J" | cut -d' ' -f1)
  COMB=$(echo "$J" | cut -d' ' -f2)
  TAG=${REGIME}_${COMB}

  # ---------- training script ----------
  cat > "$SDIR/run_${TAG}.sh" << EOF
#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=rod_${TAG}
#SBATCH --output=$ROOT/cluster_scripts/logs_rod/${TAG}_%j.out
#SBATCH --error=$ROOT/cluster_scripts/logs_rod/${TAG}_%j.err
# ============================================================
#  ROD ${TAG}  (regime=${REGIME}, combiner=${COMB})
# ============================================================
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate rawdet

export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_PROJECT=mdp-raw-preprocessing
export WANDB_MODE=offline

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROD_ROOT=/scratch/INC1526354/rod

echo "=== Starting ROD ${TAG} ==="
cd $ROOT
export PYTHONPATH="\$(pwd):\${PYTHONPATH}"

OUT=runs_rod/rod__${TAG}

python -m core.train \\
    --regime ${REGIME} \\
    --dataset rod \\
    --num-classes 5 \\
    --combine ${COMB} \\
    --epochs 50 \\
    --bs 4 \\
    --val-every 1 \\
    --val-max-batches 50 \\
    --out "\$OUT" \\
    --wandb \\
    --wandb-entity georgiasouval-university-of-warwick

echo "=== syncing W&B offline run ==="
wandb sync "\$OUT"/wandb/offline-run-* 2>/dev/null
echo "=== ROD ${TAG} finished ==="
EOF

  # ---------- eval script ----------
  cat > "$SDIR/eval_${TAG}.sh" << EOF
#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_rod_${TAG}
#SBATCH --output=$ROOT/cluster_scripts/logs_rod/eval_${TAG}_%j.out
#SBATCH --error=$ROOT/cluster_scripts/logs_rod/eval_${TAG}_%j.err
# ============================================================
#  ROD eval ${TAG}  ->  vs fcos, fasterrcnn, detr, retinanet
# ============================================================
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate rawdet

export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_PROJECT=mdp-raw-preprocessing
export WANDB_RUN_GROUP=eval__rod__${TAG}
export WANDB_MODE=offline

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROD_ROOT=/scratch/INC1526354/rod

cd $ROOT
export PYTHONPATH="\$(pwd):\${PYTHONPATH}"

PREP=runs_rod/rod__${TAG}/preprocessor.pth
mkdir -p results_rod/eval_${COMB}

if [ ! -f "\$PREP" ]; then
    echo "ERROR: checkpoint \$PREP not found — did training for ${TAG} finish?"
    exit 1
fi

echo "=== Evaluating ROD ${TAG} against all detectors ==="
for DET in fcos fasterrcnn detr retinanet; do
    echo "--- ROD ${TAG} vs \$DET ---"
    python -m core.evaluate \\
        --prep "\$PREP" \\
        --detector "\$DET" \\
        --dataset rod \\
        --num-classes 5 \\
        --out results_rod/eval_${COMB}/${TAG}__\${DET}.json \\
        --wandb \\
        --wandb-name rod__${TAG}__eval_\${DET}
done

echo "=== syncing W&B offline runs ==="
wandb sync --sync-all 2>/dev/null
echo "=== ROD eval ${TAG} finished ==="
EOF

  chmod +x "$SDIR/run_${TAG}.sh" "$SDIR/eval_${TAG}.sh"
  echo "generated: run_${TAG}.sh + eval_${TAG}.sh"
done

echo ""
echo "DONE: $(ls $SDIR/run_*.sh | wc -l) train + $(ls $SDIR/eval_*.sh | wc -l) eval scripts in $SDIR"
