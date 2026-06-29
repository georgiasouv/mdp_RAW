#!/bin/bash
# ============================================================
#  gen_combiner_scripts.sh
#  Emits run scripts for the NEW combiner experiments:
#    combiners {mgda, pcgrad, cagrad} x regimes
#    {pair_12, pair_T1, pair_T2, triple, homo_T}
#  = 15 FRESH training runs (no --resume), 80 epochs each.
#
#  homo_T is the single homo control kept to DEMONSTRATE that
#  conflict-resolution combiners are a no-op on identical gradients.
#  (homo_1/homo_2 are intentionally excluded -- same reasoning,
#  one proof regime is enough.)
#
#  Run ONCE from project root:  bash cluster_scripts/gen_combiner_scripts.sh
#  Then inspect one, then sbatch them.
# ============================================================
PROJECT_ROOT="/networkhome/WMGDS/souval_g/raw-mdp"
SCRIPT_DIR="${PROJECT_ROOT}/cluster_scripts"
LOG_DIR="${SCRIPT_DIR}/logs"
CONDA_SH="/networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="rawdet"
DATA_ROOT="/scratch/INC1526354/pascalraw"
WANDB_ENTITY="georgiasouval-university-of-warwick"
EPOCHS=80
WALLTIME="14:00:00"
PART="long"          # 80-epoch fresh runs are long; long partition (>6h cap)

# regimes that get the new combiners (diverse + ONE homo proof)
REGIMES=(pair_12 pair_T1 pair_T2 triple homo_T)
# the three sophisticated combiners (sum/normgrad already run)
COMBINERS=(mgda pcgrad cagrad)

mkdir -p "$LOG_DIR"
count=0
for REGIME in "${REGIMES[@]}"; do
  for COMBINE in "${COMBINERS[@]}"; do
    TAG="${REGIME}_${COMBINE}"
    OUTFILE="${SCRIPT_DIR}/run_${TAG}.sh"
    cat > "$OUTFILE" <<EOF
#!/bin/bash
#SBATCH --partition=${PART}
#SBATCH --gres=gpu:1
#SBATCH --time=${WALLTIME}
#SBATCH --job-name=${TAG}
#SBATCH --output=${LOG_DIR}/${TAG}_%j.out
#SBATCH --error=${LOG_DIR}/${TAG}_%j.err
# ============================================================
#  ${TAG}  (regime=${REGIME}, combiner=${COMBINE})  FRESH 0->${EPOCHS}
# ============================================================
# -- Environment --------------------------------------------------
source ${CONDA_SH}
conda activate ${CONDA_ENV}
# -- W&B (headless node: log offline, sync after) -----------------
export WANDB_ENTITY=${WANDB_ENTITY}
export WANDB_PROJECT=mdp-raw-preprocessing
export WANDB_MODE=offline
# -- Memory + data path -------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PASCALRAW_ROOT=${DATA_ROOT}
# -- Training (FRESH -- no --resume, starts at epoch 0) -----------
echo "=== Starting ${TAG} ==="
cd ${PROJECT_ROOT}
export PYTHONPATH="\$(pwd):\${PYTHONPATH}"
OUT=runs/pascalraw__${TAG}
python -m core.train \\
    --regime ${REGIME} \\
    --dataset pascalraw \\
    --num-classes 3 \\
    --combine ${COMBINE} \\
    --epochs ${EPOCHS} \\
    --bs 4 \\
    --val-every 1 \\
    --val-max-batches 50 \\
    --out "\$OUT" \\
    --wandb \\
    --wandb-entity ${WANDB_ENTITY}
# -- Sync the offline W&B run -------------------------------------
echo "=== syncing W&B offline run ==="
wandb sync "\$OUT"/wandb/offline-run-* 2>/dev/null
echo "=== ${TAG} finished ==="
EOF
    count=$((count + 1))
    echo "wrote $OUTFILE"
  done
done
echo ""
echo "Done. Generated $count combiner run scripts in ${SCRIPT_DIR}/"
echo "Inspect one:   cat ${SCRIPT_DIR}/run_pair_12_mgda.sh"
echo "Submit all 15: for r in pair_12 pair_T1 pair_T2 triple homo_T; do for c in mgda pcgrad cagrad; do sbatch ${SCRIPT_DIR}/run_\${r}_\${c}.sh; done; done"