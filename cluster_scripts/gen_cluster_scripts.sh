#!/bin/bash
# ============================================================
#  gen_cluster_scripts.sh
#  Emits 13 standalone Slurm job scripts for the A/B/C grid.
#  Run ONCE from project root:  bash gen_cluster_scripts.sh
#  Then inspect one, then sbatch them.
# ============================================================

# --- paths you may need to adjust (defaults match your cluster) ---
PROJECT_ROOT="/networkhome/WMGDS/souval_g/raw-mdp"
SCRIPT_DIR="${PROJECT_ROOT}/cluster_scripts"
LOG_DIR="${SCRIPT_DIR}/logs"
CONDA_SH="/networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="rawdet"
DATA_ROOT="/scratch/INC1526354/pascalraw"
# Partitions on THIS cluster are TIME-LIMIT tiers on shared nodes (sinfo -s):
#   short 1h | medium 6h | long 2d | xlong 14d | test 3d(gpu-04, often idle)
# Rule: a job's --time must be <= its partition's cap, else sbatch REJECTS it.
#   <=6h jobs -> medium (schedules fast).  >6h -> long.  Canary -> test (idle).

# --- the 13 jobs: "regime combiner walltime partition" -----------
# walltime per-job from the run table (rounded up with headroom).
JOBS=(
  "solo_1   normgrad 03:00:00 medium"   # 1  A baseline  FCOS
  "solo_2   normgrad 03:00:00 medium"   # 2  A baseline  Faster R-CNN
  "solo_T   normgrad 04:00:00 medium"   # 3  A baseline  DETR
  "pair_12  normgrad 04:00:00 medium"   # 4  A headline  FCOS+FRCNN
  "pair_T1  normgrad 05:00:00 medium"   # 5  A headline  DETR+FCOS
  "pair_T2  normgrad 05:00:00 medium"   # 6  A headline  DETR+FRCNN
  "triple   normgrad 04:00:00 medium"   # 7  A headline  all 3
  "homo_1   normgrad 04:00:00 medium"   # 8  B control   FCOS x3
  "homo_2   normgrad 04:00:00 medium"   # 9  B control   FRCNN x3
  "homo_T   normgrad 12:00:00 long"     # 10 B control   DETR x3 (>6h -> long)
  "pair_12  sum      04:00:00 medium"   # 11 C ablation  naive sum
  "pair_T1  sum      05:00:00 medium"   # 12 C ablation  naive sum, DETR regime
  "triple   sum      04:00:00 medium"   # 13 C ablation  naive sum, full ensemble
)

mkdir -p "$LOG_DIR"

for job in "${JOBS[@]}"; do
  read -r REGIME COMBINE TIME PART <<< "$job"
  TAG="${REGIME}_${COMBINE}"
  OUTFILE="${SCRIPT_DIR}/run_${TAG}.sh"

  cat > "$OUTFILE" <<EOF
#!/bin/bash
#SBATCH --partition=${PART}
#SBATCH --gres=gpu:1
#SBATCH --time=${TIME}
#SBATCH --job-name=${TAG}
#SBATCH --output=${LOG_DIR}/${TAG}_%j.out
#SBATCH --error=${LOG_DIR}/${TAG}_%j.err
# ============================================================
#  ${TAG}  (regime=${REGIME}, combiner=${COMBINE})
# ============================================================

# -- Environment --------------------------------------------------
source ${CONDA_SH}
conda activate ${CONDA_ENV}

# -- W&B (headless node: log offline, sync after) -----------------
export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_PROJECT=mdp-raw-preprocessing
export WANDB_MODE=offline

# -- Memory + data path -------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PASCALRAW_ROOT=${DATA_ROOT}

# -- Training -----------------------------------------------------
echo "=== Starting ${TAG} ==="
cd ${PROJECT_ROOT}
export PYTHONPATH="\$(pwd):\${PYTHONPATH}"

OUT=runs/pascalraw__${TAG}

python -m core.train \\
    --regime ${REGIME} \\
    --dataset pascalraw \\
    --num-classes 3 \\
    --combine ${COMBINE} \\
    --epochs 10 \\
    --bs 4 \\
    --out "\$OUT" \\
    --wandb

# -- Sync the offline W&B run -------------------------------------
echo "=== syncing W&B offline run ==="
wandb sync "\$OUT"/wandb/offline-run-* 2>/dev/null

echo "=== ${TAG} finished ==="
EOF

  echo "wrote $OUTFILE  (time=${TIME})"
done

echo ""
echo "Done. 13 scripts in ${SCRIPT_DIR}/"
echo "Inspect one:   cat ${SCRIPT_DIR}/run_pair_T1_normgrad.sh"
echo "Submit all:    for f in ${SCRIPT_DIR}/run_*.sh; do sbatch \$f; done"