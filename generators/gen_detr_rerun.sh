#!/bin/bash
PROJECT_ROOT="/networkhome/WMGDS/souval_g/raw-mdp"
SCRIPT_DIR="${PROJECT_ROOT}/cluster_scripts"
LOG_DIR="${SCRIPT_DIR}/logs"
CONDA_SH="/networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="rawdet"
DATA_ROOT="/scratch/INC1526354/pascalraw"
JOBS=(
  "solo_T  normgrad"
  "pair_T1 normgrad" "pair_T1 sum" "pair_T1 mgda" "pair_T1 pcgrad" "pair_T1 cagrad"
  "pair_T2 normgrad" "pair_T2 mgda" "pair_T2 pcgrad" "pair_T2 cagrad"
  "triple  normgrad" "triple  mgda" "triple  pcgrad" "triple  cagrad"
  "homo_T  normgrad" "homo_T  mgda" "homo_T  pcgrad" "homo_T  cagrad"
)
mkdir -p "$LOG_DIR"
for job in "${JOBS[@]}"; do
  read -r REGIME COMBINE <<< "$job"
  TAG="${REGIME}_${COMBINE}"
  OUTFILE="${SCRIPT_DIR}/run_${TAG}.sh"
  cat > "$OUTFILE" <<EOF
#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --job-name=${TAG}
#SBATCH --output=${LOG_DIR}/${TAG}_%j.out
#SBATCH --error=${LOG_DIR}/${TAG}_%j.err
source ${CONDA_SH}
conda activate ${CONDA_ENV}
export WANDB_ENTITY=georgiasouval-university-of-warwick
export WANDB_PROJECT=mdp-raw-preprocessing
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PASCALRAW_ROOT=${DATA_ROOT}
echo "=== Starting ${TAG} ==="
cd ${PROJECT_ROOT}
export PYTHONPATH="\$(pwd):\${PYTHONPATH}"
OUT=runs/pascalraw__${TAG}
python -m core.train \\
    --regime ${REGIME} \\
    --dataset pascalraw \\
    --num-classes 3 \\
    --combine ${COMBINE} \\
    --epochs 80 \\
    --bs 4 \\
    --val-every 1 \\
    --out "\$OUT" \\
    --wandb
echo "=== syncing W&B offline run ==="
wandb sync "\$OUT"/wandb/offline-run-* 2>/dev/null
echo "=== ${TAG} finished ==="
EOF
  echo "wrote run_${TAG}.sh"
done
echo "Done. 18 DETR run scripts (re)written."
