#!/bin/bash
PROJECT_ROOT="/networkhome/WMGDS/souval_g/raw-mdp"
SCRIPT_DIR="${PROJECT_ROOT}/cluster_scripts"
LOG_DIR="${SCRIPT_DIR}/logs"
CONDA_SH="/networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="rawdet"
DATA_ROOT="/scratch/INC1526354/pascalraw"
RES_DIR="${PROJECT_ROOT}/results/eval_final"
TAGS=(
  solo_1_normgrad solo_2_normgrad solo_T_normgrad
  pair_12_normgrad pair_12_sum pair_12_mgda pair_12_pcgrad pair_12_cagrad
  pair_T1_normgrad pair_T1_sum pair_T1_mgda pair_T1_pcgrad pair_T1_cagrad
  pair_T2_normgrad pair_T2_mgda pair_T2_pcgrad pair_T2_cagrad
  triple_normgrad triple_sum triple_mgda triple_pcgrad triple_cagrad
  homo_1_normgrad homo_2_normgrad
  homo_T_normgrad homo_T_mgda homo_T_pcgrad homo_T_cagrad
)
mkdir -p "$LOG_DIR" "$RES_DIR"
for TAG in "${TAGS[@]}"; do
  OUTFILE="${SCRIPT_DIR}/eval_${TAG}.sh"
  cat > "$OUTFILE" <<EOF
#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_${TAG}
#SBATCH --output=${LOG_DIR}/eval_${TAG}_%j.out
#SBATCH --error=${LOG_DIR}/eval_${TAG}_%j.err
source ${CONDA_SH}
conda activate ${CONDA_ENV}
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PASCALRAW_ROOT=${DATA_ROOT}
cd ${PROJECT_ROOT}
export PYTHONPATH="\$(pwd):\${PYTHONPATH}"
PREP=runs/pascalraw__${TAG}/preprocessor.pth
if [ ! -f "\$PREP" ]; then echo "MISSING \$PREP"; exit 1; fi
echo "=== Evaluating ${TAG} vs all 4 detectors (test split) ==="
for DET in fcos fasterrcnn detr retinanet; do
  echo "--- ${TAG} vs \$DET ---"
  python -m core.evaluate \\
      --prep "\$PREP" \\
      --detector "\$DET" \\
      --dataset pascalraw \\
      --num-classes 3 \\
      --split test \\
      --out ${RES_DIR}/${TAG}__\${DET}.json
done
echo "=== eval ${TAG} finished ==="
EOF
  echo "wrote eval_${TAG}.sh"
done
echo "Done. 28 eval scripts -> results/eval_final/"
