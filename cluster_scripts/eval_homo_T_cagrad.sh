#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_homo_T_cagrad
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs/eval_homo_T_cagrad_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs/eval_homo_T_cagrad_%j.err
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate rawdet
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PASCALRAW_ROOT=/scratch/INC1526354/pascalraw
cd /networkhome/WMGDS/souval_g/raw-mdp
export PYTHONPATH="$(pwd):${PYTHONPATH}"
PREP=runs/pascalraw__homo_T_cagrad/preprocessor.pth
if [ ! -f "$PREP" ]; then echo "MISSING $PREP"; exit 1; fi
echo "=== Evaluating homo_T_cagrad vs all 4 detectors (test split) ==="
for DET in fcos fasterrcnn detr retinanet; do
  echo "--- homo_T_cagrad vs $DET ---"
  python -m core.evaluate \
      --prep "$PREP" \
      --detector "$DET" \
      --dataset pascalraw \
      --num-classes 3 \
      --split test \
      --out /networkhome/WMGDS/souval_g/raw-mdp/results/eval_final/homo_T_cagrad__${DET}.json
done
echo "=== eval homo_T_cagrad finished ==="
