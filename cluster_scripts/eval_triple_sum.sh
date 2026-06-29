#!/bin/bash
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_triple_sum
#SBATCH --output=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs/eval_triple_sum_%j.out
#SBATCH --error=/networkhome/WMGDS/souval_g/raw-mdp/cluster_scripts/logs/eval_triple_sum_%j.err
source /networkhome/WMGDS/souval_g/anaconda3/etc/profile.d/conda.sh
conda activate rawdet
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PASCALRAW_ROOT=/scratch/INC1526354/pascalraw
cd /networkhome/WMGDS/souval_g/raw-mdp
export PYTHONPATH="$(pwd):${PYTHONPATH}"
PREP=runs/pascalraw__triple_sum/preprocessor.pth
if [ ! -f "$PREP" ]; then echo "MISSING $PREP"; exit 1; fi
echo "=== Evaluating triple_sum vs all 4 detectors (test split) ==="
for DET in fcos fasterrcnn detr retinanet; do
  echo "--- triple_sum vs $DET ---"
  python -m core.evaluate \
      --prep "$PREP" \
      --detector "$DET" \
      --dataset pascalraw \
      --num-classes 3 \
      --split test \
      --out /networkhome/WMGDS/souval_g/raw-mdp/results/eval_final/triple_sum__${DET}.json
done
echo "=== eval triple_sum finished ==="
