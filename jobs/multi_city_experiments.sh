#!/bin/bash
#SBATCH --job-name=climate_fix
#SBATCH --account=class
#SBATCH --partition=class
#SBATCH --qos=default
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/city_%A_%a.out
#SBATCH --error=logs/city_%A_%a.err

echo "======================================"
echo "Job: $SLURM_ARRAY_JOB_ID, Task: $SLURM_ARRAY_TASK_ID"
echo "======================================"

cd ~/projects/differentiable-climate-policy-economy-model
source venv/bin/activate

python -c "import jax; print('JAX devices:', jax.devices())"

CITIES=("nyc" "la")
CONFIGS=("balanced" "emissions_focus" "growth_focus" "aggressive")

CITY_IDX=$(($SLURM_ARRAY_TASK_ID / 4))
CONFIG_IDX=$(($SLURM_ARRAY_TASK_ID % 4))
CITY=${CITIES[$CITY_IDX]}
CONFIG=${CONFIGS[$CONFIG_IDX]}

case $CONFIG in
    "balanced")       W_E=0.5; W_G=3.0; LAMBDA=5.0 ;;
    "emissions_focus") W_E=1.0; W_G=2.0; LAMBDA=10.0 ;;
    "growth_focus")   W_E=0.3; W_G=5.0; LAMBDA=3.0 ;;
    "aggressive")     W_E=0.7; W_G=3.0; LAMBDA=7.0 ;;
esac

echo "City: $CITY, Config: $CONFIG"

OUTPUT_DIR="results/multi_city/${SLURM_ARRAY_JOB_ID}/${CITY}/${CONFIG}"
mkdir -p "$OUTPUT_DIR"

python experiments/train_city.py \
    --city "$CITY" \
    --w_E "$W_E" --w_G "$W_G" --lambda_term "$LAMBDA" \
    --lr 0.01 \
    --num_iters 200 \
    --T 30 \
    --seed $((42 + SLURM_ARRAY_TASK_ID)) \
    --output_dir "$OUTPUT_DIR" \
    --name "${CITY}_${CONFIG}" \
    --use_traffic false

if [ $? -eq 0 ]; then
    touch "$OUTPUT_DIR/COMPLETED"
    echo "✓ SUCCESS"
else
    touch "$OUTPUT_DIR/FAILED"
fi

echo "Done at $(date)"
