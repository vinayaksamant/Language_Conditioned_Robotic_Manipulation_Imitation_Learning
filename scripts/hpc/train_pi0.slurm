#!/usr/bin/env bash
#SBATCH --job-name=pi0-franka
#SBATCH --time=01:55:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=pi0-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-pi0}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data/lerobot/vla_centered_v4}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/pi0_hpc}"
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-500}"
VALIDATION_FREQUENCY="${VALIDATION_FREQUENCY:-500}"
MAX_VALIDATION_SAMPLES="${MAX_VALIDATION_SAMPLES:-100}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
CACHE_ROOT="${PI0_CACHE_DIR:-${SCRATCH:-${PROJECT_ROOT}/.cache}}"

source "${VENV_DIR}/bin/activate"
mkdir -p "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/torch" "$(dirname -- "${OUTPUT_DIR}")"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
nvidia-smi
python scripts/pi0_preflight.py     --dataset-dir "${DATASET_DIR}"     --output-dir "${OUTPUT_DIR}"     --require-cuda

python scripts/train_pi0.py     --dataset-dir "${DATASET_DIR}"     --output-dir "${OUTPUT_DIR}"     --steps "${STEPS}"     --batch-size "${BATCH_SIZE}"     --num-workers "${NUM_WORKERS}"     --save-frequency "${SAVE_FREQUENCY}"     --log-frequency 20     --validation-frequency "${VALIDATION_FREQUENCY}"     --max-validation-samples "${MAX_VALIDATION_SAMPLES}"     --resume auto     --run

python scripts/verify_pi0_training.py     --output-dir "${OUTPUT_DIR}"     --expected-steps "${STEPS}"
