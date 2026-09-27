#!/usr/bin/env bash
#SBATCH --job-name=pi0-eval
#SBATCH --time=01:55:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=pi0-eval-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-pi0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/pi0_hpc}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/checkpoints/last/pretrained_model}"
SPLIT="${SPLIT:-validation}"
EPISODES="${EPISODES:-20}"
CACHE_ROOT="${PI0_CACHE_DIR:-${SCRATCH:-${PROJECT_ROOT}/.cache}}"
RESULTS_DIR="${RESULTS_DIR:-${OUTPUT_DIR}/metrics}"

if [[ "${SPLIT}" != "validation" && "${SPLIT}" != "test" ]]; then
    echo "SPLIT must be validation or test, got: ${SPLIT}" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
mkdir -p "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/torch" "${RESULTS_DIR}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
python scripts/verify_pi0_training.py --output-dir "${OUTPUT_DIR}"
python scripts/pi0_preflight.py --output-dir "${OUTPUT_DIR}" --require-cuda
python scripts/evaluate_pi0_policy.py --checkpoint "${CHECKPOINT}" --split "${SPLIT}" --episodes "${EPISODES}" --results "${RESULTS_DIR}/${SPLIT}.json"
