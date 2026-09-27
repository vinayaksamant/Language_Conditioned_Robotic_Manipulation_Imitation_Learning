#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-pi0}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="${PYTHON_BIN}"
else
    PYTHON=""
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            PYTHON="${candidate}"
            break
        fi
    done
fi
if [[ -z "${PYTHON}" ]]; then
    echo "Python 3.10-3.12 was not found. Load your university Python module, then rerun." >&2
    exit 1
fi

"${PYTHON}" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version'
"${PYTHON}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision
fi
python -m pip install -e "${PROJECT_ROOT}[pi0,dev]"

cd "${PROJECT_ROOT}"
python -m pytest -q
python scripts/pi0_preflight.py

echo
echo "Installation complete. Activate with:"
echo "source ${VENV_DIR}/bin/activate"
