#!/usr/bin/env bash
# Fixed reproduction command for every node of this experiment tree.
# Contract (cardinal rule 2): identical on the baseline and on every child.
#   - resolves the pinned environment from the committed uv.lock (never re-resolves)
#   - runs the full cumulative claim-verification suite
#   - exits nonzero if ANY claim verifier or negative control fails
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
# Determinism: pin BLAS threading so linear-algebra reductions are bitwise stable
# across machines. Recorded in the environment stamp written by repro/common.py.
export OMP_NUM_THREADS="${REPRO_BLAS_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${REPRO_BLAS_THREADS:-1}"
export MKL_NUM_THREADS="${REPRO_BLAS_THREADS:-1}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[run.sh] uv not found; installing to \$HOME/.local/bin"
  curl -LsSf https://astral.sh/uv/0.5.11/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[run.sh] uv $(uv --version)"
# --frozen: use the committed uv.lock exactly; fail rather than silently re-resolve.
uv sync --frozen --python 3.12

exec uv run --frozen --python 3.12 python -m repro.run_all
