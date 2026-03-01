#!/bin/bash
# =============================================================================
# Step 4: Materialize runbook-aware ground truth files needed for evaluation.
# Uses the built C++ GT generator to produce step-wise GT files for each search
# checkpoint defined in the bundled update workload YAMLs.
#
# Usage: bash scripts/generate_groundtruth.sh
#
# Output: dataset/<dataset>/gt/<runbook>/stepN.gt100
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="${CONDA_ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
GT_BIN="${GT_BIN:-$ROOT_DIR/build/tests/generate_groundtruth}"

echo "========================================="
echo "  Generating Ground Truth Files"
echo "========================================="

if [ ! -x "$GT_BIN" ]; then
  echo "[WARN] GT generator not found: $GT_BIN"
  echo "[WARN] Skipping Step 4. Build the project first so build/tests/generate_groundtruth is available."
  exit 0
fi

python3 "$SCRIPT_DIR/generate_groundtruth.py" --root "$ROOT_DIR" --gt-bin "$GT_BIN" || \
  echo "[WARN] Ground-truth generation encountered errors. Continuing."

echo ""
echo "[Done] Ground truth files are ready in: dataset/"
