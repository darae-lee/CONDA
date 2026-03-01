#!/bin/bash
# =============================================================================
# End-to-end reproduction wrapper.
# =============================================================================
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "========================================="
echo "  Full Reproduction Pipeline"
echo "========================================="

echo "[1/5] Download datasets"
bash "$ROOT_DIR/scripts/download_dataset.sh" || echo "[WARN] Step 1 failed. Continuing."

echo "[2/5] Build"
cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build" -DCMAKE_BUILD_TYPE=Release || echo "[WARN] Configure failed. Continuing."
cmake --build "$ROOT_DIR/build" -j"$(nproc)" || echo "[WARN] Build failed. Continuing."

echo "[3/5] Run experiments"
bash "$ROOT_DIR/scripts/run_experiments.sh" || echo "[WARN] Step 3 failed. Continuing."

echo "[4/5] Generate runbook-aware ground truths"
bash "$ROOT_DIR/scripts/generate_groundtruth.sh" || echo "[WARN] Step 4 failed. Continuing."

echo "[5/5] Plot results"
bash "$ROOT_DIR/scripts/parse_result.sh" || echo "[WARN] Step 5 failed. Continuing."

echo "[Done] Full pipeline finished."
