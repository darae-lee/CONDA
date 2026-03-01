#!/bin/bash
# =============================================================================
# Step 5: Parse recall & latency from experiment logs and plot graphs.
# Reads results/<label>_<IDX>/run.log and *.res files, computes Recall@K
# against ground truth files, and saves PNG figures to results/.
#
# Usage: bash scripts/parse_result.sh
#
# Output: results/<dataset>_graph.png
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
ROOT_DIR="${CONDA_ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

echo "========================================="
echo "  Plotting Recall & Latency Graphs"
echo "========================================="

python3 "$SCRIPT_DIR/plot_results.py" --root "$ROOT_DIR" || \
  echo "[WARN] Plot generation encountered errors. Continuing."

echo ""
echo "[Done] Performance graphs saved to: results/plots/"
