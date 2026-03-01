#!/bin/bash
# =============================================================================
# Step 3: Run all experiments (FV, IV, CONDA)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_BIN="${BUILD_BIN:-$ROOT_DIR/build/tests/test}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/dataset}"
RUNBOOK_DIR="${RUNBOOK_DIR:-$ROOT_DIR/runbooks}"

# Build hyperparameters (Paper standard)
BUILD_L=64; R=32; C=750; ALPHA=1.2; SEARCH_L=100; SEARCH_K=10

if [ ! -f "$BUILD_BIN" ]; then
    echo "[WARN] build/tests/test not found. Skipping Step 3."
    exit 0
fi
mkdir -p "$RESULTS_DIR"

base_count() {
    python3 - "$1" <<'PY'
import struct, sys
path = sys.argv[1]
try:
    with open(path, 'rb') as f:
        raw = f.read(8)
    if len(raw) != 8:
        print(-1)
    else:
        print(struct.unpack('<II', raw)[0])
except Exception:
    print(-1)
PY
}

required_points() {
    python3 - "$1" "$2" <<'PY'
import sys, yaml
runbook, key = sys.argv[1], sys.argv[2]
with open(runbook, 'r', encoding='utf-8') as f:
    doc = yaml.safe_load(f)
if key not in doc:
    print(-1)
    raise SystemExit
steps = doc[key]
step_size = int(steps["step_size"])
mx = 0
for i in range(1, step_size + 1):
    entry = steps[i]
    if entry["operation"] == "insert":
        mx = max(mx, int(entry["end"]))
print(mx)
PY
}

run_experiment() {
    local LABEL=$1; local KEY=$2; local DATA=$3; local QUERY=$4; local DTYPE=$5; local METRIC=$6; local RUNBOOK=$7
    if [ ! -s "$DATA" ] || [ ! -s "$QUERY" ] || [ ! -s "$RUNBOOK" ]; then
        echo "  [SKIP] $LABEL: missing input files"
        return
    fi
    local BASE_NUM
    BASE_NUM="$(base_count "$DATA")"
    local REQUIRED_NUM
    REQUIRED_NUM="$(required_points "$RUNBOOK" "$KEY")"
    if [ "$BASE_NUM" -lt 0 ] || [ "$REQUIRED_NUM" -lt 0 ]; then
        echo "  [SKIP] $LABEL: failed to inspect dataset/runbook"
        return
    fi
    if [ "$BASE_NUM" -lt "$REQUIRED_NUM" ]; then
        echo "  [SKIP] $LABEL: runbook requires $REQUIRED_NUM points but base file has $BASE_NUM"
        return
    fi

    for IDX in FV IV CONDA; do
        local OUT_DIR="${RESULTS_DIR}/${LABEL}_${IDX}"
        local LOG_FILE="${OUT_DIR}/run.log"
        if [ -s "$LOG_FILE" ] && grep -q "Delete execution time" "$LOG_FILE"; then
            echo "  [SKIP] $LABEL | $IDX already completed"
            continue
        fi

        mkdir -p "$OUT_DIR"
        echo "  Running: $LABEL | $IDX"
        "$BUILD_BIN" "$RUNBOOK" "$KEY" "$DATA" "$QUERY" \
            $BUILD_L $R $C $ALPHA $SEARCH_L $SEARCH_K \
            "$OUT_DIR" "$DTYPE" "$IDX" "$METRIC" > "$LOG_FILE" 2>&1 || echo "  [WARN] $LABEL $IDX failed"
    done
}

SW="$RUNBOOK_DIR/sliding_window_runbook.yaml"
EX="$RUNBOOK_DIR/expiration_runbook.yaml"
CLU="$RUNBOOK_DIR/clustered_runbook.yaml"

echo "--- 1M Sliding Window ---"
run_experiment "bigann-1M-sw"     "bigann-1M"     "$DATASET_DIR/bigann-1M/base.1B.u8bin"         "$DATASET_DIR/bigann-1M/query.public.10K.u8bin"       "u8"    "L2" "$SW"
run_experiment "msspacev-1M-sw"   "msspacev-1M"   "$DATASET_DIR/msspacev-1M/spacev1b_base.i8bin" "$DATASET_DIR/msspacev-1M/query.i8bin"                "i8"    "L2" "$SW"
run_experiment "msturing-1M-sw"   "msturing-1M"   "$DATASET_DIR/msturing-1M/base1b.fbin"         "$DATASET_DIR/msturing-1M/testQuery10K.fbin"          "float" "L2" "$SW"
run_experiment "wikipedia-1M-sw"  "wikipedia-1M"  "$DATASET_DIR/wikipedia-1M/wikipedia_base.bin" "$DATASET_DIR/wikipedia-1M/wikipedia_query.bin"       "float" "IP" "$SW"
run_experiment "text2image-1M-sw" "text2image-1M" "$DATASET_DIR/text2image-1M/base.1B.fbin"      "$DATASET_DIR/text2image-1M/query.public.100K.fbin"   "float" "IP" "$SW"

echo "--- 1M Expiration Time ---"
run_experiment "bigann-1M-ex"     "bigann-1M"     "$DATASET_DIR/bigann-1M/base.1B.u8bin"         "$DATASET_DIR/bigann-1M/query.public.10K.u8bin"       "u8"    "L2" "$EX"
run_experiment "msspacev-1M-ex"   "msspacev-1M"   "$DATASET_DIR/msspacev-1M/spacev1b_base.i8bin" "$DATASET_DIR/msspacev-1M/query.i8bin"                "i8"    "L2" "$EX"
run_experiment "msturing-1M-ex"   "msturing-1M"   "$DATASET_DIR/msturing-1M/base1b.fbin"         "$DATASET_DIR/msturing-1M/testQuery10K.fbin"          "float" "L2" "$EX"
run_experiment "wikipedia-1M-ex"  "wikipedia-1M"  "$DATASET_DIR/wikipedia-1M/wikipedia_base.bin" "$DATASET_DIR/wikipedia-1M/wikipedia_query.bin"       "float" "IP" "$EX"
run_experiment "text2image-1M-ex" "text2image-1M" "$DATASET_DIR/text2image-1M/base.1B.fbin"      "$DATASET_DIR/text2image-1M/query.public.100K.fbin"   "float" "IP" "$EX"

echo "--- msturing Clustered ---"
run_experiment "msturing-10M-clu" "msturing-10M-clustered" "$DATASET_DIR/msturing-10M-clustered/msturing-10M-clustered.fbin" "$DATASET_DIR/msturing-10M-clustered/testQuery10K.fbin" "float" "L2" "$CLU"
run_experiment "msturing-30M-clu" "msturing-30M-clustered" "$DATASET_DIR/msturing-30M-clustered/30M-clustered64.fbin"        "$DATASET_DIR/msturing-30M-clustered/testQuery10K.fbin" "float" "L2" "$CLU"

echo "[Done] All experiments finished."
