#!/bin/bash
# =============================================================================
# Step 1: Download datasets only.
# Canonical runbooks are bundled in this repository under runbooks/.
#
# Usage: bash scripts/download_dataset.sh
# =============================================================================
set -euo pipefail

CONDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIG_ANN_DIR="${BIG_ANN_DIR:-$(dirname "$CONDA_DIR")/big-ann-benchmarks}"
DATASET_DIR="$CONDA_DIR/dataset"

mkdir -p "$DATASET_DIR"

warn() {
    echo "  [WARN] $*"
}

# Auto-clone big-ann-benchmarks if not present
if [ ! -d "$BIG_ANN_DIR" ]; then
    echo "[INFO] Cloning big-ann-benchmarks into: $BIG_ANN_DIR"
    if ! git clone https://github.com/harsha-simhadri/big-ann-benchmarks "$BIG_ANN_DIR"; then
        warn "Failed to clone big-ann-benchmarks. Existing local datasets, if any, may still be used."
    fi
fi

# Use venv python if available, otherwise system python
VENV_PYTHON="$BIG_ANN_DIR/.venv/bin/python3"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="$(which python3)"
fi

# ------------------------------------------------------------------------------
# Helper: download a file using curl with byte-range support
# ------------------------------------------------------------------------------
direct_download() {
    local URL=$1
    local DEST=$2
    local MAX_BYTES="${3:-}"
    if [ -f "$DEST" ]; then
        local SZ=$(stat -c%s "$DEST")
        if [ -n "$MAX_BYTES" ] && [ "$SZ" -ge "$MAX_BYTES" ]; then return 0;
        elif [ -z "$MAX_BYTES" ] && [ "$SZ" -gt 0 ]; then return 0; fi
    fi
    echo "  Downloading: $URL -> $DEST"
    if [ -n "$MAX_BYTES" ]; then
        curl -L --fail -# -H "Range: bytes=0-$((MAX_BYTES - 1))" "$URL" -o "$DEST"
    else
        curl -L --fail -# "$URL" -o "$DEST"
    fi
}

# ------------------------------------------------------------------------------
# Helper: patch the num_pts header
# ------------------------------------------------------------------------------
patch_header() {
    python3 -c "
import struct, sys
path, n, d = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, 'r+b') as f:
    f.seek(0); f.write(struct.pack('<II', n, d))
" "$1" "$2" "$3"
}

# ------------------------------------------------------------------------------
# Helper: download via big-ann-benchmarks dataset classes
# ------------------------------------------------------------------------------
create_dataset_download() {
    local DS_NAME=$1
    echo "--- Downloading: $DS_NAME ---"
    if [ ! -d "$BIG_ANN_DIR" ]; then
        warn "big-ann-benchmarks directory is missing. Cannot download $DS_NAME"
        return 0
    fi
    if ! (
        cd "$BIG_ANN_DIR"
        "$VENV_PYTHON" - "$DS_NAME" <<'PY'
import sys
import os
from benchmark.datasets import DATASETS
ds = DATASETS[sys.argv[1]]()
if hasattr(ds, "gt_fn"): ds.gt_fn = None
if hasattr(ds, "private_gt_url"): ds.private_gt_url = None
if hasattr(ds, "gt_url"): ds.gt_url = None

# Remove zero-byte partial files so big-ann-benchmarks doesn't treat them as valid.
paths = []
if getattr(ds, "qs_fn", None):
    paths.append(os.path.join(ds.basedir, ds.qs_fn))
if getattr(ds, "ds_fn", None):
    full_base = os.path.join(ds.basedir, ds.ds_fn)
    paths.append(full_base)
    if getattr(ds, "nb", None) is not None and getattr(ds, "d", None) is not None:
        paths.append(full_base + f".crop_nb_{ds.nb}")

for path in paths:
    if os.path.exists(path) and os.path.getsize(path) == 0:
        print(f"removing zero-byte file {path}")
        os.remove(path)

ds.prepare(False)
PY
    ); then
        warn "Dataset download failed for $DS_NAME"
    fi
}

echo "========================================="
echo "  Step 1: Downloading Datasets"
echo "========================================="

mkdir -p "$DATASET_DIR/bigann-1M"
BIGANN_BASE="$DATASET_DIR/bigann-1M/base.1B.u8bin"
BIGANN_QUERY="$DATASET_DIR/bigann-1M/query.public.10K.u8bin"
if ! direct_download "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin" "$BIGANN_BASE" 128000008; then
    warn "BIGANN base download failed"
fi
if [ -f "$BIGANN_BASE" ] && [ -s "$BIGANN_BASE" ]; then
    patch_header "$BIGANN_BASE" 1000000 128 || warn "Failed to patch BIGANN header"
else
    warn "BIGANN base download failed"
fi
if ! direct_download "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin" "$BIGANN_QUERY"; then
    warn "BIGANN query download failed"
fi

create_dataset_download "text2image-1M"
create_dataset_download "msturing-1M"
create_dataset_download "msspacev-1M"
create_dataset_download "wikipedia-1M"
create_dataset_download "msturing-10M-clustered"
create_dataset_download "msturing-30M-clustered"

# ------------------------------------------------------------------------------
# Linking files into dataset/<name>/
# ------------------------------------------------------------------------------
echo "  Linking files..."
BIG_ANN_DATA="$BIG_ANN_DIR/data"

link_file() {
    local SRC=$1
    local DST=$2
    if [ -f "$SRC" ] && [ -s "$SRC" ]; then ln -sf "$SRC" "$DST";
    else warn "Not found or empty: $SRC"; fi
}

mkdir -p "$DATASET_DIR/text2image-1M"
link_file "$BIG_ANN_DATA/text2image1B/base.1B.fbin.crop_nb_1000000"  "$DATASET_DIR/text2image-1M/base.1B.fbin"
link_file "$BIG_ANN_DATA/text2image1B/query.public.100K.fbin"         "$DATASET_DIR/text2image-1M/query.public.100K.fbin"

mkdir -p "$DATASET_DIR/msturing-1M"
link_file "$BIG_ANN_DATA/MSTuringANNS/base1b.fbin.crop_nb_1000000" "$DATASET_DIR/msturing-1M/base1b.fbin"
link_file "$BIG_ANN_DATA/MSTuringANNS/testQuery10K.fbin"            "$DATASET_DIR/msturing-1M/testQuery10K.fbin"

mkdir -p "$DATASET_DIR/msspacev-1M"
link_file "$BIG_ANN_DATA/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_1000000" "$DATASET_DIR/msspacev-1M/spacev1b_base.i8bin"
link_file "$BIG_ANN_DATA/MSSPACEV1B/query.i8bin"                          "$DATASET_DIR/msspacev-1M/query.i8bin"

mkdir -p "$DATASET_DIR/wikipedia-1M"
link_file "$BIG_ANN_DATA/wikipedia_cohere/wikipedia_base.bin.crop_nb_1000000" "$DATASET_DIR/wikipedia-1M/wikipedia_base.bin"
link_file "$BIG_ANN_DATA/wikipedia_cohere/wikipedia_query.bin"                 "$DATASET_DIR/wikipedia-1M/wikipedia_query.bin"

mkdir -p "$DATASET_DIR/msturing-10M-clustered"
link_file "$BIG_ANN_DATA/MSTuring-10M-clustered/msturing-10M-clustered.fbin" "$DATASET_DIR/msturing-10M-clustered/msturing-10M-clustered.fbin"
link_file "$BIG_ANN_DATA/MSTuring-10M-clustered/testQuery10K.fbin"            "$DATASET_DIR/msturing-10M-clustered/testQuery10K.fbin"

mkdir -p "$DATASET_DIR/msturing-30M-clustered"
link_file "$BIG_ANN_DATA/MSTuring-30M-clustered/30M-clustered64.fbin" "$DATASET_DIR/msturing-30M-clustered/30M-clustered64.fbin"
link_file "$BIG_ANN_DATA/MSTuring-30M-clustered/testQuery10K.fbin"     "$DATASET_DIR/msturing-30M-clustered/testQuery10K.fbin"

echo "  Canonical runbooks are already provided in runbooks/"
echo "  Step 1 complete!"
