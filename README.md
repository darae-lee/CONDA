# CONDA: A Connectivity-Aware Dynamic Index for Approximate Nearest Neighbor Search over Evolving Data

This repository contains the source code, datasets, and execution scripts required to reproduce the experimental results presented in our paper. Specifically, we provide scripts to compare **CONDA** with **FreshVamana** and **IPVamana**. 

This implementation builds on components from [DiskANN](https://github.com/microsoft/DiskANN) and [NSG](https://github.com/ZJULearning/nsg).

## 1. Hardware and Software Requirements

- **OS**: Linux (Ubuntu 20.04/22.04 recommended)
- **Compiler**: GCC with C++17 support
- **Dependencies**: 
  - CMake (>= 3.8)
  - Boost (`libboost-all-dev`)
  - OpenMP
  - yaml-cpp (`libyaml-cpp-dev`)
  - Python 3.8+ (for graph generation and evaluation scripts)
  
## 2. Setup and Build

Install the Python requirements used by the evaluation and plotting scripts, then build the project:

```bash
git clone https://github.com/darae-lee/CONDA.git
cd CONDA

pip install -r requirements.txt

mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j $(nproc)
cd ..
```

## 3. Dataset Preparation

Run the automated download script:

```bash
bash scripts/download_dataset.sh
```

This script will:
1. Clone `big-ann-benchmarks` (if not already present in the sibling directory)
2. Download all datasets using `create_dataset.py`
3. Create symlinks in `dataset/<dataset-name>/`

The dataset download workflow is based on the official BIG-ANN benchmark repository:
- https://github.com/harsha-simhadri/big-ann-benchmarks

The bundled update workloads are also based on `big-ann-benchmarks` and are provided under `runbooks/`:
- Sliding Window (`SW`): `sliding_window_runbook.yaml`
- Expiration (`EX`): `expiration_runbook.yaml`
- Clustered (`CS`): `clustered_runbook.yaml`

If dataset download fails in your environment, first check the upstream `big-ann-benchmarks` repository and dataset instructions above. You can also place the required files directly under `dataset/<dataset-name>/` and rerun the later steps.

## 4. Running the Experiments

The pipeline is split into modular steps:

```bash
# Step 3: Run experiments (CONDA vs FreshVamana vs IPVamana)
bash scripts/run_experiments.sh

# Step 4: Materialize Ground Truths
#  - generate runbook-aware step-wise GTs with the built C++ GT generator
#    (based on include/distance.h)
bash scripts/generate_groundtruth.sh

# Step 5: Generate performance plots
bash scripts/parse_result.sh
```

Results are saved in `results/` and plots in `results/plots/`.

For the end-to-end pipeline, run:

```bash
bash scripts/run_all.sh
```

## 5. Reproducibility Evaluation

Full reproduction pipeline:

```bash
# Step 1: Download datasets
bash scripts/download_dataset.sh

# Step 2: Build
mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc) && cd ..

# Step 3: Run all experiments
bash scripts/run_experiments.sh

# Step 4: Materialize ground truths
bash scripts/generate_groundtruth.sh

# Step 5: Generate graphs
bash scripts/parse_result.sh
```

> **Note**: If some dataset files cannot be downloaded during Step 1, for example due to network issues or upstream hosting constraints, refer to the upstream `big-ann-benchmarks` repository linked in Section 3. Missing datasets will cause the corresponding experiments to be skipped automatically.
