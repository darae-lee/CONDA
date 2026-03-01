import argparse
import glob
import os
import re
import struct

import matplotlib.pyplot as plt
import numpy as np

TYPE_COLORS = {
    "FV": "#498365",
    "IV": "#AE3C75",
    "CONDA": "#00243D",
}

TYPE_LABELS = {
    "FV": "FreshVamana",
    "IV": "IPVamana",
    "CONDA": "CONDA",
}


def load_groundtruth(gt_path):
    with open(gt_path, "rb") as f:
        try:
            num, dim = struct.unpack("<II", f.read(8))
            f.seek(0, os.SEEK_END)
            size = f.tell()
            expected = 8 + num * dim * 4
            if size == expected:
                f.seek(8)
                return np.fromfile(f, dtype=np.uint32).reshape(num, dim)
        except Exception:
            pass

    data = np.fromfile(gt_path, dtype=np.uint32)
    if data.size == 0:
        return None
    dim = data[0]
    num = len(data) // (dim + 1)
    return data.reshape(num, dim + 1)[:, 1:]


def compute_recall(res_path, gt_matrix, k=10):
    if gt_matrix is None:
        return 0.0

    data = np.fromfile(res_path, dtype=np.uint32)
    if data.size == 0:
        return 0.0

    dim = data[0]
    num = len(data) // (dim + 1)
    res = data.reshape(num, dim + 1)[:, 1:]
    effective_k = min(k, res.shape[1], gt_matrix.shape[1])
    if effective_k == 0:
        return 0.0

    recalls = []
    for i in range(min(len(res), len(gt_matrix))):
        matches = set(res[i][:effective_k]).intersection(set(gt_matrix[i][:effective_k]))
        recalls.append(len(matches) / float(effective_k))
    return float(np.mean(recalls)) if recalls else 0.0


def dataset_and_runbook_for_label(label):
    if label.endswith("-sw"):
        return label[:-3], "sliding_window_runbook"
    if label.endswith("-ex"):
        return label[:-3], "expiration_runbook"
    if label.endswith("-clu"):
        return label[:-4], "clustered_runbook"
    return None, None


def parse_run_log(log_path):
    steps = []
    current_step = None
    search_times, insert_times, delete_times = {}, {}, {}

    with open(log_path, "r") as f:
        for line in f:
            step_match = re.search(r"^(\d+)\s+:\s+(search|insert|delete)", line)
            if step_match:
                current_step = int(step_match.group(1))
                if current_step not in steps:
                    steps.append(current_step)

            if current_step is None:
                continue

            search_match = re.search(r"Search execution time[^:]*:\s+([0-9.]+)", line)
            if search_match:
                search_times[current_step] = float(search_match.group(1))

            insert_match = re.search(r"BatchInsert execution time:\s+([0-9.]+)", line)
            if insert_match:
                insert_times[current_step] = float(insert_match.group(1))

            delete_match = re.search(r"Delete execution time:\s+([0-9.]+)", line)
            if delete_match:
                delete_times[current_step] = float(delete_match.group(1))

    return steps, search_times, insert_times, delete_times


def gt_path_for_label(root_dir, label, step):
    dataset_key, runbook_name = dataset_and_runbook_for_label(label)
    if dataset_key is None:
        return None
    return os.path.join(root_dir, "dataset", dataset_key, "gt", runbook_name, f"step{step}.gt100")


def available_gt_steps(root_dir, label):
    dataset_key, runbook_name = dataset_and_runbook_for_label(label)
    if dataset_key is None:
        return set()
    gt_dir = os.path.join(root_dir, "dataset", dataset_key, "gt", runbook_name)
    if not os.path.isdir(gt_dir):
        return set()
    steps = set()
    for path in glob.glob(os.path.join(gt_dir, "step*.gt100")):
        name = os.path.basename(path)
        m = re.match(r"step(\d+)\.gt100$", name)
        if m:
            steps.add(int(m.group(1)))
    return steps


def discover_labels(results_dir):
    labels = set()
    for name in os.listdir(results_dir):
        full = os.path.join(results_dir, name)
        if not os.path.isdir(full):
            continue
        for suffix in ("_FV", "_IV", "_CONDA"):
            if name.endswith(suffix):
                labels.add(name[: -len(suffix)])
                break
    return sorted(labels)


def process_results(root_dir):
    results_dir = os.path.join(root_dir, "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    labels = discover_labels(results_dir)

    for label in labels:
        print(f"--- Processing {label} ---")
        gt_cache = {}
        label_gt_steps = available_gt_steps(root_dir, label)
        has_any_series = False
        has_recall_series = False
        has_update_series = False

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        recall_ax, update_ax = axes

        for idx_type in ["FV", "IV", "CONDA"]:
            color = TYPE_COLORS[idx_type]
            legend_label = TYPE_LABELS[idx_type]
            run_dir = os.path.join(results_dir, f"{label}_{idx_type}")
            log_path = os.path.join(run_dir, "run.log")
            if not os.path.exists(log_path):
                continue

            _, search_times, insert_times, delete_times = parse_run_log(log_path)

            search_steps = sorted(search_times.keys())
            matched_search_steps = []
            recalls = []
            for step in search_steps:
                res_files = glob.glob(os.path.join(run_dir, f"{step}-Ls*.res"))
                if not res_files or step not in label_gt_steps:
                    continue
                gt_path = gt_path_for_label(root_dir, label, step)
                if gt_path not in gt_cache:
                    gt_cache[gt_path] = load_groundtruth(gt_path) if gt_path and os.path.exists(gt_path) else None
                matched_search_steps.append(step)
                recalls.append(compute_recall(res_files[0], gt_cache[gt_path], k=10))

            if search_steps and not matched_search_steps:
                print(
                    f"  [WARN] {label} | {idx_type}: no matching GT steps for "
                    f"search steps {search_steps[:5]}{'...' if len(search_steps) > 5 else ''}"
                )

            if matched_search_steps:
                recall_ax.plot(
                    matched_search_steps,
                    recalls,
                    label=legend_label,
                    color=color,
                )
                has_any_series = True
                has_recall_series = True

            update_steps = sorted(set(insert_times) | set(delete_times))
            update_latencies = [insert_times.get(s, 0.0) + delete_times.get(s, 0.0) for s in update_steps]

            if update_steps:
                update_ax.plot(
                    update_steps,
                    update_latencies,
                    label=legend_label,
                    color=color,
                )
                has_any_series = True
                has_update_series = True

        if not has_any_series:
            print(f"Skipping {label}: no plottable recall/update data found.")
            plt.close(fig)
            continue

        recall_ax.set_title(f"Recall@10 Over Timestep ({label})", fontsize=13)
        recall_ax.set_xlabel("Timestep", fontsize=12)
        recall_ax.set_ylabel("Recall@10", fontsize=12)
        if not recall_ax.lines:
            recall_ax.text(0.5, 0.5, "No matching recall data", ha="center", va="center", transform=recall_ax.transAxes)
        recall_ax.grid()

        update_ax.set_title(f"Total Update (Insert+Delete) Latency Over Timestep ({label})", fontsize=13)
        update_ax.set_xlabel("Timestep", fontsize=12)
        update_ax.set_ylabel("Latency (sec)", fontsize=12)
        if not update_ax.lines:
            update_ax.text(0.5, 0.5, "No update data", ha="center", va="center", transform=update_ax.transAxes)
        update_ax.grid()

        handles, labels = [], []
        for ax in (recall_ax, update_ax):
            h, l = ax.get_legend_handles_labels()
            for handle, lbl in zip(h, l):
                if lbl not in labels:
                    handles.append(handle)
                    labels.append(lbl)
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.98))

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out_png = os.path.join(plots_dir, f"{label}_performance_graph.png")
        fig.savefig(out_png)
        print(f"Saved figure to {out_png}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    args = parser.parse_args()
    process_results(args.root)


if __name__ == "__main__":
    main()
