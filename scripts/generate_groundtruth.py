import argparse
import os
import struct
import subprocess

import yaml


def parse_search_steps(runbook_path, dataset_key):
    with open(runbook_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    steps = doc[dataset_key]
    step_size = int(steps["step_size"])
    search_steps = []
    for step in range(1, step_size + 1):
        if steps[step]["operation"] == "search":
            search_steps.append(step)
    return search_steps


def max_insert_end(runbook_path, dataset_key):
    with open(runbook_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    steps = doc[dataset_key]
    step_size = int(steps["step_size"])
    max_end = 0
    for step in range(1, step_size + 1):
        entry = steps[step]
        if entry["operation"] == "insert":
            max_end = max(max_end, int(entry["end"]))
    return max_end


def gt_dir(root_dir, dataset_key, runbook_path):
    runbook_name = os.path.splitext(os.path.basename(runbook_path))[0]
    return os.path.join(root_dir, "dataset", dataset_key, "gt", runbook_name)


def complete_marker(gt_output_dir):
    return os.path.join(gt_output_dir, ".complete")


def generation_specs(root_dir):
    dataset_dir = os.path.join(root_dir, "dataset")
    runbook_dir = os.path.join(root_dir, "runbooks")
    sw = os.path.join(runbook_dir, "sliding_window_runbook.yaml")
    ex = os.path.join(runbook_dir, "expiration_runbook.yaml")
    clu = os.path.join(runbook_dir, "clustered_runbook.yaml")

    return [
        ("bigann-1M", os.path.join(dataset_dir, "bigann-1M", "base.1B.u8bin"),
         os.path.join(dataset_dir, "bigann-1M", "query.public.10K.u8bin"), "u8", "L2", sw),
        ("msspacev-1M", os.path.join(dataset_dir, "msspacev-1M", "spacev1b_base.i8bin"),
         os.path.join(dataset_dir, "msspacev-1M", "query.i8bin"), "i8", "L2", sw),
        ("msturing-1M", os.path.join(dataset_dir, "msturing-1M", "base1b.fbin"),
         os.path.join(dataset_dir, "msturing-1M", "testQuery10K.fbin"), "float", "L2", sw),
        ("wikipedia-1M", os.path.join(dataset_dir, "wikipedia-1M", "wikipedia_base.bin"),
         os.path.join(dataset_dir, "wikipedia-1M", "wikipedia_query.bin"), "float", "IP", sw),
        ("text2image-1M", os.path.join(dataset_dir, "text2image-1M", "base.1B.fbin"),
         os.path.join(dataset_dir, "text2image-1M", "query.public.100K.fbin"), "float", "IP", sw),
        ("bigann-1M", os.path.join(dataset_dir, "bigann-1M", "base.1B.u8bin"),
         os.path.join(dataset_dir, "bigann-1M", "query.public.10K.u8bin"), "u8", "L2", ex),
        ("msspacev-1M", os.path.join(dataset_dir, "msspacev-1M", "spacev1b_base.i8bin"),
         os.path.join(dataset_dir, "msspacev-1M", "query.i8bin"), "i8", "L2", ex),
        ("msturing-1M", os.path.join(dataset_dir, "msturing-1M", "base1b.fbin"),
         os.path.join(dataset_dir, "msturing-1M", "testQuery10K.fbin"), "float", "L2", ex),
        ("wikipedia-1M", os.path.join(dataset_dir, "wikipedia-1M", "wikipedia_base.bin"),
         os.path.join(dataset_dir, "wikipedia-1M", "wikipedia_query.bin"), "float", "IP", ex),
        ("text2image-1M", os.path.join(dataset_dir, "text2image-1M", "base.1B.fbin"),
         os.path.join(dataset_dir, "text2image-1M", "query.public.100K.fbin"), "float", "IP", ex),
        ("msturing-10M-clustered",
         os.path.join(dataset_dir, "msturing-10M-clustered", "msturing-10M-clustered.fbin"),
         os.path.join(dataset_dir, "msturing-10M-clustered", "testQuery10K.fbin"), "float", "L2", clu),
        ("msturing-30M-clustered",
         os.path.join(dataset_dir, "msturing-30M-clustered", "30M-clustered64.fbin"),
         os.path.join(dataset_dir, "msturing-30M-clustered", "testQuery10K.fbin"), "float", "L2", clu),
    ]


def is_complete(gt_output_dir, search_steps):
    return all(os.path.exists(os.path.join(gt_output_dir, f"step{step}.gt100")) for step in search_steps)


def is_nonempty_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def read_num_dim(path):
    with open(path, "rb") as f:
        header = f.read(8)
    if len(header) != 8:
        return None, None
    return struct.unpack("<II", header)


def generate_streaming_gt(gt_bin, base_file, query_file, gt_output_dir, dtype, metric, runbook_path, dataset_key):
    os.makedirs(gt_output_dir, exist_ok=True)
    cmd = [
        gt_bin,
        base_file,
        query_file,
        gt_output_dir,
        dtype,
        metric,
        "100",
        runbook_path,
        dataset_key,
    ]
    print("Generating streaming GT:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    with open(complete_marker(gt_output_dir), "w", encoding="utf-8") as f:
        f.write("complete\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    parser.add_argument("--gt-bin", required=True)
    args = parser.parse_args()

    for dataset_key, base_file, query_file, dtype, metric, runbook_path in generation_specs(args.root):
        if not os.path.exists(runbook_path) or os.path.getsize(runbook_path) == 0:
            print(f"Skipping {dataset_key} / {os.path.basename(runbook_path)}: runbook missing or empty.")
            continue
        gt_output_dir = gt_dir(args.root, dataset_key, runbook_path)
        search_steps = parse_search_steps(runbook_path, dataset_key)
        if os.path.exists(complete_marker(gt_output_dir)):
            print(f"Skipping {dataset_key} / {os.path.basename(runbook_path)}: GT already exists.")
            continue
        if is_complete(gt_output_dir, search_steps):
            os.makedirs(gt_output_dir, exist_ok=True)
            with open(complete_marker(gt_output_dir), "w", encoding="utf-8") as f:
                f.write("complete\n")
            print(f"Skipping {dataset_key} / {os.path.basename(runbook_path)}: GT already exists.")
            continue
        if not is_nonempty_file(base_file) or not is_nonempty_file(query_file):
            print(
                f"Skipping {dataset_key} / {os.path.basename(runbook_path)}: "
                "data files missing or empty."
            )
            continue
        base_num, _ = read_num_dim(base_file)
        if base_num is None:
            print(
                f"Skipping {dataset_key} / {os.path.basename(runbook_path)}: "
                "failed to read base header."
            )
            continue
        required_points = max_insert_end(runbook_path, dataset_key)
        if required_points > base_num:
            print(
                f"Skipping {dataset_key} / {os.path.basename(runbook_path)}: "
                f"runbook requires {required_points} points but base file has {base_num}."
            )
            continue
        generate_streaming_gt(
            args.gt_bin, base_file, query_file, gt_output_dir, dtype, metric, runbook_path, dataset_key
        )


if __name__ == "__main__":
    main()
