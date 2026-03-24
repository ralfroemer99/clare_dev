"""
Compute continual learning metrics from local WandB summary files.

Reads outputs/<run_dir>/wandb/run-*/files/wandb-summary.json for each task stage,
builds the performance matrix R[stage][task], then computes AA, AUC, BWT, FWT.

Usage:
    python data_scripts/compute_cl_metrics.py \
        --outputs_dir ./outputs \
        --run_prefix dit_flow_mt_cl_seed_42_libero_spatial \
        --suite libero_spatial \
        --n_tasks 10

    # Verbose: also print the full R matrix
    python data_scripts/compute_cl_metrics.py ... --verbose

    # Compare multiple experiments:
    python data_scripts/compute_cl_metrics.py \
        --outputs_dir ./outputs \
        --run_prefix "dit_flow_mt_cl_seed_42_libero_spatial,dit_flow_mt_cl_seed_42_libero_10" \
        --suite "libero_spatial,libero_10" \
        --n_tasks 10
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load_summary(run_dir: Path) -> dict:
    """Load wandb-summary.json from a run output directory."""
    matches = list(run_dir.glob("wandb/run-*/files/wandb-summary.json"))
    if not matches:
        raise FileNotFoundError(f"No wandb-summary.json found in {run_dir}")
    # Take the most recent run if multiple exist
    return json.loads(sorted(matches)[-1].read_text())


def fetch_performance_matrix(outputs_dir, run_prefix, suite, n_tasks, verbose=False):
    """
    Returns R: numpy array of shape (n_tasks, n_tasks) where
        R[j][i] = avg_sum_reward on task i after training through stage j.
    Entries where j < i are NaN (task i not yet seen at that stage).
    """
    outputs_dir = Path(outputs_dir)

    # WandB key pattern: eval/avg_sum_reward_Libero_Spatial_Task_0
    suite_pascal = "_".join(w.capitalize() for w in suite.split("_"))  # libero_spatial -> Libero_Spatial
    task_key_template = f"eval/avg_sum_reward_{suite_pascal}_Task_{{i}}"

    R = np.full((n_tasks, n_tasks), np.nan)

    for stage in range(n_tasks):
        # Find matching output directory for this stage
        pattern = f"{run_prefix}_task_{stage}_*"
        candidates = sorted(outputs_dir.glob(pattern))
        if not candidates:
            # Also try exact match (no suffix after task number)
            exact = outputs_dir / f"{run_prefix}_task_{stage}"
            if exact.is_dir():
                candidates = [exact]
        if not candidates:
            print(f"  WARNING: no directory found matching '{pattern}' — stage {stage} skipped")
            continue
        run_dir = candidates[0]

        try:
            summary = load_summary(run_dir)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — stage {stage} skipped")
            continue

        for i in range(stage + 1):
            key = task_key_template.format(i=i)
            if key in summary:
                R[stage][i] = summary[key]
            else:
                print(f"  WARNING: key '{key}' not found in {run_dir.name}")

    if verbose:
        print(f"\n  Performance matrix R[stage][task] for '{run_prefix}':")
        header = "        " + "  ".join(f"T{i:02d} " for i in range(n_tasks))
        print(header)
        for j in range(n_tasks):
            if np.all(np.isnan(R[j])):
                continue
            row = f"  S{j:02d} |  "
            for i in range(n_tasks):
                if np.isnan(R[j][i]):
                    row += "  -- "
                else:
                    row += f"{R[j][i]:.2f} "
            print(row)

    return R


def compute_cl_metrics(R, n_tasks=None):
    """
    Metrics as defined in arXiv:2601.09512.
    R[m][n] = r(n+1 | m+1) = success rate on task n after learning m+1 tasks (0-indexed).
    NaN where m < n (task n not yet seen at stage m).

    FWT = (1/N) * sum_{n=1}^{N} r(n|n)
        = mean of diagonal R[i,i]
        Average performance right after learning each task.

    AUC = (1/N) * sum_{n=1}^{N} [ (1/(N-n+1)) * sum_{m=n}^{N} r(n|m) ]
        = for each task i, average R[j,i] over all j >= i, then mean over tasks.
        Overall performance across all stages after each task is introduced.

    NBT = (1/(N-1)) * sum_{n=1}^{N-1} [ (1/(N-n)) * sum_{m=n+1}^{N} (r(n|n) - r(n|m)) ]
        = for each task i (except last), average (R[i,i] - R[j,i]) over j > i, then mean.
        Average forgetting across tasks and subsequent stages. Lower = less forgetting.
    """
    if n_tasks is None:
        n_tasks = R.shape[0]

    last_stage = max(j for j in range(n_tasks) if not np.all(np.isnan(R[j])))

    # FWT: mean of diagonal r(n|n)
    fwt_terms = [R[i, i] for i in range(n_tasks) if not np.isnan(R[i, i])]
    fwt = float(np.mean(fwt_terms)) if fwt_terms else float("nan")

    # AUC: for each task i, average r(n|m) over all stages m >= i, then mean over tasks
    auc_per_task = []
    for i in range(n_tasks):
        vals = [R[j, i] for j in range(i, n_tasks) if not np.isnan(R[j, i])]
        if vals:
            auc_per_task.append(np.mean(vals))
    auc = float(np.mean(auc_per_task)) if auc_per_task else float("nan")

    # NBT: for each task i (except last), average (R[i,i] - R[j,i]) over j > i, then mean
    nbt_per_task = []
    for i in range(n_tasks - 1):
        if np.isnan(R[i, i]):
            continue
        forget_vals = [R[i, i] - R[j, i] for j in range(i + 1, n_tasks) if not np.isnan(R[j, i])]
        if forget_vals:
            nbt_per_task.append(np.mean(forget_vals))
    nbt = float(np.mean(nbt_per_task)) if nbt_per_task else float("nan")

    diag  = [R[i, i] for i in range(n_tasks)]
    final = list(R[last_stage, :n_tasks])

    return {"FWT": fwt, "AUC": auc, "NBT": nbt, "diag": diag, "final": final}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="./outputs")
    parser.add_argument("--run_prefix",  required=True,
                        help="Comma-separated list of run prefixes (one per experiment). "
                             "Use {seed} placeholder when --seeds is specified, e.g. 'er_seed_{seed}_libero_goal'")
    parser.add_argument("--suite",       required=True,
                        help="Comma-separated suite names matching run_prefix order")
    parser.add_argument("--n_tasks",     type=int, default=10)
    parser.add_argument("--seeds",       default=None,
                        help="Comma-separated list of seeds, e.g. '0,1,42'. "
                             "If set, run_prefix must contain {seed} placeholder. "
                             "Reports mean ± std across seeds.")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    prefixes = [p.strip() for p in args.run_prefix.split(",")]
    suites   = [s.strip() for s in args.suite.split(",")]
    assert len(prefixes) == len(suites), "--run_prefix and --suite must have the same number of entries"

    for prefix_template, suite in zip(prefixes, suites):
        print(f"\n{'='*60}")
        print(f"Experiment: {prefix_template}  |  suite: {suite}")
        print(f"{'='*60}")

        if args.seeds is not None:
            seeds = [s.strip() for s in args.seeds.split(",")]
            all_metrics = {"FWT": [], "AUC": [], "NBT": []}
            for seed in seeds:
                prefix = prefix_template.replace("{seed}", seed)
                print(f"\n  Seed {seed}: {prefix}")
                R = fetch_performance_matrix(args.outputs_dir, prefix, suite, args.n_tasks, verbose=args.verbose)
                m = compute_cl_metrics(R, n_tasks=args.n_tasks)
                print(f"    FWT={m['FWT']*100:.2f}%  AUC={m['AUC']*100:.2f}%  NBT={m['NBT']*100:.2f}%")
                for k in all_metrics:
                    all_metrics[k].append(m[k])

            print(f"\n  --- Mean ± Std across seeds {seeds} ---")
            for k in ["AUC", "FWT", "NBT"]:
                vals = np.array(all_metrics[k])
                mean, std = np.nanmean(vals), np.nanstd(vals)
                print(f"  {k:4s}: {mean*100:.2f}% ± {std*100:.2f}%")
        else:
            R = fetch_performance_matrix(args.outputs_dir, prefix_template, suite, args.n_tasks, verbose=args.verbose)
            m = compute_cl_metrics(R, n_tasks=args.n_tasks)
            print(f"\n  FWT (Forward Transfer):        {m['FWT']:.4f}  ({m['FWT']*100:.1f}%)  — avg success right after learning each task")
            print(f"  AUC (Area Under Curve):        {m['AUC']:.4f}  ({m['AUC']*100:.1f}%)  — avg success across all stages")
            print(f"  NBT (Neg. Backward Transfer):  {m['NBT']:.4f}  ({m['NBT']*100:.1f}%)  — avg forgetting (lower = better)")

        # print(f"\n  Diagonal R[i,i] — peak performance right after training each task:")
        # for i, v in enumerate(m["diag"]):
        #     print(f"    Task {i}: {v:.4f}" if not np.isnan(v) else f"    Task {i}: --")

        # print(f"\n  Final R[last,i] — performance on each task at end of experiment:")
        # for i, v in enumerate(m["final"]):
        #     print(f"    Task {i}: {v:.4f}" if not np.isnan(v) else f"    Task {i}: --")


if __name__ == "__main__":
    main()
