"""
CLARE Inference Computation Analysis
=====================================
Measures for the base policy and CLARE after each of tasks 0–9 (Libero Goal, seed 42):
  - Inference time: total, discriminator-only, policy-only (total − disc)
  - GPU memory footprint
  - Disk storage (base model + adapter)

Usage:
    conda run -n clare python data_scripts/computation_analysis.py
    conda run -n clare python data_scripts/computation_analysis.py --out data_scripts/computation_analysis.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PRETRAIN_PATH = Path("/home/ralf_roemer/projects/clare/outputs/dit_flow_mt_libero_90_pretrain_new")
OUTPUTS_DIR   = Path("/home/ralf_roemer/projects/clare_dev/outputs/libero_goal/clare")
RUN_PREFIX    = "dit_flow_mt_cl_seed_42_libero_goal"
TASK_SUFFIX   = "encoder_mlp_adapter_threshold_1_0"
N_TASKS       = 10
DEVICE        = "cuda"

# Timing
N_WARMUP = 10
N_REPS   = 50

# Batch shape (from train_config.json)
BATCH_SIZE   = 1
IMG_C, IMG_H, IMG_W = 3, 256, 256
STATE_DIM    = 8

# Discriminator input dimension (from adapter_config.json → feature_dim)
DISC_FEATURE_DIM = 2576


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def task_dir(task_id: int) -> Path:
    return OUTPUTS_DIR / f"{RUN_PREFIX}_task_{task_id}_{TASK_SUFFIX}"

def adapter_path(task_id: int) -> Path:
    return task_dir(task_id) / "checkpoints" / "last" / "adapter"

def pretrained_path(task_id: int) -> Path:
    return task_dir(task_id) / "checkpoints" / "last" / "pretrained_model"

def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_base_policy():
    from lerobot.policies.dit_flow_mt.modeling_dit_flow_mt import DiTFlowMTPolicy
    policy = DiTFlowMTPolicy.from_pretrained(PRETRAIN_PATH)
    return policy.to(DEVICE).eval()


class PeftWrapperPolicy(torch.nn.Module):
    """Thin wrapper so PEFT keys match the checkpoint prefix (base_model.model.policy.*)."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy


def load_clare_policy(task_id: int):
    from lerobot.policies.dit_flow_mt.modeling_dit_flow_mt import DiTFlowMTPolicy
    from peft import PeftModel
    # Use the original pretrain checkpoint (base weights only) for all tasks
    # to avoid loading accumulated adapter/disc weights from per-task checkpoints.
    policy = DiTFlowMTPolicy.from_pretrained(PRETRAIN_PATH)
    wrapper = PeftWrapperPolicy(policy=policy)
    peft_policy = PeftModel.from_pretrained(wrapper, adapter_path(task_id))
    return peft_policy.to(DEVICE).eval()


def get_inner_policy(peft_model):
    """Unwrap PeftModel → PeftWrapperPolicy → DiTFlowMTPolicy."""
    return peft_model.base_model.model.policy


def get_clare_modules(peft_model):
    """Return all CLARELayer instances (one per adapted linear layer)."""
    from peft.tuners.clare.layer import CLARELayer
    return [m for m in peft_model.modules() if isinstance(m, CLARELayer)]


# ─────────────────────────────────────────────────────────────────────────────
# Dummy batch
# ─────────────────────────────────────────────────────────────────────────────

def make_dummy_batch() -> dict:
    """
    Construct a single-step observation batch for select_action().
    Keys match the policy's image_features + state feature.
    Shape: (B, C, H, W) for images, (B, state_dim) for state.
    """
    return {
        "observation.images.image": torch.zeros(
            BATCH_SIZE, IMG_C, IMG_H, IMG_W, device=DEVICE,
        ),
        "observation.images.wrist_image": torch.zeros(
            BATCH_SIZE, IMG_C, IMG_H, IMG_W, device=DEVICE,
        ),
        "observation.state": torch.zeros(
            BATCH_SIZE, STATE_DIM, device=DEVICE,
        ),
        "task": ["pick and place the object"] * BATCH_SIZE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GPU timing via CUDA events
# ─────────────────────────────────────────────────────────────────────────────

def cuda_time_ms(fn, n_warmup: int = N_WARMUP, n_reps: int = N_REPS) -> float:
    """Time a callable with CUDA events, return mean ms (warmup excluded)."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.mean(times))


def time_discriminators_ms(clare_modules: list) -> float:
    """
    Time all discriminators across all CLARE adapter modules.
    Uses a dummy tensor at the discriminator's feature dimension.
    """
    x = torch.randn(BATCH_SIZE, DISC_FEATURE_DIM, device=DEVICE)

    @torch.no_grad()
    def run():
        for mod in clare_modules:
            for disc in mod.clare_discriminators[mod.adapter_name]:
                disc(x)

    return cuda_time_ms(run)


def time_select_action_ms(inner_policy, batch: dict) -> float:
    """Time a full select_action call (queue handling + flow matching ODE).

    We reset the policy each call so predict_action_chunk is always triggered
    (action queue is empty after reset).
    """
    @torch.no_grad()
    def run():
        inner_policy.reset()
        inner_policy.select_action(batch)

    return cuda_time_ms(run)


# ─────────────────────────────────────────────────────────────────────────────
# GPU memory
# ─────────────────────────────────────────────────────────────────────────────

def gpu_memory_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis() -> dict:
    results = {
        "labels":           [],
        "total_ms":         [],
        "policy_ms":        [],
        "discriminator_ms": [],
        "gpu_mb":           [],
        "model_mb":         [],
        "adapter_mb":       [],
    }

    batch = make_dummy_batch()

    # ── Base policy (no adapter, no discriminators) ──────────────────────────
    print("=== Base policy ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    policy  = load_base_policy()
    gpu_mb  = gpu_memory_mb()
    total_ms = time_select_action_ms(policy, batch)

    base_model_mb = dir_size_mb(PRETRAIN_PATH)
    print(f"  total={total_ms:.1f} ms | GPU={gpu_mb:.0f} MB | model={base_model_mb:.0f} MB")

    results["labels"].append("Base")
    results["total_ms"].append(total_ms)
    results["policy_ms"].append(total_ms)
    results["discriminator_ms"].append(0.0)
    results["gpu_mb"].append(gpu_mb)
    results["model_mb"].append(base_model_mb)
    results["adapter_mb"].append(0.0)

    del policy
    torch.cuda.empty_cache()

    # ── CLARE: one stage per task ─────────────────────────────────────────────
    for task_id in range(N_TASKS):
        print(f"=== CLARE task {task_id} ===")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        peft_model   = load_clare_policy(task_id)
        inner_policy = get_inner_policy(peft_model)
        clare_mods   = get_clare_modules(peft_model)
        n_disc       = sum(
            len(m.clare_discriminators[m.adapter_name]) for m in clare_mods
        )

        gpu_mb   = gpu_memory_mb()
        disc_ms  = time_discriminators_ms(clare_mods)
        total_ms = time_select_action_ms(inner_policy, batch)
        # Policy-only = total minus discriminator overhead
        policy_ms = max(total_ms - disc_ms, 0.0)

        adp_mb = dir_size_mb(adapter_path(task_id))

        print(
            f"  n_disc={n_disc} | total={total_ms:.1f} ms "
            f"(disc={disc_ms:.2f} ms, policy={policy_ms:.1f} ms) | "
            f"GPU={gpu_mb:.0f} MB | model={base_model_mb:.0f} MB, adapter={adp_mb:.0f} MB"
        )

        results["labels"].append(f"T{task_id}")
        results["total_ms"].append(total_ms)
        results["policy_ms"].append(policy_ms)
        results["discriminator_ms"].append(disc_ms)
        results["gpu_mb"].append(gpu_mb)
        results["model_mb"].append(base_model_mb)
        results["adapter_mb"].append(adp_mb)

        del peft_model, inner_policy
        torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_path: str) -> None:
    labels = results["labels"]
    xs     = np.arange(len(labels))

    policy_ms = np.array(results["policy_ms"])
    disc_ms   = np.array(results["discriminator_ms"])
    model_mb  = np.array(results["model_mb"])
    adapter_mb = np.array(results["adapter_mb"])
    gpu_mb    = np.array(results["gpu_mb"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "CLARE Inference Cost vs. Number of Tasks (Libero Goal, Seed 42)",
        fontsize=13, fontweight="bold",
    )

    # ── Inference time (stacked bar) ─────────────────────────────────────────
    ax = axes[0]
    ax.bar(xs, policy_ms, label="Policy (DiT + ODE)", color="steelblue")
    ax.bar(xs, disc_ms, bottom=policy_ms, label="Discriminators", color="coral")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Inference time (ms)")
    ax.set_title("Inference Time")
    ax.legend(loc="upper left")
    # Annotate discriminator time for T0 and T9
    for i in [1, len(labels) - 1]:
        ax.text(xs[i], policy_ms[i] + disc_ms[i] + 0.5,
                f"{disc_ms[i]:.2f}", ha="center", va="bottom", fontsize=8, color="coral")

    # ── Storage (stacked bar) ────────────────────────────────────────────────
    ax = axes[1]
    ax.bar(xs, model_mb, label="Base model", color="steelblue")
    ax.bar(xs, adapter_mb, bottom=model_mb, label="Adapter + discriminators", color="coral")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Disk size (MB)")
    ax.set_title("Storage")
    ax.legend(loc="upper left")
    # Annotate adapter size at T9
    i = len(labels) - 1
    ax.text(xs[i], model_mb[i] + adapter_mb[i] + 2,
            f"{adapter_mb[i]:.0f} MB", ha="center", va="bottom", fontsize=8, color="coral")

    # ── GPU memory (line) ────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(xs, gpu_mb, marker="o", color="steelblue", linewidth=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Stage")
    ax.set_ylabel("GPU memory allocated (MB)")
    ax.set_title("GPU Memory")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="data_scripts/computation_analysis.png",
        help="Output path for the plot (default: data_scripts/computation_analysis.png)",
    )
    args = parser.parse_args()

    results = run_analysis()
    plot_results(results, args.out)
