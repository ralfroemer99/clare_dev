"""
CLARE Inference Computation Analysis (v2) - Libero-40
=====================================================
Uses built-in CLARELayer timing hooks instead of external forward hooks.
Measures for the base policy and CLARE at stages 1, 5, 10, 15, …, 40
across the 40-task Libero-40 sequence (libero_10 → goal → spatial → object, seed 0).

Usage:
    conda run -n clare python data_scripts/computation_analysis_new.py
    conda run -n clare python data_scripts/computation_analysis_new.py --out data_scripts/computation_analysis.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PRETRAIN_PATH     = Path("/home/ralf_roemer/projects/clare/outputs/dit_flow_mt_libero_90_pretrain_new")
OUTPUTS_DIR       = Path("/home/ralf_roemer/projects/clare_dev/outputs/libero_40")
RUN_PREFIX        = "dit_flow_mt_cl_seed_0_libero_40"
TASK_SUFFIX       = "encoder_mlp_adapter_threshold_1_0"
DATASET_CACHE_DIR = Path.home() / ".cache/huggingface/lerobot/continuallearning"

LIBERO_40_SUITES = ["libero_10", "libero_goal", "libero_spatial", "libero_object"]
N_TASKS          = 40
TASKS_PER_SUITE  = 10

# Global task indices to measure (0-indexed): stages 1, 5, 10, 15, 20, 25, 30, 35, 40
DISPLAY_STAGES = [0, 4, 9, 14, 19, 24, 29, 34, 39]

DEVICE = "cuda"
N_WARMUP = 5
N_REPS   = 20

BATCH_SIZE          = 1
IMG_C, IMG_H, IMG_W = 3, 256, 256
STATE_DIM           = 8

DATASET_EPISODES_TOTAL = 50
DATASET_EPISODES_ER    = 5


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def global_to_suite_local(global_id: int) -> tuple[str, int]:
    suite = LIBERO_40_SUITES[global_id // TASKS_PER_SUITE]
    local = global_id % TASKS_PER_SUITE
    return suite, local


def task_dir(global_id: int) -> Path:
    suite, local = global_to_suite_local(global_id)
    return OUTPUTS_DIR / f"{RUN_PREFIX}_{suite}_task_{local}_{TASK_SUFFIX}"


def adapter_path(global_id: int) -> Path:
    return task_dir(global_id) / "checkpoints" / "last" / "adapter"


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def dataset_sizes_mb_40() -> list[float]:
    """Per-task dataset sizes in MB scaled to DATASET_EPISODES_ER episodes."""
    scale = DATASET_EPISODES_ER / DATASET_EPISODES_TOTAL
    sizes = []
    for suite in LIBERO_40_SUITES:
        for i in range(TASKS_PER_SUITE):
            p = DATASET_CACHE_DIR / f"{suite}_image_task_{i}"
            mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else 0.0
            sizes.append(mb * scale)
    return sizes


def adapter_split_mb(adapter_dir: Path) -> tuple[float, float]:
    """Return (lora_mb, disc_mb) by inspecting adapter_model.safetensors tensor names."""
    from safetensors import safe_open
    sf = adapter_dir / "adapter_model.safetensors"
    lora_bytes = disc_bytes = 0
    with safe_open(str(sf), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            nb = t.numel() * t.element_size()
            if "discriminator" in k:
                disc_bytes += nb
            else:
                lora_bytes += nb
    return lora_bytes / 1e6, disc_bytes / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_base_policy():
    from lerobot.policies.dit_flow_mt.modeling_dit_flow_mt import DiTFlowMTPolicy
    return DiTFlowMTPolicy.from_pretrained(PRETRAIN_PATH).to(DEVICE).eval()


class PeftWrapperPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy


def load_clare_policy(global_id: int):
    from lerobot.policies.dit_flow_mt.modeling_dit_flow_mt import DiTFlowMTPolicy
    from peft import PeftModel
    policy = DiTFlowMTPolicy.from_pretrained(PRETRAIN_PATH)
    wrapper = PeftWrapperPolicy(policy=policy)
    return PeftModel.from_pretrained(wrapper, adapter_path(global_id)).to(DEVICE).eval()


def get_inner_policy(peft_model):
    return peft_model.base_model.model.policy


def get_clare_modules(peft_model):
    from peft.tuners.clare.layer import CLARELayer
    return [m for m in peft_model.modules() if isinstance(m, CLARELayer)]


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers (using built-in CLARELayer timing hooks)
# ─────────────────────────────────────────────────────────────────────────────

def make_dummy_batch() -> dict:
    return {
        "observation.images.image":       torch.zeros(BATCH_SIZE, IMG_C, IMG_H, IMG_W, device=DEVICE),
        "observation.images.wrist_image": torch.zeros(BATCH_SIZE, IMG_C, IMG_H, IMG_W, device=DEVICE),
        "observation.state":              torch.zeros(BATCH_SIZE, STATE_DIM, device=DEVICE),
        "task": ["pick and place the object"] * BATCH_SIZE,
    }


def enable_clare_timing(clare_modules, enabled=True):
    """Enable or disable built-in CUDA event timing on all CLARELayer modules."""
    for mod in clare_modules:
        mod._timing_enabled = enabled


def time_select_action_ms(
    clare_modules, inner_policy, batch
) -> tuple[float, float, float, float]:
    """
    Time select_action using built-in CLARELayer timing hooks.
    Returns (total_mean_ms, total_std_ms, disc_mean_ms, adapter_mean_ms).
    """
    enable_clare_timing(clare_modules, True)

    total_times = []
    disc_times_per_rep = []
    adapter_times_per_rep = []

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    for rep in range(N_WARMUP + N_REPS):
        start.record()
        with torch.no_grad():
            inner_policy.reset()
            inner_policy.select_action(batch)
        end.record()
        torch.cuda.synchronize()

        if rep >= N_WARMUP:
            total_times.append(start.elapsed_time(end))

            rep_disc = sum(
                mod._info_dicts.get("disc_time_ms", 0.0)
                for mod in clare_modules
            )
            rep_adapter = sum(
                mod._info_dicts.get("adapter_time_ms", 0.0)
                for mod in clare_modules
            )
            disc_times_per_rep.append(rep_disc)
            adapter_times_per_rep.append(rep_adapter)

    enable_clare_timing(clare_modules, False)

    return (
        float(np.mean(total_times)),
        float(np.std(total_times)),
        float(np.mean(disc_times_per_rep)),
        float(np.mean(adapter_times_per_rep)),
    )


def gpu_memory_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis() -> dict:
    results = {
        "stage_ids":           [],   # global 0-indexed stage
        "labels":              [],   # display label (1, 5, 10, …)
        "total_ms":            [],
        "total_std":           [],
        "policy_ms":           [],
        "adapters_ms":         [],
        "disc_ms":             [],
        "backbone_vram_mb":    [],
        "adapters_vram_mb":    [],
        "disc_vram_mb":        [],
        "lora_disk_mb":        [],
        "disc_disk_mb":        [],
    }

    batch = make_dummy_batch()

    # ── Measure backbone VRAM once ───────────────────────────────────────────
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_policy = load_base_policy()
    backbone_vram = gpu_memory_mb()
    base_model_mb = dir_size_mb(PRETRAIN_PATH)
    print(f"Base policy: GPU={backbone_vram:.0f} MB | model={base_model_mb:.0f} MB")
    del base_policy
    torch.cuda.empty_cache()

    # ── CLARE stages ─────────────────────────────────────────────────────────
    for global_id in DISPLAY_STAGES:
        label = str(global_id + 1)
        adp = adapter_path(global_id)
        if not adp.exists():
            print(f"  SKIP stage {global_id + 1}: adapter not found at {adp}")
            continue

        print(f"=== CLARE stage {global_id + 1} (global task {global_id}) ===")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        peft_model   = load_clare_policy(global_id)
        inner_policy = get_inner_policy(peft_model)
        clare_mods   = get_clare_modules(peft_model)

        total_vram = gpu_memory_mb()

        total_ms, total_std, disc_ms, adapters_ms = time_select_action_ms(
            clare_mods, inner_policy, batch
        )
        policy_ms = max(total_ms - disc_ms - adapters_ms, 0.0)

        lora_mb, disc_disk = adapter_split_mb(adp)
        # Split VRAM overhead proportional to disk size
        overhead_vram = max(total_vram - backbone_vram, 0.0)
        total_disk = lora_mb + disc_disk
        adapters_vram = overhead_vram * (lora_mb / total_disk) if total_disk > 0 else 0.0
        disc_vram     = overhead_vram * (disc_disk / total_disk) if total_disk > 0 else 0.0

        print(
            f"  total={total_ms:.1f}±{total_std:.1f} ms (disc={disc_ms:.2f} ms, adapters={adapters_ms:.2f} ms, "
            f"policy={policy_ms:.1f} ms) | GPU={total_vram:.0f} MB | lora={lora_mb:.0f} MB, disc={disc_disk:.0f} MB"
        )

        results["stage_ids"].append(global_id)
        results["labels"].append(label)
        results["total_ms"].append(total_ms)
        results["total_std"].append(total_std)
        results["policy_ms"].append(policy_ms)
        results["adapters_ms"].append(adapters_ms)
        results["disc_ms"].append(disc_ms)
        results["backbone_vram_mb"].append(backbone_vram)
        results["adapters_vram_mb"].append(adapters_vram)
        results["disc_vram_mb"].append(disc_vram)
        results["lora_disk_mb"].append(lora_mb)
        results["disc_disk_mb"].append(disc_disk)

        del peft_model, inner_policy
        torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_path: str) -> None:
    labels   = results["labels"]
    xs       = np.arange(len(labels))

    policy_ms      = np.array(results["policy_ms"])
    disc_ms        = np.array(results["disc_ms"])
    total_std      = np.array(results["total_std"])
    backbone_vram  = np.array(results["backbone_vram_mb"])
    adapters_vram  = np.array(results["adapters_vram_mb"])
    disc_vram      = np.array(results["disc_vram_mb"])
    lora_disk      = np.array(results["lora_disk_mb"])
    disc_disk      = np.array(results["disc_disk_mb"])

    FS = 20
    plt.rcParams.update({
        "font.size":       FS,
        "axes.titlesize":  FS + 4,
        "axes.labelsize":  FS + 2,
        "xtick.labelsize": FS + 2,
        "ytick.labelsize": FS + 2,
        "legend.fontsize": FS,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # ── Plot 1: Inference time ────────────────────────────────────────────────
    ax = axes[0]
    adapters_ms    = np.array(results["adapters_ms"])
    ax.bar(xs, policy_ms, label="Base Policy", color="steelblue")
    ax.bar(xs, adapters_ms, bottom=policy_ms, label="Adapters", color="mediumseagreen")
    ax.bar(xs, disc_ms, bottom=policy_ms + adapters_ms, label="Discriminators", color="coral")
    ax.errorbar(xs, policy_ms + adapters_ms + disc_ms, yerr=total_std,
                fmt="none", color="black", capsize=4, linewidth=1.2)

    for i in [0, len(labels) - 1]:
        ax.text(xs[i], policy_ms[i] + adapters_ms[i] + disc_ms[i] + 0.5,
                f"{disc_ms[i]:.2f}", ha="center", va="bottom", fontsize=FS - 2, color="coral")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Inference time [ms]")
    ax.set_title("Inference Time")
    ax.legend(loc="lower left", framealpha=1, fontsize=FS - 2)

    # ── Plot 2: Disk storage ──────────────────────────────────────────────────
    ax = axes[1]
    base_model_mb = dir_size_mb(PRETRAIN_PATH)
    ax.bar(xs, np.full(len(xs), base_model_mb / 1e3), label="Base Policy", color="steelblue")
    ax.bar(xs, lora_disk / 1e3, bottom=base_model_mb / 1e3,
           label="Adapters", color="mediumseagreen")
    ax.bar(xs, disc_disk / 1e3, bottom=(base_model_mb + lora_disk) / 1e3,
           label="Discriminators", color="coral")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Stage")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_ylabel("VRAM [GB]")
    ax.set_title("GPU Memory")
    ax.legend(loc="lower left", framealpha=1, fontsize=FS - 2)

    # ── Plot 3: Cumulative disk usage (line plot) ─────────────────────────────
    ax = axes[2]

    # Adapter+disc cumulative at available stages
    stage_ids     = np.array(results["stage_ids"])
    xs_model      = stage_ids + 1          # display as 1-indexed
    cumul_model_gb = (lora_disk + disc_disk) / 1e3
    ax.plot(xs_model, cumul_model_gb, marker="o", color="mediumpurple",
            linewidth=2, label="Adapters + Discrim.")

    # Dataset cumulative (all 40 tasks on x = 1..40)
    all_dataset_mb = dataset_sizes_mb_40()
    xs_data = np.arange(1, N_TASKS + 1)
    cumul_data_gb = np.cumsum(all_dataset_mb) / 1e3
    ax.plot(xs_data, cumul_data_gb, marker="s", color="goldenrod",
            linewidth=2, label="Datasets")

    ax.set_xticks([1] + list(range(5, N_TASKS + 1, 5)))
    ax.set_xlim(0, N_TASKS + 1)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Cum. disk space [GB]")
    ax.set_title("Additional Disk Usage")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.85), framealpha=1, fontsize=FS - 2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default="figs/computation_analysis.png",
        help="Output path for the plot",
    )
    args = parser.parse_args()

    results = run_analysis()
    plot_results(results, args.out)
