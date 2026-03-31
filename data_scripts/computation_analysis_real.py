"""
CLARE Real-World Inference Computation Analysis
================================================
Reads pre-computed profiling results from outputs/real/computation_times/*.md
and plots inference time and VRAM usage, analogous to computation_analysis.py.

Usage:
    conda run -n clare python data_scripts/computation_analysis_real.py
    conda run -n clare python data_scripts/computation_analysis_real.py --out figs/computation_analysis_real.png
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, list_repo_files, get_paths_info
from safetensors import safe_open

DATA_DIR = Path("outputs/real/computation_times")


# ─────────────────────────────────────────────────────────────────────────────
# MD file parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_md(path: Path) -> dict:
    """Parse key metrics from a profiling MD file."""
    text = path.read_text()

    def find_row(metric: str) -> tuple[float, float]:
        """Return (mean, std) for a timing row. Returns (0, 0) if not found."""
        pattern = rf"\|\s*{re.escape(metric)}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
        m = re.search(pattern, text)
        return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)

    def find_vram(label: str) -> float:
        """Return VRAM MB for a bold-label row like **Backbone**."""
        pattern = rf"\|\s*\*\*{re.escape(label)}\*\*\s*\|\s*([\d.]+)\s*\|"
        m = re.search(pattern, text)
        return float(m.group(1)) if m else 0.0

    def find_process_vram(label: str) -> float:
        pattern = rf"\|\s*{re.escape(label)}\s*\|\s*([\d.]+)\s*\|"
        m = re.search(pattern, text)
        return float(m.group(1)) if m else 0.0

    total_ms, total_std        = find_row("select_action_ms")
    disc_ms, _                 = find_row("all_discriminators_total_ms")
    adapters_ms, _             = find_row("all_adapters_total_ms")
    backbone_mb                = find_vram("Backbone")
    adapters_mb                = find_vram("All adapters")
    discriminators_mb          = find_vram("All discriminators")
    allocated_mb               = find_process_vram("Allocated")

    return {
        "total_ms":        total_ms,
        "total_std":       total_std,
        "disc_ms":         disc_ms,
        "adapters_ms":     adapters_ms,
        "policy_ms":       max(total_ms - disc_ms - adapters_ms, 0.0),
        "backbone_mb":     backbone_mb,
        "adapters_mb":     adapters_mb,
        "discriminators_mb": discriminators_mb,
        "allocated_mb":    allocated_mb,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

HF_DATASETS = [
    "continuallearning/real_0_put_bowl_filtered",
    "continuallearning/real_1_stack_bowls_filtered",
    "continuallearning/real_2_put_moka_pot_filtered",
    "continuallearning/real_3_close_drawer_filtered",
    "continuallearning/real_4_put_lego_into_drawer_filtered",
]

HF_REPOS = [
    "continuallearning/dit_posttrainv2_clare_dit_real_0_put_bowl_filtered_consolidated_seed1000",
    "continuallearning/dit_posttrainv2_clare_dit_real_1_stack_bowls_filtered_consolidated_seed1000",
    "continuallearning/dit_posttrainv2_clare_dit_real_2_put_moka_pot_filtered_consolidated_seed1000",
    "continuallearning/dit_posttrainv2_clare_dit_real_3_close_drawer_filtered_consolidated_seed1000",
    "continuallearning/dit_posttrainv2_clare_dit_real_4_put_lego_into_drawer_filtered_consolidated_seed1000",
]
BACKBONE_REPO = "continuallearning/dit_fft_pretraining_v2_lerobot30_seed1000"


def load_disk_sizes() -> dict:
    """Fetch disk sizes (MB) of backbone, adapters, discriminators from HuggingFace."""
    # Backbone: model.safetensors in pretrained repo
    backbone_path = hf_hub_download(BACKBONE_REPO, "model.safetensors")
    backbone_mb = Path(backbone_path).stat().st_size / 1e6
    print(f"Backbone disk: {backbone_mb:.1f} MB")

    adapters_disk_mb = []
    disc_disk_mb = []
    for i, repo in enumerate(HF_REPOS):
        path = hf_hub_download(repo, "adapter_model.safetensors")
        adapter_bytes = 0
        disc_bytes = 0
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                nbytes = f.get_tensor(key).numel() * f.get_tensor(key).element_size()
                if "clare_discriminators" in key:
                    disc_bytes += nbytes
                elif "clare_func_adapters" in key:
                    adapter_bytes += nbytes
        adapters_disk_mb.append(adapter_bytes / 1e6)
        disc_disk_mb.append(disc_bytes / 1e6)
        print(f"Stage {i}: adapters={adapter_bytes/1e6:.2f} MB, disc={disc_bytes/1e6:.2f} MB")

    # Dataset sizes (cumulative)
    dataset_sizes_mb = []
    for ds in HF_DATASETS:
        files = list(list_repo_files(ds, repo_type="dataset"))
        infos = list(get_paths_info(ds, files, repo_type="dataset"))
        total = sum((getattr(i.lfs, "size", None) or i.size) for i in infos if i.size)
        dataset_sizes_mb.append(total / 1e6)
        print(f"Dataset {ds.split('/')[-1]}: {total/1e6:.1f} MB")

    return {
        "backbone_mb":    backbone_mb,
        "adapters_mb":    np.array(adapters_disk_mb),
        "disc_mb":        np.array(disc_disk_mb),
        "datasets_mb":    np.array(dataset_sizes_mb),
    }


def load_data() -> dict:
    # CLARE stages 0–4
    clare_files = sorted(
        DATA_DIR.glob("*clare_real_*_on_real_0*.md"),
        key=lambda p: int(re.search(r"clare_real_(\d+)_on", p.name).group(1)),
    )
    clare = [parse_md(f) for f in clare_files]
    n_clare = len(clare)
    print(f"Loaded {n_clare} CLARE stages: {[f.name for f in clare_files]}")

    # Baselines
    baseline_names = {"er": "ER", "seqfft": "SeqFFT", "seqlora": "SeqLoRA"}
    baselines = {}
    for key, label in baseline_names.items():
        files = list(DATA_DIR.glob(f"*baseline*{key}*.md"))
        if files:
            baselines[label] = parse_md(files[0])
            print(f"Loaded baseline {label}: {files[0].name}")

    disk = load_disk_sizes()
    return {"clare": clare, "baselines": baselines, "disk": disk}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(data: dict, out_path: str) -> None:
    clare     = data["clare"]
    baselines = data["baselines"]
    disk      = data["disk"]
    n_clare   = len(clare)

    # Build arrays for CLARE stages (labelled 1..N)
    clare_labels  = [str(i + 1) for i in range(n_clare)]
    policy_ms     = np.array([s["policy_ms"]    for s in clare])
    disc_ms       = np.array([s["disc_ms"]      for s in clare])
    adapters_ms   = np.array([s["adapters_ms"]  for s in clare])
    total_std     = np.array([s["total_std"]    for s in clare])
    backbone_mb   = np.array([s["backbone_mb"]  for s in clare])
    adapters_mb   = np.array([s["adapters_mb"]  for s in clare])
    disc_vram_mb  = np.array([s["discriminators_mb"] for s in clare])

    # Baseline arrays
    bl_labels = list(baselines.keys())
    bl_total  = np.array([baselines[k]["total_ms"]     for k in bl_labels])
    bl_std    = np.array([baselines[k]["total_std"]    for k in bl_labels])
    bl_alloc  = np.array([baselines[k]["allocated_mb"] for k in bl_labels])

    FS = 20
    FS_ANNOT = 18
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
    # CLARE grouped bars (stacked policy + discriminators)
    xs_clare = np.arange(n_clare)
    policy_ms_plot = policy_ms * 4
    ax.bar(xs_clare, policy_ms_plot, label="Base Policy", color="steelblue")
    ax.bar(xs_clare, adapters_ms, bottom=policy_ms_plot, label="Adapters", color="mediumseagreen")
    ax.bar(xs_clare, disc_ms, bottom=policy_ms_plot + adapters_ms, label="Discriminators", color="coral")
    ax.errorbar(xs_clare, policy_ms_plot + adapters_ms + disc_ms, yerr=total_std / 4,
                fmt="none", color="black", capsize=4, linewidth=1.2)

    # Annotate discriminator ms on first and last CLARE bar
    for i in [0, n_clare - 1]:
        ax.text(xs_clare[i], policy_ms_plot[i] + adapters_ms[i] + disc_ms[i] + 0.5,
                f"{disc_ms[i]:.2f}", ha="center", va="bottom",
                fontsize=FS_ANNOT, color="coral")

    ax.set_xticks(xs_clare)
    ax.set_xticklabels(clare_labels)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Inference time [ms]")
    ax.set_title("Inference Time")
    ax.legend(loc="lower left", framealpha=1, fontsize=FS - 2)

    # ── Extrapolation helpers ─────────────────────────────────────────────────
    xs_extrap = np.arange(n_clare, n_clare + 5)
    all_labels = clare_labels + [str(i + 1) for i in range(n_clare, n_clare + 5)]
    xs_all = np.arange(n_clare + 5)

    def linear_extrap(xs, ys):
        p = np.polyfit(xs, ys, 1)
        return np.polyval(p, xs_extrap)

    # ── Plot 2: VRAM ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.bar(xs_clare, backbone_mb / 1e3,  label="Base Policy",       color="steelblue")
    ax.bar(xs_clare, adapters_mb / 1e3,
           bottom=backbone_mb / 1e3,
           label="Adapters", color="mediumseagreen")
    ax.bar(xs_clare, disc_vram_mb / 1e3,
           bottom=(backbone_mb + adapters_mb) / 1e3,
           label="Discriminators", color="coral")

    # Extrapolation: backbone constant, adapters/disc linearly extrapolated as hatched bars
    backbone_val_gb = backbone_mb[-1] / 1e3
    extrap_adapters_gb = linear_extrap(xs_clare, adapters_mb / 1e3)
    extrap_disc_gb     = linear_extrap(xs_clare, disc_vram_mb / 1e3)
    ax.bar(xs_extrap, np.full(5, backbone_val_gb), color="steelblue")
    ax.bar(xs_extrap, extrap_adapters_gb,
           bottom=backbone_val_gb,
           facecolor="white", edgecolor="mediumseagreen", linewidth=1.5, hatch="////")
    ax.bar(xs_extrap, extrap_disc_gb,
           bottom=backbone_val_gb + extrap_adapters_gb,
           facecolor="white", edgecolor="coral", linewidth=1.5, hatch="////")

    ax.set_xticks(xs_all)
    ax.set_xticklabels(all_labels)
    ax.set_xlabel("Stage")
    ax.set_ylabel("VRAM [GB]")
    ax.set_title("GPU Memory")
    ax.legend(loc="lower left", framealpha=1, fontsize=FS - 2)

    # ── Plot 3: Cumulative disk usage (line plot) ─────────────────────────────
    ax = axes[2]
    adapters_disk  = disk["adapters_mb"]
    disc_disk      = disk["disc_mb"]
    cumul_model_gb = (adapters_disk + disc_disk) / 1e3
    cumul_data_gb  = np.cumsum(disk["datasets_mb"]) / 1e3

    extrap_model = linear_extrap(xs_clare, cumul_model_gb)
    extrap_data  = linear_extrap(xs_clare, cumul_data_gb)

    ax.plot(xs_clare, cumul_model_gb, marker="o", color="mediumpurple",
            linewidth=2, label="Adapters + Discrim.")
    ax.plot(np.append(xs_clare[-1], xs_extrap),
            np.append(cumul_model_gb[-1], extrap_model),
            color="mediumpurple", linestyle="--", linewidth=1.5)

    ax.plot(xs_clare, cumul_data_gb, marker="s", color="goldenrod",
            linewidth=2, label="Datasets")
    ax.plot(np.append(xs_clare[-1], xs_extrap),
            np.append(cumul_data_gb[-1], extrap_data),
            color="goldenrod", linestyle="--", linewidth=1.5)

    ax.set_xticks(xs_all)
    ax.set_xticklabels(all_labels)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Disk space [GB]")
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
        "--out", default="figs/computation_analysis_real.png",
        help="Output path for the plot",
    )
    args = parser.parse_args()

    data = load_data()
    plot_results(data, args.out)
