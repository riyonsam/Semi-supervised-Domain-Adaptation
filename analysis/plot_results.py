"""
plot_results.py -  Parse SLURM log files and generate result plots.

Usage:
    python plot_results.py                          # auto-find slurm-*.out files
    python plot_results.py slurm-2217459.out slurm-fixmatch-2217460.out
"""

import re
import sys
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─── Parse log file ───────────────────────────────────────────────────────────
def parse_log(path):
    """Extract epoch, loss, acc, best_acc from a SLURM log file."""
    epochs, losses, accs, bests = [], [], [], []
    per_class_data = {}

    epoch_re  = re.compile(
        r"Epoch \[\s*(\d+)/\d+\].*Loss:\s*([\d.]+).*Acc:\s*([\d.]+)%.*Best:\s*([\d.]+)%"
    )
    perclass_re = re.compile(r"Per-class acc:\s*\[(.+)\]")

    with open(path) as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                ep, loss, acc, best = m.groups()
                epochs.append(int(ep))
                losses.append(float(loss))
                accs.append(float(acc))
                bests.append(float(best))

            m2 = perclass_re.search(line)
            if m2 and epochs:
                vals = [float(x.strip().strip("'")) for x in m2.group(1).split(",")]
                per_class_data[epochs[-1]] = vals

    return {
        "epochs": epochs,
        "losses": losses,
        "accs":   accs,
        "bests":  bests,
        "per_class": per_class_data,
        "label":  os.path.basename(path),
        "best_acc": max(accs) if accs else 0.0,
        "final_acc": accs[-1] if accs else 0.0,
    }


# ─── Detect label from filename or log content ────────────────────────────────
def nice_label(path):
    # First try to extract strategy from the log's first line
    # e.g. "=== Job 2218293: strategy=random  budget=0.01 ==="
    try:
        with open(path) as f:
            for line in f:
                m = re.search(r"strategy=(\w+)", line)
                if m:
                    strategy = m.group(1)
                    budget_m = re.search(r"budget=([\d.]+)", line)
                    budget = budget_m.group(1) if budget_m else ""
                    alloc_m = re.search(r"allocation=(\w+)", line)
                    alloc = alloc_m.group(1) if alloc_m else ""
                    label = strategy
                    if budget:
                        label += f" (b={budget})"
                    if alloc and alloc != "proportional":
                        label += f" [{alloc}]"
                    return label
                break  # only check first line
    except Exception:
        pass
    # Fall back to filename
    name = os.path.basename(path).lower()
    if "fixmatch" in name:
        return "FixMatch"
    return os.path.splitext(os.path.basename(path))[0]


# ─── Collect log files ────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    log_files = sys.argv[1:]
else:
    log_files = sorted(glob.glob("slurm-*.out") + glob.glob("slurm-fixmatch-*.out"))
    if not log_files:
        print("No slurm-*.out files found. Pass log file paths as arguments.")
        sys.exit(1)

runs = []
for path in log_files:
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping.")
        continue
    data = parse_log(path)
    data["label"] = nice_label(path)
    runs.append(data)

if not runs:
    print("No valid log files parsed.")
    sys.exit(1)

# Print summary table
print(f"\n{'Run':<15} {'Best Acc':>10} {'Final Acc':>10} {'Epochs':>8}")
print("-" * 48)
for r in runs:
    print(f"{r['label']:<15} {r['best_acc']:>9.2f}% {r['final_acc']:>9.2f}% {len(r['epochs']):>8}")


# ─── Colours ──────────────────────────────────────────────────────────────────
COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("SSDA Training Results – VisDA (Synthetic → Real)", fontsize=14, fontweight="bold")

# ── Plot 1: Accuracy over epochs ─────────────────────────────────────────────
ax = axes[0]
for i, r in enumerate(runs):
    c = COLORS[i % len(COLORS)]
    ax.plot(r["epochs"], r["accs"],  color=c, linewidth=2, label=f"{r['label']} (acc)")
    ax.plot(r["epochs"], r["bests"], color=c, linewidth=1.5, linestyle="--", alpha=0.6,
            label=f"{r['label']} (best)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy vs Epoch")
ax.legend(fontsize=8)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)

# ── Plot 2: Loss over epochs ──────────────────────────────────────────────────
ax = axes[1]
for i, r in enumerate(runs):
    c = COLORS[i % len(COLORS)]
    ax.plot(r["epochs"], r["losses"], color=c, linewidth=2, label=r["label"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss vs Epoch")
ax.legend(fontsize=8)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, which="major", alpha=0.3)
ax.grid(True, which="minor", alpha=0.1)

# ── Plot 3: Best accuracy bar chart ──────────────────────────────────────────
ax = axes[2]
labels = [r["label"] for r in runs]
best_accs = [r["best_acc"] for r in runs]
bars = ax.bar(labels, best_accs, color=COLORS[:len(runs)], width=0.4, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, best_accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Best Accuracy (%)")
ax.set_title("Best Accuracy Comparison")
ax.set_ylim(0, min(100, max(best_accs) + 10))
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_path = os.environ.get("PLOT_OUT", "results_summary.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved plot → {out_path}")
plt.show()

# ── Per-class accuracy (if available) ────────────────────────────────────────
CLASS_NAMES = ["aeroplane", "bicycle", "bus", "car", "horse", "knife",
               "motorcycle", "person", "plant", "skateboard", "train", "truck"]

any_perclass = any(r["per_class"] for r in runs)
if any_perclass:
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / max(len(runs), 1)

    for i, r in enumerate(runs):
        if not r["per_class"]:
            continue
        last_ep = max(r["per_class"].keys())
        vals = r["per_class"][last_ep]
        offset = (i - len(runs) / 2 + 0.5) * width
        bars = ax2.bar(x + offset, vals, width=width * 0.9,
                       label=f"{r['label']} (ep{last_ep})", color=COLORS[i % len(COLORS)], alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax2.set_ylabel("Per-class Accuracy (%)")
    ax2.set_title("Per-class Accuracy (last logged epoch)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 105)
    plt.tight_layout()
    out2 = os.environ.get("PLOT_OUT_PERCLASS", "results_perclass.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved per-class plot → {out2}")
    plt.show()
