"""
Single-purpose script:
Plots Accuracy (left y-axis) and Loss (right y-axis)
using EXACT colors from mergerd_plot.py.

Legend is bottom-centered (Plotter style).
Metadata parsed from experiment folder names (config.py compliant).
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

# -------------------------------------------------------
# EXACT color palette copied verbatim
# -------------------------------------------------------
COLORS = [
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.6, 0.0, 1.0, 1.0],
    [1.0, 0.498, 0.0549, 1.0],
    [0.0, 0.4, 0.4, 1.0],
    [0.549, 0.337, 0.294, 1.0],
    [0.890, 0.467, 0.761, 1.0],
    [0.0, 0.85, 0.0288, 1.0],
    [0.544, 0.0, 0.0, 1.0],
    [0.0, 0.74, 0.67, 1.0],
]

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
#LOG_DIR = "logs/new_compression"
#PLOT_DIR = "plots/new_compression"
LOG_DIR = "logs/mtgc_as_per_off_repo_v2"
PLOT_DIR = "plots/mtgc_as_per_off_repo_v2"
os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------------------------------------------
# Parse experiment name (matches config.py logic)
# -------------------------------------------------------
def parse_experiment_name(name):
    dataset, clients, model = name.split("-", 2)[0:3]
    clients = clients.replace("c", "")

    part_match = re.search(r"(dirichlet|iid|pathological)_?([\d\.]+)?", name)
    if part_match:
        p = part_match.group(1).capitalize()
        a = part_match.group(2)
        partitioner = f"{p}({a})" if a else p
    else:
        partitioner = "Unknown"

    strategy = next(
        (s.capitalize() for s in ["oldmyfedavg", "gfedavg", "cfedavg", "fedavg", "fedprox", "fedmut"] if s in name),
        "FedAvg"
    )

    if "compress_quant" in name:
        bits = re.search(r"quant_(\d+)bit", name).group(1)
        compression = f"Quant({bits}b)"
    elif "compress_topk" in name:
        pct = re.search(r"topk_(\d+)pct", name).group(1)
        compression = f"TopK({pct}%)"
    elif "compress_shap" in name:
        compression = "SHAP"
    else:
        compression = "None"

    gc = "GC" if name.endswith("-gc") else "No-GC"

    legend = f"{partitioner} | {strategy} | {compression} | {gc}"
    return dataset, clients, model, legend

# -------------------------------------------------------
# Plot
# -------------------------------------------------------
fig, ax_acc = plt.subplots(figsize=(10, 6))
ax_loss = ax_acc.twinx()

legend_handles = []

experiments = sorted(
    d for d in os.listdir(LOG_DIR)
    if os.path.isdir(os.path.join(LOG_DIR, d))
)

dataset_name = num_clients = model_name = None

for idx, exp in enumerate(experiments):
    log_path = os.path.join(LOG_DIR, exp, "central", "central_server.log")
    if not os.path.exists(log_path):
        continue

    df = pd.read_csv(log_path)
    if not {"round", "accuracy", "loss"}.issubset(df.columns):
        continue

    dataset_name, num_clients, model_name, label = parse_experiment_name(exp)
    color = COLORS[idx % len(COLORS)]

    # Accuracy (solid)
    ax_acc.plot(
        df["round"],
        df["accuracy"],
        color=color,
        linewidth=1.0,
        linestyle="-"
    )

    # Loss (dotted, same color)
    ax_loss.plot(
        df["round"],
        df["loss"],
        color=color,
        linewidth=0.5,
        linestyle=":",
        alpha=0.85
    )

    # Legend handle (one per experiment)
    legend_handles.append(
        Line2D([0], [0], color=color, lw=2, linestyle="-", label=label)
    )

# -------------------------------------------------------
# Styling (matched to Plotter)
# -------------------------------------------------------
ax_acc.set_xlabel("Rounds", fontsize=12)
ax_acc.set_ylabel("Accuracy", fontsize=12)
ax_loss.set_ylabel("Loss", fontsize=12)

ax_acc.set_ylim(0.0, 1.0)
ax_acc.xaxis.set_major_locator(MultipleLocator(5))
ax_acc.yaxis.set_major_locator(MultipleLocator(0.1))

ax_acc.grid(which="major", linestyle=":", linewidth=0.8, alpha=0.8)
ax_acc.minorticks_on()
ax_acc.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.3)

# -------------------------------------------------------
# Bottom legend (Plotter style)
# -------------------------------------------------------
fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=min(3, len(legend_handles)),
    fontsize=9,
    frameon=False
)

# -------------------------------------------------------
# Title & save
# -------------------------------------------------------
plt.suptitle(
    f"{dataset_name.upper()} | {num_clients} Clients | Model: {model_name.split('-')[0].upper()}",
    fontsize=16,
    fontweight="bold",
    y=0.96
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

out_file = f"{dataset_name}_{num_clients}c_accuracy_vs_loss.png"
plt.savefig(os.path.join(PLOT_DIR, out_file), dpi=300)
plt.close()

print(f"[Saved] {PLOT_DIR}/{out_file}")

