import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Enable tex
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# Load results produced by sensitivity_agg.py
n_points = 100
sens_path = Path("sensitivity_res", f"sensitivity_agg_{n_points}.json")
with open(sens_path, "r") as f:
    results = json.load(f)

# Aggregate across all model × dataset pairs
# Entry format in JSON: results[m][ds][bucket] = [[[t2, t1], count], ...]
agg = defaultdict(lambda: {1: 0, 2: 0, 3: 0})

for m in results:
    for ds in results[m]:
        for bucket_str, entries in results[m][ds].items():
            bucket = int(bucket_str)
            for entry in entries:
                (t2, t1), count = entry
                key = (round(float(t1), 8), round(float(t2), 8))
                agg[key][bucket] += count

# Construct 2D squared grids (rows = t1, cols = t2)
all_t1 = sorted(set(k[0] for k in agg))
all_t2 = sorted(set(k[1] for k in agg))

t1_idx = {v: i for i, v in enumerate(all_t1)}
t2_idx = {v: i for i, v in enumerate(all_t2)}
n = len(all_t1)

raw_grids = {b: np.full((n, n), np.nan) for b in [1, 2, 3]}

for (t1, t2), counts in agg.items():
    i, j = t1_idx[t1], t2_idx[t2]
    for b in [1, 2, 3]:
        raw_grids[b][i, j] = counts[b]

# Normalize to proportions at each (t1, t2) cell
total = np.nansum([raw_grids[b] for b in [1, 2, 3]], axis=0)
total[total == 0] = np.nan
norm_grids = {b: raw_grids[b] / total for b in [1, 2, 3]}

# Dominant bucket at each cell
stacked = np.stack([norm_grids[b] for b in [1, 2, 3]], axis=0)
dominant = np.full((n, n), np.nan)
valid = ~np.all(np.isnan(stacked), axis=0)
dominant[valid] = np.nanargmax(stacked[:, valid], axis=0) + 1

# Marginal sensitivities (1D)
margin_t1 = {b: np.nanmean(norm_grids[b], axis=1) for b in [1, 2, 3]}
margin_t2 = {b: np.nanmean(norm_grids[b], axis=0) for b in [1, 2, 3]}

t1_arr = np.array(all_t1)
t2_arr = np.array(all_t2)

# Plot
BUCKET_LABELS = {1: "Aligned", 2: "Expanded", 3: "Divergent"}
ALIGNED_COLOR = "#4d8bff"
EXPANDED_COLOR = "#ffb84d"
DIVERGENT_COLOR = "#ff4d4d"
BUCKET_CMAPS = {
    1: LinearSegmentedColormap.from_list(
        "custom_blue_simple", ["white", ALIGNED_COLOR], N=256
    ),
    2: LinearSegmentedColormap.from_list(
        "custom_orange_simple", ["white", EXPANDED_COLOR], N=256
    ),
    3: LinearSegmentedColormap.from_list(
        "custom_red_simple", ["white", DIVERGENT_COLOR], N=256
    ),
}
BUCKET_COLORS = {1: ALIGNED_COLOR, 2: EXPANDED_COLOR, 3: DIVERGENT_COLOR}
MARKERS = {1: ".", 2: "2", 3: "1"}
PLOT_DIR = Path("sensitivity_charts")
PLOT_DIR.mkdir(exist_ok=True)

extent = [t2_arr[0], t2_arr[-1], t1_arr[0], t1_arr[-1]]
imshow_kwargs = dict(origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
imshow_kwargs_no_range = dict(origin="lower", aspect="auto", extent=extent)

# ==========================================
# 1. Individual Heatmaps (Normalized Grids)
# ==========================================
for b in [1, 2, 3]:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm_grids[b], cmap=BUCKET_CMAPS[b], **imshow_kwargs)
    ax.set_title(f"{BUCKET_LABELS[b]}", fontsize=14)
    ax.set_xlabel(r"$\tau_a$", fontsize=14)
    ax.set_ylabel(r"$\tau_e$", fontsize=14)
    ax.plot(
        [t2_arr[0], t2_arr[-1]],
        [t1_arr[0], t1_arr[-1]],
        "k--",
        lw=0.8,
        alpha=0.4,
        label="t1=t2",
    )
    plt.colorbar(im, ax=ax, label="Proportion")

    filename = f"heatmap_{BUCKET_LABELS[b].lower()}_{n_points}.pdf"
    plt.savefig(Path(PLOT_DIR, filename), bbox_inches="tight")
    plt.close(fig)

# ==========================================
# 2. Dominant Bucket Map
# ==========================================
fig, ax = plt.subplots(figsize=(6, 5))
cmap_dom = mcolors.ListedColormap([BUCKET_COLORS[b] for b in [1, 2, 3]])
bounds = [0.5, 1.5, 2.5, 3.5]
norm_dom = mcolors.BoundaryNorm(bounds, cmap_dom.N)
im_dom = ax.imshow(dominant, cmap=cmap_dom, norm=norm_dom, **imshow_kwargs_no_range)

ax.set_title("Dominant Behaviour Type", fontsize=14)
ax.set_xlabel(r"$\tau_a$", fontsize=14)
ax.set_ylabel(r"$\tau_e$", fontsize=14)
ax.plot([t2_arr[0], t2_arr[-1]], [t1_arr[0], t1_arr[-1]], "k--", lw=0.8, alpha=0.4)

cbar = plt.colorbar(im_dom, ax=ax, ticks=[1, 2, 3])
cbar.ax.set_yticklabels(["Aligned", "Expanded", "Divergent"])

filename = f"dominant_bucket_{n_points}.pdf"
plt.savefig(Path(PLOT_DIR, filename), bbox_inches="tight")
plt.close(fig)

# ==========================================
# 3. Marginal Sensitivities
# ==========================================
# Sweeping t1 (averaged over t2)
fig, ax_mt1 = plt.subplots(figsize=(7, 4))
for b in [1, 2, 3]:
    ax_mt1.plot(
        t1_arr,
        margin_t1[b],
        color=BUCKET_COLORS[b],
        lw=2,
        label=BUCKET_LABELS[b],
        marker=MARKERS[b],
        markersize=7,
    )
ax_mt1.set_xlabel(r"$\tau_a$  (averaged over $\tau_e$)", fontsize=14)
ax_mt1.set_ylabel("Mean proportion", fontsize=14)
# ax_mt1.set_title(r"Marginal sensitivity to $\tau_a$", fontsize=14)
ax_mt1.legend(fontsize=12, loc="upper right")
ax_mt1.grid(alpha=0.3)
ax_mt1.set_xlim(t1_arr[0], t1_arr[-1])
ax_mt1.set_ylim(0, 1)

filename = f"marginal_tau_a_{n_points}.pdf"
plt.savefig(Path(PLOT_DIR, filename), bbox_inches="tight")
plt.close(fig)

# Sweeping t2 (averaged over t1)
fig, ax_mt2 = plt.subplots(figsize=(7, 4))
for b in [1, 2, 3]:
    ax_mt2.plot(
        t2_arr,
        margin_t2[b],
        color=BUCKET_COLORS[b],
        lw=2,
        label=BUCKET_LABELS[b],
        marker=MARKERS[b],
        markersize=7,
    )
ax_mt2.set_xlabel(r"$\tau_e$  (averaged over $\tau_a$)", fontsize=14)
ax_mt2.set_ylabel("Mean proportion", fontsize=14)
# ax_mt2.set_title(r"Marginal sensitivity to $\tau_e$", fontsize=14)
ax_mt2.legend(fontsize=12, loc="upper right")
ax_mt2.grid(alpha=0.3)
ax_mt2.set_xlim(t2_arr[0], t2_arr[-1])
ax_mt2.set_ylim(0, 1)

filename = f"marginal_tau_e_{n_points}.pdf"
plt.savefig(Path(PLOT_DIR, filename), bbox_inches="tight")
plt.close(fig)

# ==========================================
# 4. Gradient Magnitudes (Instability Maps)
# ==========================================
for b in [1, 2, 3]:
    fig, ax = plt.subplots(figsize=(6, 5))
    g = norm_grids[b].copy()
    g[np.isnan(g)] = 0.0
    gy, gx = np.gradient(g)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag[np.isnan(norm_grids[b])] = np.nan

    im = ax.imshow(
        grad_mag, cmap="magma_r", origin="lower", aspect="auto", extent=extent
    )
    ax.set_title(f"Gradient magnitude — {BUCKET_LABELS[b]}", fontsize=12)
    ax.set_xlabel(r"$\tau_a$")
    ax.set_ylabel(r"$\tau_e$")
    ax.plot([t2_arr[0], t2_arr[-1]], [t1_arr[0], t1_arr[-1]], "w--", lw=0.8, alpha=0.5)
    plt.colorbar(im, ax=ax, label=r"$\left\lvert \nabla proportion \right\rvert$")

    filename = f"gradient_{BUCKET_LABELS[b].lower()}_{n_points}.pdf"
    plt.savefig(Path(PLOT_DIR, filename), bbox_inches="tight")
    plt.close(fig)

# ==========================================
# 5. Stability Summary Text
# ==========================================
combined_grad = np.zeros((n, n))
for b in [1, 2, 3]:
    g = norm_grids[b].copy()
    g[np.isnan(g)] = 0.0
    gy, gx = np.gradient(g)
    combined_grad += np.sqrt(gx**2 + gy**2)

combined_grad[np.isnan(norm_grids[1])] = np.nan
max_prop = np.nanmax(stacked, axis=0)
combined_grad[max_prop > 0.9] = np.nan
flat_idx = np.nanargmin(combined_grad)
best_i, best_j = np.unravel_index(flat_idx, (n, n))
best_t1, best_t2 = all_t1[best_i], all_t2[best_j]

summary_text = (
    "Most stable (t1, t2) pair\n"
    "(min total gradient magnitude)\n\n"
    rf"\noindent $\tau_e$ = {{ {best_t1:.6f} }}\\"
    rf"$\tau_a$ = {{ {best_t2:.6f} }}\\"
    f"Aligned: {norm_grids[1][best_i, best_j]:.2%}\n"
    f"Expanded: {norm_grids[2][best_i, best_j]:.2%}\n"
    f"Divergent: {norm_grids[3][best_i, best_j]:.2%}"
)

# Print it to console for quick referencing
print(summary_text.replace("\\", ""))

# Render it into a small isolated figure as well
fig, ax_stab = plt.subplots(figsize=(5, 3))
ax_stab.axis("off")
ax_stab.text(
    0.5,
    0.5,
    summary_text,
    transform=ax_stab.transAxes,
    fontsize=12,
    va="center",
    ha="center",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
)

filename = f"stability_summary_{n_points}.pdf"
plt.savefig(Path(PLOT_DIR, filename), bbox_inches="tight")
plt.close(fig)
