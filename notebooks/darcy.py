# %% Imports & config
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

REPO_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
DATA_DIR = REPO_ROOT / "data/Darcy_421"
FIGURES_DIR = REPO_ROOT / "docs/figures/darcy"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CMAP_INPUT = "viridis"
CMAP_OUTPUT = "magma"
CMAP_VECTOR = "spring"

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

# %% Load dataset
raw = scio.loadmat(
    str(DATA_DIR / "piececonst_r421_N1024_smooth1.mat"), variable_names=("coeff", "sol")
)
coeff = raw["coeff"]  # (1024, 421, 421) — permeability field a(x)
sol = raw["sol"]  # (1024, 421, 421) — pressure solution u(x)

print(f"coeff: {coeff.shape}  sol: {sol.shape}")

# %% Dataset sample — input permeability and output pressure
SAMPLE_IDX = 0

# Downsample for display (421 → 85 at stride 5)
STRIDE = 5
a = coeff[SAMPLE_IDX, ::STRIDE, ::STRIDE]
u = sol[SAMPLE_IDX, ::STRIDE, ::STRIDE]

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)

im0 = axes[0].imshow(a, cmap=CMAP_INPUT, origin="lower")
axes[0].set_title("Input: permeability $a(\\mathbf{x})$", fontsize=11)
axes[0].axis("off")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(u, cmap=CMAP_OUTPUT, origin="lower")
axes[1].set_title("Output: pressure $u(\\mathbf{x})$", fontsize=11)
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

fig.savefig(FIGURES_DIR / "darcy_dataset_sample.png", bbox_inches="tight")
plt.show()
print(f"Saved to {FIGURES_DIR / 'darcy_dataset_sample.png'}")


# %% Boundary condition — zero Dirichlet u = 0 on all edges
# Build a mask that is True on the 4 boundary edges
def boundary_mask(shape):
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


SAMPLES = [0, 10, 100]
mask = boundary_mask(u.shape)

fig, axes = plt.subplots(
    2, len(SAMPLES), figsize=(len(SAMPLES) * 3.5, 6), constrained_layout=True
)

for col, idx in enumerate(SAMPLES):
    u_s = sol[idx, ::STRIDE, ::STRIDE]

    # boundary residual: |u| on boundary, NaN elsewhere
    boundary_res = np.full_like(u_s, np.nan, dtype=np.float64)
    boundary_res[mask] = np.abs(u_s[mask])

    im0 = axes[0, col].imshow(
        u_s, cmap=CMAP_OUTPUT, origin="lower", extent=(0, 1, 0, 1)
    )
    axes[0, col].set_title(f"sample {idx}", fontsize=10)
    axes[0, col].axis("off")
    fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

    im1 = axes[1, col].imshow(
        boundary_res, cmap="Reds", origin="lower", extent=(0, 1, 0, 1)
    )
    axes[1, col].axis("off")
    fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

axes[0, 0].set_ylabel("$u(\\mathbf{x})$", fontsize=10)
axes[1, 0].set_ylabel("$|u|$ on boundary", fontsize=10)

# Print max boundary error across summary samples
n_summary = 200
max_errs = [np.abs(sol[i, ::STRIDE, ::STRIDE][mask]).max() for i in range(n_summary)]
print(
    f"Ground truth max boundary error over {n_summary} samples: "
    f"mean={np.mean(max_errs):.2e}  p95={np.percentile(max_errs, 95):.2e}  max={np.max(max_errs):.2e}"
)

fig.savefig(FIGURES_DIR / "darcy_boundary_condition.png", bbox_inches="tight")
plt.show()
print(f"Saved to {FIGURES_DIR / 'darcy_boundary_condition.png'}")

# %% Boundary condition statistics — is u ≈ 0 on the boundary?
N = coeff.shape[0]
sol_ds = sol[:, ::STRIDE, ::STRIDE]  # (N, 85, 85)

bnd_mask = boundary_mask(sol_ds.shape[1:])  # reuse mask defined above

# Per-sample boundary statistics
bnd_vals = sol_ds[:, bnd_mask]  # (N, n_boundary_pixels)
bnd_mean_abs = np.abs(bnd_vals).mean(axis=1)  # mean |u| on boundary
bnd_max_abs = np.abs(bnd_vals).max(axis=1)  # max |u| on boundary
bnd_std = bnd_vals.std(axis=1)  # std of u on boundary

print("--- Boundary u statistics (over all samples) ---")
print(
    f"  mean |u| on bnd:  mean={bnd_mean_abs.mean():.4e}  p95={np.percentile(bnd_mean_abs, 95):.4e}"
)
print(
    f"  max  |u| on bnd:  mean={bnd_max_abs.mean():.4e}   p95={np.percentile(bnd_max_abs, 95):.4e}"
)
print(
    f"  std  u   on bnd:  mean={bnd_std.mean():.4e}       p95={np.percentile(bnd_std, 95):.4e}"
)

c0 = plt.cm.magma(0.35)
c1 = plt.cm.magma(0.60)
c2 = plt.cm.magma(0.82)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

axes[0].hist(
    bnd_mean_abs, bins=40, color=c0, alpha=0.9, edgecolor="white", linewidth=0.3
)
axes[0].set_xlabel("Mean $|u|$ on boundary")
axes[0].set_ylabel("Count")
axes[0].set_title("Per-sample mean $|u|$ on $\\partial\\Omega$")

axes[1].hist(
    bnd_max_abs, bins=40, color=c1, alpha=0.9, edgecolor="white", linewidth=0.3
)
axes[1].set_xlabel("Max $|u|$ on boundary")
axes[1].set_ylabel("Count")
axes[1].set_title("Per-sample max $|u|$ on $\\partial\\Omega$")

axes[2].hist(bnd_std, bins=40, color=c2, alpha=0.9, edgecolor="white", linewidth=0.3)
axes[2].set_xlabel("Std of $u$ on boundary")
axes[2].set_ylabel("Count")
axes[2].set_title("Per-sample std of $u$ on $\\partial\\Omega$")

for ax in axes:
    ax.axvline(0, color="0.3", linewidth=1.0, linestyle="--", label="$u=0$")
axes[0].legend(fontsize=8, frameon=False)

fig.savefig(FIGURES_DIR / "darcy_dataset_statistics.png", bbox_inches="tight")
plt.show()
print(f"Saved to {FIGURES_DIR / 'darcy_dataset_statistics.png'}")

# %% Boundary statistics — LaTeX booktabs table
# Same numbers as the histograms above, summarised across all N samples. The
# final column expresses each boundary stat as a fraction of the interior RMS,
# making the "u = 0 on the boundary" claim quantitative.
interior_rms = float(np.sqrt((sol_ds[:, 1:-1, 1:-1] ** 2).mean()))

table_rows = [
    (r"mean $|u|$ on $\partial\Omega$", bnd_mean_abs, True),
    (r"max $|u|$ on $\partial\Omega$", bnd_max_abs, True),
    (r"std $u$ on $\partial\Omega$", bnd_std, False),
]


def _sci(x):
    return f"\\num{{{x:.2e}}}"


lines = [
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Quantity & Mean & p95 & Max & vs.\ interior \\",
    r"\midrule",
]
for label, vals, rel in table_rows:
    mean, p95, mx = vals.mean(), np.percentile(vals, 95), vals.max()
    rel_col = f"{mean / interior_rms:.2%}".replace("%", r"\%") if rel else "--"
    lines.append(
        f"{label} & {_sci(mean)} & {_sci(p95)} & {_sci(mx)} & {rel_col} \\\\"
    )
lines += [
    r"\midrule",
    f"\\multicolumn{{5}}{{l}}{{Interior RMS $|u| = {_sci(interior_rms)}$}} \\\\",
    r"\bottomrule",
    r"\end{tabular}",
]
latex_table = "\n".join(lines)

print(f"--- Boundary statistics table (N = {N} samples) ---")
print(latex_table)

table_path = FIGURES_DIR / "darcy_dataset_statistics_table.tex"
table_path.write_text(latex_table + "\n")
print(f"Saved to {table_path}")

# %% Boundary profile — u along each edge
# Extract the 4 edge profiles for multiple samples and plot as 1D curves.
# If the shape is consistent across samples it's systematic (solver property),
# if it varies randomly it's noise.
N_PROFILE_SAMPLES = 1000


xs = np.linspace(0, 1, sol_ds.shape[2])  # x-axis: normalised position along edge
ys = np.linspace(0, 1, sol_ds.shape[1])

edge_defs = {
    "Bottom  ($y=0$)": lambda u: u[0, :],
    "Top  ($y=1$)": lambda u: u[-1, :],
    "Left  ($x=0$)": lambda u: u[:, 0],
    "Right  ($x=1$)": lambda u: u[:, -1],
}
edge_xs = {
    "Bottom  ($y=0$)": xs,
    "Top  ($y=1$)": xs,
    "Left  ($x=0$)": ys,
    "Right  ($x=1$)": ys,
}

band_color = plt.cm.magma(0.35)
mean_color = plt.cm.magma(0.75)

fig, axes = plt.subplots(1, 4, figsize=(14, 3), constrained_layout=True)
for ax, (label, extractor) in zip(axes, edge_defs.items()):
    pos = edge_xs[label]
    profiles = np.stack([extractor(sol_ds[i]) for i in range(N_PROFILE_SAMPLES)])
    p5, p25, p75, p95 = (np.percentile(profiles, q, axis=0) for q in (5, 25, 75, 95))
    mean = profiles.mean(axis=0)
    ax.fill_between(
        pos, p5, p95, color=band_color, alpha=0.18, linewidth=0, label="5–95%"
    )
    ax.fill_between(
        pos, p25, p75, color=band_color, alpha=0.38, linewidth=0, label="25–75%"
    )
    ax.plot(pos, mean, color=mean_color, linewidth=1.8, label="mean")
    ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Position")
    ax.set_ylabel("$u$")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

axes[0].legend(fontsize=8, frameon=False)

fig.savefig(
    FIGURES_DIR / f"{N_PROFILE_SAMPLES}_darcy_boundary_profiles.png",
    bbox_inches="tight",
)
plt.show()
print(f"Saved to {FIGURES_DIR / f'{N_PROFILE_SAMPLES}_darcy_boundary_profiles.png'}")

# %% Boundary profiles in 2D context — one sample, field + edge profiles
# Show a single sample's pressure field inside the unit square, with that same
# sample's boundary profile drawn outside each edge. Each profile sits on a
# mid-margin baseline (dashed) and swings outward for u > 0; amplitude is scaled
# (actual values are O(1e-5)).
SAMPLE_2D = 100
sample_u = sol_ds[SAMPLE_2D]

M = 0.22  # margin around the field square
sq_x = np.linspace(M, 1 - M, sample_u.shape[1])  # positions along horizontal edges
sq_y = np.linspace(M, 1 - M, sample_u.shape[0])  # positions along vertical edges

edge_profiles = {
    "bottom": sample_u[0, :],
    "top": sample_u[-1, :],
    "left": sample_u[:, 0],
    "right": sample_u[:, -1],
}

# Symmetric amplitude scale: max |profile| maps to AMP within the margin.
AMP = 0.58 * M
GAP = 0.3 * M  # baseline sits mid-margin so the curve can swing both ways
_peak = max(np.abs(p).max() for p in edge_profiles.values()) or 1.0
SCALE = AMP / _peak

curve_color = plt.cm.magma(0.4)
base_color = "0.55"

fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)

# Sample field inside the square
ax.imshow(
    sample_u,
    origin="lower",
    extent=(M, 1 - M, M, 1 - M),
    cmap=CMAP_OUTPUT,
    zorder=1,
)
square = plt.Polygon(
    [(M, M), (1 - M, M), (1 - M, 1 - M), (M, 1 - M)],
    fill=False,
    edgecolor="0.3",
    linewidth=1.2,
    zorder=3,
)
ax.add_patch(square)

# bottom — baseline below the square, positive u bulges downward (outward)
by = M - GAP
ax.plot([M, 1 - M], [by, by], color=base_color, lw=0.8, ls="--", zorder=2)
ax.plot(sq_x, by - edge_profiles["bottom"] * SCALE, color=curve_color, lw=1.6, zorder=4)

# top — baseline above the square, positive u bulges upward (outward)
ty = (1 - M) + GAP
ax.plot([M, 1 - M], [ty, ty], color=base_color, lw=0.8, ls="--", zorder=2)
ax.plot(sq_x, ty + edge_profiles["top"] * SCALE, color=curve_color, lw=1.6, zorder=4)

# left — baseline left of the square, positive u bulges leftward (outward)
lx = M - GAP
ax.plot([lx, lx], [M, 1 - M], color=base_color, lw=0.8, ls="--", zorder=2)
ax.plot(lx - edge_profiles["left"] * SCALE, sq_y, color=curve_color, lw=1.6, zorder=4)

# right — baseline right of the square, positive u bulges rightward (outward)
rx = (1 - M) + GAP
ax.plot([rx, rx], [M, 1 - M], color=base_color, lw=0.8, ls="--", zorder=2)
ax.plot(rx + edge_profiles["right"] * SCALE, sq_y, color=curve_color, lw=1.6, zorder=4)

PAD = 0.05
ax.set_xlim(-PAD, 1 + PAD)
ax.set_ylim(-PAD, 1 + PAD)
ax.set_aspect("equal")
ax.set_facecolor("none")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
# ax.set_title(
#     f"Sample {SAMPLE_2D}: field + boundary profiles (amplitude scaled)", fontsize=10
# )

fig.savefig(FIGURES_DIR / "darcy_boundary_profiles_2d.png", bbox_inches="tight")
plt.show()
print(f"Saved to {FIGURES_DIR / 'darcy_boundary_profiles_2d.png'}")

# %% Combined — boundary profiles (2x2) + 2D context
# Left: per-edge percentile bands + mean over many samples.
# Right: the single-sample field with its boundary profiles on each side.
fig = plt.figure(figsize=(13, 6), constrained_layout=True)
outer = fig.add_gridspec(1, 2, width_ratios=[1, 1])

# --- Left: 2x2 grid of percentile-band edge profiles ---
left_gs = outer[0, 0].subgridspec(2, 2, hspace=0.01, wspace=0.025)
for i, (label, extractor) in enumerate(edge_defs.items()):
    axp = fig.add_subplot(left_gs[i // 2, i % 2])
    pos = edge_xs[label]
    profiles = np.stack([extractor(sol_ds[k]) for k in range(N_PROFILE_SAMPLES)])
    p5, p25, p75, p95 = (np.percentile(profiles, q, axis=0) for q in (5, 25, 75, 95))
    axp.fill_between(
        pos, p5, p95, color=band_color, alpha=0.18, linewidth=0, label="5–95%"
    )
    axp.fill_between(
        pos, p25, p75, color=band_color, alpha=0.38, linewidth=0, label="25–75%"
    )
    axp.plot(pos, profiles.mean(axis=0), color=mean_color, linewidth=1.8, label="mean")
    axp.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    axp.set_title(label, fontsize=10)
    axp.set_xlabel("Position")
    axp.set_ylabel("$u$")
    axp.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    if i == 0:
        axp.legend(fontsize=12, frameon=False)

# --- Right: single-sample field + boundary profiles ---
axr = fig.add_subplot(outer[0, 1])
axr.imshow(
    sample_u, origin="lower", extent=(M, 1 - M, M, 1 - M), cmap=CMAP_OUTPUT, zorder=1
)
axr.add_patch(
    plt.Polygon(
        [(M, M), (1 - M, M), (1 - M, 1 - M), (M, 1 - M)],
        fill=False,
        edgecolor="0.3",
        linewidth=1.2,
        zorder=3,
    )
)
by = M - GAP
axr.plot([M, 1 - M], [by, by], color=base_color, lw=0.8, ls="--", zorder=2)
axr.plot(
    sq_x, by - edge_profiles["bottom"] * SCALE, color=curve_color, lw=1.6, zorder=4
)
ty = (1 - M) + GAP
axr.plot([M, 1 - M], [ty, ty], color=base_color, lw=0.8, ls="--", zorder=2)
axr.plot(sq_x, ty + edge_profiles["top"] * SCALE, color=curve_color, lw=1.6, zorder=4)
lx = M - GAP
axr.plot([lx, lx], [M, 1 - M], color=base_color, lw=0.8, ls="--", zorder=2)
axr.plot(lx - edge_profiles["left"] * SCALE, sq_y, color=curve_color, lw=1.6, zorder=4)
rx = (1 - M) + GAP
axr.plot([rx, rx], [M, 1 - M], color=base_color, lw=0.8, ls="--", zorder=2)
axr.plot(rx + edge_profiles["right"] * SCALE, sq_y, color=curve_color, lw=1.6, zorder=4)
axr.set_xlim(-PAD, 1 + PAD)
axr.set_ylim(-PAD, 1 + PAD)
axr.set_aspect("equal")
axr.set_facecolor("none")
for spine in axr.spines.values():
    spine.set_visible(False)
axr.set_xticks([])
axr.set_yticks([])
axr.set_title(f"Sample {SAMPLE_2D} Boundary Profiles", fontsize=12)
out_path = FIGURES_DIR / "darcy_boundary_profiles_combined.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved to {out_path}")

# %% Distance functions for different power values — 2D weight fields
# The ansatz enforces u = g + l(x,y) * N(x,y), where
# l(x,y) = [x(1−x) · y(1−y)]^p  (product reduce, unit box).
# Each panel shows the full 2D weight field over [0,1]².

POWERS = [0.01, 0.05, 0.1, 0.5, 1.0]

xy = np.linspace(0, 1, 200)
X, Y = np.meshgrid(xy, xy)
base = X * (1 - X) * Y * (1 - Y)  # peaks at 1/16 at the centre

fig, axes = plt.subplots(1, len(POWERS), figsize=(14, 3), constrained_layout=True)
for ax, p in zip(axes, POWERS):
    field = base**p
    im = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="magma")
    ax.set_title(f"$p = {p}$", fontsize=10)
    ax.set_xlabel("$x$")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
axes[0].set_ylabel("$y$")

fig.savefig(FIGURES_DIR / "darcy_distance_functions.png", bbox_inches="tight")
plt.show()
print(f"Saved to {FIGURES_DIR / 'darcy_distance_functions.png'}")

# %% Distance-power ablation — val rel-L2 curves over 100 epochs
EXP_DIR = (
    REPO_ROOT
    / "outputs/darcy/dirichlet_ansatz_zero/transolver/experiments/distance_power"
)

# Map filename suffix → display label (sorted by power value)
RUNS = [
    ("0", r"$p = 0$"),
    ("001", r"$p = 0.01$"),
    ("005", r"$p = 0.05$"),
    ("01", r"$p = 0.1$"),
    ("05", r"$p = 0.5$"),
    ("1", r"$p = 1$"),
]

PALETTE = plt.cm.plasma(np.linspace(0.1, 0.85, len(RUNS)))

fig, (ax_main, ax_final) = plt.subplots(
    1,
    2,
    figsize=(11, 4),
    gridspec_kw={"width_ratios": [3, 1]},
    constrained_layout=True,
)

final_vals = {}
for (suffix, label), color in zip(RUNS, PALETTE):
    import csv

    with open(EXP_DIR / f"val_rel_l2_{suffix}.csv") as fh:
        rows = list(csv.reader(fh))
    # W&B export: quoted columns — Step (col 0), metric (col 1)
    vals = np.array([float(r[1]) for r in rows[1:]])
    epochs = np.arange(1, len(vals) + 1)
    ax_main.plot(epochs, vals, color=color, linewidth=1.5, label=label)
    final_vals[label] = vals[-1]

ax_main.set_xlabel("Epoch")
ax_main.set_ylabel("Val rel-$L_2$")
ax_main.set_title("Distance-power ablation: validation curves")
ax_main.legend(fontsize=8, frameon=False, ncol=2)

# Bar chart of final val rel-L2
labels = [r for _, r in RUNS]
bar_vals = [final_vals[r] for r in labels]
bars = ax_final.barh(labels, bar_vals, color=PALETTE, edgecolor="white", linewidth=0.4)
ax_final.set_xlabel("Final val rel-$L_2$")
ax_final.set_title("Final value")
ax_final.invert_yaxis()
for bar, v in zip(bars, bar_vals):
    ax_final.text(
        v + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{v:.4f}",
        va="center",
        fontsize=8,
    )

out_path = FIGURES_DIR / "darcy_distance_power_ablation.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved to {out_path}")

# %% Corner-value diagnostic — what is u at the 4 corners, full-res vs downsampled?
# The sine basis pins corners to exactly 0. Before trusting that, measure the
# actual corner values. NOTE: STRIDE=5 on a 421-grid samples indices 0,5,…,420,
# and 420 == 421-1 is the true last index, so the corner PIXELS coincide in both
# modes — only the interior sampling differs. We expect identical corner stats.
_sol_ds = sol[:, ::STRIDE, ::STRIDE]
_corner_modes = {
    f"full ({sol.shape[1]})": sol,
    f"downsampled ({_sol_ds.shape[1]})": _sol_ds,
}
_corner_defs = {
    "BL (0,0)": (0, 0),
    "BR (0,-1)": (0, -1),
    "TL (-1,0)": (-1, 0),
    "TR (-1,-1)": (-1, -1),
}

print("--- Corner u values (over all samples) ---")
print(f"{'mode':<18}{'corner':<12}{'mean':>14}{'std':>14}{'mean|u|':>14}")
corner_stats = {}  # (mode, corner) -> (mean, std)
for mname, arr in _corner_modes.items():
    for cname, (ci, cj) in _corner_defs.items():
        vals = arr[:, ci, cj].astype(np.float64)
        m, s, ma = vals.mean(), vals.std(), np.abs(vals).mean()
        corner_stats[(mname, cname)] = (m, s)
        print(f"{mname:<18}{cname:<12}{m:>14.4e}{s:>14.4e}{ma:>14.4e}")

mode_names = list(_corner_modes)
corner_names = list(_corner_defs)
x = np.arange(len(corner_names))
width = 0.36
bar_colors = [plt.cm.magma(0.4), plt.cm.magma(0.7)]

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
for mi, mname in enumerate(mode_names):
    means = [corner_stats[(mname, c)][0] for c in corner_names]
    stds = [corner_stats[(mname, c)][1] for c in corner_names]
    ax.bar(
        x + (mi - 0.5) * width,
        means,
        width,
        yerr=stds,
        capsize=3,
        color=bar_colors[mi],
        edgecolor="white",
        linewidth=0.4,
        label=mname,
    )
ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(corner_names)
ax.set_ylabel("Corner $u$  (mean ± std)")
ax.set_title("Corner values: full-res vs downsampled")
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
ax.legend(fontsize=8, frameon=False)

out_path = FIGURES_DIR / "darcy_corner_values.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved to {out_path}")

# %% Sine-mode order — helpers
# SineBoundaryConstraint represents each edge as u(x_j) = sum_k c_k sin(kπ j/(L-1)),
# i.e. a truncated DST-I. The right `n_modes` is set by the spectral decay of the
# TRUE Darcy boundary profiles, not guessed. Corners are taken to be exactly 0, so
# the sine basis (which vanishes at the endpoints) is the exact representation —
# no endpoint lift needed. These helpers run the analysis on any grid resolution.
TOL = 0.01


def pool_edges(sol3d):
    """Pool the 4 boundary edges of (N, H, W) into (4N, L) float64 profiles."""
    return np.concatenate(
        [
            sol3d[:, 0, :],  # bottom  (N, W)
            sol3d[:, -1, :],  # top     (N, W)
            sol3d[:, :, 0],  # left    (N, H)
            sol3d[:, :, -1],  # right   (N, H)
        ],
        axis=0,
    ).astype(np.float64)  # float64 for stable projection


def corner_report(edges, label):
    """Sanity-check the corners-are-zero assumption vs profile amplitude.

    Uses mean/p99, not max: a few outlier samples make max unrepresentative.
    """
    cv = np.abs(edges[:, [0, -1]])
    rms = np.sqrt((edges**2).mean())
    print(
        f"[{label}] corner |u|: mean={cv.mean():.2e}  p99={np.percentile(cv, 99):.2e}"
        f"  vs  profile RMS = {rms:.2e}"
    )


def sine_spectrum(edges):
    """Project pooled edges onto the DST-I sine basis (identical to
    SineBoundaryConstraint._sine_basis) and return spectral diagnostics."""
    M, L = edges.shape
    K_MAX = L - 2  # max independent interior DST-I modes
    j = np.arange(L)
    k = np.arange(1, K_MAX + 1)
    B = np.sin(np.pi * np.outer(j, k) / (L - 1))  # (L, K_MAX)
    PROJ = 2.0 / (L - 1)  # DST-I orthogonality constant

    C = PROJ * (edges @ B)  # (M, K_MAX) sine coefficients per profile
    e_unit = (L - 1) / 2.0  # per-mode energy weight (Parseval)
    tot_energy = (edges**2).sum(axis=1)  # (M,) full-profile energy

    # Global rel-L2 vs number of modes K (energy-pooled over samples/edges)
    cum = np.cumsum(C**2 * e_unit, axis=1)
    res = np.clip(tot_energy[:, None] - cum, 0, None)
    relL2 = np.sqrt(res.sum(axis=0) / tot_energy.sum())

    mean_pow = (C**2).mean(axis=0)
    mean_pow = mean_pow / mean_pow[0]
    return {
        "L": L,
        "M": M,
        "K_MAX": K_MAX,
        "Ks": np.arange(1, K_MAX + 1),
        "relL2": relL2,
        "mean_pow": mean_pow,
    }


def report_modes(spec, label, tol=TOL):
    """Print the recommended n_modes and return K_REC (or None)."""
    relL2, Ks, K_MAX = spec["relL2"], spec["Ks"], spec["K_MAX"]
    hit = np.where(relL2 < tol)[0]
    K_REC = int(Ks[hit[0]]) if hit.size else None
    print(
        f"--- Sine-mode order ({label} L={spec['L']}, edges pooled, M={spec['M']}) ---"
    )
    if K_REC is not None:
        print(f"  reaches {tol:.0%} rel-L2 at K = {K_REC}")
    else:
        print(f"  never reaches {tol:.0%} rel-L2 within K_MAX={K_MAX}")
    if K_MAX >= 16:
        print(f"  constraint default n_modes=16  ->  rel-L2 = {relL2[15]:.3%}")
    print(f"  >> recommended n_modes ≈ {K_REC}")
    return K_REC


def plot_sine_order(spec, out_path):
    c_spec = plt.cm.magma(0.45)
    c_rec = plt.cm.plasma(0.35)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    # (1) Mean power spectrum — the intrinsic decay
    axes[0].semilogy(spec["Ks"], spec["mean_pow"], color=c_spec, linewidth=1.6)
    axes[0].set_xlabel("Sine mode $k$")
    axes[0].set_ylabel("Mean power $|c_k|^2$ (normalised)")
    axes[0].set_title("Boundary sine spectrum")
    axes[0].set_xlim(1, spec["K_MAX"])

    # (2) Reconstruction rel-L2 vs number of modes
    axes[1].semilogy(spec["Ks"], spec["relL2"], color=c_rec, linewidth=1.8)
    axes[1].set_xlabel("Number of sine modes $K$")
    axes[1].set_ylabel("Global rel-$L_2$ on boundary")
    axes[1].set_title("Reconstruction vs mode count")
    axes[1].set_xlim(1, spec["K_MAX"])

    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out_path}")


# %% Sine-mode order — full-res grid (L=421, K_MAX=419)
# L=421 gives K_MAX=419 modes; corner pixels coincide with the downsampled grid,
# so the corners-are-zero conclusion is resolution-independent.
edges_full = pool_edges(sol)
corner_report(edges_full, "full-res")
spec_full = sine_spectrum(edges_full)
report_modes(spec_full, "full-res")
plot_sine_order(spec_full, FIGURES_DIR / "darcy_sine_mode_order.png")

# %% Sine-mode order — downsampled grid (the actual training resolution)
# Same analysis on the STRIDE-downsampled grid the model is trained on; L=85
# caps at K_MAX=83 modes. This is the resolution that sets the deployed n_modes.
edges_ds = pool_edges(sol[:, ::STRIDE, ::STRIDE])
corner_report(edges_ds, "downsampled")
spec_ds = sine_spectrum(edges_ds)
report_modes(spec_ds, "downsampled")
plot_sine_order(spec_ds, FIGURES_DIR / "darcy_sine_mode_order_downsampled.png")


# %% Input permeability with separated boundaries
def plot_field_boundary_decomposition(field, *, cmap, out_path):
    vmin = float(np.min(field))
    vmax = float(np.max(field))

    pieces = {
        "top-left": field[-1:, :1],
        "top": field[-1:, 1:-1],
        "top-right": field[-1:, -1:],
        "left": field[1:-1, :1],
        "interior": field[1:-1, 1:-1],
        "right": field[1:-1, -1:],
        "bottom-left": field[:1, :1],
        "bottom": field[:1, 1:-1],
        "bottom-right": field[:1, -1:],
    }
    layout = [
        ["top-left", "top", "top-right"],
        ["left", "interior", "right"],
        ["bottom-left", "bottom", "bottom-right"],
    ]

    fig = plt.figure(figsize=(5.8, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=(0.04, 1.0, 0.04),
        height_ratios=(0.04, 1.0, 0.04),
        wspace=0.12,
        hspace=0.12,
    )

    for i, row in enumerate(layout):
        for j, name in enumerate(row):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(
                pieces[name],
                cmap=cmap,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("0.35")
                spine.set_linewidth(0.8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out_path}")


plot_field_boundary_decomposition(
    a,
    cmap=CMAP_INPUT,
    out_path=FIGURES_DIR / "darcy_input_boundary_decomposition.png",
)


def plot_field_interior(field, *, cmap, out_path):
    vmin = float(np.min(field))
    vmax = float(np.max(field))
    interior = field[1:-1, 1:-1]

    fig, ax = plt.subplots(figsize=(4.8, 4.8), constrained_layout=True)
    ax.imshow(
        interior,
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out_path}")


def plot_field_boundary_strips(field, *, cmap, out_path):
    vmin = float(np.min(field))
    vmax = float(np.max(field))
    edge_strips = [
        field[-1:, 1:-1].T,
        field[:1, 1:-1].T,
        field[1:-1, :1],
        field[1:-1, -1:],
    ]

    fig = plt.figure(figsize=(1.0, 5.0), constrained_layout=True)
    strips = fig.add_gridspec(1, 4, wspace=0.22)
    for col, edge in enumerate(edge_strips):
        ax = fig.add_subplot(strips[0, col])
        ax.imshow(
            edge,
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.35")
            spine.set_linewidth(0.8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out_path}")


plot_field_interior(
    a,
    cmap=CMAP_INPUT,
    out_path=FIGURES_DIR / "darcy_input_interior.png",
)
plot_field_boundary_strips(
    a,
    cmap=CMAP_INPUT,
    out_path=FIGURES_DIR / "darcy_input_boundary_strips.png",
)
plot_field_boundary_decomposition(
    u,
    cmap=CMAP_OUTPUT,
    out_path=FIGURES_DIR / "darcy_output_boundary_decomposition.png",
)
plot_field_interior(
    u,
    cmap=CMAP_OUTPUT,
    out_path=FIGURES_DIR / "darcy_output_interior.png",
)
plot_field_boundary_strips(
    u,
    cmap=CMAP_OUTPUT,
    out_path=FIGURES_DIR / "darcy_output_boundary_strips.png",
)


# %% Sine-basis boundary approximation example
def sine_basis_matrix(length, n_modes):
    j = np.arange(length)
    k = np.arange(1, n_modes + 1)
    return np.sin(np.pi * np.outer(j, k) / (length - 1))


def sine_reconstruct_profile(profile, n_modes):
    basis = sine_basis_matrix(profile.size, n_modes)
    coeffs = (2.0 / (profile.size - 1)) * np.dot(profile, basis)
    return np.dot(coeffs, basis.T)


def plot_sine_boundary_approximation(sample, *, n_modes, out_path):
    edge_profiles = {
        "bottom": sample[0, :],
        "top": sample[-1, :],
        "left": sample[:, 0],
        "right": sample[:, -1],
    }
    pos = np.linspace(0.0, 1.0, sample.shape[0])

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), constrained_layout=True)
    true_color = plt.cm.magma(0.35)
    recon_color = plt.cm.plasma(0.70)

    for ax, (edge_name, profile) in zip(axes, edge_profiles.items()):
        recon = sine_reconstruct_profile(profile.astype(np.float64), n_modes)
        ax.plot(pos, profile, color=true_color, linewidth=1.8, label="true")
        ax.plot(
            pos,
            recon,
            color=recon_color,
            linewidth=1.4,
            linestyle="--",
            label=f"{n_modes} modes",
        )
        ax.axhline(0.0, color="0.55", linewidth=0.8)
        ax.set_title(edge_name, fontsize=10)
        ax.set_xlabel("position")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

    axes[0].set_ylabel("$u$")
    axes[0].legend(fontsize=8, frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved to {out_path}")


def plot_right_boundary_sine_diagram(sample, *, n_modes, out_path):
    profile = sample[:, -1].astype(np.float64)
    recon = sine_reconstruct_profile(profile, n_modes)
    pos = np.linspace(0.0, 1.0, sample.shape[0])

    fig, ax = plt.subplots(figsize=(3.0, 2.0), constrained_layout=True)
    ax.plot(pos, profile, color=plt.cm.magma(0.35), linewidth=5.0)
    ax.plot(pos, recon, color=plt.cm.plasma(0.70), linewidth=4.6, linestyle="--")
    ax.axhline(0.0, color="0.55", linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    print(f"Saved to {out_path}")


plot_sine_boundary_approximation(
    sol_ds[SAMPLE_IDX],
    n_modes=21,
    out_path=FIGURES_DIR / "darcy_sine_basis_boundary_example.png",
)
plot_right_boundary_sine_diagram(
    sol_ds[SAMPLE_IDX],
    n_modes=21,
    out_path=FIGURES_DIR / "darcy_sine_basis_right_boundary_diagram.png",
)

# %% Empirical sine-head sweep — best val rel-L2 vs n_modes
# Counterpart to the theoretical reconstruction curve above: this is what a
# small MLP head with a sine basis actually achieves on Darcy boundaries
# (scripts/diagnostics/darcy/darcy_boundary_n_modes.py).
import csv

SWEEP_CSV = (
    REPO_ROOT / "artifacts/darcy/darcy_boundary_n_modes/darcy_boundary_n_modes_sine.csv"
)

with open(SWEEP_CSV) as fh:
    reader = csv.DictReader(fh)
    sweep_rows = [(int(r["n_modes"]), float(r["best_val_rel_l2"])) for r in reader]

sweep_ks = np.array([r[0] for r in sweep_rows])
sweep_rl2 = np.array([r[1] for r in sweep_rows])

best_idx = int(np.argmin(sweep_rl2))
floor_rl2 = float(sweep_rl2[best_idx])
best_k = int(sweep_ks[best_idx])

DEPLOYED_K = 21
dep_idx = int(np.where(sweep_ks == DEPLOYED_K)[0][0])
dep_rl2 = float(sweep_rl2[dep_idx])

curve_color = plt.cm.magma(0.45)
floor_color = plt.cm.plasma(0.35)
mark_color = plt.cm.plasma(0.65)

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.plot(
    sweep_ks,
    sweep_rl2,
    color=curve_color,
    linewidth=1.6,
    marker="o",
    markersize=3.5,
    markerfacecolor=curve_color,
    markeredgecolor="white",
    markeredgewidth=0.6,
)
ax.axhline(
    floor_rl2,
    color=floor_color,
    linewidth=1.0,
    linestyle="--",
    label=f"floor: {floor_rl2:.3f} @ $K={best_k}$",
)
# ax.scatter(
#     [DEPLOYED_K],
#     [dep_rl2],
#     color=mark_color,
#     s=70,
#     zorder=4,
#     edgecolor="white",
#     linewidth=1.0,
#     label=f"deployed $K={DEPLOYED_K}$: {dep_rl2:.3f}",
# )
ax.set_xlabel("Number of sine modes $K$")
ax.set_ylabel("Best val rel-$L_2$")
ax.set_title("Sine head: best val rel-$L_2$ vs $n_{\\mathrm{modes}}$")
ax.set_xlim(sweep_ks.min() - 1, sweep_ks.max() + 1)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9, frameon=False, loc="upper right")

out_path = FIGURES_DIR / "darcy_sine_head_n_modes_sweep.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved to {out_path}")
print(
    f"best K={best_k} (rel-L2={floor_rl2:.4f}), "
    f"deployed K={DEPLOYED_K} (rel-L2={dep_rl2:.4f})"
)

# %% Darcy flux constraint - load constrained Transolver model
# Pulls every intermediate field of the flux pipeline from a trained run so the
# architecture block diagram can show real fields:
#   psi (stream fn, backbone output) -> v_corr = grad_perp(psi)  [div = 0]
#   v_part [fixed, div = 1] + v_corr -> v_valid  [div = 1]
#   w = -v_valid / a -> sine Poisson solve -> u  (pressure, u|bnd = 0)
import importlib
import sys

import torch
import yaml

from omni_hc.benchmarks.darcy.adapter import _prepare_batch, _runtime_overrides
from omni_hc.benchmarks.darcy.data import build_test_loader
from omni_hc.integrations.nsl import create_model
from omni_hc.integrations.nsl.modeling import ensure_nsl_path
from omni_hc.training.common import (
    forward_with_optional_aux,
    load_checkpoint_state,
    load_model_state_dict,
)

DEVICE = torch.device("cpu")
FLUX_RUN_DIR = (
    REPO_ROOT / "outputs/darcy/darcy_flux_projection/transolver/final/seed_42"
)

cfg = yaml.safe_load((FLUX_RUN_DIR / "resolved_config.yaml").read_text())
cfg["paths"]["root_dir"] = str(DATA_DIR)

# NSL's unified_pos_embedding defaults to CUDA; force the run device before the
# backbone module imports it (needed to run on CPU / Mac).
ensure_nsl_path(cfg)
import layers.Embedding as _emb

_emb = importlib.reload(_emb)
_orig_upe = _emb.unified_pos_embedding
_emb._omni_hc_original_unified_pos_embedding = _orig_upe


def _upe_device(*a, **k):
    k.setdefault("device", DEVICE)
    return _orig_upe(*a, **k)


_emb.unified_pos_embedding = _upe_device
if "models.Transolver" in sys.modules:
    sys.modules["models.Transolver"].unified_pos_embedding = _upe_device

flux_loader = build_test_loader(cfg)
meta = flux_loader.darcy_meta
x_norm = flux_loader.x_normalizer.to(DEVICE)
y_norm = flux_loader.y_normalizer.to(DEVICE)

model, _, _ = create_model(
    cfg, device=DEVICE, runtime_overrides=_runtime_overrides(meta)
)
# Wire the constraint exactly as the steady test task does.
model.constraint.set_target_normalizer(y_norm)
model.constraint.set_input_normalizer(x_norm)
model.constraint.set_grid_shape(tuple(meta["shapelist"]))
_lo, _up = meta["domain_bounds"]
model.constraint.set_domain_bounds(lower=float(_lo), upper=float(_up))

ckpt = load_checkpoint_state(FLUX_RUN_DIR / "best.pt", device=DEVICE)
load_model_state_dict(model, ckpt["model_state_dict"])
model.eval()

batch = next(iter(flux_loader))
coords_b, fx_b, target_b = _prepare_batch(batch, device=DEVICE)
with torch.no_grad():
    out = forward_with_optional_aux(model, coords_b, fx_b)

H, W = meta["shapelist"]
FLUX_SAMPLE = 0
aux = out["aux_tensors"]


def _grid(t, c):
    return t[FLUX_SAMPLE].reshape(H, W, c).cpu().numpy()


psi = _grid(aux["pred_base"], 1)[..., 0]  # stream function (backbone)
v_corr_raw = _grid(aux["stream_correction"], 2)  # grad_perp(psi), div = 0
v_valid_raw = _grid(aux["constrained_flux"], 2)  # v_part + v_corr, div = 1
v_part = v_valid_raw - v_corr_raw  # fixed particular field, div = 1
a_field = _grid(x_norm.decode(fx_b), 1)[..., 0]  # permeability a
u_field = _grid(y_norm.decode(out["pred"]), 1)[..., 0]  # recovered pressure u


def _gradient_field_from_scalar(field, lower=0.0, upper=1.0):
    """Finite-difference vector field for diagram icons, ordered as (dx, dy)."""
    dy = (upper - lower) / max(field.shape[0] - 1, 1)
    dx = (upper - lower) / max(field.shape[1] - 1, 1)
    grad_y, grad_x = np.gradient(field.astype(np.float64), dy, dx, edge_order=2)
    return np.stack([grad_x, grad_y], axis=-1)


# For the diagram, derive the final vector fields from the model's pressure
# answer. This avoids showing raw internal tensors and gives smoother visual
# cues while preserving the Darcy relationship: v = -a grad u.
w_field = _gradient_field_from_scalar(u_field, lower=float(_lo), upper=float(_up))
v_valid = -a_field[..., None] * w_field
v_corr = v_valid - v_part

print(f"psi {psi.shape}  v_valid {v_valid.shape}  u {u_field.shape}")
print(
    "diagnostics:",
    {
        k: float(v.value)
        for k, v in out["diagnostics"].items()
        if "flux_div" in k or "darcy_res" in k or "boundary" in k
    },
)

# %% Darcy flux - pipeline field icons for the block diagram
# Clean, axis-free square icons so they compose into diagrams/darcy_flux_pipeline.tex.

CMAP_VECTOR = "spring"


def _save_scalar_icon(field, fname, cmap, symmetric=False):
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    kw = {}
    if symmetric:
        m = float(np.abs(field).max()) or 1.0
        kw = dict(vmin=-m, vmax=m)
    ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, **kw)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(
        FIGURES_DIR / fname, bbox_inches="tight", pad_inches=0.02, transparent=True
    )
    plt.show()
    print(f"saved {fname}")


def _save_vector_icon(field, fname, cmap):
    min_arrow_frac = 0.5
    edge_pad = 0.06
    mag = np.linalg.norm(field, axis=-1)
    eps = (
        np.finfo(field.dtype).eps if np.issubdtype(field.dtype, np.floating) else 1e-12
    )
    unit = np.divide(
        field,
        np.maximum(mag, eps)[..., None],
        out=np.zeros_like(field, dtype=np.float64),
        where=mag[..., None] > eps,
    )
    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    Xg, Yg = np.meshgrid(xs, ys)
    s = max(H // 8, 1)
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Background: magnitude field as a scalar heatmap
    finite_all = mag[np.isfinite(mag)]
    bg_vmin = float(finite_all.min()) if finite_all.size else 0.0
    bg_vmax = float(finite_all.max()) if finite_all.size else 1.0
    if bg_vmax <= bg_vmin:
        bg_vmax = bg_vmin + 1e-12
    ax.imshow(
        mag,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="viridis",
        vmin=bg_vmin,
        vmax=bg_vmax,
        alpha=0.72,
    )

    mag_sample = mag[::s, ::s]
    finite_mag = mag_sample[np.isfinite(mag_sample)]
    vmin = float(finite_mag.min()) if finite_mag.size else 0.0
    vmax = float(finite_mag.max()) if finite_mag.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-12
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rel_mag = np.clip(mag_sample / vmax, 0.0, 1.0)
    rel_mag = np.where(mag_sample > eps, np.maximum(rel_mag, min_arrow_frac), 0.0)
    ax.quiver(
        Xg[::s, ::s],
        Yg[::s, ::s],
        unit[::s, ::s, 0] * rel_mag,
        unit[::s, ::s, 1] * rel_mag,
        mag_sample,
        cmap=cmap,
        norm=norm,
        angles="xy",
        scale_units="xy",
        scale=11,
        pivot="mid",
        alpha=0.92,
        width=0.011,
        headwidth=3.5,
        headlength=2.5,
        headaxislength=2.8,
    )
    ax.set_xlim(-edge_pad, 1 + edge_pad)
    ax.set_ylim(-edge_pad, 1 + edge_pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(
        FIGURES_DIR / fname, bbox_inches="tight", pad_inches=0.02, transparent=True
    )
    plt.show()
    print(f"saved {fname}")


_save_scalar_icon(a_field, "darcy_flux_a.png", CMAP_INPUT)
_save_scalar_icon(psi, "darcy_flux_psi.png", CMAP_OUTPUT, symmetric=True)
_save_vector_icon(v_part, "darcy_flux_vpart.png", CMAP_VECTOR)
_save_vector_icon(v_corr, "darcy_flux_vcorr.png", CMAP_VECTOR)
_save_vector_icon(v_valid, "darcy_flux_vvalid.png", CMAP_VECTOR)
_save_vector_icon(w_field, "darcy_flux_w.png", CMAP_VECTOR)
_save_scalar_icon(u_field, "darcy_flux_u.png", CMAP_OUTPUT)
