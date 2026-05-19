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

profile_color = plt.cm.magma(0.35)  # mid-magma for individual samples
mean_color = plt.cm.magma(0.75)  # bright magma for the mean line
alpha = 0.3

fig, axes = plt.subplots(1, 4, figsize=(14, 3), constrained_layout=True)
for ax, (label, extractor) in zip(axes, edge_defs.items()):
    pos = edge_xs[label]
    profiles = np.stack([extractor(sol_ds[i]) for i in range(N_PROFILE_SAMPLES)])
    for profile in profiles:
        ax.plot(pos, profile, color=profile_color, alpha=alpha, linewidth=0.8)
    ax.plot(pos, profiles.mean(axis=0), color=mean_color, linewidth=1.8, label="mean")
    ax.axhline(0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Position")
    ax.set_ylabel("$u$")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)

fig.savefig(
    FIGURES_DIR / f"{N_PROFILE_SAMPLES}_darcy_boundary_profiles.png",
    bbox_inches="tight",
)
plt.show()
print(f"Saved to {FIGURES_DIR / f'{N_PROFILE_SAMPLES}_darcy_boundary_profiles.png'}")

# %% Boundary profiles in 2D context
# Same profiles as above, but drawn on the edges of the unit square.
# Each profile extends outward from its edge; amplitude is scaled to 15% of
# the square width for legibility (actual values are O(1e-5)).

# Pre-extract all four edge profile sets
_extractors = [
    ("bottom", lambda u: u[0, :], xs, "x"),  # profiles along x, offset in y
    ("top", lambda u: u[-1, :], xs, "x"),
    ("left", lambda u: u[:, 0], ys, "y"),  # profiles along y, offset in x
    ("right", lambda u: u[:, -1], ys, "y"),
]

_all_profiles = {}
for name, fn, pos, _ in _extractors:
    _all_profiles[name] = np.stack([fn(sol_ds[i]) for i in range(N_PROFILE_SAMPLES)])

# Inset the square so profiles have room to extend outward within [0, 1]
M = 0.2  # margin on each side
sq_x = np.linspace(M, 1 - M, sol_ds.shape[2])  # positions along horizontal edges
sq_y = np.linspace(M, 1 - M, sol_ds.shape[1])  # positions along vertical edges

# Scale factor: use 99th percentile of absolute values to set the margin fill;
# rare outliers clip at the axes limits rather than crushing the typical shape.
_peak = max(np.percentile(np.abs(_all_profiles[n]), 99) for n in _all_profiles)
SCALE = (0.8 * M) / _peak

fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

# Draw domain square (inset)
square = plt.Polygon(
    [(M, M), (1 - M, M), (1 - M, 1 - M), (M, 1 - M)],
    fill=False,
    edgecolor="0.3",
    linewidth=1.2,
    zorder=2,
)
ax.add_patch(square)

_pos_map = {"bottom": sq_x, "top": sq_x, "left": sq_y, "right": sq_y}

for name, fn, _, axis in _extractors:
    pos = _pos_map[name]
    profiles = _all_profiles[name]
    mean_prof = profiles.mean(axis=0)

    for prof in profiles:
        v = np.clip(prof * SCALE, 0, M)  # never extend beyond the margin
        if name == "bottom":
            ax.plot(pos, M - v, color=profile_color, alpha=0.04, linewidth=0.6)
        elif name == "top":
            ax.plot(pos, (1 - M) + v, color=profile_color, alpha=0.04, linewidth=0.6)
        elif name == "left":
            ax.plot(M - v, pos, color=profile_color, alpha=0.04, linewidth=0.6)
        elif name == "right":
            ax.plot((1 - M) + v, pos, color=profile_color, alpha=0.04, linewidth=0.6)

    mv = np.clip(mean_prof * SCALE, 0, M)
    if name == "bottom":
        ax.plot(pos, M - mv, color=mean_color, linewidth=1.8)
    elif name == "top":
        ax.plot(pos, (1 - M) + mv, color=mean_color, linewidth=1.8)
    elif name == "left":
        ax.plot(M - mv, pos, color=mean_color, linewidth=1.8)
    elif name == "right":
        ax.plot((1 - M) + mv, pos, color=mean_color, linewidth=1.8)

PAD = 0.05
ax.set_xlim(-PAD, 1 + PAD)
ax.set_ylim(-PAD, 1 + PAD)
ax.set_aspect("equal")
# Hide spines/ticks manually so the axes clip path stays active
ax.set_facecolor("none")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Boundary profiles on $\\partial\\Omega$ (amplitude scaled)", fontsize=10)

fig.savefig(FIGURES_DIR / "darcy_boundary_profiles_2d.png", bbox_inches="tight")
plt.show()
print(f"Saved to {FIGURES_DIR / 'darcy_boundary_profiles_2d.png'}")

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
            sol3d[:, 0, :],   # bottom  (N, W)
            sol3d[:, -1, :],  # top     (N, W)
            sol3d[:, :, 0],   # left    (N, H)
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
    print(f"--- Sine-mode order ({label} L={spec['L']}, edges pooled, M={spec['M']}) ---")
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
