# %% Imports & config
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

from pathlib import Path

import matplotlib

IS_NOTEBOOK = "ipykernel" in sys.modules

if not IS_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit

from omni_hc.constraints import (
    PipeInletParabolicAnsatz,
    PipeStreamFunctionBoundaryAnsatz,
    PipeUxBoundaryAnsatz,
)
from omni_hc.diagnostics.boundary_maps import infer_boundary_ansatz_maps

REPO_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
DATA_DIR = REPO_ROOT / "data/pipe"
FIGURES_DIR = REPO_ROOT / "docs/figures/pipe"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CMAP = "GnBu"
CHANNELS = ("ux", "uy", "p")

EDGE_SLICES = {
    "inlet": (0, slice(None)),
    "outlet": (-1, slice(None)),
    "lower_wall": (slice(None), 0),
    "upper_wall": (slice(None), -1),
}
EDGE_COLORS = {
    "inlet": plt.get_cmap(CMAP)(0.65),
    "outlet": plt.get_cmap(CMAP)(0.85),
    "lower_wall": plt.get_cmap(CMAP)(0.40),
    "upper_wall": plt.get_cmap(CMAP)(0.15),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

# %% Load dataset
x_all = np.load(DATA_DIR / "Pipe_X.npy", mmap_mode="r")  # (N, H, W)
y_all = np.load(DATA_DIR / "Pipe_Y.npy", mmap_mode="r")  # (N, H, W)
q_all = np.load(DATA_DIR / "Pipe_Q.npy", mmap_mode="r")  # (N, C, H, W)

N, H, W = x_all.shape
C = q_all.shape[1]
print(f"Pipe_X: {x_all.shape}  Pipe_Y: {y_all.shape}  Pipe_Q: {q_all.shape}")
print(f"Channels: {CHANNELS[:C]}")


# %% Dataset sample — mesh boundaries and ux field
SAMPLE_IDX = 0

x = np.asarray(x_all[SAMPLE_IDX])  # (H, W)
y = np.asarray(y_all[SAMPLE_IDX])  # (H, W)
q = np.asarray(q_all[SAMPLE_IDX])  # (C, H, W)

MESH_STEP = 8

fig, (ax_mesh, ax_ux) = plt.subplots(1, 2, figsize=(12, 4))

# --- left: curvilinear mesh + boundary edges ---
for i in range(0, H, MESH_STEP):
    ax_mesh.plot(x[i, :], y[i, :], color="0.80", lw=0.5)
for j in range(0, W, MESH_STEP):
    ax_mesh.plot(x[:, j], y[:, j], color="0.80", lw=0.5)

for edge, sl in EDGE_SLICES.items():
    ax_mesh.plot(x[sl], y[sl], color=EDGE_COLORS[edge], lw=2.0, label=edge)

ax_mesh.set_aspect("equal", adjustable="box")
ax_mesh.set_xlabel("$x$")
ax_mesh.set_ylabel("$y$")
ax_mesh.set_title("Pipe mesh and boundary edges")
ax_mesh.legend(
    fontsize=8,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=4,
)

# --- right: ux field on curvilinear mesh ---
im = ax_ux.pcolormesh(x, y, q[0], shading="gouraud", cmap=CMAP)
fig.colorbar(im, ax=ax_ux, label="$u_x$", shrink=0.6)
ax_ux.set_aspect("equal", adjustable="box")
ax_ux.set_xlabel("$x$")
ax_ux.set_ylabel("$y$")
ax_ux.set_title("Streamwise velocity $u_x$")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "pipe_dataset_sample.png", bbox_inches="tight")
if IS_NOTEBOOK:
    plt.show()
else:
    plt.close(fig)
print(f"Saved to {FIGURES_DIR / 'pipe_dataset_sample.png'}")


# %% Pipe flow constraint maps
def plot_pipe_constraint_maps(constraint_specs, *, coords_tensor, out_path):
    maps = [
        infer_boundary_ansatz_maps(
            constraint,
            pred_shape=(1, H * W, 1),
            grid_shape=(H, W),
            coords=coords_tensor,
            dtype=coords_tensor.dtype,
            device=coords_tensor.device,
        )
        for _, constraint in constraint_specs
    ]

    fig, axes = plt.subplots(
        2,
        len(maps),
        figsize=(14.0, 5.0),
        gridspec_kw={"hspace": 0.32, "wspace": 0.12},
        sharex=True,
        sharey=True,
    )
    fig.suptitle("Pipe flow hard-constraint maps", y=0.98)

    for col, ((title, _constraint), ansatz_maps) in enumerate(
        zip(constraint_specs, maps)
    ):
        for row, (map_name, values, cmap) in enumerate(
            (
                ("g", ansatz_maps.g[..., 0].numpy(), "coolwarm"),
                ("l", ansatz_maps.l[..., 0].numpy(), CMAP),
            )
        ):
            ax = axes[row, col]
            im = ax.pcolormesh(x, y, values, shading="gouraud", cmap=cmap)
            ax.plot(x[:, 0], y[:, 0], color="0.15", lw=0.7)
            ax.plot(x[:, -1], y[:, -1], color="0.15", lw=0.7)
            ax.plot(x[0, :], y[0, :], color="0.15", lw=0.7)
            ax.plot(x[-1, :], y[-1, :], color="0.15", lw=0.7)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{title}\n{map_name} ({ansatz_maps.space})", fontsize=9)
            ax.set_xlabel("$x$")
            if col == 0:
                ax.set_ylabel("$y$")
            fig.colorbar(im, ax=ax, shrink=0.8, fraction=0.035, pad=0.02)

    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.055, right=0.985)
    fig.savefig(out_path, bbox_inches="tight")
    if IS_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved to {out_path}")


coords_tensor = torch.as_tensor(
    np.stack([x, y], axis=-1).reshape(1, H * W, 2),
    dtype=torch.float32,
)
pipe_constraint_specs = [
    (
        "Inlet parabolic",
        PipeInletParabolicAnsatz(out_dim=1, grid_shape=(H, W), amplitude=0.25),
    ),
    (
        "Inlet + wall",
        PipeUxBoundaryAnsatz(out_dim=1, grid_shape=(H, W), amplitude=0.25),
    ),
    (
        "Stream function",
        PipeStreamFunctionBoundaryAnsatz(shapelist=(H, W), amplitude=0.25),
    ),
]
plot_pipe_constraint_maps(
    pipe_constraint_specs,
    coords_tensor=coords_tensor,
    out_path=FIGURES_DIR / "pipe_flow_constraint_maps.png",
)


# %% Boundary velocity profiles across the dataset
def edge_profile_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0),
        "q05": np.quantile(values, 0.05, axis=0),
        "q95": np.quantile(values, 0.95, axis=0),
    }


def edge_summary_stats(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    return {
        "mean": values.mean(),
        "mean_abs": np.abs(values).mean(),
        "max_abs": np.abs(values).max(),
        "min": values.min(),
        "max": values.max(),
        "std": values.std(),
    }


def print_markdown_stats_table(rows, columns):
    headers = ["Boundary", *columns]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---", *["---:" for _ in columns]]) + " |")
    for label, stats in rows:
        values = [f"{stats[col]:.4e}" for col in columns]
        print("| " + " | ".join([label, *values]) + " |")


def wall_summary_rows(channel_values):
    return [
        ("bottom wall (j=0)", edge_summary_stats(channel_values[:, :, 0])),
        ("bottom first interior (j=1)", edge_summary_stats(channel_values[:, :, 1])),
        (
            "top first interior (j=W-2)",
            edge_summary_stats(channel_values[:, :, -2]),
        ),
        ("top wall (j=W-1)", edge_summary_stats(channel_values[:, :, -1])),
    ]


def write_wall_statistics(channel_name, channel_values):
    columns = ["mean", "mean_abs", "max_abs", "min", "max", "std"]
    rows = wall_summary_rows(channel_values)

    print(f"--- Pipe wall {channel_name} statistics for report table ---")
    print_markdown_stats_table(rows, columns)

    stats_path = FIGURES_DIR / f"pipe_wall_{channel_name}_statistics.csv"
    with open(stats_path, "w", newline="") as fh:
        import csv

        writer = csv.writer(fh)
        writer.writerow(["boundary", *columns])
        for label, stats in rows:
            writer.writerow([label, *[f"{stats[col]:.6e}" for col in columns]])
    print(f"Saved to {stats_path}")


def plot_boundary_edge_profiles(channel_name, channel_values, label):
    edge_stats = {
        "inlet": edge_profile_stats(channel_values[:, 0, :]),
        "outlet": edge_profile_stats(channel_values[:, -1, :]),
    }
    transverse_idx = np.arange(W)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    fig.suptitle(f"Pipe ${label}$ boundary profiles", y=1.04)

    for ax, edge in zip(axes, ("inlet", "outlet")):
        stats = edge_stats[edge]
        ax.fill_between(
            transverse_idx,
            stats["q05"],
            stats["q95"],
            color=EDGE_COLORS[edge],
            alpha=0.16,
        )
        ax.plot(transverse_idx, stats["mean"], color=EDGE_COLORS[edge], lw=1.8)
        ax.set_title(edge)
        ax.set_xlabel("transverse edge index")
        ax.axhline(0.0, color="0.25", lw=0.8)
        ax.grid(True, alpha=0.22)

    axes[0].set_ylabel(f"${label}$")

    fig.tight_layout()
    output_path = FIGURES_DIR / f"pipe_{channel_name}_boundary_edge_profiles.png"
    fig.savefig(output_path, bbox_inches="tight")
    if IS_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved to {output_path}")


velocity_channels = [
    ("ux", 0, "u_x"),
    ("uy", 1, "u_y"),
]
for channel_name, channel_idx, label in velocity_channels:
    channel_values = q_all[:, channel_idx]
    write_wall_statistics(channel_name, channel_values)
    plot_boundary_edge_profiles(channel_name, channel_values, label)


# %% Inlet ux candidate shape fits
def normalized_edge_coordinate(values):
    v_min = float(values.min())
    v_max = float(values.max())
    if v_max <= v_min:
        raise ValueError("Edge coordinate has zero extent.")
    return (values - v_min) / (v_max - v_min)


def gaussian_profile(t, amplitude, center, sigma, offset):
    sigma = np.maximum(sigma, 1e-12)
    return offset + amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def wall_zero_gaussian_profile(t, amplitude, center, sigma):
    raw = gaussian_profile(t, 1.0, center, sigma, 0.0)
    left = gaussian_profile(0.0, 1.0, center, sigma, 0.0)
    right = gaussian_profile(1.0, 1.0, center, sigma, 0.0)
    chord = left + (right - left) * t
    shaped = raw - chord
    max_abs = np.max(np.abs(shaped))
    if max_abs <= 1e-12:
        return np.zeros_like(t)
    return amplitude * shaped / max_abs


def parabola_profile(t, amplitude):
    return amplitude * 4.0 * t * (1.0 - t)


def shifted_parabola_profile(t, amplitude, center):
    left = max(float(center), 1e-12)
    right = max(float(1.0 - center), 1e-12)
    z = np.where(t <= center, (t - center) / left, (t - center) / right)
    return amplitude * np.clip(1.0 - z**2, 0.0, None)


def profile_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_inlet_candidates(t, ux):
    amp0 = max(float(ux.max() - ux.min()), 1e-8)
    center0 = float(t[np.argmax(ux)])
    sigma0 = 0.25
    offset0 = float(min(ux[0], ux[-1]))

    candidates = {}

    popt, _ = curve_fit(
        gaussian_profile,
        t,
        ux,
        p0=(amp0, center0, sigma0, offset0),
        bounds=([0.0, 0.0, 1e-3, -np.inf], [np.inf, 1.0, 10.0, np.inf]),
        maxfev=20000,
    )
    pred = gaussian_profile(t, *popt)
    candidates["gaussian"] = {"pred": pred, "rmse": profile_rmse(ux, pred)}

    popt, _ = curve_fit(
        wall_zero_gaussian_profile,
        t,
        ux,
        p0=(amp0, center0, sigma0),
        bounds=([0.0, 0.0, 1e-3], [np.inf, 1.0, 10.0]),
        maxfev=20000,
    )
    pred = wall_zero_gaussian_profile(t, *popt)
    candidates["wall-zero gaussian"] = {"pred": pred, "rmse": profile_rmse(ux, pred)}

    popt, _ = curve_fit(
        parabola_profile,
        t,
        ux,
        p0=(max(float(ux.max()), 1e-8),),
        bounds=([0.0], [np.inf]),
        maxfev=20000,
    )
    pred = parabola_profile(t, *popt)
    candidates["wall-zero parabola"] = {"pred": pred, "rmse": profile_rmse(ux, pred)}

    popt, _ = curve_fit(
        shifted_parabola_profile,
        t,
        ux,
        p0=(max(float(ux.max()), 1e-8), center0),
        bounds=([0.0, 0.0], [np.inf, 1.0]),
        maxfev=20000,
    )
    pred = shifted_parabola_profile(t, *popt)
    candidates["shifted wall-zero parabola"] = {
        "pred": pred,
        "rmse": profile_rmse(ux, pred),
    }

    return candidates


y_inlet = np.asarray(y_all[0, 0, :], dtype=np.float64)
t_inlet = normalized_edge_coordinate(y_inlet)
order = np.argsort(t_inlet)
t_inlet = t_inlet[order]

ux_inlet_all = np.asarray(q_all[:, 0, 0, :], dtype=np.float64)[:, order]
ux_inlet_mean = ux_inlet_all.mean(axis=0)
ux_inlet_q05 = np.quantile(ux_inlet_all, 0.05, axis=0)
ux_inlet_q95 = np.quantile(ux_inlet_all, 0.95, axis=0)
candidate_fits = fit_inlet_candidates(t_inlet, ux_inlet_mean)

fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
ax.fill_between(
    t_inlet,
    ux_inlet_q05,
    ux_inlet_q95,
    color=EDGE_COLORS["inlet"],
    alpha=0.16,
    label="dataset 5-95%",
)
ax.plot(
    t_inlet,
    ux_inlet_mean,
    "ko",
    ms=3.0,
    label="dataset mean",
)

fit_colors = {
    "gaussian": "#7c3aed",
    "wall-zero gaussian": "#0f766e",
    "wall-zero parabola": "#dc2626",
    "shifted wall-zero parabola": "#ea580c",
}
for name, fit in sorted(candidate_fits.items(), key=lambda item: item[1]["rmse"]):
    ax.plot(
        t_inlet,
        fit["pred"],
        lw=1.7,
        color=fit_colors[name],
        label=f"{name}  RMSE={fit['rmse']:.2e}",
    )

ax.set_title("inlet $u_x$ candidate shape fits")
ax.set_xlabel("normalized inlet coordinate")
ax.set_ylabel("$u_x$")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "pipe_inlet_candidate_shape_fits.png", bbox_inches="tight")
if IS_NOTEBOOK:
    plt.show()
else:
    plt.close(fig)
print(f"Saved to {FIGURES_DIR / 'pipe_inlet_candidate_shape_fits.png'}")
