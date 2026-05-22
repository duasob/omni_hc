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
    StructuredWallDirichletAnsatz,
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
        gridspec_kw={"hspace": 0.12, "wspace": 0.25},
        sharex=True,
        sharey=True,
    )
    # fig.suptitle("Pipe flow hard-constraint maps", y=0.98)

    row_defs = (
        ("g", "coolwarm", lambda m: m.g[..., 0].numpy()),
        ("l", CMAP, lambda m: m.l[..., 0].numpy()),
    )
    # Shared colour range per row so a single colourbar describes the whole row.
    row_ranges = [
        (
            min(float(getter(m).min()) for m in maps),
            max(float(getter(m).max()) for m in maps),
        )
        for _, _, getter in row_defs
    ]
    row_meshes = [None] * len(row_defs)

    for col, ((title, _constraint), ansatz_maps) in enumerate(
        zip(constraint_specs, maps)
    ):
        for row, (map_name, cmap, getter) in enumerate(row_defs):
            ax = axes[row, col]
            vmin, vmax = row_ranges[row]
            im = ax.pcolormesh(
                x,
                y,
                getter(ansatz_maps),
                shading="gouraud",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            row_meshes[row] = im
            ax.plot(x[:, 0], y[:, 0], color="0.15", lw=0.7)
            ax.plot(x[:, -1], y[:, -1], color="0.15", lw=0.7)
            ax.plot(x[0, :], y[0, :], color="0.15", lw=0.7)
            ax.plot(x[-1, :], y[-1, :], color="0.15", lw=0.7)
            ax.set_aspect("equal", adjustable="box")

            if row == len(maps) - 1:
                ax.set_title(f"{map_name}", fontsize=12)
                ax.set_xlabel("$x$")
            else:
                ax.set_title(f"{title}\n{map_name}", fontsize=12)
            if col == 0:
                ax.set_ylabel("$y$")

    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.055, right=0.90)
    # One shared colourbar per row, placed to the right of the entire row.
    for row, mesh in enumerate(row_meshes):
        fig.colorbar(mesh, ax=list(axes[row, :]), shrink=0.8, fraction=0.035, pad=0.02)

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
        "Wall",
        StructuredWallDirichletAnsatz(out_dim=1, grid_shape=(H, W)),
    ),
    (
        "Inlet",
        PipeInletParabolicAnsatz(out_dim=1, grid_shape=(H, W), amplitude=0.25),
    ),
    (
        "Inlet + wall",
        PipeUxBoundaryAnsatz(out_dim=1, grid_shape=(H, W), amplitude=0.25),
    ),
    # (
    #     "Stream function",
    #     PipeStreamFunctionBoundaryAnsatz(shapelist=(H, W), amplitude=0.25),
    # ),
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


# %% Global vorticity from velocity (FD on curvilinear mesh)
# ω = ∂v/∂x − ∂u/∂y on a curvilinear (i, j) mesh, via chain rule through
# logical coordinates (ξ=i, η=j with Δξ=Δη=1):
#     ∂f/∂x = ( y_η f_ξ − y_ξ f_η) / det(J)
#     ∂f/∂y = (−x_η f_ξ + x_ξ f_η) / det(J)
# Area element: dA = |det(J)| dξ dη. Cross-check uses Stokes / Green's theorem:
#     ∫∫_Ω ω dA = ∮_∂Ω (u dx + v dy)
# (matches FD up to discretization error on the curvilinear boundary).
def compute_vorticity_field(x, y, ux, uy):
    x_xi = np.gradient(x, axis=0)
    x_eta = np.gradient(x, axis=1)
    y_xi = np.gradient(y, axis=0)
    y_eta = np.gradient(y, axis=1)
    det_J = x_xi * y_eta - x_eta * y_xi

    u_xi = np.gradient(ux, axis=0)
    u_eta = np.gradient(ux, axis=1)
    v_xi = np.gradient(uy, axis=0)
    v_eta = np.gradient(uy, axis=1)

    dv_dx = (y_eta * v_xi - y_xi * v_eta) / det_J
    du_dy = (-x_eta * u_xi + x_xi * u_eta) / det_J
    omega = dv_dx - du_dy
    return omega, det_J


def vorticity_integral(omega, det_J):
    return float((omega * np.abs(det_J)).sum())


def stokes_circulation(x, y, ux, uy):
    """Counter-clockwise line integral ∮ (u dx + v dy) over the mesh boundary."""

    def line(xs, ys, us, vs):
        dx = np.diff(xs)
        dy = np.diff(ys)
        um = 0.5 * (us[:-1] + us[1:])
        vm = 0.5 * (vs[:-1] + vs[1:])
        return float((um * dx + vm * dy).sum())

    bw = line(x[:, 0], y[:, 0], ux[:, 0], uy[:, 0])  # bottom wall
    ou = line(x[-1, :], y[-1, :], ux[-1, :], uy[-1, :])  # outlet
    tw = line(x[::-1, -1], y[::-1, -1], ux[::-1, -1], uy[::-1, -1])  # top wall
    iw = line(x[0, ::-1], y[0, ::-1], ux[0, ::-1], uy[0, ::-1])  # inlet
    return bw + ou + tw + iw


N_VORT = min(N, 500)
print(f"Computing global vorticity over {N_VORT} samples...")
integrals_fd = np.empty(N_VORT)
integrals_stokes = np.empty(N_VORT)
for i in range(N_VORT):
    xi = np.asarray(x_all[i], dtype=np.float64)
    yi = np.asarray(y_all[i], dtype=np.float64)
    ui = np.asarray(q_all[i, 0], dtype=np.float64)
    vi = np.asarray(q_all[i, 1], dtype=np.float64)
    omega_i, det_i = compute_vorticity_field(xi, yi, ui, vi)
    integrals_fd[i] = vorticity_integral(omega_i, det_i)
    integrals_stokes[i] = stokes_circulation(xi, yi, ui, vi)

print(
    f"  ∫ω dA   (FD)    : mean={integrals_fd.mean():+.3e}  "
    f"std={integrals_fd.std():.3e}  "
    f"|·| mean={np.abs(integrals_fd).mean():.3e}"
)
print(
    f"  ∮(u dx+v dy) (Stokes): mean={integrals_stokes.mean():+.3e}  "
    f"std={integrals_stokes.std():.3e}  "
    f"|·| mean={np.abs(integrals_stokes).mean():.3e}"
)

# Reference scale for normalisation: typical |inlet flux| ≈ ∫|u| dy at inlet.
ref_flux = float(
    np.mean(
        [
            np.abs(np.trapz(q_all[i, 0, 0, :], y_all[i, 0, :]))
            for i in range(min(N_VORT, 100))
        ]
    )
)
print(
    f"  reference inlet |∫u dy| ≈ {ref_flux:.3e}  "
    f"→ |∫ω|/ref ≈ {np.abs(integrals_fd).mean() / max(ref_flux, 1e-12):.2%}"
)

# --- Vorticity field for sample 0 ---
SAMPLE_VORT = SAMPLE_IDX
xs0 = np.asarray(x_all[SAMPLE_VORT], dtype=np.float64)
ys0 = np.asarray(y_all[SAMPLE_VORT], dtype=np.float64)
u0 = np.asarray(q_all[SAMPLE_VORT, 0], dtype=np.float64)
v0 = np.asarray(q_all[SAMPLE_VORT, 1], dtype=np.float64)
omega0, det0 = compute_vorticity_field(xs0, ys0, u0, v0)
I0 = vorticity_integral(omega0, det0)
S0 = stokes_circulation(xs0, ys0, u0, v0)
v_abs = float(np.percentile(np.abs(omega0), 99))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))

# Panel 1: vorticity field on the curvilinear mesh
ax = axes[0]
im = ax.pcolormesh(
    xs0, ys0, omega0, shading="gouraud", cmap="RdBu_r", vmin=-v_abs, vmax=v_abs
)
fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\omega$")
for sl in EDGE_SLICES.values():
    ax.plot(xs0[sl], ys0[sl], color="0.15", lw=0.7)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
title_line2 = (
    rf"$\int\omega\,dA = {I0:+.3e}$,  "
    rf"$\oint(u\,dx + v\,dy) = {S0:+.3e}$"
)
ax.set_title(f"sample {SAMPLE_VORT}: $\\omega(x,y)$\n{title_line2}", fontsize=10)

# Panel 2: distribution of ∫ω dA across samples
ax = axes[1]
ax.hist(
    integrals_fd,
    bins=40,
    color=plt.get_cmap(CMAP)(0.55),
    alpha=0.85,
    edgecolor="white",
    linewidth=0.4,
    label=r"$\int\omega\,dA$ (FD)",
)
ax.axvline(0.0, color="0.25", lw=1.0, linestyle="--")
ax.axvline(
    integrals_fd.mean(),
    color=plt.get_cmap(CMAP)(0.85),
    lw=1.5,
    label=f"mean = {integrals_fd.mean():+.2e}",
)
ax.set_xlabel(r"$\int\omega\,dA$")
ax.set_ylabel("count")
ax.set_title(f"Global vorticity across {N_VORT} samples")
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.22)

# Panel 3: FD ∫ω dA vs Stokes ∮(u dx + v dy)
ax = axes[2]
lim = max(np.abs(integrals_fd).max(), np.abs(integrals_stokes).max())
ax.plot([-lim, lim], [-lim, lim], color="0.5", lw=0.8, linestyle="--", label="$y=x$")
ax.scatter(
    integrals_fd,
    integrals_stokes,
    s=10,
    alpha=0.55,
    color=plt.get_cmap(CMAP)(0.7),
    edgecolor="none",
)
ax.set_xlabel(r"$\int\omega\,dA$ (FD)")
ax.set_ylabel(r"$\oint(u\,dx + v\,dy)$ (Stokes)")
ax.set_title("Green's theorem cross-check")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.22)
ax.legend(fontsize=8, frameon=False, loc="upper left")

fig.tight_layout()
out_path = FIGURES_DIR / "pipe_global_vorticity.png"
fig.savefig(out_path, bbox_inches="tight")
if IS_NOTEBOOK:
    plt.show()
else:
    plt.close(fig)
print(f"Saved to {out_path}")
