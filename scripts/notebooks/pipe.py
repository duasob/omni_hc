# %% Imports & config
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

from pathlib import Path

import matplotlib
from matplotlib.collections import LineCollection

IS_NOTEBOOK = "ipykernel" in sys.modules

if not IS_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
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
    "inlet": "orange",  # green
    "outlet": plt.get_cmap(CMAP)(0.85),
    "lower_wall": "yellowgreen",
    "upper_wall": "green",
}
EDGE_LABELS = {
    "inlet": "Inlet",
    "outlet": "Outlet",
    "lower_wall": "Lower wall",
    "upper_wall": "Upper wall",
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
    fontsize=12,
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


# %% Presentation asset — moving tracer animation inside the pipe
PIPE_FLOW_SAMPLE_IDX = [0, 100, 1000]
# May also be a sequence, e.g. PIPE_FLOW_SAMPLE_IDX = [0, 100, 1000].
PIPE_FLOW_FRAME_COUNT = 20
PIPE_FLOW_FPS = 10
PIPE_FLOW_STREAM_COUNT = 5
PIPE_FLOW_PARTICLES_PER_STREAM = 3


def _render_rgba_frames(fig, update_fn, frame_count: int, *, dpi: int = 150):
    """Render full RGBA frames so moving artists do not accumulate."""
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Saving the pipe-flow GIF requires Pillow.") from exc

    fig.set_dpi(dpi)
    frames = []
    for frame_idx in range(frame_count):
        update_fn(frame_idx)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        frames.append(Image.fromarray(rgba))
    return frames


def _save_rgba_animation(frames, out_path: Path, *, fps: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(1000 / max(int(fps), 1)), 1)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return out_path


def _normalised_curve_coordinate(xs, ys):
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    if arc[-1] <= 1e-12:
        return np.linspace(0.0, 1.0, len(xs))
    return arc / arc[-1]


def _curve_point(curve, phase):
    s, xs, ys = curve
    phase = float(np.clip(phase, 0.0, 1.0))
    return np.interp(phase, s, xs), np.interp(phase, s, ys)


def _normalise_pipe_flow_sample_indices(sample_indices):
    if np.isscalar(sample_indices):
        return (int(sample_indices),)
    return tuple(int(idx) for idx in sample_indices)


def _make_pipe_flow_tracer_frames(sample_idx: int, frame_count: int):
    xi = np.asarray(x_all[sample_idx], dtype=np.float64)
    yi = np.asarray(y_all[sample_idx], dtype=np.float64)
    ux = np.asarray(q_all[sample_idx, 0], dtype=np.float64)
    uy = np.asarray(q_all[sample_idx, 1], dtype=np.float64)
    speed = np.hypot(ux, uy)

    stream_columns = np.linspace(
        max(1, int(0.16 * (W - 1))),
        min(W - 2, int(0.84 * (W - 1))),
        PIPE_FLOW_STREAM_COUNT,
        dtype=int,
    )
    stream_curves = [
        (
            _normalised_curve_coordinate(xi[:, j], yi[:, j]),
            xi[:, j],
            yi[:, j],
        )
        for j in stream_columns
    ]
    stream_speeds = np.array(
        [np.mean(np.maximum(ux[:, j], 0.0)) for j in stream_columns],
        dtype=np.float64,
    )
    if float(stream_speeds.max()) > 1e-12:
        stream_speeds = 0.78 + 0.44 * stream_speeds / stream_speeds.max()
    else:
        stream_speeds = np.ones_like(stream_speeds)

    fig, ax = plt.subplots(figsize=(9.0, 2.2), facecolor="none")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.text(
        0.5,
        0.91,
        f"Sample {sample_idx}",
        ha="center",
        va="center",
        color="0.12",
        fontsize=16,
        fontweight="semibold",
    )

    vmax = float(np.percentile(speed, 99))
    ax.pcolormesh(
        xi,
        yi,
        speed,
        shading="gouraud",
        cmap=CMAP,
        vmin=0.0,
        vmax=max(vmax, 1e-12),
        alpha=0.95,
        zorder=1,
    )
    ax.plot(xi[:, 0], yi[:, 0], color="0.08", lw=1.8, zorder=4)
    ax.plot(xi[:, -1], yi[:, -1], color="0.08", lw=1.8, zorder=4)
    ax.plot(xi[0, :], yi[0, :], color=EDGE_COLORS["inlet"], lw=1.4, zorder=4)
    ax.plot(xi[-1, :], yi[-1, :], color=EDGE_COLORS["outlet"], lw=1.4, zorder=4)

    for _, xs, ys in stream_curves:
        ax.plot(xs, ys, color="white", lw=0.45, alpha=0.18, zorder=2)

    trails = LineCollection(
        [],
        colors=[(1.0, 1.0, 1.0, 0.58)],
        linewidths=1.5,
        capstyle="round",
        zorder=5,
    )
    ax.add_collection(trails)
    particles = ax.scatter(
        [],
        [],
        s=18,
        c="white",
        edgecolors=plt.get_cmap(CMAP)(0.95),
        linewidths=0.55,
        zorder=6,
    )

    # direction_arrow = ax.annotate(
    #     "",
    #     xy=(0.82, 0.84),
    #     xytext=(0.68, 0.84),
    #     xycoords="axes fraction",
    #     arrowprops=dict(
    #         arrowstyle="-|>",
    #         color="0.15",
    #         lw=1.7,
    #         mutation_scale=14,
    #         shrinkA=0,
    #         shrinkB=0,
    #     ),
    #     zorder=7,
    # )
    # direction_arrow.arrow_patch.set_alpha(0.9)

    x_pad = 0.02 * float(xi.max() - xi.min())
    y_pad = 0.24 * float(yi.max() - yi.min())
    ax.set_xlim(float(xi.min() - x_pad), float(xi.max() + x_pad))
    ax.set_ylim(float(yi.min() - y_pad), float(yi.max() + y_pad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    particle_offsets = np.linspace(
        0.0,
        1.0,
        PIPE_FLOW_PARTICLES_PER_STREAM,
        endpoint=False,
    )

    def update_frame(frame_idx: int):
        progress = frame_idx / frame_count
        points = []
        segments = []
        for stream_idx, (curve, speed_factor) in enumerate(
            zip(stream_curves, stream_speeds)
        ):
            row_offset = 0.035 * stream_idx
            for offset in particle_offsets:
                phase = (offset + row_offset + 1.65 * speed_factor * progress) % 1.0
                head = _curve_point(curve, phase)
                tail_phase = phase - 0.045
                tail = _curve_point(curve, tail_phase) if tail_phase >= 0.0 else head
                points.append(head)
                segments.append([tail, head])
        particles.set_offsets(np.asarray(points))
        trails.set_segments(segments)
        return particles, trails

    frames = _render_rgba_frames(
        fig,
        update_frame,
        frame_count,
        dpi=150,
    )
    if IS_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)
    return frames


def _make_pipe_flow_tracer_animation(sample_indices, out_path: Path) -> Path:
    sample_indices = _normalise_pipe_flow_sample_indices(sample_indices)
    if not sample_indices:
        raise ValueError("PIPE_FLOW_SAMPLE_IDX must contain at least one sample index.")

    frames = []
    for sample_idx in sample_indices:
        frames.extend(_make_pipe_flow_tracer_frames(sample_idx, PIPE_FLOW_FRAME_COUNT))
    return _save_rgba_animation(frames, out_path, fps=PIPE_FLOW_FPS)


pipe_flow_gif = _make_pipe_flow_tracer_animation(
    PIPE_FLOW_SAMPLE_IDX,
    FIGURES_DIR / "pipe_flow_tracers.gif",
)
print(f"Saved pipe flow GIF to {pipe_flow_gif}")


# %% Presentation asset — wall boundary constraint construction
PIPE_BOUNDARY_CONSTRAINT_SAMPLE_IDX = 1000
PIPE_BOUNDARY_CONSTRAINT_SEED = 7


def _pipe_constraint_maps(constraint, xi, yi):
    coords = torch.as_tensor(
        np.stack([xi, yi], axis=-1).reshape(1, H * W, 2),
        dtype=torch.float32,
    )
    maps = infer_boundary_ansatz_maps(
        constraint,
        pred_shape=(1, H * W, 1),
        grid_shape=(H, W),
        coords=coords,
        dtype=coords.dtype,
        device=coords.device,
    )
    return maps.g[..., 0].numpy(), maps.l[..., 0].numpy()


def _draw_pipe_field(ax, xi, yi, values, *, cmap, vmin=None, vmax=None):
    im = ax.pcolormesh(
        xi,
        yi,
        values,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.plot(xi[:, 0], yi[:, 0], color="0.08", lw=1.5)
    ax.plot(xi[:, -1], yi[:, -1], color="0.08", lw=1.5)
    ax.plot(xi[0, :], yi[0, :], color=EDGE_COLORS["inlet"], lw=1.0)
    ax.plot(xi[-1, :], yi[-1, :], color=EDGE_COLORS["outlet"], lw=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return im


def plot_pipe_boundary_constraint_construction(out_path: Path):
    xi = np.asarray(x_all[PIPE_BOUNDARY_CONSTRAINT_SAMPLE_IDX], dtype=np.float64)
    yi = np.asarray(y_all[PIPE_BOUNDARY_CONSTRAINT_SAMPLE_IDX], dtype=np.float64)

    _, l_wall = _pipe_constraint_maps(
        StructuredWallDirichletAnsatz(out_dim=1, grid_shape=(H, W)),
        xi,
        yi,
    )
    rng = np.random.default_rng(PIPE_BOUNDARY_CONSTRAINT_SEED)
    model_raw = gaussian_filter(rng.normal(size=(H, W)), sigma=4.0, mode="nearest")
    model_raw -= float(model_raw.min())
    model_range = float(model_raw.max())
    if model_range > 1e-12:
        model_raw /= model_range
    constrained = l_wall * model_raw

    fig = plt.figure(figsize=(12.0, 2.5), facecolor="white")
    fig.patch.set_alpha(1)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)

    axes = [
        fig.add_axes([0.08, 0.30, 0.25, 0.46]),
        fig.add_axes([0.375, 0.30, 0.25, 0.46]),
        fig.add_axes([0.67, 0.30, 0.25, 0.46]),
    ]
    _draw_pipe_field(
        axes[0],
        xi,
        yi,
        l_wall,
        cmap=CMAP,
        vmin=0.0,
        vmax=max(float(l_wall.max()), 1e-12),
    )
    _draw_pipe_field(
        axes[1],
        xi,
        yi,
        model_raw,
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
    )
    _draw_pipe_field(
        axes[2],
        xi,
        yi,
        constrained,
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
    )

    canvas.text(0.205, 0.18, r"$l$", ha="center", va="center", fontsize=28)
    canvas.text(0.50, 0.18, r"$N$", ha="center", va="center", fontsize=28)
    canvas.text(0.795, 0.18, r"$u$", ha="center", va="center", fontsize=28)
    canvas.text(0.352, 0.52, r"$\times$", ha="center", va="center", fontsize=34)
    canvas.text(0.648, 0.52, r"$=$", ha="center", va="center", fontsize=34)

    # TODO add colourbar

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    if IS_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved to {out_path}")


pipe_boundary_constraint_path = (
    FIGURES_DIR / "pipe_boundary_constraint_construction.png"
)
plot_pipe_boundary_constraint_construction(pipe_boundary_constraint_path)


def plot_pipe_inlet_wall_constraint_construction(out_path: Path):
    xi = np.asarray(x_all[PIPE_BOUNDARY_CONSTRAINT_SAMPLE_IDX], dtype=np.float64)
    yi = np.asarray(y_all[PIPE_BOUNDARY_CONSTRAINT_SAMPLE_IDX], dtype=np.float64)

    g_ux, l_ux = _pipe_constraint_maps(
        PipeUxBoundaryAnsatz(out_dim=1, grid_shape=(H, W), amplitude=0.25),
        xi,
        yi,
    )
    rng = np.random.default_rng(PIPE_BOUNDARY_CONSTRAINT_SEED)
    model_raw = gaussian_filter(rng.normal(size=(H, W)), sigma=4.0, mode="nearest")
    model_raw -= float(model_raw.min())
    model_range = float(model_raw.max())
    if model_range > 1e-12:
        model_raw /= model_range
    constrained = g_ux + l_ux * model_raw

    vmax = max(float(g_ux.max()), float(l_ux.max()), 1.0, float(constrained.max()))

    fig = plt.figure(figsize=(14.0, 2.5), facecolor="white")
    fig.patch.set_alpha(1)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_axis_off()
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)

    axes = [
        fig.add_axes([0.035, 0.30, 0.18, 0.46]),
        fig.add_axes([0.275, 0.30, 0.18, 0.46]),
        fig.add_axes([0.515, 0.30, 0.18, 0.46]),
        fig.add_axes([0.755, 0.30, 0.18, 0.46]),
    ]
    g_mesh = _draw_pipe_field(
        axes[0],
        xi,
        yi,
        g_ux,
        cmap="YlOrBr",
        vmin=0.0,
        vmax=max(float(g_ux.max()), 1e-12),
    )
    shared_mesh = None
    for ax, values in zip(axes[1:], (l_ux, model_raw, constrained)):
        shared_mesh = _draw_pipe_field(
            ax,
            xi,
            yi,
            values,
            cmap=CMAP,
            vmin=0.0,
            vmax=vmax,
        )

    g_cax = fig.add_axes([0.222, 0.36, 0.010, 0.30])
    shared_cax = fig.add_axes([0.948, 0.36, 0.010, 0.30])
    for cbar in (
        fig.colorbar(g_mesh, cax=g_cax),
        fig.colorbar(shared_mesh, cax=shared_cax),
    ):
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=7, length=2, width=0.5, pad=1)

    canvas.text(0.125, 0.18, r"$g$", ha="center", va="center", fontsize=28)
    canvas.text(0.365, 0.18, r"$l$", ha="center", va="center", fontsize=28)
    canvas.text(0.605, 0.18, r"$N$", ha="center", va="center", fontsize=28)
    canvas.text(0.845, 0.18, r"$u$", ha="center", va="center", fontsize=28)
    canvas.text(0.245, 0.52, r"$+$", ha="center", va="center", fontsize=34)
    canvas.text(0.485, 0.52, r"$\times$", ha="center", va="center", fontsize=34)
    canvas.text(0.725, 0.52, r"$=$", ha="center", va="center", fontsize=34)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    if IS_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved to {out_path}")


pipe_inlet_wall_constraint_path = (
    FIGURES_DIR / "pipe_inlet_wall_constraint_construction.png"
)
plot_pipe_inlet_wall_constraint_construction(pipe_inlet_wall_constraint_path)


# %% Mesh structure variation across samples (2×5 grid)
N_MESH_ROWS, N_MESH_COLS = 3, 3
MESH_SAMPLE_INDICES = np.linspace(0, N - 1, N_MESH_ROWS * N_MESH_COLS, dtype=int)

fig, axes = plt.subplots(
    N_MESH_ROWS,
    N_MESH_COLS,
    figsize=(N_MESH_COLS * 4.5, N_MESH_ROWS * 1.8),
    constrained_layout=True,
)

legend_handles = []
for ax_idx, (ax, idx) in enumerate(zip(axes.ravel(), MESH_SAMPLE_INDICES)):
    xi = np.asarray(x_all[idx])
    yi = np.asarray(y_all[idx])

    for i in range(0, H, MESH_STEP):
        ax.plot(xi[i, :], yi[i, :], color="0.78", lw=0.35)
    for j in range(0, W, MESH_STEP):
        ax.plot(xi[:, j], yi[:, j], color="0.78", lw=0.35)

    for edge, sl in EDGE_SLICES.items():
        (line,) = ax.plot(xi[sl], yi[sl], color=EDGE_COLORS[edge], lw=1.6)
        if ax_idx == 0:
            line.set_label(edge)
            legend_handles.append(line)

    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

fig.legend(
    handles=legend_handles,
    fontsize=16,
    frameon=False,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.04),
)

out_path = FIGURES_DIR / "pipe_mesh_variation.png"
fig.savefig(out_path, bbox_inches="tight")
if IS_NOTEBOOK:
    plt.show()
else:
    plt.close(fig)
print(f"Saved to {out_path}")


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
        ("$g(i,j)$", "coolwarm", lambda m: m.g[..., 0].numpy()),
        ("$l(i,j)$", CMAP, lambda m: m.l[..., 0].numpy()),
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
        ("inlet (i=0)", edge_summary_stats(channel_values[:, 0, :])),
        ("outlet (i=L-1)", edge_summary_stats(channel_values[:, -1, :])),
    ]


def write_wall_statistics(channel_name, channel_values):
    columns = ["mean", "mean_abs", "max_abs", "min", "max", "std"]
    rows = wall_summary_rows(channel_values)

    print(f"--- Pipe boundary {channel_name} statistics for report table ---")
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
    transverse_coord = np.linspace(0.0, 1.0, W)

    fig, ax = plt.subplots(figsize=(5.8, 3.4))

    for edge in ("inlet", "outlet"):
        stats = edge_stats[edge]
        ax.fill_between(
            transverse_coord,
            stats["q05"],
            stats["q95"],
            color=EDGE_COLORS[edge],
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            transverse_coord,
            stats["mean"],
            color=EDGE_COLORS[edge],
            lw=2.0,
            label=EDGE_LABELS[edge],
        )

    ax.axhline(0.0, color="0.20", lw=0.8, alpha=0.75)
    ax.set_title(f"Pipe ${label}$ boundary profiles", pad=8)
    ax.set_xlabel("normalized transverse coordinate")
    ax.set_ylabel(f"${label}$")
    ax.legend(frameon=False, loc="best")
    ax.margins(x=0.02)
    # ax.grid(axis="y", color="0.88", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.20")
        spine.set_linewidth(0.8)

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


# %% Pipe sample geometries with inlet and outlet ux profiles
PROFILE_SAMPLE_COUNT = 3
PROFILE_SAMPLE_INDICES = [100, 500, 1000]
PROFILE_MESH_STEP = 10


PROFILE_DISPLAY_INLET_SPAN = 0.18


def standardize_profile_geometry(xi, yi):
    inlet_center_x = float(np.nanmean(xi[0, :]))
    inlet_center_y = float(np.nanmean(yi[0, :]))
    stream_hint = np.array(
        [
            float(np.nanmean(xi[-1, :] - xi[0, :])),
            float(np.nanmean(yi[-1, :] - yi[0, :])),
        ],
        dtype=np.float64,
    )
    stream_norm = float(np.hypot(*stream_hint))
    if stream_norm <= 1e-12:
        stream_unit = np.array([1.0, 0.0], dtype=np.float64)
    else:
        stream_unit = stream_hint / stream_norm
    transverse_unit = np.array([-stream_unit[1], stream_unit[0]], dtype=np.float64)

    dx = xi - inlet_center_x
    dy = yi - inlet_center_y
    display_x = dx * stream_unit[0] + dy * stream_unit[1]
    display_y = dx * transverse_unit[0] + dy * transverse_unit[1]

    inlet_span = float(
        np.hypot(
            display_x[0, -1] - display_x[0, 0],
            display_y[0, -1] - display_y[0, 0],
        )
    )
    if inlet_span <= 1e-12:
        inlet_span = 1.0
    scale = PROFILE_DISPLAY_INLET_SPAN / inlet_span
    return display_x * scale, display_y * scale


def profile_curve_scale(xi, yi, ux_values):
    pipe_length = float(
        np.nanmedian(np.hypot(xi[-1, :] - xi[0, :], yi[-1, :] - yi[0, :]))
    )
    max_abs_ux = float(np.nanmax(np.abs(ux_values)))
    if max_abs_ux <= 1e-12:
        return 0.0
    return 0.11 * pipe_length / max_abs_ux


def edge_frame(edge_x, edge_y, outward_hint):
    tangent = np.array(
        [float(edge_x[-1] - edge_x[0]), float(edge_y[-1] - edge_y[0])],
        dtype=np.float64,
    )
    tangent_norm = float(np.hypot(*tangent))
    if tangent_norm <= 1e-12:
        tangent = np.array([0.0, 1.0])
    else:
        tangent /= tangent_norm

    normal = np.array([tangent[1], -tangent[0]])
    if float(np.dot(normal, outward_hint)) < 0.0:
        normal *= -1.0
    return tangent, normal


def detached_profile_geometry(edge_x, edge_y, ux_edge, scale, outward_hint):
    tangent, normal = edge_frame(edge_x, edge_y, outward_hint)
    center = np.array([float(np.mean(edge_x)), float(np.mean(edge_y))])
    edge_span = float(np.hypot(edge_x[-1] - edge_x[0], edge_y[-1] - edge_y[0]))
    side_coord = np.linspace(-0.5 * edge_span, 0.5 * edge_span, W)
    base_center = center + 0.07 * normal
    base = base_center[:, None] + tangent[:, None] * side_coord[None, :]
    curve = base + normal[:, None] * (scale * ux_edge)[None, :]
    return base[0], base[1], curve[0], curve[1]


def draw_pipe_sample_with_ux_profiles(ax, sample_idx):
    xi = np.asarray(x_all[sample_idx], dtype=np.float64)
    yi = np.asarray(y_all[sample_idx], dtype=np.float64)
    xi, yi = standardize_profile_geometry(xi, yi)
    ux = np.asarray(q_all[sample_idx, 0], dtype=np.float64)

    for i in range(0, H, PROFILE_MESH_STEP):
        ax.plot(xi[i, :], yi[i, :], color="0.86", lw=0.35, zorder=1)
    for j in range(0, W, PROFILE_MESH_STEP):
        ax.plot(xi[:, j], yi[:, j], color="0.86", lw=0.35, zorder=1)

    ax.plot(xi[0, :], yi[0, :], color=EDGE_COLORS["inlet"], lw=1.5, zorder=3)
    ax.plot(xi[-1, :], yi[-1, :], color=EDGE_COLORS["outlet"], lw=1.5, zorder=3)
    ax.plot(xi[:, 0], yi[:, 0], color="0.15", lw=1.1, zorder=3)
    ax.plot(xi[:, -1], yi[:, -1], color="0.15", lw=1.1, zorder=3)

    scale = profile_curve_scale(xi, yi, np.r_[ux[0, :], ux[-1, :]])
    stream_hint = np.array(
        [
            float(np.nanmean(xi[-1, :] - xi[0, :])),
            float(np.nanmean(yi[-1, :] - yi[0, :])),
        ]
    )
    inlet_base_x, inlet_base_y, inlet_x, inlet_y = detached_profile_geometry(
        xi[0, :], yi[0, :], ux[0, :], scale, -stream_hint
    )
    outlet_base_x, outlet_base_y, outlet_x, outlet_y = detached_profile_geometry(
        xi[-1, :], yi[-1, :], ux[-1, :], scale, stream_hint
    )

    ax.fill(
        np.r_[inlet_base_x, inlet_x[::-1]],
        np.r_[inlet_base_y, inlet_y[::-1]],
        color=EDGE_COLORS["inlet"],
        alpha=0.12,
        linewidth=0,
        zorder=2,
    )
    ax.fill(
        np.r_[outlet_base_x, outlet_x[::-1]],
        np.r_[outlet_base_y, outlet_y[::-1]],
        color=EDGE_COLORS["outlet"],
        alpha=0.12,
        linewidth=0,
        zorder=2,
    )
    ax.plot(inlet_base_x, inlet_base_y, color=EDGE_COLORS["inlet"], lw=1.5, zorder=4)
    ax.plot(outlet_base_x, outlet_base_y, color=EDGE_COLORS["outlet"], lw=1.5, zorder=4)
    ax.plot(inlet_x, inlet_y, color=EDGE_COLORS["inlet"], lw=2.0, zorder=4)
    ax.plot(outlet_x, outlet_y, color=EDGE_COLORS["outlet"], lw=2.0, zorder=4)

    inlet_edge_center = np.array([float(np.mean(xi[0, :])), float(np.mean(yi[0, :]))])
    inlet_base_center = np.array(
        [float(np.mean(inlet_base_x)), float(np.mean(inlet_base_y))]
    )
    outlet_edge_center = np.array(
        [float(np.mean(xi[-1, :])), float(np.mean(yi[-1, :]))]
    )
    outlet_base_center = np.array(
        [float(np.mean(outlet_base_x)), float(np.mean(outlet_base_y))]
    )
    inlet_gap = inlet_edge_center - inlet_base_center
    outlet_gap = outlet_base_center - outlet_edge_center
    arrow_style = dict(
        arrowstyle="-|>",
        color="0.10",
        lw=1.1,
        mutation_scale=11,
        shrinkA=0,
        shrinkB=0,
        zorder=6,
    )
    ax.annotate(
        "",
        xy=inlet_edge_center - 0.18 * inlet_gap,
        xytext=inlet_base_center + 0.18 * inlet_gap,
        arrowprops=arrow_style,
    )
    ax.annotate(
        "",
        xy=outlet_base_center - 0.18 * outlet_gap,
        xytext=outlet_edge_center + 0.18 * outlet_gap,
        arrowprops=arrow_style,
    )

    # ax.text(
    #     0.01,
    #     0.92,
    #     f"sample {sample_idx}",
    #     transform=ax.transAxes,
    #     fontsize=8,
    #     ha="left",
    #     va="top",
    #     color="0.15",
    # )
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


fig, axes = plt.subplots(
    PROFILE_SAMPLE_COUNT,
    1,
    figsize=(8.0, 1.45 * PROFILE_SAMPLE_COUNT),
    constrained_layout=True,
)
axes = np.atleast_1d(axes)
for ax, sample_idx in zip(axes, PROFILE_SAMPLE_INDICES):
    draw_pipe_sample_with_ux_profiles(ax, int(sample_idx))

x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
for ax in axes:
    x0, y0, width, height = ax.dataLim.bounds
    x_mins.append(x0)
    y_mins.append(y0)
    x_maxs.append(x0 + width)
    y_maxs.append(y0 + height)

pad = 0.04
x_lim = (min(x_mins) - pad, max(x_maxs) + pad)
y_lim = (min(y_mins) - pad, max(y_maxs) + pad)
for ax in axes:
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

legend_handles = [
    plt.Line2D([], [], color=EDGE_COLORS["inlet"], lw=2.0, label="Inlet $u_x$"),
    plt.Line2D([], [], color=EDGE_COLORS["outlet"], lw=2.0, label="Outlet $u_x$"),
    # plt.Line2D([], [], color="0.15", lw=1.1, label="Pipe wall"),
]
fig.legend(
    handles=legend_handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.07),
    ncol=3,
)

out_path = FIGURES_DIR / "pipe_sample_geometry_ux_profiles.png"
fig.savefig(out_path, bbox_inches="tight")
if IS_NOTEBOOK:
    plt.show()
else:
    plt.close(fig)
print(f"Saved to {out_path}")


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
# ax.grid(True, alpha=0.25)
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

# %% Ground-truth divergence ∇·u on the curvilinear mesh
# Replicates the metric used by omni_hc.constraints.metrics.pipe.compute (curvilinear
# chain rule via the Jacobian). ch5 GT reports div_rmse ≈ 6.62e-1; we want to see
# whether that is intrinsic to the dataset or a discretisation/normalisation issue.

SAMPLE_IDX = 1000


def compute_divergence_field(x, y, ux, uy):
    x_xi = np.gradient(x, axis=0)
    x_eta = np.gradient(x, axis=1)
    y_xi = np.gradient(y, axis=0)
    y_eta = np.gradient(y, axis=1)
    det_J = x_xi * y_eta - x_eta * y_xi
    det_J_safe = np.where(np.abs(det_J) < 1e-12, 1e-12, det_J)

    u_xi = np.gradient(ux, axis=0)
    u_eta = np.gradient(ux, axis=1)
    v_xi = np.gradient(uy, axis=0)
    v_eta = np.gradient(uy, axis=1)

    du_dx = (y_eta * u_xi - y_xi * u_eta) / det_J_safe
    dv_dy = (-x_eta * v_xi + x_xi * v_eta) / det_J_safe
    return du_dx + dv_dy, du_dx, dv_dy, det_J


N_DIV = min(N, 1000)
print(f"Computing ∇·u over {N_DIV} samples...")
div_abs_mean = np.empty(N_DIV)
div_abs_max = np.empty(N_DIV)
div_rmse = np.empty(N_DIV)
speed_mean = np.empty(N_DIV)
dudx_abs_mean = np.empty(N_DIV)
dvdy_abs_mean = np.empty(N_DIV)
for i in range(N_DIV):
    xi = np.asarray(x_all[i], dtype=np.float64)
    yi = np.asarray(y_all[i], dtype=np.float64)
    ui = np.asarray(q_all[i, 0], dtype=np.float64)
    vi = np.asarray(q_all[i, 1], dtype=np.float64)
    div_i, dudx_i, dvdy_i, _ = compute_divergence_field(xi, yi, ui, vi)
    div_abs_mean[i] = np.abs(div_i).mean()
    div_abs_max[i] = np.abs(div_i).max()
    div_rmse[i] = float(np.sqrt(np.mean(div_i**2)))
    speed_mean[i] = float(np.mean(np.hypot(ui, vi)))
    dudx_abs_mean[i] = np.abs(dudx_i).mean()
    dvdy_abs_mean[i] = np.abs(dvdy_i).mean()

# Reference scales for interpretation.
# 1. |∂u/∂x| + |∂v/∂y| sets the natural denominator (catastrophic cancellation upper bound).
# 2. mean speed gives a "is the field even close to zero" sanity check.
ref_grad = (dudx_abs_mean + dvdy_abs_mean).mean()
print(
    f"  |∇·u| mean             : {div_abs_mean.mean():.3e}"
    f"  (std {div_abs_mean.std():.3e})"
)
print(
    f"  |∇·u| max              : {div_abs_max.mean():.3e}"
    f"  (std {div_abs_max.std():.3e})"
)
print(f"  ‖∇·u‖₂ per sample (rmse): {div_rmse.mean():.3e}  (std {div_rmse.std():.3e})")
print(
    f"  reference |∂u/∂x|+|∂v/∂y| mean = {ref_grad:.3e}"
    f"  → relative |∇·u|/ref = {div_abs_mean.mean() / max(ref_grad, 1e-12):.2%}"
)
print(
    f"  mean speed |u|         : {speed_mean.mean():.3e}"
    f"  → ‖∇·u‖₂ / mean|u|     = {div_rmse.mean() / max(speed_mean.mean(), 1e-12):.2%}"
)

# Sample 0 visualisation.
xs0 = np.asarray(x_all[SAMPLE_IDX], dtype=np.float64)
ys0 = np.asarray(y_all[SAMPLE_IDX], dtype=np.float64)
u0 = np.asarray(q_all[SAMPLE_IDX, 0], dtype=np.float64)
v0 = np.asarray(q_all[SAMPLE_IDX, 1], dtype=np.float64)
div0, dudx0, dvdy0, det0 = compute_divergence_field(xs0, ys0, u0, v0)
v_abs = float(np.percentile(np.abs(div0), 99))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))

ax = axes[0]
im = ax.pcolormesh(
    xs0, ys0, div0, shading="gouraud", cmap="RdBu_r", vmin=-v_abs, vmax=v_abs
)
fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\nabla\cdot u$")
for sl in EDGE_SLICES.values():
    ax.plot(xs0[sl], ys0[sl], color="0.15", lw=0.7)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title(
    f"sample {SAMPLE_IDX}: $\\nabla\\cdot u$\n"
    rf"$|\nabla\cdot u|_\mathrm{{mean}}={np.abs(div0).mean():.2e}$,"
    rf"  rmse$={float(np.sqrt(np.mean(div0**2))):.2e}$",
    fontsize=10,
)

ax = axes[1]
ax.hist(
    div_rmse,
    bins=40,
    color=plt.get_cmap(CMAP)(0.55),
    alpha=0.85,
    edgecolor="white",
    linewidth=0.4,
    label=r"$\|\nabla\cdot u\|_2$ per sample",
)
ax.axvline(
    div_rmse.mean(),
    color=plt.get_cmap(CMAP)(0.85),
    lw=1.5,
    label=f"mean = {div_rmse.mean():.2e}",
)
ax.set_xlabel(r"$\|\nabla\cdot u\|_2$")
ax.set_ylabel("count")
ax.set_title(f"Per-sample divergence RMSE across {N_DIV} samples")
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.22)

ax = axes[2]
lim = max(np.abs(dudx0).max(), np.abs(dvdy0).max())
ax.plot(
    [-lim, lim],
    [lim, -lim],
    color="0.5",
    lw=0.8,
    linestyle="--",
    label=r"$\partial_x u=-\partial_y v$",
)
ax.scatter(
    dudx0.ravel(),
    dvdy0.ravel(),
    s=6,
    alpha=0.35,
    color=plt.get_cmap(CMAP)(0.7),
    edgecolor="none",
)
ax.set_xlabel(r"$\partial u_x/\partial x$")
ax.set_ylabel(r"$\partial u_y/\partial y$")
ax.set_title(f"sample {SAMPLE_IDX}: incompressibility scatter")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.22)
ax.legend(fontsize=8, frameon=False, loc="upper left")

fig.tight_layout()
out_path = FIGURES_DIR / "pipe_divergence.png"
fig.savefig(out_path, bbox_inches="tight")
if IS_NOTEBOOK:
    plt.show()
else:
    plt.close(fig)
print(f"Saved to {out_path}")


# %% Mass conservation across streamwise cross-sections
# For incompressible 2D flow with no-slip walls, the volumetric flux through
# every cross-section i must equal the inlet flux:
#     Q_i = ∮_slice u·n dℓ = ∫_j (u_x dy - u_y dx)  =  Q_inlet     ∀ i
# Discretely, along row i (j = 0..W-2):
#     Q_i = Σ_j [ 0.5(u_x[i,j] + u_x[i,j+1]) * (y[i,j+1] - y[i,j])
#                 - 0.5(u_y[i,j] + u_y[i,j+1]) * (x[i,j+1] - x[i,j]) ]
def streamwise_flux_profile(x, y, ux, uy):
    dx = np.diff(x, axis=1)  # (H, W-1) along j
    dy = np.diff(y, axis=1)
    ux_mid = 0.5 * (ux[:, :-1] + ux[:, 1:])
    uy_mid = 0.5 * (uy[:, :-1] + uy[:, 1:])
    return (ux_mid * dy - uy_mid * dx).sum(axis=1)  # (H,)


N_MASS = min(N, 1000)
print(f"Computing streamwise flux Q_i over {N_MASS} samples...")
q_inlet_all = np.empty(N_MASS)
q_outlet_all = np.empty(N_MASS)
q_range_all = np.empty(N_MASS)
q_std_all = np.empty(N_MASS)
for i in range(N_MASS):
    xi = np.asarray(x_all[i], dtype=np.float64)
    yi = np.asarray(y_all[i], dtype=np.float64)
    ui = np.asarray(q_all[i, 0], dtype=np.float64)
    vi = np.asarray(q_all[i, 1], dtype=np.float64)
    q_profile = streamwise_flux_profile(xi, yi, ui, vi)
    q_inlet_all[i] = q_profile[0]
    q_outlet_all[i] = q_profile[-1]
    q_range_all[i] = q_profile.max() - q_profile.min()
    q_std_all[i] = q_profile.std()

# Inlet flux is the reference. Relative spread (max - min)/|Q_inlet| tells you
# what fraction of the through-flow is being "lost" or "gained" along the pipe.
rel_range = q_range_all / np.abs(q_inlet_all).clip(min=1e-12)
rel_std = q_std_all / np.abs(q_inlet_all).clip(min=1e-12)
rel_outlet = (q_outlet_all - q_inlet_all) / np.abs(q_inlet_all).clip(min=1e-12)
print(
    f"  Q_inlet                 : mean={q_inlet_all.mean():+.3e}  std={q_inlet_all.std():.3e}"
)
print(
    f"  Q_outlet                : mean={q_outlet_all.mean():+.3e}  std={q_outlet_all.std():.3e}"
)
print(
    f"  (Q_outlet - Q_inlet)/Q_inlet: mean={rel_outlet.mean():+.2%}  |·| mean={np.abs(rel_outlet).mean():.2%}"
)
print(
    f"  (max Q_i - min Q_i)/|Q_inlet|: mean={rel_range.mean():.2%}  median={np.median(rel_range):.2%}"
)
print(
    f"  std(Q_i)/|Q_inlet|            : mean={rel_std.mean():.2%}  median={np.median(rel_std):.2%}"
)

# Visualise.
xs0 = np.asarray(x_all[SAMPLE_IDX], dtype=np.float64)
ys0 = np.asarray(y_all[SAMPLE_IDX], dtype=np.float64)
u0 = np.asarray(q_all[SAMPLE_IDX, 0], dtype=np.float64)
v0 = np.asarray(q_all[SAMPLE_IDX, 1], dtype=np.float64)
q_profile_0 = streamwise_flux_profile(xs0, ys0, u0, v0)
i_axis = np.arange(H)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))

ax = axes[0]
ax.plot(i_axis, q_profile_0, color=plt.get_cmap(CMAP)(0.75), lw=1.6)
ax.axhline(
    q_profile_0[0],
    color="0.25",
    lw=0.8,
    linestyle="--",
    label=f"$Q_\\mathrm{{inlet}}={q_profile_0[0]:.3e}$",
)
ax.set_xlabel("streamwise index $i$")
ax.set_ylabel("$Q_i$")
_rel = (q_profile_0.max() - q_profile_0.min()) / max(abs(q_profile_0[0]), 1e-12)
ax.set_title(
    f"sample {SAMPLE_IDX}: cross-section flux\n"
    rf"$(Q_\mathrm{{max}}-Q_\mathrm{{min}})/|Q_\mathrm{{inlet}}|$ = {_rel:.2%}",
    fontsize=10,
)
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.22)

ax = axes[1]
sample_indices = np.linspace(0, N - 1, 8, dtype=int)
for idx in sample_indices:
    xi = np.asarray(x_all[idx], dtype=np.float64)
    yi = np.asarray(y_all[idx], dtype=np.float64)
    ui = np.asarray(q_all[idx, 0], dtype=np.float64)
    vi = np.asarray(q_all[idx, 1], dtype=np.float64)
    qp = streamwise_flux_profile(xi, yi, ui, vi)
    ax.plot(i_axis, qp / max(abs(qp[0]), 1e-12), lw=0.9, alpha=0.7)
ax.axhline(1.0, color="0.25", lw=0.8, linestyle="--", label=r"$Q_i/Q_\mathrm{inlet}=1$")
ax.set_xlabel("streamwise index $i$")
ax.set_ylabel(r"$Q_i / Q_\mathrm{inlet}$")
ax.set_title(f"Normalised flux profile, {len(sample_indices)} samples")
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.22)

ax = axes[2]
ax.hist(
    rel_range * 100,
    bins=40,
    color=plt.get_cmap(CMAP)(0.55),
    alpha=0.85,
    edgecolor="white",
    linewidth=0.4,
)
ax.axvline(
    rel_range.mean() * 100,
    color=plt.get_cmap(CMAP)(0.85),
    lw=1.5,
    label=f"mean = {rel_range.mean():.2%}",
)
ax.axvline(
    np.median(rel_range) * 100,
    color="0.25",
    lw=1.2,
    linestyle="--",
    label=f"median = {np.median(rel_range):.2%}",
)
ax.set_xlabel(r"$(Q_\mathrm{max} - Q_\mathrm{min})/|Q_\mathrm{inlet}|$  [%]")
ax.set_ylabel("count")
ax.set_title(f"Relative flux spread across {N_MASS} samples")
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.22)

fig.tight_layout()
out_path = FIGURES_DIR / "pipe_mass_conservation.png"
fig.savefig(out_path, bbox_inches="tight")
if IS_NOTEBOOK:
    plt.show()
else:
    plt.close(fig)
print(f"Saved to {out_path}")


# %% Divergence heatmaps for multiple samples
DIV_SAMPLE_INDICES = PROFILE_SAMPLE_INDICES

fig, axes = plt.subplots(
    1, len(DIV_SAMPLE_INDICES), figsize=(5.0 * len(DIV_SAMPLE_INDICES), 4.0)
)
if len(DIV_SAMPLE_INDICES) == 1:
    axes = [axes]
for ax, idx in zip(axes, DIV_SAMPLE_INDICES):
    xi = np.asarray(x_all[idx], dtype=np.float64)
    yi = np.asarray(y_all[idx], dtype=np.float64)
    ui = np.asarray(q_all[idx, 0], dtype=np.float64)
    vi = np.asarray(q_all[idx, 1], dtype=np.float64)
    div_i, _, _, _ = compute_divergence_field(xi, yi, ui, vi)
    v_abs_i = float(np.percentile(np.abs(div_i), 99))
    im = ax.pcolormesh(
        xi, yi, div_i, shading="gouraud", cmap="RdBu_r", vmin=-v_abs_i, vmax=v_abs_i
    )
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\nabla\cdot u$")
    for sl in EDGE_SLICES.values():
        ax.plot(xi[sl], yi[sl], color="0.15", lw=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(
        f"sample {idx}: $\\nabla\\cdot u$\n"
        rf"$|\nabla\cdot u|_\mathrm{{mean}}={np.abs(div_i).mean():.2e}$,"
        rf"  rmse$={float(np.sqrt(np.mean(div_i**2))):.2e}$",
        fontsize=10,
    )

fig.tight_layout()
out_path = FIGURES_DIR / "pipe_divergence_samples.png"
fig.savefig(out_path, bbox_inches="tight")
if IS_NOTEBOOK:
    plt.show()
else:
    plt.close(fig)
print(f"Saved to {out_path}")


# %% Mean streamwise flux Q_i averaged across all samples (with percentile bands)
print(f"Computing per-sample Q_i profiles across {N_MASS} samples for averaged plot...")
q_profiles_all = np.empty((N_MASS, H), dtype=np.float64)
for i in range(N_MASS):
    xi = np.asarray(x_all[i], dtype=np.float64)
    yi = np.asarray(y_all[i], dtype=np.float64)
    ui = np.asarray(q_all[i, 0], dtype=np.float64)
    vi = np.asarray(q_all[i, 1], dtype=np.float64)
    q_profiles_all[i] = streamwise_flux_profile(xi, yi, ui, vi)

# Normalise each profile by its own inlet flux so samples with different scales
# can be averaged meaningfully. Sign-correct by inlet direction.
inlet_ref = q_profiles_all[:, 0]
sign = np.where(inlet_ref >= 0, 1.0, -1.0)
q_profiles_norm = (q_profiles_all * sign[:, None]) / np.abs(inlet_ref).clip(min=1e-12)[
    :, None
]

q_mean = q_profiles_norm.mean(axis=0)
q_p05 = np.quantile(q_profiles_norm, 0.05, axis=0)
q_p95 = np.quantile(q_profiles_norm, 0.95, axis=0)
q_p25 = np.quantile(q_profiles_norm, 0.25, axis=0)
q_p75 = np.quantile(q_profiles_norm, 0.75, axis=0)

fig, ax = plt.subplots(figsize=(6.0, 4.0))
band_color = plt.get_cmap(CMAP)(0.55)
line_color = plt.get_cmap(CMAP)(0.85)
ax.fill_between(
    i_axis, q_p05, q_p95, color=band_color, alpha=0.18, linewidth=0, label="5–95%"
)
# ax.fill_between(
#     i_axis, q_p25, q_p75, color=band_color, alpha=0.32, linewidth=0, label="25–75%"
# )
ax.plot(i_axis, q_mean, color=line_color, lw=1.8, label="mean")
ax.axhline(1.0, color="0.25", lw=0.8, linestyle="--", label=r"$Q_i/Q_\mathrm{inlet}=1$")
ax.set_xlabel("streamwise index $i$")
ax.set_ylabel(r"$Q_i / Q_\mathrm{inlet}$")
ax.set_title(f"Normalised flux $Q_i/Q_\\mathrm{{inlet}}$ across {N_MASS} samples")
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.22)
fig.tight_layout()
out_path = FIGURES_DIR / "pipe_mass_conservation_mean.png"
fig.savefig(out_path, bbox_inches="tight")
if IS_NOTEBOOK:
    plt.show()
else:
    plt.close(fig)
print(f"Saved to {out_path}")


# %% Cross-section flux Q_i painted on the pipe geometry for a few samples
FLUX_SAMPLE_INDICES = PROFILE_SAMPLE_INDICES

fig, axes = plt.subplots(
    1, len(FLUX_SAMPLE_INDICES), figsize=(5.0 * len(FLUX_SAMPLE_INDICES), 4.0)
)
if len(FLUX_SAMPLE_INDICES) == 1:
    axes = [axes]
for ax, idx in zip(axes, FLUX_SAMPLE_INDICES):
    xi = np.asarray(x_all[idx], dtype=np.float64)
    yi = np.asarray(y_all[idx], dtype=np.float64)
    ui = np.asarray(q_all[idx, 0], dtype=np.float64)
    vi = np.asarray(q_all[idx, 1], dtype=np.float64)
    qp = streamwise_flux_profile(xi, yi, ui, vi)  # (H,)
    q_field = np.broadcast_to(qp[:, None], xi.shape)
    v_abs_q = float(np.max(np.abs(qp)))
    v_abs_q = v_abs_q if v_abs_q > 1e-12 else 1.0
    im = ax.pcolormesh(
        xi, yi, q_field, shading="gouraud", cmap="RdBu_r", vmin=-v_abs_q, vmax=v_abs_q
    )
    fig.colorbar(im, ax=ax, shrink=0.85, label="$Q_i$")
    for sl in EDGE_SLICES.values():
        ax.plot(xi[sl], yi[sl], color="0.15", lw=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    _rel = (qp.max() - qp.min()) / max(abs(qp[0]), 1e-12)
    ax.set_title(
        f"sample {idx}: cross-section flux\n"
        rf"$Q_\mathrm{{inlet}}={qp[0]:.2e}$,  "
        rf"$(Q_\mathrm{{max}}-Q_\mathrm{{min}})/|Q_\mathrm{{inlet}}|$ = {_rel:.2%}",
        fontsize=10,
    )
fig.tight_layout()
out_path = FIGURES_DIR / "pipe_mass_conservation_samples.png"
fig.savefig(out_path, bbox_inches="tight")
if IS_NOTEBOOK:
    plt.show()
else:
    plt.close(fig)
print(f"Saved to {out_path}")


# %% Mass conservation on a trained Transolver (unconstrained baseline)
# Reuses streamwise_flux_profile from above on predicted (u_x, u_y) instead of
# GT, to see whether a trained baseline that has no flux constraint actually
# learns mass conservation as well as the (already-imperfect) ground truth.
from omni_hc.benchmarks.pipe import adapter as pipe_adapter
from omni_hc.benchmarks.pipe.data import build_test_loader as build_pipe_test_loader
from omni_hc.benchmarks.pipe.data import (
    build_train_val_loaders as build_pipe_train_val_loaders,
)
from omni_hc.core import compose_run_config, load_yaml_file
from omni_hc.integrations.nsl.modeling import create_model
from omni_hc.training.common import (
    build_optimizer,
    forward_with_optional_aux,
    load_checkpoint_state,
    load_model_state_dict,
    relative_l2_per_sample,
)
from omni_hc.training.reproducibility import seed_everything, training_seed

# %% Qualitative pipe predictions - GT / prediction / error across models
QUALITATIVE_SAMPLE_INDICES = (0, 100)
QUALITATIVE_MODEL_SPECS = (
    ("Unconstrained", "outputs/pipe/none/transolver/{budget}/seed_42"),
    (
        "$u_x$ ansatz",
        "outputs/pipe/pipe_ux_boundary_ansatz/transolver/{budget}/seed_42",
    ),
    (
        "Stream ansatz",
        "outputs/pipe/pipe_stream_function_boundary_ansatz/transolver/{budget}/seed_42",
    ),
)
QUALITATIVE_BUDGETS = (
    ("e50_t500", "50 epochs / 500 train samples", "pipe_qualitative_e50.png"),
    ("final", "500 epochs / 1000 train samples", "pipe_qualitative_final.png"),
)


def _move_normalizer_to_device(loader, name, device):
    normalizer = getattr(loader, name, None)
    return normalizer.to(device) if normalizer is not None else None


def _decode_tensor(normalizer, tensor):
    return normalizer.decode(tensor) if normalizer is not None else tensor


def _configure_pipe_constraint_for_plot(
    model, *, meta, x_normalizer, y_normalizer, uy_normalizer
):
    constraint = getattr(model, "constraint", None)
    if constraint is None:
        return
    if y_normalizer is not None and hasattr(constraint, "set_target_normalizer"):
        constraint.set_target_normalizer(y_normalizer)
    if x_normalizer is not None and hasattr(constraint, "set_input_normalizer"):
        constraint.set_input_normalizer(x_normalizer)
    if uy_normalizer is not None and hasattr(constraint, "set_uy_normalizer"):
        constraint.set_uy_normalizer(uy_normalizer)
    if hasattr(constraint, "set_grid_shape"):
        constraint.set_grid_shape(tuple(meta["shapelist"]))
    if hasattr(constraint, "set_domain_bounds"):
        domain_bounds = meta.get("domain_bounds")
        if domain_bounds is not None and len(domain_bounds) == 2:
            constraint.set_domain_bounds(
                float(domain_bounds[0]), float(domain_bounds[1])
            )


def load_pipe_predictions_for_samples(run_dir, sample_indices):
    """Load decoded x/y mesh, GT ux, and predicted ux for test samples."""
    ckpt_path = run_dir / "best.pt"
    cfg_path = run_dir / "resolved_config.yaml"
    if not ckpt_path.exists() or not cfg_path.exists():
        return None

    cfg = load_yaml_file(cfg_path)
    cfg.setdefault("paths", {})["output_dir"] = str(run_dir)
    cfg["paths"]["root_dir"] = str(DATA_DIR)
    cfg.setdefault("data", {})["load_uy"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_pipe_test_loader(cfg)
    meta = pipe_adapter._get_meta(loader)
    h_grid, w_grid = tuple(meta["shapelist"])

    x_normalizer = _move_normalizer_to_device(loader, "x_normalizer", device)
    y_normalizer = _move_normalizer_to_device(loader, "y_normalizer", device)
    uy_normalizer = _move_normalizer_to_device(loader, "uy_normalizer", device)

    model, _, _ = create_model(
        cfg, device=device, runtime_overrides=pipe_adapter._runtime_overrides(meta)
    )
    _configure_pipe_constraint_for_plot(
        model,
        meta=meta,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        uy_normalizer=uy_normalizer,
    )
    ckpt = load_checkpoint_state(ckpt_path, device=device)
    load_model_state_dict(model, ckpt["model_state_dict"])
    model.eval()

    wanted = set(int(i) for i in sample_indices)
    found = {}
    seen = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch["coords"].shape[0])
            needed = [i for i in range(batch_size) if seen + i in wanted]
            if not needed:
                seen += batch_size
                continue

            coords = batch["coords"].to(device)
            fx = batch["x"].to(device)
            uy_target = batch.get("y_uy")
            if uy_target is not None:
                uy_target = uy_target.to(device)
            out = forward_with_optional_aux(model, coords, fx, uy_target=uy_target)
            pred_decoded = _decode_tensor(y_normalizer, out["pred"])
            target_decoded = _decode_tensor(y_normalizer, batch["y"].to(device))
            coords_decoded = _decode_tensor(x_normalizer, coords)

            for local_idx in needed:
                sample_idx = seen + local_idx
                found[sample_idx] = {
                    "x": coords_decoded[local_idx, :, 0]
                    .reshape(h_grid, w_grid)
                    .detach()
                    .cpu()
                    .numpy(),
                    "y": coords_decoded[local_idx, :, 1]
                    .reshape(h_grid, w_grid)
                    .detach()
                    .cpu()
                    .numpy(),
                    "gt": target_decoded[local_idx, :, 0]
                    .reshape(h_grid, w_grid)
                    .detach()
                    .cpu()
                    .numpy(),
                    "pred": pred_decoded[local_idx, :, 0]
                    .reshape(h_grid, w_grid)
                    .detach()
                    .cpu()
                    .numpy(),
                }
            if wanted.issubset(found):
                break
            seen += batch_size
    return found


def _plot_pipe_qual_field(ax, x_grid, y_grid, values, *, cmap, vmin, vmax):
    mesh = ax.pcolormesh(
        x_grid,
        y_grid,
        values,
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.plot(x_grid[0, :], y_grid[0, :], color="0.18", lw=0.45)
    ax.plot(x_grid[-1, :], y_grid[-1, :], color="0.18", lw=0.45)
    ax.plot(x_grid[:, 0], y_grid[:, 0], color="0.18", lw=0.45)
    ax.plot(x_grid[:, -1], y_grid[:, -1], color="0.18", lw=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def plot_pipe_qualitative_budget(budget, title, out_name, *, sample_indices):
    loaded = []
    for model_label, path_template in QUALITATIVE_MODEL_SPECS:
        run_dir = REPO_ROOT / path_template.format(budget=budget)
        samples = load_pipe_predictions_for_samples(run_dir, sample_indices)
        loaded.append((model_label, run_dir, samples))
        if samples is None:
            print(f"{title} | {model_label}: missing checkpoint/config at {run_dir}")
        else:
            print(f"{title} | {model_label}: loaded {len(samples)} sample(s)")

    available = [
        sample
        for _label, _run_dir, samples in loaded
        if samples is not None
        for sample in samples.values()
    ]
    if not available:
        print(f"{title}: no available predictions; skipping qualitative plot")
        return None

    field_values = np.concatenate(
        [np.ravel(sample["gt"]) for sample in available]
        + [np.ravel(sample["pred"]) for sample in available]
    )
    field_vmin = float(np.nanmin(field_values))
    field_vmax = float(np.nanmax(field_values))
    error_scale = max(
        float(np.nanmax(np.abs(sample["pred"] - sample["gt"]))) for sample in available
    )
    error_scale = max(error_scale, 1e-12)

    row_count = len(sample_indices) * len(QUALITATIVE_MODEL_SPECS)
    fig, axes = plt.subplots(
        row_count,
        3,
        figsize=(10.2, 2.15 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    for sample_pos, sample_idx in enumerate(sample_indices):
        for model_pos, (model_label, _run_dir, samples) in enumerate(loaded):
            row = sample_pos * len(QUALITATIVE_MODEL_SPECS) + model_pos
            row_axes = axes[row]
            row_axes[0].set_ylabel(
                f"sample {sample_idx}\n{model_label}",
                rotation=0,
                ha="right",
                va="center",
                labelpad=48,
                fontsize=9,
            )
            if samples is None or sample_idx not in samples:
                for ax in row_axes:
                    ax.axis("off")
                    ax.text(
                        0.5,
                        0.5,
                        "missing run",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="0.35",
                    )
                continue

            sample = samples[sample_idx]
            error = sample["pred"] - sample["gt"]
            field_mesh = _plot_pipe_qual_field(
                row_axes[0],
                sample["x"],
                sample["y"],
                sample["gt"],
                cmap=CMAP,
                vmin=field_vmin,
                vmax=field_vmax,
            )
            _plot_pipe_qual_field(
                row_axes[1],
                sample["x"],
                sample["y"],
                sample["pred"],
                cmap=CMAP,
                vmin=field_vmin,
                vmax=field_vmax,
            )
            error_mesh = _plot_pipe_qual_field(
                row_axes[2],
                sample["x"],
                sample["y"],
                error,
                cmap="coolwarm",
                vmin=-error_scale,
                vmax=error_scale,
            )
            fig.colorbar(field_mesh, ax=row_axes[:2].tolist(), shrink=0.82, pad=0.01)
            fig.colorbar(error_mesh, ax=row_axes[2], shrink=0.82, pad=0.01)

    for ax, col_title in zip(axes[0], ("GT $u_x$", "Pred $u_x$", "Pred - GT")):
        ax.set_title(col_title, fontsize=11)
    fig.suptitle(f"Pipe qualitative predictions ({title})", fontsize=13)

    out_path = FIGURES_DIR / out_name
    fig.savefig(out_path, bbox_inches="tight")
    if IS_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved to {out_path}")
    return out_path


for qualitative_budget, qualitative_title, qualitative_out_name in QUALITATIVE_BUDGETS:
    plot_pipe_qualitative_budget(
        qualitative_budget,
        qualitative_title,
        qualitative_out_name,
        sample_indices=QUALITATIVE_SAMPLE_INDICES,
    )


# %% Fast learning trace — one epoch, unconstrained vs stream-function ansatz
# Enable this cell/block with:
#   OMNI_HC_RUN_FAST_PIPE_QUAL=1 python scripts/notebooks/pipe.py
# or set RUN_FAST_PIPE_QUALITATIVE = True in a notebook before running this section.
RUN_FAST_PIPE_QUALITATIVE = True
RUN_FAST_PIPE_QUALITATIVE = bool(
    int(os.environ.get("OMNI_HC_RUN_FAST_PIPE_QUAL", "0"))
) or bool(globals().get("RUN_FAST_PIPE_QUALITATIVE", False))
FAST_QUALITATIVE_BACKBONE = "FNO"
FAST_QUALITATIVE_SAMPLE_IDX = 0
FAST_QUALITATIVE_NTRAIN = 100
FAST_QUALITATIVE_BATCH_SIZE = 2
FAST_QUALITATIVE_OUTPUT_ROOT = Path(
    os.environ.get(
        "OMNI_HC_FAST_PIPE_QUAL_OUTPUT_ROOT",
        "/private/tmp/omni_hc_pipe_fast_qualitative",
    )
)
FAST_QUALITATIVE_STORYBOARD_NAME = "pipe_fast_learning_trace_epoch1.png"
FAST_QUALITATIVE_GIF_NAME = "pipe_fast_learning_trace_epoch1.gif"
FAST_QUALITATIVE_GIF_FPS = 5


def _fast_pipe_qualitative_overrides(output_dir: Path) -> dict:
    return {
        "paths": {
            "root_dir": str(DATA_DIR),
            "output_dir": str(output_dir),
        },
        "data": {
            "ntrain": FAST_QUALITATIVE_NTRAIN,
            "ntest": 16,
            "load_uy": True,
        },
        "model": {
            "args": {
                "n_hidden": 32,
                "n_heads": 4,
                "n_layers": 2,
                "modes": 8,
            },
        },
        "training": {
            "batch_size": FAST_QUALITATIVE_BATCH_SIZE,
            "val_size": 0,
            "num_epochs": 1,
            "scheduler": "none",
            "seed": 42,
        },
        "evaluation": {
            "batch_size": 2,
        },
        "wandb_logging": {
            "wandb": False,
            "log_every": None,
            "image_log_every": None,
        },
    }


def _compose_fast_pipe_qualitative_config(*, label: str, constraint: str | None):
    output_dir = FAST_QUALITATIVE_OUTPUT_ROOT / label / "seed_42"
    return compose_run_config(
        benchmark="pipe",
        backbone=FAST_QUALITATIVE_BACKBONE,
        constraint=constraint,
        budget="debug",
        mode="train",
        seed=42,
        extra_overrides=_fast_pipe_qualitative_overrides(output_dir),
    )


def _select_fast_pipe_test_sample(loader, sample_idx: int, device: torch.device):
    sample = loader.dataset[int(sample_idx)]
    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if key in {"coords", "x", "y", "y_uy"}
    }
    return batch


def _capture_fast_pipe_prediction(
    model,
    *,
    sample_batch,
    x_normalizer,
    y_normalizer,
    h_grid: int,
    w_grid: int,
    label: str,
    step: int,
    train_loss: float | None = None,
):
    model.eval()
    with torch.no_grad():
        uy_target = sample_batch.get("y_uy")
        out = forward_with_optional_aux(
            model,
            sample_batch["coords"],
            sample_batch["x"],
            uy_target=uy_target,
        )
        pred_decoded = _decode_tensor(y_normalizer, out["pred"])
        target_decoded = _decode_tensor(y_normalizer, sample_batch["y"])
        coords_decoded = _decode_tensor(x_normalizer, sample_batch["coords"])
        rel_l2 = float(
            relative_l2_per_sample(pred_decoded, target_decoded).mean().item()
        )

    return {
        "label": label,
        "step": int(step),
        "train_loss": train_loss,
        "rel_l2": rel_l2,
        "x": coords_decoded[0, :, 0].reshape(h_grid, w_grid).detach().cpu().numpy(),
        "y": coords_decoded[0, :, 1].reshape(h_grid, w_grid).detach().cpu().numpy(),
        "target": target_decoded[0, :, 0]
        .reshape(h_grid, w_grid)
        .detach()
        .cpu()
        .numpy(),
        "pred": pred_decoded[0, :, 0].reshape(h_grid, w_grid).detach().cpu().numpy(),
    }


def train_fast_pipe_learning_trace():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_specs = (
        (
            "Unconstrained",
            "unconstrained",
            None,
        ),
        (
            "Stream ansatz",
            "stream_ansatz",
            "pipe_stream_function_boundary_ansatz",
        ),
    )
    traces = {}
    for model_label, run_label, constraint_name in run_specs:
        cfg = _compose_fast_pipe_qualitative_config(
            label=run_label,
            constraint=constraint_name,
        )
        seed_everything(training_seed(cfg))
        print(
            f"Training fast pipe learning trace: {model_label} "
            f"({FAST_QUALITATIVE_NTRAIN} samples, one epoch)"
        )
        train_loader, _val_loader = build_pipe_train_val_loaders(cfg)
        test_loader = build_pipe_test_loader(cfg)
        meta = pipe_adapter._get_meta(train_loader)
        h_grid, w_grid = tuple(meta["shapelist"])

        x_normalizer = _move_normalizer_to_device(train_loader, "x_normalizer", device)
        y_normalizer = _move_normalizer_to_device(train_loader, "y_normalizer", device)
        uy_normalizer = _move_normalizer_to_device(
            train_loader, "uy_normalizer", device
        )

        model, _model_args, _resolved_nsl_root = create_model(
            cfg,
            device=device,
            runtime_overrides=pipe_adapter._runtime_overrides(meta),
        )
        _configure_pipe_constraint_for_plot(
            model,
            meta=meta,
            x_normalizer=x_normalizer,
            y_normalizer=y_normalizer,
            uy_normalizer=uy_normalizer,
        )

        optimizer = build_optimizer(model, cfg.get("training", {}))
        sample_batch = _select_fast_pipe_test_sample(
            test_loader, FAST_QUALITATIVE_SAMPLE_IDX, device
        )
        trace = [
            _capture_fast_pipe_prediction(
                model,
                sample_batch=sample_batch,
                x_normalizer=x_normalizer,
                y_normalizer=y_normalizer,
                h_grid=h_grid,
                w_grid=w_grid,
                label="init",
                step=0,
            )
        ]

        model.train()
        for step, batch in enumerate(train_loader, start=1):
            coords = batch["coords"].to(device)
            fx = batch["x"].to(device)
            target = batch["y"].to(device)
            uy_target = batch.get("y_uy")
            if uy_target is not None:
                uy_target = uy_target.to(device)

            out = forward_with_optional_aux(
                model,
                coords,
                fx,
                uy_target=uy_target,
            )
            pred_decoded = _decode_tensor(y_normalizer, out["pred"])
            target_decoded = _decode_tensor(y_normalizer, target)
            loss = torch.nn.functional.mse_loss(pred_decoded, target_decoded)
            if out.get("extra_loss") is not None:
                loss = loss + out["extra_loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            trace.append(
                _capture_fast_pipe_prediction(
                    model,
                    sample_batch=sample_batch,
                    x_normalizer=x_normalizer,
                    y_normalizer=y_normalizer,
                    h_grid=h_grid,
                    w_grid=w_grid,
                    label=f"step {step}",
                    step=step,
                    train_loss=float(loss.detach().item()),
                )
            )
            model.train()

        traces[model_label] = trace
        print(
            f"{model_label}: captured {len(trace)} frames; "
            f"final rel. L2={trace[-1]['rel_l2']:.3f}"
        )
    return traces


def _fast_trace_field_ranges(traces):
    all_frames = [frame for trace in traces.values() for frame in trace]
    field_values = np.concatenate(
        [np.ravel(frame["target"]) for frame in all_frames]
        + [np.ravel(frame["pred"]) for frame in all_frames]
    )
    field_vmin = float(np.nanmin(field_values))
    field_vmax = float(np.nanmax(field_values))
    error_scale = max(
        float(np.nanmax(np.abs(frame["pred"] - frame["target"])))
        for frame in all_frames
    )
    return field_vmin, field_vmax, max(error_scale, 1e-12)


def plot_fast_pipe_learning_storyboard(traces, *, out_path: Path):
    field_vmin, field_vmax, error_scale = _fast_trace_field_ranges(traces)
    max_frames = max(len(trace) for trace in traces.values())
    model_labels = list(traces)
    fig, axes = plt.subplots(
        len(model_labels),
        max_frames,
        figsize=(2.4 * max_frames, 2.35 * len(model_labels)),
        squeeze=False,
        constrained_layout=True,
    )
    mesh = None
    for row, model_label in enumerate(model_labels):
        trace = traces[model_label]
        for col in range(max_frames):
            ax = axes[row, col]
            if col >= len(trace):
                ax.axis("off")
                continue
            frame = trace[col]
            mesh = _plot_pipe_qual_field(
                ax,
                frame["x"],
                frame["y"],
                frame["pred"],
                cmap=CMAP,
                vmin=field_vmin,
                vmax=field_vmax,
            )
            ax.set_title(
                f"{frame['label']}\nrel. $L_2$={frame['rel_l2']:.2f}",
                fontsize=9,
            )
        axes[row, 0].set_ylabel(
            model_label,
            rotation=0,
            ha="right",
            va="center",
            labelpad=54,
            fontsize=10,
        )
    fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.52, pad=0.01)
    fig.suptitle(
        "One-epoch pipe learning trace on one held-out sample",
        fontsize=13,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    if IS_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved to {out_path}")
    return out_path


def _draw_fast_pipe_learning_frame(
    traces, frame_idx, *, field_vmin, field_vmax, error_scale
):
    model_labels = list(traces)
    fig = plt.figure(figsize=(12.2, 2.6), facecolor="white")
    row_y = [0.55, 0.29]
    pred_left = 0.19
    err_left = 0.52
    field_width = 0.23
    field_height = 0.22
    axes = np.empty((len(model_labels), 2), dtype=object)
    field_mesh = None
    error_mesh = None
    for row, model_label in enumerate(model_labels):
        trace = traces[model_label]
        frame = trace[min(frame_idx, len(trace) - 1)]
        error = frame["pred"] - frame["target"]
        axes[row, 0] = fig.add_axes(
            [pred_left, row_y[row], field_width, field_height]
        )
        axes[row, 1] = fig.add_axes([err_left, row_y[row], field_width, field_height])
        for ax in axes[row]:
            ax.set_facecolor("white")
        field_mesh = _plot_pipe_qual_field(
            axes[row, 0],
            frame["x"],
            frame["y"],
            frame["pred"],
            cmap=CMAP,
            vmin=field_vmin,
            vmax=field_vmax,
        )
        error_mesh = _plot_pipe_qual_field(
            axes[row, 1],
            frame["x"],
            frame["y"],
            error,
            cmap="coolwarm",
            vmin=-error_scale,
            vmax=error_scale,
        )
        fig.text(
            0.135,
            row_y[row] + 0.5 * field_height,
            f"{model_label}\n{frame['label']}\nrel. $L_2$={frame['rel_l2']:.3f}",
            ha="right",
            va="center",
            color="#4f4f55",
            fontsize=11,
        )
    fig.text(
        pred_left + 0.5 * field_width,
        0.86,
        "Pred $u_x$",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.text(
        err_left + 0.5 * field_width,
        0.86,
        "Pred - GT",
        ha="center",
        va="center",
        fontsize=11,
    )
    field_cax = fig.add_axes([0.445, 0.31, 0.008, 0.42])
    error_cax = fig.add_axes([0.785, 0.31, 0.008, 0.42])
    field_bar = fig.colorbar(
        field_mesh,
        cax=field_cax,
    )
    error_bar = fig.colorbar(
        error_mesh,
        cax=error_cax,
    )
    for cbar in (field_bar, error_bar):
        cbar.ax.tick_params(labelsize=7, length=2, pad=1)
    ax_curve = fig.add_axes([0.835, 0.28, 0.135, 0.50])
    ax_curve.set_facecolor("white")
    curve_colors = {
        model_labels[0]: plt.get_cmap(CMAP)(0.85),
        model_labels[1]: EDGE_COLORS["inlet"],
    }
    for model_label in model_labels:
        trace = traces[model_label]
        shown_idx = min(frame_idx, len(trace) - 1)
        steps = [f["step"] for f in trace[: shown_idx + 1]]
        values = [f["rel_l2"] for f in trace[: shown_idx + 1]]
        ax_curve.plot(
            steps,
            values,
            marker="o",
            markersize=2.6,
            linewidth=1.5,
            color=curve_colors.get(model_label),
            label="Base" if model_label == model_labels[0] else "Stream",
        )
    all_steps = [f["step"] for trace in traces.values() for f in trace]
    all_values = [f["rel_l2"] for trace in traces.values() for f in trace]
    ax_curve.set_xlim(min(all_steps), max(all_steps))
    value_pad = 0.06 * max(max(all_values) - min(all_values), 1e-12)
    ax_curve.set_ylim(min(all_values) - value_pad, max(all_values) + value_pad)
    ax_curve.set_title("Test rel. $L_2$", fontsize=9, pad=4)
    ax_curve.set_xlabel("step", fontsize=8, labelpad=1)
    ax_curve.tick_params(axis="both", labelsize=7, length=2, pad=1)
    ax_curve.grid(color="0.88", linewidth=0.55)
    ax_curve.legend(frameon=False, loc="upper right", fontsize=7, handlelength=1.2)
    return fig


def save_fast_pipe_learning_gif(traces, *, out_path: Path):
    field_vmin, field_vmax, error_scale = _fast_trace_field_ranges(traces)
    max_frames = max(len(trace) for trace in traces.values())
    frames = []
    for frame_idx in range(max_frames):
        fig = _draw_fast_pipe_learning_frame(
            traces,
            frame_idx,
            field_vmin=field_vmin,
            field_vmax=field_vmax,
            error_scale=error_scale,
        )
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        from PIL import Image

        frames.append(Image.fromarray(rgba))
        if IS_NOTEBOOK:
            plt.show()
        else:
            plt.close(fig)
    saved = _save_rgba_animation(frames, out_path, fps=FAST_QUALITATIVE_GIF_FPS)
    print(f"Saved to {saved}")
    return saved


def plot_fast_pipe_learning_final_2x2(traces, *, out_path: Path):
    loaded = [(model_label, trace[-1]) for model_label, trace in traces.items()]
    field_vmin, field_vmax, error_scale = _fast_trace_field_ranges(traces)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 4.9), constrained_layout=True)
    field_mesh = None
    error_mesh = None
    for col, (model_label, frame) in enumerate(loaded):
        error = frame["pred"] - frame["target"]
        field_mesh = _plot_pipe_qual_field(
            axes[0, col],
            frame["x"],
            frame["y"],
            frame["pred"],
            cmap=CMAP,
            vmin=field_vmin,
            vmax=field_vmax,
        )
        axes[0, col].set_title(f"{model_label} final prediction", fontsize=11)
        error_mesh = _plot_pipe_qual_field(
            axes[1, col],
            frame["x"],
            frame["y"],
            error,
            cmap="coolwarm",
            vmin=-error_scale,
            vmax=error_scale,
        )
        axes[1, col].set_title(
            f"error, rel. $L_2$={frame['rel_l2']:.3f}",
            fontsize=11,
        )

    fig.colorbar(field_mesh, ax=axes[0, :].tolist(), shrink=0.82, pad=0.01)
    error_bar = fig.colorbar(error_mesh, ax=axes[1, :].tolist(), shrink=0.82, pad=0.01)
    error_bar.set_label("pred - GT", fontsize=9)
    fig.suptitle(
        "Pipe flow: one-epoch qualitative comparison "
        f"({FAST_QUALITATIVE_BACKBONE}, {FAST_QUALITATIVE_NTRAIN} train samples)",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    if IS_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved to {out_path}")
    return out_path


if RUN_FAST_PIPE_QUALITATIVE:
    fast_learning_traces = train_fast_pipe_learning_trace()
    fast_storyboard_path = plot_fast_pipe_learning_storyboard(
        fast_learning_traces,
        out_path=FIGURES_DIR / FAST_QUALITATIVE_STORYBOARD_NAME,
    )
    fast_learning_gif_path = save_fast_pipe_learning_gif(
        fast_learning_traces,
        out_path=FIGURES_DIR / FAST_QUALITATIVE_GIF_NAME,
    )


# %% Mass conservation on saved unconstrained Transolver checkpoints
candidates = sorted(REPO_ROOT.glob("outputs/pipe/none/transolver/*/seed_*/best.pt"))
print(f"Found {len(candidates)} trained transolver pipe checkpoints (constraint=none):")
for c in candidates:
    print(f"  {c.relative_to(REPO_ROOT)}")

if not candidates:
    print("No trained Transolver pipe checkpoint found; skipping.")
else:
    ckpt_path = candidates[-1]
    cfg_path = ckpt_path.parent / "resolved_config.yaml"
    print(f"\nUsing: {ckpt_path.relative_to(REPO_ROOT)}")

    cfg = load_yaml_file(cfg_path)
    cfg.setdefault("paths", {})["output_dir"] = str(ckpt_path.parent)
    cfg["paths"]["root_dir"] = str(DATA_DIR)
    cfg.setdefault("data", {})["load_uy"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_pipe_test_loader(cfg)
    meta = pipe_adapter._get_meta(loader)
    H_grid, W_grid = tuple(meta["shapelist"])
    y_normalizer = getattr(loader, "y_normalizer", None)
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)
    uy_normalizer = getattr(loader, "uy_normalizer", None)
    if uy_normalizer is not None:
        uy_normalizer = uy_normalizer.to(device)

    model, _, _ = create_model(
        cfg, device=device, runtime_overrides=pipe_adapter._runtime_overrides(meta)
    )
    ckpt = load_checkpoint_state(ckpt_path, device=device)
    load_model_state_dict(model, ckpt["model_state_dict"])
    model.eval()

    rel_range_pred, rel_outlet_pred, q_inlet_pred = [], [], []
    rel_range_gt, rel_outlet_gt = [], []

    with torch.no_grad():
        for batch in loader:
            coords = batch["coords"].to(device)
            fx = batch["x"].to(device)
            out = model(coords, fx)
            if isinstance(out, tuple):
                out = out[0]
            elif isinstance(out, dict):
                out = out.get("pred", out)
            if y_normalizer is not None and out.shape[-1] == 1:
                out_dec = y_normalizer.decode(out)
            else:
                out_dec = out  # multi-channel model output is already physical
            B = out_dec.shape[0]
            C = out_dec.shape[-1]
            grid = out_dec.reshape(B, H_grid, W_grid, C)
            if C >= 2:
                ux_pred = grid[..., 0].cpu().numpy()
                uy_pred = grid[..., 1].cpu().numpy()
            else:
                # Single-channel model: only u_x is predicted; pad u_y with GT to
                # isolate the u_x reconstruction quality on the flux integral.
                ux_pred = grid[..., 0].cpu().numpy()
                uy_pred = batch["y_uy"].reshape(B, H_grid, W_grid).cpu().numpy()

            xy = coords.reshape(B, H_grid, W_grid, 2).cpu().numpy()

            ux_gt = batch["y"].reshape(B, H_grid, W_grid).cpu().numpy()
            uy_gt = batch["y_uy"].reshape(B, H_grid, W_grid).cpu().numpy()
            if y_normalizer is not None:
                ux_gt = (
                    y_normalizer.decode(batch["y"].to(device))
                    .reshape(B, H_grid, W_grid)
                    .cpu()
                    .numpy()
                )
            if uy_normalizer is not None:
                uy_gt = (
                    uy_normalizer.decode(batch["y_uy"].to(device))
                    .reshape(B, H_grid, W_grid)
                    .cpu()
                    .numpy()
                )

            for b in range(B):
                qp = streamwise_flux_profile(
                    xy[b, ..., 0], xy[b, ..., 1], ux_pred[b], uy_pred[b]
                )
                qg = streamwise_flux_profile(
                    xy[b, ..., 0], xy[b, ..., 1], ux_gt[b], uy_gt[b]
                )
                q0 = max(abs(qp[0]), 1e-12)
                rel_range_pred.append((qp.max() - qp.min()) / q0)
                rel_outlet_pred.append((qp[-1] - qp[0]) / q0)
                q_inlet_pred.append(qp[0])
                qg0 = max(abs(qg[0]), 1e-12)
                rel_range_gt.append((qg.max() - qg.min()) / qg0)
                rel_outlet_gt.append((qg[-1] - qg[0]) / qg0)

    rel_range_pred = np.asarray(rel_range_pred)
    rel_outlet_pred = np.asarray(rel_outlet_pred)
    rel_range_gt = np.asarray(rel_range_gt)
    rel_outlet_gt = np.asarray(rel_outlet_gt)
    q_inlet_pred = np.asarray(q_inlet_pred)

    print(f"\nMass conservation on {len(rel_range_pred)} test samples:")
    print(
        f"  Q_inlet (pred)                            : mean={q_inlet_pred.mean():+.3e}  std={q_inlet_pred.std():.3e}"
    )
    print("  Predicted:")
    print(
        f"    (Q_outlet - Q_inlet)/Q_inlet            : mean={rel_outlet_pred.mean():+.2%}  |·| mean={np.abs(rel_outlet_pred).mean():.2%}"
    )
    print(
        f"    (max Q_i - min Q_i)/|Q_inlet|           : mean={rel_range_pred.mean():.2%}  median={np.median(rel_range_pred):.2%}"
    )
    print("  Ground truth (same test split):")
    print(
        f"    (Q_outlet - Q_inlet)/Q_inlet            : mean={rel_outlet_gt.mean():+.2%}  |·| mean={np.abs(rel_outlet_gt).mean():.2%}"
    )
    print(
        f"    (max Q_i - min Q_i)/|Q_inlet|           : mean={rel_range_gt.mean():.2%}  median={np.median(rel_range_gt):.2%}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    bins = np.linspace(
        0, np.percentile(np.r_[rel_range_pred, rel_range_gt] * 100, 99), 40
    )
    ax.hist(
        rel_range_gt * 100,
        bins=bins,
        color="0.45",
        alpha=0.55,
        edgecolor="white",
        linewidth=0.4,
        label="GT",
    )
    ax.hist(
        rel_range_pred * 100,
        bins=bins,
        color=plt.get_cmap(CMAP)(0.7),
        alpha=0.7,
        edgecolor="white",
        linewidth=0.4,
        label="Transolver pred",
    )
    ax.axvline(
        rel_range_gt.mean() * 100,
        color="0.25",
        lw=1.2,
        linestyle="--",
        label=f"GT mean = {rel_range_gt.mean():.2%}",
    )
    ax.axvline(
        rel_range_pred.mean() * 100,
        color=plt.get_cmap(CMAP)(0.95),
        lw=1.4,
        label=f"pred mean = {rel_range_pred.mean():.2%}",
    )
    ax.set_xlabel(r"$(Q_\mathrm{max} - Q_\mathrm{min})/|Q_\mathrm{inlet}|$  [%]")
    ax.set_ylabel("count")
    ax.set_title("Cross-section flux spread (lower = better mass conservation)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.22)

    ax = axes[1]
    ax.scatter(
        rel_outlet_gt * 100,
        rel_outlet_pred * 100,
        s=12,
        alpha=0.5,
        color=plt.get_cmap(CMAP)(0.75),
        edgecolor="none",
    )
    lim = max(np.abs(np.r_[rel_outlet_gt, rel_outlet_pred] * 100).max(), 1.0)
    ax.plot(
        [-lim, lim], [-lim, lim], color="0.5", lw=0.8, linestyle="--", label="y = x"
    )
    ax.axhline(0.0, color="0.25", lw=0.7)
    ax.axvline(0.0, color="0.25", lw=0.7)
    ax.set_xlabel("GT  $(Q_\\mathrm{outlet}-Q_\\mathrm{inlet})/Q_\\mathrm{inlet}$  [%]")
    ax.set_ylabel(
        "pred  $(Q_\\mathrm{outlet}-Q_\\mathrm{inlet})/Q_\\mathrm{inlet}$  [%]"
    )
    ax.set_title("Per-sample outlet-inlet drift: pred vs GT")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(True, alpha=0.22)

    fig.tight_layout()
    out_path = FIGURES_DIR / "pipe_mass_conservation_transolver.png"
    fig.savefig(out_path, bbox_inches="tight")
    if IS_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)
    print(f"\nSaved to {out_path}")


# %% Stream function boundary ansatz — g and l fields on the pipe mesh
# PipeStreamFunctionBoundaryAnsatz uses psi = g + l * N where:
#   g = psi_bc(eta)             : boundary lift (stream fn that recovers parabolic inlet)
#   l = xi^p * eta^2*(1-eta)^2  : correction window (zero at all hard boundaries)
#   N = backbone scalar output  : unconstrained network prediction
# Uses sample 0 mesh (x, y already loaded).
SFBA_AMPLITUDE = 0.25
SFBA_DECAY_POWER = 0.25

# eta: transverse coordinate normalized row-wise (wall-to-wall in each streamwise slice)
y_min_row = y.min(axis=1, keepdims=True)  # (H, 1)
y_max_row = y.max(axis=1, keepdims=True)  # (H, 1)
eta_field = (y - y_min_row) / np.maximum(y_max_row - y_min_row, 1e-12)  # (H, W)

# xi: streamwise coordinate, 0 at inlet (row 0) → 1 at outlet (row H-1)
xi_field = np.linspace(0.0, 1.0, H)[:, None] * np.ones((1, W))  # (H, W)

# inlet_extent: physical wall-to-wall span at the inlet row
inlet_extent = float(y[0, :].max() - y[0, :].min())

# g = psi_bc(eta) = A * H * (2*eta^2 - 4/3*eta^3)
g_field = (
    SFBA_AMPLITUDE * inlet_extent * (2.0 * eta_field**2 - (4.0 / 3.0) * eta_field**3)
)

# l = xi^p * eta^2 * (1 - eta)^2
l_field = xi_field**SFBA_DECAY_POWER * eta_field**2 * (1.0 - eta_field) ** 2

fig, (ax_g, ax_l) = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

im_g = ax_g.pcolormesh(x, y, g_field, shading="gouraud", cmap="coolwarm")
for sl in EDGE_SLICES.values():
    ax_g.plot(x[sl], y[sl], color="0.15", lw=0.7)
fig.colorbar(im_g, ax=ax_g, shrink=0.7)
ax_g.set_aspect("equal", adjustable="box")
ax_g.set_xlabel("$x$")
ax_g.set_ylabel("$y$")
# ax_g.set_title(r"$g(i,j)$", fontsize=10)

im_l = ax_l.pcolormesh(x, y, l_field, shading="gouraud", cmap=CMAP)
for sl in EDGE_SLICES.values():
    ax_l.plot(x[sl], y[sl], color="0.15", lw=0.7)
fig.colorbar(im_l, ax=ax_l, shrink=0.7)
ax_l.set_aspect("equal", adjustable="box")
ax_l.set_xlabel("$x$")
ax_l.set_ylabel("$y$")
# ax_l.set_title(
#     rf"$l(i,j)$,  $p={SFBA_DECAY_POWER}$",
#     fontsize=10,
# )

out_path = FIGURES_DIR / "pipe_stream_g_l_fields.png"
fig.savefig(out_path, bbox_inches="tight")
if IS_NOTEBOOK:
    plt.show()
else:
    plt.close(fig)
print(f"Saved to {out_path}")

# %% Validation rel-L2 curves — e100_t900 across constraint families
# Reads the W&B-exported val_rel_l2.csv in each e100 run directory and overlays
# the constraint families on one axes. If a run has no exported CSV, fall back
# to the best validation rel-L2 recorded in checkpoint_summary.txt.
import csv as _csv

VAL_CURVE_RUNS = {
    "Unconstrained": REPO_ROOT / "outputs/pipe/none/transolver/e100_t900/seed_42",
    "$u_x$ ansatz": REPO_ROOT
    / "outputs/pipe/pipe_ux_boundary_ansatz/transolver/e100_t900/seed_42",
    "Stream ansatz": REPO_ROOT
    / "outputs/pipe/pipe_stream_function_boundary_ansatz/transolver/e100_t900/seed_42",
    "Stream + $u_y$": REPO_ROOT
    / "outputs/pipe/pipe_stream_function_boundary_ansatz_uy/transolver/e100_t900/seed_42",
}
VAL_CURVE_COLORS = dict(
    zip(
        VAL_CURVE_RUNS, plt.get_cmap(CMAP)(np.linspace(0.45, 0.95, len(VAL_CURVE_RUNS)))
    )
)


def _is_float(token):
    try:
        float(token)
        return True
    except (TypeError, ValueError):
        return False


def read_val_curve(csv_path):
    """Return (steps, values) from a W&B-exported val_rel_l2.csv.

    The value column is the first whose header contains both "val" and "rel"
    (skipping the ``__MIN``/``__MAX`` aggregates, which come later); the x-axis
    uses the "Step" column when present, else the 1-based row index.
    """
    with open(csv_path, newline="") as fh:
        rows = [r for r in _csv.reader(fh) if r]
    if not rows:
        raise ValueError(f"empty csv: {csv_path}")

    header = None
    if not all(_is_float(tok) for tok in rows[0]):
        header, rows = rows[0], rows[1:]

    def _col(predicate, default):
        if header is None:
            return default
        for i, name in enumerate(header):
            if predicate(name.strip().lower()):
                return i
        return default

    val_col = _col(
        lambda n: "val" in n and "rel" in n and not n.endswith(("__min", "__max")),
        len(rows[0]) - 1,
    )
    step_col = _col(
        lambda n: n in {"step", "epoch", "epochs", "iter", "iteration"}, None
    )

    values = np.array([float(r[val_col]) for r in rows], dtype=float)
    if step_col is not None:
        steps = np.array([float(r[step_col]) for r in rows], dtype=float)
    else:
        steps = np.arange(1, len(values) + 1, dtype=float)
    return steps, values


def read_checkpoint_summary_val(summary_path):
    """Return a single best-val point from checkpoint_summary.txt."""
    section = None
    in_val_metrics = False
    epoch = None
    rel_l2 = None
    for raw_line in summary_path.read_text().splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "[BEST]":
            section = "best"
            in_val_metrics = False
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = None
            in_val_metrics = False
            continue
        if section != "best":
            continue
        if stripped.startswith("epoch:"):
            epoch = float(stripped.split(":", 1)[1])
            continue
        if stripped == "val_metrics:":
            in_val_metrics = True
            continue
        if in_val_metrics and stripped.startswith("rel_l2:"):
            rel_l2 = float(stripped.split(":", 1)[1])
            break
    if epoch is None or rel_l2 is None:
        raise ValueError(f"could not read BEST val rel-L2 from {summary_path}")
    return np.array([epoch], dtype=float), np.array([rel_l2], dtype=float)


def read_val_run(run_dir):
    csv_path = run_dir / "val_rel_l2.csv"
    if csv_path.exists():
        steps, values = read_val_curve(csv_path)
        return steps, values, "curve"
    summary_path = run_dir / "checkpoint_summary.txt"
    if summary_path.exists():
        steps, values = read_checkpoint_summary_val(summary_path)
        return steps, values, "summary"
    raise FileNotFoundError(
        f"missing val_rel_l2.csv and checkpoint_summary.txt: {run_dir}"
    )


fig, ax = plt.subplots(figsize=(6.4, 4.0))
plotted = 0
for label, run_dir in VAL_CURVE_RUNS.items():
    if not run_dir.exists():
        print(f"skip (missing): {run_dir}")
        continue
    steps, values, source = read_val_run(run_dir)
    if source == "curve":
        ax.plot(steps, values, label=label, color=VAL_CURVE_COLORS[label], lw=1.8)
    else:
        ax.scatter(
            steps,
            values,
            label=f"{label} (best)",
            color=VAL_CURVE_COLORS[label],
            s=34,
            zorder=5,
        )
    print(
        f"{label}: {len(values)} {source} point(s), best val rel-L2 = {values.min():.4g}"
    )
    plotted += 1

ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel(r"Validation relative $L_2$")
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
if plotted:
    ax.legend(frameon=False)

out_path = FIGURES_DIR / "pipe_val_rel_l2_e100.png"
if plotted:
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved to {out_path}")
else:
    print("no val curves found — nothing saved")
if IS_NOTEBOOK:
    plt.show()
else:
    plt.close(fig)
