# %% Imports & config
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
PLASTICITY_DIAGNOSTICS = REPO_ROOT / "scripts" / "diagnostics" / "plasticity"
sys.path.insert(0, str(PLASTICITY_DIAGNOSTICS))

from _common import resolve_plasticity_mat
from plasticity_forging_gif import infer_material_grid

DATA_DIR = REPO_ROOT / "data/plasticity"
FIGURES_DIR = REPO_ROOT / "docs/figures/plasticity"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CMAP = "Greys"
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "figure.dpi": 150,
    }
)

# %% Load dataset
mat_path = resolve_plasticity_mat(DATA_DIR)
raw = scio.loadmat(str(mat_path), variable_names=("input", "output"))
die = np.asarray(raw["input"], dtype=np.float64)  # (N, X)
output = np.asarray(raw["output"], dtype=np.float64)  # (N, X, Y, T, C)

print(f"Loaded {mat_path}")
print(f"die: {die.shape}  output: {output.shape}")
print("output channels used here: x, y, u_x, u_y")


# %% Helpers
def die_position(
    die_profile, timestep, *, die_speed=6.0, time_duration=1.0, t_count=20
):
    die_y_initial = np.asarray(die_profile, dtype=np.float64)
    die_drop_per_step = float(die_speed) * float(time_duration) / max(int(t_count), 1)
    return die_y_initial - die_drop_per_step * int(timestep)


def snapshot_timesteps(t_count, *, n_snapshots, step_size=None):
    if step_size is None:
        timesteps = np.linspace(0, max(int(t_count) - 1, 0), int(n_snapshots))
        return list(np.unique(timesteps.astype(int)))
    timesteps = list(
        range(0, min(int(t_count), int(n_snapshots) * int(step_size)), int(step_size))
    )
    if len(timesteps) != int(n_snapshots):
        raise ValueError(
            f"Expected {n_snapshots} snapshots with step size {step_size}, "
            f"but T={t_count} only gives {len(timesteps)}."
        )
    return timesteps


def point_plot_bounds(coords, material, die_y_all, *, pad_top_fraction=0.12):
    block_x = coords[..., 0]
    block_y = coords[..., 1]
    material_x = material[:, 0, 0, 0]
    ref_x_min, ref_x_max = float(np.nanmin(material_x)), float(np.nanmax(material_x))
    x_min = min(float(np.nanmin(block_x)), ref_x_min)
    x_max = max(float(np.nanmax(block_x)), ref_x_max)
    y_min = float(np.nanmin(block_y))
    y_max = float(np.nanmax(block_y))
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    die_y_top = max(float(np.nanmax(dy)) for dy in die_y_all)

    pad_x = 0.06 * x_span
    pad_bottom = 0.08 * y_span
    pad_top = max(die_y_top - y_max, 0.0) + float(pad_top_fraction) * y_span
    return {
        "xlim": (x_min - pad_x, x_max + pad_x),
        "ylim": (y_min - pad_bottom, y_max + pad_top),
        "die_fill_top": y_max + pad_top,
        "ref_x_min": ref_x_min,
        "ref_x_max": ref_x_max,
    }


def fractional_window(xlim, ylim, fractions):
    x0, x1, y0, y1 = (float(value) for value in fractions)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("fractional zoom bounds must satisfy 0 <= min < max <= 1")
    left, right = xlim
    bottom, top = ylim
    return (
        left + x0 * (right - left),
        left + x1 * (right - left),
        bottom + y0 * (top - bottom),
        bottom + y1 * (top - bottom),
    )


def draw_point_snapshot(
    ax,
    *,
    coords,
    material,
    disp_mag,
    timestep,
    die_x,
    die_y,
    die_fill_top,
    xlim,
    ylim,
    vmin,
    vmax,
    point_size=8.0,
    reference_point_size=2.8,
    title=None,
):
    ax.set_facecolor("#ffffff")
    ax.scatter(
        material[:, :, 0, 0].reshape(-1),
        material[:, :, 0, 1].reshape(-1),
        s=reference_point_size,
        c="#bdbdbd",
        linewidths=0,
        alpha=0.42,
    )
    scatter = ax.scatter(
        coords[:, :, timestep, 0].reshape(-1),
        coords[:, :, timestep, 1].reshape(-1),
        c=disp_mag[:, :, timestep].reshape(-1),
        s=point_size,
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        alpha=0.96,
    )

    die_fill_x = np.concatenate([die_x, die_x[::-1]])
    die_fill_y = np.concatenate([die_y, np.full_like(die_y, die_fill_top)])
    ax.fill(die_fill_x, die_fill_y, color="#111111", alpha=0.16, linewidth=0)
    ax.plot(die_x, die_y, color="#111111", linewidth=1.5)

    if title is not None:
        ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(0.8)
    return scatter


# %% First figure - die descent and material compression
SAMPLE_IDX = 0
N_SNAPSHOTS = 10
STEP_SIZE = 2
DIE_SPEED = 6.0
TIME_DURATION = 1.0

sample = output[SAMPLE_IDX]
die_profile = die[SAMPLE_IDX]

coords = sample[..., 0:2]
disp = sample[..., 2:4]
disp_mag = np.linalg.norm(disp, axis=-1)
material = infer_material_grid(sample)

t_count = int(sample.shape[2])
timesteps = snapshot_timesteps(
    t_count,
    n_snapshots=N_SNAPSHOTS,
    step_size=STEP_SIZE,
)

material_x = material[:, 0, 0, 0]
ref_x_min = float(np.nanmin(material_x))
ref_x_max = float(np.nanmax(material_x))
die_x = np.linspace(ref_x_min, ref_x_max, die_profile.shape[0])
die_y_all = [
    die_position(
        die_profile,
        timestep,
        die_speed=DIE_SPEED,
        time_duration=TIME_DURATION,
        t_count=t_count,
    )
    for timestep in timesteps
]
plot_bounds = point_plot_bounds(coords, material, die_y_all)

point_values = [disp_mag[:, :, timestep].reshape(-1) for timestep in timesteps]
vmin = min(float(np.nanmin(values)) for values in point_values)
vmax = max(float(np.nanmax(values)) for values in point_values)
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
    vmin, vmax = 0.0, 1.0

fig, axes = plt.subplots(
    2,
    5,
    figsize=(11.5, 3.25),
    sharex=True,
    sharey=True,
)
axes = axes.reshape(-1)
fig.subplots_adjust(hspace=0.00, wspace=0.0)

last_scatter = None
for ax, timestep, die_y in zip(axes, timesteps, die_y_all):
    last_scatter = draw_point_snapshot(
        ax,
        coords=coords,
        material=material,
        disp_mag=disp_mag,
        timestep=timestep,
        die_x=die_x,
        die_y=die_y,
        die_fill_top=plot_bounds["die_fill_top"],
        xlim=plot_bounds["xlim"],
        ylim=plot_bounds["ylim"],
        vmin=vmin,
        vmax=vmax,
        title=f"$t={timestep}$",
    )

axes[0].set_ylabel("y")
axes[5].set_ylabel("y")
for ax in axes[5:]:
    ax.set_xlabel("x")

fig.colorbar(
    last_scatter,
    ax=axes,
    fraction=0.018,
    pad=0.01,
    label=r"point displacement magnitude $\|u\|$",
)
# fig.suptitle(
#     f"Plasticity forging sample {SAMPLE_IDX}: die descent and compression",
#     y=1.05,
#     fontsize=12,
# )

out_path = FIGURES_DIR / "plasticity_forging_snapshots.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved {out_path}")


# %% Full and zoomed time evolution
TIME_EVOLUTION_N = 4
TIME_EVOLUTION_TIMESTEPS = snapshot_timesteps(
    t_count,
    n_snapshots=TIME_EVOLUTION_N,
    step_size=None,
)

# Edit these coordinates to choose the zoomed-in region. If left as None, the
# notebook uses the upper-middle part of the full plot.
ZOOM_WINDOW = (-47.5, -30, 5, 16)  # (x_min, x_max, y_min, y_max)
ZOOM_FRACTIONS = (0.32, 0.68, 0.56, 0.92)

time_die_y_all = [
    die_position(
        die_profile,
        timestep,
        die_speed=DIE_SPEED,
        time_duration=TIME_DURATION,
        t_count=t_count,
    )
    for timestep in TIME_EVOLUTION_TIMESTEPS
]
plot_bounds = point_plot_bounds(coords, material, time_die_y_all)
full_xlim = plot_bounds["xlim"]
full_ylim = plot_bounds["ylim"]
zoom_xlim = None
zoom_ylim = None
if ZOOM_WINDOW is None:
    zoom_xlim_min, zoom_xlim_max, zoom_ylim_min, zoom_ylim_max = fractional_window(
        full_xlim,
        full_ylim,
        ZOOM_FRACTIONS,
    )
else:
    zoom_xlim_min, zoom_xlim_max, zoom_ylim_min, zoom_ylim_max = ZOOM_WINDOW
zoom_xlim = (zoom_xlim_min, zoom_xlim_max)
zoom_ylim = (zoom_ylim_min, zoom_ylim_max)

time_values = [
    disp_mag[:, :, timestep].reshape(-1) for timestep in TIME_EVOLUTION_TIMESTEPS
]
time_vmin = min(float(np.nanmin(values)) for values in time_values)
time_vmax = max(float(np.nanmax(values)) for values in time_values)
if not np.isfinite(time_vmin) or not np.isfinite(time_vmax) or time_vmin == time_vmax:
    time_vmin, time_vmax = 0.0, 1.0

fig, axes = plt.subplots(
    2,
    len(TIME_EVOLUTION_TIMESTEPS),
    figsize=(2.35 * len(TIME_EVOLUTION_TIMESTEPS), 3.15),
    sharex=False,
    sharey=False,
)
axes = np.asarray(axes).reshape(2, -1)
fig.subplots_adjust(hspace=0.0, wspace=0.18)

last_scatter = None
for col, (timestep, die_y) in enumerate(zip(TIME_EVOLUTION_TIMESTEPS, time_die_y_all)):
    last_scatter = draw_point_snapshot(
        axes[0, col],
        coords=coords,
        material=material,
        disp_mag=disp_mag,
        timestep=timestep,
        die_x=die_x,
        die_y=die_y,
        die_fill_top=plot_bounds["die_fill_top"],
        xlim=full_xlim,
        ylim=full_ylim,
        vmin=time_vmin,
        vmax=time_vmax,
        title=f"$t={timestep}$",
    )
    last_scatter = draw_point_snapshot(
        axes[1, col],
        coords=coords,
        material=material,
        disp_mag=disp_mag,
        timestep=timestep,
        die_x=die_x,
        die_y=die_y,
        die_fill_top=plot_bounds["die_fill_top"],
        xlim=zoom_xlim,
        ylim=zoom_ylim,
        vmin=time_vmin,
        vmax=time_vmax,
        title=None,
        point_size=9.5,
        reference_point_size=3.4,
    )

axes[0, 0].set_ylabel("y")
axes[1, 0].set_ylabel("y")
for ax in axes[:, 1:].reshape(-1):
    ax.tick_params(labelleft=False)
for ax in axes[1, :]:
    ax.set_xlabel("x")

fig.colorbar(
    last_scatter,
    ax=axes.reshape(-1),
    fraction=0.018,
    pad=0.03,
    label=r"$\|u\|$",
)
# fig.suptitle(
#     f"Plasticity forging sample {SAMPLE_IDX}: full and zoomed time evolution",
#     y=1.03,
#     fontsize=12,
# )


out_path = FIGURES_DIR / "plasticity_forging_snapshots_zoomed.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved {out_path}")


# %% Physically valid versus invalid local cell evolution
POINT_COLORS = {
    "p00": "#2563eb",
    "p10": "#dc2626",
    "p11": "#16a34a",
    "p01": "#9333ea",
}
POINT_LABELS = {
    "p00": r"$p_{00}$",
    "p10": r"$p_{10}$",
    "p11": r"$p_{11}$",
    "p01": r"$p_{01}$",
}
CELL_ORDER = ("p00", "p10", "p11", "p01")


def local_reference_grid():
    x, y = np.meshgrid(np.arange(4, dtype=np.float64), np.arange(4, dtype=np.float64))
    return np.stack([x, y], axis=-1)


def set_cell_points(grid, points):
    updated = np.array(grid, copy=True)
    index_by_point = {
        "p00": (1, 1),
        "p10": (1, 2),
        "p11": (2, 2),
        "p01": (2, 1),
    }
    for point_id, index in index_by_point.items():
        updated[index] = points[point_id]
    return updated


def grid_cells(grid):
    cells = []
    for row in range(grid.shape[0] - 1):
        for col in range(grid.shape[1] - 1):
            cells.append(
                np.array(
                    [
                        grid[row, col],
                        grid[row, col + 1],
                        grid[row + 1, col + 1],
                        grid[row + 1, col],
                        grid[row, col],
                    ]
                )
            )
    return cells


def draw_local_cell_state(ax, *, grid, points, title, show_point_labels=False):
    for cell in grid_cells(grid):
        ax.plot(cell[:, 0], cell[:, 1], color="#64748b", linewidth=0.9, alpha=0.10)

    central = np.array([points[key] for key in (*CELL_ORDER, CELL_ORDER[0])])
    ax.plot(central[:, 0], central[:, 1], color="#111827", linewidth=2.2, alpha=0.95)

    for point_id, point in points.items():
        ax.scatter(
            point[0],
            point[1],
            s=72,
            color=POINT_COLORS[point_id],
            edgecolors="white",
            linewidths=0.9,
            zorder=5,
        )
        if show_point_labels:
            ax.text(
                point[0] + 0.05,
                point[1] + 0.05,
                POINT_LABELS[point_id],
                color=POINT_COLORS[point_id],
                fontsize=9,
                weight="bold",
            )

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.55, 2.45)
    ax.set_ylim(0.55, 2.45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f8fafc")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_alpha(0.2)


initial_points = {
    "p00": np.array([1.0, 1.0]),
    "p10": np.array([2.0, 1.0]),
    "p11": np.array([2.0, 2.0]),
    "p01": np.array([1.0, 2.0]),
}
valid_points = {
    "p00": np.array([0.95, 1.03]),
    "p10": np.array([2.08, 1.10]),
    "p11": np.array([1.97, 1.88]),
    "p01": np.array([0.88, 1.98]),
}
invalid_points = {
    "p00": np.array([0.95, 1.03]),
    "p10": np.array([2.08, 1.88]),
    "p11": np.array([1.03, 1.95]),
    "p01": np.array([1.97, 1.10]),
}

base_grid = local_reference_grid()
initial_grid = set_cell_points(base_grid, initial_points)
valid_grid = set_cell_points(base_grid, valid_points)
invalid_grid = set_cell_points(base_grid, invalid_points)

fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.15), sharex=True, sharey=True)
fig.subplots_adjust(wspace=0.06)

draw_local_cell_state(
    axes[0],
    grid=initial_grid,
    points=initial_points,
    title="Initial Cell",
    show_point_labels=True,
)
draw_local_cell_state(
    axes[1],
    grid=valid_grid,
    points=valid_points,
    title="Ordered Prediction",
)
draw_local_cell_state(
    axes[2],
    grid=invalid_grid,
    points=invalid_points,
    title="Unordered Prediction",
)

out_path = FIGURES_DIR / "plasticity_valid_invalid_cell_evolution.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved {out_path}")


# %% Constraint output channels to cell geometry
SCHEMATIC_H = 4
SCHEMATIC_W = 5
EXAMPLE_I = 1
EXAMPLE_J = 2
DX_COLOR = "#2563eb"
DY_COLOR = "#dc2626"
REF_COLOR = "#f59e0b"
UNUSED_COLOR = "#94a3b8"
CELL_COLOR = "#16a34a"
CHANNEL_FILL = "#FFFFFF"  # "#e5e7eb"
CHANNEL_HIGHLIGHT = "#e5e7eb"  # "#d1d5db"
CHANNEL_EDGE = "#64748b"
CHANNEL_BOLD_EDGE = "#111827"
POINT_GRAY = "#334155"
CELL_FILL_GRAY = "#9ca3af"


def draw_channel_grid(ax, *, title, entries, border_rect=None):
    ax.set_aspect("equal")
    ax.set_xlim(-0.15, SCHEMATIC_W + 0.15)
    ax.set_ylim(-0.15, SCHEMATIC_H + 0.15)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    ax.set_facecolor("#f8fafc")
    for spine in ax.spines.values():
        spine.set_alpha(0.18)

    for i in range(SCHEMATIC_H):
        for j in range(SCHEMATIC_W):
            color = "#ffffff"
            alpha = 1.0
            label = ""
            for selector, entry_color, entry_alpha, entry_label in entries:
                if selector(i, j):
                    color = entry_color
                    alpha = entry_alpha
                    label = entry_label(i, j) if callable(entry_label) else entry_label
                    break
            ax.add_patch(
                Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=color,
                    edgecolor=CHANNEL_EDGE,
                    linewidth=0.8,
                    alpha=alpha,
                )
            )
            if label:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#0f172a",
                )

    if border_rect is not None:
        x, y, w, h = border_rect
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                facecolor="none",
                edgecolor=CHANNEL_BOLD_EDGE,
                linewidth=2.2,
                zorder=5,
            )
        )


def draw_output_cell(ax):
    ax.set_aspect("equal")
    ax.set_anchor("W")
    ax.set_xlim(-0.10, 4.7)
    ax.set_ylim(-0.65, 5.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Cell Grid", fontsize=10, pad=-12)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)

    core_points = {
        (0, 0): np.array([0.0, 0.0]),
        (0, 1): np.array([0.95, 0.0]),
        (0, 2): np.array([2.12, 0.0]),
        (1, 0): np.array([0.15, 1.05]),
        (1, 1): np.array([1.06, 1.02]),
        (1, 2): np.array([2.22, 1.10]),
        (2, 0): np.array([0.28, 2.04]),
        (2, 1): np.array([1.14, 2.14]),
        (2, 2): np.array([2.28, 2.22]),
    }

    # Extrapolate core 3×3 to 5×5 using constant last-delta rule
    N = 5
    all_points = {k: np.array(v) for k, v in core_points.items()}
    for c in range(3):
        delta = all_points[(2, c)] - all_points[(1, c)]
        for r in range(3, N):
            all_points[(r, c)] = all_points[(r - 1, c)] + delta
    for r in range(N):
        delta = all_points[(r, 2)] - all_points[(r, 1)]
        for c in range(3, N):
            all_points[(r, c)] = all_points[(r, c - 1)] + delta

    # Core mesh lines (3×3 grid)
    for r in range(3):
        row = np.array([core_points[(r, c)] for c in range(3)])
        ax.plot(row[:, 0], row[:, 1], color="#64748b", linewidth=1.0, alpha=0.32)
    for c in range(3):
        col = np.array([core_points[(r, c)] for r in range(3)])
        ax.plot(col[:, 0], col[:, 1], color="#64748b", linewidth=1.0, alpha=0.32)

    # Extended mesh lines with low alpha
    EXT_ALPHA = 0.14
    for r in range(N):
        seg = np.array([all_points[(r, c)] for c in range(2 if r < 3 else 0, N)])
        ax.plot(seg[:, 0], seg[:, 1], color="#64748b", linewidth=1.0, alpha=EXT_ALPHA)
    for c in range(N):
        seg = np.array([all_points[(r, c)] for r in range(2 if c < 3 else 0, N)])
        ax.plot(seg[:, 0], seg[:, 1], color="#64748b", linewidth=1.0, alpha=EXT_ALPHA)

    cell = np.array(
        [
            core_points[(0, 0)],
            core_points[(0, 1)],
            core_points[(1, 1)],
            core_points[(1, 0)],
            core_points[(0, 0)],
        ]
    )
    ax.fill(cell[:, 0], cell[:, 1], color=CELL_FILL_GRAY, alpha=0.34, linewidth=0)
    ax.plot(cell[:, 0], cell[:, 1], color="#111827", linewidth=2.0)

    # Core points at full opacity
    core_coords = np.array(list(core_points.values()))
    ax.scatter(
        core_coords[:, 0],
        core_coords[:, 1],
        s=34,
        color=POINT_GRAY,
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
    )

    # Extended points at low opacity (no labels)
    ext_coords = np.array([v for (r, c), v in all_points.items() if r >= 3 or c >= 3])
    ax.scatter(
        ext_coords[:, 0],
        ext_coords[:, 1],
        s=18,
        color=POINT_GRAY,
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
        alpha=0.28,
    )

    labels = {
        (0, 0): r"$(x_0, 0)$",
        (0, 1): r"$(x_0+\Delta x_{00}, 0)$",
        (0, 2): r"$(x_0+\Delta x_{00}+\Delta x_{01}, 0)$",
        (1, 0): r"$(x_1, \Delta y_{00})$",
        (2, 0): r"$(x_2, \Delta y_{00}+\Delta y_{01})$",
        (1, 1): r"$(x_0+\Delta x_{10}, \Delta y_{10})$",
    }
    offsets = {
        (0, 0): (-0.25, -0.20),
        (0, 1): (-0.35, -0.20),
        (0, 2): (-0.52, -0.18),
        (1, 0): (-0.30, 0.12),
        (2, 0): (-0.28, 0.14),
        (1, 1): (-0.18, 0.12),
    }
    for key, label in labels.items():
        point = core_points[key]
        dx, dy = offsets[key]
        ax.text(point[0] + dx, point[1] + dy, label, fontsize=7, color="#0f172a")


fig = plt.figure(figsize=(11.2, 6.5))
grid_spec = fig.add_gridspec(
    2,
    2,
    width_ratios=(1.0, 1.0),
    hspace=0.18,
    wspace=0.08,
)
channel_x_ax = fig.add_subplot(grid_spec[0, 0])
channel_y_ax = fig.add_subplot(grid_spec[1, 0])
cell_ax = fig.add_subplot(grid_spec[:, 1])

draw_channel_grid(
    channel_x_ax,
    title=r" Channel $x$",
    entries=[
        (
            lambda _i, j: j == 0,
            CHANNEL_HIGHLIGHT,
            1.0,
            lambda i, _j: rf"$x_{i}$",
        ),
        (
            lambda _i, j: j > 0,
            CHANNEL_FILL,
            1.0,
            lambda i, j: rf"$\Delta x_{{{i},{j - 1}}}$",
        ),
    ],
    border_rect=(0, 0, 1, SCHEMATIC_H),
)
draw_channel_grid(
    channel_y_ax,
    title=r"Channel $y$",
    entries=[
        (
            lambda i, _j: i == SCHEMATIC_H - 1,
            CHANNEL_HIGHLIGHT,
            1.0,
            r"$0$",
        ),
        (
            lambda i, _j: i < SCHEMATIC_H - 1,
            CHANNEL_FILL,
            1.0,
            lambda i, j: rf"$\Delta y_{{{i},{j}}}$",
        ),
    ],
    border_rect=(0, SCHEMATIC_H - 1, SCHEMATIC_W, 1),
)
draw_output_cell(cell_ax)

out_path = FIGURES_DIR / "plasticity_constraint_channel_cell_mapping.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved {out_path}")


# %% Envelope constraint schematic
GAP_COLOR = "#7c3aed"
ENVELOPE_COLOR = "#0f766e"
DIE_COLOR = "#111827"
BOTTOM_COLOR = "#334155"
ALLOC_COLOR = "#dc2626"


def draw_envelope_profile(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 4.55)
    ax.set_ylim(-0.45, 4.55)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title("Envelope Reconstruction", fontsize=10)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)

    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    bottom_y = 0.0
    die_y = np.array([3.85, 3.95, 3.55, 3.22, 3.34])
    gap = np.array([0.48, 0.56, 0.44, 0.38, 0.42])
    envelope_y = die_y - gap
    weights = np.array(
        [
            [0.28, 0.35, 0.37],
            [0.22, 0.43, 0.35],
            [0.26, 0.30, 0.44],
            [0.34, 0.30, 0.36],
            [0.31, 0.36, 0.33],
        ]
    )
    free_height = envelope_y - bottom_y
    y_rows = np.zeros((x.size, 4))
    y_rows[:, -1] = bottom_y
    for k in range(2, -1, -1):
        y_rows[:, k] = y_rows[:, k + 1] + weights[:, k] * free_height

    die_fill_top = 4.35
    ax.fill_between(
        x,
        die_y,
        die_fill_top,
        color="#e5e7eb",
        alpha=0.72,
        linewidth=0,
        zorder=0,
    )
    ax.plot(x, die_y, color=DIE_COLOR, linewidth=2.2, label="moving die")
    ax.plot(
        x,
        envelope_y,
        color=ENVELOPE_COLOR,
        linewidth=2.2,
        linestyle=(0, (5, 3)),
        label="usable envelope",
    )
    ax.plot(
        [x[0], x[-1]],
        [bottom_y, bottom_y],
        color=BOTTOM_COLOR,
        linewidth=2.0,
    )

    for row in range(y_rows.shape[1]):
        alpha = 0.36 if row in {0, y_rows.shape[1] - 1} else 0.22
        ax.plot(x, y_rows[:, row], color="#64748b", linewidth=1.0, alpha=alpha)
    for col in range(x.size):
        ax.plot(
            np.repeat(x[col], y_rows.shape[1]),
            y_rows[col],
            color="#64748b",
            linewidth=1.0,
            alpha=0.25,
        )
        ax.scatter(
            np.repeat(x[col], y_rows.shape[1]),
            y_rows[col],
            s=32,
            color=POINT_GRAY,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )

    cell_x = np.array([x[1], x[2], x[2], x[1], x[1]])
    cell_y = np.array(
        [y_rows[1, 1], y_rows[2, 1], y_rows[2, 2], y_rows[1, 2], y_rows[1, 1]]
    )
    ax.fill(cell_x, cell_y, color=CELL_FILL_GRAY, alpha=0.30, linewidth=0, zorder=2)
    ax.plot(cell_x, cell_y, color="#111827", linewidth=1.8, zorder=3)

    col = 3
    ax.add_patch(
        FancyArrowPatch(
            (x[col] + 0.32, die_y[col]),
            (x[col] + 0.32, envelope_y[col]),
            arrowstyle="<->",
            mutation_scale=9,
            linewidth=1.2,
            color=GAP_COLOR,
        )
    )
    ax.text(
        x[col] + 0.42,
        0.5 * (die_y[col] + envelope_y[col]),
        r"$g_i$",
        color=GAP_COLOR,
        fontsize=8,
        va="center",
    )
    for start, stop in zip(y_rows[col, 1:], y_rows[col, :-1]):
        ax.add_patch(
            FancyArrowPatch(
                (x[col] - 0.22, start),
                (x[col] - 0.22, stop),
                arrowstyle="<->",
                mutation_scale=8,
                linewidth=1.0,
                color=ALLOC_COLOR,
            )
        )
    ax.text(
        x[col] - 0.6,
        0.5 * envelope_y[col],
        r"$\Delta y_{i,j}$",
        color=ALLOC_COLOR,
        fontsize=8,
        va="center",
    )
    ax.text(
        0.02,
        bottom_y - 0.17,
        r"$y_\mathrm{bottom}$ fixed",
        fontsize=8,
        color=BOTTOM_COLOR,
    )
    ax.text(0.04, die_y[0] + 0.14, "moving die", fontsize=8, color=DIE_COLOR)
    ax.text(
        0.04,
        envelope_y[0] - 0.28,
        "die minus gap",
        fontsize=8,
        color=ENVELOPE_COLOR,
    )


fig, profile_ax = plt.subplots(figsize=(6.4, 5.2))
draw_envelope_profile(profile_ax)

out_path = FIGURES_DIR / "plasticity_envelope_constraint_schematic.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved {out_path}")
