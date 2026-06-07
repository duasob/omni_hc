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

CMAP = "plasma"
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


# %% Model mesh overlay: unconstrained vs envelope-constrained
# Loads the two trained Transolver checkpoints, rolls each out on the same test
# sample, and overlays the predicted mesh on the ground-truth mesh. The point is
# the learning difference: the unconstrained model can drift off the true mesh,
# while the envelope-constrained model is ordered and capped by construction.
import plasticity_model_gif as pmg
import torch
from matplotlib.collections import LineCollection  # noqa: F401  (used via pmg.add_mesh)
from matplotlib.lines import Line2D

from omni_hc.benchmarks.plasticity.data import build_test_loader
from omni_hc.core import compose_run_config, parse_dotted_overrides
from omni_hc.integrations.nsl import create_model
from omni_hc.training.common import load_checkpoint_state, load_model_state_dict

OVERLAY_SAMPLE_IDX = 0
OVERLAY_TIMESTEP = -1  # last timestep (fully deformed); set to an int in [0, T)

# (row label, run directory). The unconstrained baseline is on top.
MODEL_RUNS = (
    (
        "Unconstrained",
        REPO_ROOT / "outputs/plasticity/none/Transolver/debug/seed_42",
    ),
    (
        "Envelope constrained",
        REPO_ROOT
        / "outputs/plasticity/plasticity_envelope_constraint/transolver/debug/seed_42",
    ),
)

GT_MESH_COLOR = "#94a3b8"
PRED_MESH_COLOR = "#dc2626"


PRED_CACHE_DIR = REPO_ROOT / "artifacts/plasticity/pred_cache"


def _pred_cache_path(run_dir, sample_idx):
    # Slug from the trailing <family>/<backbone>/<budget>/<seed> path parts.
    slug = "_".join(Path(run_dir).parts[-4:])
    return PRED_CACHE_DIR / f"{slug}__sample{int(sample_idx):04d}.npz"


def _checkpoint_path(run_dir):
    run_dir = Path(run_dir)
    best = run_dir / "best.pt"
    return best if best.exists() else run_dir / "latest.pt"


def predict_model_sequences(run_dir, sample_idx, *, device=None, use_cache=True):
    """Roll a trained checkpoint out on one test sample.

    Returns ``(pred_seq, target_seq, meta)`` with the raw 4-channel sequences
    shaped ``(H, W, T, out_dim)`` (coords + displacement), reconstructed the same
    way as the diagnostics GIFs.

    Results are cached under ``artifacts/plasticity/pred_cache`` keyed by run and
    sample (rolling a checkpoint out is the slow part). The cache is invalidated
    automatically when the checkpoint is newer than the cached file.
    """
    run_dir = Path(run_dir)
    cache_path = _pred_cache_path(run_dir, sample_idx)
    checkpoint = _checkpoint_path(run_dir)
    if (
        use_cache
        and cache_path.exists()
        and (
            not checkpoint.exists()
            or checkpoint.stat().st_mtime <= cache_path.stat().st_mtime
        )
    ):
        cached = np.load(cache_path)
        meta = {
            "shapelist": tuple(int(v) for v in cached["shapelist"]),
            "t_out": int(cached["t_out"]),
            "out_dim": int(cached["out_dim"]),
        }
        print(f"[pred cache] HIT  {cache_path.name}")
        return cached["pred_seq"], cached["target_seq"], meta

    print(f"[pred cache] MISS {cache_path.name} (rolling out checkpoint...)")
    device = device or torch.device("cpu")
    cfg = compose_run_config(
        experiment=str(run_dir / "resolved_config.yaml"),
        mode="test",
        extra_overrides=parse_dotted_overrides([f"paths.root_dir={DATA_DIR}"]),
    )
    loader = build_test_loader(cfg)
    meta = loader.plasticity_meta
    x_normalizer = getattr(loader, "x_normalizer", None)
    if x_normalizer is not None:
        x_normalizer = x_normalizer.to(device)
    y_normalizer = getattr(loader, "y_normalizer", None)
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)

    model, _, _ = create_model(
        cfg,
        device=device,
        runtime_overrides=pmg.runtime_overrides(meta),
    )
    if (
        x_normalizer is not None
        and hasattr(model, "constraint")
        and hasattr(model.constraint, "set_input_normalizer")
    ):
        model.constraint.set_input_normalizer(x_normalizer)

    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "latest.pt"
    checkpoint = load_checkpoint_state(checkpoint_path, device=device)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()

    batch = pmg.get_sample_batch(loader, sample_idx, device=device)
    pred, target, _ = pmg.predict_sequence(
        model,
        batch,
        y_normalizer=y_normalizer,
        t_out=int(meta["t_out"]),
    )
    h, w = tuple(meta["shapelist"])
    t_out, out_dim = int(meta["t_out"]), int(meta["out_dim"])
    pred_seq = pred[0].detach().cpu().reshape(h, w, t_out, out_dim).numpy()
    target_seq = target[0].detach().cpu().reshape(h, w, t_out, out_dim).numpy()
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            pred_seq=pred_seq,
            target_seq=target_seq,
            shapelist=np.asarray([h, w], dtype=np.int64),
            t_out=np.int64(t_out),
            out_dim=np.int64(out_dim),
        )
    return pred_seq, target_seq, meta


def load_model_prediction(run_dir, sample_idx, *, device=None):
    """Return ``(pred_coords, target_coords)`` in physical space, each (H, W, T, 2)."""
    pred_seq, target_seq, _meta = predict_model_sequences(
        run_dir, sample_idx, device=device
    )
    material = pmg.infer_material(target_seq)
    return pmg.plot_coords(pred_seq, material), pmg.plot_coords(target_seq, material)


overlay_predictions = [
    (label, *load_model_prediction(run_dir, OVERLAY_SAMPLE_IDX))
    for label, run_dir in MODEL_RUNS
]

fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.12)

overlay_coord_sets = []
for _label, pred_coords, target_coords in overlay_predictions:
    overlay_coord_sets.append(pred_coords[:, :, OVERLAY_TIMESTEP])
    overlay_coord_sets.append(target_coords[:, :, OVERLAY_TIMESTEP])

for ax, (label, pred_coords, target_coords) in zip(axes, overlay_predictions):
    pred_t = pred_coords[:, :, OVERLAY_TIMESTEP]
    target_t = target_coords[:, :, OVERLAY_TIMESTEP]
    pmg.add_mesh(ax, target_t, color=GT_MESH_COLOR, linewidth=0.5, alpha=0.95, step=1)
    pmg.add_mesh(ax, pred_t, color=PRED_MESH_COLOR, linewidth=0.5, alpha=0.8, step=1)
    rel_l2 = float(
        np.linalg.norm((pred_t - target_t).reshape(-1))
        / max(np.linalg.norm(target_t.reshape(-1)), 1.0e-12)
    )
    ax.set_title(f"{label}  (rel. $L_2$ = {rel_l2:.4f})", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylabel("y")
    ax.tick_params(labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(0.8)

axes[-1].set_xlabel("x")
fig.legend(
    handles=[
        Line2D([], [], color=GT_MESH_COLOR, linewidth=1.6, label="ground truth"),
        Line2D([], [], color=PRED_MESH_COLOR, linewidth=1.2, label="prediction"),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=2,
    fontsize=9,
    frameon=False,
)
pmg.set_shared_limits(axes, *overlay_coord_sets)

out_path = FIGURES_DIR / "plasticity_model_mesh_overlay.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved {out_path}")


# %% Validation rel-L2 curves: none vs mesh vs envelope (e50_t500)
# Reads the wandb-exported val/rel_l2 history (column 1 of val_rel_l2.csv) for
# the three constraint families at the e50_t500 budget and plots them together
# on a log y-axis, showing the early inductive-bias head start of the
# constrained models versus the unconstrained baseline.
import csv

VAL_CURVE_BUDGET = "e50_t500"
# (label, constraint dir)
VAL_CURVE_RUNS = (
    ("Unconstrained", "none"),
    ("Mesh consistency", "plasticity_mesh_consistency_constraint"),
    ("Envelope", "plasticity_envelope_constraint"),
)
# Sample one colour per run from CMAP, avoiding the very light/dark extremes.
val_curve_colors = plt.get_cmap(CMAP)(np.linspace(0.1, 0.8, len(VAL_CURVE_RUNS)))


def load_val_curve(csv_path):
    """Return (steps, val_rel_l2) from a wandb-exported val_rel_l2.csv.

    Column 0 is the step; column 1 is the run's val/rel_l2 (the __MIN/__MAX
    columns duplicate it for single-seed exports and are ignored).
    """
    steps, values = [], []
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2 or not row[0] or not row[1]:
                continue
            steps.append(float(row[0]))
            values.append(float(row[1]))
    return np.asarray(steps), np.asarray(values)


fig, ax = plt.subplots(figsize=(6.6, 4.2))
for (label, constraint_dir), color in zip(VAL_CURVE_RUNS, val_curve_colors):
    csv_path = (
        REPO_ROOT
        / "outputs/plasticity"
        / constraint_dir
        / "transolver"
        / VAL_CURVE_BUDGET
        / "seed_42"
        / "val_rel_l2.csv"
    )
    if not csv_path.exists():
        print(f"skip (missing): {csv_path}")
        continue
    steps, values = load_val_curve(csv_path)
    ax.plot(steps, values, color=color, linewidth=1.6, label=label)

ax.set_yscale("log")
ax.set_xlabel("training step")
ax.set_ylabel(r"validation rel. $L_2$")
ax.set_title(f"Validation rel. $L_2$ over training ({VAL_CURVE_BUDGET})", fontsize=10)
ax.grid(True, which="both", linewidth=0.4, alpha=0.25)
ax.legend(fontsize=9, frameon=False)
for spine in ax.spines.values():
    spine.set_color("#111111")
    spine.set_linewidth(0.8)

out_path = FIGURES_DIR / f"plasticity_val_rel_l2_{VAL_CURVE_BUDGET}.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved {out_path}")


# %% Failure mode 1: cell collapse (filmstrip from the constraint-failure GIF)
# The pre-rendered diagnostic GIF (left: prediction mesh with inverted cells in
# red; right: oriented cell-area matrix) is the clearest illustration, so we
# simply lay a few of its frames out as a static figure.
from PIL import Image

FIG1_GIF = FIGURES_DIR / "sample_0100_constraint_failures.gif"
FIG1_FRAMES = [0, 6, 13, 19]  # which GIF frames (timesteps) to lay out
FIG1_NCOLS = 1  # 1 = vertical filmstrip; 2 = grid
# Optional per-frame pixel crop (left, top, right, bottom) to trim titles/margins.
FIG1_CROP = None

f1_gif = Image.open(FIG1_GIF)
f1_sel = [f for f in FIG1_FRAMES if 0 <= f < f1_gif.n_frames]
f1_imgs = []
for f in f1_sel:
    f1_gif.seek(f)
    frame = f1_gif.convert("RGB")
    if FIG1_CROP is not None:
        frame = frame.crop(tuple(FIG1_CROP))
    f1_imgs.append(np.asarray(frame))

f1_ncols = max(1, int(FIG1_NCOLS))
f1_nrows = int(np.ceil(len(f1_imgs) / f1_ncols))
f1_h, f1_w = f1_imgs[0].shape[:2]
f1_cell_w = 7.5 / f1_ncols
f1_cell_h = f1_cell_w * (f1_h / f1_w)
fig, axes = plt.subplots(
    f1_nrows,
    f1_ncols,
    figsize=(f1_cell_w * f1_ncols, f1_cell_h * f1_nrows),
)
axes = np.atleast_1d(axes).reshape(-1)
for ax in axes:
    ax.set_axis_off()
for ax, img in zip(axes, f1_imgs):
    ax.imshow(img)
fig.subplots_adjust(hspace=0.02, wspace=0.02)

out_path = FIGURES_DIR / "plasticity_failure_cell_collapse.png"
fig.savefig(out_path, bbox_inches="tight", dpi=200)
plt.show()
print(f"Saved {out_path}")


# %% Failure mode 2: boundary constraint diagnostics (top envelope + bottom pin)
# Panel A: top-row y vs x against the moving-die cap; the unconstrained surface
# pokes above the die (shaded). Panel B: bottom-row y vs x against the pinned
# y_bottom, zoomed tight so the unconstrained drift off the pin is visible.
FIG2_SAMPLE = 0
FIG2_BUDGET = "final"
FIG2_TIMESTEP = -1  # last (most deformed) step
NTEST = 80
TOP_HEIGHT = 15.1
Y_BOTTOM = -0.1

FIG2_RUNS = (
    ("Unconstrained", "none", "#dc2626"),
    # ("Mesh", "plasticity_mesh_consistency_constraint", "#2563eb"),
    ("Envelope", "plasticity_envelope_constraint", "#0f766e"),
)

f2_coords = {}
f2_material = None
for label, fam, color in FIG2_RUNS:
    run = REPO_ROOT / f"outputs/plasticity/{fam}/transolver/{FIG2_BUDGET}/seed_42"
    ps, ts, _ = predict_model_sequences(run, FIG2_SAMPLE)
    f2_material = pmg.infer_material(ts)
    f2_coords[label] = (pmg.plot_coords(ps, f2_material), color)

f2_t_count = f2_material.shape[2]
f2_t = FIG2_TIMESTEP if FIG2_TIMESTEP >= 0 else f2_t_count + FIG2_TIMESTEP

# Die cap: input die profile dropped to this timestep, clamped at TOP_HEIGHT.
f2_die_profile = die[die.shape[0] - NTEST + FIG2_SAMPLE]
f2_top_x_ref = f2_material[:, 0, 0, 0]
f2_ref_x_min, f2_ref_x_max = (
    float(np.nanmin(f2_top_x_ref)),
    float(np.nanmax(f2_top_x_ref)),
)
f2_die_x = np.linspace(f2_ref_x_min, f2_ref_x_max, f2_die_profile.shape[0])
f2_die_y = die_position(
    f2_die_profile,
    f2_t,
    die_speed=DIE_SPEED,
    time_duration=TIME_DURATION,
    t_count=f2_t_count,
)
f2_dord = np.argsort(f2_die_x)


def f2_cap(x_query):
    cap = np.interp(x_query, f2_die_x[f2_dord], f2_die_y[f2_dord])
    return np.minimum(cap, TOP_HEIGHT)


f2_order = np.argsort(f2_top_x_ref)
f2_xs = f2_top_x_ref[f2_order]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.2, 6.0))

# Panel A: top envelope
ax_top.plot(
    f2_die_x[f2_dord],
    np.minimum(f2_die_y[f2_dord], TOP_HEIGHT),
    color="#111827",
    linewidth=1.8,
    label="moving-die cap",
)
for label, (coords, color) in f2_coords.items():
    top_x = coords[:, 0, f2_t, 0][f2_order]
    top_y = coords[:, 0, f2_t, 1][f2_order]
    ax_top.plot(top_x, top_y, color=color, linewidth=1.3, label=label)
unc_top_y = f2_coords["Unconstrained"][0][:, 0, f2_t, 1][f2_order]
cap_xs = f2_cap(f2_xs)
ax_top.fill_between(
    f2_xs,
    cap_xs,
    unc_top_y,
    where=unc_top_y > cap_xs,
    color="#dc2626",
    alpha=0.18,
    linewidth=0,
    label="penetration",
)
ax_top.set_title(f"Top surface vs moving-die cap ($t={f2_t}$)", fontsize=10)
ax_top.set_xlabel("x")
ax_top.set_ylabel("y")
ax_top.legend(fontsize=8, frameon=False, ncol=2)

# Panel B: bottom pin (zoomed)
ax_bot.axhline(
    Y_BOTTOM,
    color="#111827",
    linewidth=1.4,
    linestyle=(0, (5, 3)),
    label=r"$y_\mathrm{bottom}$",
)
f2_dev = 0.0
for label, (coords, color) in f2_coords.items():
    bot_x = coords[:, -1, f2_t, 0]
    bot_y = coords[:, -1, f2_t, 1]
    border = np.argsort(bot_x)
    ax_bot.plot(bot_x[border], bot_y[border], color=color, linewidth=1.3, label=label)
    f2_dev = max(f2_dev, float(np.nanmax(np.abs(bot_y - Y_BOTTOM))))
f2_margin = max(1.2 * f2_dev, 0.02)
ax_bot.set_ylim(Y_BOTTOM - f2_margin, Y_BOTTOM + f2_margin)
ax_bot.set_title(
    f"Bottom row vs pinned $y_\\mathrm{{bottom}}$ ($t={f2_t}$)", fontsize=10
)
ax_bot.set_xlabel("x")
ax_bot.set_ylabel("y")
ax_bot.legend(fontsize=8, frameon=False, ncol=2)

fig.tight_layout()
out_path = FIGURES_DIR / "plasticity_failure_boundary_diagnostics.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved {out_path}")


# %% Failure mode 3: qualitative predictions across budgets (family x budget)
# Each cell: predicted mesh nodes coloured by per-node error over a faint GT mesh,
# with the rel-L2 printed. Rows are constraint families, columns are data budgets.
FIG3_SAMPLE = 0
FIG3_TIMESTEP = -1
FIG3_FAMILIES = (
    ("Unconstrained", "none"),
    ("Mesh", "plasticity_mesh_consistency_constraint"),
    ("Envelope", "plasticity_envelope_constraint"),
)
FIG3_BUDGETS = (
    ("Tiny", "e5_t50"),
    ("Small", "e10_t100"),
    ("Medium", "e50_t500"),
    ("Large", "e100_t900"),
    ("Full", "final"),
)

f3_cells = {}
f3_err_max = 0.0
for family_label, family_dir in FIG3_FAMILIES:
    for budget_label, budget_dir in FIG3_BUDGETS:
        run = (
            REPO_ROOT
            / f"outputs/plasticity/{family_dir}/transolver/{budget_dir}/seed_42"
        )
        pred_coords, target_coords = load_model_prediction(run, FIG3_SAMPLE)
        pred_t = pred_coords[:, :, FIG3_TIMESTEP]
        target_t = target_coords[:, :, FIG3_TIMESTEP]
        node_err = np.linalg.norm(pred_t - target_t, axis=-1)
        rel_l2 = float(
            np.linalg.norm((pred_t - target_t).reshape(-1))
            / max(np.linalg.norm(target_t.reshape(-1)), 1.0e-12)
        )
        f3_cells[(family_label, budget_label)] = (pred_t, target_t, node_err, rel_l2)
        f3_err_max = max(f3_err_max, float(np.nanpercentile(node_err, 99)))

# Budgets down the rows, the three families across the columns: each row is a
# like-for-like comparison of the families at one data budget.
f3_rows = len(FIG3_BUDGETS)
f3_cols = len(FIG3_FAMILIES)
# col_width/row_height control the cell box; the meshes are wide and short, so a
# small row_height keeps the box close to the content and removes the big gaps
# between rows. f3_hspace is the residual vertical gap between rows (wspace the
# horizontal gap between columns). Tune these three to taste.
col_width = 2.5
row_height = 1.15
f3_hspace = 0.05
f3_wspace = 0.05
fig, axes = plt.subplots(
    f3_rows,
    f3_cols,
    figsize=(col_width * f3_cols, row_height * f3_rows),
    sharex=True,
    sharey=True,
)
axes = np.atleast_2d(axes)
last_scatter = None
for r, (budget_label, _budget_dir) in enumerate(FIG3_BUDGETS):
    for c, (family_label, _family_dir) in enumerate(FIG3_FAMILIES):
        ax = axes[r, c]
        pred_t, target_t, node_err, rel_l2 = f3_cells[(family_label, budget_label)]
        pmg.add_mesh(ax, target_t, color="#cbd5e1", linewidth=0.3, alpha=0.85, step=1)
        last_scatter = ax.scatter(
            pred_t[..., 0].reshape(-1),
            pred_t[..., 1].reshape(-1),
            c=node_err.reshape(-1),
            s=3.0,
            cmap="plasma",
            vmin=0.0,
            vmax=f3_err_max,
            linewidths=0,
        )
        ax.set_aspect("equal", adjustable="box")
        if r == 0:
            ax.set_title(family_label, fontsize=10)
        if c == 0:
            ax.set_ylabel(budget_label, fontsize=10)
        # ax.text(
        #     0.03,
        #     0.96,
        #     rf"rel $L_2$={rel_l2:.3f}",
        #     transform=ax.transAxes,
        #     fontsize=7,
        #     va="top",
        #     ha="left",
        # )
        ax.tick_params(labelsize=6, length=2)

f3_coord_sets = []
for pred_t, target_t, _err, _rel in f3_cells.values():
    f3_coord_sets.append(pred_t)
    f3_coord_sets.append(target_t)
pmg.set_shared_limits(axes.reshape(-1), *f3_coord_sets)
fig.subplots_adjust(hspace=f3_hspace, wspace=f3_wspace)
fig.colorbar(
    last_scatter,
    ax=axes,
    fraction=0.015,
    pad=0.01,
    label=r"node error $\|\hat{p} - p\|$",
)

out_path = FIGURES_DIR / "plasticity_failure_budget_qualitative.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved {out_path}")
