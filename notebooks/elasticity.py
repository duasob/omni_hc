# %% Imports & config
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
DATA_DIR = REPO_ROOT / "data/elasticity"
FIGURES_DIR = REPO_ROOT / "docs/figures/elasticity"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CMAP = "viridis"
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

# %% Load dataset
# Reuse the benchmark's own orientation logic so the notebook matches training.
from omni_hc.benchmarks.elasticity.data import (
    _orient_elasticity_arrays,
    resolve_elasticity_files,
)

sigma_path, xy_path = resolve_elasticity_files(DATA_DIR)
sigma_raw = np.load(sigma_path)
xy_raw = np.load(xy_path)
coords, sigma = _orient_elasticity_arrays(sigma_raw, xy_raw)  # (N, P, 2), (N, P)

N, P, _ = coords.shape
print(f"coords: {coords.shape}  sigma: {sigma.shape}  ({N} samples, {P} points)")
print(f"sigma range: [{sigma.min():.3e}, {sigma.max():.3e}]")

# %% Dataset sample - point cloud coloured by von Mises stress
SAMPLE_IDX = [0]

fig, axes = plt.subplots(1, len(SAMPLE_IDX), figsize=(4 * len(SAMPLE_IDX), 4))
if len(SAMPLE_IDX) == 1:
    axes = [axes]

vmin = min(sigma[i].min() for i in SAMPLE_IDX)
vmax = max(sigma[i].max() for i in SAMPLE_IDX)
for ax, idx in zip(axes, SAMPLE_IDX):
    sc = ax.scatter(
        coords[idx, :, 0],
        coords[idx, :, 1],
        c=sigma[idx],
        cmap=CMAP,
        s=10,
        vmin=vmin,
        vmax=vmax,
    )
    # ax.set_title(f"sample {idx}", fontsize=10)

    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

fig.colorbar(
    plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax)),
    ax=axes,
    fraction=0.025,
    pad=0.02,
    label=r"$\sigma_{VM}$",
)
fig.savefig(FIGURES_DIR / "elasticity_dataset_sample.png", bbox_inches="tight")
plt.show()
print(f"Saved {FIGURES_DIR / 'elasticity_dataset_sample.png'}")

# %% Dataset summary statistics - stress distribution
per_sample_mean = sigma.mean(axis=1)
per_sample_max = sigma.max(axis=1)

fig, (ax_hist, ax_band) = plt.subplots(1, 2, figsize=(12, 3.5))

ax_hist.hist(sigma.reshape(-1), bins=80, color=plt.get_cmap(CMAP)(0.4))
ax_hist.set_xlabel(r"$\sigma_{VM}$ (all points, all samples)")
ax_hist.set_ylabel("count")
ax_hist.set_title("Stress distribution")

order = np.argsort(per_sample_mean)
ax_band.plot(
    per_sample_mean[order], color=plt.get_cmap(CMAP)(0.3), label="per-sample mean"
)
ax_band.plot(
    per_sample_max[order], color=plt.get_cmap(CMAP)(0.75), label="per-sample max"
)
ax_band.set_xlabel("sample (sorted by mean stress)")
ax_band.set_ylabel(r"$\sigma_{VM}$")
ax_band.set_title("Per-sample stress")
ax_band.legend(frameon=False)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "elasticity_dataset_summary.png", bbox_inches="tight")
plt.show()
print(f"Saved {FIGURES_DIR / 'elasticity_dataset_summary.png'}")

# %% [SCAFFOLD] Inference - load a constrained checkpoint and run a forward pass
# TODO: point RUN_DIR at a finished run once experiments are done.
# Mirrors the steady-task forward used in training; elasticity is single-step
# (not autoregressive), so we use forward_with_optional_aux directly.
import torch
import yaml

from omni_hc.benchmarks.elasticity.adapter import _prepare_batch, _runtime_overrides
from omni_hc.benchmarks.elasticity.data import build_test_loader
from omni_hc.integrations.nsl import create_model
from omni_hc.training.common import (
    forward_with_optional_aux,
    load_checkpoint_state,
    load_model_state_dict,
)

DEVICE = torch.device("cpu")
RUN_DIR = (
    REPO_ROOT
    / "outputs/elasticity/elasticity_plane_stress_vm_constraint/transolver/smoke/seed_42"
)

if (RUN_DIR / "resolved_config.yaml").exists():
    cfg = yaml.safe_load((RUN_DIR / "resolved_config.yaml").read_text())
    cfg["paths"]["root_dir"] = str(DATA_DIR)

    test_loader = build_test_loader(cfg)
    meta = test_loader.elasticity_meta

    model, _, _ = create_model(
        cfg, device=DEVICE, runtime_overrides=_runtime_overrides(meta)
    )
    model.constraint.set_target_normalizer(test_loader.y_normalizer.to(DEVICE))
    ckpt = load_checkpoint_state(RUN_DIR / "best.pt", device=DEVICE)
    load_model_state_dict(model, ckpt["model_state_dict"])
    model.eval()

    batch = next(iter(test_loader))
    coords_b, fx_b, target_b = _prepare_batch(batch, device=DEVICE)
    with torch.no_grad():
        out = forward_with_optional_aux(model, coords_b, fx_b)
    print("aux keys:", sorted(out.get("aux_tensors", {}).keys()))
else:
    print(f"[scaffold] no run at {RUN_DIR}; skip inference until a run exists.")

# %% 3D latent membrane - relative thickness change from lambda_3
# The plane-stress model gives local principal stretch magnitudes, including
# lambda_3 = exp(-2m). If an undeformed membrane has thickness h0, the local
# plane-stress model implies h/h0 = lambda_3. The dataset does not provide h0,
# so the physically meaningful quantity here is the relative thickness change
# lambda_3 - 1, not an absolute thickness.
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if "out" not in globals():
    print("[3D visualization] run the inference cell first.")
else:
    VIS_SAMPLE = 0
    REFERENCE_DISPLAY_THICKNESS = 0.08
    THICKNESS_VARIATION_EXAGGERATION = 10000.0

    aux = out["aux_tensors"]
    coords_3d = coords_b[VIS_SAMPLE].cpu().numpy()
    lambda_1 = aux["lambda_1"][VIS_SAMPLE, :, 0].cpu().numpy()
    lambda_2 = aux["lambda_2"][VIS_SAMPLE, :, 0].cpu().numpy()
    lambda_3 = aux["lambda_3"][VIS_SAMPLE, :, 0].cpu().numpy()
    sigma_vm = aux["sigma_physical"][VIS_SAMPLE, :, 0].cpu().numpy()

    relative_thickness_change = lambda_3 - 1.0

    # REFERENCE_DISPLAY_THICKNESS is a drawing scale, not a measured h0.
    # Exaggerate only the variation around lambda_3=1.
    lambda_3_visual = 1.0 + THICKNESS_VARIATION_EXAGGERATION * (lambda_3 - 1.0)
    lambda_3_visual = np.maximum(lambda_3_visual, 0.05)
    half_thickness = 0.5 * REFERENCE_DISPLAY_THICKNESS * lambda_3_visual

    triangulation = mtri.Triangulation(coords_3d[:, 0], coords_3d[:, 1])
    triangles = triangulation.triangles

    # Delaunay triangulation bridges the central void. Mask triangles containing
    # an unusually long edge so the hole remains visible.
    triangle_xy = coords_3d[triangles]
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangle_xy[:, 0] - triangle_xy[:, 1], axis=1),
            np.linalg.norm(triangle_xy[:, 1] - triangle_xy[:, 2], axis=1),
            np.linalg.norm(triangle_xy[:, 2] - triangle_xy[:, 0], axis=1),
        ),
        axis=1,
    )
    max_edge = edge_lengths.max(axis=1)
    local_edge_scale = np.median(edge_lengths)
    triangulation.set_mask(max_edge > 3.0 * local_edge_scale)

    visible_triangles = triangles[~triangulation.mask]
    triangle_sigma = sigma_vm[visible_triangles].mean(axis=1)
    stress_norm = plt.Normalize(vmin=sigma_vm.min(), vmax=sigma_vm.max())
    stress_cmap = plt.get_cmap(CMAP)

    top_vertices = np.stack(
        (
            coords_3d[visible_triangles, 0],
            coords_3d[visible_triangles, 1],
            half_thickness[visible_triangles],
        ),
        axis=-1,
    )
    bottom_vertices = top_vertices.copy()
    bottom_vertices[..., 2] *= -1.0

    fig = plt.figure(figsize=(16, 6))
    ax_3d = fig.add_subplot(121, projection="3d")
    top_surface = Poly3DCollection(
        top_vertices,
        facecolors=stress_cmap(stress_norm(triangle_sigma)),
        edgecolors=(1.0, 1.0, 1.0, 0.12),
        linewidths=0.08,
    )
    bottom_surface = Poly3DCollection(
        bottom_vertices,
        facecolors=stress_cmap(stress_norm(triangle_sigma)),
        edgecolors="none",
        alpha=0.35,
    )
    ax_3d.add_collection3d(top_surface)
    ax_3d.add_collection3d(bottom_surface)
    ax_3d.set_xlim(coords_3d[:, 0].min(), coords_3d[:, 0].max())
    ax_3d.set_ylim(coords_3d[:, 1].min(), coords_3d[:, 1].max())
    ax_3d.set_zlim(-half_thickness.max() * 1.15, half_thickness.max() * 1.15)
    ax_3d.set_xlabel("$x$")
    ax_3d.set_ylabel("$y$")
    ax_3d.set_zlabel("display thickness")
    ax_3d.set_title(
        "Stress-colored membrane\n"
        + rf"thickness variation exaggerated "
        + rf"$\times {THICKNESS_VARIATION_EXAGGERATION:.0f}$"
    )
    ax_3d.view_init(elev=25, azim=-58)
    ax_3d.set_box_aspect((1.0, 1.0, 0.35))
    fig.colorbar(
        plt.cm.ScalarMappable(norm=stress_norm, cmap=stress_cmap),
        ax=ax_3d,
        fraction=0.035,
        pad=0.08,
        label=r"predicted $\sigma_{\mathrm{VM}}$",
    )

    ax_change = fig.add_subplot(122)
    change_scale = max(
        abs(relative_thickness_change.min()),
        abs(relative_thickness_change.max()),
    )
    thickness_plot = ax_change.tripcolor(
        triangulation,
        100.0 * relative_thickness_change,
        shading="gouraud",
        cmap="coolwarm",
        vmin=-100.0 * change_scale,
        vmax=100.0 * change_scale,
    )
    ax_change.set_aspect("equal")
    ax_change.set_xlabel("$x$")
    ax_change.set_ylabel("$y$")
    ax_change.set_title(r"Relative thickness change $100(\lambda_3-1)$")
    fig.colorbar(
        thickness_plot,
        ax=ax_change,
        fraction=0.046,
        pad=0.04,
        label="thickness change (%)",
    )

    annotation = (
        rf"$\lambda_1\in[{lambda_1.min():.6f},{lambda_1.max():.6f}]$"
        + "\n"
        + rf"$\lambda_2\in[{lambda_2.min():.6f},{lambda_2.max():.6f}]$"
        + "\n"
        + rf"$\lambda_3\in[{lambda_3.min():.6f},{lambda_3.max():.6f}]$"
        + "\n"
        + r"principal stretch magnitudes; rotation/directions unknown"
    )
    ax_3d.text2D(
        0.02,
        0.98,
        annotation,
        transform=ax_3d.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
    )

    fig.tight_layout()
    output_path = FIGURES_DIR / "elasticity_latent_membrane_3d.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.show()
    print(f"Saved {output_path}")

# %% Thin 3D solid - reference versus lambda_3 thickness transformation
# This closes the triangulated membrane with side walls, including the central
# void boundary. The in-plane coordinates are unchanged because the current
# latent representation does not identify principal directions or displacement.
if "out" not in globals():
    print("[thin solid visualization] run the inference cell first.")
else:
    SOLID_SAMPLE = 0
    SOLID_REFERENCE_THICKNESS = 0.08
    SOLID_THICKNESS_EXAGGERATION = 1500.0

    solid_aux = out["aux_tensors"]
    solid_xy = coords_b[SOLID_SAMPLE].cpu().numpy()
    solid_lambda_3 = solid_aux["lambda_3"][SOLID_SAMPLE, :, 0].cpu().numpy()
    solid_change = solid_lambda_3 - 1.0

    solid_tri = mtri.Triangulation(solid_xy[:, 0], solid_xy[:, 1])
    solid_triangles = solid_tri.triangles
    solid_triangle_xy = solid_xy[solid_triangles]
    solid_edges = np.stack(
        (
            np.linalg.norm(
                solid_triangle_xy[:, 0] - solid_triangle_xy[:, 1], axis=1
            ),
            np.linalg.norm(
                solid_triangle_xy[:, 1] - solid_triangle_xy[:, 2], axis=1
            ),
            np.linalg.norm(
                solid_triangle_xy[:, 2] - solid_triangle_xy[:, 0], axis=1
            ),
        ),
        axis=1,
    )
    solid_tri.set_mask(
        solid_edges.max(axis=1) > 3.0 * np.median(solid_edges)
    )
    solid_visible_triangles = solid_triangles[~solid_tri.mask]

    # Edges used by one visible triangle form either the external boundary or
    # the boundary of the void. Both need side faces to make a closed solid.
    edge_counts = {}
    for triangle in solid_visible_triangles:
        for start, end in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(start), int(end))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_edges = np.asarray(
        [edge for edge, count in edge_counts.items() if count == 1],
        dtype=int,
    )

    reference_half = np.full(solid_xy.shape[0], 0.5 * SOLID_REFERENCE_THICKNESS)
    transformed_scale = 1.0 + SOLID_THICKNESS_EXAGGERATION * solid_change
    transformed_scale = np.maximum(transformed_scale, 0.05)
    transformed_half = 0.5 * SOLID_REFERENCE_THICKNESS * transformed_scale

    change_limit = max(abs(solid_change.min()), abs(solid_change.max()))
    change_norm = plt.Normalize(vmin=-change_limit, vmax=change_limit)
    change_cmap = plt.get_cmap("coolwarm")

    def solid_vertices(half_thickness):
        top_vertices = np.stack(
            (
                solid_xy[solid_visible_triangles, 0],
                solid_xy[solid_visible_triangles, 1],
                half_thickness[solid_visible_triangles],
            ),
            axis=-1,
        )
        bottom_vertices = top_vertices.copy()
        bottom_vertices[..., 2] *= -1.0

        side_vertices = []
        for start, end in boundary_edges:
            side_vertices.append(
                [
                    (solid_xy[start, 0], solid_xy[start, 1], half_thickness[start]),
                    (solid_xy[end, 0], solid_xy[end, 1], half_thickness[end]),
                    (solid_xy[end, 0], solid_xy[end, 1], -half_thickness[end]),
                    (solid_xy[start, 0], solid_xy[start, 1], -half_thickness[start]),
                ]
            )
        return top_vertices, bottom_vertices, side_vertices

    ref_top, ref_bottom, ref_sides = solid_vertices(reference_half)
    def_top, def_bottom, def_sides = solid_vertices(transformed_half)
    triangle_change = solid_change[solid_visible_triangles].mean(axis=1)
    boundary_change = np.asarray(
        [
            0.5 * (solid_change[start] + solid_change[end])
            for start, end in boundary_edges
        ]
    )

    fig = plt.figure(figsize=(11, 8))
    axis = fig.add_subplot(111, projection="3d")

    # Transparent grey surfaces show the reference thickness.
    for vertices in (ref_top, ref_bottom):
        axis.add_collection3d(
            Poly3DCollection(
                vertices,
                facecolors=(0.65, 0.65, 0.65, 0.08),
                edgecolors=(0.25, 0.25, 0.25, 0.18),
                linewidths=0.08,
            )
        )
    axis.add_collection3d(
        Poly3DCollection(
            ref_sides,
            facecolors=(0.65, 0.65, 0.65, 0.04),
            edgecolors=(0.20, 0.20, 0.20, 0.42),
            linewidths=0.25,
        )
    )

    # Colored surfaces show z' = lambda_3 z, with the displacement exaggerated.
    transformed_colors = change_cmap(change_norm(triangle_change))
    axis.add_collection3d(
        Poly3DCollection(
            def_top,
            facecolors=transformed_colors,
            edgecolors=(1.0, 1.0, 1.0, 0.10),
            linewidths=0.06,
            alpha=0.88,
        )
    )
    axis.add_collection3d(
        Poly3DCollection(
            def_bottom,
            facecolors=transformed_colors,
            edgecolors="none",
            alpha=0.42,
        )
    )
    axis.add_collection3d(
        Poly3DCollection(
            def_sides,
            facecolors=change_cmap(change_norm(boundary_change)),
            edgecolors=(0.1, 0.1, 0.1, 0.18),
            linewidths=0.12,
            alpha=0.78,
        )
    )

    # Arrows explicitly show the movement from z to z' on both faces.
    arrow_indices = np.linspace(0, solid_xy.shape[0] - 1, 18, dtype=int)
    arrow_delta = transformed_half[arrow_indices] - reference_half[arrow_indices]
    for sign in (1.0, -1.0):
        axis.quiver(
            solid_xy[arrow_indices, 0],
            solid_xy[arrow_indices, 1],
            sign * reference_half[arrow_indices],
            np.zeros_like(arrow_delta),
            np.zeros_like(arrow_delta),
            sign * arrow_delta,
            color="black",
            linewidth=0.8,
            arrow_length_ratio=0.25,
            normalize=False,
        )

    axis.set_xlim(solid_xy[:, 0].min(), solid_xy[:, 0].max())
    axis.set_ylim(solid_xy[:, 1].min(), solid_xy[:, 1].max())
    z_limit = max(reference_half.max(), transformed_half.max()) * 1.2
    axis.set_zlim(-z_limit, z_limit)
    axis.set_xlabel("$x$")
    axis.set_ylabel("$y$")
    axis.set_zlabel("through-thickness coordinate $z$")
    axis.set_title(
        r"Thickness transformation $z'=\lambda_3(x,y)z$"
        + "\n"
        + rf"displacement exaggerated "
        + rf"$\times {SOLID_THICKNESS_EXAGGERATION:.0f}$"
    )
    axis.view_init(elev=24, azim=-58)
    axis.set_box_aspect((1.0, 1.0, 0.38))

    colorbar_axis = fig.add_axes((0.88, 0.18, 0.022, 0.62))
    fig.colorbar(
        plt.cm.ScalarMappable(norm=change_norm, cmap=change_cmap),
        cax=colorbar_axis,
        label=r"relative thickness change $\lambda_3-1$",
    )
    fig.text(
        0.48,
        0.04,
        "Grey: reference solid. Color: transformed solid. "
        "Arrows: top/bottom surface movement.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.88, bottom=0.10, right=0.84)
    solid_output_path = FIGURES_DIR / "elasticity_thin_solid_3d.png"
    fig.savefig(solid_output_path, bbox_inches="tight", dpi=220)
    plt.show()
    print(f"Saved {solid_output_path}")

# %% Diagram figures - input point cloud and output sigma_VM
IMPL_DIR = FIGURES_DIR

DIAG_SAMPLE = 0
_coords = coords[DIAG_SAMPLE]  # (P, 2)
_sigma = sigma[DIAG_SAMPLE]  # (P,)

_figsize = (4, 4)
_s = 20  # point size

# Input: geometry only, no field colour
fig, ax = plt.subplots(figsize=_figsize)
ax.scatter(_coords[:, 0], _coords[:, 1], s=_s, color="0.3", linewidths=0)
ax.set_aspect("equal")
ax.axis("off")
fig.savefig(IMPL_DIR / "elasticity_diagram_input.png", bbox_inches="tight", dpi=200)
plt.show()
print(f"Saved {IMPL_DIR / 'elasticity_diagram_input.png'}")

# Output: same geometry coloured by von Mises stress
fig, ax = plt.subplots(figsize=_figsize)
sc = ax.scatter(
    _coords[:, 0],
    _coords[:, 1],
    c=_sigma,
    cmap=CMAP,
    s=_s,
    linewidths=0,
    vmin=_sigma.min(),
    vmax=_sigma.max(),
)
ax.set_aspect("equal")
ax.axis("off")
# fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label=r"$\sigma_{VM}$")
fig.savefig(IMPL_DIR / "elasticity_diagram_output.png", bbox_inches="tight", dpi=200)
plt.show()
print(f"Saved {IMPL_DIR / 'elasticity_diagram_output.png'}")

# %% [SCAFFOLD] Prediction vs ground truth + latent kinematics
# TODO: scatter predicted sigma vs target sigma and abs error on the point cloud;
# scatter m, d, lambda_1/2/3, and det(F) from out["aux_tensors"] to confirm
# 3D incompressibility and the plane-stress condition.
# (exact incompressibility by construction). Save:
#   elasticity_prediction.png, elasticity_kinematics.png

# %% [SCAFFOLD] Results / param + FLOPs accounting
# TODO: collect baseline vs constrained rel-L2 across backbones into a table;
# the constraint head is small + pointwise, so report its param/FLOPs overhead
# (analogous to the NS cost table).
