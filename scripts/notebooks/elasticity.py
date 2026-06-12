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

# %% Input-geometry gallery - point clouds coloured by von Mises stress
# An N x M grid of different samples (ground truth, no predictions), showing the
# variety of plate-with-hole geometries and their von Mises stress fields. A
# shared colour scale lets stress magnitudes be compared across samples.
GEO_ROWS = 4
GEO_COLS = 4
GEO_POINT_SIZE = 4

n_geo = coords.shape[0]
geo_indices = np.unique(np.linspace(0, n_geo - 1, GEO_ROWS * GEO_COLS).astype(int))

geo_vmin = float(min(sigma[idx].min() for idx in geo_indices))
geo_vmax = float(max(sigma[idx].max() for idx in geo_indices))

fig, axes = plt.subplots(
    GEO_ROWS,
    GEO_COLS,
    figsize=(2.0 * GEO_COLS, 2.0 * GEO_ROWS),
    sharex=True,
    sharey=True,
)
axes = np.atleast_1d(axes).reshape(-1)
for ax in axes:
    ax.set_axis_off()

geo_scatter = None
for ax, idx in zip(axes, geo_indices):
    geo_scatter = ax.scatter(
        coords[idx, :, 0],
        coords[idx, :, 1],
        c=sigma[idx],
        cmap=CMAP,
        s=GEO_POINT_SIZE,
        linewidths=0,
        vmin=geo_vmin,
        vmax=geo_vmax,
    )
    ax.set_aspect("equal")
    ax.set_title(f"sample {idx}", fontsize=8)

fig.subplots_adjust(hspace=0.12, wspace=0.06)
fig.colorbar(
    geo_scatter,
    ax=axes,
    fraction=0.025,
    pad=0.02,
    label=r"$\sigma_{VM}$",
)
out_path = FIGURES_DIR / "elasticity_geometry_gallery.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved {out_path}")

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

# %% Inference - load the trained latent plane-stress model
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
    / "outputs/elasticity/elasticity_plane_stress_vm_latent/transolver/e100_t900/seed_42"
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

# %% Constraint comparison — input / GT / prediction / error per model
# Qualitative panel over one test sample comparing the unconstrained baseline
# against the plane-stress von-Mises constraint in its two decoder variants:
# the scalar-output head (no latent) and the engineered latent decoder (which
# reads the constraint head from Transolver's internal representations). Columns
# are the input geometry, GT von-Mises stress, prediction, and absolute error;
# rows are the models. Run paths match scripts/reporting/registry/
# elasticity_report.py (the scalar row uses the backbone_out_dim_1 ablation).
# Predictions are cached under artifacts/elasticity/pred_cache and invalidated
# when a checkpoint is newer than its cache, so re-running the figure is cheap.
ELAS_PRED_CACHE_DIR = REPO_ROOT / "artifacts/elasticity/pred_cache"

CONSTRAINT_COMPARISON_RUNS = [
    ("Baseline", REPO_ROOT / "outputs/elasticity/none/transolver/final/seed_42"),
    (
        "Constrained (no latent)",
        REPO_ROOT
        / "outputs/elasticity/elasticity_plane_stress_vm_constraint/transolver/backbone_out_dim_1/final/seed_42",
    ),
    (
        "Constrained (latent)",
        REPO_ROOT
        / "outputs/elasticity/elasticity_plane_stress_vm_latent/transolver/final/seed_42",
    ),
]
CONSTRAINT_COMPARISON_SAMPLE = 0  # index into the canonical test split
CONSTRAINT_COMPARISON_POINT_SIZE = 8
ELAS_COMPARISON_DEVICE = torch.device("cpu")


def _load_elasticity_predictions(run_dir, sample_idxs, *, device):
    """Roll a trained elasticity checkpoint out on a few test samples.

    Returns ``{"coords", "pred", "target"}`` with per-sample point clouds. The
    predicted von-Mises stress is taken from the constraint's ``sigma_physical``
    aux tensor when present, and otherwise from the decoded scalar output (the
    unconstrained baseline), so all three variants are returned in physical units.
    """
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    cfg["paths"]["root_dir"] = str(DATA_DIR)
    loader = build_test_loader(cfg)
    meta = loader.elasticity_meta
    model, _, _ = create_model(
        cfg, device=device, runtime_overrides=_runtime_overrides(meta)
    )
    y_normalizer = loader.y_normalizer.to(device)
    constraint = getattr(model, "constraint", None)
    if constraint is not None and hasattr(constraint, "set_target_normalizer"):
        constraint.set_target_normalizer(y_normalizer)
    ckpt = load_checkpoint_state(run_dir / "best.pt", device=device)
    load_model_state_dict(model, ckpt["model_state_dict"])
    model.eval()

    keys = [key for key in ("coords", "x", "y") if key in loader.dataset[0]]
    batch = {
        key: torch.stack([loader.dataset[int(i)][key] for i in sample_idxs], dim=0)
        for key in keys
    }
    coords_b, fx_b, target_b = _prepare_batch(batch, device=device)
    with torch.no_grad():
        out = forward_with_optional_aux(model, coords_b, fx_b)
    aux = out.get("aux_tensors", {})
    if "sigma_physical" in aux:
        pred = aux["sigma_physical"][..., 0]
    else:
        pred = y_normalizer.decode(out["pred"])[..., 0]
    target = y_normalizer.decode(target_b)[..., 0]
    return {
        "coords": coords_b.cpu().numpy(),
        "pred": pred.cpu().numpy(),
        "target": target.cpu().numpy(),
    }


def _elas_pred_cache_path(run_dir, sample_idxs):
    # Slug from the trailing path parts (family .. seed), keeping the
    # backbone_out_dim_* ablation directory so variants do not collide.
    slug = "_".join(Path(run_dir).parts[-5:])
    idx_tag = "-".join(str(int(i)) for i in sample_idxs)
    return ELAS_PRED_CACHE_DIR / f"{slug}__samples_{idx_tag}.npz"


def load_elasticity_predictions_cached(run_dir, sample_idxs, *, device, use_cache=True):
    run_dir = Path(run_dir)
    cache_path = _elas_pred_cache_path(run_dir, sample_idxs)
    checkpoint = run_dir / "best.pt"
    if (
        use_cache
        and cache_path.exists()
        and (
            not checkpoint.exists()
            or checkpoint.stat().st_mtime <= cache_path.stat().st_mtime
        )
    ):
        cached = np.load(cache_path)
        print(f"[elasticity pred cache] HIT  {cache_path.name}")
        return {key: cached[key] for key in ("coords", "pred", "target")}

    print(f"[elasticity pred cache] MISS {cache_path.name} (rolling out checkpoint...)")
    fields = _load_elasticity_predictions(run_dir, sample_idxs, device=device)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **fields)
    return fields


ec_sample_idxs = [CONSTRAINT_COMPARISON_SAMPLE]
ec_fields = {
    label: load_elasticity_predictions_cached(
        run_dir, ec_sample_idxs, device=ELAS_COMPARISON_DEVICE
    )
    for label, run_dir in CONSTRAINT_COMPARISON_RUNS
}

# Geometry and GT are shared across models; take them from the first run.
ec_first = next(iter(ec_fields.values()))
ec_coords = ec_first["coords"][0]  # (P, 2)
ec_gt = ec_first["target"][0]  # (P,)
ec_preds = {label: f["pred"][0] for label, f in ec_fields.items()}
ec_errors = {
    label: np.abs(f["pred"][0] - f["target"][0]) for label, f in ec_fields.items()
}

# Shared colour scales: stress across GT + all predictions, error across models.
ec_svmin = float(min(ec_gt.min(), *(p.min() for p in ec_preds.values())))
ec_svmax = float(max(ec_gt.max(), *(p.max() for p in ec_preds.values())))
ec_evmax = float(max(e.max() for e in ec_errors.values())) or 1.0

ec_n = len(CONSTRAINT_COMPARISON_RUNS)
fig, axes = plt.subplots(ec_n, 3, figsize=(10.0, ec_n * 3.1), constrained_layout=True)
if ec_n == 1:
    axes = axes[None, :]

ec_titles = [
    r"GT $\sigma_{VM}$",
    r"Prediction $\hat{\sigma}_{VM}$",
    r"Error $|\hat{\sigma}-\sigma|$",
]
for row, (label, _run_dir) in enumerate(CONSTRAINT_COMPARISON_RUNS):
    sc_gt = axes[row, 0].scatter(
        ec_coords[:, 0], ec_coords[:, 1], c=ec_gt, cmap=CMAP,
        s=CONSTRAINT_COMPARISON_POINT_SIZE, linewidths=0, vmin=ec_svmin, vmax=ec_svmax,
    )
    fig.colorbar(sc_gt, ax=axes[row, 0], fraction=0.046, pad=0.04)
    sc_pred = axes[row, 1].scatter(
        ec_coords[:, 0], ec_coords[:, 1], c=ec_preds[label], cmap=CMAP,
        s=CONSTRAINT_COMPARISON_POINT_SIZE, linewidths=0, vmin=ec_svmin, vmax=ec_svmax,
    )
    fig.colorbar(sc_pred, ax=axes[row, 1], fraction=0.046, pad=0.04)
    sc_err = axes[row, 2].scatter(
        ec_coords[:, 0], ec_coords[:, 1], c=ec_errors[label], cmap="inferno",
        s=CONSTRAINT_COMPARISON_POINT_SIZE, linewidths=0, vmin=0.0, vmax=ec_evmax,
    )
    fig.colorbar(sc_err, ax=axes[row, 2], fraction=0.046, pad=0.04)

    rel_l2 = float(
        np.linalg.norm(ec_preds[label] - ec_gt) / max(np.linalg.norm(ec_gt), 1e-12)
    )
    axes[row, 2].set_title(rf"rel $L_2$={rel_l2:.3f}", fontsize=9)
    axes[row, 0].set_ylabel(label, fontsize=11)
    for col in range(3):
        axes[row, col].set_aspect("equal")
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

# Column headers on the top row; the error column header sits above the first
# row's rel-L2 so both stay visible.
for col, title in enumerate(ec_titles):
    if col == 2:
        axes[0, col].set_title(f"{title}\n{axes[0, col].get_title()}", fontsize=11)
    else:
        axes[0, col].set_title(title, fontsize=11)

out_path = FIGURES_DIR / "elasticity_constraint_comparison.png"
fig.savefig(out_path, bbox_inches="tight")
plt.show()
print(f"Saved {out_path}")

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
            np.linalg.norm(solid_triangle_xy[:, 0] - solid_triangle_xy[:, 1], axis=1),
            np.linalg.norm(solid_triangle_xy[:, 1] - solid_triangle_xy[:, 2], axis=1),
            np.linalg.norm(solid_triangle_xy[:, 2] - solid_triangle_xy[:, 0], axis=1),
        ),
        axis=1,
    )
    solid_tri.set_mask(solid_edges.max(axis=1) > 3.0 * np.median(solid_edges))
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

# %% Conceptual reparameterization - upright reference and deformed solids
# This is an explanatory illustration, not a model prediction. We prescribe a
# uniform principal-stretch state aligned with the specimen axes so that the
# roles of lambda_1 (width), lambda_2 (height), and lambda_3 (thickness) are
# visually clear. Incompressibility determines lambda_3 exactly.
import matplotlib.tri as mtri
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CONCEPT_SAMPLE = 0
CONCEPT_LAMBDA_1 = 0.90
CONCEPT_LAMBDA_2 = 1.40
CONCEPT_LAMBDA_3 = 1.0 / (CONCEPT_LAMBDA_1 * CONCEPT_LAMBDA_2)
CONCEPT_REFERENCE_THICKNESS = 0.16

concept_xy = coords[CONCEPT_SAMPLE].copy()
concept_x_min, concept_x_max = concept_xy[:, 0].min(), concept_xy[:, 0].max()
concept_y_min, concept_y_max = concept_xy[:, 1].min(), concept_xy[:, 1].max()
concept_x_mid = 0.5 * (concept_x_min + concept_x_max)

# Affine deformation for the illustration: horizontal contraction, vertical
# extension from the fixed lower boundary, and incompressible thickness change.
concept_xy_deformed = concept_xy.copy()
concept_xy_deformed[:, 0] = concept_x_mid + CONCEPT_LAMBDA_1 * (
    concept_xy[:, 0] - concept_x_mid
)
concept_xy_deformed[:, 1] = concept_y_min + CONCEPT_LAMBDA_2 * (
    concept_xy[:, 1] - concept_y_min
)

concept_tri = mtri.Triangulation(concept_xy[:, 0], concept_xy[:, 1])
concept_triangles = concept_tri.triangles
concept_triangle_xy = concept_xy[concept_triangles]
concept_edge_lengths = np.stack(
    (
        np.linalg.norm(concept_triangle_xy[:, 0] - concept_triangle_xy[:, 1], axis=1),
        np.linalg.norm(concept_triangle_xy[:, 1] - concept_triangle_xy[:, 2], axis=1),
        np.linalg.norm(concept_triangle_xy[:, 2] - concept_triangle_xy[:, 0], axis=1),
    ),
    axis=1,
)
concept_tri.set_mask(
    concept_edge_lengths.max(axis=1) > 3.0 * np.median(concept_edge_lengths)
)
concept_visible_triangles = concept_triangles[~concept_tri.mask]

concept_edge_counts = {}
for triangle in concept_visible_triangles:
    for start, end in (
        (triangle[0], triangle[1]),
        (triangle[1], triangle[2]),
        (triangle[2], triangle[0]),
    ):
        edge = tuple(sorted((int(start), int(end))))
        concept_edge_counts[edge] = concept_edge_counts.get(edge, 0) + 1
concept_boundary_edges = np.asarray(
    [edge for edge, count in concept_edge_counts.items() if count == 1],
    dtype=int,
)


def conceptual_solid(xy, half_thickness):
    """Return front/back faces and walls with vertical specimen coordinate."""
    front = np.stack(
        (
            xy[concept_visible_triangles, 0],
            np.full_like(xy[concept_visible_triangles, 0], -half_thickness),
            xy[concept_visible_triangles, 1],
        ),
        axis=-1,
    )
    back = front.copy()
    back[..., 1] = half_thickness

    walls = []
    for start, end in concept_boundary_edges:
        walls.append(
            [
                (xy[start, 0], -half_thickness, xy[start, 1]),
                (xy[end, 0], -half_thickness, xy[end, 1]),
                (xy[end, 0], half_thickness, xy[end, 1]),
                (xy[start, 0], half_thickness, xy[start, 1]),
            ]
        )
    return front, back, walls


concept_ref_half = 0.5 * CONCEPT_REFERENCE_THICKNESS
concept_def_half = concept_ref_half * CONCEPT_LAMBDA_3
concept_ref_front, concept_ref_back, concept_ref_walls = conceptual_solid(
    concept_xy, concept_ref_half
)
concept_def_front, concept_def_back, concept_def_walls = conceptual_solid(
    concept_xy_deformed, concept_def_half
)

fig = plt.figure(figsize=(13, 6))
concept_axes = [
    fig.add_subplot(121, projection="3d"),
    fig.add_subplot(122, projection="3d"),
]


def draw_concept_solid(
    axis,
    *,
    xy,
    half_thickness,
    front,
    back,
    walls,
    face_color,
    side_color,
    title,
    show_boundary_conditions,
):
    axis.add_collection3d(
        Poly3DCollection(
            back,
            facecolors=face_color,
            edgecolors="none",
            alpha=0.28,
        )
    )
    axis.add_collection3d(
        Poly3DCollection(
            front,
            facecolors=face_color,
            edgecolors=(1.0, 1.0, 1.0, 0.18),
            linewidths=0.08,
            alpha=0.92,
        )
    )
    axis.add_collection3d(
        Poly3DCollection(
            walls,
            facecolors=side_color,
            edgecolors=(0.15, 0.15, 0.15, 0.28),
            linewidths=0.18,
            alpha=0.82,
        )
    )

    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    z_min, z_max = xy[:, 1].min(), xy[:, 1].max()
    x_span = x_max - x_min
    z_span = z_max - z_min
    front_depth = -half_thickness

    # Fixed support at the lower edge.
    axis.plot(
        [x_min, x_max],
        [front_depth, front_depth],
        [z_min, z_min],
        color="0.15",
        lw=2.0,
        zorder=10,
    )
    for support_x in np.linspace(x_min, x_max, 13):
        axis.plot(
            [support_x, support_x - 0.035 * x_span],
            [front_depth, front_depth],
            [z_min, z_min - 0.045 * z_span],
            color="0.15",
            lw=1.0,
            zorder=10,
        )

    if show_boundary_conditions:
        # Upward applied traction on the top edge.
        traction_x = np.linspace(x_min + 0.08 * x_span, x_max - 0.08 * x_span, 6)
        axis.quiver(
            traction_x,
            np.full_like(traction_x, front_depth),
            np.full_like(traction_x, z_max + 0.01 * z_span),
            np.zeros_like(traction_x),
            np.zeros_like(traction_x),
            np.full_like(traction_x, 0.15 * z_span),
            color="tab:red",
            linewidth=1.5,
            arrow_length_ratio=0.25,
        )

    common_x_min = min(concept_xy[:, 0].min(), concept_xy_deformed[:, 0].min())
    common_x_max = max(concept_xy[:, 0].max(), concept_xy_deformed[:, 0].max())
    common_z_max = concept_xy_deformed[:, 1].max()
    axis.set_xlim(
        common_x_min - 0.12 * x_span,
        common_x_max + 0.12 * x_span,
    )
    axis.set_ylim(-0.17, 0.17)
    axis.set_zlim(
        concept_y_min - 0.08 * z_span,
        common_z_max + 0.22 * z_span,
    )
    axis.set_box_aspect((1.0, 0.42, 1.35))
    axis.view_init(elev=10, azim=-62)
    axis.set_title(title, fontsize=14, pad=10)
    axis.set_axis_off()


draw_concept_solid(
    concept_axes[0],
    xy=concept_xy,
    half_thickness=concept_ref_half,
    front=concept_ref_front,
    back=concept_ref_back,
    walls=concept_ref_walls,
    face_color=(0.62, 0.65, 0.69, 1.0),
    side_color=(0.42, 0.45, 0.50, 1.0),
    title="Reference membrane",
    show_boundary_conditions=True,
)
draw_concept_solid(
    concept_axes[1],
    xy=concept_xy_deformed,
    half_thickness=concept_def_half,
    front=concept_def_front,
    back=concept_def_back,
    walls=concept_def_walls,
    face_color=(0.20, 0.52, 0.82, 1.0),
    side_color=(0.09, 0.30, 0.56, 1.0),
    title="Illustrative latent deformation",
    show_boundary_conditions=True,
)

# Mark the out-of-plane thickness direction on the deformed side face.
concept_def_x_max = concept_xy_deformed[:, 0].max()
concept_def_z_mid = 0.5 * (
    concept_xy_deformed[:, 1].min() + concept_xy_deformed[:, 1].max()
)
concept_axes[1].quiver(
    concept_def_x_max,
    -concept_def_half,
    concept_def_z_mid,
    0.0,
    2.0 * concept_def_half,
    0.0,
    color="tab:purple",
    linewidth=2.0,
    arrow_length_ratio=0.25,
)
concept_axes[1].text(
    concept_def_x_max + 0.015,
    0.0,
    concept_def_z_mid,
    r"$\lambda_3$",
    color="tab:purple",
    fontsize=12,
)

# Flow arrow between the reference and reparameterized states.
flow_arrow = FancyArrowPatch(
    (0.475, 0.51),
    (0.535, 0.51),
    transform=fig.transFigure,
    arrowstyle="-|>",
    mutation_scale=22,
    linewidth=1.8,
    color="0.2",
)
fig.add_artist(flow_arrow)

fig.text(0.245, 0.10, "fixed", ha="center", fontsize=12)
fig.text(0.755, 0.10, "fixed", ha="center", fontsize=12)
fig.text(
    0.5,
    0.885,
    r"applied tension traction $t$",
    ha="center",
    color="tab:red",
    fontsize=12,
)

stretch_text = (
    rf"$\lambda_1={CONCEPT_LAMBDA_1:.2f}$  width"
    + "\n"
    + rf"$\lambda_2={CONCEPT_LAMBDA_2:.2f}$  height"
    + "\n"
    + rf"$\lambda_3={CONCEPT_LAMBDA_3:.3f}$  thickness"
    + "\n"
    + r"$\lambda_1\lambda_2\lambda_3=1$"
)
fig.text(
    0.505,
    0.30,
    stretch_text,
    ha="center",
    va="center",
    fontsize=10.5,
    bbox={
        "boxstyle": "round,pad=0.45",
        "facecolor": "white",
        "edgecolor": "0.75",
        "alpha": 0.92,
    },
)
fig.suptitle(
    "Plane-stress reparameterisation: an interpretable local deformation",
    fontsize=16,
    y=0.995,
)
fig.text(
    0.5,
    0.035,
    "Illustrative stretches aligned with the specimen axes; values are not model predictions.",
    ha="center",
    fontsize=10,
    color="0.3",
)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.90, wspace=0.02)
concept_output_path = FIGURES_DIR / "elasticity_reparameterization_intuition.png"
fig.savefig(concept_output_path, bbox_inches="tight", dpi=220)
plt.show()
print(f"Saved {concept_output_path}")

# %% Real latent reparameterization - upright transformed solid
# The model predicts local principal-stretch magnitudes but not their directions
# or a globally compatible displacement field. We therefore retain the supplied
# in-plane point-cloud geometry and visualize the identifiable through-thickness
# transformation implied by the predicted lambda_3 field.
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageChops

if "out" not in globals():
    print("[real latent visualization] run the inference cell first.")
else:
    REAL_SAMPLE = 0
    # Plotting controls. These are display units because the dataset does not
    # provide a physical reference thickness h_0.
    REAL_REFERENCE_THICKNESS = 0.16  # total thickness when lambda_3 = 1
    REAL_DEPTH_HALF_RANGE = 0.30  # fixed camera window in the thickness direction
    REAL_THICKNESS_EXAGGERATION = 1_000.0  # exaggerates lambda_3 - 1 only
    REAL_COLOR_SCALE_POWER = 4  # display 10^4 (lambda_3 - 1) on the colorbar

    real_aux = out["aux_tensors"]
    real_xy = coords_b[REAL_SAMPLE].cpu().numpy()
    real_lambda_1 = real_aux["lambda_1"][REAL_SAMPLE, :, 0].cpu().numpy()
    real_lambda_2 = real_aux["lambda_2"][REAL_SAMPLE, :, 0].cpu().numpy()
    real_lambda_3 = real_aux["lambda_3"][REAL_SAMPLE, :, 0].cpu().numpy()
    real_det_f = real_aux["full_det_f"][REAL_SAMPLE, :, 0].cpu().numpy()
    real_thickness_change = real_lambda_3 - 1.0

    # The plotted thickness is
    #   h_plot = h_ref [1 + E (lambda_3 - 1)].
    # REAL_REFERENCE_THICKNESS controls its baseline size, while E controls how
    # strongly the small predicted variations alter that size.
    real_lambda_3_visual = 1.0 + REAL_THICKNESS_EXAGGERATION * real_thickness_change
    real_lambda_3_visual = np.maximum(real_lambda_3_visual, 0.08)
    real_half_thickness = 0.5 * REAL_REFERENCE_THICKNESS * real_lambda_3_visual

    real_tri = mtri.Triangulation(real_xy[:, 0], real_xy[:, 1])
    real_triangles = real_tri.triangles
    real_triangle_xy = real_xy[real_triangles]
    real_edge_lengths = np.stack(
        (
            np.linalg.norm(real_triangle_xy[:, 0] - real_triangle_xy[:, 1], axis=1),
            np.linalg.norm(real_triangle_xy[:, 1] - real_triangle_xy[:, 2], axis=1),
            np.linalg.norm(real_triangle_xy[:, 2] - real_triangle_xy[:, 0], axis=1),
        ),
        axis=1,
    )
    real_tri.set_mask(
        real_edge_lengths.max(axis=1) > 3.0 * np.median(real_edge_lengths)
    )
    real_visible_triangles = real_triangles[~real_tri.mask]

    real_edge_counts = {}
    for triangle in real_visible_triangles:
        for start, end in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(start), int(end))))
            real_edge_counts[edge] = real_edge_counts.get(edge, 0) + 1
    real_boundary_edges = np.asarray(
        [edge for edge, count in real_edge_counts.items() if count == 1],
        dtype=int,
    )

    # Plot coordinates are (specimen x, through-thickness z, specimen y), so
    # the fixed-to-loaded direction is vertical in the rendered figure.
    real_front = np.stack(
        (
            real_xy[real_visible_triangles, 0],
            -real_half_thickness[real_visible_triangles],
            real_xy[real_visible_triangles, 1],
        ),
        axis=-1,
    )
    real_back = real_front.copy()
    real_back[..., 1] = real_half_thickness[real_visible_triangles]

    real_walls = []
    real_wall_values = []
    for start, end in real_boundary_edges:
        real_walls.append(
            [
                (
                    real_xy[start, 0],
                    -real_half_thickness[start],
                    real_xy[start, 1],
                ),
                (
                    real_xy[end, 0],
                    -real_half_thickness[end],
                    real_xy[end, 1],
                ),
                (
                    real_xy[end, 0],
                    real_half_thickness[end],
                    real_xy[end, 1],
                ),
                (
                    real_xy[start, 0],
                    real_half_thickness[start],
                    real_xy[start, 1],
                ),
            ]
        )
        real_wall_values.append(
            0.5 * (real_thickness_change[start] + real_thickness_change[end])
        )

    real_color_multiplier = 10.0**REAL_COLOR_SCALE_POWER
    real_color_values = real_color_multiplier * real_thickness_change
    real_face_values = real_color_values[real_visible_triangles].mean(axis=1)
    real_wall_values = real_color_multiplier * np.asarray(real_wall_values)
    real_abs_limit = max(
        abs(float(real_color_values.min())),
        abs(float(real_color_values.max())),
    )
    real_norm = TwoSlopeNorm(
        vmin=-real_abs_limit,
        vcenter=0.0,
        vmax=real_abs_limit,
    )
    real_cmap = plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(6.2, 7.0))
    real_ax = fig.add_axes((-0.10, -0.08, 0.98, 1.14), projection="3d")
    real_ax.add_collection3d(
        Poly3DCollection(
            real_back,
            facecolors=real_cmap(real_norm(real_face_values)),
            edgecolors="none",
            alpha=0.25,
        )
    )
    real_ax.add_collection3d(
        Poly3DCollection(
            real_front,
            facecolors=real_cmap(real_norm(real_face_values)),
            edgecolors=(1.0, 1.0, 1.0, 0.16),
            linewidths=0.08,
            alpha=0.96,
        )
    )
    real_ax.add_collection3d(
        Poly3DCollection(
            real_walls,
            facecolors=real_cmap(real_norm(real_wall_values)),
            edgecolors=(0.12, 0.12, 0.12, 0.28),
            linewidths=0.18,
            alpha=0.88,
        )
    )

    real_x_min, real_x_max = real_xy[:, 0].min(), real_xy[:, 0].max()
    real_y_min, real_y_max = real_xy[:, 1].min(), real_xy[:, 1].max()
    real_x_span = real_x_max - real_x_min
    real_y_span = real_y_max - real_y_min
    real_front_depth = -float(real_half_thickness.max())

    # Fixed lower edge.
    real_ax.plot(
        [real_x_min, real_x_max],
        [real_front_depth, real_front_depth],
        [real_y_min, real_y_min],
        color="0.12",
        lw=2.2,
        zorder=10,
    )
    for support_x in np.linspace(real_x_min, real_x_max, 13):
        real_ax.plot(
            [support_x, support_x - 0.035 * real_x_span],
            [real_front_depth, real_front_depth],
            [real_y_min, real_y_min - 0.045 * real_y_span],
            color="0.12",
            lw=1.0,
            zorder=10,
        )

    # Applied tension traction on the upper edge.
    real_traction_x = np.linspace(
        real_x_min + 0.08 * real_x_span,
        real_x_max - 0.08 * real_x_span,
        7,
    )
    real_ax.quiver(
        real_traction_x,
        np.full_like(real_traction_x, real_front_depth),
        np.full_like(real_traction_x, real_y_max + 0.01 * real_y_span),
        np.zeros_like(real_traction_x),
        np.zeros_like(real_traction_x),
        np.full_like(real_traction_x, 0.14 * real_y_span),
        color="k",
        linewidth=1.5,
        arrow_length_ratio=0.25,
    )

    real_ax.set_xlim(
        real_x_min - 0.12 * real_x_span,
        real_x_max + 0.12 * real_x_span,
    )
    # Keep this view window fixed. If it followed real_half_thickness.max(),
    # Matplotlib would rescale the axis and visually cancel thickness changes.
    real_ax.set_ylim(-REAL_DEPTH_HALF_RANGE, REAL_DEPTH_HALF_RANGE)
    real_ax.set_zlim(
        real_y_min - 0.08 * real_y_span,
        real_y_max + 0.22 * real_y_span,
    )
    real_ax.set_box_aspect((1.0, 0.42, 1.35))
    real_ax.view_init(elev=10, azim=-62)
    real_ax.set_axis_off()
    # real_ax.set_title(
    #     "Predicted latent thickness transformation\n"
    #     rf"visible thickness variation exaggerated "
    #     rf"$\times{REAL_THICKNESS_EXAGGERATION:.0f}$",
    #     fontsize=15,
    #     pad=12,
    # )

    real_scalar_mappable = plt.cm.ScalarMappable(norm=real_norm, cmap=real_cmap)
    real_scalar_mappable.set_array(real_color_values)
    real_colorbar_ax = fig.add_axes((0.82, 0.14, 0.035, 0.72))
    real_colorbar = fig.colorbar(
        real_scalar_mappable,
        cax=real_colorbar_ax,
    )
    real_colorbar.set_label(
        rf"$10^{{{REAL_COLOR_SCALE_POWER}}}(\lambda_3-1)$",
        fontsize=11,
    )

    # real_stretch_text = (
    #     rf"$\lambda_1 \in [{real_lambda_1.min():.6f},"
    #     rf" {real_lambda_1.max():.6f}]$"
    #     + "\n"
    #     + rf"$\lambda_2 \in [{real_lambda_2.min():.6f},"
    #     rf" {real_lambda_2.max():.6f}]$"
    #     + "\n"
    #     + rf"$\lambda_3 \in [{real_lambda_3.min():.6f},"
    #     rf" {real_lambda_3.max():.6f}]$"
    #     + "\n"
    #     + rf"$\max|\det F-1|={np.max(np.abs(real_det_f - 1.0)):.1e}$"
    # )
    # fig.text(
    #     0.04,
    #     0.22,
    #     real_stretch_text,
    #     fontsize=10.5,
    #     bbox={
    #         "boxstyle": "round,pad=0.45",
    #         "facecolor": "white",
    #         "edgecolor": "0.75",
    #         "alpha": 0.94,
    #     },
    # )
    # fig.text(
    #     0.57,
    #     0.815,
    #     r"$t$",
    #     ha="center",
    #     color="k",
    #     fontsize=11.5,
    # )
    # fig.text(0.55, 0.15, "fixed", ha="center", fontsize=11.5)
    # fig.text(
    #     0.5,
    #     0.015,
    #     "In-plane geometry is retained because principal directions and "
    #     "a compatible displacement field are not observed.",
    #     ha="center",
    #     fontsize=9.5,
    #     color="0.3",
    # )
    real_output_path = FIGURES_DIR / "elasticity_reparameterization_real_output.png"
    fig.savefig(real_output_path, bbox_inches="tight", pad_inches=0.02, dpi=220)

    # A hidden 3D axes rectangle can survive bbox_inches="tight". Crop the
    # remaining near-white border to the actual rendered content.
    with Image.open(real_output_path) as real_image:
        real_rgb = real_image.convert("RGB")
        real_background = Image.new("RGB", real_rgb.size, "white")
        real_difference = ImageChops.difference(real_rgb, real_background).convert("L")
        real_content_bbox = real_difference.point(
            lambda value: 255 if value > 8 else 0
        ).getbbox()
        if real_content_bbox is not None:
            real_crop_padding = 12
            left, top, right, bottom = real_content_bbox
            real_content_bbox = (
                max(0, left - real_crop_padding),
                max(0, top - real_crop_padding),
                min(real_rgb.width, right + real_crop_padding),
                min(real_rgb.height, bottom + real_crop_padding),
            )
            real_rgb.crop(real_content_bbox).save(real_output_path)

    plt.show()
    print(f"Saved {real_output_path}")

# %% Diagram figures - input point cloud and output sigma_VM
# Use the same sample as the 3D latent visualization (coords_b[REAL_SAMPLE]) so
# the pipeline diagram and the 3D figure in the report show the same geometry.
# The test loader uses the last ntest dataset samples, so coords_b[0] is NOT
# coords[0]; tie both figures to the test batch. Colour the output by the
# model's predicted sigma_VM to match the $\hat{\sigma}_{VM}$ label.
IMPL_DIR = FIGURES_DIR

DIAG_SAMPLE = 0  # index into the test batch; matches REAL/VIS/SOLID_SAMPLE
if "coords_b" in globals():
    _coords = coords_b[DIAG_SAMPLE].cpu().numpy()  # (P, 2)
    if "out" in globals():
        _sigma = out["aux_tensors"]["sigma_physical"][DIAG_SAMPLE, :, 0].cpu().numpy()
    else:
        _sigma = target_b[DIAG_SAMPLE, :, 0].cpu().numpy()
else:
    print("[diagram] no model run; falling back to raw dataset sample 0.")
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

# %% Setup schematic - input (BCs) -> Neural operator -> stress output
# A single figure mirroring the canonical elasticity setup: a fixed bottom edge,
# a tension traction on the top edge, the geometry point cloud as input, and the
# predicted von Mises stress field as output.
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

SCHEM_SAMPLE = 0
pts = coords[SCHEM_SAMPLE]  # (P, 2)
field = sigma[SCHEM_SAMPLE]  # (P,)

x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
span = x_max - x_min

fig = plt.figure(figsize=(12, 4.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.75, 1.0], wspace=0.05)
ax_in = fig.add_subplot(gs[0])
ax_mid = fig.add_subplot(gs[1])
ax_out = fig.add_subplot(gs[2])

# A framing box drawn around each point cloud at the geometry bounds.
pad = 0.04 * span


def frame(ax):
    ax.add_patch(
        Rectangle(
            (x_min - pad, y_min - pad),
            (x_max - x_min) + 2 * pad,
            (y_max - y_min) + 2 * pad,
            fill=False,
            edgecolor="0.2",
            linewidth=1.2,
        )
    )


# --- Input panel: geometry + boundary conditions ---
ax_in.scatter(pts[:, 0], pts[:, 1], s=14, color="0.25", linewidths=0)
frame(ax_in)

# Tension traction: upward arrows rising from the top of the input box.
arrow_x = np.linspace(x_min + 0.06 * span, x_max - 0.06 * span, 6)
arrow_base = y_max + pad
arrow_len = 0.16 * span
for ax_x in arrow_x:
    ax_in.annotate(
        "",
        xy=(ax_x, arrow_base + arrow_len),
        xytext=(ax_x, arrow_base),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "k",
            "lw": 1.6,
            "mutation_scale": 12,
        },
        annotation_clip=False,
        zorder=5,
    )
ax_in.text(
    0.5 * (x_min + x_max),
    arrow_base + arrow_len + 0.03 * span,
    r"Tension traction $t$",
    ha="center",
    va="bottom",
    color="k",
    fontsize=12,
    clip_on=False,
)

# Fixed support: a baseline with hatch ticks under the bottom edge.
base_y = y_min - pad
ax_in.plot([x_min, x_max], [base_y, base_y], color="0.1", lw=1.4)
tick = 0.045 * span
for hx in np.linspace(x_min, x_max, 14):
    ax_in.plot([hx, hx - tick], [base_y, base_y - tick], color="0.1", lw=1.0)
ax_in.text(
    0.5 * (x_min + x_max),
    base_y - tick - 0.04 * span,
    "fixed",
    ha="center",
    va="top",
    fontsize=12,
)

ax_in.set_aspect("equal")
ax_in.set_xlim(x_min - pad - 0.02 * span, x_max + pad + 0.02 * span)
ax_in.set_ylim(
    base_y - tick - 0.10 * span,
    arrow_base + arrow_len + 0.12 * span,
)
ax_in.axis("off")

# --- Middle panel: neural operator box with flow arrows ---
ax_mid.set_xlim(0, 1)
ax_mid.set_ylim(0, 1)
ax_mid.axis("off")
box = FancyBboxPatch(
    (0.18, 0.42),
    0.64,
    0.16,
    boxstyle="round,pad=0.02",
    linewidth=1.2,
    edgecolor="0.4",
    facecolor="white",
)
ax_mid.add_patch(box)
ax_mid.text(0.5, 0.50, "Model", ha="center", va="center", fontsize=12)
ax_mid.add_patch(
    FancyArrowPatch(
        (0.0, 0.50),
        (0.17, 0.50),
        arrowstyle="-|>",
        mutation_scale=16,
        color="0.2",
        lw=1.4,
    )
)
ax_mid.add_patch(
    FancyArrowPatch(
        (0.83, 0.50),
        (1.0, 0.50),
        arrowstyle="-|>",
        mutation_scale=16,
        color="0.2",
        lw=1.4,
    )
)

# --- Output panel: predicted von Mises stress ---
sc = ax_out.scatter(
    pts[:, 0],
    pts[:, 1],
    c=field,
    cmap=CMAP,
    s=14,
    linewidths=0,
    vmin=field.min(),
    vmax=field.max(),
)
frame(ax_out)
ax_out.set_aspect("equal")
ax_out.axis("off")
ax_out.set_title("Stress", fontsize=12)
fig.colorbar(sc, ax=ax_out, fraction=0.046, pad=0.04, label=r"$\sigma_{VM}$")

fig.savefig(
    FIGURES_DIR / "elasticity_setup_schematic.png", bbox_inches="tight", dpi=200
)
plt.show()
print(f"Saved {FIGURES_DIR / 'elasticity_setup_schematic.png'}")
