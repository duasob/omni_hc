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

    fig = plt.figure(figsize=(14, 6))
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
