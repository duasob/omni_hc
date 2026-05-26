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
RUN_DIR = REPO_ROOT / "outputs/elasticity/fno_deviatoric_stress"  # TODO: real run

if (RUN_DIR / "resolved_config.yaml").exists():
    cfg = yaml.safe_load((RUN_DIR / "resolved_config.yaml").read_text())
    cfg["paths"]["root_dir"] = str(DATA_DIR)

    test_loader = build_test_loader(cfg)
    meta = test_loader.elasticity_meta

    model, _, _ = create_model(
        cfg, device=DEVICE, runtime_overrides=_runtime_overrides(meta)
    )
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

# %% Strain-ellipse diagram - what theta and lambda mean per point
# C = R(theta) diag(lambda^2, lambda^-2) R(theta)^T maps a unit circle of
# material directions to an ellipse with semi-axes (lambda, 1/lambda) tilted by
# theta. Area is preserved (lambda * 1/lambda = 1), so incompressibility is
# directly visible as constant glyph area.
from matplotlib.collections import PatchCollection
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch

DIAGRAM_SAMPLE = 0
GLYPH_FRACTION = 1 / 65  # base glyph radius as a fraction of the cloud extent
AR_EXAGGERATE = 12.0  # aspect-ratio exponent; lambda sits near 1 so amplify it
# The trained checkpoint currently collapses to a near-constant (theta, lambda);
# default to a smoothly varying illustrative field that actually shows what the
# two parameters mean. Flip to True once a run predicts a non-degenerate field.
USE_MODEL_PREDICTIONS = False

aux = out.get("aux_tensors", {}) if "out" in globals() else {}
if USE_MODEL_PREDICTIONS and aux.get("lambda") is not None:
    coords_d = coords_b[DIAGRAM_SAMPLE].cpu().numpy()
    theta_d = aux["theta"][DIAGRAM_SAMPLE, :, 0].cpu().numpy()
    lambda_d = aux["lambda"][DIAGRAM_SAMPLE, :, 0].cpu().numpy()
    source = "model prediction"
    if lambda_d.std() < 1e-3:
        print(
            f"[warn] predicted lambda nearly constant "
            f"(std={lambda_d.std():.2e}); glyph field will look uniform."
        )
else:
    coords_d = coords[DIAGRAM_SAMPLE]
    cx, cy = coords_d[:, 0].mean(), coords_d[:, 1].mean()
    rel = coords_d - np.array([cx, cy])
    radius = np.hypot(rel[:, 0], rel[:, 1])
    theta_d = np.arctan2(rel[:, 1], rel[:, 0])  # orientation fans out radially
    lambda_d = 1.0 + 0.06 * (radius / (radius.max() + 1e-9))  # stretch grows out
    source = ""

extent = max(
    coords_d[:, 0].max() - coords_d[:, 0].min(),
    coords_d[:, 1].max() - coords_d[:, 1].min(),
)
glyph_r = GLYPH_FRACTION * extent

sub = np.arange(len(coords_d))

fig, (ax_concept, ax_field) = plt.subplots(
    1,
    2,
    figsize=(6, 10),
)

# --- concept inset: unit circle -> strain ellipse, defining theta and lambda --
lam0, theta0 = 1.6, np.deg2rad(35.0)
deg0 = np.rad2deg(theta0)
accent = plt.get_cmap(CMAP)(0.55)

ax_concept.add_patch(Circle((0, 0), 1.0, fill=False, ls="--", lw=1.0, ec="0.55"))
ax_concept.add_patch(
    Ellipse(
        (0, 0),
        width=2 * lam0,
        height=2 / lam0,
        angle=deg0,
        facecolor=accent,
        alpha=0.22,
        edgecolor=accent,
        lw=1.8,
    )
)
maj = lam0 * np.array([np.cos(theta0), np.sin(theta0)])
minr = (1 / lam0) * np.array([-np.sin(theta0), np.cos(theta0)])
for vec, lab in ((maj, r"$\lambda$"), (minr, r"$\lambda^{-1}$")):
    ax_concept.add_patch(
        FancyArrowPatch(
            (0, 0), vec, arrowstyle="-|>", mutation_scale=12, lw=1.6, color="0.15"
        )
    )
    ax_concept.annotate(lab, vec * 1.18, ha="center", va="center", fontsize=12)
ax_concept.add_patch(Arc((0, 0), 0.9, 0.9, theta1=0.0, theta2=deg0, color="0.15"))
ax_concept.plot([0, 1.3], [0, 0], color="0.6", lw=0.8, ls=":")
ax_concept.annotate(
    r"$\theta$",
    (0.62 * np.cos(theta0 / 2), 0.62 * np.sin(theta0 / 2)),
    ha="center",
    va="center",
    fontsize=12,
)
ax_concept.annotate(
    "unit circle",
    (0, -1.0),
    xytext=(0, -1.5),
    ha="center",
    fontsize=12,
    color="0.25",
    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8),
)
ax_concept.set_xlim(-1.9, 1.9)
ax_concept.set_ylim(-1.9, 1.9)
ax_concept.set_aspect("equal")
ax_concept.axis("off")
ax_concept.set_title(r"$\mathbf{C}$ mapping", fontsize=16, pad=-20)

# --- field: one glyph per (subsampled) point over the body --------------------
ax_field.scatter(coords_d[:, 0], coords_d[:, 1], s=3, color="0.85", zorder=0)
a = glyph_r * lambda_d[sub] ** AR_EXAGGERATE
b = glyph_r * lambda_d[sub] ** -AR_EXAGGERATE
glyphs = [
    Ellipse(
        (coords_d[i, 0], coords_d[i, 1]),
        width=2 * ai,
        height=2 * bi,
        angle=np.rad2deg(theta_d[i]),
    )
    for i, ai, bi in zip(sub, a, b)
]
pc = PatchCollection(glyphs, cmap=CMAP, zorder=1, edgecolor="white", linewidth=0.4)
pc.set_array(lambda_d[sub])
ax_field.add_collection(pc)
ax_field.autoscale_view()
ax_field.set_aspect("equal")
ax_field.set_xlabel("$x$")
ax_field.set_ylabel("$y$")
ax_field.set_title(
    r"angle $=\theta$, aspect $=\lambda{:}\lambda^{-1}$",
    fontsize=16,
)
ax_field.text(
    0.98,
    0.02,
    source,
    transform=ax_field.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    color="0.4",
)
fig.colorbar(pc, ax=ax_field, fraction=0.046, pad=0.02, label=r"$\lambda$")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "elasticity_spectral_glyphs.png", bbox_inches="tight")
plt.show()
print(f"Saved {FIGURES_DIR / 'elasticity_spectral_glyphs.png'}  [{source}]")

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

# %% [SCAFFOLD] Prediction vs ground truth + kinematics (det C ~ 1)
# TODO: scatter predicted sigma vs target sigma and abs error on the point cloud;
# scatter theta, lambda, det_c from out["aux_tensors"] to confirm det_c ~ 1
# (exact incompressibility by construction). Save:
#   elasticity_prediction.png, elasticity_kinematics.png

# %% [SCAFFOLD] Results / param + FLOPs accounting
# TODO: collect baseline vs constrained rel-L2 across backbones into a table;
# the constraint head is small + pointwise, so report its param/FLOPs overhead
# (analogous to the NS cost table).
