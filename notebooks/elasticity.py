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
SAMPLE_IDX = [0, 10, 100, 500, 1000]

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
    label=r"von Mises stress $\sigma_{VM}$",
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
        out = forward_with_optional_aux(model, coords_b, fx_b, return_aux=True)
    print("aux keys:", sorted(out.get("aux_tensors", {}).keys()))
else:
    print(f"[scaffold] no run at {RUN_DIR}; skip inference until a run exists.")

# %% [SCAFFOLD] Prediction vs ground truth + kinematics (det C ~ 1)
# TODO: scatter predicted sigma vs target sigma and abs error on the point cloud;
# scatter theta, lambda, det_c from out["aux_tensors"] to confirm det_c ~ 1
# (exact incompressibility by construction). Save:
#   elasticity_prediction.png, elasticity_kinematics.png

# %% [SCAFFOLD] Results / param + FLOPs accounting
# TODO: collect baseline vs constrained rel-L2 across backbones into a table;
# the constraint head is small + pointwise, so report its param/FLOPs overhead
# (analogous to the NS cost table).
