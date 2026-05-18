# %% Imports & config
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import scipy.io as scio
import yaml

REPO_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
DATA_PATH = (
    REPO_ROOT / "data/NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat"
)
OUTPUTS_ROOT = REPO_ROOT / "outputs/navier_stokes/mean_correction"
FIGURES_DIR = REPO_ROOT / "docs/figures/ns"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CMAP = "viridis"
MODELS = {
    "Factformer": OUTPUTS_ROOT / "factformer/final/seed_42",
    "Transolver": OUTPUTS_ROOT / "transolver/final/seed_42",
    "Galerkin-T": OUTPUTS_ROOT / "galerkin_transformer/validation/seed_42",
    "ONO": OUTPUTS_ROOT / "ono/final/seed_42",
    "GNOT": OUTPUTS_ROOT / "gnot/final/seed_42",
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
raw = scio.loadmat(str(DATA_PATH))
u = raw["u"]  # (1200, 64, 64, 20) — vorticity
a = raw["a"]  # (1200, 64, 64)     — initial condition
t_vals = raw["t"].ravel()  # (20,)

N, H, W, T_FULL = u.shape
T_IN, T_OUT = 10, 10
print(f"u: {u.shape}  a: {a.shape}  t: {t_vals}")

# %% Dataset sample — vorticity evolution
SAMPLE_IDX = 0
INPUT_FRAMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

TARGET_FRAMES = [t + T_IN for t in INPUT_FRAMES]

_cell = 1.5  # inches per image
fig, axes = plt.subplots(
    2,
    len(INPUT_FRAMES),
    figsize=(len(INPUT_FRAMES) * _cell + 1, 2 * _cell),
    constrained_layout=True,
)

vmin, vmax = u[SAMPLE_IDX].min(), u[SAMPLE_IDX].max()
for col, (ti, tj) in enumerate(zip(INPUT_FRAMES, TARGET_FRAMES)):
    axes[0, col].imshow(
        u[SAMPLE_IDX, :, :, ti], cmap=CMAP, origin="lower", vmin=vmin, vmax=vmax
    )
    axes[0, col].set_title(f"$x_{{{ti + 1}}}$", fontsize=10)
    axes[0, col].axis("off")
    axes[1, col].imshow(
        u[SAMPLE_IDX, :, :, tj], cmap=CMAP, origin="lower", vmin=vmin, vmax=vmax
    )
    axes[1, col].set_title(f"$x_{{{tj + 1}}}$", fontsize=10)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Input context", fontsize=9)
axes[1, 0].set_ylabel("Target", fontsize=9)

fig.colorbar(
    plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax)),
    ax=axes.ravel().tolist(),
    fraction=0.015,
    pad=0.02,
    label="Vorticity",
)
# fig.suptitle("Navier-Stokes: vorticity field evolution (sample 0)", fontsize=11)
fig.savefig(FIGURES_DIR / "ns_dataset_sample.png", bbox_inches="tight")
plt.show()
print(f"Saved image to {FIGURES_DIR / 'ns_dataset_sample.pdf'}")


# %% Training curves
TRAINING_MODELS = {
    "Factformer": OUTPUTS_ROOT / "factformer/final/seed_42",
    "Transolver": OUTPUTS_ROOT / "transolver/final/seed_42",
    "Galerkin-T": OUTPUTS_ROOT / "galerkin_transformer/validation/seed_42",
    "ONO": OUTPUTS_ROOT / "ono/final/seed_42",
}


def load_train_csv(run_dir: Path) -> list[float]:
    vals = []
    with open(run_dir / "train_rel_l2.csv") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            vals.append(float(row[1]))
    return vals


def load_csv(run_dir: Path, filename: str) -> list[float]:
    vals = []
    with open(run_dir / filename) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            vals.append(float(row[1]))
    return vals


GT_DIR = OUTPUTS_ROOT / "galerkin_transformer/validation/seed_42"

fig, (ax, ax_gt) = plt.subplots(1, 2, figsize=(13, 4))

curve_colors = plt.get_cmap(CMAP)(np.linspace(0.15, 0.85, len(TRAINING_MODELS)))
for (name, run_dir), color in zip(TRAINING_MODELS.items(), curve_colors):
    vals = load_csv(run_dir, "train_rel_l2.csv")
    epochs = np.arange(1, len(vals) + 1)
    ax.plot(epochs, vals, label=name, color=color, lw=1.5)

ax.set_xlabel("Epoch")
ax.set_ylabel("Train relative $L_2$")
ax.set_title("Navier-Stokes: training curves")
ax.legend(frameon=False)

gt_train = load_csv(GT_DIR, "train_rel_l2.csv")
gt_val = load_csv(GT_DIR, "val_rel_l2.csv")
ax_gt.plot(
    np.arange(1, len(gt_train) + 1),
    gt_train,
    color=plt.get_cmap(CMAP)(0.3),
    lw=1.5,
    label="Train",
)
ax_gt.plot(
    np.arange(1, len(gt_val) + 1),
    gt_val,
    color=plt.get_cmap(CMAP)(0.7),
    lw=1.5,
    label="Validation",
)
ax_gt.set_xlabel("Epoch")
ax_gt.set_ylabel("Relative $L_2$")
ax_gt.set_title("Galerkin-T: train vs validation")
ax_gt.legend(frameon=False)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "ns_training_curves.png", bbox_inches="tight")
plt.show()


# %% Rollout inference — Factformer
import torch

from omni_hc.benchmarks.navier_stokes.adapter import (
    _init_task_state,
    _prepare_batch,
    _runtime_overrides,
)
from omni_hc.benchmarks.navier_stokes.data import build_test_loader
from omni_hc.integrations.nsl import create_model
from omni_hc.training.common import load_checkpoint_state, load_model_state_dict
from omni_hc.training.tasks.autoregressive import rollout_autoregressive

DEVICE = torch.device("cpu")
INFER_MODEL = "ONO"
RUN_DIR = MODELS[INFER_MODEL]

cfg = yaml.safe_load((RUN_DIR / "resolved_config.yaml").read_text())
cfg["paths"]["root_dir"] = str(DATA_PATH.parent)

test_loader = build_test_loader(cfg)
meta = test_loader.ns_meta

model, _, _ = create_model(
    cfg, device=DEVICE, runtime_overrides=_runtime_overrides(meta)
)
ckpt = load_checkpoint_state(RUN_DIR / "best.pt", device=DEVICE)
load_model_state_dict(model, ckpt["model_state_dict"])
model.eval()

task_state = _init_task_state(meta, sample_dtype=torch.float32, device=DEVICE)
batch = next(iter(test_loader))
coords, fx, target = _prepare_batch(batch, device=DEVICE, task_state=task_state)

with torch.no_grad():
    pred, _, _ = rollout_autoregressive(
        model,
        coords,
        fx,
        target,
        t_out=meta["t_out"],
        out_dim=meta["out_dim"],
        teacher_forcing=False,
    )

h, w = meta["shapelist"]
t_out = meta["t_out"]
pred_field = pred[0].view(h, w, t_out).cpu().numpy()
gt_field = target[0].view(h, w, t_out).cpu().numpy()

# %% Rollout visualisation — pred vs GT
SHOW_STEPS = [0, 3, 6, 9]

vmin_f = min(pred_field[..., SHOW_STEPS].min(), gt_field[..., SHOW_STEPS].min())
vmax_f = max(pred_field[..., SHOW_STEPS].max(), gt_field[..., SHOW_STEPS].max())
err_abs = np.abs(pred_field[..., SHOW_STEPS] - gt_field[..., SHOW_STEPS])
err_scale = err_abs.max()

fig, axes = plt.subplots(3, len(SHOW_STEPS), figsize=(12, 8))
row_labels = ["Ground truth", "Prediction", "Absolute error"]

for col, step in enumerate(SHOW_STEPS):
    gt = gt_field[:, :, step]
    pr = pred_field[:, :, step]
    err = np.abs(pr - gt)
    t_label = f"$t_{{{step + T_IN + 1}}}$"

    axes[0, col].imshow(gt, cmap=CMAP, origin="lower", vmin=vmin_f, vmax=vmax_f)
    axes[0, col].set_title(t_label, fontsize=10)
    axes[0, col].axis("off")

    axes[1, col].imshow(pr, cmap=CMAP, origin="lower", vmin=vmin_f, vmax=vmax_f)
    axes[1, col].axis("off")

    axes[2, col].imshow(err, cmap=CMAP, origin="lower", vmin=0, vmax=err_scale)
    axes[2, col].axis("off")

for row_idx, label in enumerate(row_labels):
    axes[row_idx, 0].set_ylabel(label, fontsize=9)

fig.colorbar(
    plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin_f, vmax=vmax_f)),
    ax=axes[:2],
    fraction=0.015,
    pad=0.02,
    label="Vorticity",
)
fig.colorbar(
    plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=0, vmax=err_scale)),
    ax=axes[2:],
    fraction=0.015,
    pad=0.02,
    label="|Error|",
)
fig.suptitle(f"NS rollout — {INFER_MODEL} (test sample 0)", fontsize=11)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "ns_rollout_prediction.png", bbox_inches="tight")
plt.show()


# %% Correction anatomy — ground truth + correction field
from omni_hc.training.common import forward_with_optional_aux

out_dim = meta["out_dim"]

with torch.no_grad():
    step_out = forward_with_optional_aux(model, coords, fx)

corr_np = step_out["aux_tensors"]["corr"][0].reshape(h, w).cpu().numpy()
gt_np = target[0, :, :out_dim].reshape(h, w).cpu().numpy()

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.imshow(gt_np, cmap=CMAP, origin="lower")
ax.axis("off")
fig.savefig(FIGURES_DIR / "ns_ground_truth.png", bbox_inches="tight", pad_inches=0)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.imshow(corr_np, cmap="Reds", origin="lower")
ax.axis("off")
fig.savefig(FIGURES_DIR / "ns_correction.png", bbox_inches="tight", pad_inches=0)
plt.show()
