# %% Imports & config
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as patches
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
OUTPUTS_ROOT = REPO_ROOT / "outputs/navier_stokes/mean_constraint"
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
    axes[0, col].set_title(f"$t_{{{ti + 1}}}$", fontsize=14)
    axes[0, col].axis("off")
    axes[1, col].imshow(
        u[SAMPLE_IDX, :, :, tj], cmap=CMAP, origin="lower", vmin=vmin, vmax=vmax
    )
    axes[1, col].set_title(f"$t_{{{tj + 1}}}$", fontsize=14)
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


# %% Presentation asset — autoregressive problem animation
DIAGRAM_TEXT = "#1a1a1a"


def _render_rgba_frames(
    fig,
    update_fn,
    frame_count: int,
    *,
    dpi: int = 150,
):
    """Render full RGBA frames so transparent text does not accumulate."""
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Saving transparent animation frames requires Pillow."
        ) from exc

    fig.set_dpi(dpi)
    frames = []
    for frame_idx in range(frame_count):
        update_fn(frame_idx)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        frames.append(Image.fromarray(rgba, mode="RGBA"))
    return frames


def _save_rgba_animation(frames, out_path: Path, *, fps: int) -> Path:
    """Save RGBA frames as WebP or GIF. Prefer WebP for reliable transparency."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(1000 / max(int(fps), 1)), 1)
    suffix = out_path.suffix.lower()
    if suffix == ".webp":
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            lossless=True,
            quality=100,
            method=6,
        )
    elif suffix == ".gif":
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
        )
    else:
        raise ValueError(f"Unsupported animation format: {out_path.suffix}")
    return out_path


PRESENTATION_SAMPLE_IDX = 0
PRESENTATION_FRAME_COUNT = min(T_FULL - 1, 19)
PRESENTATION_FPS = 2

presentation_seq = u[PRESENTATION_SAMPLE_IDX]
pres_vmin = float(presentation_seq.min())
pres_vmax = float(presentation_seq.max())

fig = plt.figure(figsize=(10.5, 3.2), facecolor="none")
fig.patch.set_alpha(0)
canvas = fig.add_axes([0, 0, 1, 1])
canvas.patch.set_alpha(0)
canvas.set_axis_off()
canvas.set_xlim(0, 1)
canvas.set_ylim(0, 1)

left_ax = fig.add_axes([0.075, 0.24, 0.27, 0.56], facecolor="none")
right_ax = fig.add_axes([0.655, 0.24, 0.27, 0.56], facecolor="none")
for field_ax in (left_ax, right_ax):
    field_ax.patch.set_alpha(0)
    field_ax.set_axis_off()

left_img = left_ax.imshow(
    presentation_seq[:, :, 0],
    cmap=CMAP,
    origin="lower",
    vmin=pres_vmin,
    vmax=pres_vmax,
)
right_img = right_ax.imshow(
    presentation_seq[:, :, 1],
    cmap=CMAP,
    origin="lower",
    vmin=pres_vmin,
    vmax=pres_vmax,
)

model_box = patches.FancyBboxPatch(
    (0.425, 0.415),
    0.15,
    0.16,
    boxstyle="round,pad=0.02,rounding_size=0.025",
    linewidth=1.8,
    edgecolor=DIAGRAM_TEXT,
    facecolor="none",
)
canvas.add_patch(model_box)
canvas.text(
    0.5,
    0.495,
    "model",
    ha="center",
    va="center",
    color=DIAGRAM_TEXT,
    fontsize=18,
    fontweight="semibold",
)

arrow_style = dict(arrowstyle="->", color=DIAGRAM_TEXT, lw=2.0, mutation_scale=18)
canvas.annotate("", xy=(0.405, 0.495), xytext=(0.3, 0.495), arrowprops=arrow_style)
canvas.annotate("", xy=(0.695, 0.495), xytext=(0.595, 0.495), arrowprops=arrow_style)
canvas.plot(
    [0.795, 0.795, 0.205],
    [0.80, 0.90, 0.90],
    color=DIAGRAM_TEXT,
    lw=2.0,
    solid_capstyle="round",
)
canvas.annotate(
    "",
    xy=(0.205, 0.80),
    xytext=(0.205, 0.90),
    arrowprops=arrow_style,
)

left_label = canvas.text(
    0.21,
    0.135,
    r"$\omega_{t_i}$",
    ha="center",
    va="center",
    color=DIAGRAM_TEXT,
    fontsize=16,
)
right_label = canvas.text(
    0.79,
    0.135,
    r"$\hat{\omega}_{t_{i+1}}$",
    ha="center",
    va="center",
    color=DIAGRAM_TEXT,
    fontsize=16,
)


def _update_presentation_frame(frame_idx: int):
    left_img.set_data(presentation_seq[:, :, frame_idx])
    right_img.set_data(presentation_seq[:, :, frame_idx + 1])
    left_label.set_text(rf"$\omega_{{{frame_idx}}}$")
    right_label.set_text(rf"$\omega_{{{frame_idx + 1}}}$")
    return left_img, right_img, left_label, right_label


ns_problem_frames = _render_rgba_frames(
    fig,
    _update_presentation_frame,
    PRESENTATION_FRAME_COUNT,
)

ns_problem_webp = _save_rgba_animation(
    ns_problem_frames,
    FIGURES_DIR / "ns_problem_rollout.webp",
    fps=PRESENTATION_FPS,
)
ns_problem_gif = _save_rgba_animation(
    ns_problem_frames,
    FIGURES_DIR / "ns_problem_rollout.gif",
    fps=PRESENTATION_FPS,
)
plt.show()
print(f"Saved presentation WebP to {ns_problem_webp}")
print(f"Saved fallback GIF to {ns_problem_gif}")


# %% Spatial-mean vorticity statistics across timesteps
# Left:  spatial mean of ω  — should hover near zero (conservation law).
# Right: spatial mean of |ω| — captures the growing magnitude of the field.
u_flat = u.reshape(N, H * W, T_FULL)

spatial_mean = u_flat.mean(axis=1)  # (N, T)
spatial_mag = np.abs(u_flat).mean(axis=1)  # (N, T)


def _band(arr):
    m = arr.mean(axis=0)
    p5 = np.percentile(arr, 5, axis=0)
    p95 = np.percentile(arr, 95, axis=0)
    return m, p5, p95


mean_t, mean_p5, mean_p95 = _band(spatial_mean)
mag_mean_t, mag_p5, mag_p95 = _band(spatial_mag)

t_idx = np.arange(1, T_FULL + 1)

c_mean = plt.get_cmap(CMAP)(0.3)
c_mag = plt.get_cmap(CMAP)(0.75)

fig, (ax_mean, ax_mag) = plt.subplots(1, 2, figsize=(12, 3.5))

# --- left: signed mean ---
ax_mean.fill_between(
    t_idx, mean_p5, mean_p95, alpha=0.25, color=c_mean, label="5–95th percentile"
)
ax_mean.plot(t_idx, mean_t, color=c_mean, lw=1.8, label="Mean")
ax_mean.axhline(0, color="0.4", lw=0.9, ls="--", label=r"$\bar\omega = 0$")
ax_mean.set_xlabel("Timestep")
ax_mean.set_ylabel(r"Spatial mean $\bar\omega$")
ax_mean.set_title("Signed mean vorticity")
ax_mean.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax_mean.legend(frameon=False)

# --- right: magnitude mean ---
ax_mag.fill_between(
    t_idx, mag_p5, mag_p95, alpha=0.25, color=c_mag, label="5–95th percentile"
)
ax_mag.plot(t_idx, mag_mean_t, color=c_mag, lw=1.8, label="Mean")
ax_mag.set_xlabel("Timestep")
ax_mag.set_ylabel(r"Spatial mean $|\bar\omega|$")
ax_mag.set_title("Vorticity magnitude")
ax_mag.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax_mag.legend(frameon=False)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "ns_mean_vorticity.png", bbox_inches="tight")
plt.show()
print(f"Saved to {FIGURES_DIR / 'ns_mean_vorticity.png'}")


# %% Training curves
TRAINING_MODELS = {
    # "Factformer": OUTPUTS_ROOT / "factformer/final/seed_42",
    "Transolver": OUTPUTS_ROOT / "transolver/final/seed_42",
    # "Galerkin-T": OUTPUTS_ROOT / "galerkin_transformer/validation/seed_42",
    # "ONO": OUTPUTS_ROOT / "ono/final/seed_42",
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
INFER_MODEL = "Transolver"
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
SHOW_STEPS = [3, 7, 11, 15, 19]

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


# %% Diagnosis — multi-model 3×N rollout grid (rel. L2 error)
# Rows: target | pred_1 | pred_2 | ... | rel_l2_1 | rel_l2_2 | ...
# Adjust DIAG_MODELS to any subset of keys from MODELS.
DIAG_MODELS = ["ONO"]


def load_ns_model(run_dir: Path) -> torch.nn.Module:
    cfg_m = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    cfg_m["paths"]["root_dir"] = str(DATA_PATH.parent)
    m, _, _ = create_model(
        cfg_m, device=DEVICE, runtime_overrides=_runtime_overrides(meta)
    )
    ckpt_m = load_checkpoint_state(run_dir / "best.pt", device=DEVICE)
    load_model_state_dict(m, ckpt_m["model_state_dict"])
    m.eval()
    return m


def rel_l2_field(pred_f: np.ndarray, gt_f: np.ndarray) -> np.ndarray:
    """Spatial rel-L2 map: |pred - gt| / ||gt||_F per timestep. Shape: (h, w, t_out)."""
    out = np.empty_like(pred_f)
    for t in range(pred_f.shape[2]):
        gt_t = gt_f[:, :, t]
        out[:, :, t] = np.abs(pred_f[:, :, t] - gt_t) / (np.linalg.norm(gt_t) + 1e-8)
    return out


diag_batch = next(iter(test_loader))
diag_preds = {}
for _name in DIAG_MODELS:
    _m = load_ns_model(MODELS[_name])
    _ts = _init_task_state(meta, sample_dtype=torch.float32, device=DEVICE)
    _coords, _fx, _tgt = _prepare_batch(diag_batch, device=DEVICE, task_state=_ts)
    with torch.no_grad():
        _pred, _, _ = rollout_autoregressive(
            _m,
            _coords,
            _fx,
            _tgt,
            t_out=meta["t_out"],
            out_dim=meta["out_dim"],
            teacher_forcing=False,
        )
    diag_preds[_name] = _pred[0].view(h, w, t_out).cpu().numpy()

diag_gt = _tgt[0].view(h, w, t_out).cpu().numpy()
diag_errors = {n: rel_l2_field(diag_preds[n], diag_gt) for n in DIAG_MODELS}

n_m = len(DIAG_MODELS)
n_rows = 1 + 2 * n_m
_cell = 1.5

vmin_d = min(diag_gt.min(), *(diag_preds[n].min() for n in DIAG_MODELS))
vmax_d = max(diag_gt.max(), *(diag_preds[n].max() for n in DIAG_MODELS))
err_max_d = max(diag_errors[n].max() for n in DIAG_MODELS)

fig, axes = plt.subplots(
    n_rows,
    t_out,
    figsize=(t_out * _cell + 1.5, n_rows * _cell + 0.5),
    constrained_layout=True,
)

for col in range(t_out):
    t_label = f"$t_{{{col + T_IN + 1}}}$"
    axes[0, col].imshow(
        diag_gt[:, :, col], cmap=CMAP, origin="lower", vmin=vmin_d, vmax=vmax_d
    )
    axes[0, col].set_title(t_label, fontsize=9)
    axes[0, col].axis("off")
    for i, name in enumerate(DIAG_MODELS):
        axes[1 + i, col].imshow(
            diag_preds[name][:, :, col],
            cmap=CMAP,
            origin="lower",
            vmin=vmin_d,
            vmax=vmax_d,
        )
        axes[1 + i, col].axis("off")
        axes[1 + n_m + i, col].imshow(
            diag_errors[name][:, :, col],
            cmap=CMAP,
            origin="lower",
            vmin=0,
            vmax=err_max_d,
        )
        axes[1 + n_m + i, col].axis("off")

axes[0, 0].set_ylabel("Ground truth", fontsize=9)
for i, name in enumerate(DIAG_MODELS):
    axes[1 + i, 0].set_ylabel(name, fontsize=9)
    axes[1 + n_m + i, 0].set_ylabel(f"{name} rel. $L_2$", fontsize=9)

fig.colorbar(
    plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin_d, vmax=vmax_d)),
    ax=axes[: 1 + n_m],
    fraction=0.015,
    pad=0.02,
    label="Vorticity",
)
fig.colorbar(
    plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=0, vmax=err_max_d)),
    ax=axes[1 + n_m :],
    fraction=0.015,
    pad=0.02,
    label=r"Rel. $L_2$",
)
fig.suptitle("NS diagnosis — test sample 0", fontsize=11)
fig.savefig(FIGURES_DIR / "ns_diag_rollout.png", bbox_inches="tight")
plt.show()


# %% Diagnosis — per-timestep rel. L2 over the full test set (multi-model)
def compute_step_rel_l2(m, loader, h, w, t_out, meta, device):
    """Returns (N_samples, t_out) per-sample per-timestep relative L2."""
    ts = _init_task_state(meta, sample_dtype=torch.float32, device=device)
    errs = []
    for batch in loader:
        coords_b, fx_b, target_b = _prepare_batch(batch, device=device, task_state=ts)
        with torch.no_grad():
            pred_b, _, _ = rollout_autoregressive(
                m,
                coords_b,
                fx_b,
                target_b,
                t_out=meta["t_out"],
                out_dim=meta["out_dim"],
                teacher_forcing=False,
            )
        bsz = pred_b.shape[0]
        p = pred_b.cpu().numpy().reshape(bsz, h, w, t_out)
        g = target_b.cpu().numpy().reshape(bsz, h, w, t_out)
        num = np.sqrt(((p - g) ** 2).sum(axis=(1, 2)))  # (bsz, t_out)
        den = np.sqrt((g**2).sum(axis=(1, 2))) + 1e-8  # (bsz, t_out)
        errs.append(num / den)
    return np.concatenate(errs, axis=0)  # (N_test, t_out)


step_idx = np.arange(1, t_out + 1)
curve_colors_d = plt.get_cmap(CMAP)(np.linspace(0.15, 0.85, len(DIAG_MODELS)))

fig, ax = plt.subplots(figsize=(7, 3.5))
for name, color in zip(DIAG_MODELS, curve_colors_d):
    step_errs = compute_step_rel_l2(
        load_ns_model(MODELS[name]), test_loader, h, w, t_out, meta, DEVICE
    )
    mean_e, p5_e, p95_e = _band(step_errs)
    ax.fill_between(step_idx, p5_e, p95_e, alpha=0.2, color=color)
    ax.plot(step_idx, mean_e, color=color, lw=1.8, label=name)

ax.set_xlabel("Output timestep")
ax.set_ylabel(r"Rel. $L_2$")
ax.set_title(r"NS per-timestep rel. $L_2$ error")
ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ns_diag_rel_l2_curve.png", bbox_inches="tight")
plt.show()
print(f"Saved diagnosis figures to {FIGURES_DIR}")
