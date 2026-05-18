# %% Imports & config
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath

REPO_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
DATA_DIR = REPO_ROOT / "data/airfoil/naca"
FIGURES_DIR = REPO_ROOT / "docs/figures/airfoil"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Physical constants
GAMMA = 1.4
M_INF = 0.8
A_INF = np.sqrt(GAMMA)      # ≈ 1.1832
U_INF = M_INF * A_INF       # ≈ 0.9466
RHO_INF = 1.0
P_INF = 1.0
H_INF = P_INF / ((GAMMA - 1.0) * RHO_INF) + 0.5 * U_INF**2  # ≈ 2.948

# Q channels: 0=ρ, 1=u, 2=v, 3=p, 4=M
CH_NAMES = [r"$\rho$", r"$u$", r"$v$", r"$p$", r"$M$"]
CH_FREESTREAM = [RHO_INF, U_INF, 0.0, P_INF, M_INF]
CH_CMAPS = ["RdBu_r", "RdBu_r", "RdBu_r", "RdBu_r", "RdBu_r"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

# %% Load cylinder-grid data (memory-mapped — avoids pulling 4 GB into RAM)
print("Loading NACA cylinder arrays (memory-mapped) ...")
CX = np.load(DATA_DIR / "NACA_Cylinder_X.npy", mmap_mode="r")  # (2490, 221, 51)
CY = np.load(DATA_DIR / "NACA_Cylinder_Y.npy", mmap_mode="r")
CQ = np.load(DATA_DIR / "NACA_Cylinder_Q.npy", mmap_mode="r")  # (2490, 5, 221, 51)
print(f"  CX: {CX.shape}  CY: {CY.shape}  CQ: {CQ.shape}")

N_SAMPLES, N_I, N_J = CX.shape   # 2490, 221, 51
# j=0  → airfoil wall (r ≈ 0)
# j=50 → far-field boundary (r ≈ 40)
# i=0..220 wraps circumferentially; row 220 closes back onto row 0

# %% Dataset sample — airfoil wall profiles
# j=0 includes both the airfoil surface (r < 2) and the TE wake slit (r up to 40).
# Mask to airfoil surface points so the aspect ratio stays sensible.
GEOM_SAMPLES = [0, 100, 500, 999]
colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(GEOM_SAMPLES)))

# Compute surface mask once from sample 0 — the slit i-indices are fixed for all samples
_wx0 = np.array(CX[0, :, 0])
_wy0 = np.array(CY[0, :, 0])
_r0  = np.sqrt(_wx0**2 + _wy0**2)
surf_mask = _r0 < 2.0   # True for airfoil surface, False for wake-slit extension

fig, ax = plt.subplots(figsize=(6, 3))
for idx, color in zip(GEOM_SAMPLES, colors):
    wx = np.array(CX[idx, :, 0])[surf_mask]
    wy = np.array(CY[idx, :, 0])[surf_mask]
    ax.fill(wx, wy, color=color, alpha=0.25)
    ax.plot(wx, wy, lw=1.0, color=color, label=f"sample {idx}")
ax.set_aspect("equal")
ax.set_xlim(-0.05, 1.1)
ax.set_ylim(-0.18, 0.18)
ax.set_xlabel("x / chord")
ax.set_ylabel("y / chord")
ax.set_title("Airfoil wall profiles (j = 0, surface points only)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "airfoil_geometries.png", bbox_inches="tight")
plt.show()
print(f"Saved → {FIGURES_DIR / 'airfoil_geometries.png'}")

# %% Mach field on the physical C-grid (near-airfoil physical view)
SAMPLE = 0
cx   = np.array(CX[SAMPLE])      # (221, 51) physical x-coords
cy   = np.array(CY[SAMPLE])      # (221, 51) physical y-coords
mach = np.array(CQ[SAMPLE, 4])   # (221, 51) Mach number
mach[-1, :] = np.nan              # suppress wake-slit wrap cell

wall_r = np.sqrt(cx[:, 0] ** 2 + cy[:, 0] ** 2)
surface = wall_r < 2.0

fig, ax = plt.subplots(figsize=(7.2, 3.8))
pc = ax.pcolormesh(
    cx, cy, mach,
    cmap="viridis", vmin=0.1, vmax=1.55,
    shading="gouraud", rasterized=True,
)
pc.set_clip_path(Circle((0.5, 0.0), 2.0, transform=ax.transData))

ax.fill(cx[surface, 0], cy[surface, 0], color="white", zorder=5)
ax.plot(cx[surface, 0], cy[surface, 0], color="0.35", lw=0.45, zorder=6)

ax.set_aspect("equal")
ax.set_xlim(-1.5, 2.5)
ax.set_ylim(-1.05, 1.05)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("0.55")
    spine.set_linewidth(0.7)
ax.tick_params(
    axis="both",
    which="both",
    labelbottom=False,
    labelleft=False,
    length=1.8,
    width=0.4,
    color="0.45",
)
fig.subplots_adjust(left=0.02, right=0.99, bottom=0.08, top=0.98)
fig.savefig(FIGURES_DIR / "airfoil_mach_physical.png", bbox_inches="tight", dpi=200)
plt.show()
print(f"Saved → {FIGURES_DIR / 'airfoil_mach_physical.png'}")

# %% All 5 flow channels on the structured C-grid (imshow)
# Axes: i (circumferential, horizontal) × j (radial, vertical)
# CQ[n, ch, :, :] is shape (221, 51); .T gives (51, 221) for imshow
fig, axes = plt.subplots(1, 5, figsize=(18, 3.2))
for ch_idx, (ax, name, cmap) in enumerate(zip(axes, CH_NAMES, CH_CMAPS)):
    field = np.array(CQ[SAMPLE, ch_idx]).T   # (51, 221)  — j on y-axis
    im = ax.imshow(field, origin="lower", aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(name, fontsize=13)
    ax.set_xlabel("i (circ.)", fontsize=8)
axes[0].set_ylabel("j (radial)")
axes[0].set_yticks([0, 25, 50])
axes[0].set_yticklabels(["0\n(wall)", "25", "50\n(far-field)"])
fig.suptitle(f"Flow field — sample {SAMPLE}  (structured C-grid, 221 × 51)", fontsize=11)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "airfoil_channels.png", bbox_inches="tight")
plt.show()
print(f"Saved → {FIGURES_DIR / 'airfoil_channels.png'}")

# %% Far-field decay function — the constraint backbone
# d(j) = 1 - j/(J-1): 1 at wall, 0 at far-field
j = np.arange(N_J)                   # 0, 1, ..., 50
d_linear = 1.0 - j / (N_J - 1)      # linear decay
d_quad   = d_linear ** 2             # power-law alternative

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(j, d_linear, lw=2, label=r"$d(j) = 1 - j/50$  (power=1)")
ax.plot(j, d_quad,   lw=2, ls="--", label=r"$d(j)^2$  (power=2)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("j  (radial index)")
ax.set_ylabel("Distance weight  d(j)")
ax.set_title("Far-field decay function")
ax.legend(frameon=False)

d_grid = np.tile(d_linear[None, :], (N_I, 1))   # (221, 51)
ax = axes[1]
im = ax.imshow(d_grid.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
fig.colorbar(im, ax=ax, label="d(j)")
ax.set_xlabel("i (circumferential)")
ax.set_ylabel("j (radial)")
ax.set_yticks([0, 25, 50])
ax.set_yticklabels(["0\n(wall)", "25", "50\n(far-field)"])
ax.set_title("Decay weight on C-grid")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "airfoil_far_field_decay.png", bbox_inches="tight")
plt.show()
print(f"Saved → {FIGURES_DIR / 'airfoil_far_field_decay.png'}")

# %% Far-field ansatz anatomy — decompose ground-truth Mach
# Ansatz:  M = M_inf + d(j) * N(x, y)
# => N(x, y) = (M - M_inf) / d(j)   [undefined at j=50 where d=0]
mach_gt = np.array(CQ[SAMPLE, 4])              # (221, 51)
d = 1.0 - j / (N_J - 1)                        # (51,)

# Broadcast d across i-axis
d_bcast = d[np.newaxis, :]                     # (1, 51)
safe_d = np.where(d_bcast > 1e-8, d_bcast, 1.0)
backbone_correction = (mach_gt - M_INF) / safe_d
backbone_correction[:, -1] = 0.0              # undefined at far-field; set to 0

mach_reconstructed = M_INF + d_bcast * backbone_correction

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
panels = [
    (mach_gt,              "RdBu_r",  "Ground truth  $M$"),
    (mach_reconstructed,  "RdBu_r",  r"$M_\infty + d(j)\cdot N$  (reconstruction)"),
    (backbone_correction, "coolwarm", r"Backbone correction  $N(x,y)$"),
]
for ax, (field, cmap, title) in zip(axes, panels):
    vabs = np.nanpercentile(np.abs(field), 98)
    im = ax.imshow(
        field.T, origin="lower", aspect="auto", cmap=cmap,
        vmin=-vabs, vmax=vabs,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("i (circumferential)")
    ax.set_ylabel("j (radial)")
    ax.set_yticks([0, 25, 50])
    ax.set_yticklabels(["0\n(wall)", "25", "50\n(ff)"])
    ax.set_title(title, fontsize=10)

fig.suptitle(
    r"Far-field ansatz: $M = M_\infty + d(j)\cdot N(x,y)$  —  sample "
    + str(SAMPLE),
    fontsize=12,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "airfoil_far_field_anatomy.png", bbox_inches="tight")
plt.show()
print(f"Saved → {FIGURES_DIR / 'airfoil_far_field_anatomy.png'}")

# %% Mach definition verification: M == |q| / sqrt(γ p/ρ)
rho = np.array(CQ[SAMPLE, 0])    # (221, 51)
u   = np.array(CQ[SAMPLE, 1])
v   = np.array(CQ[SAMPLE, 2])
p   = np.array(CQ[SAMPLE, 3])
M   = np.array(CQ[SAMPLE, 4])

a = np.sqrt(GAMMA * p / rho)
M_derived = np.sqrt(u**2 + v**2) / a
residual = np.abs(M - M_derived)

print(f"\nMach definition residual (sample {SAMPLE}):")
print(f"  max  = {residual.max():.2e}")
print(f"  mean = {residual.mean():.2e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
panels = [
    (M,         "RdBu_r", f"Stored  $M$  (CQ channel 4)"),
    (M_derived, "RdBu_r", r"Derived  $|q|/a$"),
    (residual,  "hot",    r"$|M_\mathrm{stored} - M_\mathrm{derived}|$"),
]
for ax, (field, cmap, title) in zip(axes, panels):
    im = ax.imshow(field.T, origin="lower", aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("i (circumferential)")
    ax.set_ylabel("j (radial)")
    ax.set_title(title, fontsize=10)

fig.suptitle(f"Mach definition check — sample {SAMPLE}", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "airfoil_mach_definition.png", bbox_inches="tight")
plt.show()
print(f"Saved → {FIGURES_DIR / 'airfoil_mach_definition.png'}")

# %% Far-field residuals across the dataset
# Check deviation from Q_inf at j=50 (upper j-edge) for all 2490 samples.
# CQ is channels-first: (N, 5, 221, 51) → far-field row = CQ[:, :, :, -1]
print("\nComputing far-field residuals over all samples ...")
N_CHECK = N_SAMPLES
far_field = np.array(CQ[:N_CHECK, :4, :, -1])   # (N, 4, 221) — primitive channels
Q_inf_vec = np.array([RHO_INF, U_INF, 0.0, P_INF])  # (4,)
far_residual = np.abs(far_field - Q_inf_vec[None, :, None])  # (N, 4, 221)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ch, (ax, name, q_inf) in enumerate(
    zip(axes, CH_NAMES[:4], CH_FREESTREAM[:4])
):
    vals = far_residual[:, ch, :].ravel()
    ax.hist(vals, bins=80, color=plt.get_cmap("viridis")(0.4), alpha=0.85)
    ax.set_xlabel(rf"$|{name} - {name}_\infty|$  at j=50", fontsize=9)
    ax.set_ylabel("count")
    ax.set_title(
        f"{name}  (Q_inf = {q_inf:.4f})\n"
        f"mean={vals.mean():.3e}  p99={np.percentile(vals, 99):.3e}",
        fontsize=8,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1e3 else f"{x:.0f}"))

fig.suptitle("Far-field residuals at j = 50 (r ≈ 40) across all samples", fontsize=11)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "airfoil_far_field_residuals.png", bbox_inches="tight")
plt.show()
print(f"Saved → {FIGURES_DIR / 'airfoil_far_field_residuals.png'}")

# %% Positivity check — ρ and p across the dataset
print("\nChecking positivity of ρ and p ...")
rho_all = np.array(CQ[:, 0, :, :])   # (N, 221, 51)
p_all   = np.array(CQ[:, 3, :, :])

print(f"  ρ  min={rho_all.min():.4f}  max={rho_all.max():.4f}  negatives={int((rho_all <= 0).sum())}")
print(f"  p  min={p_all.min():.4f}   max={p_all.max():.4f}   negatives={int((p_all <= 0).sum())}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, vals) in zip(axes, [(r"$\rho$", rho_all.ravel()), (r"$p$", p_all.ravel())]):
    ax.hist(vals, bins=100, color=plt.get_cmap("viridis")(0.55), alpha=0.85)
    ax.axvline(0, color="r", lw=1.0, ls="--", label="zero")
    ax.set_xlabel(name)
    ax.set_ylabel("count")
    ax.set_title(
        f"{name}  min={vals.min():.3f}  max={vals.max():.3f}",
        fontsize=9,
    )
    ax.legend(frameon=False, fontsize=8)

fig.suptitle(
    r"Positivity of $\rho$ and $p$ — all 2490 samples (softplus enforces this for free)",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "airfoil_positivity.png", bbox_inches="tight")
plt.show()
print(f"Saved → {FIGURES_DIR / 'airfoil_positivity.png'}")

# %% Paper-style helpers
from matplotlib.patches import Circle

# Shared geometry: surface mask (wall points on the airfoil, not the wake slit)
_r0 = np.sqrt(np.array(CX[0, :, 0]) ** 2 + np.array(CY[0, :, 0]) ** 2)
SURF_MASK = _r0 < 2.0

# Colormap norm: blue → white at M∞ → red (matches the benchmark papers)
MACH_NORM = TwoSlopeNorm(vmin=0.0, vcenter=M_INF, vmax=1.7)
ERR_NORM  = TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25)

# Fan clip: circle centred at chord midpoint (0.5, 0), radius = 2 chord lengths.
# set_clip_path clips the pcolormesh collection directly — no layering tricks needed.
FAN_CX, FAN_CY = 0.5, 0.0
R_SHOW = 2.0
PAD    = 0.12


def _fan_axes(ax):
    ax.set_xlim(FAN_CX - R_SHOW - PAD, FAN_CX + R_SHOW + PAD)
    ax.set_ylim(FAN_CY - R_SHOW - PAD, FAN_CY + R_SHOW + PAD)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_fan(ax, sample_idx, field, norm, cmap):
    """Core render: pcolormesh + circular clip + airfoil fill."""
    data = np.array(field, dtype=float)
    data[-1, :] = np.nan                  # suppress wake-slit wrap cell

    pc = ax.pcolormesh(
        np.array(CX[sample_idx]),
        np.array(CY[sample_idx]),
        data,
        norm=norm,
        cmap=cmap,
        shading="gouraud",
        rasterized=True,
    )

    # Clip pcolormesh to circle in data coordinates
    clip_circle = Circle((FAN_CX, FAN_CY), R_SHOW, transform=ax.transData)
    pc.set_clip_path(clip_circle)

    # Thin arc border showing the clip boundary
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(
        FAN_CX + R_SHOW * np.cos(theta),
        FAN_CY + R_SHOW * np.sin(theta),
        color="0.65", lw=0.5, zorder=4,
    )

    # White airfoil body
    sx = np.array(CX[sample_idx, SURF_MASK, 0])
    sy = np.array(CY[sample_idx, SURF_MASK, 0])
    ax.fill(sx, sy, color="white", zorder=5)
    ax.plot(sx, sy, color="0.25", lw=0.5, zorder=6)

    _fan_axes(ax)
    return pc


def plot_mach_fan(ax, sample_idx, *, norm=MACH_NORM, cmap="RdBu_r", title=None):
    """Paper-style Mach field panel."""
    pc = _draw_fan(ax, sample_idx, CQ[sample_idx, 4], norm, cmap)
    if title:
        ax.set_title(title, fontsize=9, pad=3)
    return pc


def plot_error_fan(ax, sample_idx, pred_field, *, norm=ERR_NORM, cmap="RdBu_r", title=None):
    """Paper-style error map panel (pred − GT)."""
    gt  = np.array(CQ[sample_idx, 4], dtype=float)
    err = pred_field.astype(float) - gt
    pc  = _draw_fan(ax, sample_idx, err, norm, cmap)
    if title:
        ax.set_title(title, fontsize=9, pad=3)
    return pc


# %% Paper-style: single ground-truth sample
SAMPLE = 0
fig, ax = plt.subplots(figsize=(4.5, 4.5))
pc = plot_mach_fan(ax, SAMPLE, title=f"Ground truth — sample {SAMPLE}")
cbar = fig.colorbar(pc, ax=ax, fraction=0.03, pad=0.01, extend="neither")
cbar.set_label("Mach", fontsize=8)
cbar.set_ticks([0.0, 0.4, 0.8, 1.2, 1.6])
fig.tight_layout()
fig.savefig(FIGURES_DIR / "airfoil_paper_single.png", bbox_inches="tight", dpi=200)
plt.show()
print(f"Saved → {FIGURES_DIR / 'airfoil_paper_single.png'}")

# %% Paper-style: four ground-truth samples side-by-side (like the GT column in papers)
SAMPLES = [0, 100, 500, 999]
fig, axes = plt.subplots(1, len(SAMPLES), figsize=(4 * len(SAMPLES), 4.2))
for ax, s in zip(axes, SAMPLES):
    pc = plot_mach_fan(ax, s, title=f"sample {s}")
# Shared colorbar on the right
cbar = fig.colorbar(pc, ax=axes.tolist(), fraction=0.015, pad=0.01, extend="neither")
cbar.set_label("Mach", fontsize=9)
cbar.set_ticks([0.0, 0.4, 0.8, 1.2, 1.6])
fig.suptitle("Mach field — ground truth", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "airfoil_paper_gt_row.png", bbox_inches="tight", dpi=200)
plt.show()
print(f"Saved → {FIGURES_DIR / 'airfoil_paper_gt_row.png'}")

# %% Paper-style: GT + error map layout (2 rows × N cols)
# Since we have no model predictions yet, row 2 shows the difference between
# a nearby sample and sample 0 — purely to validate the error-map rendering.
# Replace `pseudo_pred` with actual model output once training is done.
REF = 0
PRED_SAMPLES = [1, 2, 3]   # stand-in "predictions" — swap with real model outputs

n_cols = 1 + len(PRED_SAMPLES)
fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols, 9))

# Row 0: Mach fields
pc_m = plot_mach_fan(axes[0, 0], REF, title="Ground truth")
for col, s in enumerate(PRED_SAMPLES, start=1):
    plot_mach_fan(axes[0, col], s, title=f"Sample {s} (stand-in pred)")

# Row 1: error maps (each stand-in sample − sample 0)
axes[1, 0].axis("off")   # no error for GT itself
axes[1, 0].text(0.5, 0.5, "Error maps", ha="center", va="center",
                transform=axes[1, 0].transAxes, fontsize=10, color="0.4")
for col, s in enumerate(PRED_SAMPLES, start=1):
    pseudo_pred = np.array(CQ[s, 4])
    pc_e = plot_error_fan(axes[1, col], REF, pseudo_pred, title=f"Δ sample {s} − {REF}")

# Colorbars
fig.colorbar(pc_m, ax=axes[0].tolist(), fraction=0.012, pad=0.01,
             label="Mach").set_ticks([0.0, 0.4, 0.8, 1.2, 1.6])
fig.colorbar(pc_e, ax=axes[1, 1:].tolist(), fraction=0.015, pad=0.01,
             label="ΔMach").set_ticks([-0.2, -0.1, 0.0, 0.1, 0.2])
fig.tight_layout()
fig.savefig(FIGURES_DIR / "airfoil_paper_grid.png", bbox_inches="tight", dpi=200)
plt.show()
print(f"Saved → {FIGURES_DIR / 'airfoil_paper_grid.png'}")
