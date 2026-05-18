# %% Imports & config
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
DATA_DIR = REPO_ROOT / "data/pipe"
FIGURES_DIR = REPO_ROOT / "docs/figures/pipe"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CMAP = "viridis"
CHANNELS = ("ux", "uy", "p")

EDGE_SLICES = {
    "inlet": (0, slice(None)),
    "outlet": (-1, slice(None)),
    "lower_wall": (slice(None), 0),
    "upper_wall": (slice(None), -1),
}
EDGE_COLORS = {
    "inlet": plt.get_cmap(CMAP)(0.15),
    "outlet": plt.get_cmap(CMAP)(0.85),
    "lower_wall": plt.get_cmap(CMAP)(0.40),
    "upper_wall": plt.get_cmap(CMAP)(0.65),
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
x_all = np.load(DATA_DIR / "Pipe_X.npy", mmap_mode="r")  # (N, H, W)
y_all = np.load(DATA_DIR / "Pipe_Y.npy", mmap_mode="r")  # (N, H, W)
q_all = np.load(DATA_DIR / "Pipe_Q.npy", mmap_mode="r")  # (N, C, H, W)

N, H, W = x_all.shape
C = q_all.shape[1]
print(f"Pipe_X: {x_all.shape}  Pipe_Y: {y_all.shape}  Pipe_Q: {q_all.shape}")
print(f"Channels: {CHANNELS[:C]}")


# %% Dataset sample — mesh boundaries and ux field
SAMPLE_IDX = 0

x = np.asarray(x_all[SAMPLE_IDX])  # (H, W)
y = np.asarray(y_all[SAMPLE_IDX])  # (H, W)
q = np.asarray(q_all[SAMPLE_IDX])  # (C, H, W)

MESH_STEP = 8

fig, (ax_mesh, ax_ux) = plt.subplots(1, 2, figsize=(12, 4))

# --- left: curvilinear mesh + boundary edges ---
for i in range(0, H, MESH_STEP):
    ax_mesh.plot(x[i, :], y[i, :], color="0.80", lw=0.5)
for j in range(0, W, MESH_STEP):
    ax_mesh.plot(x[:, j], y[:, j], color="0.80", lw=0.5)

for edge, sl in EDGE_SLICES.items():
    ax_mesh.plot(x[sl], y[sl], color=EDGE_COLORS[edge], lw=2.0, label=edge)

ax_mesh.set_aspect("equal", adjustable="box")
ax_mesh.set_xlabel("$x$")
ax_mesh.set_ylabel("$y$")
ax_mesh.set_title("Pipe mesh and boundary edges")
ax_mesh.legend(
    fontsize=8,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=4,
)

# --- right: ux field on curvilinear mesh ---
im = ax_ux.pcolormesh(x, y, q[0], shading="gouraud", cmap=CMAP)
fig.colorbar(im, ax=ax_ux, label="$u_x$", shrink=0.6)
ax_ux.set_aspect("equal", adjustable="box")
ax_ux.set_xlabel("$x$")
ax_ux.set_ylabel("$y$")
ax_ux.set_title("Streamwise velocity $u_x$")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "pipe_dataset_sample.png", bbox_inches="tight")
plt.show()
print(f"Saved to {FIGURES_DIR / 'pipe_dataset_sample.png'}")
