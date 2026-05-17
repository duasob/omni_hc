from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


GAMMA = 1.4
# Reference freestream: rho_inf=1, p_inf=1, M_inf=0.8, AoA=0
# => a_inf = sqrt(gamma) ≈ 1.1832, u_inf = M_inf * a_inf ≈ 0.9466
A_INF = float(np.sqrt(GAMMA))
M_INF = 0.8
U_INF = M_INF * A_INF  # ≈ 0.9466
# Total enthalpy at freestream: H_inf = p/(rho*(gamma-1)) + 0.5*u^2
H_INF = 1.0 / ((GAMMA - 1) * 1.0) + 0.5 * U_INF**2  # ≈ 2.948


def _naca_dir(data_dir: Path) -> tuple[Path, Path]:
    for root in [data_dir / "naca", data_dir]:
        if (root / "NACA_X.npy").exists():
            return root, data_dir / "naca_interp"
    raise FileNotFoundError(
        f"Missing NACA_X.npy under {data_dir}/naca or {data_dir}"
    )


def load_naca_arrays(
    data_dir: Path,
    *,
    mmap_mode: str | None = "r",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load unstructured C-grid data.

    Returns:
        X:     (N, 11220) x-coordinates of unstructured points
        Y:     (N, 11220) y-coordinates
        Q:     (N, 5, 11220) flow quantities [rho, u, v, p, Mach]
        theta: (N, 8) airfoil shape parameters (col 0 is always 0)
        res:   (N,) solver residuals
    """
    naca_dir, _ = _naca_dir(data_dir)
    X = np.load(naca_dir / "NACA_X.npy", mmap_mode=mmap_mode)
    Y = np.load(naca_dir / "NACA_Y.npy", mmap_mode=mmap_mode)
    Q = np.load(naca_dir / "NACA_Q.npy", mmap_mode=mmap_mode)
    theta = np.load(naca_dir / "NACA_theta.npy", mmap_mode=mmap_mode)
    res = np.load(naca_dir / "NACA_res.npy", mmap_mode=mmap_mode)
    return X, Y, Q, theta, res


def load_naca_cylinder_arrays(
    data_dir: Path,
    *,
    mmap_mode: str | None = "r",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load structured C-grid in cylinder (computational) coordinates.

    Returns:
        CX: (N, 221, 51) x-coords on cylinder grid
        CY: (N, 221, 51) y-coords
        CQ: (N, 5, 221, 51) flow quantities [rho, u, v, p, Mach]
    Wall (airfoil surface) is j=0; far-field is j=50.
    """
    naca_dir, _ = _naca_dir(data_dir)
    CX = np.load(naca_dir / "NACA_Cylinder_X.npy", mmap_mode=mmap_mode)
    CY = np.load(naca_dir / "NACA_Cylinder_Y.npy", mmap_mode=mmap_mode)
    CQ = np.load(naca_dir / "NACA_Cylinder_Q.npy", mmap_mode=mmap_mode)
    return CX, CY, CQ


def load_naca_interp_arrays(
    data_dir: Path,
    *,
    mmap_mode: str | None = "r",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load regular-grid interpolated data (1200 samples, subset of naca/).

    Returns:
        Xi:   (N, 101, 101) x-coordinates, range [-0.5, 1.5]
        Yi:   (N, 101, 101) y-coordinates, range [-1, 1]
        Qi:   (N, 101, 101) Mach number; exactly 0 inside airfoil mask
        mask: (N, 101, 101) bool, True=fluid, False=inside airfoil
    """
    _, interp_dir = _naca_dir(data_dir)
    if not interp_dir.exists():
        raise FileNotFoundError(f"Missing naca_interp directory: {interp_dir}")
    Xi = np.load(interp_dir / "NACA_X_interp.npy", mmap_mode=mmap_mode)
    Yi = np.load(interp_dir / "NACA_Y_interp.npy", mmap_mode=mmap_mode)
    Qi = np.load(interp_dir / "NACA_Q_interp.npy", mmap_mode=mmap_mode)
    mask = np.load(interp_dir / "NACA_mask_interp.npy", mmap_mode=mmap_mode)
    return Xi, Yi, Qi, mask


def select_split(
    *arrays: np.ndarray,
    split: str,
    ntrain: int,
    ntest: int,
) -> tuple[np.ndarray, ...]:
    n_samples = int(arrays[0].shape[0])
    if split == "all":
        return arrays
    if split == "train":
        return tuple(a[:ntrain] for a in arrays)
    if split == "test":
        return tuple(a[-ntest:] for a in arrays)
    raise ValueError(f"split must be 'all', 'train', or 'test', got {split!r}")


def wall_normal_velocity(
    CX: np.ndarray,
    CY: np.ndarray,
    CQ: np.ndarray,
) -> np.ndarray:
    """Compute normal velocity at the airfoil wall (j=0) for each cylinder-grid sample.

    Normal direction estimated from the along-wall tangent via finite differences.

    Returns:
        V_n: (N, 221) signed normal velocity at j=0 (should be ~0 for no-penetration)
    """
    wx = CX[:, :, 0].astype(np.float64)  # (N, 221)
    wy = CY[:, :, 0].astype(np.float64)
    wu = CQ[:, 1, :, 0].astype(np.float64)
    wv = CQ[:, 2, :, 0].astype(np.float64)

    dx_ds = np.gradient(wx, axis=1)
    dy_ds = np.gradient(wy, axis=1)
    norm = np.sqrt(dx_ds**2 + dy_ds**2) + 1e-20
    # Outward normal from wall into fluid: rotate tangent 90° outward
    nx = -dy_ds / norm
    ny = dx_ds / norm
    return wu * nx + wv * ny


def write_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def require_matplotlib(plt):
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plots. Install matplotlib or rerun without --no-plot."
        )
