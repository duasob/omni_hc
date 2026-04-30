from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


SIGMA_FILE = "Random_UnitCell_sigma_10.npy"
XY_FILE = "Random_UnitCell_XY_10.npy"


class UnitTransformer:
    def __init__(self, x: torch.Tensor):
        self.mean = x.mean(dim=(0, 1), keepdim=True)
        self.std = x.std(dim=(0, 1), keepdim=True) + 1e-8

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def to(self, device: torch.device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class IdentityTransformer:
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def to(self, device: torch.device):
        return self


class ElasticityDataset(Dataset):
    def __init__(self, coords: torch.Tensor, y: torch.Tensor) -> None:
        self._coords = coords
        self._y = y

    def __len__(self) -> int:
        return int(self._y.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "coords": self._coords[idx],
            "x": self._coords[idx],
            "y": self._y[idx],
        }


def resolve_elasticity_files(root_dir: str | Path) -> tuple[Path, Path]:
    root = Path(root_dir)
    candidates = (
        root / "Meshes",
        root,
        root / "elasticity" / "Meshes",
    )
    for candidate in candidates:
        sigma_path = candidate / SIGMA_FILE
        xy_path = candidate / XY_FILE
        if sigma_path.exists() and xy_path.exists():
            return sigma_path, xy_path
    raise FileNotFoundError(
        "Expected elasticity files Random_UnitCell_sigma_10.npy and "
        "Random_UnitCell_XY_10.npy under root_dir, root_dir/Meshes, or "
        "root_dir/elasticity/Meshes."
    )


def _orient_elasticity_arrays(
    sigma_raw: np.ndarray,
    xy_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if xy_raw.ndim != 3:
        raise ValueError(f"Expected XY with 3 dimensions, got {xy_raw.shape}.")
    if xy_raw.shape[1] == 2:
        coords = np.transpose(xy_raw, (2, 0, 1))
    elif xy_raw.shape[2] == 2:
        coords = xy_raw
    elif xy_raw.shape[0] == 2:
        coords = np.transpose(xy_raw, (2, 1, 0))
    else:
        raise ValueError(
            "Expected XY in shape (P, 2, N), (N, P, 2), or (2, P, N); "
            f"got {xy_raw.shape}."
        )

    n_samples, n_points, _ = coords.shape
    if sigma_raw.ndim == 3 and sigma_raw.shape[-1] == 1:
        sigma_raw = sigma_raw[..., 0]
    if sigma_raw.ndim != 2:
        raise ValueError(f"Expected scalar sigma with 2 dimensions, got {sigma_raw.shape}.")
    if sigma_raw.shape == (n_points, n_samples):
        sigma = sigma_raw.T
    elif sigma_raw.shape == (n_samples, n_points):
        sigma = sigma_raw
    elif sigma_raw.shape[0] == n_samples:
        sigma = sigma_raw
    elif sigma_raw.shape[1] == n_samples:
        sigma = sigma_raw.T
    else:
        raise ValueError(
            "Could not align sigma with XY. Expected sigma to contain "
            f"{n_samples} samples and {n_points} points; got {sigma_raw.shape}."
        )
    if sigma.shape != (n_samples, n_points):
        raise ValueError(
            f"Aligned sigma shape {sigma.shape} does not match coords {coords.shape}."
        )
    return coords, sigma


def _as_int(value, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _load_elasticity_raw(cfg: dict):
    sigma_path, xy_path = resolve_elasticity_files(cfg["paths"]["root_dir"])
    sigma_raw = np.load(sigma_path, mmap_mode="r")
    xy_raw = np.load(xy_path, mmap_mode="r")
    coords, sigma = _orient_elasticity_arrays(sigma_raw, xy_raw)

    data_cfg = cfg.get("data", {})
    n_samples, n_points, _ = coords.shape
    ntrain = _as_int(data_cfg.get("ntrain"), 1000)
    ntest = _as_int(data_cfg.get("ntest"), 200)
    if ntrain <= 0 or ntest <= 0:
        raise ValueError(f"ntrain and ntest must be positive, got {ntrain} and {ntest}.")
    if ntrain + ntest > n_samples:
        raise ValueError(
            f"ntrain + ntest exceeds available samples: {ntrain} + {ntest} > {n_samples}."
        )

    x_train = torch.from_numpy(np.array(coords[:ntrain])).float()
    y_train = torch.from_numpy(np.array(sigma[:ntrain, :, None])).float()
    x_test = torch.from_numpy(np.array(coords[-ntest:])).float()
    y_test = torch.from_numpy(np.array(sigma[-ntest:, :, None])).float()

    meta = {
        "shapelist": (n_points,),
        "fun_dim": 0,
        "out_dim": 1,
        "space_dim": 2,
        "domain_bounds": (0.0, 1.0),
        "task": "steady",
        "loader": "elas",
        "geotype": "unstructured",
        "ntrain": ntrain,
        "ntest": ntest,
        "sigma_path": str(sigma_path),
        "xy_path": str(xy_path),
    }
    return x_train, y_train, x_test, y_test, meta


def _build_normalizers(cfg: dict, y_train: torch.Tensor):
    data_cfg = cfg.get("data", {})
    normalize = bool(data_cfg.get("normalize", True))
    if not normalize:
        return IdentityTransformer(), IdentityTransformer()
    norm_type = str(data_cfg.get("norm_type", "UnitTransformer"))
    if norm_type != "UnitTransformer":
        raise ValueError(
            "Elasticity loader currently supports only data.norm_type='UnitTransformer'."
        )
    return IdentityTransformer(), UnitTransformer(y_train)


def build_train_val_loaders(cfg: dict):
    x_train, y_train, _, _, meta = _load_elasticity_raw(cfg)
    x_normalizer, y_normalizer = _build_normalizers(cfg, y_train)
    y_train = y_normalizer.encode(y_train)

    dataset = ElasticityDataset(x_train, y_train)
    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 4))
    val_size = int(training_cfg.get("val_size", 16))
    val_size = min(max(val_size, 1), len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(int(training_cfg.get("seed", 42)))
    train_subset, val_subset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    train_loader.elasticity_meta = meta
    val_loader.elasticity_meta = meta
    train_loader.x_normalizer = x_normalizer
    val_loader.x_normalizer = x_normalizer
    train_loader.y_normalizer = y_normalizer
    val_loader.y_normalizer = y_normalizer
    return train_loader, val_loader


def build_test_loader(cfg: dict):
    _, y_train, x_test, y_test, meta = _load_elasticity_raw(cfg)
    x_normalizer, y_normalizer = _build_normalizers(cfg, y_train)
    y_test = y_normalizer.encode(y_test)

    dataset = ElasticityDataset(x_test, y_test)
    batch_size = int(
        cfg.get("evaluation", {}).get(
            "batch_size", cfg.get("training", {}).get("batch_size", 4)
        )
    )
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader.elasticity_meta = meta
    test_loader.x_normalizer = x_normalizer
    test_loader.y_normalizer = y_normalizer
    return test_loader
