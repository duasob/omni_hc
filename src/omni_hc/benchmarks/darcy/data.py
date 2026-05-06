from __future__ import annotations

from pathlib import Path

import scipy.io as scio
import torch
from torch.utils.data import DataLoader, Dataset, random_split


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


class DarcyDataset(Dataset):
    def __init__(
        self, coords: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> None:
        self._coords = coords
        self._x = x
        self._y = y

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "coords": self._coords[idx],
            "x": self._x[idx],
            "y": self._y[idx],
        }


def resolve_darcy_mat_files(root_dir: str | Path) -> tuple[Path, Path]:
    root = Path(root_dir)
    if root.is_dir():
        train_path = root / "piececonst_r421_N1024_smooth1.mat"
        test_path = root / "piececonst_r421_N1024_smooth2.mat"
        if train_path.exists() and test_path.exists():
            return train_path, test_path
    raise FileNotFoundError(
        "Expected Darcy directory containing 'piececonst_r421_N1024_smooth1.mat' "
        "and 'piececonst_r421_N1024_smooth2.mat'."
    )


def _as_int(value, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _grid(h: int, w: int) -> torch.Tensor:
    ys = torch.linspace(0, 1, h)
    xs = torch.linspace(0, 1, w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).view(-1, 2)


def _load_darcy_mat(path: Path) -> dict:
    raw = scio.loadmat(str(path), variable_names=("coeff", "sol"))
    if "coeff" not in raw or "sol" not in raw:
        raise KeyError(f"Expected 'coeff' and 'sol' in {path}")
    return raw


def _darcy_sampling(cfg: dict, coeff_shape: tuple[int, ...]):
    data_cfg = cfg.get("data", {})
    model_args = cfg.get("model", {}).get("args", {})
    h_full, w_full = int(coeff_shape[1]), int(coeff_shape[2])
    r1 = _as_int(data_cfg.get("downsamplex", model_args.get("downsamplex")), 1)
    r2 = _as_int(data_cfg.get("downsampley", model_args.get("downsampley")), 1)
    s1 = int(((h_full - 1) / r1) + 1)
    s2 = int(((w_full - 1) / r2) + 1)
    return r1, r2, s1, s2


def _slice_field(raw: dict, key: str, n: int, r1: int, r2: int, s1: int, s2: int):
    return raw[key][:n, ::r1, ::r2][:, :s1, :s2]


def _load_darcy_train_raw(cfg: dict):
    train_path, test_path = resolve_darcy_mat_files(cfg["paths"]["root_dir"])
    train_raw = _load_darcy_mat(train_path)

    data_cfg = cfg.get("data", {})
    r1, r2, s1, s2 = _darcy_sampling(cfg, train_raw["coeff"].shape)
    ntrain = _as_int(data_cfg.get("ntrain"), train_raw["coeff"].shape[0])

    x_train = _slice_field(train_raw, "coeff", ntrain, r1, r2, s1, s2)
    y_train = _slice_field(train_raw, "sol", ntrain, r1, r2, s1, s2)

    x_train = torch.from_numpy(x_train.reshape(ntrain, -1, 1)).float()
    y_train = torch.from_numpy(y_train.reshape(ntrain, -1, 1)).float()

    meta = {
        "shapelist": (s1, s2),
        "fun_dim": 1,
        "out_dim": 1,
        "space_dim": 2,
        "domain_bounds": (0.0, 1.0),
        "task": "steady",
        "loader": "darcy",
        "geotype": "structured_2D",
        "ntrain": ntrain,
        "ntest": _as_int(data_cfg.get("ntest"), 200),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    return x_train, y_train, meta


def _load_darcy_raw(cfg: dict):
    train_path, test_path = resolve_darcy_mat_files(cfg["paths"]["root_dir"])
    x_train, y_train, meta = _load_darcy_train_raw(cfg)
    test_raw = _load_darcy_mat(test_path)

    data_cfg = cfg.get("data", {})
    r1, r2, s1, s2 = _darcy_sampling(cfg, test_raw["coeff"].shape)
    ntest = _as_int(data_cfg.get("ntest"), test_raw["coeff"].shape[0])
    x_test = _slice_field(test_raw, "coeff", ntest, r1, r2, s1, s2)
    y_test = _slice_field(test_raw, "sol", ntest, r1, r2, s1, s2)

    x_test = torch.from_numpy(x_test.reshape(ntest, -1, 1)).float()
    y_test = torch.from_numpy(y_test.reshape(ntest, -1, 1)).float()

    meta["ntest"] = ntest
    meta["train_path"] = str(train_path)
    meta["test_path"] = str(test_path)
    return x_train, y_train, x_test, y_test, meta


def _build_normalizers(cfg: dict, x_train: torch.Tensor, y_train: torch.Tensor):
    data_cfg = cfg.get("data", {})
    normalize = bool(data_cfg.get("normalize", True))
    if not normalize:
        return IdentityTransformer(), IdentityTransformer()
    norm_type = str(data_cfg.get("norm_type", "UnitTransformer"))
    if norm_type != "UnitTransformer":
        raise ValueError(
            "Darcy smoke loader currently supports only data.norm_type='UnitTransformer'."
        )
    return UnitTransformer(x_train), UnitTransformer(y_train)


def build_train_val_loaders(cfg: dict):
    x_train, y_train, meta = _load_darcy_train_raw(cfg)
    x_normalizer, y_normalizer = _build_normalizers(cfg, x_train, y_train)
    x_train = x_normalizer.encode(x_train)
    y_train = y_normalizer.encode(y_train)

    coords = _grid(*meta["shapelist"]).unsqueeze(0).repeat(int(x_train.shape[0]), 1, 1)
    dataset = DarcyDataset(coords, x_train, y_train)

    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 4))
    val_size = int(training_cfg.get("val_size", 100))
    if val_size < 0:
        raise ValueError(f"Invalid val_size={val_size}")
    if val_size == 0:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        train_loader.darcy_meta = meta
        train_loader.x_normalizer = x_normalizer
        train_loader.y_normalizer = y_normalizer
        return train_loader, None

    val_size = min(max(val_size, 1), len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(int(training_cfg.get("seed", 42)))
    train_subset, val_subset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    train_loader.darcy_meta = meta
    val_loader.darcy_meta = meta
    train_loader.x_normalizer = x_normalizer
    val_loader.x_normalizer = x_normalizer
    train_loader.y_normalizer = y_normalizer
    val_loader.y_normalizer = y_normalizer
    return train_loader, val_loader


def build_test_loader(cfg: dict):
    x_train, y_train, x_test, y_test, meta = _load_darcy_raw(cfg)
    x_normalizer, y_normalizer = _build_normalizers(cfg, x_train, y_train)
    x_test = x_normalizer.encode(x_test)
    y_test = y_normalizer.encode(y_test)

    coords = _grid(*meta["shapelist"]).unsqueeze(0).repeat(int(x_test.shape[0]), 1, 1)
    dataset = DarcyDataset(coords, x_test, y_test)
    batch_size = int(
        cfg.get("evaluation", {}).get(
            "batch_size", cfg.get("training", {}).get("batch_size", 4)
        )
    )
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader.darcy_meta = meta
    test_loader.x_normalizer = x_normalizer
    test_loader.y_normalizer = y_normalizer
    return test_loader
