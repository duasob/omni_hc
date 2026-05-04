from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader, Dataset, random_split


PLASTICITY_FILE = "plas_N987_T20.mat"


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


class PlasticityDataset(Dataset):
    def __init__(
        self,
        coords: torch.Tensor,
        time: torch.Tensor,
        fx: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        self._coords = coords
        self._time = time
        self._fx = fx
        self._y = y

    def __len__(self) -> int:
        return int(self._fx.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "coords": self._coords[idx],
            "time": self._time[idx],
            "x": self._fx[idx],
            "y": self._y[idx],
        }


def resolve_plasticity_mat_file(root_dir: str | Path) -> Path:
    root = Path(root_dir)
    if root.is_file() and root.suffix == ".mat":
        return root
    candidates = (
        root / PLASTICITY_FILE,
        root / "plasticity" / PLASTICITY_FILE,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Expected {PLASTICITY_FILE} under root_dir or root_dir/plasticity."
    )


def make_grid(h: int, w: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    xs = torch.linspace(0, 1, w, device=device, dtype=dtype)
    ys = torch.linspace(0, 1, h, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    return grid.reshape(-1, 2)


def _as_int(value, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _load_plasticity_mat(path: Path):
    raw = scio.loadmat(str(path), variable_names=("input", "output"))
    if "input" not in raw or "output" not in raw:
        raise KeyError(f"Expected keys 'input' and 'output' in {path}")
    input_raw = raw["input"]
    output_raw = raw["output"]
    if input_raw.ndim != 2:
        raise ValueError(f"Expected input with shape (N, X), got {input_raw.shape}")
    if output_raw.ndim != 5:
        raise ValueError(
            f"Expected output with shape (N, X, Y, T, C), got {output_raw.shape}"
        )
    return input_raw, output_raw


def _load_plasticity_input(path: Path):
    raw = scio.loadmat(str(path), variable_names=("input",))
    if "input" not in raw:
        raise KeyError(f"Expected key 'input' in {path}")
    input_raw = raw["input"]
    if input_raw.ndim != 2:
        raise ValueError(f"Expected input with shape (N, X), got {input_raw.shape}")
    return input_raw


def _prepare_fx(input_raw, *, sample_slice, n_split: int, r1: int, s1: int, s2: int):
    die = input_raw[sample_slice, ::r1][:, :s1]
    fx = die.reshape(n_split, s1, 1).repeat(s2, axis=2).reshape(n_split, -1, 1)
    return torch.from_numpy(np.array(fx)).float()


def _load_plasticity_raw(cfg: dict, *, split: str):
    mat_path = resolve_plasticity_mat_file(cfg["paths"]["root_dir"])
    input_raw, output_raw = _load_plasticity_mat(mat_path)
    data_cfg = cfg.get("data", {}) or {}
    model_args = cfg.get("model", {}).get("args", {}) or {}

    n_samples, x_full = input_raw.shape
    out_samples, out_x, out_y, t_full, channel_full = output_raw.shape
    if out_samples != n_samples or out_x != x_full:
        raise ValueError(
            "Plasticity input/output sample or X dimensions do not match: "
            f"input={input_raw.shape}, output={output_raw.shape}"
        )

    r1 = _as_int(data_cfg.get("downsamplex", model_args.get("downsamplex")), 1)
    r2 = _as_int(data_cfg.get("downsampley", model_args.get("downsampley")), 1)
    t_out = _as_int(data_cfg.get("T_out", model_args.get("T_out")), 20)
    out_dim = _as_int(data_cfg.get("out_dim", model_args.get("out_dim")), 4)
    ntrain = _as_int(data_cfg.get("ntrain"), 900)
    ntest = _as_int(data_cfg.get("ntest"), 80)

    if r1 <= 0 or r2 <= 0:
        raise ValueError(f"downsample factors must be positive, got {r1}, {r2}.")
    if t_out > t_full:
        raise ValueError(f"T_out exceeds available time steps: {t_out} > {t_full}.")
    if out_dim > channel_full:
        raise ValueError(f"out_dim exceeds output channels: {out_dim} > {channel_full}.")
    if ntrain <= 1 or ntest <= 0:
        raise ValueError(f"ntrain and ntest must be positive, got {ntrain}, {ntest}.")
    if ntrain + ntest > n_samples:
        raise ValueError(
            f"ntrain + ntest exceeds available samples: {ntrain} + {ntest} > {n_samples}."
        )

    s1 = int(((x_full - 1) / r1) + 1)
    s2 = int(((out_y - 1) / r2) + 1)
    sample_slice = slice(0, ntrain) if split == "train" else slice(-ntest, None)
    n_split = ntrain if split == "train" else ntest
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")

    fx_tensor = _prepare_fx(
        input_raw,
        sample_slice=sample_slice,
        n_split=n_split,
        r1=r1,
        s1=s1,
        s2=s2,
    )
    y = output_raw[
        sample_slice,
        ::r1,
        ::r2,
        :t_out,
        :out_dim,
    ][:, :s1, :s2]
    y = y.reshape(n_split, -1, t_out * out_dim)

    coords = make_grid(s1, s2).unsqueeze(0).repeat(n_split, 1, 1)
    time = torch.linspace(0, 1, t_out).unsqueeze(0).repeat(n_split, 1)

    y_tensor = torch.from_numpy(np.array(y)).float()
    meta = {
        "shapelist": (int(s1), int(s2)),
        "t_out": int(t_out),
        "out_dim": int(out_dim),
        "fun_dim": 1,
        "space_dim": 2,
        "domain_bounds": (0.0, 1.0),
        "task": "dynamic_conditional",
        "loader": "plas",
        "geotype": "structured_2D",
        "ntrain": int(ntrain),
        "ntest": int(ntest),
        "path": str(mat_path),
        "input_shape": tuple(int(v) for v in input_raw.shape),
        "output_shape": tuple(int(v) for v in output_raw.shape),
        "channel_semantics": str(
            data_cfg.get(
                "channel_semantics",
                "unknown; NSL visualizes final channels 0:2 as coordinates and 2:4 as a vector magnitude",
            )
        ),
    }
    return coords, time.float(), fx_tensor, y_tensor, meta


def _build_normalizers(cfg: dict, fx_train: torch.Tensor, y_train: torch.Tensor):
    data_cfg = cfg.get("data", {}) or {}
    normalize_input = bool(data_cfg.get("normalize_input", True))
    normalize_target = bool(data_cfg.get("normalize", False))
    if normalize_input:
        x_normalizer = UnitTransformer(fx_train)
    else:
        x_normalizer = IdentityTransformer()
    if normalize_target:
        y_normalizer = UnitTransformer(y_train)
    else:
        y_normalizer = IdentityTransformer()
    return x_normalizer, y_normalizer


def build_train_val_loaders(cfg: dict):
    coords, time, fx_train, y_train, meta = _load_plasticity_raw(cfg, split="train")
    x_normalizer, y_normalizer = _build_normalizers(cfg, fx_train, y_train)
    fx_train = x_normalizer.encode(fx_train)
    y_train = y_normalizer.encode(y_train)

    dataset = PlasticityDataset(coords, time, fx_train, y_train)
    training_cfg = cfg.get("training", {}) or {}
    batch_size = int(training_cfg.get("batch_size", 8))
    val_size = int(training_cfg.get("val_size", 100))
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
    train_loader.plasticity_meta = meta
    val_loader.plasticity_meta = meta
    train_loader.x_normalizer = x_normalizer
    val_loader.x_normalizer = x_normalizer
    train_loader.y_normalizer = y_normalizer
    val_loader.y_normalizer = y_normalizer
    return train_loader, val_loader


def build_test_loader(cfg: dict):
    coords, time, fx_test, y_test, meta = _load_plasticity_raw(cfg, split="test")
    mat_path = resolve_plasticity_mat_file(cfg["paths"]["root_dir"])
    input_raw = _load_plasticity_input(mat_path)
    fx_train = _prepare_fx(
        input_raw,
        sample_slice=slice(0, int(meta["ntrain"])),
        n_split=int(meta["ntrain"]),
        r1=int(cfg.get("data", {}).get("downsamplex", 1)),
        s1=int(meta["shapelist"][0]),
        s2=int(meta["shapelist"][1]),
    )
    y_train = y_test if not bool(cfg.get("data", {}).get("normalize", False)) else None
    if y_train is None:
        _, _, _, y_train, _ = _load_plasticity_raw(cfg, split="train")
    x_normalizer, y_normalizer = _build_normalizers(cfg, fx_train, y_train)
    fx_test = x_normalizer.encode(fx_test)
    y_test = y_normalizer.encode(y_test)

    dataset = PlasticityDataset(coords, time, fx_test, y_test)
    batch_size = int(
        cfg.get("evaluation", {}).get(
            "batch_size", cfg.get("training", {}).get("batch_size", 8)
        )
    )
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader.plasticity_meta = meta
    test_loader.x_normalizer = x_normalizer
    test_loader.y_normalizer = y_normalizer
    return test_loader
