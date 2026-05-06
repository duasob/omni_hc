from __future__ import annotations

from pathlib import Path

import numpy as np
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


class PipeDataset(Dataset):
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


def resolve_pipe_dir(root_dir: str | Path) -> Path:
    root = Path(root_dir)
    required = ("Pipe_X.npy", "Pipe_Y.npy", "Pipe_Q.npy")
    if root.is_dir() and all((root / name).exists() for name in required):
        return root
    raise FileNotFoundError(
        "Expected a pipe data directory containing Pipe_X.npy, Pipe_Y.npy, and Pipe_Q.npy."
    )


def _as_int(value, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _load_pipe_raw(cfg: dict):
    root = resolve_pipe_dir(cfg["paths"]["root_dir"])
    input_x = np.load(root / "Pipe_X.npy", mmap_mode="r")
    input_y = np.load(root / "Pipe_Y.npy", mmap_mode="r")
    output_q = np.load(root / "Pipe_Q.npy", mmap_mode="r")

    if input_x.shape != input_y.shape:
        raise ValueError(
            f"Pipe_X and Pipe_Y shapes must match, got {input_x.shape} and {input_y.shape}."
        )
    if output_q.ndim != 4:
        raise ValueError(f"Expected Pipe_Q with shape (N,C,H,W), got {output_q.shape}.")

    data_cfg = cfg.get("data", {})
    model_args = cfg.get("model", {}).get("args", {})
    n_samples, h_full, w_full = input_x.shape
    r1 = _as_int(data_cfg.get("downsamplex", model_args.get("downsamplex")), 1)
    r2 = _as_int(data_cfg.get("downsampley", model_args.get("downsampley")), 1)
    s1 = int(((h_full - 1) / r1) + 1)
    s2 = int(((w_full - 1) / r2) + 1)
    ntrain = _as_int(data_cfg.get("ntrain"), 1000)
    ntest = _as_int(data_cfg.get("ntest"), 200)
    target_channel = _as_int(data_cfg.get("target_channel"), 0)

    if target_channel < 0 or target_channel >= int(output_q.shape[1]):
        raise ValueError(
            f"target_channel={target_channel} is out of range for Pipe_Q with {output_q.shape[1]} channels."
        )
    if ntrain <= 0 or ntest <= 0:
        raise ValueError(f"ntrain and ntest must be positive, got {ntrain} and {ntest}.")
    if ntrain + ntest > n_samples:
        raise ValueError(
            f"ntrain + ntest exceeds available samples: {ntrain} + {ntest} > {n_samples}."
        )

    x_train = np.stack(
        [
            input_x[:ntrain, ::r1, ::r2][:, :s1, :s2],
            input_y[:ntrain, ::r1, ::r2][:, :s1, :s2],
        ],
        axis=-1,
    )
    y_train = output_q[:ntrain, target_channel, ::r1, ::r2][:, :s1, :s2]

    test_start = ntrain
    test_stop = ntrain + ntest
    x_test = np.stack(
        [
            input_x[test_start:test_stop, ::r1, ::r2][:, :s1, :s2],
            input_y[test_start:test_stop, ::r1, ::r2][:, :s1, :s2],
        ],
        axis=-1,
    )
    y_test = output_q[test_start:test_stop, target_channel, ::r1, ::r2][:, :s1, :s2]

    x_train = torch.from_numpy(x_train.reshape(ntrain, -1, 2)).float()
    y_train = torch.from_numpy(y_train.reshape(ntrain, -1, 1)).float()
    x_test = torch.from_numpy(x_test.reshape(ntest, -1, 2)).float()
    y_test = torch.from_numpy(y_test.reshape(ntest, -1, 1)).float()

    meta = {
        "shapelist": (s1, s2),
        "fun_dim": 2,
        "out_dim": 1,
        "space_dim": 2,
        "task": "steady",
        "loader": "pipe",
        "geotype": "structured_2D",
        "ntrain": ntrain,
        "ntest": ntest,
        "root_dir": str(root),
        "target_channel": target_channel,
    }
    return x_train, y_train, x_test, y_test, meta


def _build_normalizers(cfg: dict, x_train: torch.Tensor, y_train: torch.Tensor):
    data_cfg = cfg.get("data", {})
    normalize = bool(data_cfg.get("normalize", True))
    if not normalize:
        return IdentityTransformer(), IdentityTransformer()
    norm_type = str(data_cfg.get("norm_type", "UnitTransformer"))
    if norm_type != "UnitTransformer":
        raise ValueError(
            "Pipe loader currently supports only data.norm_type='UnitTransformer'."
        )
    return UnitTransformer(x_train), UnitTransformer(y_train)


def build_train_val_loaders(cfg: dict):
    x_train, y_train, _, _, meta = _load_pipe_raw(cfg)
    x_normalizer, y_normalizer = _build_normalizers(cfg, x_train, y_train)
    x_train = x_normalizer.encode(x_train)
    y_train = y_normalizer.encode(y_train)

    dataset = PipeDataset(x_train, x_train, y_train)
    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 4))
    val_size = int(training_cfg.get("val_size", 16))
    if val_size < 0:
        raise ValueError(f"Invalid val_size={val_size}")
    if val_size == 0:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        train_loader.pipe_meta = meta
        train_loader.x_normalizer = x_normalizer
        train_loader.y_normalizer = y_normalizer
        return train_loader, None

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
    train_loader.pipe_meta = meta
    val_loader.pipe_meta = meta
    train_loader.x_normalizer = x_normalizer
    val_loader.x_normalizer = x_normalizer
    train_loader.y_normalizer = y_normalizer
    val_loader.y_normalizer = y_normalizer
    return train_loader, val_loader


def build_test_loader(cfg: dict):
    x_train, y_train, x_test, y_test, meta = _load_pipe_raw(cfg)
    x_normalizer, y_normalizer = _build_normalizers(cfg, x_train, y_train)
    x_test = x_normalizer.encode(x_test)
    y_test = y_normalizer.encode(y_test)

    dataset = PipeDataset(x_test, x_test, y_test)
    batch_size = int(
        cfg.get("evaluation", {}).get(
            "batch_size", cfg.get("training", {}).get("batch_size", 4)
        )
    )
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader.pipe_meta = meta
    test_loader.x_normalizer = x_normalizer
    test_loader.y_normalizer = y_normalizer
    return test_loader
