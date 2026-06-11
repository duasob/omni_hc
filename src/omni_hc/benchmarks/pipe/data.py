from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from omni_hc.training.reproducibility import (
    DATA_SHUFFLE_SEED_OFFSET,
    DATA_SPLIT_SEED_OFFSET,
    seeded_generator,
    training_seed,
)


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
        self,
        coords: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        y_uy: torch.Tensor | None = None,
    ) -> None:
        self._coords = coords
        self._x = x
        self._y = y
        self._y_uy = y_uy

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            "coords": self._coords[idx],
            "x": self._x[idx],
            "y": self._y[idx],
        }
        if self._y_uy is not None:
            item["y_uy"] = self._y_uy[idx]
        return item


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
    load_uy = bool(data_cfg.get("load_uy", False))

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

    uy_channel = 1
    if load_uy and (uy_channel < 0 or uy_channel >= int(output_q.shape[1])):
        raise ValueError(
            f"load_uy=True but uy channel index {uy_channel} is out of range for "
            f"Pipe_Q with {output_q.shape[1]} channels."
        )

    x_train = np.stack(
        [
            input_x[:ntrain, ::r1, ::r2][:, :s1, :s2],
            input_y[:ntrain, ::r1, ::r2][:, :s1, :s2],
        ],
        axis=-1,
    )
    y_train = output_q[:ntrain, target_channel, ::r1, ::r2][:, :s1, :s2]
    y_uy_train = (
        output_q[:ntrain, uy_channel, ::r1, ::r2][:, :s1, :s2] if load_uy else None
    )

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
    y_uy_test = (
        output_q[test_start:test_stop, uy_channel, ::r1, ::r2][:, :s1, :s2]
        if load_uy
        else None
    )

    x_train = torch.from_numpy(x_train.reshape(ntrain, -1, 2)).float()
    y_train = torch.from_numpy(y_train.reshape(ntrain, -1, 1)).float()
    y_uy_train = (
        torch.from_numpy(y_uy_train.reshape(ntrain, -1, 1)).float()
        if y_uy_train is not None
        else None
    )
    x_test = torch.from_numpy(x_test.reshape(ntest, -1, 2)).float()
    y_test = torch.from_numpy(y_test.reshape(ntest, -1, 1)).float()
    y_uy_test = (
        torch.from_numpy(y_uy_test.reshape(ntest, -1, 1)).float()
        if y_uy_test is not None
        else None
    )

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
        "load_uy": load_uy,
    }
    return x_train, y_train, y_uy_train, x_test, y_test, y_uy_test, meta


def _build_normalizers(cfg: dict, x_train: torch.Tensor, y_train: torch.Tensor, y_uy_train: torch.Tensor | None):
    data_cfg = cfg.get("data", {})
    normalize = bool(data_cfg.get("normalize", True))
    if not normalize:
        uy_normalizer = IdentityTransformer() if y_uy_train is not None else None
        return IdentityTransformer(), IdentityTransformer(), uy_normalizer
    norm_type = str(data_cfg.get("norm_type", "UnitTransformer"))
    if norm_type != "UnitTransformer":
        raise ValueError(
            "Pipe loader currently supports only data.norm_type='UnitTransformer'."
        )
    uy_normalizer = UnitTransformer(y_uy_train) if y_uy_train is not None else None
    return UnitTransformer(x_train), UnitTransformer(y_train), uy_normalizer


def _attach_normalizers(loader, *, x_normalizer, y_normalizer, uy_normalizer):
    loader.x_normalizer = x_normalizer
    loader.y_normalizer = y_normalizer
    if uy_normalizer is not None:
        loader.uy_normalizer = uy_normalizer


def build_train_val_loaders(cfg: dict):
    x_train, y_train, y_uy_train, _, _, _, meta = _load_pipe_raw(cfg)
    x_normalizer, y_normalizer, uy_normalizer = _build_normalizers(
        cfg, x_train, y_train, y_uy_train
    )
    x_train = x_normalizer.encode(x_train)
    y_train = y_normalizer.encode(y_train)

    dataset = PipeDataset(x_train, x_train, y_train, y_uy=y_uy_train)
    training_cfg = cfg.get("training", {})
    seed = training_seed(cfg)
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
            generator=seeded_generator(seed, offset=DATA_SHUFFLE_SEED_OFFSET),
        )
        train_loader.pipe_meta = meta
        _attach_normalizers(
            train_loader,
            x_normalizer=x_normalizer,
            y_normalizer=y_normalizer,
            uy_normalizer=uy_normalizer,
        )
        return train_loader, None

    val_size = min(max(val_size, 1), len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = seeded_generator(seed, offset=DATA_SPLIT_SEED_OFFSET)
    train_subset, val_subset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=seeded_generator(seed, offset=DATA_SHUFFLE_SEED_OFFSET),
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    train_loader.pipe_meta = meta
    val_loader.pipe_meta = meta
    _attach_normalizers(
        train_loader,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        uy_normalizer=uy_normalizer,
    )
    _attach_normalizers(
        val_loader,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        uy_normalizer=uy_normalizer,
    )
    return train_loader, val_loader


def build_test_loader(cfg: dict):
    x_train, y_train, y_uy_train, x_test, y_test, y_uy_test, meta = _load_pipe_raw(cfg)
    x_normalizer, y_normalizer, uy_normalizer = _build_normalizers(
        cfg, x_train, y_train, y_uy_train
    )
    x_test = x_normalizer.encode(x_test)
    y_test = y_normalizer.encode(y_test)

    dataset = PipeDataset(x_test, x_test, y_test, y_uy=y_uy_test)
    batch_size = int(
        cfg.get("evaluation", {}).get(
            "batch_size", cfg.get("training", {}).get("batch_size", 4)
        )
    )
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader.pipe_meta = meta
    _attach_normalizers(
        test_loader,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        uy_normalizer=uy_normalizer,
    )
    return test_loader
