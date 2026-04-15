from pathlib import Path

import scipy.io as scio
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class DictTensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._x = x
        self._y = y

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self._x[idx], "y": self._y[idx]}


def make_grid(h: int, w: int, *, device, dtype):
    ys = torch.linspace(0, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    return grid.view(-1, 2)


def resolve_ns_mat_file(root_dir: str | Path) -> Path:
    root = Path(root_dir)
    if root.is_file() and root.suffix == ".mat":
        return root
    candidate = root / "NavierStokes_V1e-5_N1200_T20.mat"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Expected a Navier-Stokes .mat file or a directory containing "
        "'NavierStokes_V1e-5_N1200_T20.mat'."
    )


def _as_int(value, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _load_ns_dataset_info(cfg: dict):
    mat_path = resolve_ns_mat_file(cfg["paths"]["root_dir"])
    raw = scio.loadmat(str(mat_path))
    if "u" not in raw:
        raise KeyError(f"Expected key 'u' in {mat_path}")

    u = raw["u"]
    if u.ndim != 4:
        raise ValueError(f"Expected 'u' with shape (N,H,W,T), got {u.shape}")

    data_cfg = cfg.get("data", {})
    model_args = cfg.get("model", {}).get("args", {})
    n_samples, h_full, w_full, t_full = u.shape
    r1 = _as_int(data_cfg.get("downsamplex", model_args.get("downsamplex")), 1)
    r2 = _as_int(data_cfg.get("downsampley", model_args.get("downsampley")), 1)
    t_in = _as_int(data_cfg.get("T_in", model_args.get("T_in")), 10)
    t_out = _as_int(data_cfg.get("T_out", model_args.get("T_out")), 10)
    out_dim = _as_int(data_cfg.get("out_dim", model_args.get("out_dim")), 1)
    ntrain = _as_int(data_cfg.get("ntrain"), 1000)
    ntest = _as_int(data_cfg.get("ntest"), 200)

    if out_dim != 1:
        raise ValueError("The demo loader currently supports out_dim=1 only.")
    if t_in + t_out > t_full:
        raise ValueError(f"T_in + T_out exceeds available time steps: {t_in + t_out} > {t_full}")

    s1 = int(((h_full - 1) / r1) + 1)
    s2 = int(((w_full - 1) / r2) + 1)

    meta = {
        "shapelist": (int(s1), int(s2)),
        "t_in": int(t_in),
        "t_out": int(t_out),
        "out_dim": int(out_dim),
        "fun_dim": int(out_dim * t_in),
        "ntrain": int(min(ntrain, n_samples)),
        "ntest": int(min(ntest, n_samples)),
        "path": str(mat_path),
        "u_shape": tuple(int(v) for v in u.shape),
    }
    return u[:, ::r1, ::r2][:, :s1, :s2], meta


def _as_autoregressive_dataset(u: torch.Tensor, *, t_in: int, t_out: int, out_dim: int):
    data_a = u[..., None, :t_in]
    data_y = u[..., None, t_in : t_in + t_out]
    x = torch.from_numpy(data_a.reshape(data_a.shape[0], -1, out_dim * t_in)).float()
    y = torch.from_numpy(data_y.reshape(data_y.shape[0], -1, out_dim * t_out)).float()
    return DictTensorDataset(x, y)


def load_ns_autoregressive_dataset(cfg: dict, *, split: str):
    u, meta = _load_ns_dataset_info(cfg)
    n_samples = int(u.shape[0])
    ntrain = int(meta["ntrain"])
    ntest = int(meta["ntest"])

    if split == "train":
        if ntrain <= 1 or ntrain >= n_samples:
            raise ValueError(f"Invalid ntrain={ntrain} for n_samples={n_samples}")
        subset = u[:ntrain]
    elif split == "test":
        if ntest <= 0:
            raise ValueError(f"Invalid ntest={ntest}")
        subset = u[-ntest:]
    else:
        raise ValueError(f"Unknown split: {split}")

    dataset = _as_autoregressive_dataset(
        subset,
        t_in=int(meta["t_in"]),
        t_out=int(meta["t_out"]),
        out_dim=int(meta["out_dim"]),
    )
    return dataset, meta


def build_train_val_loaders(cfg: dict):
    dataset, meta = load_ns_autoregressive_dataset(cfg, split="train")
    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 4))
    val_size = int(training_cfg.get("val_size", 100))
    val_size = min(max(val_size, 1), len(dataset) - 1)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(int(training_cfg.get("seed", 42)))
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    train_loader.ns_meta = meta
    val_loader.ns_meta = meta
    return train_loader, val_loader


def build_test_loader(cfg: dict):
    dataset, meta = load_ns_autoregressive_dataset(cfg, split="test")
    batch_size = int(cfg.get("evaluation", {}).get("batch_size", cfg.get("training", {}).get("batch_size", 4)))
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader.ns_meta = meta
    return test_loader
