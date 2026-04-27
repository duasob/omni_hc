from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

from omni_hc.core import load_yaml_file
from omni_hc.diagnostics import infer_boundary_ansatz_maps, plot_boundary_ansatz_maps
from omni_hc.integrations.nsl.modeling import _build_constraint


DEFAULT_CONFIGS = (
    "configs/constraints/dirichlet_ansatz_zero.yaml",
    "configs/constraints/pipe_wall_no_slip.yaml",
    "configs/constraints/pipe_inlet_parabolic.yaml",
    "configs/constraints/pipe_ux_boundary.yaml",
    "configs/constraints/pipe_stream_function_boundary.yaml",
)


class ZeroBackbone(torch.nn.Module):
    def __init__(self, pred: torch.Tensor):
        super().__init__()
        self.register_buffer("pred", pred)

    def forward(self, *args, **kwargs):
        return self.pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot g and l maps for boundary ansatz constraints."
    )
    parser.add_argument(
        "--constraint-configs",
        type=Path,
        nargs="+",
        default=[Path(path) for path in DEFAULT_CONFIGS],
        help="Constraint YAML files to visualize.",
    )
    parser.add_argument(
        "--grid-shape",
        type=int,
        nargs=2,
        default=(129, 129),
        metavar=("H", "W"),
        help="Structured grid shape used for synthetic maps.",
    )
    parser.add_argument(
        "--pipe-data-dir",
        type=Path,
        default=Path("data/pipe"),
        help="Optional pipe data directory used to draw maps on the real pipe mesh.",
    )
    parser.add_argument(
        "--pipe-sample",
        type=int,
        default=0,
        help="Pipe sample index used when pipe data is available.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/boundary_ansatz_maps"),
        help="Directory where figures are written.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Output channel to plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for config_path in args.constraint_configs:
        cfg = load_yaml_file(config_path)
        name = str(cfg.get("constraint", {}).get("name", config_path.stem))
        grid_shape, coords = coords_for_constraint(
            name,
            requested_grid_shape=tuple(args.grid_shape),
            pipe_data_dir=args.pipe_data_dir,
            pipe_sample=args.pipe_sample,
        )
        constraint = build_constraint_from_config(cfg, grid_shape=grid_shape)
        maps = infer_boundary_ansatz_maps(
            constraint,
            pred_shape=(1, grid_shape[0] * grid_shape[1], 1),
            grid_shape=grid_shape,
            coords=coords,
        )
        out_path = args.out_dir / f"{config_path.stem}_g_l.png"
        plot_boundary_ansatz_maps(
            maps,
            out_path=out_path,
            coords=coords,
            channel=args.channel,
            title=f"{config_path.stem}: {maps.space}",
            show=args.show,
        )
        print(
            f"Wrote {out_path} "
            f"(constraint={maps.constraint_name}, space={maps.space}, grid={grid_shape})"
        )


def build_constraint_from_config(cfg: dict, *, grid_shape: tuple[int, int]):
    pred = torch.zeros(1, grid_shape[0] * grid_shape[1], 1)
    args = SimpleNamespace(out_dim=1, shapelist=grid_shape, n_hidden=8)
    model = _build_constraint(ZeroBackbone(pred), args, cfg)
    if not hasattr(model, "constraint"):
        raise ValueError("Config did not build a constrained model")
    return model.constraint


def coords_for_constraint(
    name: str,
    *,
    requested_grid_shape: tuple[int, int],
    pipe_data_dir: Path,
    pipe_sample: int,
) -> tuple[tuple[int, int], torch.Tensor]:
    if name.startswith("pipe") and _pipe_arrays_exist(pipe_data_dir):
        x = np.load(pipe_data_dir / "Pipe_X.npy", mmap_mode="r")[pipe_sample]
        y = np.load(pipe_data_dir / "Pipe_Y.npy", mmap_mode="r")[pipe_sample]
        coords = np.stack([x, y], axis=-1)
        grid_shape = tuple(int(v) for v in x.shape)
        return grid_shape, torch.as_tensor(coords, dtype=torch.float32).reshape(
            1, grid_shape[0] * grid_shape[1], 2
        )
    return requested_grid_shape, unit_square_coords(*requested_grid_shape)


def unit_square_coords(height: int, width: int) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, height)
    y = torch.linspace(0.0, 1.0, width)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2)


def _pipe_arrays_exist(data_dir: Path) -> bool:
    return (data_dir / "Pipe_X.npy").exists() and (data_dir / "Pipe_Y.npy").exists()


if __name__ == "__main__":
    main()
