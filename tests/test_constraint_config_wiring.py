from types import SimpleNamespace

import torch

from omni_hc.constraints import ConstrainedModel
from omni_hc.core import load_yaml_file
from omni_hc.integrations.nsl.modeling import _build_constraint


class DummyBackbone(torch.nn.Module):
    def __init__(self, pred: torch.Tensor):
        super().__init__()
        self.register_buffer("pred", pred)

    def forward(self, *args, **kwargs):
        return self.pred


def _args(*, out_dim=1, shapelist=(4, 5)):
    return SimpleNamespace(out_dim=out_dim, shapelist=shapelist, n_hidden=8)


def test_pipe_ux_boundary_config_builds_working_constraint():
    height, width = 4, 5
    pred = torch.randn(2, height * width, 1)
    coords = _structured_coords(height, width).expand(2, -1, -1)
    cfg = load_yaml_file("configs/constraints/pipe_ux_boundary.yaml")

    model = _build_constraint(DummyBackbone(pred), _args(shapelist=(height, width)), cfg)
    out = model(coords)
    out_grid = out.reshape(2, height, width, 1)

    assert isinstance(model, ConstrainedModel)
    assert torch.allclose(out_grid[:, :, 0, 0], torch.zeros(2, height), atol=1e-8)
    assert torch.allclose(out_grid[:, :, -1, 0], torch.zeros(2, height), atol=1e-8)


def test_pipe_stream_boundary_config_builds_working_constraint():
    height, width = 8, 9
    pred = torch.zeros(1, height * width, 1)
    coords = _structured_coords(height, width)
    cfg = load_yaml_file("configs/constraints/pipe_stream_function_boundary.yaml")

    model = _build_constraint(DummyBackbone(pred), _args(shapelist=(height, width)), cfg)
    out = model(coords, return_aux=True)

    assert isinstance(model, ConstrainedModel)
    assert "constraint/stream_inlet_abs_max" in out.diagnostics
    assert "stream_psi_bc" in out.aux


def test_darcy_flux_config_builds_pressure_with_dirichlet_boundary():
    height = width = 8
    pred = torch.randn(1, height * width, 1)
    permeability = torch.full((1, height * width, 1), 4.0)
    cfg = load_yaml_file("configs/constraints/darcy_flux_fft_pad.yaml")
    cfg["constraint"]["padding"] = 2

    model = _build_constraint(DummyBackbone(pred), _args(shapelist=(height, width)), cfg)
    pressure = model(fx=permeability)
    pressure_grid = pressure.reshape(1, height, width, 1)

    assert isinstance(model, ConstrainedModel)
    assert torch.allclose(
        pressure_grid[:, 0, :, 0],
        torch.zeros(1, width),
        atol=1e-6,
    )
    assert torch.allclose(
        pressure_grid[:, -1, :, 0],
        torch.zeros(1, width),
        atol=1e-6,
    )


def _structured_coords(height: int, width: int):
    x = torch.linspace(0.0, 10.0, height)
    y = torch.linspace(-2.0, 2.0, width)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2)
