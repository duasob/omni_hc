from types import SimpleNamespace

import torch

from omni_hc.constraints import ConstrainedModel
from omni_hc.core import load_yaml_file
from omni_hc.integrations.nsl.modeling import _build_constraint, build_model_args


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
    cfg = load_yaml_file("configs/constraints/pipe_ux_boundary_ansatz.yaml")

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
    cfg = load_yaml_file("configs/constraints/pipe_stream_function_boundary_ansatz.yaml")

    model = _build_constraint(DummyBackbone(pred), _args(shapelist=(height, width)), cfg)
    out = model(coords, return_aux=True)

    assert isinstance(model, ConstrainedModel)
    assert "constraint/stream_inlet_abs_max" in out.diagnostics
    assert "stream_psi_bc" in out.aux


def test_darcy_flux_config_builds_pressure_with_dirichlet_boundary():
    height = width = 8
    pred = torch.randn(1, height * width, 1)
    permeability = torch.full((1, height * width, 1), 4.0)
    cfg = load_yaml_file("configs/constraints/darcy_flux_constraint.yaml")
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


def test_elasticity_deviatoric_stress_config_builds_scalar_constraint():
    pred = torch.randn(2, 13, 1)
    coords = torch.rand(2, 13, 2)
    cfg = load_yaml_file("configs/constraints/elasticity_deviatoric_stress_constraint.yaml")

    model = _build_constraint(DummyBackbone(pred), _args(out_dim=1), cfg)
    out = model(coords, return_aux=True)

    assert isinstance(model, ConstrainedModel)
    assert out.pred.shape == (2, 13, 1)
    assert "constraint/det_c_abs_error_max" in out.diagnostics


def test_plasticity_mesh_consistency_config_builds_ordered_mesh():
    height, width = 4, 3
    pred = torch.zeros(2, height * width, 3)
    cfg = load_yaml_file("configs/constraints/plasticity_mesh_consistency_constraint.yaml")
    cfg["constraint"]["x_left"] = 0.35
    cfg["constraint"]["x_right"] = -1.15
    cfg["constraint"]["y_top"] = 0.9
    cfg["constraint"]["y_bottom"] = -0.1

    model = _build_constraint(
        DummyBackbone(pred),
        _args(out_dim=3, shapelist=(height, width)),
        cfg,
    )
    out = model(return_aux=True)

    assert isinstance(model, ConstrainedModel)
    assert out.pred.shape == (2, height * width, 4)
    assert "constraint/bottom_y_abs_error_max" in out.diagnostics
    assert "constraint/axis_order_margin_min" in out.diagnostics


def test_plasticity_envelope_config_builds_bottom_fixed_envelope_mesh():
    height, width = 4, 3
    pred = torch.zeros(2, height * width, 4)
    fx = torch.ones(2, height * width, 1)
    cfg = load_yaml_file("configs/constraints/plasticity_envelope_constraint.yaml")
    cfg["constraint"]["x_left"] = 0.35
    cfg["constraint"]["x_right"] = -1.15
    cfg["constraint"]["y_top"] = 1.0
    cfg["constraint"]["top_height"] = 1.0
    cfg["constraint"]["y_bottom"] = 0.0
    cfg["constraint"]["die_speed"] = 0.2

    model = _build_constraint(
        DummyBackbone(pred),
        _args(out_dim=4, shapelist=(height, width)),
        cfg,
    )
    out = model(fx=fx, T=torch.ones(2, 1), return_aux=True)

    assert isinstance(model, ConstrainedModel)
    assert out.pred.shape == (2, height * width, 4)
    assert "constraint/bottom_y_abs_error_max" in out.diagnostics
    assert "constraint/top_clearance_min" in out.diagnostics
    assert out.diagnostics["constraint/top_violation_count"].value == 0.0


def test_constraint_backbone_out_dim_overrides_backbone_output_only():
    cfg = {
        "model": {
            "backbone": "FNO",
            "args": {
                "n_hidden": 8,
                "modes": 4,
                "out_dim": 1,
            },
        },
        "constraint": {
            "name": "elasticity_deviatoric_stress_constraint",
            "backbone_out_dim": 2,
        },
    }

    args = build_model_args(cfg)

    assert args.out_dim == 2
    assert args.constraint_target_out_dim == 1


def _structured_coords(height: int, width: int):
    x = torch.linspace(0.0, 10.0, height)
    y = torch.linspace(-2.0, 2.0, width)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2)
