import pytest
import torch

from omni_hc.constraints import ConstraintOutput, DarcyDefectCorrectionConstraint


class AffineNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


def _make_coords(height, width, lower=0.0, upper=1.0):
    y = torch.linspace(lower, upper, height)
    x = torch.linspace(lower, upper, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return coords.unsqueeze(0)  # (1, H*W, 2)


def test_dirichlet_bcs_are_exactly_zero():
    """Dirichlet ansatz enforces u=0 on all four boundary edges."""
    torch.manual_seed(0)
    height = width = 10
    n_points = height * width

    constraint = DarcyDefectCorrectionConstraint(
        force_value=1.0,
        n_correction_steps=0,   # ansatz only, no correction
        shapelist=(height, width),
    )
    coords = _make_coords(height, width).expand(2, -1, -1)
    pred = torch.randn(2, n_points, 1)
    fx = torch.full((2, n_points, 1), 5.0)

    out_flat = constraint(pred=pred, coords=coords, fx=fx)
    out = out_flat.transpose(1, 2).reshape(2, 1, height, width)

    assert out.shape == (2, 1, height, width)
    assert torch.allclose(out[:, :, 0, :], torch.zeros_like(out[:, :, 0, :]), atol=1e-6)
    assert torch.allclose(out[:, :, -1, :], torch.zeros_like(out[:, :, -1, :]), atol=1e-6)
    assert torch.allclose(out[:, :, :, 0], torch.zeros_like(out[:, :, :, 0]), atol=1e-6)
    assert torch.allclose(out[:, :, :, -1], torch.zeros_like(out[:, :, :, -1]), atol=1e-6)


def test_correction_reduces_residual_for_constant_permeability():
    """
    For a constant permeability field and a non-trivial backbone output,
    one correction step should reduce the Darcy residual substantially.
    """
    torch.manual_seed(1)
    height = width = 16
    n_points = height * width
    a_val = 5.0

    constraint = DarcyDefectCorrectionConstraint(
        force_value=1.0,
        n_correction_steps=1,
        shapelist=(height, width),
    )
    coords = _make_coords(height, width)
    pred = torch.randn(1, n_points, 1)
    fx = torch.full((1, n_points, 1), a_val)

    out_no_corr = constraint.__class__(
        force_value=1.0,
        n_correction_steps=0,
        shapelist=(height, width),
    )
    out_1step = constraint

    res_before = _darcy_res_mean(out_no_corr(pred=pred, coords=coords, fx=fx), fx, height, width, a_val)
    res_after = _darcy_res_mean(out_1step(pred=pred, coords=coords, fx=fx), fx, height, width, a_val)

    assert res_after < res_before * 0.01, (
        f"Correction should reduce residual by >99% for constant a; "
        f"before={res_before:.4g}, after={res_after:.4g}"
    )


def _darcy_res_mean(u_flat, fx_flat, height, width, a_val):
    """Helper: compute mean |r| for scalar constant-a Darcy."""
    u = u_flat.transpose(1, 2).reshape(1, 1, height, width)
    dy = dx = 1.0 / (height - 1)
    flux_x = a_val * (u[:, :, :, 1:] - u[:, :, :, :-1]) / dx
    flux_y = a_val * (u[:, :, 1:, :] - u[:, :, :-1, :]) / dy
    div_x = (flux_x[:, :, 1:-1, 1:] - flux_x[:, :, 1:-1, :-1]) / dx
    div_y = (flux_y[:, :, 1:, 1:-1] - flux_y[:, :, :-1, 1:-1]) / dy
    r = -(div_x + div_y) - 1.0
    return float(r.abs().mean())


def test_constraint_output_with_diagnostics():
    """return_aux=True returns ConstraintOutput with expected diagnostic keys."""
    torch.manual_seed(2)
    height = width = 12
    n_points = height * width

    constraint = DarcyDefectCorrectionConstraint(
        force_value=1.0,
        n_correction_steps=1,
        shapelist=(height, width),
    )
    coords = _make_coords(height, width)
    pred = torch.randn(1, n_points, 1)
    fx = torch.full((1, n_points, 1), 4.0)

    out = constraint(pred=pred, coords=coords, fx=fx, return_aux=True)

    assert isinstance(out, ConstraintOutput)
    assert out.pred.shape == (1, n_points, 1)
    assert "pred_ansatz" in out.aux
    assert "constraint/darcy_res_abs_mean" in out.diagnostics
    assert "constraint/darcy_res_bulk_abs_mean" in out.diagnostics
    assert "constraint/darcy_res_intf_abs_mean" in out.diagnostics
    assert "constraint/boundary_abs_mean" in out.diagnostics
    assert "constraint/correction_norm_mean" in out.diagnostics


def test_normalizers_encode_decode_correctly():
    """Constraint re-encodes the output so loss operates in normalized space."""
    torch.manual_seed(3)
    height = width = 8
    n_points = height * width

    constraint = DarcyDefectCorrectionConstraint(
        force_value=1.0,
        n_correction_steps=1,
        shapelist=(height, width),
    )
    constraint.set_input_normalizer(AffineNormalizer(mean=7.5, std=4.5))
    constraint.set_target_normalizer(AffineNormalizer(mean=0.005, std=0.003))

    coords = _make_coords(height, width)
    pred_norm = torch.randn(1, n_points, 1)
    # Encode a permeability of ~7.5 (mean of {3,12} = 7.5)
    fx_norm = constraint.input_normalizer.encode(torch.full((1, n_points, 1), 7.5))

    out_norm = constraint(pred=pred_norm, coords=coords, fx=fx_norm)

    assert out_norm.shape == (1, n_points, 1)
    # Decoded output should have physical BCs = 0
    out_phys = constraint.target_normalizer.decode(out_norm)
    out_grid = out_phys.transpose(1, 2).reshape(1, 1, height, width)
    assert torch.allclose(out_grid[:, :, 0, :], torch.zeros_like(out_grid[:, :, 0, :]), atol=1e-5)
    assert torch.allclose(out_grid[:, :, -1, :], torch.zeros_like(out_grid[:, :, -1, :]), atol=1e-5)


def test_requires_fx_and_coords():
    constraint = DarcyDefectCorrectionConstraint(shapelist=(8, 8))
    pred = torch.randn(1, 64, 1)
    coords = _make_coords(8, 8)
    fx = torch.ones(1, 64, 1)

    with pytest.raises(ValueError, match="fx"):
        constraint(pred=pred, coords=coords)

    with pytest.raises(ValueError, match="coords"):
        constraint(pred=pred, fx=fx)
