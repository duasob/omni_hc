import pytest
import torch

from omni_hc.constraints import ConstraintOutput, DarcyFluxConstraint


class AffineNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


def test_darcy_flux_constraint_recovers_scalar_pressure_with_dirichlet_boundary():
    torch.manual_seed(0)
    height = width = 8
    n_points = height * width

    constraint = DarcyFluxConstraint(
        spectral_backend="helmholtz_sine",
        padding=2,
        padding_mode="reflect",
        particular_field="y_only",
        enforce_boundary=True,
        boundary_value=0.0,
        shapelist=(height, width),
    )
    constraint.set_input_normalizer(AffineNormalizer(mean=3.0, std=2.0))
    constraint.set_target_normalizer(AffineNormalizer(mean=1.0, std=4.0))

    psi_pred = torch.randn(2, n_points, 1)
    permeability_physical = torch.full((2, n_points, 1), 4.0)
    permeability_encoded = constraint.input_normalizer.encode(permeability_physical)

    pressure_encoded = constraint(pred=psi_pred, fx=permeability_encoded)
    pressure = constraint.target_normalizer.decode(pressure_encoded)
    pressure_grid = pressure.transpose(1, 2).reshape(2, 1, height, width)

    assert pressure_encoded.shape == (2, n_points, 1)
    assert torch.allclose(
        pressure_grid[..., 0, :],
        torch.zeros_like(pressure_grid[..., 0, :]),
        atol=1e-6,
    )
    assert torch.allclose(
        pressure_grid[..., -1, :],
        torch.zeros_like(pressure_grid[..., -1, :]),
        atol=1e-6,
    )
    assert torch.allclose(
        pressure_grid[..., :, 0],
        torch.zeros_like(pressure_grid[..., :, 0]),
        atol=1e-6,
    )
    assert torch.allclose(
        pressure_grid[..., :, -1],
        torch.zeros_like(pressure_grid[..., :, -1]),
        atol=1e-6,
    )


def test_darcy_flux_fd_derivative_conserves_mass_discretely():
    # The FD grad_perp(psi) is consistent with the central-difference divergence
    # used downstream, so div(q) == force_value exactly on the interior (the
    # spectral path only holds in the spectral sense and leaves a large residual
    # under the FD divergence the Poisson RHS actually uses).
    from omni_hc.constraints.utils.spectral import (
        finite_difference_divergence_2d,
        reshape_channels_last_to_grid,
    )

    torch.manual_seed(0)
    height = width = 16
    n_points = height * width
    force_value = 1.0

    constraint = DarcyFluxConstraint(
        stream_derivative="fd",
        padding=0,
        force_value=force_value,
        particular_field="y_only",
        shapelist=(height, width),
    )
    psi_pred = torch.randn(2, n_points, 1)
    perm = torch.full((2, n_points, 1), 4.0)

    out = constraint(pred=psi_pred, fx=perm, return_aux=True)
    flux = reshape_channels_last_to_grid(
        out.aux["constrained_flux"], shapelist=(height, width)
    )
    dy = dx = 1.0 / (height - 1)
    div = finite_difference_divergence_2d(flux, dy=dy, dx=dx)
    residual = (div[..., 1:-1, 1:-1] - force_value).abs()
    assert float(residual.max()) < 1e-3


def test_darcy_flux_constraint_requires_scalar_stream_output():
    constraint = DarcyFluxConstraint(
        spectral_backend="helmholtz_sine",
        shapelist=(8, 8),
    )
    bad_pred = torch.randn(1, 64, 2)
    fx = torch.ones(1, 64, 1)

    with pytest.raises(ValueError, match="scalar backbone output"):
        constraint(pred=bad_pred, fx=fx)


def test_darcy_flux_constraint_emits_helmholtz_diagnostics():
    torch.manual_seed(0)
    height = width = 8
    n_points = height * width

    constraint = DarcyFluxConstraint(
        spectral_backend="helmholtz_sine",
        padding=2,
        padding_mode="reflect",
        particular_field="y_only",
        enforce_boundary=True,
        boundary_value=0.0,
        shapelist=(height, width),
    )

    psi_pred = torch.randn(1, n_points, 1)
    permeability_physical = torch.full((1, n_points, 1), 4.0)

    out = constraint(
        pred=psi_pred,
        fx=permeability_physical,
        return_aux=True,
    )

    assert isinstance(out, ConstraintOutput)
    assert out.pred.shape == (1, n_points, 1)
    assert "stream_correction" in out.aux
    assert "constrained_flux" in out.aux
    assert "constraint/stream_div_abs_mean" in out.diagnostics
    assert "constraint/flux_div_abs_mean" in out.diagnostics
    assert "constraint/flux_rmse" in out.diagnostics
    assert "constraint/w_error_abs_mean" in out.diagnostics
    assert "constraint/w_curl_abs_mean" in out.diagnostics
    assert "constraint/darcy_res_abs_mean" in out.diagnostics
    assert "constraint/boundary_abs_mean" in out.diagnostics
    assert float(out.diagnostics["constraint/stream_div_abs_mean"].value) < 1.0e-3
    assert float(out.diagnostics["constraint/boundary_abs_max"].value) == 0.0
