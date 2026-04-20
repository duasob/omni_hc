import torch

from omni_hc.constraints import ConstraintOutput, DarcyFluxConstraint, fft_leray_project_2d
from omni_hc.constraints.spectral import spectral_divergence_2d


class AffineNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


def test_fft_leray_projection_zeroes_divergence_of_correction():
    torch.manual_seed(0)
    field = torch.randn(2, 2, 16, 16)
    projected = fft_leray_project_2d(field, dy=1.0 / 15.0, dx=1.0 / 15.0)
    divergence = spectral_divergence_2d(projected, dy=1.0 / 15.0, dx=1.0 / 15.0)

    assert divergence.abs().max().item() < 1e-4


def test_darcy_flux_constraint_recovers_scalar_pressure_with_zero_boundary():
    torch.manual_seed(0)
    height = width = 8
    n_points = height * width

    constraint = DarcyFluxConstraint(
        spectral_backend="fft_pad",
        padding=2,
        padding_mode="reflect",
        enforce_boundary=True,
        boundary_value=0.0,
        shapelist=(height, width),
    )
    constraint.set_input_normalizer(AffineNormalizer(mean=3.0, std=2.0))
    constraint.set_target_normalizer(AffineNormalizer(mean=1.0, std=4.0))

    flux_pred = torch.randn(2, n_points, 2)
    permeability_physical = torch.full((2, n_points, 1), 4.0)
    permeability_encoded = constraint.input_normalizer.encode(permeability_physical)

    pressure_encoded = constraint(pred=flux_pred, fx=permeability_encoded)
    pressure = constraint.target_normalizer.decode(pressure_encoded)
    pressure_grid = pressure.reshape(2, height, width, 1)

    assert pressure_encoded.shape == (2, n_points, 1)
    assert torch.allclose(
        pressure_grid[:, 0, :, :],
        torch.zeros_like(pressure_grid[:, 0, :, :]),
        atol=1e-6,
    )
    assert torch.allclose(
        pressure_grid[:, -1, :, :],
        torch.zeros_like(pressure_grid[:, -1, :, :]),
        atol=1e-6,
    )
    assert torch.allclose(
        pressure_grid[:, :, 0, :],
        torch.zeros_like(pressure_grid[:, :, 0, :]),
        atol=1e-6,
    )
    assert torch.allclose(
        pressure_grid[:, :, -1, :],
        torch.zeros_like(pressure_grid[:, :, -1, :]),
        atol=1e-6,
    )


def test_darcy_flux_constraint_emits_physics_diagnostics():
    torch.manual_seed(0)
    height = width = 8
    n_points = height * width

    constraint = DarcyFluxConstraint(
        spectral_backend="fft_pad",
        padding=2,
        padding_mode="reflect",
        enforce_boundary=False,
        boundary_value=0.0,
        shapelist=(height, width),
    )

    flux_pred = torch.randn(1, n_points, 2)
    permeability_physical = torch.full((1, n_points, 1), 4.0)
    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, height),
            torch.linspace(0.0, 1.0, width),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(1, n_points, 2)

    out = constraint(
        pred=flux_pred,
        fx=permeability_physical,
        coords=coords,
        return_aux=True,
    )

    assert isinstance(out, ConstraintOutput)
    assert "constraint/flux_div_abs_mean" in out.diagnostics
    assert "constraint/darcy_res_abs_mean" in out.diagnostics
    assert "constraint/grad_curl_abs_mean" in out.diagnostics
    assert "constraint/debug/fft_particular_div_padded_abs_mean" in out.diagnostics
    assert "constraint/debug/fft_correction_div_padded_abs_mean" in out.diagnostics
    assert "constraint/debug/fft_constrained_div_padded_abs_mean" in out.diagnostics
