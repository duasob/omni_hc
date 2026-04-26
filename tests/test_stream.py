import torch

from omni_hc.constraints import (
    PipeStreamFunctionBoundaryAnsatz,
    PipeStreamFunctionUxConstraint,
)
from omni_hc.constraints.utils.stream_ops import (
    finite_volume_divergence_curvilinear,
    stream_velocity_from_psi_cartesian_spectral,
    stream_velocity_from_psi_curvilinear,
)
from omni_hc.constraints.utils.spectral import reshape_channels_last_to_grid


class AffineNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


class ChannelAffineNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).reshape(1, 1, -1)
        self.std = torch.tensor(std, dtype=torch.float32).reshape(1, 1, -1)

    def encode(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def decode(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


def _structured_coords(height: int, width: int):
    x = torch.linspace(0.0, 10.0, height)
    y = torch.linspace(-2.0, 2.0, width)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2)


def test_stream_velocity_from_psi_cartesian_spectral_matches_periodic_stream():
    height = width = 16
    y = torch.arange(height, dtype=torch.float32) / float(height)
    x = torch.arange(width, dtype=torch.float32) / float(width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    psi = (torch.sin(2.0 * torch.pi * xx) * torch.sin(2.0 * torch.pi * yy)).reshape(
        1, 1, height, width
    )

    velocity = stream_velocity_from_psi_cartesian_spectral(
        psi,
        dy=1.0 / height,
        dx=1.0 / width,
    )

    ux_expected = (
        2.0
        * torch.pi
        * torch.sin(2.0 * torch.pi * xx)
        * torch.cos(2.0 * torch.pi * yy)
    ).reshape(1, 1, height, width)
    uy_expected = (
        -2.0
        * torch.pi
        * torch.cos(2.0 * torch.pi * xx)
        * torch.sin(2.0 * torch.pi * yy)
    ).reshape(1, 1, height, width)

    assert torch.allclose(velocity[:, 0:1], ux_expected, atol=1e-3)
    assert torch.allclose(velocity[:, 1:2], uy_expected, atol=1e-3)


def test_stream_velocity_from_psi_curvilinear_recovers_cartesian_stream():
    height, width = 8, 9
    coords = _structured_coords(height, width)
    coords_grid = reshape_channels_last_to_grid(coords, shapelist=(height, width))
    x = coords_grid[:, 0:1]
    y = coords_grid[:, 1:2]
    psi = x * y

    velocity, jac = stream_velocity_from_psi_curvilinear(psi, coords_grid)

    assert torch.allclose(velocity[:, 0:1], x, atol=1e-5)
    assert torch.allclose(velocity[:, 1:2], -y, atol=1e-5)
    assert torch.all(jac > 0.0)


def test_finite_volume_divergence_curvilinear_is_small_for_stream_velocity():
    height, width = 8, 9
    coords = _structured_coords(height, width)
    coords_grid = reshape_channels_last_to_grid(coords, shapelist=(height, width))
    x = coords_grid[:, 0:1]
    y = coords_grid[:, 1:2]
    psi = x * y
    velocity, _ = stream_velocity_from_psi_curvilinear(psi, coords_grid)

    div = finite_volume_divergence_curvilinear(velocity, coords_grid)

    assert float(div.abs().mean()) < 1e-5


def test_pipe_stream_function_constraint_returns_ux_from_stream_function():
    height, width = 8, 9
    coords = _structured_coords(height, width)
    coords_grid = reshape_channels_last_to_grid(coords, shapelist=(height, width))
    x = coords_grid[:, 0:1]
    y = coords_grid[:, 1:2]
    psi = x * y
    pred = psi.reshape(1, 1, height * width).transpose(1, 2)

    constraint = PipeStreamFunctionUxConstraint(shapelist=(height, width))
    out = constraint(pred=pred, coords=coords)
    ux = out.reshape(1, height, width, 1)

    expected = x.reshape(1, height, width, 1)
    assert torch.allclose(ux, expected, atol=1e-5)


def test_pipe_stream_function_constraint_supports_normalized_inputs_and_targets():
    height, width = 8, 9
    coords = _structured_coords(height, width)
    coords_grid = reshape_channels_last_to_grid(coords, shapelist=(height, width))
    x = coords_grid[:, 0:1]
    y = coords_grid[:, 1:2]
    psi = x * y
    pred = psi.reshape(1, 1, height * width).transpose(1, 2)

    input_normalizer = ChannelAffineNormalizer(mean=[5.0, -1.0], std=[2.0, 3.0])
    target_normalizer = AffineNormalizer(mean=2.0, std=4.0)
    encoded_coords = input_normalizer.encode(coords)

    constraint = PipeStreamFunctionUxConstraint(shapelist=(height, width))
    constraint.set_input_normalizer(input_normalizer)
    constraint.set_target_normalizer(target_normalizer)
    out_encoded = constraint(pred=pred, coords=encoded_coords)
    out_physical = target_normalizer.decode(out_encoded).reshape(1, height, width, 1)

    expected = x.reshape(1, height, width, 1)
    assert torch.allclose(out_physical, expected, atol=1e-5)


def test_pipe_stream_function_boundary_constraint_preserves_psi_bc_and_mask():
    height, width = 8, 9
    coords = _structured_coords(height, width)
    pred = torch.ones(1, height * width, 1)

    constraint = PipeStreamFunctionBoundaryAnsatz(
        shapelist=(height, width),
        amplitude=0.25,
        decay_power=2.0,
    )
    out = constraint(pred=pred, coords=coords, return_aux=True)
    mask = out.aux["stream_mask"].reshape(1, height, width, 1)

    assert torch.allclose(mask[:, 0], torch.zeros_like(mask[:, 0]), atol=1e-6)
    assert torch.allclose(mask[:, :, 0], torch.zeros_like(mask[:, :, 0]), atol=1e-6)
    assert torch.allclose(mask[:, :, -1], torch.zeros_like(mask[:, :, -1]), atol=1e-6)
    assert float(out.diagnostics["constraint/stream_inlet_abs_max"].value) < 6.0e-2
    assert float(out.diagnostics["constraint/stream_wall_ux_abs_max"].value) < 9.0e-2
    assert "stream_psi_bc" in out.aux
    assert "stream_mask" in out.aux


def test_pipe_stream_function_constraint_emits_stream_diagnostics():
    height, width = 8, 9
    coords = _structured_coords(height, width)
    coords_grid = reshape_channels_last_to_grid(coords, shapelist=(height, width))
    x = coords_grid[:, 0:1]
    y = coords_grid[:, 1:2]
    psi = x * y
    pred = psi.reshape(1, 1, height * width).transpose(1, 2)

    constraint = PipeStreamFunctionUxConstraint(shapelist=(height, width))
    out = constraint(pred=pred, coords=coords, return_aux=True)

    assert "constraint/stream_div_abs_mean" in out.diagnostics
    assert "constraint/stream_div_abs_max" in out.diagnostics
    assert "constraint/stream_uy_abs_mean" in out.diagnostics
    assert "constraint/stream_psi_std" in out.diagnostics
    assert "stream_psi" in out.aux
    assert "stream_uy" in out.aux
    assert "stream_div" in out.aux


def test_pipe_stream_function_boundary_constraint_emits_boundary_diagnostics():
    height, width = 8, 9
    coords = _structured_coords(height, width)
    pred = torch.zeros(1, height * width, 1)

    constraint = PipeStreamFunctionBoundaryAnsatz(shapelist=(height, width))
    out = constraint(pred=pred, coords=coords, return_aux=True)

    assert "constraint/stream_div_abs_mean" in out.diagnostics
    assert "constraint/stream_inlet_abs_max" in out.diagnostics
    assert "constraint/stream_wall_ux_abs_max" in out.diagnostics
    assert "constraint/stream_mask_max" in out.diagnostics
    assert "stream_psi" in out.aux
    assert "stream_uy" in out.aux
    assert "stream_div" in out.aux
    assert "stream_psi_bc" in out.aux
    assert "stream_mask" in out.aux
