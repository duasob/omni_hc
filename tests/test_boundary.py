import torch

from omni_hc.constraints import (
    ConstraintOutput,
    DirichletBoundaryAnsatz,
    PipeInletParabolicAnsatz,
    PipeUxBoundaryAnsatz,
    StructuredWallDirichletAnsatz,
    boundary_stats,
    is_boundary_point,
    structured_wall_distance,
    structured_wall_mask,
    structured_wall_stats,
    unit_box_distance,
)


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


def test_unit_box_distance_is_zero_on_boundary():
    coords = torch.tensor(
        [[[0.0, 0.3], [1.0, 0.7], [0.4, 0.0], [0.6, 1.0], [0.5, 0.5]]]
    )

    dist = unit_box_distance(coords)

    assert torch.allclose(dist[0, :4], torch.zeros_like(dist[0, :4]))
    assert dist[0, 4, 0] > 0.0


def test_dirichlet_ansatz_enforces_zero_boundary():
    coords = torch.tensor(
        [[[0.0, 0.3], [1.0, 0.7], [0.4, 0.0], [0.6, 1.0], [0.5, 0.5]]]
    )
    latent_pred = torch.randn(1, 5, 1)
    constraint = DirichletBoundaryAnsatz(out_dim=1, boundary_value=0.0)

    out = constraint(pred=latent_pred, coords=coords)
    mask = is_boundary_point(coords)

    assert torch.allclose(out[mask], torch.zeros_like(out[mask]), atol=1e-8)


def test_boundary_stats_report_zero_for_exact_boundary_match():
    coords = torch.tensor(
        [[[0.0, 0.3], [1.0, 0.7], [0.4, 0.0], [0.6, 1.0], [0.5, 0.5]]]
    )
    field = torch.tensor([[[0.0], [0.0], [0.0], [0.0], [2.0]]])

    stats = boundary_stats(field, coords, target_value=0.0)

    assert stats["boundary_abs_mean"] == 0.0
    assert stats["boundary_abs_max"] == 0.0


def test_dirichlet_ansatz_enforces_physical_boundary_with_normalizer():
    coords = torch.tensor(
        [[[0.0, 0.3], [1.0, 0.7], [0.4, 0.0], [0.6, 1.0], [0.5, 0.5]]]
    )
    latent_pred = torch.randn(1, 5, 1)
    constraint = DirichletBoundaryAnsatz(out_dim=1, boundary_value=0.0)
    normalizer = AffineNormalizer(mean=2.0, std=4.0)
    constraint.set_target_normalizer(normalizer)

    out_encoded = constraint(pred=latent_pred, coords=coords)
    out_physical = normalizer.decode(out_encoded)
    mask = is_boundary_point(coords)

    assert torch.allclose(
        out_physical[mask], torch.zeros_like(out_physical[mask]), atol=1e-8
    )


def test_dirichlet_ansatz_emits_boundary_diagnostics():
    coords = torch.tensor(
        [[[0.0, 0.3], [1.0, 0.7], [0.4, 0.0], [0.6, 1.0], [0.5, 0.5]]]
    )
    latent_pred = torch.randn(1, 5, 1)
    constraint = DirichletBoundaryAnsatz(out_dim=1, boundary_value=0.0)

    out = constraint(pred=latent_pred, coords=coords, return_aux=True)

    assert isinstance(out, ConstraintOutput)
    assert "constraint/boundary_abs_mean" in out.diagnostics
    assert "constraint/boundary_abs_max" in out.diagnostics


def test_structured_wall_distance_is_zero_on_transverse_walls():
    distance = structured_wall_distance((4, 5), transverse_axis=1)
    distance_grid = distance.reshape(4, 5)

    assert torch.allclose(distance_grid[:, 0], torch.zeros(4))
    assert torch.allclose(distance_grid[:, -1], torch.zeros(4))
    assert torch.all(distance_grid[:, 1:-1] > 0.0)


def test_structured_wall_mask_selects_only_transverse_walls():
    mask = structured_wall_mask((4, 5), transverse_axis=1).reshape(4, 5)

    assert mask.sum().item() == 8
    assert torch.all(mask[:, 0])
    assert torch.all(mask[:, -1])
    assert not torch.any(mask[:, 1:-1])


def test_structured_wall_ansatz_enforces_zero_walls_on_scalar_field():
    height, width = 4, 5
    pred = torch.randn(2, height * width, 1)
    constraint = StructuredWallDirichletAnsatz(
        out_dim=1,
        grid_shape=(height, width),
        boundary_value=0.0,
    )

    out = constraint(pred=pred)
    out_grid = out.reshape(2, height, width, 1)

    assert torch.allclose(out_grid[:, :, 0], torch.zeros_like(out_grid[:, :, 0]))
    assert torch.allclose(out_grid[:, :, -1], torch.zeros_like(out_grid[:, :, -1]))


def test_structured_wall_ansatz_enforces_physical_zero_with_normalizer():
    height, width = 4, 5
    pred = torch.randn(2, height * width, 1)
    constraint = StructuredWallDirichletAnsatz(
        out_dim=1,
        grid_shape=(height, width),
        boundary_value=0.0,
    )
    normalizer = AffineNormalizer(mean=2.0, std=4.0)
    constraint.set_target_normalizer(normalizer)

    out_encoded = constraint(pred=pred)
    out_physical = normalizer.decode(out_encoded).reshape(2, height, width, 1)

    assert torch.allclose(
        out_physical[:, :, 0],
        torch.zeros_like(out_physical[:, :, 0]),
        atol=1e-8,
    )
    assert torch.allclose(
        out_physical[:, :, -1],
        torch.zeros_like(out_physical[:, :, -1]),
        atol=1e-8,
    )


def test_structured_wall_ansatz_can_target_selected_channels():
    height, width = 4, 5
    pred = torch.randn(2, height * width, 3)
    constraint = StructuredWallDirichletAnsatz(
        out_dim=3,
        grid_shape=(height, width),
        boundary_value=0.0,
        channel_indices=[0, 1],
    )

    out = constraint(pred=pred)
    out_grid = out.reshape(2, height, width, 3)
    pred_grid = pred.reshape(2, height, width, 3)

    assert torch.allclose(out_grid[:, :, 0, :2], torch.zeros_like(out_grid[:, :, 0, :2]))
    assert torch.allclose(
        out_grid[:, :, -1, :2],
        torch.zeros_like(out_grid[:, :, -1, :2]),
    )
    assert torch.allclose(out_grid[..., 2], pred_grid[..., 2])


def test_structured_wall_stats_report_zero_for_exact_wall_match():
    height, width = 4, 5
    field = torch.randn(2, height, width, 1)
    field[:, :, 0] = 0.0
    field[:, :, -1] = 0.0

    stats = structured_wall_stats(field.reshape(2, height * width, 1), (height, width))

    assert stats["wall_abs_mean"] == 0.0
    assert stats["wall_abs_max"] == 0.0


def test_structured_wall_ansatz_emits_wall_diagnostics():
    height, width = 4, 5
    pred = torch.randn(2, height * width, 1)
    constraint = StructuredWallDirichletAnsatz(
        out_dim=1,
        grid_shape=(height, width),
        boundary_value=0.0,
    )

    out = constraint(pred=pred, return_aux=True)

    assert isinstance(out, ConstraintOutput)
    assert "constraint/wall_abs_mean" in out.diagnostics
    assert "constraint/wall_abs_max" in out.diagnostics
    assert "constraint/wall_base_abs_mean" in out.diagnostics
    assert "constraint/wall_base_lower_abs_mean" in out.diagnostics
    assert "constraint/wall_base_upper_abs_mean" in out.diagnostics
    assert "constraint/interior_abs_delta_mean" in out.diagnostics


def _structured_coords(height: int, width: int):
    x = torch.linspace(0.0, 10.0, height)
    y = torch.linspace(-2.0, 2.0, width)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2)


def test_pipe_inlet_parabolic_ansatz_enforces_inlet_profile():
    height, width = 4, 5
    pred = torch.randn(2, height * width, 1)
    coords = _structured_coords(height, width).expand(2, -1, -1)
    constraint = PipeInletParabolicAnsatz(
        out_dim=1,
        grid_shape=(height, width),
        amplitude=0.25,
        decay_power=4.0,
    )

    out = constraint(pred=pred, coords=coords)
    out_grid = out.reshape(2, height, width, 1)
    t = torch.linspace(0.0, 1.0, width)
    expected = 0.25 * 4.0 * t * (1.0 - t)

    assert torch.allclose(out_grid[:, 0, :, 0], expected.expand(2, -1), atol=1e-8)


def test_pipe_inlet_parabolic_ansatz_uses_normalized_inputs_and_targets():
    height, width = 4, 5
    pred = torch.randn(2, height * width, 1)
    coords = _structured_coords(height, width).expand(2, -1, -1)
    input_normalizer = ChannelAffineNormalizer(mean=[5.0, -1.0], std=[2.0, 3.0])
    target_normalizer = AffineNormalizer(mean=2.0, std=4.0)
    encoded_coords = input_normalizer.encode(coords)
    constraint = PipeInletParabolicAnsatz(
        out_dim=1,
        grid_shape=(height, width),
        amplitude=0.25,
        decay_power=4.0,
    )
    constraint.set_input_normalizer(input_normalizer)
    constraint.set_target_normalizer(target_normalizer)

    out_encoded = constraint(pred=pred, coords=encoded_coords)
    out_physical = target_normalizer.decode(out_encoded).reshape(2, height, width, 1)
    t = torch.linspace(0.0, 1.0, width)
    expected = 0.25 * 4.0 * t * (1.0 - t)

    assert torch.allclose(
        out_physical[:, 0, :, 0],
        expected.expand(2, -1),
        atol=1e-8,
    )


def test_pipe_inlet_parabolic_ansatz_can_target_selected_channels():
    height, width = 4, 5
    pred = torch.randn(2, height * width, 3)
    coords = _structured_coords(height, width).expand(2, -1, -1)
    constraint = PipeInletParabolicAnsatz(
        out_dim=3,
        grid_shape=(height, width),
        amplitude=0.25,
        channel_indices=[0],
    )

    out = constraint(pred=pred, coords=coords)
    out_grid = out.reshape(2, height, width, 3)
    pred_grid = pred.reshape(2, height, width, 3)
    t = torch.linspace(0.0, 1.0, width)
    expected = 0.25 * 4.0 * t * (1.0 - t)

    assert torch.allclose(out_grid[:, 0, :, 0], expected.expand(2, -1), atol=1e-8)
    assert torch.allclose(out_grid[..., 1:], pred_grid[..., 1:])


def test_pipe_inlet_parabolic_ansatz_emits_inlet_diagnostics():
    height, width = 4, 5
    pred = torch.randn(2, height * width, 1)
    coords = _structured_coords(height, width).expand(2, -1, -1)
    constraint = PipeInletParabolicAnsatz(
        out_dim=1,
        grid_shape=(height, width),
        amplitude=0.25,
    )

    out = constraint(pred=pred, coords=coords, return_aux=True)

    assert isinstance(out, ConstraintOutput)
    assert "constraint/inlet_abs_mean" in out.diagnostics
    assert "constraint/inlet_abs_max" in out.diagnostics
    assert "constraint/inlet_base_abs_mean" in out.diagnostics
    assert "constraint/inlet_alpha_mean" in out.diagnostics
    assert "constraint/inlet_decay_power" in out.diagnostics
    assert torch.allclose(out.diagnostics["constraint/inlet_abs_max"].value, torch.tensor(0.0))


def test_pipe_ux_boundary_ansatz_enforces_inlet_and_walls():
    height, width = 4, 5
    pred = torch.randn(2, height * width, 1)
    coords = _structured_coords(height, width).expand(2, -1, -1)
    constraint = PipeUxBoundaryAnsatz(
        out_dim=1,
        grid_shape=(height, width),
        amplitude=0.25,
        inlet_decay_power=4.0,
    )

    out = constraint(pred=pred, coords=coords)
    out_grid = out.reshape(2, height, width, 1)
    t = torch.linspace(0.0, 1.0, width)
    expected_inlet = 0.25 * 4.0 * t * (1.0 - t)

    assert torch.allclose(
        out_grid[:, 0, :, 0],
        expected_inlet.expand(2, -1),
        atol=1e-8,
    )
    assert torch.allclose(out_grid[:, :, 0, 0], torch.zeros(2, height), atol=1e-8)
    assert torch.allclose(out_grid[:, :, -1, 0], torch.zeros(2, height), atol=1e-8)


def test_pipe_ux_boundary_ansatz_uses_normalized_inputs_and_targets():
    height, width = 4, 5
    pred = torch.randn(2, height * width, 1)
    coords = _structured_coords(height, width).expand(2, -1, -1)
    input_normalizer = ChannelAffineNormalizer(mean=[5.0, -1.0], std=[2.0, 3.0])
    target_normalizer = AffineNormalizer(mean=2.0, std=4.0)
    encoded_coords = input_normalizer.encode(coords)
    constraint = PipeUxBoundaryAnsatz(
        out_dim=1,
        grid_shape=(height, width),
        amplitude=0.25,
    )
    constraint.set_input_normalizer(input_normalizer)
    constraint.set_target_normalizer(target_normalizer)

    out_encoded = constraint(pred=pred, coords=encoded_coords)
    out_physical = target_normalizer.decode(out_encoded).reshape(2, height, width, 1)
    t = torch.linspace(0.0, 1.0, width)
    expected_inlet = 0.25 * 4.0 * t * (1.0 - t)

    assert torch.allclose(
        out_physical[:, 0, :, 0],
        expected_inlet.expand(2, -1),
        atol=1e-8,
    )
    assert torch.allclose(out_physical[:, :, 0, 0], torch.zeros(2, height), atol=1e-8)
    assert torch.allclose(out_physical[:, :, -1, 0], torch.zeros(2, height), atol=1e-8)


def test_pipe_ux_boundary_ansatz_emits_combined_diagnostics():
    height, width = 4, 5
    pred = torch.randn(2, height * width, 1)
    coords = _structured_coords(height, width).expand(2, -1, -1)
    constraint = PipeUxBoundaryAnsatz(
        out_dim=1,
        grid_shape=(height, width),
        amplitude=0.25,
    )

    out = constraint(pred=pred, coords=coords, return_aux=True)

    assert isinstance(out, ConstraintOutput)
    assert "constraint/inlet_abs_max" in out.diagnostics
    assert "constraint/wall_abs_max" in out.diagnostics
    assert "constraint/boundary_distance_min" in out.diagnostics
    assert "constraint/boundary_distance_max" in out.diagnostics
    assert torch.allclose(out.diagnostics["constraint/inlet_abs_max"].value, torch.tensor(0.0))
    assert out.diagnostics["constraint/wall_abs_max"].value == 0.0
