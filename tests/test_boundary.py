import torch

from omni_hc.constraints import (
    ConstraintOutput,
    DirichletBoundaryAnsatz,
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
