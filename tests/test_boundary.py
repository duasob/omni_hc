import torch

from omni_hc.constraints import (
    DirichletBoundaryAnsatz,
    boundary_stats,
    is_boundary_point,
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
