import torch

from omni_hc.constraints import PipeStreamFunctionBoundaryAnsatz, PipeUxBoundaryAnsatz
from omni_hc.diagnostics import infer_boundary_ansatz_maps


def test_direct_boundary_maps_are_inferred_from_constraint_outputs():
    height, width = 5, 7
    coords = _structured_coords(height, width)
    constraint = PipeUxBoundaryAnsatz(out_dim=1, grid_shape=(height, width))

    maps = infer_boundary_ansatz_maps(
        constraint,
        pred_shape=(1, height * width, 1),
        grid_shape=(height, width),
        coords=coords,
    )

    assert maps.space == "output"
    assert maps.g.shape == (height, width, 1)
    assert maps.l.shape == (height, width, 1)
    assert torch.allclose(maps.l[:, 0, 0], torch.zeros(height), atol=1e-8)
    assert torch.allclose(maps.l[:, -1, 0], torch.zeros(height), atol=1e-8)
    assert torch.allclose(maps.l[0, :, 0], torch.zeros(width), atol=1e-8)


def test_stream_boundary_maps_use_latent_stream_aux_fields():
    height, width = 8, 9
    coords = _structured_coords(height, width)
    constraint = PipeStreamFunctionBoundaryAnsatz(shapelist=(height, width))

    maps = infer_boundary_ansatz_maps(
        constraint,
        pred_shape=(1, height * width, 1),
        grid_shape=(height, width),
        coords=coords,
    )

    assert maps.space == "stream_function"
    assert maps.g.shape == (height, width, 1)
    assert maps.l.shape == (height, width, 1)
    assert torch.allclose(maps.l[:, 0, 0], torch.zeros(height), atol=1e-8)
    assert torch.allclose(maps.l[:, -1, 0], torch.zeros(height), atol=1e-8)
    assert torch.allclose(maps.l[0, :, 0], torch.zeros(width), atol=1e-8)


def _structured_coords(height: int, width: int):
    x = torch.linspace(0.0, 10.0, height)
    y = torch.linspace(-2.0, 2.0, width)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2)
