import torch

from omni_hc.constraints import PlasticityMeshConsistencyConstraint


def test_plasticity_mesh_consistency_reconstructs_ordered_coordinates():
    constraint = PlasticityMeshConsistencyConstraint(
        shapelist=(4, 3),
        x_left=0.35,
        x_right=-1.15,
        y_top=0.9,
        y_bottom=-0.1,
        min_spacing=1.0e-4,
    )
    pred = torch.zeros(2, 12, 3)

    out = constraint(pred=pred)
    field = out.reshape(2, 4, 3, 4)
    coords = field[..., :2]
    displacement = field[..., 2:4]

    dx = coords[:, 1:, :, 0] - coords[:, :-1, :, 0]
    dy = coords[:, :, 1:, 1] - coords[:, :, :-1, 1]

    assert torch.all(dx < 0.0)
    assert torch.all(dy < 0.0)
    assert torch.allclose(
        coords[:, :, -1, 1],
        torch.full_like(coords[:, :, -1, 1], -0.1),
    )
    assert torch.allclose(
        displacement,
        coords - constraint.material_grid.unsqueeze(0),
        atol=1.0e-6,
    )


def test_plasticity_mesh_consistency_aux_reports_invariant_diagnostics():
    constraint = PlasticityMeshConsistencyConstraint(shapelist=(4, 3))
    pred = torch.zeros(1, 12, 3)

    output = constraint(pred=pred, return_aux=True)

    assert output.pred.shape == (1, 12, 4)
    assert output.aux["dx"].shape == (1, 3, 3)
    assert output.aux["dy"].shape == (1, 4, 2)
    assert output.diagnostics["constraint/min_dx"].value > 0.0
    assert output.diagnostics["constraint/min_dy"].value > 0.0
    assert output.diagnostics["constraint/bottom_y_abs_error_max"].value == 0.0
    assert output.diagnostics["constraint/axis_order_margin_min"].value > 0.0
    assert output.diagnostics["constraint/min_oriented_cell_area"].value > 0.0
    assert output.diagnostics["constraint/flipped_cell_count_worst"].value == 0.0
    assert output.diagnostics["constraint/flipped_cell_fraction_worst"].value == 0.0
