import torch

from omni_hc.constraints import (
    PlasticityEnvelopeConstraint,
    PlasticityMeshConsistencyConstraint,
)


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


def test_plasticity_envelope_reconstructs_bottom_and_top_clearance():
    constraint = PlasticityEnvelopeConstraint(
        shapelist=(4, 3),
        x_left=0.35,
        x_right=-1.15,
        y_top=1.0,
        y_bottom=0.0,
        min_spacing=1.0e-4,
        min_gap=1.0e-4,
        envelope_source="fx",
        die_speed=0.2,
        time_duration=1.0,
    )
    pred = torch.zeros(2, 12, 4)
    fx = torch.ones(2, 12, 1)
    time = torch.ones(2, 1)

    output = constraint(pred=pred, fx=fx, T=time, return_aux=True)
    field = output.pred.reshape(2, 4, 3, 4)
    coords = field[..., :2]
    displacement = field[..., 2:4]

    dx = coords[:, 1:, :, 0] - coords[:, :-1, :, 0]
    dy = coords[:, :, 1:, 1] - coords[:, :, :-1, 1]

    assert torch.all(dx < 0.0)
    assert torch.all(dy < 0.0)
    assert torch.allclose(
        coords[:, :, -1, 1],
        torch.zeros_like(coords[:, :, -1, 1]),
    )
    assert torch.all(output.aux["top_clearance"] >= 0.0)
    assert torch.allclose(
        output.aux["envelope_x"],
        coords[:, :, 0, 0],
    )
    assert torch.allclose(output.aux["envelope_y"], torch.full((2, 4), 0.8))
    assert torch.allclose(
        displacement,
        coords - constraint.material_grid.unsqueeze(0),
        atol=1.0e-6,
    )
    assert output.diagnostics["constraint/top_violation_count"].value == 0.0


def test_plasticity_envelope_aligns_input_die_to_physical_x_order():
    constraint = PlasticityEnvelopeConstraint(
        shapelist=(4, 3),
        x_left=0.0,
        x_right=-3.0,
        y_top=40.0,
        y_bottom=0.0,
        spacing_activation="exp",
        min_spacing=0.0,
        min_gap=0.0,
        envelope_source="fx",
        die_speed=0.0,
    )
    pred = torch.zeros(1, 12, 4)
    die = torch.tensor([10.0, 20.0, 30.0, 40.0])
    fx = die.reshape(1, 4, 1).repeat(1, 1, 3).reshape(1, 12, 1)

    output = constraint(pred=pred, fx=fx, T=torch.zeros(1, 1), return_aux=True)

    assert torch.allclose(
        output.aux["envelope_y"],
        torch.tensor([[40.0, 30.0, 20.0, 10.0]]),
    )


def test_plasticity_envelope_samples_die_at_predicted_top_x():
    constraint = PlasticityEnvelopeConstraint(
        shapelist=(4, 3),
        x_left=0.0,
        x_right=-3.0,
        y_top=30.0,
        y_bottom=0.0,
        spacing_activation="exp",
        min_spacing=0.0,
        min_gap=0.0,
        envelope_source="fx",
        die_speed=0.0,
    )
    pred = torch.zeros(1, 12, 4)
    raw = pred.reshape(1, 4, 3, 4)
    raw[:, 0, :, 0] = -0.5
    raw[:, :-1, :, 1] = 0.0
    die = torch.tensor([0.0, 10.0, 20.0, 30.0])
    fx = die.reshape(1, 4, 1).repeat(1, 1, 3).reshape(1, 12, 1)

    output = constraint(pred=pred, fx=fx, T=torch.zeros(1, 1), return_aux=True)

    assert torch.allclose(
        output.aux["envelope_x"],
        torch.tensor([[-0.5, -1.5, -2.5, -3.5]]),
    )
    assert torch.allclose(
        output.aux["envelope_y"],
        torch.tensor([[25.0, 15.0, 5.0, 0.0]]),
    )


def test_plasticity_envelope_caps_sampled_die_at_top_height():
    constraint = PlasticityEnvelopeConstraint(
        shapelist=(4, 3),
        x_left=0.0,
        x_right=-3.0,
        y_top=100.0,
        top_height=15.0,
        y_bottom=0.0,
        spacing_activation="exp",
        min_spacing=0.0,
        min_gap=0.0,
        envelope_source="fx",
        die_speed=0.0,
    )
    pred = torch.zeros(1, 12, 4)
    die = torch.tensor([10.0, 20.0, 30.0, 40.0])
    fx = die.reshape(1, 4, 1).repeat(1, 1, 3).reshape(1, 12, 1)

    output = constraint(pred=pred, fx=fx, T=torch.zeros(1, 1), return_aux=True)

    assert torch.allclose(
        output.aux["envelope_y"],
        torch.tensor([[15.0, 15.0, 15.0, 10.0]]),
    )


def test_plasticity_envelope_accepts_per_index_top_height():
    constraint = PlasticityEnvelopeConstraint(
        shapelist=(4, 3),
        x_left=0.0,
        x_right=-3.0,
        y_top=100.0,
        top_height=[35.0, 25.0, 15.0, 5.0],
        y_bottom=0.0,
        spacing_activation="exp",
        min_spacing=0.0,
        min_gap=0.0,
        envelope_source="fx",
        die_speed=0.0,
    )
    pred = torch.zeros(1, 12, 4)
    die = torch.tensor([10.0, 20.0, 30.0, 40.0])
    fx = die.reshape(1, 4, 1).repeat(1, 1, 3).reshape(1, 12, 1)

    output = constraint(pred=pred, fx=fx, T=torch.zeros(1, 1), return_aux=True)

    assert torch.allclose(
        output.aux["envelope_y"],
        torch.tensor([[35.0, 25.0, 15.0, 5.0]]),
    )


def test_plasticity_envelope_caps_moved_die_profile_per_index():
    constraint = PlasticityEnvelopeConstraint(
        shapelist=(4, 3),
        x_left=0.0,
        x_right=-3.0,
        y_top=100.0,
        top_height=15.0,
        y_bottom=0.0,
        spacing_activation="exp",
        min_spacing=0.0,
        min_gap=0.0,
        envelope_source="fx",
        die_speed=10.0,
        time_duration=1.0,
    )
    pred = torch.zeros(1, 12, 4)
    die = torch.tensor([10.0, 20.0, 30.0, 40.0])
    fx = die.reshape(1, 4, 1).repeat(1, 1, 3).reshape(1, 12, 1)

    output = constraint(pred=pred, fx=fx, T=torch.ones(1, 1), return_aux=True)

    assert torch.allclose(
        output.aux["envelope_y"],
        torch.tensor([[15.0, 15.0, 10.0, 0.0]]),
    )
