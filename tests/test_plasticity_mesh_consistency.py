import torch

from omni_hc.constraints import (
    PlasticityEnvelopeConstraint,
    PlasticityIsotonicRegression,
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


def test_plasticity_envelope_uses_absolute_die_index_gap_profile():
    constraint = PlasticityEnvelopeConstraint(
        shapelist=(4, 3),
        x_left=0.0,
        x_right=-3.0,
        y_top=100.0,
        y_bottom=0.0,
        spacing_activation="exp",
        min_spacing=0.0,
        min_gap=0.0,
        max_gap=4.5,
        envelope_source="fx",
        die_speed=0.0,
    )
    pred = torch.zeros(1, 12, 4)
    target_gap = torch.tensor([0.45, 0.90, 1.35, 1.80])
    raw = pred.reshape(1, 4, 3, 4)
    raw[:, :, 0, 3] = torch.logit(target_gap / 4.5)
    die = torch.tensor([10.0, 20.0, 30.0, 40.0])
    fx = die.reshape(1, 4, 1).repeat(1, 1, 3).reshape(1, 12, 1)

    output = constraint(pred=pred, fx=fx, T=torch.zeros(1, 1), return_aux=True)
    field = output.pred.reshape(1, 4, 3, 4)

    expected_die = torch.tensor([[40.0, 30.0, 20.0, 10.0]])
    expected_gap = target_gap[None, :]
    expected_envelope = expected_die - expected_gap
    assert torch.allclose(output.aux["die_envelope_y"], expected_die)
    assert torch.allclose(output.aux["gap_profile"], expected_gap, atol=1.0e-6)
    assert torch.allclose(output.aux["gap"], expected_gap, atol=1.0e-6)
    assert torch.allclose(output.aux["envelope_y"], expected_envelope, atol=1.0e-6)
    assert torch.allclose(field[:, :, 0, 1], expected_envelope, atol=1.0e-6)


def test_plasticity_envelope_uses_separate_x_and_y_spacing_floors():
    constraint = PlasticityEnvelopeConstraint(
        shapelist=(4, 3),
        x_left=0.0,
        x_right=-3.0,
        y_top=10.0,
        y_bottom=0.0,
        spacing_activation="exp",
        min_spacing=0.0,
        min_x_spacing=0.2,
        min_y_spacing=0.05,
        min_gap=0.0,
        max_gap=0.0,
        envelope_source="constant",
        die_speed=0.0,
    )
    pred = torch.zeros(1, 12, 4)

    output = constraint(pred=pred, T=torch.zeros(1, 1), return_aux=True)

    assert torch.all(output.aux["dx"] >= 1.2 - 1.0e-6)
    assert torch.all(output.aux["dy"] >= 0.05 - 1.0e-6)
    assert torch.isclose(output.aux["dx"].min(), torch.tensor(1.2), atol=1.0e-6)


def test_plasticity_isotonic_regression_projects_coordinates_below_envelope():
    constraint = PlasticityIsotonicRegression(
        shapelist=(4, 3),
        x_left=0.0,
        x_right=-3.0,
        y_top=10.0,
        y_bottom=0.0,
        top_height=10.0,
        envelope_source="constant",
        die_speed=0.0,
        min_x_spacing=1.0e-4,
        min_y_spacing=1.0e-4,
        coordinate_mode="absolute",
    )
    raw = torch.zeros(1, 4, 3, 2)
    raw[..., 0] = torch.tensor(
        [
            [0.0, 0.2, 0.4],
            [-2.0, -2.1, -2.2],
            [-1.0, -1.2, -1.4],
            [-3.0, -2.8, -2.6],
        ]
    )
    raw[..., 1] = torch.tensor(
        [
            [12.0, 8.0, 3.0],
            [11.0, 9.0, 1.0],
            [12.0, 7.0, 2.0],
            [13.0, 8.0, 4.0],
        ]
    )
    pred = raw.reshape(1, 12, 2)

    output = constraint(pred=pred, T=torch.zeros(1, 1), return_aux=True)
    field = output.pred.reshape(1, 4, 3, 4)
    coords = field[..., :2]
    displacement = field[..., 2:4]
    dx = coords[:, :-1, :, 0] - coords[:, 1:, :, 0]
    dy = coords[:, :, :-1, 1] - coords[:, :, 1:, 1]

    assert torch.all(dx >= constraint.min_x_spacing - 1.0e-6)
    assert torch.all(dy >= constraint.min_y_spacing - 1.0e-6)
    assert torch.all(coords[:, :, 0, 1] <= output.aux["envelope_y"] + 1.0e-6)
    assert torch.allclose(coords[:, :, -1, 1], torch.zeros_like(coords[:, :, -1, 1]))
    assert torch.allclose(
        displacement,
        coords - constraint.material_grid.unsqueeze(0),
        atol=1.0e-6,
    )
    assert output.diagnostics["constraint/top_violation_count"].value == 0.0
    assert output.diagnostics["constraint/bottom_y_abs_error_max"].value == 0.0


def test_plasticity_isotonic_regression_reduces_spacing_when_cap_is_infeasible():
    constraint = PlasticityIsotonicRegression(
        shapelist=(3, 4),
        x_left=0.0,
        x_right=-2.0,
        y_top=0.0,
        y_bottom=0.0,
        top_height=0.0,
        envelope_source="constant",
        die_speed=0.0,
        min_x_spacing=0.0,
        min_y_spacing=1.0,
        collapse_spacing_threshold=1.0e-3,
        top_collapse_rows=2,
        coordinate_mode="absolute",
    )
    pred = torch.zeros(1, 12, 2)

    output = constraint(pred=pred, T=torch.zeros(1, 1), return_aux=True)
    field = output.pred.reshape(1, 3, 4, 4)
    coords = field[..., :2]

    assert torch.allclose(coords[:, :, 0, 1], torch.zeros_like(coords[:, :, 0, 1]))
    assert torch.allclose(coords[:, :, -1, 1], torch.zeros_like(coords[:, :, -1, 1]))
    assert torch.all(coords[:, :, :-1, 1] >= coords[:, :, 1:, 1])
    assert output.diagnostics["constraint/top_violation_count"].value == 0.0
    assert output.diagnostics["constraint/top_dy_min"].value == 0.0
    assert output.diagnostics["constraint/top_dy_collapse_fraction"].value == 1.0
    assert output.diagnostics["constraint/top_dy_collapse_worst_sample_fraction"].value == 1.0


def test_plasticity_isotonic_regression_displacement_mode_anchors_material_grid():
    constraint = PlasticityIsotonicRegression(
        shapelist=(4, 3),
        x_left=0.0,
        x_right=-3.0,
        y_top=2.0,
        y_bottom=0.0,
        top_height=2.0,
        envelope_source="constant",
        die_speed=0.0,
        min_x_spacing=1.0e-6,
        min_y_spacing=1.0e-6,
        coordinate_mode="displacement",
    )
    pred = torch.zeros(1, 12, 2)

    output = constraint(pred=pred, T=torch.zeros(1, 1), return_aux=True)
    field = output.pred.reshape(1, 4, 3, 4)
    coords = field[..., :2]
    displacement = field[..., 2:4]

    assert torch.allclose(coords, constraint.material_grid.unsqueeze(0), atol=1.0e-6)
    assert torch.allclose(displacement, torch.zeros_like(displacement), atol=1.0e-6)
    assert output.diagnostics["constraint/projection_correction_max"].value < 1.0e-5
    assert output.diagnostics["constraint/top_dy_collapse_fraction"].value == 0.0
