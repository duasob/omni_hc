import torch

from omni_hc.constraints import ConstraintDiagnostic
from omni_hc.constraints.metrics.darcy import compute as compute_darcy_metrics
from omni_hc.constraints.metrics.plasticity import compute as compute_plasticity_metrics
from omni_hc.training.benchmark_diagnostics import make_benchmark_diagnostic_fn
from omni_hc.training.common import MetricAccumulator


def test_metric_accumulator_respects_mean_and_max_reductions():
    metrics = MetricAccumulator()

    metrics.update(
        {
            "constraint/a": ConstraintDiagnostic(value=2.0, reduce="mean"),
            "constraint/b": ConstraintDiagnostic(value=1.0, reduce="max"),
        },
        weight=2,
    )
    metrics.update(
        {
            "constraint/a": ConstraintDiagnostic(value=4.0, reduce="mean"),
            "constraint/b": ConstraintDiagnostic(value=3.0, reduce="max"),
        },
        weight=1,
    )

    reduced = metrics.compute()

    assert reduced["constraint/a"] == (2.0 * 2.0 + 4.0) / 3.0
    assert reduced["constraint/b"] == 3.0


def test_darcy_metric_reports_pressure_induced_residual_keys():
    import torch

    h = w = 5
    x = torch.linspace(0.0, 1.0, w)
    y = torch.linspace(0.0, 1.0, h)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    pressure = 0.25 * (xx * (1.0 - xx) + yy * (1.0 - yy))
    pred = pressure.reshape(1, h * w, 1)
    permeability = torch.ones_like(pred)

    metrics = compute_darcy_metrics(
        pred,
        {"x": permeability},
        {"shapelist": (h, w)},
    )

    assert "constraint/darcy_res_abs_mean" in metrics
    assert "constraint/darcy_res_abs_max" in metrics
    assert "constraint/darcy_res_rmse" in metrics
    assert "constraint/flux_rmse" in metrics


def test_plasticity_metric_reports_zero_for_monotone_grid():
    pred = _plasticity_pred_from_coords(
        [
            [[1.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )

    metrics = compute_plasticity_metrics(pred, {}, {"shapelist": (2, 2), "t_out": 1, "out_dim": 4})

    assert metrics["constraint/x_order_violation_count"].value == 0
    assert metrics["constraint/y_order_violation_count"].value == 0
    assert metrics["constraint/axis_order_violation_count"].value == 0
    assert metrics["constraint/x_order_margin_min"].value == 1
    assert metrics["constraint/y_order_margin_min"].value == 1
    assert metrics["constraint/neg_spacing_count"].value == 0
    assert metrics["constraint/neg_spacing_worst_sample_fraction"].value == 0
    assert metrics["constraint/neg_spacing_fraction"].value == 0
    assert metrics["constraint/bottom_boundary_violation_count"].value == 0
    assert metrics["constraint/bottom_envelope_violation_max"].value == 0


def test_plasticity_metric_reports_fraction_for_inverted_grid():
    pred = _plasticity_pred_from_coords(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ]
    )

    metrics = compute_plasticity_metrics(pred, {}, {"shapelist": (2, 2), "t_out": 1, "out_dim": 4})

    assert metrics["constraint/x_order_violation_count"].value == 2
    assert metrics["constraint/y_order_violation_count"].value == 2
    assert metrics["constraint/axis_order_violation_count"].value == 4
    assert metrics["constraint/x_order_violation_fraction"].value == 1
    assert metrics["constraint/y_order_violation_fraction"].value == 1
    assert metrics["constraint/axis_order_violation_fraction"].value == 1
    assert metrics["constraint/neg_spacing_count"].value == 4
    assert metrics["constraint/neg_spacing_worst_sample_fraction"].value == 1
    assert metrics["constraint/neg_spacing_fraction"].value == 1


def test_plasticity_metric_reports_top_and_bottom_envelope_violations():
    import torch

    pred = _plasticity_pred_from_coords(
        [
            [[1.0, 1.2], [1.0, 0.2]],
            [[0.0, 1.1], [0.0, 0.1]],
        ]
    )
    target = _plasticity_pred_from_coords(
        [
            [[1.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )
    fx = torch.tensor([[[1.0], [1.0], [1.0], [1.0]]])
    time = torch.zeros(1, 1)

    metrics = compute_plasticity_metrics(
        pred,
        {"target": target, "x": fx, "time": time},
        {
            "shapelist": (2, 2),
            "t_out": 1,
            "out_dim": 4,
            "die_speed": 0.0,
            "top_height": 1.0,
            "top_envelope_tolerance": 0.0,
        },
    )

    assert metrics["constraint/top_envelope_violation_count"].value == 2
    assert metrics["constraint/top_envelope_violation_mean"].value > 0
    assert metrics["constraint/top_envelope_violation_max"].value == 0.20000004768371582
    assert metrics["constraint/bottom_boundary_violation_count"].value == 2
    assert metrics["constraint/bottom_boundary_violation_fraction"].value == 1
    assert metrics["constraint/bottom_boundary_abs_error_mean"].value == 0.15000000596046448
    assert metrics["constraint/bottom_boundary_abs_error_max"].value == 0.20000000298023224
    assert metrics["constraint/bottom_envelope_violation_mean"].value == 0.15000000596046448
    assert metrics["constraint/bottom_envelope_violation_max"].value == 0.20000000298023224


def test_plasticity_metric_bottom_boundary_tolerance_ignores_roundoff():
    pred = _plasticity_pred_from_coords(
        [
            [[1.0, 1.0], [1.0, 0.000007]],
            [[0.0, 1.0], [0.0, -0.000007]],
        ]
    )
    target = _plasticity_pred_from_coords(
        [
            [[1.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )

    metrics = compute_plasticity_metrics(
        pred,
        {"target": target},
        {
            "shapelist": (2, 2),
            "t_out": 1,
            "out_dim": 4,
            "bottom_boundary_tolerance": 1.0e-4,
        },
    )

    assert metrics["constraint/bottom_boundary_violation_count"].value == 0
    assert metrics["constraint/bottom_boundary_violation_fraction"].value == 0
    assert metrics["constraint/bottom_boundary_violation_max"].value == 0
    assert metrics["constraint/bottom_envelope_violation_max"].value > 0


def test_plasticity_metric_counts_nodes_below_y_bottom():
    pred = _plasticity_pred_from_coords(
        [
            [[1.0, 1.0], [1.0, -0.2]],
            [[0.0, 0.5], [0.0, -0.3]],
        ]
    )

    metrics = compute_plasticity_metrics(
        pred,
        {},
        {
            "shapelist": (2, 2),
            "t_out": 1,
            "out_dim": 4,
            "y_bottom": -0.1,
            "below_y_bottom_tolerance": 0.0,
        },
    )

    assert metrics["constraint/below_y_bottom_violation_count"].value == 2
    assert metrics["constraint/below_y_bottom_violation_fraction"].value == 0.5
    assert metrics["constraint/below_y_bottom_violation_max"].value == 0.20000001788139343


def test_plasticity_metric_floor_tolerance_ignores_roundoff():
    pred = _plasticity_pred_from_coords(
        [
            [[1.0, 1.0], [1.0, -0.100007]],
            [[0.0, 0.5], [0.0, -0.099993]],
        ]
    )

    metrics = compute_plasticity_metrics(
        pred,
        {},
        {
            "shapelist": (2, 2),
            "t_out": 1,
            "out_dim": 4,
            "y_bottom": -0.1,
            "below_y_bottom_tolerance": 1.0e-4,
        },
    )

    assert metrics["constraint/below_y_bottom_violation_count"].value == 0
    assert metrics["constraint/below_y_bottom_violation_fraction"].value == 0
    assert metrics["constraint/below_y_bottom_violation_max"].value == 0
    assert metrics["constraint/below_y_bottom_excess_raw_max"].value > 0


def test_plasticity_metric_top_envelope_tolerance_ignores_small_excess():
    pred = _plasticity_pred_from_coords(
        [
            [[1.0, 1.000007], [1.0, 0.0]],
            [[0.0, 1.000007], [0.0, 0.0]],
        ]
    )
    fx = torch.tensor([[[1.0], [1.0], [1.0], [1.0]]])

    metrics = compute_plasticity_metrics(
        pred,
        {"x": fx, "time": torch.zeros(1, 1)},
        {
            "shapelist": (2, 2),
            "t_out": 1,
            "out_dim": 4,
            "die_speed": 0.0,
            "top_height": 1.0,
            "top_envelope_tolerance": 1.0e-4,
        },
    )

    assert metrics["constraint/top_envelope_violation_count"].value == 0
    assert metrics["constraint/top_envelope_violation_fraction"].value == 0
    assert metrics["constraint/top_envelope_violation_max"].value == 0
    assert metrics["constraint/top_envelope_excess_raw_max"].value > 0


def test_plasticity_metric_distinguishes_floating_bottom_from_floor_penetration():
    pred = _plasticity_pred_from_coords(
        [
            [[1.0, 4.0], [1.0, 3.0]],
            [[0.0, 4.0], [0.0, 3.0]],
        ]
    )
    target = _plasticity_pred_from_coords(
        [
            [[1.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )

    metrics = compute_plasticity_metrics(
        pred,
        {"target": target},
        {"shapelist": (2, 2), "t_out": 1, "out_dim": 4, "y_bottom": 0.0},
    )

    assert metrics["constraint/bottom_boundary_violation_count"].value == 2
    assert metrics["constraint/bottom_boundary_violation_fraction"].value == 1
    assert metrics["constraint/below_y_bottom_violation_count"].value == 0


def test_plasticity_metric_bottom_band_membership_without_reference():
    # Coords-only field (no displacement channels, no target): the pinned bottom
    # row is checked for membership in the GT band [-0.10001, -0.09999], so an
    # in-band node registers zero offset while a below-band node violates.
    pred = torch.tensor(
        [  # (B, T, H, W, C) with C=2; W=-1 is the bottom row
            [
                [
                    [[1.0, 1.0], [1.0, -0.10000]],  # in-band bottom -> ok
                    [[0.0, 1.0], [0.0, -0.20000]],  # below band     -> violation
                ]
            ]
        ],
        dtype=torch.float32,
    )

    metrics = compute_plasticity_metrics(
        pred,
        {},
        {"shapelist": (2, 2), "t_out": 1, "out_dim": 2},
    )

    assert metrics["constraint/bottom_boundary_violation_count"].value == 1
    assert metrics["constraint/bottom_boundary_violation_fraction"].value == 0.5
    assert metrics["constraint/bottom_boundary_abs_error_max"].value > 0


def test_benchmark_diagnostics_uses_constraint_metric_parameters():
    pred = _plasticity_pred_from_coords(
        [
            [[1.0, 1.0], [1.0, -0.2]],
            [[0.0, 1.0], [0.0, -0.2]],
        ]
    )
    diagnostic_fn = make_benchmark_diagnostic_fn(
        {
            "benchmark": {"name": "plasticity_2d"},
            "diagnostics": {"below_y_bottom_tolerance": 0.15},
            "constraint": {"y_bottom": -0.1},
        },
        {"shapelist": (2, 2), "t_out": 1, "out_dim": 4, "y_bottom": 0.5},
    )

    metrics = diagnostic_fn(pred=pred)

    assert metrics["constraint/below_y_bottom_violation_count"].value == 0


def _plasticity_pred_from_coords(coords):
    import torch

    xy = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    disp = torch.zeros_like(xy)
    return torch.cat([xy, disp], dim=-1).reshape(1, 4, 4)
