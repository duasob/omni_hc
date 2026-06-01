from omni_hc.constraints import ConstraintDiagnostic
from omni_hc.constraints.metrics.darcy import compute as compute_darcy_metrics
from omni_hc.constraints.metrics.plasticity import compute as compute_plasticity_metrics
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

    assert metrics["constraint/neg_spacing_count"].value == 0
    assert metrics["constraint/neg_spacing_worst_sample_fraction"].value == 0
    assert metrics["constraint/neg_spacing_fraction"].value == 0
    assert metrics["constraint/flipped_cell_count_worst"].value == 0
    assert metrics["constraint/flipped_cell_fraction_worst"].value == 0


def test_plasticity_metric_reports_fraction_for_inverted_grid():
    pred = _plasticity_pred_from_coords(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ]
    )

    metrics = compute_plasticity_metrics(pred, {}, {"shapelist": (2, 2), "t_out": 1, "out_dim": 4})

    assert metrics["constraint/neg_spacing_count"].value == 4
    assert metrics["constraint/neg_spacing_worst_sample_fraction"].value == 1
    assert metrics["constraint/neg_spacing_fraction"].value == 1


def test_plasticity_metric_reports_worst_sample_flip_count():
    import torch

    valid = _plasticity_pred_from_coords(
        [
            [[1.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )
    invalid = _plasticity_pred_from_coords(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[1.0, 1.0], [1.0, 0.0]],
        ]
    )
    pred = torch.cat([valid, invalid], dim=0)
    target = torch.cat([valid, valid], dim=0)

    metrics = compute_plasticity_metrics(
        pred,
        {"target": target},
        {"shapelist": (2, 2), "t_out": 1, "out_dim": 4},
    )

    assert metrics["constraint/flipped_cell_count_mean"].value == 0.5
    assert metrics["constraint/flipped_cell_count_worst"].value == 1
    assert metrics["constraint/flipped_cell_fraction_mean"].value == 0.5
    assert metrics["constraint/flipped_cell_fraction_worst"].value == 1


def _plasticity_pred_from_coords(coords):
    import torch

    xy = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    disp = torch.zeros_like(xy)
    return torch.cat([xy, disp], dim=-1).reshape(1, 4, 4)
