from omni_hc.constraints import ConstraintDiagnostic
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


def test_plasticity_metric_reports_zero_for_monotone_grid():
    pred = _plasticity_pred_from_coords(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ]
    )

    metrics = compute_plasticity_metrics(pred, {}, {"shapelist": (2, 2), "t_out": 1, "out_dim": 4})

    assert metrics["constraint/neg_spacing_count"].value == 0
    assert metrics["constraint/neg_spacing_fraction"].value == 0


def test_plasticity_metric_reports_fraction_for_inverted_grid():
    pred = _plasticity_pred_from_coords(
        [
            [[0.0, 0.0], [0.0, -1.0]],
            [[-1.0, 0.0], [-1.0, -1.0]],
        ]
    )

    metrics = compute_plasticity_metrics(pred, {}, {"shapelist": (2, 2), "t_out": 1, "out_dim": 4})

    assert metrics["constraint/neg_spacing_count"].value == 4
    assert metrics["constraint/neg_spacing_fraction"].value == 1


def _plasticity_pred_from_coords(coords):
    import torch

    xy = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    disp = torch.zeros_like(xy)
    return torch.cat([xy, disp], dim=-1).reshape(1, 4, 4)
