from omni_hc.constraints import ConstraintDiagnostic
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
