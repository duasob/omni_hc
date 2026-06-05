import pytest
import torch

from omni_hc.training.tasks.dynamic_conditional import (
    _plasticity_material_grid,
    plasticity_component_loss_metrics,
)


def test_plasticity_component_loss_metrics_split_position_and_deviation():
    target = torch.ones(2, 3, 2 * 4)
    pred = target.clone()
    pred_view = pred.reshape(2, 3, 2, 4)
    pred_view[:, :, :, 0:2] = 2.0
    pred_view[:, :, :, 2:4] = 3.0

    metrics = plasticity_component_loss_metrics(pred, target, t_out=2, out_dim=4)

    assert metrics["loss_position"] == pytest.approx(1.0)
    assert metrics["loss_deviation"] == pytest.approx(2.0)


def test_plasticity_component_loss_metrics_ignores_short_targets():
    pred = torch.zeros(2, 3, 2 * 2)
    target = torch.ones_like(pred)

    assert plasticity_component_loss_metrics(pred, target, t_out=2, out_dim=2) == {}


def test_plasticity_component_loss_metrics_reports_deviation_consistency():
    material = _plasticity_material_grid(
        (2, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
        x_left=0.0,
        x_right=-1.0,
        y_top=1.0,
        y_bottom=0.0,
    )
    target_view = torch.zeros(1, 4, 1, 4)
    target_view[..., 0:2] = material
    target_view[..., 2:4] = 0.0
    pred_view = target_view.clone()
    pred_view[..., 0] = pred_view[..., 0] - 0.5
    pred_view[..., 2] = pred_view[..., 0] - material[..., 0]
    pred_view[..., 3] = pred_view[..., 1] - material[..., 1]
    pred = pred_view.reshape(1, 4, 4)
    target = target_view.reshape(1, 4, 4)

    metrics = plasticity_component_loss_metrics(
        pred,
        target,
        t_out=1,
        out_dim=4,
        material_grid=material,
    )

    assert metrics["deviation_consistency_pred_mse"] == pytest.approx(0.0)
    assert metrics["deviation_consistency_pred_abs_max"] == pytest.approx(0.0)
    assert metrics["deviation_consistency_target_mse"] == pytest.approx(0.0)
    assert metrics["deviation_consistency_target_abs_max"] == pytest.approx(0.0)


def test_plasticity_component_loss_metrics_catches_inconsistent_deviation():
    material = _plasticity_material_grid(
        (1, 1),
        device=torch.device("cpu"),
        dtype=torch.float32,
        x_left=0.0,
        x_right=0.0,
        y_top=0.0,
        y_bottom=0.0,
    )
    target = torch.tensor([[[1.0, 2.0, 1.0, 2.0]]])
    pred = torch.tensor([[[1.0, 2.0, 0.0, 0.0]]])

    metrics = plasticity_component_loss_metrics(
        pred,
        target,
        t_out=1,
        out_dim=4,
        material_grid=material,
    )

    assert metrics["deviation_consistency_pred_mse"] == pytest.approx(2.5)
    assert metrics["deviation_consistency_pred_abs_max"] == pytest.approx(2.0)
    assert metrics["deviation_consistency_target_mse"] == pytest.approx(0.0)
