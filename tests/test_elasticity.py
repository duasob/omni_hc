import math

import torch

from omni_hc.constraints import ElasticityDeviatoricStressConstraint


def test_elasticity_deviatoric_stress_returns_scalar_stress():
    constraint = ElasticityDeviatoricStressConstraint(backbone_out_dim=2)
    pred = torch.randn(3, 11, 2)

    out = constraint(pred=pred, return_aux=True)

    assert out.pred.shape == (3, 11, 1)
    assert torch.all(out.pred >= 0.0)
    assert out.aux["theta"].shape == (3, 11, 1)
    assert out.aux["lambda"].shape == (3, 11, 1)
    assert out.aux["det_c"].shape == (3, 11, 1)
    assert out.aux["stress_dev_11"].shape == (3, 11, 1)
    assert "constraint/det_c_abs_error_max" in out.diagnostics
    assert "constraint/stress_dev_inner_mean" in out.diagnostics


def test_elasticity_deviatoric_stress_has_unit_c_determinant():
    constraint = ElasticityDeviatoricStressConstraint(backbone_out_dim=2)
    pred = 0.2 * torch.randn(2, 17, 2)

    out = constraint(pred=pred, return_aux=True)

    assert torch.allclose(
        out.aux["det_c"],
        torch.ones_like(out.aux["det_c"]),
        atol=1e-4,
        rtol=1e-4,
    )


def test_elasticity_deviatoric_stress_identity_has_zero_von_mises():
    constraint = ElasticityDeviatoricStressConstraint(backbone_out_dim=2)
    pred = torch.zeros(2, 5, 2)

    out = constraint(pred=pred, return_aux=True)

    assert torch.allclose(out.aux["i1"], torch.full_like(out.aux["i1"], 2.0))
    assert torch.allclose(out.aux["i2"], torch.ones_like(out.aux["i2"]))
    assert torch.allclose(out.aux["stress_dev_inner"], torch.zeros_like(out.aux["stress_dev_inner"]))
    assert torch.allclose(out.pred, torch.zeros_like(out.pred), atol=1e-6)


def test_elasticity_deviatoric_stress_matches_diagonal_closed_form():
    c2 = 9.79e3
    constraint = ElasticityDeviatoricStressConstraint(
        backbone_out_dim=2,
        c2=c2,
        max_log_lambda=1.0,
    )
    log_lambda = torch.tensor(0.2)
    pred = torch.tensor([[[0.0, float(log_lambda)]]])

    out = constraint(pred=pred, return_aux=True)

    bounded_log_lambda = torch.tanh(log_lambda)
    lambda_sq = torch.exp(2.0 * bounded_log_lambda)
    inv_lambda_sq = torch.exp(-2.0 * bounded_log_lambda)
    expected = math.sqrt(3.0) * c2 * torch.abs(lambda_sq - inv_lambda_sq)
    assert torch.allclose(out.pred[0, 0, 0], expected, atol=1e-3, rtol=1e-5)


def test_elasticity_deviatoric_stress_is_periodic_in_theta():
    constraint = ElasticityDeviatoricStressConstraint(
        backbone_out_dim=2,
        max_log_lambda=1.0,
    )
    pred = torch.tensor([[[0.2, 0.7]]])
    shifted = pred.clone()
    shifted[..., 0] = shifted[..., 0] + math.pi

    out = constraint(pred=pred)
    shifted_out = constraint(pred=shifted)

    assert torch.allclose(out, shifted_out, atol=1e-4, rtol=1e-5)


def test_elasticity_deviatoric_stress_head_uses_scalar_backbone_and_coords():
    constraint = ElasticityDeviatoricStressConstraint(
        backbone_out_dim=1,
        head_hidden_dim=8,
        head_layers=1,
        head_init_scale=1e-3,
        max_log_lambda=0.03,
    )
    pred = torch.randn(2, 7, 1)
    coords = torch.rand(2, 7, 2)

    out = constraint(pred=pred, coords=coords, return_aux=True)

    assert out.pred.shape == (2, 7, 1)
    assert out.aux["theta_raw"].shape == (2, 7, 1)
    assert out.aux["log_lambda_raw"].shape == (2, 7, 1)
    assert out.aux["param_head_input_z"].shape == (2, 7, 1)
    assert torch.all(out.aux["log_lambda"].abs() <= 0.03)
