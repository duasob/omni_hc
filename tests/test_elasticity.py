import torch

from omni_hc.constraints import ElasticityPlaneStressVMConstraint


def test_plane_stress_vm_returns_scalar_stress_and_latents():
    constraint = ElasticityPlaneStressVMConstraint(backbone_out_dim=2)
    pred = torch.randn(3, 11, 2)

    out = constraint(pred=pred, return_aux=True)

    assert out.pred.shape == (3, 11, 1)
    assert torch.all(out.pred >= 0.0)
    for key in (
        "mean_log_stretch",
        "deviatoric_log_stretch",
        "lambda_1",
        "lambda_2",
        "lambda_3",
        "full_det_f",
        "principal_cauchy_stress_3",
    ):
        assert out.aux[key].shape == (3, 11, 1)
    assert "constraint/full_det_f_abs_error_max" in out.diagnostics
    assert "constraint/plane_stress_abs_error_max" in out.diagnostics


def test_plane_stress_vm_enforces_3d_incompressibility():
    constraint = ElasticityPlaneStressVMConstraint(backbone_out_dim=2)
    pred = 0.2 * torch.randn(2, 17, 2)

    out = constraint(pred=pred, return_aux=True)

    assert torch.allclose(
        out.aux["full_det_f"],
        torch.ones_like(out.aux["full_det_f"]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        out.aux["lambda_3"],
        out.aux["in_plane_det_f"].reciprocal(),
        atol=1e-6,
        rtol=1e-6,
    )


def test_plane_stress_vm_enforces_zero_thickness_stress():
    constraint = ElasticityPlaneStressVMConstraint(backbone_out_dim=2)
    pred = 0.2 * torch.randn(2, 17, 2)

    out = constraint(pred=pred, return_aux=True)

    assert torch.allclose(
        out.aux["principal_cauchy_stress_3"],
        torch.zeros_like(out.aux["principal_cauchy_stress_3"]),
        atol=1e-5,
        rtol=0.0,
    )


def test_plane_stress_vm_identity_has_zero_stress_up_to_smoothing_epsilon():
    constraint = ElasticityPlaneStressVMConstraint(backbone_out_dim=2)
    pred = torch.zeros(2, 5, 2)

    out = constraint(pred=pred, return_aux=True)

    for key in ("lambda_1", "lambda_2", "lambda_3", "full_det_f"):
        assert torch.allclose(out.aux[key], torch.ones_like(out.aux[key]))
    assert torch.allclose(
        out.pred,
        torch.full_like(out.pred, 1e-4),
        atol=1e-7,
        rtol=0.0,
    )


def test_plane_stress_vm_matches_closed_form():
    c1 = 12.0
    c2 = 3.0
    constraint = ElasticityPlaneStressVMConstraint(
        backbone_out_dim=2,
        c1=c1,
        c2=c2,
        max_mean_log_stretch=1.0,
        max_deviatoric_log_stretch=1.0,
    )
    pred = torch.tensor([[[0.2, -0.35]]])

    out = constraint(pred=pred, return_aux=True)

    m = torch.tanh(pred[..., 0])
    d = torch.tanh(pred[..., 1])
    lambda_1 = torch.exp(m + d)
    lambda_2 = torch.exp(m - d)
    lambda_3 = torch.exp(-2.0 * m)
    pressure = 2.0 * c1 * lambda_3.square() - 2.0 * c2 / lambda_3.square()
    sigma_1 = -pressure + 2.0 * c1 * lambda_1.square() - 2.0 * c2 / lambda_1.square()
    sigma_2 = -pressure + 2.0 * c1 * lambda_2.square() - 2.0 * c2 / lambda_2.square()
    expected = (
        sigma_1.square() - sigma_1 * sigma_2 + sigma_2.square() + 1e-8
    ).sqrt()

    assert torch.allclose(out.pred[..., 0], expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out.aux["pressure"][..., 0], pressure)


def test_plane_stress_vm_is_invariant_to_deviatoric_stretch_sign():
    constraint = ElasticityPlaneStressVMConstraint(
        backbone_out_dim=2,
        max_mean_log_stretch=1.0,
        max_deviatoric_log_stretch=1.0,
    )
    pred = torch.tensor([[[0.2, 0.7]]])
    swapped = pred.clone()
    swapped[..., 1] *= -1.0

    out = constraint(pred=pred)
    swapped_out = constraint(pred=swapped)

    assert torch.allclose(out, swapped_out, atol=1e-4, rtol=1e-5)


def test_plane_stress_vm_depends_on_both_material_parameters():
    pred = torch.tensor([[[0.2, 0.35]]])
    base = ElasticityPlaneStressVMConstraint(
        backbone_out_dim=2,
        c1=12.0,
        c2=3.0,
        max_mean_log_stretch=1.0,
        max_deviatoric_log_stretch=1.0,
    )
    changed_c1 = ElasticityPlaneStressVMConstraint(
        backbone_out_dim=2,
        c1=18.0,
        c2=3.0,
        max_mean_log_stretch=1.0,
        max_deviatoric_log_stretch=1.0,
    )
    changed_c2 = ElasticityPlaneStressVMConstraint(
        backbone_out_dim=2,
        c1=12.0,
        c2=5.0,
        max_mean_log_stretch=1.0,
        max_deviatoric_log_stretch=1.0,
    )

    assert not torch.allclose(base(pred=pred), changed_c1(pred=pred))
    assert not torch.allclose(base(pred=pred), changed_c2(pred=pred))


def test_plane_stress_vm_head_uses_scalar_backbone_and_coords():
    constraint = ElasticityPlaneStressVMConstraint(
        backbone_out_dim=1,
        head_hidden_dim=8,
        head_layers=1,
        head_init_scale=1e-3,
        max_mean_log_stretch=2e-3,
        max_deviatoric_log_stretch=3e-3,
    )
    pred = torch.randn(2, 7, 1)
    coords = torch.rand(2, 7, 2)

    out = constraint(pred=pred, coords=coords, return_aux=True)

    assert out.pred.shape == (2, 7, 1)
    assert out.aux["mean_log_stretch_raw"].shape == (2, 7, 1)
    assert out.aux["deviatoric_log_stretch_raw"].shape == (2, 7, 1)
    assert out.aux["param_head_input_z"].shape == (2, 7, 1)
    assert torch.all(out.aux["mean_log_stretch"].abs() <= 2e-3)
    assert torch.all(out.aux["deviatoric_log_stretch"].abs() <= 3e-3)


def test_plane_stress_vm_default_initialization_has_finite_gradients():
    constraint = ElasticityPlaneStressVMConstraint(backbone_out_dim=1)
    pred = torch.randn(2, 7, 1, requires_grad=True)
    coords = torch.rand(2, 7, 2)

    loss = constraint(pred=pred, coords=coords).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))
    for parameter in constraint.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))
