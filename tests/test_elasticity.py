import torch
import torch.nn as nn

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
    d = -torch.tanh(pred[..., 1]).square()
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


def test_plane_stress_vm_canonically_orders_in_plane_stretches():
    constraint = ElasticityPlaneStressVMConstraint(
        backbone_out_dim=2,
        max_mean_log_stretch=1.0,
        max_deviatoric_log_stretch=1.0,
    )
    pred = torch.randn(2, 7, 2)

    out = constraint(pred=pred, return_aux=True)

    assert torch.all(out.aux["deviatoric_log_stretch"] <= 0.0)
    assert torch.all(out.aux["lambda_1"] <= out.aux["lambda_2"])


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


def test_plane_stress_vm_head_uses_vector_backbone_latent_and_coords():
    constraint = ElasticityPlaneStressVMConstraint(
        backbone_out_dim=32,
        head_hidden_dim=8,
        head_layers=1,
        head_init_scale=1e-3,
        max_mean_log_stretch=2e-3,
        max_deviatoric_log_stretch=3e-3,
    )
    pred = torch.randn(2, 7, 32)
    coords = torch.rand(2, 7, 2)

    out = constraint(pred=pred, coords=coords, return_aux=True)

    assert out.pred.shape == (2, 7, 1)
    assert out.aux["mean_log_stretch_raw"].shape == (2, 7, 1)
    assert out.aux["deviatoric_log_stretch_raw"].shape == (2, 7, 1)
    assert out.aux["param_head_input_z"].shape == (2, 7, 32)
    assert torch.all(out.aux["mean_log_stretch"].abs() <= 2e-3)
    assert torch.all(out.aux["deviatoric_log_stretch"] <= 0.0)
    assert torch.all(out.aux["deviatoric_log_stretch"] >= -3e-3)


def test_plane_stress_vm_default_initialization_has_finite_gradients():
    constraint = ElasticityPlaneStressVMConstraint(backbone_out_dim=32)
    pred = torch.randn(2, 7, 32, requires_grad=True)
    coords = torch.rand(2, 7, 2)

    loss = constraint(pred=pred, coords=coords).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.all(torch.isfinite(pred.grad))
    for parameter in constraint.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))


class _FixedExtractor:
    """Minimal stand-in for ForwardHookLatentExtractor in unit tests."""

    def __init__(self, latent):
        self._latent = latent

    def get(self):
        return self._latent


def test_engineered_decoder_decodes_from_latent_and_preserves_guarantees():
    latent_dim = 24
    latent = torch.randn(2, 9, latent_dim)
    constraint = ElasticityPlaneStressVMConstraint(
        backbone_out_dim=1,
        latent_dim=latent_dim,
        head_hidden_dim=16,
        head_layers=1,
        extractor=_FixedExtractor(latent),
    )
    pred = torch.randn(2, 9, 1)
    coords = torch.rand(2, 9, 2)

    out = constraint(pred=pred, coords=coords, return_aux=True)

    assert out.pred.shape == (2, 9, 1)
    assert constraint.param_head[0].in_features == latent_dim + 2
    assert out.aux["param_head_latent"].shape == (2, 9, latent_dim)
    assert "param_head_input_z" not in out.aux
    assert torch.allclose(
        out.aux["full_det_f"], torch.ones_like(out.aux["full_det_f"]), atol=1e-5
    )
    assert torch.allclose(
        out.aux["principal_cauchy_stress_3"],
        torch.zeros_like(out.aux["principal_cauchy_stress_3"]),
        atol=1e-5,
    )


def test_engineered_decoder_without_coords_consumes_latent_only():
    latent_dim = 12
    latent = torch.randn(2, 5, latent_dim)
    constraint = ElasticityPlaneStressVMConstraint(
        backbone_out_dim=1,
        latent_dim=latent_dim,
        head_hidden_dim=8,
        head_layers=1,
        decoder_include_coords=False,
        extractor=_FixedExtractor(latent),
    )
    pred = torch.randn(2, 5, 1)

    out = constraint(pred=pred, return_aux=True)

    assert out.pred.shape == (2, 5, 1)
    assert constraint.param_head[0].in_features == latent_dim


class _MidBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, fx):
        return self.lin(fx)


class _LastBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Attn = nn.Linear(dim, dim)
        self.ln_3 = nn.Identity()
        self.out = nn.Linear(dim, 1)

    def forward(self, fx):
        fx = self.Attn(fx) + fx
        return self.out(self.ln_3(fx))


class _FakeTransolver(nn.Module):
    """Exposes the blocks.-2 / blocks.-1.Attn / blocks.-1.ln_3 hook paths."""

    def __init__(self, dim, space_dim=2):
        super().__init__()
        self.n_hidden = dim
        self.embed = nn.Linear(space_dim, dim)
        self.blocks = nn.ModuleList([_MidBlock(dim), _MidBlock(dim), _LastBlock(dim)])

    def forward(self, coords, fx=None):
        h = self.embed(coords)
        for block in self.blocks:
            h = block(h)
        return h


def test_engineered_decoder_build_wires_extractor_and_infers_latent_dim():
    dim = 8
    backbone = _FakeTransolver(dim=dim)
    cfg = {
        "constraint": {
            "name": "elasticity_plane_stress_vm_constraint",
            "backbone_out_dim": 1,
            "latent_module": ["blocks.-2", "blocks.-1.Attn", "blocks.-1.ln_3"],
            "head_hidden_dim": 16,
            "head_layers": 1,
        }
    }
    model_context = {"backbone_out_dim": 1, "target_out_dim": 1, "n_hidden": dim}

    model = ElasticityPlaneStressVMConstraint.build(backbone, model_context, cfg)

    assert model.constraint.latent_dim == dim * 3
    assert model.constraint.param_head[0].in_features == dim * 3 + 2

    coords = torch.rand(2, 6, 2)
    out = model(coords, return_aux=True)

    assert out.pred.shape == (2, 6, 1)
    assert out.aux["param_head_latent"].shape == (2, 6, dim * 3)
