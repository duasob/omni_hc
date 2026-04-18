import torch

from omni_hc.constraints import ConstraintOutput, MeanConstraint, match_mean


def test_match_mean_aligns_to_target_mean():
    x = torch.randn(2, 16, 1)
    target = torch.randn(2, 16, 1)

    aligned = match_mean(x, target)

    assert torch.allclose(aligned.mean(dim=1), target.mean(dim=1))


def test_post_output_constraint_zeroes_mean():
    pred = torch.randn(4, 32, 1)
    constraint = MeanConstraint(mode="post_output", out_dim=1)

    out = constraint(pred=pred)

    zeros = torch.zeros_like(out.mean(dim=1))
    assert torch.allclose(out.mean(dim=1), zeros, atol=1e-6)


def test_post_output_constraint_emits_structured_diagnostics():
    pred = torch.randn(2, 16, 1)
    constraint = MeanConstraint(mode="post_output", out_dim=1)

    out = constraint(pred=pred, return_aux=True)

    assert isinstance(out, ConstraintOutput)
    assert "constraint/pred_base_mean" in out.diagnostics
    assert "constraint/corr_mean" in out.diagnostics
