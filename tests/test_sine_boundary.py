import torch

from omni_hc.constraints import ConstraintOutput, SineBoundaryConstraint


def _legacy_forward(c: SineBoundaryConstraint, pred, fx):
    """Original hard-overwrite implementation, kept as the oracle for the
    g + l*N refactor (l = perimeter indicator must reproduce this exactly)."""
    B = pred.shape[0]
    feats = c._boundary_feats(fx)
    coeffs = c.coeff_head(feats).view(B, 4, c.n_modes)
    u_bottom = coeffs[:, 0] @ c.basis_h.T
    u_top = coeffs[:, 1] @ c.basis_h.T
    u_left = coeffs[:, 2] @ c.basis_v.T
    u_right = coeffs[:, 3] @ c.basis_v.T
    out = pred.clone()
    out[:, c.idx_bottom, 0] = u_bottom
    out[:, c.idx_top, 0] = u_top
    out[:, c.idx_left, 0] = u_left
    out[:, c.idx_right, 0] = u_right
    return out


def _make(grid=(7, 9), n_modes=4, channels=3, batch=2, seed=0):
    torch.manual_seed(seed)
    H, W = grid
    c = SineBoundaryConstraint(
        n_modes=n_modes, grid_shape=grid, hidden_dim=16, n_layers=1
    )
    c.eval()
    pred = torch.randn(batch, H * W, channels)
    fx = torch.randn(batch, H * W, 1)
    return c, pred, fx, (H, W)


def test_ansatz_matches_legacy_overwrite():
    c, pred, fx, _ = _make()
    with torch.no_grad():
        got = c(pred=pred, fx=fx)
        want = _legacy_forward(c, pred, fx)
    assert torch.equal(got, want), (got - want).abs().max().item()


def test_other_channels_and_interior_untouched():
    c, pred, fx, (H, W) = _make()
    with torch.no_grad():
        out = c(pred=pred, fx=fx)
    # Non-zero channels are passed through verbatim.
    assert torch.equal(out[..., 1:], pred[..., 1:])
    # Interior of channel 0 is the backbone prediction, unchanged.
    field = out[..., 0].view(-1, H, W)
    base = pred[..., 0].view(-1, H, W)
    assert torch.equal(field[:, 1:-1, 1:-1], base[:, 1:-1, 1:-1])


def test_boundary_is_exactly_g_independent_of_pred():
    c, pred, fx, _ = _make()
    with torch.no_grad():
        out_a = c(pred=pred, fx=fx)
        out_b = c(pred=pred + 100.0, fx=fx)
    bidx = c.idx_all_boundary
    # Boundary is hard-set to g, so it must not depend on the backbone pred.
    assert torch.equal(out_a[:, bidx, 0], out_b[:, bidx, 0])
    # Corners are zero in every edge sine basis.
    for corner in (0, c.W - 1, (c.H - 1) * c.W, c.H * c.W - 1):
        assert torch.allclose(
            out_a[:, corner, 0], torch.zeros_like(out_a[:, corner, 0]), atol=1e-6
        )


class _Normalizer:
    """Per-channel affine, mirrors benchmarks.darcy.data.UnitTransformer."""

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).reshape(1, 1, -1)
        self.std = torch.tensor(std).reshape(1, 1, -1)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


def test_no_normalizer_path_is_identity_to_legacy():
    # set_target_normalizer(None) must not change the bit-for-bit equivalence.
    c, pred, fx, _ = _make()
    c.set_target_normalizer(None)
    with torch.no_grad():
        got = c(pred=pred, fx=fx)
        want = _legacy_forward(c, pred, fx)
    assert torch.equal(got, want)


def test_normalizer_decoded_boundary_corners_are_zero():
    # The bug: in normalized space the sine basis pinned corners to 0, which
    # decodes to the field mean (non-zero). With encode_target, decoded
    # corners must be ~0 even when the normalizer mean is large.
    c, pred, fx, (H, W) = _make(channels=1)
    norm = _Normalizer([0.7], [0.2])  # mean != 0 is the failure case
    c.set_target_normalizer(norm)
    with torch.no_grad():
        out = c(pred=pred, fx=fx)
    decoded = norm.decode(out)[..., 0]
    for corner in (0, W - 1, (H - 1) * W, H * W - 1):
        assert torch.allclose(
            decoded[:, corner], torch.zeros_like(decoded[:, corner]), atol=1e-5
        )


def test_normalizer_interior_is_exact_backbone():
    # Interior must remain the raw (normalized) backbone prediction: the
    # particular's interior is a hard 0, not encode(0) = -mu/sigma.
    c, pred, fx, (H, W) = _make(channels=1)
    c.set_target_normalizer(_Normalizer([0.7], [0.2]))
    with torch.no_grad():
        out = c(pred=pred, fx=fx)
    field = out[..., 0].view(-1, H, W)
    base = pred[..., 0].view(-1, H, W)
    assert torch.equal(field[:, 1:-1, 1:-1], base[:, 1:-1, 1:-1])


def test_return_aux_contract():
    c, pred, fx, _ = _make()
    with torch.no_grad():
        out = c(pred=pred, fx=fx, return_aux=True)
    assert isinstance(out, ConstraintOutput)
    assert "boundary_pred" in out.aux
    assert "constraint/boundary_correction_mean_abs" in out.diagnostics
    assert torch.equal(out.pred, c(pred=pred, fx=fx))


def test_feature_modes_preserve_forward_shape():
    pred = torch.randn(2, 7 * 9, 1)
    fx = torch.randn(2, 7 * 9, 1)
    for mode in (
        "boundary",
        "boundary_inner",
        "boundary_stats",
        "boundary_inner_stats",
        "full",
    ):
        c = SineBoundaryConstraint(
            n_modes=4,
            grid_shape=(7, 9),
            hidden_dim=16,
            n_layers=1,
            feature_mode=mode,
        )
        with torch.no_grad():
            out = c(pred=pred, fx=fx)
        assert out.shape == pred.shape


def test_unknown_feature_mode_fails_fast():
    try:
        SineBoundaryConstraint(
            n_modes=4,
            grid_shape=(7, 9),
            hidden_dim=16,
            feature_mode="unknown",
        )
    except ValueError as exc:
        assert "feature_mode" in str(exc)
    else:
        raise AssertionError("unknown feature_mode should fail")
