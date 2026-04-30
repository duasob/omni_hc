from __future__ import annotations

import torch
import torch.nn as nn

from .base import ConstraintDiagnostic, ConstraintModule
from .utils.boundary_ops import encode_target


def _det2(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]


def _identity_like(matrix: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(2, dtype=matrix.dtype, device=matrix.device)
    return eye.expand(*matrix.shape[:-2], 2, 2)


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _right_cauchy_green_from_spectral_params(
    theta: torch.Tensor,
    log_lambda: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return C = R diag(lambda^2, lambda^-2) R^T and lambda."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    lambda_stretch = torch.exp(log_lambda)
    lambda_sq = torch.exp(2.0 * log_lambda)
    inv_lambda_sq = torch.exp(-2.0 * log_lambda)

    c11 = c.square() * lambda_sq + s.square() * inv_lambda_sq
    c22 = s.square() * lambda_sq + c.square() * inv_lambda_sq
    c12 = c * s * (lambda_sq - inv_lambda_sq)
    right_cauchy_green = torch.stack(
        (
            torch.stack((c11, c12), dim=-1),
            torch.stack((c12, c22), dim=-1),
        ),
        dim=-2,
    )
    return right_cauchy_green, lambda_stretch


def _make_mlp(
    *,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.GELU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


class ElasticityDeviatoricStressConstraint(ConstraintModule):
    """
    Computes 2D incompressible hyperelastic von Mises stress.

    With backbone_out_dim=1, the backbone emits a scalar latent z per point. The
    constraint predicts (theta, log_lambda) with an internal head over [z, x, y].

    With backbone_out_dim=2, the backbone output is interpreted directly as
    (theta_raw, log_lambda_raw).

    In both modes, the Right Cauchy-Green tensor is constructed as

        C = R(theta) diag(lambda^2, lambda^-2) R(theta)^T.

    This enforces C symmetric positive definite and det(C) = 1 by construction.
    The returned scalar is computed from the 2D deviatoric Second
    Piola-Kirchhoff stress.
    """

    name = "elasticity_deviatoric_stress"

    def __init__(
        self,
        *,
        backbone_out_dim: int = 1,
        target_out_dim: int = 1,
        c1: float = 1.863e5,
        c2: float = 9.79e3,
        max_log_lambda: float = 0.03,
        head_hidden_dim: int = 32,
        head_layers: int = 2,
        head_init_scale: float = 1e-3,
        theta_bias: float = 0.0,
        log_lambda_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone_out_dim = int(backbone_out_dim)
        self.target_out_dim = int(target_out_dim)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.max_log_lambda = float(max_log_lambda)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_layers = int(head_layers)
        self.head_init_scale = float(head_init_scale)
        self.theta_bias = float(theta_bias)
        self.log_lambda_bias = float(log_lambda_bias)
        self.target_normalizer = None

        if self.backbone_out_dim not in {1, 2}:
            raise ValueError(
                "ElasticityDeviatoricStressConstraint expects "
                f"backbone_out_dim=1 or 2, got {self.backbone_out_dim}."
            )
        if self.target_out_dim != 1:
            raise ValueError(
                "ElasticityDeviatoricStressConstraint returns one scalar stress "
                f"channel, got target_out_dim={self.target_out_dim}."
            )

        self.param_head = None
        if self.backbone_out_dim == 1:
            self.param_head = _make_mlp(
                in_dim=3,
                hidden_dim=self.head_hidden_dim,
                out_dim=2,
                n_layers=self.head_layers,
            )
            self._init_param_head()

    def _init_param_head(self) -> None:
        if self.param_head is None:
            return
        final = self.param_head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Elasticity parameter head must end with nn.Linear.")
        with torch.no_grad():
            final.weight.mul_(self.head_init_scale)
            final.bias.copy_(
                torch.tensor(
                    [self.theta_bias, self.log_lambda_bias],
                    dtype=final.bias.dtype,
                    device=final.bias.device,
                )
            )

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def _validate_pred(self, pred: torch.Tensor) -> None:
        if pred.ndim != 3 or pred.shape[-1] != self.backbone_out_dim:
            raise ValueError(
                "ElasticityDeviatoricStressConstraint expects backbone output "
                f"with shape (batch, n_points, {self.backbone_out_dim}), got "
                f"{tuple(pred.shape)!r}."
            )

    def _validate_coords(self, pred: torch.Tensor, coords: torch.Tensor | None) -> None:
        if self.backbone_out_dim != 1:
            return
        if coords is None:
            raise ValueError(
                "ElasticityDeviatoricStressConstraint with backbone_out_dim=1 "
                "requires coords so the parameter head can consume [z, x, y]."
            )
        if (
            coords.ndim != 3
            or coords.shape[:2] != pred.shape[:2]
            or coords.shape[-1] < 2
        ):
            raise ValueError(
                "ElasticityDeviatoricStressConstraint expects coords with shape "
                f"(batch, n_points, >=2), got {tuple(coords.shape)!r}."
            )

    def _spectral_params(
        self,
        pred: torch.Tensor,
        coords: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        self._validate_pred(pred)
        if self.backbone_out_dim == 2:
            theta_raw, log_lambda_raw = pred.unbind(dim=-1)
            aux = {}
        else:
            self._validate_coords(pred, coords)
            assert coords is not None
            assert self.param_head is not None
            head_input = torch.cat((pred, coords[..., :2]), dim=-1)
            theta_raw, log_lambda_raw = self.param_head(head_input).unbind(dim=-1)
            aux = {
                "param_head_input_z": pred,
                "param_head_input_x": coords[..., 0:1],
                "param_head_input_y": coords[..., 1:2],
            }

        log_lambda = self.max_log_lambda * torch.tanh(log_lambda_raw)
        return theta_raw, log_lambda, {"log_lambda_raw": log_lambda_raw, **aux}

    def _physical_stress(
        self,
        pred: torch.Tensor,
        coords: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        theta_raw, log_lambda, aux = self._spectral_params(pred, coords)

        right_cauchy_green, lambda_stretch = _right_cauchy_green_from_spectral_params(
            theta_raw,
            log_lambda,
        )
        i1 = right_cauchy_green[..., 0, 0] + right_cauchy_green[..., 1, 1]
        c_squared = torch.matmul(right_cauchy_green, right_cauchy_green)
        tr_c_squared = c_squared[..., 0, 0] + c_squared[..., 1, 1]
        i2 = 0.5 * (i1.square() - tr_c_squared)

        identity = _identity_like(right_cauchy_green)
        stress = 2.0 * self.c1 * identity + 2.0 * self.c2 * (
            i1[..., None, None] * identity - right_cauchy_green
        )
        stress_trace = stress[..., 0, 0] + stress[..., 1, 1]
        stress_dev = stress - 0.5 * stress_trace[..., None, None] * identity
        stress_dev_inner = stress_dev.square().sum(dim=(-1, -2))
        sigma_vm = (1.5 * stress_dev_inner).clamp_min(0.0).sqrt().unsqueeze(-1)
        det_c = _det2(right_cauchy_green)

        aux.update(
            {
                "theta_raw": theta_raw.unsqueeze(-1),
                "theta": _wrap_angle(theta_raw).unsqueeze(-1),
                "log_lambda": log_lambda.unsqueeze(-1),
                "lambda": lambda_stretch.unsqueeze(-1),
                "right_cauchy_green_c11": right_cauchy_green[..., 0, 0].unsqueeze(-1),
                "right_cauchy_green_c12": right_cauchy_green[..., 0, 1].unsqueeze(-1),
                "right_cauchy_green_c22": right_cauchy_green[..., 1, 1].unsqueeze(-1),
                "det_c": det_c.unsqueeze(-1),
                "i1": i1.unsqueeze(-1),
                "i2": i2.unsqueeze(-1),
                "stress_11": stress[..., 0, 0].unsqueeze(-1),
                "stress_22": stress[..., 1, 1].unsqueeze(-1),
                "stress_12": stress[..., 0, 1].unsqueeze(-1),
                "stress_trace": stress_trace.unsqueeze(-1),
                "stress_dev_11": stress_dev[..., 0, 0].unsqueeze(-1),
                "stress_dev_22": stress_dev[..., 1, 1].unsqueeze(-1),
                "stress_dev_12": stress_dev[..., 0, 1].unsqueeze(-1),
                "stress_dev_inner": stress_dev_inner.unsqueeze(-1),
            }
        )
        aux["log_lambda_raw"] = aux["log_lambda_raw"].unsqueeze(-1)
        return sigma_vm, aux

    def _diagnostics(self, sigma_physical: torch.Tensor, aux: dict) -> dict:
        det_error = (aux["det_c"] - 1.0).abs()
        diagnostics = {
            "constraint/sigma_min": ConstraintDiagnostic(
                value=sigma_physical.min(),
                reduce="min",
            ),
            "constraint/sigma_mean": ConstraintDiagnostic(
                value=sigma_physical.mean(),
                reduce="mean",
            ),
            "constraint/sigma_max": ConstraintDiagnostic(
                value=sigma_physical.max(),
                reduce="max",
            ),
            "constraint/det_c_abs_error_mean": ConstraintDiagnostic(
                value=det_error.mean(),
                reduce="mean",
            ),
            "constraint/det_c_abs_error_max": ConstraintDiagnostic(
                value=det_error.max(),
                reduce="max",
            ),
            "constraint/i1_mean": ConstraintDiagnostic(
                value=aux["i1"].mean(),
                reduce="mean",
            ),
            "constraint/i2_mean": ConstraintDiagnostic(
                value=aux["i2"].mean(),
                reduce="mean",
            ),
            "constraint/lambda_mean": ConstraintDiagnostic(
                value=aux["lambda"].mean(),
                reduce="mean",
            ),
            "constraint/lambda_min": ConstraintDiagnostic(
                value=aux["lambda"].min(),
                reduce="min",
            ),
            "constraint/lambda_max": ConstraintDiagnostic(
                value=aux["lambda"].max(),
                reduce="max",
            ),
            "constraint/stress_trace_mean": ConstraintDiagnostic(
                value=aux["stress_trace"].mean(),
                reduce="mean",
            ),
            "constraint/stress_dev_inner_mean": ConstraintDiagnostic(
                value=aux["stress_dev_inner"].mean(),
                reduce="mean",
            ),
            "constraint/theta_std": ConstraintDiagnostic(
                value=aux["theta"].std(unbiased=False),
                reduce="mean",
            ),
            "constraint/log_lambda_std": ConstraintDiagnostic(
                value=aux["log_lambda"].std(unbiased=False),
                reduce="mean",
            ),
            "constraint/log_lambda_raw_mean": ConstraintDiagnostic(
                value=aux["log_lambda_raw"].mean(),
                reduce="mean",
            ),
            "constraint/log_lambda_raw_std": ConstraintDiagnostic(
                value=aux["log_lambda_raw"].std(unbiased=False),
                reduce="mean",
            ),
        }
        return diagnostics

    def forward(self, *, pred, coords=None, return_aux=False, **_unused):
        sigma_physical, aux = self._physical_stress(pred, coords)
        sigma = encode_target(sigma_physical, self.target_normalizer)

        if return_aux:
            return self.as_output(
                sigma,
                aux={"pred_base": pred, "sigma_physical": sigma_physical, **aux},
                diagnostics=self._diagnostics(sigma_physical, aux),
            )
        return sigma
