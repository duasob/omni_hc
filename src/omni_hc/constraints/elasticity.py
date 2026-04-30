from __future__ import annotations

import torch

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


class ElasticityDeviatoricStressConstraint(ConstraintModule):
    """
    Maps two backbone channels to 2D incompressible hyperelastic von Mises stress.

    The backbone output is interpreted as (theta, log_lambda), which defines
    the Right Cauchy-Green tensor

        C = R(theta) diag(lambda^2, lambda^-2) R(theta)^T.

    This enforces C symmetric positive definite and det(C) = 1 by construction.
    The returned scalar is computed from the 2D deviatoric Second
    Piola-Kirchhoff stress.
    """

    name = "elasticity_deviatoric_stress"

    def __init__(
        self,
        *,
        backbone_out_dim: int = 2,
        target_out_dim: int = 1,
        c1: float = 1.863e5,
        c2: float = 9.79e3,
        max_log_lambda: float = 8.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.backbone_out_dim = int(backbone_out_dim)
        self.target_out_dim = int(target_out_dim)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.max_log_lambda = float(max_log_lambda)
        self.eps = float(eps)
        self.target_normalizer = None

        if self.backbone_out_dim != 2:
            raise ValueError(
                "ElasticityDeviatoricStressConstraint expects "
                f"backbone_out_dim=2, got {self.backbone_out_dim}."
            )
        if self.target_out_dim != 1:
            raise ValueError(
                "ElasticityDeviatoricStressConstraint returns one scalar stress "
                f"channel, got target_out_dim={self.target_out_dim}."
            )

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def _validate_pred(self, pred: torch.Tensor) -> None:
        if pred.ndim != 3 or pred.shape[-1] != self.backbone_out_dim:
            raise ValueError(
                "ElasticityDeviatoricStressConstraint expects backbone output "
                f"with shape (batch, n_points, 2), got {tuple(pred.shape)!r}."
            )

    def _physical_stress(self, pred: torch.Tensor) -> tuple[torch.Tensor, dict]:
        self._validate_pred(pred)
        theta_raw, log_lambda_raw = pred.unbind(dim=-1)
        # This is a bit hacky. 
        # Large values lead to numerical instability
        log_lambda = log_lambda_raw.clamp( 
            -self.max_log_lambda,
            self.max_log_lambda,
        )
q
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

        aux = {
            "theta_raw": theta_raw.unsqueeze(-1),
            "theta": _wrap_angle(theta_raw).unsqueeze(-1),
            "log_lambda_raw": log_lambda_raw.unsqueeze(-1),
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
        }
        return diagnostics

    def forward(self, *, pred, return_aux=False, **_unused):
        sigma_physical, aux = self._physical_stress(pred)
        sigma = encode_target(sigma_physical, self.target_normalizer)

        if return_aux:
            return self.as_output(
                sigma,
                aux={"pred_base": pred, "sigma_physical": sigma_physical, **aux},
                diagnostics=self._diagnostics(sigma_physical, aux),
            )
        return sigma


class ElasticityDeviatoricStressFromCConstraint(ConstraintModule):
    """
    Maps backbone channels to a raw 2x2 C tensor and computes von Mises stress.

    The backbone output is interpreted as either (c11, c12, c22) or
    (c11, c12, c21, c22). No SPD or det(C)=1 enforcement is applied.
    """

    name = "elasticity_deviatoric_stress_from_c"

    def __init__(
        self,
        *,
        backbone_out_dim: int = 3,
        target_out_dim: int = 1,
        # From geo-fno paper
        c1: float = 1.863e5,
        c2: float = 9.79e3,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.backbone_out_dim = int(backbone_out_dim)
        self.target_out_dim = int(target_out_dim)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.eps = float(eps)
        self.target_normalizer = None

        if self.backbone_out_dim not in {3, 4}:
            raise ValueError(
                "ElasticityDeviatoricStressFromCConstraint expects "
                f"backbone_out_dim=3 or 4, got {self.backbone_out_dim}."
            )
        if self.target_out_dim != 1:
            raise ValueError(
                "ElasticityDeviatoricStressFromCConstraint returns one scalar stress "
                f"channel, got target_out_dim={self.target_out_dim}."
            )

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def _validate_pred(self, pred: torch.Tensor) -> None:
        if pred.ndim != 3 or pred.shape[-1] != self.backbone_out_dim:
            raise ValueError(
                "ElasticityDeviatoricStressFromCConstraint expects backbone output "
                f"with shape (batch, n_points, {self.backbone_out_dim}), got "
                f"{tuple(pred.shape)!r}."
            )

    def _assemble_c(self, pred: torch.Tensor) -> tuple[torch.Tensor, dict]:
        self._validate_pred(pred)
        if self.backbone_out_dim == 3:
            c11, c12, c22 = pred.unbind(dim=-1)
            c21 = c12
        else:
            c11, c12, c21, c22 = pred.unbind(dim=-1)
        right_cauchy_green = torch.stack(
            (
                torch.stack((c11, c12), dim=-1),
                torch.stack((c21, c22), dim=-1),
            ),
            dim=-2,
        )
        aux = {
            "right_cauchy_green_c11": c11.unsqueeze(-1),
            "right_cauchy_green_c12": c12.unsqueeze(-1),
            "right_cauchy_green_c21": c21.unsqueeze(-1),
            "right_cauchy_green_c22": c22.unsqueeze(-1),
        }
        return right_cauchy_green, aux

    def _physical_stress(self, pred: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Compute von Mises stress from C:
            stress = 2 * c1 * I + 2 * c2 * (tr(C) * I - C)
            stress_dev = stress - 0.5 * tr(stress) * I
        """

        right_cauchy_green, aux = self._assemble_c(pred)
        i1 = right_cauchy_green[..., 0, 0] + right_cauchy_green[..., 1, 1]
        c_squared = torch.matmul(right_cauchy_green, right_cauchy_green)
        tr_c_squared = c_squared[..., 0, 0] + c_squared[..., 1, 1]

        i2 = 0.5 * (i1.square() - tr_c_squared)  # i2 not used. just for diagnostics

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
            "constraint/stress_trace_mean": ConstraintDiagnostic(
                value=aux["stress_trace"].mean(),
                reduce="mean",
            ),
            "constraint/stress_dev_inner_mean": ConstraintDiagnostic(
                value=aux["stress_dev_inner"].mean(),
                reduce="mean",
            ),
        }
        return diagnostics

    def forward(self, *, pred, return_aux=False, **_unused):
        sigma_physical, aux = self._physical_stress(pred)
        sigma = encode_target(sigma_physical, self.target_normalizer)

        if return_aux:
            return self.as_output(
                sigma,
                aux={"pred_base": pred, "sigma_physical": sigma_physical, **aux},
                diagnostics=self._diagnostics(sigma_physical, aux),
            )
        return sigma
