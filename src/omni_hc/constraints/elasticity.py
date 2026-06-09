from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn as nn

from .base import (
    _BUILD_META_KEYS,
    ConstrainedModel,
    ConstraintDiagnostic,
    ConstraintModule,
)
from .utils.boundary_ops import encode_target
from .utils.hooks import ForwardHookLatentExtractor


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


def _principal_stretches(
    mean_log_stretch: torch.Tensor,
    deviatoric_log_stretch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Recover in-plane stretches and the incompressible thickness stretch.

    The sign of d only swaps the two in-plane principal directions.
    """
    lambda_1 = torch.exp(mean_log_stretch + deviatoric_log_stretch)
    lambda_2 = torch.exp(mean_log_stretch - deviatoric_log_stretch)
    lambda_3 = torch.exp(-2.0 * mean_log_stretch)
    return lambda_1, lambda_2, lambda_3


def _plane_stress_principal_cauchy(
    *,
    lambda_1: torch.Tensor,
    lambda_2: torch.Tensor,
    lambda_3: torch.Tensor,
    c1: float,
    c2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the principal Cauchy stresses and incompressibility pressure."""
    lambda_1_sq = lambda_1.square()
    lambda_2_sq = lambda_2.square()
    lambda_3_sq = lambda_3.square()
    inv_lambda_1_sq = lambda_1_sq.reciprocal()
    inv_lambda_2_sq = lambda_2_sq.reciprocal()
    inv_lambda_3_sq = lambda_3_sq.reciprocal()

    # This pressure makes the out-of-plane Cauchy stress exactly zero.
    pressure = 2.0 * c1 * lambda_3_sq - 2.0 * c2 * inv_lambda_3_sq
    # Substitute the plane-stress pressure before evaluating the in-plane
    # stresses. This avoids subtracting two nearly equal O(C1) terms.
    sigma_1 = (
        2.0 * c1 * (lambda_1_sq - lambda_3_sq)
        - 2.0 * c2 * (inv_lambda_1_sq - inv_lambda_3_sq)
    )
    sigma_2 = (
        2.0 * c1 * (lambda_2_sq - lambda_3_sq)
        - 2.0 * c2 * (inv_lambda_2_sq - inv_lambda_3_sq)
    )
    sigma_3 = torch.zeros_like(sigma_1)
    return sigma_1, sigma_2, sigma_3, pressure


class ElasticityPlaneStressVMConstraint(ConstraintModule):
    """
    Plane-stress von Mises reparameterization for an incompressible membrane.

    With backbone_out_dim != 2, a pointwise head maps the vector latent and
    physical coordinates [z, x, y] to raw mean and deviatoric in-plane log
    stretches (m, d). With backbone_out_dim=2, the backbone output is
    interpreted directly as the raw (m, d) parameters.

        lambda_1 = exp(m + d)
        lambda_2 = exp(m - d)
        lambda_3 = exp(-2m)

    This enforces 3D incompressibility. The pressure is selected to enforce the
    plane-stress condition sigma_3 = 0 before computing von Mises stress.
    """

    name = "elasticity_plane_stress_vm"

    def __init__(
        self,
        *,
        backbone_out_dim: int = 32,
        target_out_dim: int = 1,
        c1: float = 1.863e5,
        c2: float = 9.79e3,
        max_mean_log_stretch: float = 1.0e-3,
        max_deviatoric_log_stretch: float = 1.0e-3,
        head_hidden_dim: int = 32,
        head_layers: int = 2,
        head_init_scale: float = 1e-3,
        mean_log_stretch_bias: float = 0.0,
        deviatoric_log_stretch_bias: float = 0.4,
    ) -> None:
        super().__init__()
        self.backbone_out_dim = int(backbone_out_dim)
        self.target_out_dim = int(target_out_dim)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.max_mean_log_stretch = float(max_mean_log_stretch)
        self.max_deviatoric_log_stretch = float(max_deviatoric_log_stretch)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_layers = int(head_layers)
        self.head_init_scale = float(head_init_scale)
        self.mean_log_stretch_bias = float(mean_log_stretch_bias)
        self.deviatoric_log_stretch_bias = float(deviatoric_log_stretch_bias)
        self.target_normalizer = None

        if self.backbone_out_dim <= 0:
            raise ValueError("backbone_out_dim must be positive.")
        if self.target_out_dim != 1:
            raise ValueError(
                "ElasticityPlaneStressVMConstraint returns one scalar stress "
                f"channel, got target_out_dim={self.target_out_dim}."
            )
        if self.max_mean_log_stretch <= 0.0:
            raise ValueError("max_mean_log_stretch must be positive.")
        if self.max_deviatoric_log_stretch <= 0.0:
            raise ValueError("max_deviatoric_log_stretch must be positive.")

        self.param_head = None
        if self.backbone_out_dim != 2:
            self.param_head = _make_mlp(
                in_dim=self.backbone_out_dim + 2,
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
                    [
                        self.mean_log_stretch_bias,
                        self.deviatoric_log_stretch_bias,
                    ],
                    dtype=final.bias.dtype,
                    device=final.bias.device,
                )
            )

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def _validate_pred(self, pred: torch.Tensor) -> None:
        if pred.ndim != 3 or pred.shape[-1] != self.backbone_out_dim:
            raise ValueError(
                "ElasticityPlaneStressVMConstraint expects backbone output "
                f"with shape (batch, n_points, {self.backbone_out_dim}), got "
                f"{tuple(pred.shape)!r}."
            )

    def _validate_coords(self, pred: torch.Tensor, coords: torch.Tensor | None) -> None:
        if self.backbone_out_dim == 2:
            return
        if coords is None:
            raise ValueError(
                "ElasticityPlaneStressVMConstraint with a learned parameter "
                "head requires coords so it can consume [z, x, y]."
            )
        if (
            coords.ndim != 3
            or coords.shape[:2] != pred.shape[:2]
            or coords.shape[-1] < 2
        ):
            raise ValueError(
                "ElasticityPlaneStressVMConstraint expects coords with shape "
                f"(batch, n_points, >=2), got {tuple(coords.shape)!r}."
            )

    def _stretch_params(
        self,
        pred: torch.Tensor,
        coords: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        self._validate_pred(pred)
        if self.backbone_out_dim == 2:
            mean_raw, deviatoric_raw = pred.unbind(dim=-1)
            aux = {}
        else:
            self._validate_coords(pred, coords)
            assert coords is not None
            assert self.param_head is not None
            head_input = torch.cat((pred, coords[..., :2]), dim=-1)
            mean_raw, deviatoric_raw = self.param_head(head_input).unbind(dim=-1)
            aux = {
                "param_head_input_z": pred,
                "param_head_input_x": coords[..., 0:1],
                "param_head_input_y": coords[..., 1:2],
            }

        mean_log_stretch = self.max_mean_log_stretch * torch.tanh(mean_raw)
        # Canonically order the in-plane stretches as lambda_1 <= lambda_2.
        # Squaring keeps the map smooth and bounded while retaining d = 0.
        deviatoric_log_stretch = (
            -self.max_deviatoric_log_stretch * torch.tanh(deviatoric_raw).square()
        )
        return mean_log_stretch, deviatoric_log_stretch, {
            "mean_log_stretch_raw": mean_raw,
            "deviatoric_log_stretch_raw": deviatoric_raw,
            **aux,
        }

    def _physical_stress(
        self,
        pred: torch.Tensor,
        coords: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        mean_log_stretch, deviatoric_log_stretch, aux = self._stretch_params(
            pred, coords
        )
        lambda_1, lambda_2, lambda_3 = _principal_stretches(
            mean_log_stretch,
            deviatoric_log_stretch,
        )
        sigma_1, sigma_2, sigma_3, pressure = _plane_stress_principal_cauchy(
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            c1=self.c1,
            c2=self.c2,
        )

        sigma_vm_sq = sigma_1.square() - sigma_1 * sigma_2 + sigma_2.square()
        sigma_vm = (sigma_vm_sq.clamp_min(0.0) + 1e-8).sqrt().unsqueeze(-1)
        full_det_f = lambda_1 * lambda_2 * lambda_3
        in_plane_det_f = lambda_1 * lambda_2

        aux.update(
            {
                "mean_log_stretch": mean_log_stretch.unsqueeze(-1),
                "deviatoric_log_stretch": deviatoric_log_stretch.unsqueeze(-1),
                "lambda_1": lambda_1.unsqueeze(-1),
                "lambda_2": lambda_2.unsqueeze(-1),
                "lambda_3": lambda_3.unsqueeze(-1),
                "in_plane_det_f": in_plane_det_f.unsqueeze(-1),
                "full_det_f": full_det_f.unsqueeze(-1),
                "c_33": lambda_3.square().unsqueeze(-1),
                "pressure": pressure.unsqueeze(-1),
                "principal_cauchy_stress_1": sigma_1.unsqueeze(-1),
                "principal_cauchy_stress_2": sigma_2.unsqueeze(-1),
                "principal_cauchy_stress_3": sigma_3.unsqueeze(-1),
                "sigma_vm_squared": sigma_vm_sq.unsqueeze(-1),
            }
        )
        aux["mean_log_stretch_raw"] = aux["mean_log_stretch_raw"].unsqueeze(-1)
        aux["deviatoric_log_stretch_raw"] = aux[
            "deviatoric_log_stretch_raw"
        ].unsqueeze(-1)
        return sigma_vm, aux

    def _diagnostics(self, sigma_physical: torch.Tensor, aux: dict) -> dict:
        det_error = (aux["full_det_f"] - 1.0).abs()
        plane_stress_error = aux["principal_cauchy_stress_3"].abs()
        return {
            "constraint/sigma_min": ConstraintDiagnostic(
                value=sigma_physical.min(), reduce="min"
            ),
            "constraint/sigma_mean": ConstraintDiagnostic(
                value=sigma_physical.mean(), reduce="mean"
            ),
            "constraint/sigma_max": ConstraintDiagnostic(
                value=sigma_physical.max(), reduce="max"
            ),
            "constraint/full_det_f_abs_error_mean": ConstraintDiagnostic(
                value=det_error.mean(), reduce="mean"
            ),
            "constraint/full_det_f_abs_error_max": ConstraintDiagnostic(
                value=det_error.max(), reduce="max"
            ),
            "constraint/plane_stress_abs_error_mean": ConstraintDiagnostic(
                value=plane_stress_error.mean(), reduce="mean"
            ),
            "constraint/plane_stress_abs_error_max": ConstraintDiagnostic(
                value=plane_stress_error.max(), reduce="max"
            ),
            "constraint/lambda_1_min": ConstraintDiagnostic(
                value=aux["lambda_1"].min(), reduce="min"
            ),
            "constraint/lambda_1_max": ConstraintDiagnostic(
                value=aux["lambda_1"].max(), reduce="max"
            ),
            "constraint/lambda_2_min": ConstraintDiagnostic(
                value=aux["lambda_2"].min(), reduce="min"
            ),
            "constraint/lambda_2_max": ConstraintDiagnostic(
                value=aux["lambda_2"].max(), reduce="max"
            ),
            "constraint/lambda_3_min": ConstraintDiagnostic(
                value=aux["lambda_3"].min(), reduce="min"
            ),
            "constraint/lambda_3_max": ConstraintDiagnostic(
                value=aux["lambda_3"].max(), reduce="max"
            ),
            "constraint/in_plane_det_f_mean": ConstraintDiagnostic(
                value=aux["in_plane_det_f"].mean(), reduce="mean"
            ),
            "constraint/mean_log_stretch_raw_mean": ConstraintDiagnostic(
                value=aux["mean_log_stretch_raw"].mean(), reduce="mean"
            ),
            "constraint/mean_log_stretch_raw_std": ConstraintDiagnostic(
                value=aux["mean_log_stretch_raw"].std(unbiased=False),
                reduce="mean",
            ),
            "constraint/deviatoric_log_stretch_raw_mean": ConstraintDiagnostic(
                value=aux["deviatoric_log_stretch_raw"].mean(), reduce="mean"
            ),
            "constraint/deviatoric_log_stretch_raw_std": ConstraintDiagnostic(
                value=aux["deviatoric_log_stretch_raw"].std(unbiased=False),
                reduce="mean",
            ),
        }

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

    @classmethod
    def log_media(cls, ctx) -> dict[str, str]:
        aux = ctx.aux_tensors
        if not aux:
            return {}
        if ctx.out_dir is None:
            from omni_hc.training.logging_utils import log_elasticity_latent_panels

            log_elasticity_latent_panels(
                ctx.coords,
                aux,
                prefix=ctx.prefix,
                epoch=ctx.epoch,
                step=ctx.step,
                point_size=float(
                    (ctx.cfg.get("wandb_logging") or {}).get("point_size", 24.0)
                ),
            )
            return {}
        from omni_hc.training.logging_utils import save_elasticity_latent_panels

        return save_elasticity_latent_panels(
            ctx.coords,
            aux,
            out_dir=ctx.out_dir,
            prefix=ctx.prefix,
            point_size=float(
                (ctx.cfg.get("wandb_logging") or {}).get("point_size", 24.0)
            ),
        )
