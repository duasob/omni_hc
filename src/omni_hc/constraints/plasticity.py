from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import ConstraintDiagnostic, ConstraintModule


class PlasticityMeshConsistencyConstraint(ConstraintModule):
    """
    Reconstruct plasticity coordinates from ordered spacings.

    The backbone emits three channels on the structured grid:
    - channel 0: x anchor values on the i=0 edge, read at every j
    - channel 1: raw positive spacing along i, read on i=0..I-2
    - channel 2: raw positive spacing along j, read on j=0..J-2

    Output channels match the dataset target: deformed x/y coordinates followed
    by displacement u_x/u_y relative to the undeformed physical grid.
    """

    name = "plasticity_mesh_consistency"

    def __init__(
        self,
        *,
        shapelist,
        backbone_out_dim: int = 3,
        target_out_dim: int = 4,
        x_left: float = 0.35000038,
        x_right: float = -49.65,
        y_top: float = 14.9,
        y_bottom: float = -0.099999905,
        spacing_activation: str = "softplus",
        min_spacing: float = 1.0e-6,
    ):
        super().__init__()
        if shapelist is None or len(tuple(shapelist)) != 2:
            raise ValueError(
                "PlasticityMeshConsistencyConstraint requires a 2D shapelist."
            )
        self.shapelist = tuple(int(v) for v in shapelist)
        self.backbone_out_dim = int(backbone_out_dim)
        self.target_out_dim = int(target_out_dim)
        if self.backbone_out_dim != 3:
            raise ValueError(
                "PlasticityMeshConsistencyConstraint expects backbone_out_dim=3, "
                f"got {self.backbone_out_dim}."
            )
        if self.target_out_dim != 4:
            raise ValueError(
                "PlasticityMeshConsistencyConstraint outputs the 4 plasticity target "
                f"channels, got target_out_dim={self.target_out_dim}."
            )
        self.spacing_activation = str(spacing_activation).lower()
        if self.spacing_activation not in {"softplus", "exp"}:
            raise ValueError(
                "spacing_activation must be one of: softplus, exp; "
                f"got {spacing_activation!r}."
            )
        self.min_spacing = float(min_spacing)

        i_count, j_count = self.shapelist
        if i_count < 2 or j_count < 2:
            raise ValueError(
                f"Plasticity grid must have at least 2x2 points, got {self.shapelist}."
            )

        material = self._make_material_grid(
            i_count=i_count,
            j_count=j_count,
            x_left=float(x_left),
            x_right=float(x_right),
            y_top=float(y_top),
            y_bottom=float(y_bottom),
        )
        orientation = torch.sign(self._signed_cell_areas(material[None, ...]).mean())
        if float(orientation.item()) == 0.0:
            orientation = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("material_grid", material, persistent=False)
        self.register_buffer("orientation_sign", orientation, persistent=False)
        self.register_buffer(
            "bottom_y",
            torch.tensor(float(y_bottom), dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _make_material_grid(
        *,
        i_count: int,
        j_count: int,
        x_left: float,
        x_right: float,
        y_top: float,
        y_bottom: float,
    ) -> torch.Tensor:
        x = torch.linspace(x_left, x_right, i_count, dtype=torch.float32)
        y = torch.linspace(y_top, y_bottom, j_count, dtype=torch.float32)
        xx = x[:, None].expand(i_count, j_count)
        yy = y[None, :].expand(i_count, j_count)
        return torch.stack((xx, yy), dim=-1)

    @staticmethod
    def _signed_cell_areas(coords: torch.Tensor) -> torch.Tensor:
        p00 = coords[:, :-1, :-1]
        p10 = coords[:, 1:, :-1]
        p11 = coords[:, 1:, 1:]
        p01 = coords[:, :-1, 1:]
        vertices = torch.stack((p00, p10, p11, p01), dim=-2)
        x = vertices[..., 0]
        y = vertices[..., 1]
        return 0.5 * (
            x.mul(torch.roll(y, shifts=-1, dims=-1)).sum(dim=-1)
            - y.mul(torch.roll(x, shifts=-1, dims=-1)).sum(dim=-1)
        )

    def _positive_spacing(self, raw: torch.Tensor) -> torch.Tensor:
        if self.spacing_activation == "exp":
            spacing = torch.exp(raw)
        else:
            spacing = F.softplus(raw)
        return spacing + self.min_spacing

    def forward(self, *, pred, return_aux=False, **_unused):
        if pred.ndim != 3 or pred.shape[-1] != self.backbone_out_dim:
            raise ValueError(
                "PlasticityMeshConsistencyConstraint expects pred with shape "
                f"(batch, n_points, {self.backbone_out_dim}), got {tuple(pred.shape)}."
            )

        batch_size, n_points, _ = pred.shape
        i_count, j_count = self.shapelist
        expected_points = i_count * j_count
        if n_points != expected_points:
            raise ValueError(
                f"PlasticityMeshConsistencyConstraint expected {expected_points} points "
                f"for shapelist={self.shapelist}, got {n_points}."
            )

        raw = pred.reshape(batch_size, i_count, j_count, self.backbone_out_dim)
        x_anchor = raw[:, 0, :, 0]
        dx = self._positive_spacing(raw[:, :-1, :, 1])
        dy = self._positive_spacing(raw[:, :, :-1, 2])

        x = torch.empty(
            batch_size,
            i_count,
            j_count,
            device=pred.device,
            dtype=pred.dtype,
        )
        x[:, 0, :] = x_anchor
        x[:, 1:, :] = x_anchor[:, None, :] - torch.cumsum(dx, dim=1)

        y = torch.empty(
            batch_size,
            i_count,
            j_count,
            device=pred.device,
            dtype=pred.dtype,
        )
        bottom_y = self.bottom_y.to(device=pred.device, dtype=pred.dtype)
        y[:, :, -1] = bottom_y
        y[:, :, :-1] = bottom_y + torch.flip(
            torch.cumsum(torch.flip(dy, dims=(2,)), dim=2),
            dims=(2,),
        )

        coords = torch.stack((x, y), dim=-1)
        material = self.material_grid.to(device=pred.device, dtype=pred.dtype)
        displacement = coords - material[None, ...]
        out = torch.cat((coords, displacement), dim=-1).reshape(
            batch_size,
            expected_points,
            self.target_out_dim,
        )

        if return_aux:
            orientation_sign = self.orientation_sign.to(
                device=pred.device,
                dtype=pred.dtype,
            )
            oriented_cell_area = self._signed_cell_areas(coords) * orientation_sign
            bottom_y_error = (y[:, :, -1] - bottom_y).abs()
            axis_order_margin = torch.minimum(dx.min(), dy.min())
            neg_spacing = torch.cat([dx.reshape(-1), dy.reshape(-1)], dim=0) < 0
            neg_spacing_count = neg_spacing.to(torch.float32).sum()
            diagnostics = {
                "constraint/min_dx": ConstraintDiagnostic(
                    value=dx.min(),
                    reduce="min",
                ),
                "constraint/min_dy": ConstraintDiagnostic(
                    value=dy.min(),
                    reduce="min",
                ),
                "constraint/min_oriented_cell_area": ConstraintDiagnostic(
                    value=oriented_cell_area.min(),
                    reduce="min",
                ),
                "constraint/bottom_y_abs_error_max": ConstraintDiagnostic(
                    value=bottom_y_error.max(),
                    reduce="max",
                ),
                "constraint/axis_order_margin_min": ConstraintDiagnostic(
                    value=axis_order_margin,
                    reduce="min",
                ),
                "constraint/neg_spacing_count": ConstraintDiagnostic(
                    value=neg_spacing_count,
                    reduce="sum",
                ),
                "constraint/neg_spacing_fraction": ConstraintDiagnostic(
                    value=neg_spacing.to(torch.float32).mean(),
                    reduce="mean",
                ),
            }
            return self.as_output(
                out,
                aux={
                    "pred_base": pred,
                    "x_anchor": x_anchor,
                    "dx": dx,
                    "dy": dy,
                },
                diagnostics=diagnostics,
            )
        return out
