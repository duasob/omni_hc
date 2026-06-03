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
        # Structure of the grid
        # for reference to the orientation
        # and bottom anchor
        x_left: float = 0.35,
        x_right: float = -49.65,
        y_top: float = 14.9,
        y_bottom: float = -0.1,  # Used as bottom anchor
        spacing_activation: str = "softplus",
        # On the data around 0.1. Could enforce it or let the model learn it
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
            flipped_cell = oriented_cell_area < 0
            flipped_cell_count = flipped_cell.to(torch.float32).sum(dim=(-1, -2))
            flipped_cell_fraction = flipped_cell.to(torch.float32).mean(dim=(-1, -2))
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
                "constraint/neg_spacing_worst_sample_fraction": ConstraintDiagnostic(
                    value=neg_spacing.to(torch.float32).mean(),
                    reduce="max",
                ),
                "constraint/neg_spacing_fraction": ConstraintDiagnostic(
                    value=neg_spacing.to(torch.float32).mean(),
                    reduce="max",
                ),
                "constraint/flipped_cell_count_mean": ConstraintDiagnostic(
                    value=flipped_cell_count.mean(),
                    reduce="mean",
                ),
                "constraint/flipped_cell_count_worst": ConstraintDiagnostic(
                    value=flipped_cell_count.max(),
                    reduce="max",
                ),
                "constraint/flipped_cell_fraction_mean": ConstraintDiagnostic(
                    value=flipped_cell_fraction.mean(),
                    reduce="mean",
                ),
                "constraint/flipped_cell_fraction_worst": ConstraintDiagnostic(
                    value=flipped_cell_fraction.max(),
                    reduce="max",
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


class PlasticityEnvelopeConstraint(ConstraintModule):
    """
    Reconstruct plasticity coordinates under a moving die envelope.

    The backbone emits four channels on the structured grid:
    - channel 0: x anchor values on the i=0 edge, read at every j
    - channel 1: raw positive spacing along i, read on i=0..I-2
    - channel 2: raw vertical spacing allocation, read on j=0..J-2
    - channel 3: raw top clearance/gap, read on the j=0 edge for every i

    The bottom edge is fixed at ``y_bottom``.  Each column receives a vertical
    envelope height from either the input die profile or a constant moving
    inlet height.  Positive vertical spacings are normalized to fill the
    envelope height minus a learned non-negative top gap.
    """

    name = "plasticity_envelope"

    def __init__(
        self,
        *,
        shapelist,
        backbone_out_dim: int = 4,
        target_out_dim: int = 4,
        # Mesh structure
        x_left: float = 0.35,
        x_right: float = -49.65,
        y_top: float = 14.9,
        y_bottom: float = -0.1,
        spacing_activation: str = "softplus",
        min_spacing: float = 1.0e-6,
        min_gap: float = 1.0e-6,
        top_height: float | None = None,
        envelope_source: str = "fx",
        die_speed: float = 6.0,
        time_duration: float = 1.0,
    ):
        super().__init__()
        if shapelist is None or len(tuple(shapelist)) != 2:
            raise ValueError("PlasticityEnvelopeConstraint requires a 2D shapelist.")
        self.shapelist = tuple(int(v) for v in shapelist)
        self.backbone_out_dim = int(backbone_out_dim)
        self.target_out_dim = int(target_out_dim)
        if self.backbone_out_dim != 4:
            raise ValueError(
                "PlasticityEnvelopeConstraint expects backbone_out_dim=4, "
                f"got {self.backbone_out_dim}."
            )
        if self.target_out_dim != 4:
            raise ValueError(
                "PlasticityEnvelopeConstraint outputs the 4 plasticity target "
                f"channels, got target_out_dim={self.target_out_dim}."
            )
        self.spacing_activation = str(spacing_activation).lower()
        if self.spacing_activation not in {"softplus", "exp"}:
            raise ValueError(
                "spacing_activation must be one of: softplus, exp; "
                f"got {spacing_activation!r}."
            )
        self.min_spacing = float(min_spacing)
        self.min_gap = float(min_gap)
        if self.min_spacing < 0.0:
            raise ValueError(f"min_spacing must be non-negative, got {min_spacing}.")
        if self.min_gap < 0.0:
            raise ValueError(f"min_gap must be non-negative, got {min_gap}.")
        self.envelope_source = str(envelope_source).lower()
        if self.envelope_source not in {"fx", "constant"}:
            raise ValueError(
                "envelope_source must be one of: fx, constant; "
                f"got {envelope_source!r}."
            )
        self.die_speed = float(die_speed)
        self.time_duration = float(time_duration)
        self.input_normalizer = None

        i_count, j_count = self.shapelist
        if i_count < 2 or j_count < 2:
            raise ValueError(
                f"Plasticity grid must have at least 2x2 points, got {self.shapelist}."
            )

        # Build on top of previous constraint
        material = PlasticityMeshConsistencyConstraint._make_material_grid(
            i_count=i_count,
            j_count=j_count,
            x_left=float(x_left),
            x_right=float(x_right),
            y_top=float(y_top),
            y_bottom=float(y_bottom),
        )
        orientation = torch.sign(
            PlasticityMeshConsistencyConstraint._signed_cell_areas(
                material[None, ...],
            ).mean()
        )
        if float(orientation.item()) == 0.0:
            orientation = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("material_grid", material, persistent=False)
        self.register_buffer("orientation_sign", orientation, persistent=False)
        self.register_buffer(
            "bottom_y",
            torch.tensor(float(y_bottom), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "constant_envelope_y",
            torch.tensor(float(y_top), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "top_height",
            self._make_top_height_profile(
                top_height=top_height,
                y_top=float(y_top),
                i_count=i_count,
            ),
            persistent=False,
        )

    @staticmethod
    def _make_top_height_profile(
        *,
        top_height,
        y_top: float,
        i_count: int,
    ) -> torch.Tensor:
        if top_height is None:
            return torch.full((i_count,), float(y_top), dtype=torch.float32)
        top_height_tensor = torch.as_tensor(top_height, dtype=torch.float32)
        if top_height_tensor.ndim == 0:
            return top_height_tensor.expand(i_count).clone()
        top_height_tensor = top_height_tensor.reshape(-1)
        if top_height_tensor.numel() != i_count:
            raise ValueError(
                "top_height must be a scalar or a sequence with one value per "
                f"i index; got {top_height_tensor.numel()} values for i_count={i_count}."
            )
        return top_height_tensor

    def set_input_normalizer(self, normalizer) -> None:
        self.input_normalizer = normalizer

    def _decode_fx(self, fx: torch.Tensor) -> torch.Tensor:
        if self.input_normalizer is None:
            return fx
        return self.input_normalizer.decode(fx)

    def _positive_spacing(self, raw: torch.Tensor) -> torch.Tensor:
        if self.spacing_activation == "exp":
            return torch.exp(raw)
        return F.softplus(raw)

    def _time(self, T: torch.Tensor | None, *, pred: torch.Tensor) -> torch.Tensor:
        batch_size = int(pred.shape[0])
        if T is None:
            return torch.zeros(batch_size, 1, device=pred.device, dtype=pred.dtype)
        time = T.to(device=pred.device, dtype=pred.dtype)
        if time.ndim == 1:
            time = time[:, None]
        return time.reshape(batch_size, -1)[:, 0:1]

    def _envelope_y(
        self,
        *,
        pred: torch.Tensor,
        fx: torch.Tensor | None,
        T: torch.Tensor | None,
        x_query: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_points, _ = pred.shape
        i_count, j_count = self.shapelist
        time = self._time(T, pred=pred)
        drop = self.die_speed * self.time_duration * time
        top_height_profile = self.top_height.to(
            device=pred.device,
            dtype=pred.dtype,
        ).expand(batch_size, -1)
        if self.envelope_source == "constant":
            envelope = self.constant_envelope_y.to(
                device=pred.device,
                dtype=pred.dtype,
            )
            profile = envelope.expand(batch_size, i_count) - drop
            capped_profile = torch.minimum(profile, top_height_profile)
            return self._interpolate_envelope_y(capped_profile, x_query)
        if fx is None:
            raise ValueError(
                "PlasticityEnvelopeConstraint with envelope_source='fx' requires fx."
            )
        if fx.ndim != 3 or fx.shape[0] != batch_size or fx.shape[1] != n_points:
            raise ValueError(
                "PlasticityEnvelopeConstraint expects fx with shape "
                f"(batch, {n_points}, channels), got {tuple(fx.shape)}."
            )
        fx_physical = self._decode_fx(fx).to(device=pred.device, dtype=pred.dtype)

        # Die is defined backward
        die_grid = fx_physical.reshape(batch_size, i_count, j_count, -1)[..., 0]
        die_profile = torch.flip(die_grid[:, :, 0], dims=(1,))
        moved_die_profile = die_profile - drop
        capped_profile = torch.minimum(moved_die_profile, top_height_profile)
        return self._interpolate_envelope_y(capped_profile, x_query)

    def _interpolate_envelope_y(
        self,
        envelope_y: torch.Tensor,
        x_query: torch.Tensor,
    ) -> torch.Tensor:
        reference_x = self.material_grid[:, 0, 0].to(
            device=x_query.device,
            dtype=x_query.dtype,
        )
        profile_y = envelope_y.to(device=x_query.device, dtype=x_query.dtype)
        if reference_x[0] > reference_x[-1]:
            interp_x = torch.flip(reference_x, dims=(0,))
            interp_y = torch.flip(profile_y, dims=(1,))
        else:
            interp_x = reference_x
            interp_y = profile_y

        x_clamped = x_query.clamp(
            min=float(interp_x[0].item()),
            max=float(interp_x[-1].item()),
        )
        upper = torch.searchsorted(interp_x, x_clamped.contiguous()).clamp(
            min=1,
            max=interp_x.numel() - 1,
        )
        lower = upper - 1
        x_lower = interp_x[lower]
        x_upper = interp_x[upper]
        y_lower = torch.gather(interp_y, dim=1, index=lower)
        y_upper = torch.gather(interp_y, dim=1, index=upper)
        weight = (x_clamped - x_lower) / (x_upper - x_lower).clamp_min(1.0e-12)
        return y_lower + weight * (y_upper - y_lower)

    def forward(self, *, pred, fx=None, T=None, return_aux=False, **_unused):
        if pred.ndim != 3 or pred.shape[-1] != self.backbone_out_dim:
            raise ValueError(
                "PlasticityEnvelopeConstraint expects pred with shape "
                f"(batch, n_points, {self.backbone_out_dim}), got {tuple(pred.shape)}."
            )

        batch_size, n_points, _ = pred.shape
        i_count, j_count = self.shapelist
        expected_points = i_count * j_count
        if n_points != expected_points:
            raise ValueError(
                f"PlasticityEnvelopeConstraint expected {expected_points} points "
                f"for shapelist={self.shapelist}, got {n_points}."
            )

        raw = pred.reshape(batch_size, i_count, j_count, self.backbone_out_dim)
        x_anchor = raw[:, 0, :, 0]
        dx = self._positive_spacing(raw[:, :-1, :, 1]) + self.min_spacing
        raw_dy = raw[:, :, :-1, 2]
        vertical_score = self._positive_spacing(raw_dy).clamp_min(1.0e-12)
        weights = vertical_score / vertical_score.sum(dim=2, keepdim=True)
        gap_fraction = torch.sigmoid(raw[:, :, 0, 3])

        x = torch.empty(
            batch_size,
            i_count,
            j_count,
            device=pred.device,
            dtype=pred.dtype,
        )
        x[:, 0, :] = x_anchor
        x[:, 1:, :] = x_anchor[:, None, :] - torch.cumsum(dx, dim=1)
        envelope_x = x[:, :, 0]
        envelope_y = self._envelope_y(pred=pred, fx=fx, T=T, x_query=envelope_x)

        bottom_y = self.bottom_y.to(device=pred.device, dtype=pred.dtype)
        envelope_height = envelope_y - bottom_y
        min_spacing_total = float(j_count - 1) * self.min_spacing
        min_budget_total = min_spacing_total + self.min_gap
        free_budget = (envelope_height - min_budget_total).clamp_min(0.0)
        gap = self.min_gap + free_budget * gap_fraction
        alloc_budget = free_budget * (1.0 - gap_fraction)
        dy = self.min_spacing + weights * alloc_budget[:, :, None]

        y = torch.empty(
            batch_size,
            i_count,
            j_count,
            device=pred.device,
            dtype=pred.dtype,
        )
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
            oriented_cell_area = (
                PlasticityMeshConsistencyConstraint._signed_cell_areas(coords)
                * orientation_sign
            )
            bottom_y_error = (y[:, :, -1] - bottom_y).abs()
            top_clearance = envelope_y - y[:, :, 0]
            top_violation = top_clearance < 0
            axis_order_margin = torch.minimum(dx.min(), dy.min())
            flipped_cell = oriented_cell_area < 0
            flipped_cell_count = flipped_cell.to(torch.float32).sum(dim=(-1, -2))
            flipped_cell_fraction = flipped_cell.to(torch.float32).mean(dim=(-1, -2))
            diagnostics = {
                "constraint/min_dx": ConstraintDiagnostic(value=dx.min(), reduce="min"),
                "constraint/min_dy": ConstraintDiagnostic(value=dy.min(), reduce="min"),
                "constraint/min_gap": ConstraintDiagnostic(
                    value=gap.min(), reduce="min"
                ),
                "constraint/mean_gap": ConstraintDiagnostic(
                    value=gap.mean(),
                    reduce="mean",
                ),
                "constraint/min_envelope_height": ConstraintDiagnostic(
                    value=envelope_height.min(),
                    reduce="min",
                ),
                "constraint/top_clearance_min": ConstraintDiagnostic(
                    value=top_clearance.min(),
                    reduce="min",
                ),
                "constraint/top_violation_count": ConstraintDiagnostic(
                    value=top_violation.to(torch.float32).sum(),
                    reduce="sum",
                ),
                "constraint/top_violation_fraction": ConstraintDiagnostic(
                    value=top_violation.to(torch.float32).mean(),
                    reduce="max",
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
                "constraint/flipped_cell_count_mean": ConstraintDiagnostic(
                    value=flipped_cell_count.mean(),
                    reduce="mean",
                ),
                "constraint/flipped_cell_count_worst": ConstraintDiagnostic(
                    value=flipped_cell_count.max(),
                    reduce="max",
                ),
                "constraint/flipped_cell_fraction_mean": ConstraintDiagnostic(
                    value=flipped_cell_fraction.mean(),
                    reduce="mean",
                ),
                "constraint/flipped_cell_fraction_worst": ConstraintDiagnostic(
                    value=flipped_cell_fraction.max(),
                    reduce="max",
                ),
            }
            return self.as_output(
                out,
                aux={
                    "pred_base": pred,
                    "x_anchor": x_anchor,
                    "envelope_x": envelope_x,
                    "dx": dx,
                    "dy": dy,
                    "gap": gap,
                    "gap_fraction": gap_fraction,
                    "envelope_y": envelope_y,
                    "top_clearance": top_clearance,
                },
                diagnostics=diagnostics,
            )
        return out
