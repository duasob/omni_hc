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
    available height.  When ``max_gap`` is set, channel 3 predicts an absolute
    die-index gap profile before interpolation; otherwise it uses the legacy
    budget-fraction top gap.
    """

    name = "plasticity_envelope"
    expected_backbone_out_dim = 4

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
        min_x_spacing: float | None = None,
        min_y_spacing: float | None = None,
        min_gap: float = 1.0e-6,
        max_gap: float | None = None,
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
        expected_backbone_out_dim = int(self.expected_backbone_out_dim)
        if self.backbone_out_dim != expected_backbone_out_dim:
            raise ValueError(
                f"{self.__class__.__name__} expects "
                f"backbone_out_dim={expected_backbone_out_dim}, "
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
        self.min_x_spacing = (
            self.min_spacing if min_x_spacing is None else float(min_x_spacing)
        )
        self.min_y_spacing = (
            self.min_spacing if min_y_spacing is None else float(min_y_spacing)
        )
        self.min_gap = float(min_gap)
        if self.min_spacing < 0.0:
            raise ValueError(f"min_spacing must be non-negative, got {min_spacing}.")
        if self.min_x_spacing < 0.0:
            raise ValueError(
                f"min_x_spacing must be non-negative, got {min_x_spacing}."
            )
        if self.min_y_spacing < 0.0:
            raise ValueError(
                f"min_y_spacing must be non-negative, got {min_y_spacing}."
            )
        if self.min_gap < 0.0:
            raise ValueError(f"min_gap must be non-negative, got {min_gap}.")
        self.max_gap = None if max_gap is None else float(max_gap)
        if self.max_gap is not None:
            if self.max_gap < 0.0:
                raise ValueError(f"max_gap must be non-negative, got {max_gap}.")
            if self.max_gap < self.min_gap:
                raise ValueError(
                    f"max_gap must be at least min_gap={self.min_gap}, got {max_gap}."
                )
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

    def _envelope_profile(
        self,
        *,
        pred: torch.Tensor,
        fx: torch.Tensor | None,
        T: torch.Tensor | None,
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
            return capped_profile
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
        return capped_profile

    def _envelope_y(
        self,
        *,
        pred: torch.Tensor,
        fx: torch.Tensor | None,
        T: torch.Tensor | None,
        x_query: torch.Tensor,
    ) -> torch.Tensor:
        capped_profile = self._envelope_profile(pred=pred, fx=fx, T=T)
        return self._interpolate_envelope_y(capped_profile, x_query)

    def _absolute_gap_profile(self, raw_gap: torch.Tensor) -> torch.Tensor:
        if self.max_gap is None:
            raise RuntimeError("absolute gap profile requires max_gap to be set.")
        if self.max_gap == self.min_gap:
            return torch.full_like(raw_gap, self.min_gap)
        gap_fraction = torch.sigmoid(raw_gap)
        return self.min_gap + (self.max_gap - self.min_gap) * gap_fraction

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
        dx = self._positive_spacing(raw[:, :-1, :, 1]) + self.min_x_spacing
        raw_dy = raw[:, :, :-1, 2]
        vertical_score = self._positive_spacing(raw_dy).clamp_min(1.0e-12)
        weights = vertical_score / vertical_score.sum(dim=2, keepdim=True)
        raw_gap = raw[:, :, 0, 3]
        gap_fraction = torch.sigmoid(raw_gap)

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
        die_envelope_profile = self._envelope_profile(pred=pred, fx=fx, T=T)
        die_envelope_y = self._interpolate_envelope_y(die_envelope_profile, envelope_x)
        if self.max_gap is None:
            envelope_y = die_envelope_y
        else:
            gap_profile = self._absolute_gap_profile(raw_gap)
            gapped_profile = die_envelope_profile - gap_profile
            envelope_y = self._interpolate_envelope_y(gapped_profile, envelope_x)

        bottom_y = self.bottom_y.to(device=pred.device, dtype=pred.dtype)
        envelope_height = envelope_y - bottom_y
        min_spacing_total = float(j_count - 1) * self.min_y_spacing
        if self.max_gap is None:
            min_budget_total = min_spacing_total + self.min_gap
            free_budget = (envelope_height - min_budget_total).clamp_min(0.0)
            gap = self.min_gap + free_budget * gap_fraction
            alloc_budget = free_budget * (1.0 - gap_fraction)
            gap_profile = gap
        else:
            free_budget = (envelope_height - min_spacing_total).clamp_min(0.0)
            gap = die_envelope_y - envelope_y
            alloc_budget = free_budget
        dy = self.min_y_spacing + weights * alloc_budget[:, :, None]

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
                    "gap_profile": gap_profile,
                    "die_envelope_y": die_envelope_y,
                    "envelope_y": envelope_y,
                    "top_clearance": top_clearance,
                },
                diagnostics=diagnostics,
            )
        return out


class PlasticityEnvelopeYFreeXConstraint(PlasticityEnvelopeConstraint):
    """
    Constrain only the vertical coordinate to the moving envelope.

    The backbone emits two channels:
    - channel 0: free horizontal displacement ``u_x``
    - channel 1: pointwise vertical logit offset

    The output still matches the plasticity target convention
    ``[x, y, u_x, u_y]``.  Unlike :class:`PlasticityEnvelopeConstraint`, this
    ablation does not reconstruct x or y with cumulative sums.
    """

    name = "plasticity_envelope_y_free_x"
    expected_backbone_out_dim = 2

    def __init__(
        self,
        *,
        shapelist,
        backbone_out_dim: int = 2,
        target_out_dim: int = 4,
        fix_bottom: bool = True,
        envelope_query: str = "pred_x",
        fraction_eps: float = 1.0e-4,
        **kwargs,
    ):
        super().__init__(
            shapelist=shapelist,
            backbone_out_dim=backbone_out_dim,
            target_out_dim=target_out_dim,
            **kwargs,
        )
        if self.backbone_out_dim != 2:
            raise ValueError(
                "PlasticityEnvelopeYFreeXConstraint expects backbone_out_dim=2, "
                f"got {self.backbone_out_dim}."
            )
        self.fix_bottom = bool(fix_bottom)
        self.envelope_query = str(envelope_query).lower()
        if self.envelope_query not in {"pred_x", "material_x"}:
            raise ValueError(
                "envelope_query must be one of: pred_x, material_x; "
                f"got {envelope_query!r}."
            )
        self.fraction_eps = float(fraction_eps)
        if not 0.0 < self.fraction_eps < 0.5:
            raise ValueError(
                f"fraction_eps must be in (0, 0.5), got {fraction_eps}."
            )

        material_y = self.material_grid[..., 1]
        bottom_y = self.bottom_y.to(dtype=material_y.dtype)
        top_y = self.constant_envelope_y.to(dtype=material_y.dtype)
        base_fraction = ((material_y - bottom_y) / (top_y - bottom_y)).clamp(
            self.fraction_eps,
            1.0 - self.fraction_eps,
        )
        self.register_buffer(
            "base_y_logit",
            torch.logit(base_fraction),
            persistent=False,
        )

    def forward(self, *, pred, fx=None, T=None, return_aux=False, **_unused):
        if pred.ndim != 3 or pred.shape[-1] != self.backbone_out_dim:
            raise ValueError(
                "PlasticityEnvelopeYFreeXConstraint expects pred with shape "
                f"(batch, n_points, {self.backbone_out_dim}), got {tuple(pred.shape)}."
            )

        batch_size, n_points, _ = pred.shape
        i_count, j_count = self.shapelist
        expected_points = i_count * j_count
        if n_points != expected_points:
            raise ValueError(
                f"PlasticityEnvelopeYFreeXConstraint expected {expected_points} points "
                f"for shapelist={self.shapelist}, got {n_points}."
            )

        raw = pred.reshape(batch_size, i_count, j_count, self.backbone_out_dim)
        material = self.material_grid.to(device=pred.device, dtype=pred.dtype)
        free_ux = raw[..., 0]
        x = material[None, ..., 0] + free_ux

        if self.envelope_query == "pred_x":
            envelope_x = x[:, :, 0]
        else:
            envelope_x = material[None, :, 0, 0].expand(batch_size, -1)
        die_envelope_profile = self._envelope_profile(pred=pred, fx=fx, T=T)
        envelope_y = self._interpolate_envelope_y(die_envelope_profile, envelope_x)

        bottom_y = self.bottom_y.to(device=pred.device, dtype=pred.dtype)
        envelope_height = (envelope_y - bottom_y).clamp_min(0.0)
        base_logit = self.base_y_logit.to(device=pred.device, dtype=pred.dtype)
        y_fraction = torch.sigmoid(raw[..., 1] + base_logit[None, ...])
        y = bottom_y + y_fraction * envelope_height[:, :, None]
        if self.fix_bottom:
            y[:, :, -1] = bottom_y

        coords = torch.stack((x, y), dim=-1)
        displacement = coords - material[None, ...]
        out = torch.cat((coords, displacement), dim=-1).reshape(
            batch_size,
            expected_points,
            self.target_out_dim,
        )

        if return_aux:
            top_clearance = envelope_y - y[:, :, 0]
            top_violation = top_clearance < 0
            bottom_y_error = (y[:, :, -1] - bottom_y).abs()
            dy = y[:, :, :-1] - y[:, :, 1:]
            diagnostics = {
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
                "constraint/bottom_y_abs_error_max": ConstraintDiagnostic(
                    value=bottom_y_error.max(),
                    reduce="max",
                ),
                "constraint/min_dy_signed": ConstraintDiagnostic(
                    value=dy.min(),
                    reduce="min",
                ),
                "constraint/y_fraction_min": ConstraintDiagnostic(
                    value=y_fraction.min(),
                    reduce="min",
                ),
                "constraint/y_fraction_max": ConstraintDiagnostic(
                    value=y_fraction.max(),
                    reduce="max",
                ),
            }
            return self.as_output(
                out,
                aux={
                    "pred_base": pred,
                    "free_ux": free_ux,
                    "envelope_x": envelope_x,
                    "die_envelope_y": self._interpolate_envelope_y(
                        die_envelope_profile,
                        envelope_x,
                    ),
                    "envelope_y": envelope_y,
                    "top_clearance": top_clearance,
                    "y_fraction": y_fraction,
                    "dy": dy,
                },
                diagnostics=diagnostics,
            )
        return out


class PlasticityIsotonicRegression(ConstraintModule):
    """
    Project directly predicted plasticity coordinates onto ordered coordinates.

    By default the backbone predicts coordinate displacement residuals, which
    are added to the material grid before projection.  This keeps the projection
    in coordinate space without requiring the backbone to produce absolute
    physical x/y coordinates from an unanchored initialization.  Set
    coordinate_mode="absolute" to interpret the first two backbone channels as
    raw physical coordinates directly.

    This constraint minimally repairs invalid raw coordinates with 1D isotonic projections:
    - x is projected to be decreasing along the material i-axis.
    - y is projected per material column to be decreasing from top to bottom,
      fixed at y_bottom, and no higher than the moving die envelope at the top.

    Displacements are recomputed from the projected coordinates so the returned
    four channels remain consistent with the dataset target convention.
    """

    name = "plasticity_isotonic_regression"

    def __init__(
        self,
        *,
        shapelist,
        backbone_out_dim: int = 2,
        target_out_dim: int = 4,
        x_left: float = 0.35,
        x_right: float = -49.65,
        y_top: float = 14.9,
        y_bottom: float = -0.1,
        top_height: float | None = None,
        envelope_source: str = "fx",
        die_speed: float = 6.0,
        time_duration: float = 1.0,
        min_x_spacing: float = 1.0e-6,
        min_y_spacing: float = 1.0e-6,
        collapse_spacing_threshold: float = 1.0e-3,
        top_collapse_rows: int = 3,
        coordinate_mode: str = "displacement",
        projection_device: str = "auto",
    ):
        super().__init__()
        if shapelist is None or len(tuple(shapelist)) != 2:
            raise ValueError("PlasticityIsotonicRegression requires a 2D shapelist.")
        self.shapelist = tuple(int(v) for v in shapelist)
        self.backbone_out_dim = int(backbone_out_dim)
        self.target_out_dim = int(target_out_dim)
        if self.backbone_out_dim < 2:
            raise ValueError(
                "PlasticityIsotonicRegression expects at least 2 backbone channels "
                f"for raw x/y coordinates, got {self.backbone_out_dim}."
            )
        if self.target_out_dim != 4:
            raise ValueError(
                "PlasticityIsotonicRegression outputs the 4 plasticity target "
                f"channels, got target_out_dim={self.target_out_dim}."
            )
        self.envelope_source = str(envelope_source).lower()
        if self.envelope_source not in {"fx", "constant"}:
            raise ValueError(
                "envelope_source must be one of: fx, constant; "
                f"got {envelope_source!r}."
            )
        self.die_speed = float(die_speed)
        self.time_duration = float(time_duration)
        self.min_x_spacing = float(min_x_spacing)
        self.min_y_spacing = float(min_y_spacing)
        self.collapse_spacing_threshold = float(collapse_spacing_threshold)
        self.top_collapse_rows = int(top_collapse_rows)
        self.coordinate_mode = str(coordinate_mode).lower()
        self.projection_device = str(projection_device).lower()
        if self.coordinate_mode not in {"displacement", "absolute"}:
            raise ValueError(
                "coordinate_mode must be one of: displacement, absolute; "
                f"got {coordinate_mode!r}."
            )
        if self.projection_device not in {"auto", "native", "cpu"}:
            raise ValueError(
                "projection_device must be one of: auto, native, cpu; "
                f"got {projection_device!r}."
            )
        if self.min_x_spacing < 0.0:
            raise ValueError(f"min_x_spacing must be non-negative, got {min_x_spacing}.")
        if self.min_y_spacing < 0.0:
            raise ValueError(f"min_y_spacing must be non-negative, got {min_y_spacing}.")
        if self.collapse_spacing_threshold < 0.0:
            raise ValueError(
                "collapse_spacing_threshold must be non-negative, "
                f"got {collapse_spacing_threshold}."
            )
        if self.top_collapse_rows < 1:
            raise ValueError(f"top_collapse_rows must be >= 1, got {top_collapse_rows}.")
        self.input_normalizer = None

        i_count, j_count = self.shapelist
        if i_count < 2 or j_count < 2:
            raise ValueError(
                f"Plasticity grid must have at least 2x2 points, got {self.shapelist}."
            )
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
            PlasticityEnvelopeConstraint._make_top_height_profile(
                top_height=top_height,
                y_top=float(y_top),
                i_count=i_count,
            ),
            persistent=False,
        )

    def set_input_normalizer(self, normalizer) -> None:
        self.input_normalizer = normalizer

    def _decode_fx(self, fx: torch.Tensor) -> torch.Tensor:
        if self.input_normalizer is None:
            return fx
        return self.input_normalizer.decode(fx)

    def _time(self, T: torch.Tensor | None, *, pred: torch.Tensor) -> torch.Tensor:
        batch_size = int(pred.shape[0])
        if T is None:
            return torch.zeros(batch_size, 1, device=pred.device, dtype=pred.dtype)
        time = T.to(device=pred.device, dtype=pred.dtype)
        if time.ndim == 1:
            time = time[:, None]
        return time.reshape(batch_size, -1)[:, 0:1]

    def _should_project_on_cpu(self, tensor: torch.Tensor) -> bool:
        if self.projection_device == "cpu":
            return tensor.device.type != "cpu"
        if self.projection_device == "auto":
            return tensor.device.type == "cuda"
        return False

    def _project_x_backend(self, raw_x: torch.Tensor) -> torch.Tensor:
        if not self._should_project_on_cpu(raw_x):
            return self._project_x(raw_x, min_spacing=self.min_x_spacing)
        projected = self._project_x(
            raw_x.to("cpu"),
            min_spacing=self.min_x_spacing,
        )
        return projected.to(device=raw_x.device)

    def _project_y_backend(
        self,
        raw_y: torch.Tensor,
        *,
        envelope_y: torch.Tensor,
        bottom_y: torch.Tensor,
    ) -> torch.Tensor:
        if not self._should_project_on_cpu(raw_y):
            return self._project_y(
                raw_y,
                envelope_y=envelope_y,
                bottom_y=bottom_y,
                min_spacing=self.min_y_spacing,
            )
        projected = self._project_y(
            raw_y.to("cpu"),
            envelope_y=envelope_y.to("cpu"),
            bottom_y=bottom_y.to("cpu"),
            min_spacing=self.min_y_spacing,
        )
        return projected.to(device=raw_y.device)

    @staticmethod
    def _pava_increasing(values: torch.Tensor) -> torch.Tensor:
        blocks: list[tuple[torch.Tensor, int]] = []
        for value in values:
            blocks.append((value, 1))
            while len(blocks) >= 2:
                left_sum, left_count = blocks[-2]
                right_sum, right_count = blocks[-1]
                left_mean = left_sum / float(left_count)
                right_mean = right_sum / float(right_count)
                if float(left_mean.detach()) <= float(right_mean.detach()):
                    break
                blocks[-2:] = [(left_sum + right_sum, left_count + right_count)]
        projected = []
        for block_sum, block_count in blocks:
            projected.extend([block_sum / float(block_count)] * block_count)
        return torch.stack(projected)

    @classmethod
    def _project_decreasing_1d(cls, values: torch.Tensor) -> torch.Tensor:
        return -cls._pava_increasing(-values)

    @classmethod
    def _project_x(cls, raw_x: torch.Tensor, *, min_spacing: float) -> torch.Tensor:
        batch_size, i_count, j_count = raw_x.shape
        spacing = torch.arange(
            i_count,
            device=raw_x.device,
            dtype=raw_x.dtype,
        ) * float(min_spacing)
        projected = torch.empty_like(raw_x)
        for batch_idx in range(batch_size):
            for j_idx in range(j_count):
                z = raw_x[batch_idx, :, j_idx] + spacing
                projected[batch_idx, :, j_idx] = (
                    cls._project_decreasing_1d(z) - spacing
                )
        return projected

    @classmethod
    def _project_y(
        cls,
        raw_y: torch.Tensor,
        *,
        envelope_y: torch.Tensor,
        bottom_y: torch.Tensor,
        min_spacing: float,
    ) -> torch.Tensor:
        batch_size, i_count, j_count = raw_y.shape
        projected = torch.empty_like(raw_y)
        base_spacing = torch.arange(
            j_count,
            device=raw_y.device,
            dtype=raw_y.dtype,
        ) * float(min_spacing)
        for batch_idx in range(batch_size):
            for i_idx in range(i_count):
                cap = envelope_y[batch_idx, i_idx]
                bottom = bottom_y.to(device=raw_y.device, dtype=raw_y.dtype)
                max_spacing = (cap - bottom).clamp_min(0.0) / float(j_count - 1)
                effective_spacing = min(float(min_spacing), float(max_spacing.detach()))
                if effective_spacing == float(min_spacing):
                    spacing = base_spacing
                else:
                    spacing = torch.arange(
                        j_count,
                        device=raw_y.device,
                        dtype=raw_y.dtype,
                    ) * effective_spacing
                z_bottom = bottom + spacing[-1]
                z_cap = torch.maximum(cap, z_bottom)
                z_free = raw_y[batch_idx, i_idx, :-1] + spacing[:-1]
                z_free = z_free.clamp(min=z_bottom, max=z_cap)
                z_projected = cls._project_decreasing_1d(z_free)
                z = torch.cat([z_projected, z_bottom.reshape(1)], dim=0)
                projected[batch_idx, i_idx] = z - spacing
        return projected

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
                "PlasticityIsotonicRegression with envelope_source='fx' requires fx."
            )
        if fx.ndim != 3 or fx.shape[0] != batch_size or fx.shape[1] != n_points:
            raise ValueError(
                "PlasticityIsotonicRegression expects fx with shape "
                f"(batch, {n_points}, channels), got {tuple(fx.shape)}."
            )
        fx_physical = self._decode_fx(fx).to(device=pred.device, dtype=pred.dtype)
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
        if pred.ndim != 3 or pred.shape[-1] < 2:
            raise ValueError(
                "PlasticityIsotonicRegression expects pred with shape "
                "(batch, n_points, channels>=2), "
                f"got {tuple(pred.shape)}."
            )
        batch_size, n_points, _ = pred.shape
        i_count, j_count = self.shapelist
        expected_points = i_count * j_count
        if n_points != expected_points:
            raise ValueError(
                f"PlasticityIsotonicRegression expected {expected_points} points "
                f"for shapelist={self.shapelist}, got {n_points}."
            )

        raw = pred.reshape(batch_size, i_count, j_count, -1)
        material = self.material_grid.to(device=pred.device, dtype=pred.dtype)
        if self.coordinate_mode == "displacement":
            raw_coords = material[None, ...] + raw[..., :2]
        else:
            raw_coords = raw[..., :2]
        raw_x = raw_coords[..., 0]
        raw_y = raw_coords[..., 1]
        x = self._project_x_backend(raw_x)
        envelope_x = x[:, :, 0]
        envelope_y = self._envelope_y(pred=pred, fx=fx, T=T, x_query=envelope_x)
        bottom_y = self.bottom_y.to(device=pred.device, dtype=pred.dtype)
        y = self._project_y_backend(
            raw_y,
            envelope_y=envelope_y,
            bottom_y=bottom_y,
        )

        coords = torch.stack((x, y), dim=-1)
        displacement = coords - material[None, ...]
        out = torch.cat((coords, displacement), dim=-1).reshape(
            batch_size,
            expected_points,
            self.target_out_dim,
        )

        if return_aux:
            dx = x[:, :-1, :] - x[:, 1:, :]
            dy = y[:, :, :-1] - y[:, :, 1:]
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
            top_violation = top_clearance < -1.0e-6
            collapsed_dy = dy <= self.collapse_spacing_threshold
            top_rows = min(self.top_collapse_rows, int(dy.shape[-1]))
            top_dy = dy[:, :, :top_rows]
            collapsed_top_dy = top_dy <= self.collapse_spacing_threshold
            collapsed_dy_per_sample = (
                collapsed_dy.reshape(batch_size, -1).to(torch.float32).mean(dim=1)
            )
            collapsed_top_dy_per_sample = (
                collapsed_top_dy.reshape(batch_size, -1).to(torch.float32).mean(dim=1)
            )
            correction = torch.linalg.vector_norm(
                coords - torch.stack((raw_x, raw_y), dim=-1),
                dim=-1,
            )
            correction_total = correction.reshape(batch_size, -1).sum(dim=1)
            flipped_cell = oriented_cell_area < 0
            flipped_cell_count = flipped_cell.to(torch.float32).sum(dim=(-1, -2))
            flipped_cell_fraction = flipped_cell.to(torch.float32).mean(dim=(-1, -2))
            diagnostics = {
                "constraint/min_dx": ConstraintDiagnostic(value=dx.min(), reduce="min"),
                "constraint/min_dy": ConstraintDiagnostic(value=dy.min(), reduce="min"),
                "constraint/mean_dy": ConstraintDiagnostic(value=dy.mean(), reduce="mean"),
                "constraint/top_dy_min": ConstraintDiagnostic(
                    value=top_dy.min(),
                    reduce="min",
                ),
                "constraint/top_dy_mean": ConstraintDiagnostic(
                    value=top_dy.mean(),
                    reduce="mean",
                ),
                "constraint/dy_collapse_count": ConstraintDiagnostic(
                    value=collapsed_dy.to(torch.float32).sum(),
                    reduce="sum",
                ),
                "constraint/dy_collapse_fraction": ConstraintDiagnostic(
                    value=collapsed_dy.to(torch.float32).mean(),
                    reduce="mean",
                ),
                "constraint/dy_collapse_worst_sample_fraction": ConstraintDiagnostic(
                    value=collapsed_dy_per_sample.max(),
                    reduce="max",
                ),
                "constraint/top_dy_collapse_count": ConstraintDiagnostic(
                    value=collapsed_top_dy.to(torch.float32).sum(),
                    reduce="sum",
                ),
                "constraint/top_dy_collapse_fraction": ConstraintDiagnostic(
                    value=collapsed_top_dy.to(torch.float32).mean(),
                    reduce="mean",
                ),
                "constraint/top_dy_collapse_worst_sample_fraction": ConstraintDiagnostic(
                    value=collapsed_top_dy_per_sample.max(),
                    reduce="max",
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
                "constraint/projection_correction_mean": ConstraintDiagnostic(
                    value=correction.mean(),
                    reduce="mean",
                ),
                "constraint/projection_correction_max": ConstraintDiagnostic(
                    value=correction.max(),
                    reduce="max",
                ),
                "constraint/projection_correction_total_mean": ConstraintDiagnostic(
                    value=correction_total.mean(),
                    reduce="mean",
                ),
                "constraint/projection_correction_total_worst": ConstraintDiagnostic(
                    value=correction_total.max(),
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
                    "raw_coords": torch.stack((raw_x, raw_y), dim=-1),
                    "coords": coords,
                    "envelope_x": envelope_x,
                    "envelope_y": envelope_y,
                    "top_clearance": top_clearance,
                    "dx": dx,
                    "dy": dy,
                    "projection_correction": correction,
                },
                diagnostics=diagnostics,
            )
        return out
