from __future__ import annotations

import math
from typing import Any

import torch

from .base import (
    _BUILD_META_KEYS,
    ConstrainedModel,
    ConstraintDiagnostic,
    ConstraintModule,
)
from .mean import build_mlp
from .utils.boundary_ops import apply_boundary_ansatz, channel_mask, encode_target
from .utils.hooks import ForwardHookLatentExtractor


def _sine_basis(n_points: int, n_modes: int) -> torch.Tensor:
    """B[j, k] = sin(k * pi * j / (n_points - 1)).  Zero at j=0 and j=n_points-1."""
    j = torch.arange(n_points, dtype=torch.float32)
    k = torch.arange(1, n_modes + 1, dtype=torch.float32)
    return torch.sin(math.pi * j.unsqueeze(1) * k.unsqueeze(0) / (n_points - 1))


def _boundary_indices_2d(H: int, W: int) -> dict[str, torch.Tensor]:
    """Flat indices for each edge of an H×W row-major grid (y-first ordering)."""
    return {
        "bottom": torch.arange(W),
        "top":    torch.arange((H - 1) * W, H * W),
        "left":   torch.arange(0, H * W, W),
        "right":  torch.arange(W - 1, H * W, W),
    }


class SineBoundaryConstraint(ConstraintModule):
    """
    Predicts per-edge boundary values via a sine basis expansion and hard-enforces
    them on the main model prediction.

    For each edge of length L, the predicted profile is:
        u(x_j) = sum_{k=1}^{n_modes} c_k * sin(k * pi * j / (L - 1))

    Corner-zero is satisfied by construction for all k.

    coeff_head always receives physical-space boundary permeability features,
    Z-score normalised using statistics computed during pretrain_coeff_head.
    If set_input_normalizer is called (benchmark x_normalizer), _boundary_feats
    first decodes the normalised input back to physical space before extracting
    features, ensuring the same distribution as during standalone pretrain.

    coeff_head can be pre-trained from raw numpy arrays via build() when
    coeff_head_pretrain is set in the constraint config (see config comments).
    A saved checkpoint can be loaded via pretrained_coeff_head.
    """

    name = "sine_boundary_constraint"

    def __init__(
        self,
        *,
        n_modes: int,
        grid_shape: tuple[int, int],
        hidden_dim: int,
        n_layers: int = 0,
        act: str = "gelu",
        latent_dim: int = 0,
        feature_mode: str = "boundary",
        inner_depth: int = 1,
        extractor=None,
    ):
        super().__init__()
        H, W = int(grid_shape[0]), int(grid_shape[1])
        self.H, self.W = H, W
        self.n_modes = n_modes
        self.latent_dim = latent_dim
        self.feature_mode = str(feature_mode)
        # Number of interior rings (depth 1..inner_depth) fed to coeff_head when
        # feature_mode includes "inner". depth 1 == the ring adjacent to the edge.
        self.inner_depth = int(inner_depth)
        if self.inner_depth < 1:
            raise ValueError(f"inner_depth must be >= 1, got {self.inner_depth}")
        self.extractor = extractor

        # Sine basis matrices — (n_points, n_modes)
        self.register_buffer("basis_h", _sine_basis(W, n_modes))
        self.register_buffer("basis_v", _sine_basis(H, n_modes))

        # Flat boundary indices per edge
        idx = _boundary_indices_2d(H, W)
        for edge, tensor in idx.items():
            self.register_buffer(f"idx_{edge}", tensor)

        # Unique boundary nodes for latent pooling (corners deduplicated)
        all_boundary = torch.cat([
            idx["bottom"],
            idx["top"],
            idx["left"][1:-1],
            idx["right"][1:-1],
        ])
        self.register_buffer("idx_all_boundary", all_boundary)

        # Ansatz distance l for f = g + l * N.
        distance = torch.ones(H * W, dtype=torch.float32)
        distance[all_boundary] = 0.0
        self.register_buffer("boundary_distance", distance.view(1, H * W, 1))
        self.register_buffer("boundary_keep", (1.0 - distance).view(1, H * W, 1))

        # The sine basis is zero-endpoint and zero-mean, so it can only express
        # a corner-zero profile in *physical* space (where the Darcy boundary
        # is ~0). Predictions are normalized, where that same boundary is a
        # large near-constant -mu/sigma the sine series cannot represent. We
        # therefore build g in physical units and encode it, exactly like the
        # other Dirichlet ansatz constraints; encode_target injects the DC
        # offset the basis is structurally unable to produce.
        self.target_normalizer = None
        self.input_normalizer  = None  # set via set_input_normalizer

        # Z-score stats for boundary permeability features (physical space).
        # Populated by pretrain_coeff_head; None means no normalisation.
        self.register_buffer("_feat_mean", None)
        self.register_buffer("_feat_std",  None)

        # MLP input: selected physical-space permeability features + latent.
        boundary_feat_dim = self._permeability_feature_dim() + latent_dim
        self.coeff_head = build_mlp(
            boundary_feat_dim, hidden_dim, 4 * n_modes, n_layers=n_layers, act=act
        )

    # ------------------------------------------------------------------
    # Normalizer setters (called by train_steady_task after model build)
    # ------------------------------------------------------------------

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def set_input_normalizer(self, normalizer) -> None:
        self.input_normalizer = normalizer

    # ------------------------------------------------------------------
    # coeff_head lifecycle
    # ------------------------------------------------------------------

    def freeze_coeff_head(self) -> None:
        for p in self.coeff_head.parameters():
            p.requires_grad = False

    def pretrain_coeff_head(
        self,
        fx_arr,
        sol_arr,
        *,
        device,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_samples: int | None = None,
        val_frac: float = 0.1,
        val_seed: int = 0,
        seed: int = 0,
        save_path=None,
        log_path=None,
    ) -> dict[str, Any]:
        """Train coeff_head standalone from raw numpy arrays (physical space).

        fx_arr:   [N, H, W] or [N, H*W] permeability (physical units)
        sol_arr:  [N, H, W] or [N, H*W] pressure solution (physical units)

        Computes and stores Z-score feature statistics so that _boundary_feats
        produces the same distribution during main training (after the benchmark
        normalizer is applied and then inverted).

        If val_frac > 0, a held-out split is used to track validation rel-L2
        (boundary, unique-corner) and the checkpoint with the lowest val rel-L2
        is restored at the end. A per-epoch train/val log is written to
        log_path (YAML).
        """
        import copy

        import torch.nn.functional as F

        N = int(fx_arr.shape[0])
        if max_samples is not None:
            N = min(N, int(max_samples))
            fx_arr, sol_arr = fx_arr[:N], sol_arr[:N]

        fx_t  = torch.as_tensor(fx_arr.reshape(N, -1, 1),  dtype=torch.float32).to(device)
        sol_t = torch.as_tensor(sol_arr.reshape(N, -1, 1), dtype=torch.float32).to(device)

        # Deterministic train/val split.
        val_frac = float(max(0.0, min(val_frac, 0.9)))
        n_val = int(round(N * val_frac))
        n_train = N - n_val
        g = torch.Generator(device="cpu").manual_seed(int(val_seed))
        perm0 = torch.randperm(N, generator=g).to(device)
        train_idx = perm0[:n_train]
        val_idx   = perm0[n_train:]
        fx_tr, sol_tr = fx_t[train_idx], sol_t[train_idx]
        fx_va, sol_va = (fx_t[val_idx], sol_t[val_idx]) if n_val > 0 else (None, None)

        # Compute Z-score stats from the selected training permeability features.
        with torch.no_grad():
            raw_feats = self._raw_permeability_feats(fx_tr)
            self._feat_mean = raw_feats.mean(0)
            self._feat_std  = raw_feats.std(0).clamp(min=1e-6)

            # Target rescale (matches scripts/diagnostics/darcy/darcy_boundary_models.py).
            # Darcy boundary values are O(1e-5); without rescale, MSE ~ 1e-10 and
            # Adam's epsilon dominates sqrt(v), collapsing the effective step.
            # We train on y / y_scale and bake y_scale into the head's last Linear
            # after training, so inference still produces physical-unit output.
            y_tr_boundary = torch.cat([
                sol_tr[:, self.idx_bottom,       0],
                sol_tr[:, self.idx_top,          0],
                sol_tr[:, self.idx_left[1:-1],   0],
                sol_tr[:, self.idx_right[1:-1],  0],
            ], dim=-1)
            y_scale = float(
                torch.sqrt((y_tr_boundary ** 2).mean()).item()
            ) + 1e-12

        def _boundary_pred_target(fx_b, sol_b):
            B = fx_b.shape[0]
            feats = self._boundary_feats(fx_b)
            if self.latent_dim > 0:
                feats = torch.cat(
                    [feats, feats.new_zeros(B, self.latent_dim)], dim=-1
                )
            coeffs = self.coeff_head(feats).view(B, 4, self.n_modes)
            u_bottom = coeffs[:, 0] @ self.basis_h.T
            u_top    = coeffs[:, 1] @ self.basis_h.T
            u_left   = coeffs[:, 2] @ self.basis_v.T
            u_right  = coeffs[:, 3] @ self.basis_v.T
            # Unique-boundary representation (corners deduplicated), matching
            # scripts/diagnostics/darcy/darcy_boundary_n_modes.py.
            pred = torch.cat(
                [u_bottom, u_top, u_left[:, 1:-1], u_right[:, 1:-1]], dim=-1
            )  # [B, 2W + 2(H-2)]
            targ = torch.cat([
                sol_b[:, self.idx_bottom,       0],
                sol_b[:, self.idx_top,          0],
                sol_b[:, self.idx_left[1:-1],   0],
                sol_b[:, self.idx_right[1:-1],  0],
            ], dim=-1) / y_scale
            return pred, targ

        def _rel_l2(pred, targ):
            diff = (pred - targ).pow(2).sum(dim=-1).sqrt()
            norm = targ.pow(2).sum(dim=-1).sqrt().clamp(min=1e-12)
            return (diff / norm).mean()

        @torch.no_grad()
        def _eval_metrics(fx_all, sol_all):
            self.coeff_head.eval()
            mse_sum = 0.0
            rl2_sum = 0.0
            nb = 0
            for i in range(0, fx_all.shape[0], batch_size):
                pred, targ = _boundary_pred_target(
                    fx_all[i : i + batch_size], sol_all[i : i + batch_size]
                )
                mse_sum += float(F.mse_loss(pred, targ).item())
                rl2_sum += float(_rel_l2(pred, targ).item())
                nb += 1
            return mse_sum / max(nb, 1), rl2_sum / max(nb, 1)

        opt = torch.optim.Adam(
            self.coeff_head.parameters(), lr=lr, weight_decay=weight_decay
        )
        print(
            f"[coeff_head pretrain] {epochs} epochs  lr={lr}  n_train={n_train}  n_val={n_val}",
            flush=True,
        )

        history: list[dict[str, float]] = []
        best_rl2 = float("inf")
        best_epoch = -1
        best_state = None
        train_generator = torch.Generator(device="cpu").manual_seed(int(seed))

        for epoch in range(epochs):
            self.coeff_head.train()
            perm = torch.randperm(
                n_train,
                generator=train_generator,
                device="cpu",
            ).to(device)
            mse_sum = 0.0
            rl2_sum = 0.0
            n_batches = 0

            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                pred, targ = _boundary_pred_target(fx_tr[idx], sol_tr[idx])
                loss = F.mse_loss(pred, targ)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                mse_sum   += float(loss.item())
                with torch.no_grad():
                    rl2_sum += float(_rel_l2(pred, targ).item())
                n_batches += 1

            train_mse = mse_sum / max(n_batches, 1)
            train_rl2 = rl2_sum / max(n_batches, 1)
            if n_val > 0:
                val_mse, val_rl2 = _eval_metrics(fx_va, sol_va)
            else:
                val_mse, val_rl2 = float("nan"), float("nan")
            history.append({
                "epoch": epoch + 1,
                "train_mse": train_mse,
                "train_rel_l2": train_rl2,
                "val_mse": val_mse,
                "val_rel_l2": val_rl2,
            })

            if n_val > 0 and val_rl2 < best_rl2:
                best_rl2 = val_rl2
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.coeff_head.state_dict())

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = (
                    f"[coeff_head pretrain] epoch {epoch + 1}/{epochs}"
                    f"  train_rel_l2={train_rl2:.4e}"
                )
                if n_val > 0:
                    msg += (
                        f"  val_rel_l2={val_rl2:.4e}"
                        f"  best_val_rel_l2={best_rl2:.4e}@{best_epoch}"
                    )
                print(msg, flush=True)

        if best_state is not None:
            self.coeff_head.load_state_dict(best_state)
            print(
                f"[coeff_head pretrain] restored best epoch {best_epoch}"
                f"  val_rel_l2={best_rl2:.4e}",
                flush=True,
            )

        # Bake y_scale into the head's last Linear so that the head produces
        # boundary values in physical units at inference time (the constraint's
        # forward expects physical units; encode_target adds the DC offset).
        with torch.no_grad():
            last_linear = None
            for m in reversed(list(self.coeff_head.modules())):
                if isinstance(m, torch.nn.Linear):
                    last_linear = m
                    break
            if last_linear is None:
                raise RuntimeError("coeff_head has no Linear layer to absorb y_scale")
            last_linear.weight.mul_(y_scale)
            if last_linear.bias is not None:
                last_linear.bias.mul_(y_scale)
        print(f"[coeff_head pretrain] absorbed y_scale={y_scale:.4e} into last Linear", flush=True)

        from pathlib import Path

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self.coeff_head.state_dict(),
                    "feat_mean":  self._feat_mean.cpu(),
                    "feat_std":   self._feat_std.cpu(),
                },
                str(save_path),
            )
            print(f"[coeff_head pretrain] saved to {save_path}", flush=True)

        if log_path is None and save_path is not None:
            log_path = Path(save_path).with_name(Path(save_path).stem + "_log.yaml")
        if log_path is not None:
            import yaml
            log_path = Path(log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = {
                "feature_mode": self.feature_mode,
                "n_train": n_train,
                "n_val": n_val,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "val_frac": val_frac,
                "val_seed": val_seed,
                "best_epoch":        best_epoch if best_state is not None else None,
                "best_val_rel_l2":   best_rl2   if best_state is not None else None,
                "final_train_mse":    history[-1]["train_mse"]    if history else None,
                "final_train_rel_l2": history[-1]["train_rel_l2"] if history else None,
                "final_val_mse":      history[-1]["val_mse"]      if history else None,
                "final_val_rel_l2":   history[-1]["val_rel_l2"]   if history else None,
                "history": history,
            }
            with open(log_path, "w") as f:
                yaml.safe_dump(log, f, sort_keys=False)
            print(f"[coeff_head pretrain] log written to {log_path}", flush=True)
        else:
            log = {
                "feature_mode": self.feature_mode,
                "n_train": n_train,
                "n_val": n_val,
                "epochs": epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "val_frac": val_frac,
                "val_seed": val_seed,
                "best_epoch":        best_epoch if best_state is not None else None,
                "best_val_rel_l2":   best_rl2   if best_state is not None else None,
                "final_train_mse":    history[-1]["train_mse"]    if history else None,
                "final_train_rel_l2": history[-1]["train_rel_l2"] if history else None,
                "final_val_mse":      history[-1]["val_mse"]      if history else None,
                "final_val_rel_l2":   history[-1]["val_rel_l2"]   if history else None,
                "history": history,
            }

        self.freeze_coeff_head()
        print("[coeff_head pretrain] coeff_head frozen", flush=True)
        return log

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        backbone: torch.nn.Module,
        model_context: dict[str, Any],
        cfg: dict[str, Any],
    ) -> ConstrainedModel:
        constraint_section = cfg.get("constraint", {}) or {}
        _SINE_META = _BUILD_META_KEYS | {
            "latent_module", "pretrained_coeff_head", "coeff_head_pretrain",
        }
        params = {k: v for k, v in constraint_section.items() if k not in _SINE_META}

        if "grid_shape" not in params:
            shapelist = model_context.get("shapelist")
            if shapelist is None:
                raise ValueError(
                    "model_context must contain 'shapelist' for SineBoundaryConstraint"
                )
            params["grid_shape"] = tuple(int(s) for s in shapelist)

        extractor = None
        latent_module = constraint_section.get("latent_module")
        if latent_module:
            extractor = ForwardHookLatentExtractor(backbone, latent_module)
            backbone.register_forward_pre_hook(lambda *_: extractor.reset())
            if "latent_dim" not in params:
                n_hidden = model_context.get("n_hidden")
                if n_hidden is not None:
                    params["latent_dim"] = int(n_hidden)

        constraint = cls(**params, extractor=extractor)

        pretrained_path = constraint_section.get("pretrained_coeff_head")
        pretrain_cfg    = constraint_section.get("coeff_head_pretrain")

        if pretrained_path is not None:
            data = torch.load(str(pretrained_path), map_location="cpu", weights_only=True)
            if isinstance(data, dict) and "state_dict" in data:
                constraint.coeff_head.load_state_dict(data["state_dict"])
                if "feat_mean" in data:
                    constraint._feat_mean = data["feat_mean"]
                    constraint._feat_std  = data["feat_std"]
            else:
                constraint.coeff_head.load_state_dict(data)
            constraint.freeze_coeff_head()
            print(
                f"[SineBoundaryConstraint] loaded pretrained coeff_head from {pretrained_path}",
                flush=True,
            )

        elif pretrain_cfg is not None:
            from omni_hc.benchmarks.darcy.data import _load_darcy_train_raw
            x_train, y_train, _ = _load_darcy_train_raw(cfg)
            # _load_darcy_train_raw returns physical-space tensors [N, H*W, 1]
            fx_arr  = x_train.numpy()
            sol_arr = y_train.numpy()

            try:
                device = next(backbone.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

            pretrain_kwargs = {k: v for k, v in pretrain_cfg.items()}
            pretrain_kwargs.setdefault(
                "seed",
                int((cfg.get("training") or {}).get("seed", 42)),
            )
            constraint.pretrain_coeff_head(fx_arr, sol_arr, device=device, **pretrain_kwargs)

        wrapped = ConstrainedModel(backbone=backbone, constraint=constraint)
        if constraint_section.get("freeze_base", False):
            for param in wrapped.backbone.parameters():
                param.requires_grad = False
        return wrapped

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def _permeability_feature_dim(self) -> int:
        base = 2 * self.W + 2 * (self.H - 2)
        mode = self.feature_mode
        if mode == "boundary":
            return base
        if mode == "boundary_inner":
            return (1 + self.inner_depth) * base
        if mode == "boundary_stats":
            return base + 8
        if mode == "boundary_inner_stats":
            return (1 + self.inner_depth) * base + 8
        if mode == "full":
            return self.H * self.W
        raise ValueError(
            f"Unknown sine boundary feature_mode={mode!r}. "
            "Use one of: boundary, boundary_inner, boundary_stats, "
            "boundary_inner_stats, full."
        )

    def _edge_feats(self, a: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            a[:, self.idx_bottom],          # [B, W]
            a[:, self.idx_top],             # [B, W]
            a[:, self.idx_left[1:-1]],      # [B, H-2]
            a[:, self.idx_right[1:-1]],     # [B, H-2]
        ], dim=-1)

    def _inner_edge_feats(self, a: torch.Tensor) -> torch.Tensor:
        d = self.inner_depth
        if self.H < 2 * d + 1 or self.W < 2 * d + 1:
            raise ValueError(
                f"feature_mode with inner_depth={d} requires "
                f"H, W >= {2 * d + 1}; got H={self.H}, W={self.W}"
            )
        pieces: list[torch.Tensor] = []
        for k in range(1, d + 1):
            pieces.append(a[:, k * self.W : (k + 1) * self.W])               # bottom + k
            pieces.append(a[:, (self.H - 1 - k) * self.W : (self.H - k) * self.W])  # top - k
            pieces.append(a[:, self.idx_left[1:-1] + k])                     # left + k
            pieces.append(a[:, self.idx_right[1:-1] - k])                    # right - k
        return torch.cat(pieces, dim=-1)

    def _global_stats_feats(self, a: torch.Tensor) -> torch.Tensor:
        field = a.view(a.shape[0], self.H, self.W)
        flat = field.reshape(field.shape[0], -1)
        interior = field[:, 1:-1, 1:-1].reshape(field.shape[0], -1)
        return torch.stack(
            [
                flat.mean(dim=-1),
                flat.std(dim=-1),
                flat.amin(dim=-1),
                flat.amax(dim=-1),
                interior.mean(dim=-1),
                interior.std(dim=-1),
                interior.amin(dim=-1),
                interior.amax(dim=-1),
            ],
            dim=-1,
        )

    def _raw_permeability_feats(self, fx: torch.Tensor) -> torch.Tensor:
        a = fx[:, :, 0]
        mode = self.feature_mode
        if mode == "full":
            return a

        pieces = [self._edge_feats(a)]
        if mode in {"boundary_inner", "boundary_inner_stats"}:
            pieces.append(self._inner_edge_feats(a))
        if mode in {"boundary_stats", "boundary_inner_stats"}:
            pieces.append(self._global_stats_feats(a))
        if mode == "boundary" or len(pieces) > 1:
            return torch.cat(pieces, dim=-1)
        raise ValueError(f"Unknown sine boundary feature_mode={mode!r}")

    def _boundary_feats(self, fx: torch.Tensor) -> torch.Tensor:
        # Decode to physical space if the benchmark normalizer was applied upstream.
        # coeff_head always expects physical-space inputs, Z-score normalised.
        if self.input_normalizer is not None:
            fx = self.input_normalizer.decode(fx)
        feats = self._raw_permeability_feats(fx)
        if self._feat_std is not None:
            feats = (feats - self._feat_mean) / self._feat_std
        return feats

    def forward(self, *, pred, return_aux=False, **kwargs):
        fx = kwargs.get("fx")
        if fx is None:
            raise ValueError("SineBoundaryConstraint requires fx (input permeability)")

        B = pred.shape[0]
        feats = self._boundary_feats(fx)

        if self.extractor is not None:
            latent = self.extractor.get()
            if latent is not None:
                boundary_latent = latent[:, self.idx_all_boundary, :]  # [B, n_bnd, n_hidden]
                feats = torch.cat([feats, boundary_latent.mean(dim=1)], dim=-1)

        coeffs = self.coeff_head(feats).view(B, 4, self.n_modes)  # [B, 4, n_modes]

        u_bottom = coeffs[:, 0] @ self.basis_h.T   # [B, W]
        u_top    = coeffs[:, 1] @ self.basis_h.T   # [B, W]
        u_left   = coeffs[:, 2] @ self.basis_v.T   # [B, H]
        u_right  = coeffs[:, 3] @ self.basis_v.T   # [B, H]

        g_phys = pred.new_zeros(B, self.H * self.W)
        g_phys[:, self.idx_bottom] = u_bottom
        g_phys[:, self.idx_top]    = u_top
        g_phys[:, self.idx_left]   = u_left
        g_phys[:, self.idx_right]  = u_right

        g_enc = encode_target(g_phys.unsqueeze(-1), self.target_normalizer)
        particular = g_enc * self.boundary_keep

        out = apply_boundary_ansatz(
            pred=pred,
            particular=particular,
            distance=self.boundary_distance,
            channel_mask=channel_mask(pred, [0]),
        )

        if return_aux:
            correction = (out[:, self.idx_all_boundary, 0] - pred[:, self.idx_all_boundary, 0]).abs()
            diagnostics = {
                "constraint/boundary_correction_mean_abs": ConstraintDiagnostic(
                    value=correction.mean(), reduce="mean"
                ),
            }
            return self.as_output(
                out,
                aux={"boundary_pred": out[:, self.idx_all_boundary, 0]},
                diagnostics=diagnostics,
            )
        return out
