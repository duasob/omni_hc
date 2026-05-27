from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
import torch.nn as nn

# TODO: revise this
# Keys in constraint YAML blocks that are not constructor parameters.
# Used by the default ConstraintModule.build classmethod.
_BUILD_META_KEYS: frozenset[str] = frozenset({"name", "freeze_base"})

# kwargs passed to ConstrainedModel.forward that are meant for the constraint only
# and must not be forwarded to the backbone.
_CONSTRAINT_ONLY_KWARGS: frozenset[str] = frozenset({"uy_target"})


@dataclass
class ConstraintDiagnostic:
    value: torch.Tensor | float
    reduce: str = "mean"


@dataclass
class ConstraintOutput:
    pred: torch.Tensor
    aux: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, ConstraintDiagnostic] = field(default_factory=dict)
    extra_loss: torch.Tensor | None = None


class ConstraintModule(nn.Module):
    """Base class for constraint operators applied around a backbone."""

    name = "constraint"

    @classmethod
    def log_media(cls, ctx) -> dict[str, str]:
        """Override to emit constraint-specific W&B images or save files.

        ctx is a MediaLogContext (omni_hc.benchmarks.base). Return a dict of
        {key: saved_path} for any files written; return {} for W&B-only logging.
        The default does nothing.
        """
        return {}

    @classmethod
    def build(
        cls,
        backbone: nn.Module,
        model_context: dict[str, Any],
        cfg: dict[str, Any],
    ) -> "ConstrainedModel":
        """Construct a ConstrainedModel from a full resolved run config.

        Subclasses override this when construction requires extra logic (e.g.
        wiring up a latent extractor). The default implementation injects
        model-derived keys that appear in the subclass __init__ signature and
        are absent from the YAML config.
        """
        constraint_section = cfg.get("constraint", {}) or {}
        params = {k: v for k, v in constraint_section.items() if k not in _BUILD_META_KEYS}
        sig = inspect.signature(cls.__init__)
        for key, value in model_context.items():
            if key in sig.parameters and key not in params:
                params[key] = value
        try:
            constraint = cls(**params)
        except TypeError as exc:
            raise ValueError(
                f"Failed to construct {cls.__name__} — ensure all required parameters "
                f"are present in the constraint YAML config. Detail: {exc}"
            ) from exc
        wrapped = ConstrainedModel(backbone=backbone, constraint=constraint)
        if constraint_section.get("freeze_base", False):
            for param in wrapped.backbone.parameters():
                param.requires_grad = False
        return wrapped

    def as_output(
        self,
        pred: torch.Tensor,
        *,
        aux: dict[str, Any] | None = None,
        diagnostics: dict[str, ConstraintDiagnostic] | None = None,
        extra_loss: torch.Tensor | None = None,
    ) -> ConstraintOutput:
        return ConstraintOutput(
            pred=pred,
            aux={} if aux is None else aux,
            diagnostics={} if diagnostics is None else diagnostics,
            extra_loss=extra_loss,
        )


class ConstrainedModel(nn.Module):
    """
    Wraps a backbone and applies a constraint to its prediction.

    The wrapper keeps backbone construction separate from constraint logic.
    Constraints that require latent features (e.g. latent_head mode) own their
    extractor directly and register their own hooks on the backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        constraint: nn.Module | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.constraint = constraint
        self.supports_aux = constraint is not None

    def forward(self, *args, return_aux=False, **kwargs):
        if self.constraint is None:
            return self.backbone(*args, **kwargs)

        backbone_kwargs = {k: v for k, v in kwargs.items() if k not in _CONSTRAINT_ONLY_KWARGS}
        pred = self.backbone(*args, **backbone_kwargs)
        coords = args[0] if len(args) > 0 else kwargs.get("coords")
        fx = args[1] if len(args) > 1 else kwargs.get("fx")
        return self.constraint(
            pred=pred,
            coords=coords,
            fx=fx,
            return_aux=return_aux,
            **kwargs,
        )
