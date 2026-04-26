from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
import torch.nn as nn


@dataclass
class ConstraintDiagnostic:
    value: torch.Tensor | float
    reduce: str = "mean"


@dataclass
class ConstraintOutput:
    pred: torch.Tensor
    aux: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, ConstraintDiagnostic] = field(default_factory=dict)


class ConstraintModule(nn.Module):
    """Base class for constraint operators applied around a backbone."""

    name = "constraint"

    def as_output(
        self,
        pred: torch.Tensor,
        *,
        aux: dict[str, Any] | None = None,
        diagnostics: dict[str, ConstraintDiagnostic] | None = None,
    ) -> ConstraintOutput:
        return ConstraintOutput(
            pred=pred,
            aux={} if aux is None else aux,
            diagnostics={} if diagnostics is None else diagnostics,
        )


class LatentExtractor(Protocol):
    def reset(self) -> None: ...

    def get(self) -> torch.Tensor | None: ...


class ConstrainedModel(nn.Module):
    """
    Wraps a backbone and applies a constraint to its prediction.

    The wrapper keeps backbone construction separate from constraint logic. If
    a latent extractor is supplied, it is reset before the backbone forward pass
    and its captured tensor is passed to the constraint.
    """

    def __init__(
        self,
        backbone: nn.Module,
        constraint: nn.Module | None = None,
        latent_extractor: LatentExtractor | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.constraint = constraint
        self.latent_extractor = latent_extractor
        self.supports_aux = constraint is not None

    def forward(self, *args, return_aux=False, **kwargs):
        if self.constraint is None:
            return self.backbone(*args, **kwargs)

        if self.latent_extractor is not None:
            self.latent_extractor.reset()

        pred = self.backbone(*args, **kwargs)
        latent = None if self.latent_extractor is None else self.latent_extractor.get()
        coords = args[0] if len(args) > 0 else kwargs.get("coords")
        fx = args[1] if len(args) > 1 else kwargs.get("fx")
        return self.constraint(
            pred=pred,
            latent=latent,
            coords=coords,
            fx=fx,
            return_aux=return_aux,
        )
