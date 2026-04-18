from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
