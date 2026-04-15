from __future__ import annotations

import torch.nn as nn


class ConstraintModule(nn.Module):
    """Base class for constraint operators applied around a backbone."""

    name = "constraint"

