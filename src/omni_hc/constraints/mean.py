from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .base import ConstraintDiagnostic, ConstraintModule

_ACTIVATIONS = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


def _get_activation(act):
    if act is None:
        return nn.Identity
    if isinstance(act, str):
        key = act.lower()
        if key not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation: {act}")
        factory = _ACTIVATIONS[key]
        return factory if isinstance(factory, type) else factory
    if issubclass(act, nn.Module):
        return act
    raise TypeError("act must be a string or nn.Module class")


def build_mlp(in_dim, hidden_dim, out_dim, *, n_layers=0, act="gelu"):
    act_cls = _get_activation(act)
    layers = [nn.Linear(in_dim, hidden_dim), act_cls()]
    for _ in range(n_layers):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act_cls())
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def _infer_reduce_dims(tensor, channel_dim):
    if channel_dim < 0:
        channel_dim = tensor.ndim + channel_dim
    return tuple(i for i in range(1, tensor.ndim) if i != channel_dim)


def match_mean(x, target, *, channel_dim=-1, reduce_dims=None):
    if reduce_dims is None:
        reduce_dims = _infer_reduce_dims(x, channel_dim)
    x_mean = x.mean(dim=reduce_dims, keepdim=True)
    target_mean = target.mean(dim=reduce_dims, keepdim=True)
    return x - x_mean + target_mean


def _resolve_reduce_dims(tensor: torch.Tensor, channel_dim: int, reduce_dims):
    if reduce_dims is not None:
        return tuple(int(d) for d in reduce_dims)
    if channel_dim < 0:
        channel_dim = tensor.ndim + channel_dim
    return tuple(i for i in range(1, tensor.ndim) if i != channel_dim)


class MeanConstraint(ConstraintModule):
    """
    Generic mean-preserving wrapper constraint.

    Modes:
    - post_output
    - post_output_learned
    - latent_head
    """

    name = "mean_constraint"

    def __init__(
        self,
        *,
        mode: str,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        n_layers: int = 0,
        act: str = "gelu",
        latent_dim: Optional[int] = None,
        channel_dim: int = -1,
        reduce_dims: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.mode = mode
        self.channel_dim = channel_dim
        self.reduce_dims = tuple(reduce_dims) if reduce_dims is not None else None

        if mode == "post_output":
            self.corr_head = None
        elif mode == "post_output_learned":
            head_hidden = hidden_dim if hidden_dim is not None else max(2 * out_dim, 8)
            self.corr_head = build_mlp(
                out_dim, head_hidden, out_dim, n_layers=n_layers, act=act
            )
        elif mode == "latent_head":
            if latent_dim is None:
                raise ValueError("latent_dim is required for mode='latent_head'")
            head_hidden = hidden_dim if hidden_dim is not None else latent_dim
            self.corr_head = build_mlp(
                latent_dim, head_hidden, out_dim, n_layers=n_layers, act=act
            )
        else:
            raise ValueError(
                f"Unknown mean constraint mode '{mode}'. "
                "Use one of: post_output, post_output_learned, latent_head."
            )

    def forward(self, *, pred, latent=None, return_aux=False, **_unused):
        reduce_dims = _resolve_reduce_dims(pred, self.channel_dim, self.reduce_dims)

        if self.mode == "post_output":
            corr = pred.mean(dim=reduce_dims, keepdim=True)
            out = pred - corr
        else:
            if self.mode == "post_output_learned":
                corr_raw = self.corr_head(pred)
            else:
                if latent is None:
                    raise ValueError("latent is required for mode='latent_head'")
                corr_raw = self.corr_head(latent)
            corr = match_mean(
                corr_raw,
                pred,
                channel_dim=self.channel_dim,
                reduce_dims=reduce_dims,
            )
            out = pred - corr

        if return_aux:
            diagnostics = {
                "constraint/pred_base_mean": ConstraintDiagnostic(
                    value=pred.mean(),
                    reduce="mean",
                ),
                "constraint/corr_mean": ConstraintDiagnostic(
                    value=corr.mean(),
                    reduce="mean",
                ),
            }
            return self.as_output(
                out,
                aux={"pred_base": pred, "corr": corr},
                diagnostics=diagnostics,
            )
        return out


class MeanCorrection(ConstraintModule):
    """
    Mean-correction head that enforces mean(out) == target mean.
    """

    name = "mean_correction"

    def __init__(
        self,
        *,
        latent_dim=None,
        out_dim=None,
        hidden_dim=None,
        n_layers=0,
        act="gelu",
        pred_head=None,
        corr_head=None,
        channel_dim=-1,
        reduce_dims=None,
    ):
        super().__init__()
        if pred_head is None or corr_head is None:
            if latent_dim is None or out_dim is None:
                raise ValueError(
                    "latent_dim and out_dim are required to build missing heads"
                )
        if pred_head is None:
            head_hidden = hidden_dim if hidden_dim is not None else latent_dim
            pred_head = build_mlp(
                latent_dim, head_hidden, out_dim, n_layers=n_layers, act=act
            )
        if corr_head is None:
            head_hidden = hidden_dim if hidden_dim is not None else latent_dim
            corr_head = build_mlp(
                latent_dim, head_hidden, out_dim, n_layers=n_layers, act=act
            )
        self.pred_head = pred_head
        self.corr_head = corr_head
        self.channel_dim = channel_dim
        self.reduce_dims = reduce_dims

    def forward(self, latent, pred=None, *, return_aux=False):
        if pred is None:
            pred = self.pred_head(latent)
        corr_raw = self.corr_head(latent)
        corr = match_mean(
            corr_raw,
            pred,
            channel_dim=self.channel_dim,
            reduce_dims=self.reduce_dims,
        )
        out = pred - corr
        if return_aux:
            diagnostics = {
                "constraint/pred_base_mean": ConstraintDiagnostic(
                    value=pred.mean(),
                    reduce="mean",
                ),
                "constraint/corr_mean": ConstraintDiagnostic(
                    value=corr.mean(),
                    reduce="mean",
                ),
            }
            return self.as_output(
                out,
                aux={"pred_base": pred, "corr": corr},
                diagnostics=diagnostics,
            )
        return out
