from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml

from omni_hc.constraints import ConstraintDiagnostic, ConstraintOutput


def build_optimizer(model, cfg: dict):
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict):
    name = str(cfg.get("scheduler", "none")).lower()
    if name in {"", "none", "null"}:
        return None, False
    if name == "onecyclelr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg["learning_rate"]),
            epochs=int(cfg["num_epochs"]),
            steps_per_epoch=max(int(cfg["steps_per_epoch"]), 1),
            pct_start=float(cfg.get("pct_start", 0.3)),
        )
        return scheduler, True
    if name == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 100)),
            gamma=float(cfg.get("gamma", 0.5)),
        )
        return scheduler, False
    raise ValueError(f"Unknown scheduler: {name}")


def _as_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.numel() != 1:
            raise ValueError(
                f"Expected a scalar tensor metric, got shape {tuple(detached.shape)!r}"
            )
        return float(detached.item())
    return float(value)


def _coerce_diagnostic(value) -> ConstraintDiagnostic:
    if isinstance(value, ConstraintDiagnostic):
        return value
    return ConstraintDiagnostic(value=value, reduce="mean")


class MetricAccumulator:
    def __init__(self) -> None:
        self._state: dict[str, dict[str, float | str]] = {}

    def update(self, metrics: dict[str, Any], *, weight: int = 1) -> None:
        for name, raw in metrics.items():
            diagnostic = _coerce_diagnostic(raw)
            reduce = str(diagnostic.reduce).lower()
            value = _as_scalar(diagnostic.value)
            state = self._state.get(name)
            if state is None:
                if reduce == "mean":
                    self._state[name] = {
                        "reduce": reduce,
                        "total": value * float(weight),
                        "count": float(weight),
                    }
                elif reduce in {"max", "min", "last", "sum"}:
                    self._state[name] = {"reduce": reduce, "value": value}
                else:
                    raise ValueError(f"Unsupported metric reduction '{reduce}'")
                continue

            if state["reduce"] != reduce:
                raise ValueError(
                    f"Mismatched reduction for metric '{name}': "
                    f"{state['reduce']} vs {reduce}"
                )

            if reduce == "mean":
                state["total"] = float(state["total"]) + value * float(weight)
                state["count"] = float(state["count"]) + float(weight)
            elif reduce == "sum":
                state["value"] = float(state["value"]) + value
            elif reduce == "max":
                state["value"] = max(float(state["value"]), value)
            elif reduce == "min":
                state["value"] = min(float(state["value"]), value)
            elif reduce == "last":
                state["value"] = value

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, state in self._state.items():
            reduce = str(state["reduce"])
            if reduce == "mean":
                count = max(float(state["count"]), 1.0)
                metrics[name] = float(state["total"]) / count
            else:
                metrics[name] = float(state["value"])
        return metrics

    def as_diagnostics(self) -> dict[str, ConstraintDiagnostic]:
        aggregated = self.compute()
        return {
            name: ConstraintDiagnostic(value=value, reduce=str(self._state[name]["reduce"]))
            for name, value in aggregated.items()
        }


def prefix_metric_names(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


def diagnostic_values(diagnostics: dict[str, Any]) -> dict[str, float]:
    return {name: _as_scalar(_coerce_diagnostic(value).value) for name, value in diagnostics.items()}


def _normalize_forward_output(output) -> dict[str, Any]:
    if isinstance(output, ConstraintOutput):
        pred = output.pred
        return {
            "pred": pred,
            "pred_mean": float(pred.mean().item()),
            "aux_tensors": dict(output.aux),
            "diagnostics": dict(output.diagnostics),
        }
    if isinstance(output, tuple) and len(output) == 3:
        pred, pred_base, corr = output
        diagnostics = {
            "constraint/pred_base_mean": ConstraintDiagnostic(
                value=pred_base.mean(),
                reduce="mean",
            ),
            "constraint/corr_mean": ConstraintDiagnostic(
                value=corr.mean(),
                reduce="mean",
            ),
        }
        return {
            "pred": pred,
            "pred_mean": float(pred.mean().item()),
            "aux_tensors": {"pred_base": pred_base, "corr": corr},
            "diagnostics": diagnostics,
        }
    pred = output
    return {
        "pred": pred,
        "pred_mean": float(pred.mean().item()),
        "aux_tensors": {},
        "diagnostics": {},
    }


def forward_with_optional_aux(model, coords, fx):
    if bool(getattr(model, "supports_aux", False)):
        return _normalize_forward_output(model(coords, fx, return_aux=True))
    return _normalize_forward_output(model(coords, fx))


def relative_l2_per_sample(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
):
    diff = pred.reshape(pred.shape[0], -1) - target.reshape(target.shape[0], -1)
    diff_norm = torch.norm(diff, p=2, dim=1)
    tgt_norm = torch.norm(target.reshape(target.shape[0], -1), p=2, dim=1).clamp_min(
        eps
    )
    return diff_norm / tgt_norm


def resolve_output_dir(cfg: dict) -> Path:
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def normalize_interval(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None
        return int(text)
    return int(value)


def write_resolved_config(cfg: dict, *, output_dir: Path, resolved_nsl_root: Path):
    payload = deepcopy(cfg)
    payload.setdefault("backend", {})
    payload["backend"]["resolved_nsl_root"] = str(resolved_nsl_root)
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_checkpoint_state(checkpoint_path: str | Path, *, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Expected dict checkpoint at {checkpoint_path}, received {type(checkpoint).__name__}"
        )
    return checkpoint


def save_checkpoint_bundle(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    is_best: bool,
):
    latest_path = output_dir / "latest.pt"
    torch.save(payload, latest_path)
    if is_best:
        torch.save(torch.load(latest_path, map_location="cpu"), output_dir / "best.pt")
