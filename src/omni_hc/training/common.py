from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml


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


def forward_with_optional_aux(model, coords, fx):
    if bool(getattr(model, "supports_aux", False)):
        out = model(coords, fx, return_aux=True)
        if isinstance(out, tuple) and len(out) == 3:
            pred, pred_base, corr = out
            return {
                "pred": pred,
                "pred_mean": float(pred.mean().item()),
                "pred_base_mean": float(pred_base.mean().item()),
                "corr_mean": float(corr.mean().item()),
            }
    pred = model(coords, fx)
    return {
        "pred": pred,
        "pred_mean": float(pred.mean().item()),
        "pred_base_mean": None,
        "corr_mean": None,
    }


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
