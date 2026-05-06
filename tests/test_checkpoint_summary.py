from __future__ import annotations

import torch

from omni_hc.training.common import save_checkpoint_bundle


def _payload(epoch: int, *, train_rel_l2: float, val_rel_l2: float):
    return {
        "epoch": epoch,
        "model_state_dict": {"weight": torch.tensor([float(epoch)])},
        "optimizer_state_dict": {},
        "scheduler_state_dict": None,
        "train_metrics": {"rel_l2": train_rel_l2, "loss": train_rel_l2 + 1.0},
        "val_metrics": {"rel_l2": val_rel_l2, "mse": val_rel_l2 + 2.0},
    }


def test_save_checkpoint_bundle_writes_latest_and_best_summary(tmp_path):
    save_checkpoint_bundle(
        tmp_path,
        _payload(1, train_rel_l2=0.5, val_rel_l2=0.4),
        is_best=True,
    )
    save_checkpoint_bundle(
        tmp_path,
        _payload(2, train_rel_l2=0.3, val_rel_l2=0.6),
        is_best=False,
    )

    summary = (tmp_path / "checkpoint_summary.txt").read_text(encoding="utf-8")

    assert "[LATEST]" in summary
    assert "epoch: 2" in summary
    assert "rel_l2: 0.3" in summary
    assert "[BEST]" in summary
    assert "epoch: 1" in summary
    assert "rel_l2: 0.4" in summary
