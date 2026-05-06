from __future__ import annotations

from pathlib import Path

from omni_hc.training.optuna_utils import apply_optuna_search_space


def run_optuna_search(
    cfg: dict,
    *,
    nsl_root: str | Path | None,
    device,
    train_fn,
):
    try:
        import optuna
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "Optuna is required for search. Install it with `pip install optuna` "
            "or `pip install -e '.[experiments]'`."
        ) from exc

    optuna_cfg = cfg.get("optuna", {}) or {}
    search_space = optuna_cfg.get("search_space", {})
    if not search_space:
        raise ValueError("optuna.search_space is required")
    num_trials = int(optuna_cfg.get("num_trials", 10))
    direction = str(optuna_cfg.get("direction", "minimize"))
    base_output_dir = Path(optuna_cfg.get("save_dir", cfg["paths"]["output_dir"]))

    def objective(trial):
        trial_cfg = apply_optuna_search_space(cfg, trial, search_space)
        trial_cfg.setdefault("paths", {})
        trial_cfg["paths"]["output_dir"] = str(
            base_output_dir / f"trial_{trial.number:03d}"
        )
        if "wandb_logging" in trial_cfg:
            trial_cfg["wandb_logging"] = {
                **trial_cfg["wandb_logging"],
                "run_name": (
                    f"{optuna_cfg.get('run_name', 'optuna')}_trial_{trial.number:03d}"
                ),
            }
        result = train_fn(trial_cfg, nsl_root=nsl_root, device=device)
        if "best_score" in result:
            return float(result["best_score"])
        return float(result["best_val_rel_l2"])

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=num_trials)
    return study
