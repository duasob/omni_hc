from omni_hc.core import compose_run_config


def test_compose_train_config_from_named_components():
    cfg = compose_run_config(
        benchmark="darcy",
        backbone="FNO",
        constraint="darcy_flux_fft_pad",
        budget="debug",
    )

    assert cfg["benchmark"]["name"] == "darcy_2d"
    assert cfg["model"]["backbone"] == "FNO"
    assert cfg["constraint"]["name"] == "darcy_flux_projection"
    assert cfg["training"]["num_epochs"] == 1
    assert cfg["wandb_logging"]["run_name"] == "darcy_fno_darcy_flux_fft_pad_debug_seed_42"
    assert cfg["experiment"]["source_configs"] == [
        "configs/benchmarks/darcy/base.yaml",
        "configs/backbones/darcy/FNO.yaml",
        "configs/constraints/darcy_flux_fft_pad.yaml",
        "configs/budgets/debug.yaml",
    ]


def test_compose_tune_config_adds_search_space_and_trial_dir():
    cfg = compose_run_config(
        benchmark="darcy",
        backbone="Galerkin_Transformer",
        constraint="darcy_flux_fft_pad",
        budget="tune_debug",
        mode="tune",
    )

    assert cfg["optuna"]["num_trials"] == 2
    assert "constraint.padding" in cfg["optuna"]["search_space"]
    assert cfg["optuna"]["save_dir"].endswith("/trials")
    assert "configs/optuna/darcy/darcy_flux_fft_pad.yaml" in cfg["experiment"]["source_configs"]


def test_experiment_config_applies_overrides():
    cfg = compose_run_config(
        experiment="configs/experiments/navier_stokes/fno_small_mean.yaml",
    )

    assert cfg["model"]["backbone"] == "FNO"
    assert cfg["constraint"]["name"] == "mean_correction"
    assert cfg["constraint"]["mode"] == "post_output"
    assert cfg["wandb_logging"]["run_name"] == "navier_stokes_fno_mean"
