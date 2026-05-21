import pytest
import yaml

from omni_hc.core import compose_run_config, parse_dotted_overrides


def test_compose_train_config_from_named_components():
    cfg = compose_run_config(
        benchmark="darcy",
        backbone="FNO",
        constraint="darcy_flux_constraint",
        budget="debug",
    )

    assert cfg["benchmark"]["name"] == "darcy_2d"
    assert cfg["model"]["backbone"] == "FNO"
    assert cfg["constraint"]["name"] == "darcy_flux_constraint"
    assert cfg["training"]["num_epochs"] == 1
    assert cfg["wandb_logging"]["run_name"] == "darcy_fno_darcy_flux_constraint_debug_seed_42"
    assert cfg["paths"]["output_dir"].endswith(
        "outputs/darcy/darcy_flux_constraint/fno/debug/seed_42"
    )
    assert cfg["experiment"]["source_configs"] == [
        "configs/benchmarks/darcy/base.yaml",
        "configs/backbones/darcy/FNO.yaml",
        "configs/constraints/darcy_flux_constraint.yaml",
        "configs/budgets/debug.yaml",
    ]



def test_constraint_alias_resolves_to_config_parameters():
    cfg = compose_run_config(
        benchmark="darcy",
        backbone="Galerkin_Transformer",
        constraint="darcy_flux_constraint",
        budget="final",
    )

    assert cfg["constraint"]["name"] == "darcy_flux_constraint"
    assert cfg["constraint"]["spectral_backend"] == "helmholtz_sine"
    assert cfg["constraint"]["padding"] == 8
    assert "configs/constraints/darcy_flux_constraint.yaml" in cfg["experiment"]["source_configs"]


def test_plasticity_mesh_consistency_alias_resolves_to_config_parameters():
    cfg = compose_run_config(
        benchmark="plasticity",
        backbone="FNO",
        constraint="plasticity_mesh_consistency_constraint",
        budget="debug",
    )

    assert cfg["constraint"]["name"] == "plasticity_mesh_consistency_constraint"
    assert cfg["constraint"]["backbone_out_dim"] == 3
    assert cfg["constraint"]["target_out_dim"] == 4
    assert "configs/constraints/plasticity_mesh_consistency_constraint.yaml" in cfg[
        "experiment"
    ]["source_configs"]


def test_explicit_unknown_constraint_fails_fast():
    with pytest.raises(FileNotFoundError):
        compose_run_config(
            benchmark="darcy",
            backbone="FNO",
            constraint="definitely_missing",
            budget="debug",
        )


def test_experiment_config_applies_overrides():
    cfg = compose_run_config(
        experiment="configs/experiments/navier_stokes/fno_small_mean.yaml",
    )

    assert cfg["model"]["backbone"] == "FNO"
    assert cfg["constraint"]["name"] == "mean_constraint"
    assert cfg["constraint"]["mode"] == "post_output"
    assert cfg["wandb_logging"]["run_name"] == "navier_stokes_fno_mean"


def test_ono_mean_latent_engineered_config_applies_architecture_taps():
    cfg = compose_run_config(
        experiment="configs/experiments/navier_stokes/ono_mean_latent_engineered.yaml",
    )

    assert cfg["model"]["backbone"] == "ONO"
    assert cfg["constraint"]["name"] == "mean_constraint"
    assert cfg["constraint"]["mode"] == "latent_head"
    assert cfg["constraint"]["latent_module"] == [
        "blocks.-2[1]",
        "blocks.-1.Attn",
        "blocks.-1[0]",
        "blocks.-1.proj",
        "blocks.-1.ln_3",
    ]
    assert cfg["constraint"]["latent_dim"] == 528
    assert cfg["wandb_logging"]["run_name"] == "navier_stokes_ono_mean_latent_engineered"


def test_dotted_cli_overrides_apply_to_composed_config():
    cfg = compose_run_config(
        benchmark="navier_stokes",
        backbone="Galerkin_Transformer",
        constraint="mean_constraint",
        budget="debug",
        extra_overrides=parse_dotted_overrides(
            [
                "constraint.mode=latent_head",
                "constraint.latent_module=blocks.-1.ln_3",
            ]
        ),
    )

    assert cfg["constraint"]["mode"] == "latent_head"
    assert cfg["constraint"]["latent_module"] == "blocks.-1.ln_3"


def test_resolved_config_can_be_loaded_as_experiment(tmp_path):
    resolved = compose_run_config(
        benchmark="darcy",
        backbone="Galerkin_Transformer",
        constraint="dirichlet_boundary_ansatz",
        budget="final",
    )
    resolved_path = tmp_path / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")

    cfg = compose_run_config(experiment=resolved_path, mode="test")

    assert cfg["benchmark"]["name"] == "darcy_2d"
    assert cfg["model"]["backbone"] == "Galerkin_Transformer"
    assert cfg["constraint"]["name"] == "dirichlet_boundary_ansatz"
    assert cfg["experiment"]["mode"] == "test"
