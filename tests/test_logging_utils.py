import torch
import pytest

import omni_hc.training.logging_utils as logging_utils


class FakeWandb:
    def __init__(self):
        self.run = None
        self.init_kwargs = None
        self.finished = False
        self.logged = None
        self.logged_step = None

    class Image:
        def __init__(self, value):
            self.value = value

    def finish(self):
        self.finished = True
        self.run = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        self.run = object()

    def log(self, payload, *, step=None):
        self.logged = payload
        self.logged_step = step


def test_wandb_init_uses_defaults_for_project_and_run_name(monkeypatch):
    fake_wandb = FakeWandb()
    monkeypatch.setattr(logging_utils, "wandb", fake_wandb)

    enabled = logging_utils.init_wandb_if_enabled(
        {
            "paths": {"output_dir": "outputs/pipe/fno_small"},
            "wandb_logging": {"wandb": True},
        }
    )

    assert enabled is True
    assert fake_wandb.init_kwargs["project"] == "omni_hc"
    assert fake_wandb.init_kwargs["name"] == "outputs_pipe_fno_small"


def test_steady_field_logging_adds_shared_comparison_panel(monkeypatch):
    if logging_utils.plt is None:
        pytest.skip("matplotlib is unavailable")

    fake_wandb = FakeWandb()
    fake_wandb.run = object()
    monkeypatch.setattr(logging_utils, "wandb", fake_wandb)

    coeff = torch.linspace(0.0, 1.0, 16).view(1, 16, 1)
    target = torch.linspace(1.0, 2.0, 16).view(1, 16, 1)
    pred = target + 0.1

    logging_utils.log_steady_field_images(
        coeff,
        pred,
        target,
        4,
        4,
        prefix="validation",
        epoch=2,
        step=10,
    )

    assert fake_wandb.logged_step == 10
    assert fake_wandb.logged["epoch"] == 3
    assert "validation/fields" in fake_wandb.logged
    assert "validation/pred" in fake_wandb.logged
    assert "validation/target" in fake_wandb.logged
    assert "validation/error_signed" in fake_wandb.logged
