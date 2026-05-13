import torch
import pytest
from pathlib import Path

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

    class Video:
        def __init__(self, value, *, fps=None, format=None):
            self.value = value
            self.fps = fps
            self.format = format

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


def test_plasticity_mesh_media_logs_images_and_gif(monkeypatch):
    if logging_utils.plt is None:
        pytest.skip("matplotlib is unavailable")

    fake_wandb = FakeWandb()
    fake_wandb.run = object()
    monkeypatch.setattr(logging_utils, "wandb", fake_wandb)

    h, w, t_out, out_dim = 4, 3, 2, 4
    x = -torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
    y = -torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
    coords = torch.stack((x, y), dim=-1)
    sequence = []
    for step in range(t_out):
        step_coords = coords + torch.tensor([0.1 * step, 0.05 * step])
        disp = step_coords - coords
        sequence.append(torch.cat((step_coords, disp), dim=-1))
    pred = torch.stack(sequence, dim=2).reshape(1, h * w, t_out * out_dim)
    target = pred.clone()
    aux = {
        "dx": torch.ones(1, h - 1, w),
        "dy": torch.ones(1, h, w - 1),
    }

    logging_utils.log_plasticity_mesh_consistency_media(
        pred,
        target,
        h,
        w,
        t_out=t_out,
        out_dim=out_dim,
        prefix="validation",
        epoch=0,
        aux_tensors=aux,
        step=1,
        fps=3,
    )

    assert fake_wandb.logged_step == 1
    assert "validation/plasticity_mesh_final" in fake_wandb.logged
    assert "validation/plasticity_mesh_consistency" in fake_wandb.logged
    video = fake_wandb.logged["validation/plasticity_mesh_rollout"]
    assert video.format == "gif"
    assert video.fps == 3
    assert Path(video.value).suffix == ".gif"
    assert Path(video.value).exists()
