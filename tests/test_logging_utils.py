import omni_hc.training.logging_utils as logging_utils


class FakeWandb:
    def __init__(self):
        self.run = None
        self.init_kwargs = None
        self.finished = False

    def finish(self):
        self.finished = True
        self.run = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        self.run = object()


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
