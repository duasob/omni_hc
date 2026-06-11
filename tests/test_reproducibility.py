import random

import numpy as np
import pytest
import torch

from omni_hc.training.reproducibility import seed_everything, seeded_generator
from omni_hc.training.tasks import autoregressive, dynamic_conditional, steady


def test_seed_everything_repeats_python_numpy_and_torch_draws():
    seed_everything(17)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(4),
    )

    seed_everything(17)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(4),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_seeded_generators_are_isolated_from_global_rng_draws():
    first = torch.randperm(12, generator=seeded_generator(23, offset=1))

    torch.manual_seed(999)
    _ = torch.rand(100)

    second = torch.randperm(12, generator=seeded_generator(23, offset=1))

    assert torch.equal(first, second)


@pytest.mark.parametrize(
    ("task_module", "train_fn", "extra_kwargs"),
    [
        (
            steady,
            steady.train_steady_task,
            {},
        ),
        (
            autoregressive,
            autoregressive.train_autoregressive_task,
            {"init_task_state": lambda *args, **kwargs: {}},
        ),
        (
            dynamic_conditional,
            dynamic_conditional.train_dynamic_conditional_task,
            {},
        ),
    ],
)
def test_task_seeds_before_loader_and_model(
    monkeypatch,
    task_module,
    train_fn,
    extra_kwargs,
):
    calls = []
    loader = object()

    monkeypatch.setattr(
        task_module,
        "seed_everything",
        lambda seed: calls.append(("seed", seed)),
    )

    def build_loaders(cfg):
        calls.append(("loaders", None))
        return loader, None

    def create_model(*args, **kwargs):
        calls.append(("model", None))
        raise RuntimeError("stop after model ordering check")

    monkeypatch.setattr(task_module, "create_model", create_model)

    with pytest.raises(RuntimeError, match="ordering check"):
        train_fn(
            {"training": {"seed": 73}},
            device=torch.device("cpu"),
            build_train_val_loaders=build_loaders,
            get_meta=lambda unused_loader: {},
            runtime_overrides=lambda meta: {},
            prepare_batch=lambda *args, **kwargs: None,
            **extra_kwargs,
        )

    assert calls == [
        ("seed", 73),
        ("loaders", None),
        ("model", None),
    ]
