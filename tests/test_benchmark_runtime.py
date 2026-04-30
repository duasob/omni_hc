from omni_hc.benchmarks import BENCHMARKS
from omni_hc.training import get_benchmark_adapter


def test_navier_stokes_adapter_is_registered():
    adapter = BENCHMARKS.get("navier_stokes_2d")

    assert adapter.spec.name == "navier_stokes_2d"
    assert callable(adapter.train_fn)
    assert callable(adapter.test_fn)
    assert callable(adapter.tune_fn)


def test_darcy_adapter_is_registered():
    adapter = BENCHMARKS.get("darcy_2d")

    assert adapter.spec.name == "darcy_2d"
    assert callable(adapter.train_fn)
    assert callable(adapter.test_fn)
    assert callable(adapter.tune_fn)


def test_pipe_adapter_is_registered():
    adapter = BENCHMARKS.get("pipe_2d")

    assert adapter.spec.name == "pipe_2d"
    assert callable(adapter.train_fn)
    assert callable(adapter.test_fn)
    assert callable(adapter.tune_fn)


def test_elasticity_adapter_is_registered():
    adapter = BENCHMARKS.get("elasticity_2d")

    assert adapter.spec.name == "elasticity_2d"
    assert callable(adapter.train_fn)
    assert callable(adapter.test_fn)
    assert callable(adapter.tune_fn)


def test_adapter_resolution_uses_benchmark_name():
    adapter = get_benchmark_adapter({"benchmark": {"name": "navier_stokes_2d"}})

    assert adapter.spec.dataset_family == "fno"


def test_adapter_resolution_supports_darcy():
    adapter = get_benchmark_adapter({"benchmark": {"name": "darcy_2d"}})

    assert adapter.spec.dataset_family == "fno"


def test_adapter_resolution_supports_pipe():
    adapter = get_benchmark_adapter({"benchmark": {"name": "pipe_2d"}})

    assert adapter.spec.dataset_family == "fno"


def test_adapter_resolution_supports_elasticity():
    adapter = get_benchmark_adapter({"benchmark": {"name": "elasticity_2d"}})

    assert adapter.spec.dataset_family == "geo-fno"
