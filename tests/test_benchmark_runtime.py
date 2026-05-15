from omni_hc.benchmarks import BENCHMARKS
from omni_hc.training import get_benchmark_adapter


def test_navier_stokes_adapter_is_registered():
    adapter = BENCHMARKS["navier_stokes_2d"]

    assert adapter.name == "navier_stokes_2d"
    assert callable(adapter.train_fn)
    assert callable(adapter.test_fn)
    assert callable(adapter.tune_fn)


def test_darcy_adapter_is_registered():
    adapter = BENCHMARKS["darcy_2d"]

    assert adapter.name == "darcy_2d"
    assert callable(adapter.train_fn)
    assert callable(adapter.test_fn)
    assert callable(adapter.tune_fn)


def test_pipe_adapter_is_registered():
    adapter = BENCHMARKS["pipe_2d"]

    assert adapter.name == "pipe_2d"
    assert callable(adapter.train_fn)
    assert callable(adapter.test_fn)
    assert callable(adapter.tune_fn)


def test_elasticity_adapter_is_registered():
    adapter = BENCHMARKS["elasticity_2d"]

    assert adapter.name == "elasticity_2d"
    assert callable(adapter.train_fn)
    assert callable(adapter.test_fn)
    assert callable(adapter.tune_fn)


def test_adapter_resolution_uses_benchmark_name():
    adapter = get_benchmark_adapter({"benchmark": {"name": "navier_stokes_2d"}})

    assert adapter.name == "navier_stokes_2d"
    assert "flow" in adapter.domain


def test_adapter_resolution_supports_darcy():
    adapter = get_benchmark_adapter({"benchmark": {"name": "darcy_2d"}})

    assert adapter.name == "darcy_2d"


def test_adapter_resolution_supports_pipe():
    adapter = get_benchmark_adapter({"benchmark": {"name": "pipe_2d"}})

    assert adapter.name == "pipe_2d"


def test_adapter_resolution_supports_elasticity():
    adapter = get_benchmark_adapter({"benchmark": {"name": "elasticity_2d"}})

    assert adapter.name == "elasticity_2d"
    assert "elasticity" in adapter.domain
