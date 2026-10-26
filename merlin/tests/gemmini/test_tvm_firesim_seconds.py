"""Raw guest cycles must not be mislabeled as RunnerResult seconds."""

import importlib.util
import math

import pytest

from merlin.common.paths import repo_root

tvm = pytest.importorskip("tvm")


@pytest.fixture
def units():
    path = repo_root() / "examples/gemmini/comparisons/tvm/firesim_runner_seconds.py"
    spec = importlib.util.spec_from_file_location("seconds_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_future_converts_actual_cost_and_preserves_error(units):
    from tvm import meta_schedule as ms

    class Future:
        def done(self):
            return True

        def result(self):
            return ms.runner.RunnerResult([1200.0], None)

    result = units.SecondsFuture(Future(), 1000).result()
    assert float(result.run_secs[0]) == 1.2
    assert result.error_msg is None

    class Failed(Future):
        def result(self):
            return ms.runner.RunnerResult(None, "censored at incumbent")

    result = units.SecondsFuture(Failed(), 1000).result()
    assert not result.run_secs
    assert "censored" in result.error_msg


@pytest.mark.parametrize("frequency", [0, -1, math.nan, math.inf])
def test_missing_or_invalid_target_frequency_is_rejected(units, frequency):
    with pytest.raises(ValueError):
        units.cycles_to_seconds(1200, frequency)
