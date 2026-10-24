"""Checks for diagnostic denominators and failure handling."""

import importlib.util

import numpy as np
import pytest

from merlin.common.paths import repo_root


def load_comparison():
    path = repo_root() / "examples/gemmini/comparisons/tvm/compare_tinyllama_outputs.py"
    spec = importlib.util.spec_from_file_location("compare_tinyllama_outputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    return {
        "out0": np.ones((1, 4), dtype=np.float32),
        "out1": np.zeros((2, 2, 1, 1, 10, 2), dtype=np.float32),
        "out2": np.array([1, 1], dtype=np.int64),
    }


def test_padding_does_not_hide_an_active_key_failure():
    comparison = load_comparison()
    reference, actual = fixture(), fixture()
    actual["out1"][0, 1, 0, 0, 0, 1] = 0.1
    result = comparison.compare_call(reference, actual)
    assert not result["passed"]
    assert result["cache_all_slots"]["within_tolerance"] == 79
    assert result["first_active_cache_failure"] == 1
    assert result["cache_by_layer"][1]["key_active"]["within_tolerance"] == 1
    assert result["cache_by_layer"][1]["key_active"]["count"] == 2
    assert result["cache_by_layer"][1]["key_padding"]["passed"]


def test_nonfinite_pairs_and_changed_positions_fail():
    comparison = load_comparison()
    reference, actual = fixture(), fixture()
    reference["out0"][0, 0] = actual["out0"][0, 0] = np.nan
    actual["out2"][1] = 2
    result = comparison.compare_call(reference, actual)
    assert not result["logits"]["passed"]
    assert result["logits"]["nonfinite"] == 1
    assert not result["positions"]["passed"]
    assert not result["passed"]


def test_integer_comparison_remains_exact_above_float64_precision():
    result = load_comparison().statistics(np.array([2**60], dtype=np.int64), np.array([2**60 + 1], dtype=np.int64))
    assert not result["passed"]
    assert result["within_tolerance"] == 0
    assert result["max_abs_finite"] == 1


def test_empty_padding_is_valid_and_bad_abi_is_rejected():
    comparison = load_comparison()
    reference, actual = fixture(), fixture()
    reference["out2"][:] = actual["out2"][:] = 10
    assert comparison.compare_call(reference, actual)["passed"]
    actual["out2"][0] = 11
    with pytest.raises(ValueError, match="capacity"):
        comparison.compare_call(reference, actual)


def test_all_arrays_and_shapes_are_required():
    comparison = load_comparison()
    reference, actual = fixture(), fixture()
    del actual["out1"]
    with pytest.raises(ValueError, match="exactly"):
        comparison.compare_call(reference, actual)


def test_compiled_rounding_preserves_the_observed_threshold_crossing():
    torch = pytest.importorskip("torch")
    tvm = pytest.importorskip("tvm")
    from tvm import relax
    from tvm.relax.frontend.torch import from_exported_program

    class Encode(torch.nn.Module):
        def forward(self, value, scale):
            return torch.round(value / scale).clamp(-127, 127).to(torch.int8)

    value = np.array(
        [[-0.7109373807907104, -0.7109375596046448, 1.0], [-0.5, 0.5, 127.0], [1.5, 2.5, -127.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    scale = np.array([[0.015625], [1.0], [1.0], [1.0]], dtype=np.float32)
    exported = torch.export.export(Encode(), (torch.from_numpy(value), torch.from_numpy(scale)))
    mod = from_exported_program(exported, unwrap_unit_return_tuple=True)
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm.cpu())
    actual = vm["main"](tvm.nd.array(value), tvm.nd.array(scale)).numpy()
    np.testing.assert_array_equal(
        actual, np.array([[-45, -46, 64], [0, 0, 127], [2, 2, -127], [0, 0, 0]], dtype=np.int8)
    )
