"""Reject unsafe native model buffers before invoking a compiler or library."""

import copy

import numpy as np
import pytest
from larger_model_mlp import fixture, inputs_and_reference
from larger_model_native_demo import compare, execute, validate_contract


@pytest.mark.parametrize("fault", ["argument-shape", "argument-dtype", "output-shape", "missing-argument"])
def test_native_contract_mismatch_refuses_before_execution(tmp_path, monkeypatch, fault):
    import larger_model_native_demo as native

    model = fixture(layers=2, width=4, batch=2, classes=3)
    contract = {"entry": "forward", "arguments": copy.deepcopy(model["arguments"]), "outputs": model["outputs"]}
    source = tmp_path / "original-model.mlir"
    source.write_text(model["source"])
    validate_contract(source, contract)
    if fault == "argument-shape":
        contract["arguments"][0]["shape"] = [2, 1]
    elif fault == "argument-dtype":
        contract["arguments"][0]["dtype"] = "float32"
    elif fault == "output-shape":
        contract["outputs"][0]["shape"] = [2, 1]
    else:
        contract["arguments"].pop()

    def forbidden(*args, **kwargs):
        pytest.fail("a mismatched native source contract reached execution")

    monkeypatch.setattr(native, "execute", forbidden)
    with pytest.raises(ValueError, match="native"):
        compare(tmp_path, tmp_path / "comparison", contract, {})


@pytest.mark.parametrize("fault", ["missing", "extra", "reordered", "broadcast", "convert"])
def test_native_input_mismatch_refuses_before_abi_or_library(tmp_path, monkeypatch, fault):
    import test_kernel_benchmark_formal_gates as kernel

    model = fixture(layers=2, width=4, batch=2, classes=3)
    inputs, _, _ = inputs_and_reference(model)
    contract = {"entry": "forward", "arguments": model["arguments"], "outputs": model["outputs"]}
    if fault == "missing":
        inputs.pop("x")
    elif fault == "extra":
        inputs["foreign"] = np.zeros((1,), dtype=np.float16)
    elif fault == "reordered":
        inputs = dict(reversed(list(inputs.items())))
    elif fault == "broadcast":
        inputs["x"] = np.zeros((1, 4), dtype=np.float16)
    else:
        inputs["x"] = inputs["x"].astype(np.float32)
    llvm = tmp_path / "model.ll"
    llvm.write_text("opaque pointer arity alone cannot authorize buffers")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid inputs reached native ABI handling")

    monkeypatch.setattr(kernel, "_abi_arguments", forbidden)
    with pytest.raises(ValueError, match="provided model inputs|dtype or shape"):
        execute(llvm, tmp_path / "native", contract, inputs)


def test_scenario_limits_are_inclusive_without_array_allocation(monkeypatch):
    import larger_model_native_demo as native

    def forbidden(*args, **kwargs):
        pytest.fail("metadata byte counting allocated an array")

    for name in ("empty", "zeros", "ones"):
        monkeypatch.setattr(np, name, forbidden)
    input_limit = native.MAX_SCENARIO_INPUT_BYTES
    contract = {
        "arguments": [{"shape": [input_limit], "dtype": "int8"}],
        "outputs": [{"shape": [1 << 24], "dtype": "float32"}],
    }
    assert native.require_array_budget(contract) == {"input_bytes": input_limit, "output_bytes": 64 << 20}
    contract["arguments"][0]["shape"][0] += 1
    with pytest.raises(ValueError, match="arguments array byte budget"):
        native.require_array_budget(contract)


def test_budget_counts_actual_dtype_widths_and_sums_all_buffers(monkeypatch):
    import larger_model_native_demo as native

    monkeypatch.setattr(native, "MAX_SCENARIO_INPUT_BYTES", 8)
    monkeypatch.setattr(native, "MAX_SCENARIO_OUTPUT_BYTES", 8)
    contract = {
        "arguments": [{"shape": [2], "dtype": "float16"}, {"shape": [4], "dtype": "bool"}],
        "outputs": [{"shape": [], "dtype": "int64"}],
    }
    assert native.require_array_budget(contract) == {"input_bytes": 8, "output_bytes": 8}
    contract["arguments"].append({"shape": [], "dtype": "int8"})
    with pytest.raises(ValueError, match="arguments array byte budget"):
        native.require_array_budget(contract)


@pytest.mark.parametrize("scope", ["arguments", "outputs"])
def test_generation_budget_refuses_before_rng_even_with_tiny_test_limits(monkeypatch, scope):
    import larger_model_native_demo as native

    monkeypatch.setattr(native, "MAX_SCENARIO_INPUT_BYTES", 8)
    monkeypatch.setattr(native, "MAX_SCENARIO_OUTPUT_BYTES", 8)
    contract = {
        "arguments": [{"name": "arg0", "canonical_input_name": "arg0", "shape": [1], "dtype": "float32"}],
        "outputs": [{"name": "result0", "shape": [1], "dtype": "float32"}],
    }
    contract[scope][0]["shape"] = [3]
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: pytest.fail("over-budget scenario reached RNG"))
    with pytest.raises(ValueError, match=scope + " array byte budget"):
        native.generated_inputs(contract, seed=1)


@pytest.mark.parametrize("scope", ["arguments", "outputs"])
def test_execution_budget_refuses_before_abi_or_guarded_copies(tmp_path, monkeypatch, scope):
    import larger_model_native_demo as native
    import test_kernel_benchmark_formal_gates as kernel

    monkeypatch.setattr(native, "MAX_SCENARIO_INPUT_BYTES", 8)
    monkeypatch.setattr(native, "MAX_SCENARIO_OUTPUT_BYTES", 8)
    contract = {
        "entry": "forward",
        "arguments": [{"name": "arg0", "canonical_input_name": "arg0", "shape": [1], "dtype": "float32"}],
        "outputs": [{"name": "result0", "shape": [1], "dtype": "float32"}],
    }
    contract[scope][0]["shape"] = [3]
    inputs = {"arg0": np.zeros(contract["arguments"][0]["shape"], dtype=np.float32)}
    llvm = tmp_path / "model.ll"
    llvm.write_text("host-only sentinel; no compiler or library")

    def forbidden(*args, **kwargs):
        pytest.fail("over-budget native scenario reached ABI/copies/library")

    monkeypatch.setattr(kernel, "_abi_arguments", forbidden)
    monkeypatch.setattr(kernel, "_Guarded", forbidden)
    monkeypatch.setattr(native, "_runtime_library", forbidden)
    with pytest.raises(ValueError, match=scope + " array byte budget"):
        native.execute(llvm, tmp_path / "native", contract, inputs)


@pytest.mark.parametrize("scope", ["arguments", "outputs"])
def test_mlp_reference_budget_precedes_independent_array_generation(tmp_path, monkeypatch, scope):
    import larger_model_campaign as campaign
    import larger_model_mlp as mlp
    import larger_model_native_demo as native

    model = fixture(layers=1, width=2, batch=2, classes=3)
    source = tmp_path / "model.mlir"
    source.write_text(model["source"])
    contract = {"arguments": model["arguments"], "outputs": model["outputs"]}
    monkeypatch.setattr(native, "MAX_SCENARIO_INPUT_BYTES" if scope == "arguments" else "MAX_SCENARIO_OUTPUT_BYTES", 1)
    monkeypatch.setattr(mlp, "inputs_and_reference", lambda *a, **k: pytest.fail("budget refusal reached MLP arrays"))
    manifest = {
        "source": native.ref(source),
        "model_parameters": {"layers": 1, "width": 2, "batch": 2, "classes": 3},
    }
    with pytest.raises(ValueError, match=scope + " array byte budget"):
        campaign._mlp_inputs(manifest, contract, 1, tmp_path / "reference")


def test_huge_source_signature_stays_metadata_only_without_scenario_budget(tmp_path, monkeypatch):
    import larger_model_native_demo as native

    extent = (1 << 30) + 1
    source = tmp_path / "model.mlir"
    source.write_text(
        f"module {{func.func @forward(%x:tensor<{extent}xf32>)->tensor<{extent}xf32>"
        f" {{func.return %x:tensor<{extent}xf32>}}}}"
    )
    contract = {
        "entry": "forward",
        "arguments": [{"name": "arg0", "shape": [extent], "dtype": "float32"}],
        "outputs": [{"name": "result0", "shape": [extent], "dtype": "float32"}],
    }
    monkeypatch.setattr(
        native, "require_array_budget", lambda *a: pytest.fail("ABI metadata invoked allocation policy")
    )
    assert native.validate_contract(source, contract) == native.ref(source)


@pytest.mark.parametrize("outcome", ["timeout", "failed"])
def test_host_shared_build_budget_matches_owned_run_and_record(tmp_path, monkeypatch, outcome):
    """Host protocol only: never execute a compiler, native library or proof."""
    import ctypes
    import json
    import subprocess
    from types import SimpleNamespace

    import larger_model_native_demo as native

    from merlin.common import owned_process, paths
    from merlin.llvmlower import codegen, toolchain

    tool = tmp_path / "selected-compiler"
    tool.write_bytes(b"host-only tool-source sentinel")
    llvm = tmp_path / "model.ll"
    llvm.write_text("host-only LLVM sentinel")
    directory = tmp_path / "build"
    directory.mkdir()
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(toolchain, "clang", lambda: tool)
    monkeypatch.setattr(toolchain, "host_llc", lambda: None)
    monkeypatch.setattr(paths, "module_source_path", lambda name: tool)
    monkeypatch.setattr(codegen, "mlir_runtime_c", lambda: tool)
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: pytest.fail("failed fake build reached native loading"))
    cleanup = {"complete": True, "guardian_reaped": True}
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        if outcome == "timeout":
            error = subprocess.TimeoutExpired(command, kwargs["timeout"])
            error.owned_process_cleanup = cleanup
            raise error
        return SimpleNamespace(
            returncode=1, stdout="host sentinel", stderr="failed sentinel", owned_process_cleanup=cleanup
        )

    monkeypatch.setattr(owned_process, "run", fake_run)
    expected_error = subprocess.TimeoutExpired if outcome == "timeout" else RuntimeError
    with pytest.raises(expected_error):
        native._runtime_library(llvm, directory)
    assert native.HOST_SHARED_BUILD_TIMEOUT_SECONDS == 1200
    assert len(observed) == 1
    command, kwargs = observed[0]
    assert kwargs == {"capture_output": True, "text": True, "timeout": native.HOST_SHARED_BUILD_TIMEOUT_SECONDS}
    assert command[1] == "-c"
    assert command[2].endswith("build_host_shared(Path(sys.argv[1]), Path(sys.argv[2]))")
    assert command[3:] == [str(llvm), str(directory / "execution.so")]
    record = json.loads((directory / "BUILD-PROCESS.json").read_text())
    assert record["command"] == command
    assert record["timeout_seconds"] == kwargs["timeout"]
    assert record["builder"] == "merlin.llvmlower.codegen.build_host_shared"
    assert record["owned_process_cleanup"] == cleanup
    assert record["error_type"] == expected_error.__name__
