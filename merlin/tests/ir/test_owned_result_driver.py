"""Generated runtime inputs and observed-shape frames execute real source."""

import json
import subprocess
from copy import deepcopy

import numpy as np
import pytest
from owned_result_fixture import COMPACT

from merlin.common.paths import runtime_dir
from merlin.frontends.execution_result_signature import MODE, select_signature
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import toolchain
from merlin.llvmlower.owned_result_driver import generate, verify
from merlin.llvmlower.owned_result_lowering import lower_owned_results
from merlin.llvmlower.target_data_layout import of
from merlin.runtime.owned_result_readback import decode

pytestmark = pytest.mark.skipif(not toolchain.available(), reason="native MLIR unavailable")

def signature(source, rows):
    return select_signature(parse_mlir_text(source), {
        "entry": "forward", "inputs": rows,
        "dimension_bindings": {"n": {"argument": 0, "axis": 0}}
        if rows and rows[0]["shape"] == ["n"] else {},
    }, MODE)


def compile_driver(root, source, contract, arrays):
    generated = root / "driver"
    record = generate(source, contract, arrays, generated)
    lowered = lower_owned_results(
        source, workdir=root / "lower", data_layout=of(toolchain.clang(), ("-O2", "-fPIC")),
        index_bits=64, timeout=30,
    )
    executable = root / "program"
    abi, runtime = runtime_dir() / "abi", runtime_dir() / "c"
    done = subprocess.run(
        [str(toolchain.clang()), "-std=c11", "-Wall", "-Wextra", "-O2",
         "-I", str(abi), "-I", str(runtime), str(generated / "model_owned.c"),
         str(runtime / "owned_result_host_main.c"), str(lowered.llvm),
         *(str(runtime / name) for name in ("owned_result_driver.c", "owned_result_readback.c")),
         *(str(abi / name) for name in ("owned_results.c", "owned_result_views.c", "mlir_runtime.c")),
         "-lm", "-o", str(executable)],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr
    verify(source, contract, arrays, generated)
    return executable, generated, record


@pytest.fixture(scope="module", params=[0, 6])
def compact(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("owned-driver")
    contract = signature(COMPACT, [{"dtype": "i32", "shape": ["n"]}])
    arrays = [np.arange(request.param, dtype=np.int32)]
    return root, contract, *compile_driver(root, COMPACT, contract, arrays), arrays


def test_one_generated_driver_observes_changed_content_and_empty_results(compact):
    root, contract, executable, generated, record, original_arrays = compact
    capacity = original_arrays[0].size
    cases = [np.zeros(capacity, np.int32), np.arange(capacity, dtype=np.int32) - 2,
             np.resize(np.array([-(1 << 31), (1 << 31) - 1, -1, 7], np.int32), capacity)]
    source_before = (generated / "model_owned.c").read_bytes()
    for ordinal, x in enumerate(cases):
        blob = bytearray((generated / "invocation.bin").read_bytes())
        start = record["inputs"][0]["offset"]
        blob[start:start + x.nbytes] = x.tobytes()
        arguments = root / f"arguments-{ordinal}.bin"
        arguments.write_bytes(blob)
        done = subprocess.run([str(executable), str(arguments)], capture_output=True, timeout=5)
        assert done.returncode == 0, done.stderr
        outputs, observation = decode(COMPACT, contract, [x], done.stdout)
        for actual, expected in zip(outputs, (x[x > 0], x[x > 0], x), strict=True):
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes()
        assert observation["functional_inputs_preserved"]
        (root / f"stdout-{ordinal}.bin").write_bytes(done.stdout)
    assert (generated / "model_owned.c").read_bytes() == source_before
    verify(COMPACT, contract, original_arrays, generated)


SCALAR = '''module {func.func @forward(%x:tensor<?xi32> {bufferization.writable=false},
 %n:i64,%value:f64,%flag:i1)->(tensor<?xi32>,tensor<f64>) {
 %scalar="tensor.from_elements"(%value):(f64)->tensor<f64>
 return %x,%scalar:tensor<?xi32>,tensor<f64>
}}'''


@pytest.fixture(scope="module")
def scalars(tmp_path_factory):
    root = tmp_path_factory.mktemp("owned-driver-scalars")
    contract = signature(SCALAR, [
        {"dtype": "i32", "shape": ["n"]},
        {"kind": "scalar", "dtype": "i64", "shape": [], "scalar": "n", "bounds": [0, 31]},
        {"kind": "scalar", "dtype": "f64", "shape": []},
        {"kind": "scalar", "dtype": "i1", "shape": []},
    ])
    arrays = [np.arange(6, dtype=np.int32), np.array(6, np.int64), np.array(1.5), np.array(True)]
    return root, contract, *compile_driver(root, SCALAR, contract, arrays), arrays


@pytest.mark.parametrize("bits", [0x7FF0000000000123, 0xFFF8000000000456, 0x8000000000000000, 1])
def test_native_scalar_and_rank_zero_results_keep_exact_words(scalars, bits):
    root, contract, executable, generated, record, arrays = scalars
    value = np.array(bits, np.uint64).view(np.float64)
    actual_arrays = [*arrays[:2], value, arrays[3]]
    blob = bytearray((generated / "invocation.bin").read_bytes())
    offset = record["inputs"][2]["offset"]
    blob[offset:offset + 8] = value.tobytes()
    path = root / f"words-{bits}.bin"
    path.write_bytes(blob)
    done = subprocess.run([str(executable), str(path)], capture_output=True, timeout=5)
    assert done.returncode == 0, done.stderr
    outputs, _ = decode(SCALAR, contract, actual_arrays, done.stdout)
    assert outputs[1].shape == () and outputs[1].tobytes() == value.tobytes()


@pytest.mark.parametrize("index,value", [(1, -1), (1, 32), (1, 5), (3, 2)])
def test_runtime_scalar_contract_changes_refuse_before_source_publication(scalars, index, value):
    root, _contract, executable, generated, record, _arrays = scalars
    blob = bytearray((generated / "invocation.bin").read_bytes())
    row = record["inputs"][index]
    blob[row["offset"]:row["offset"] + row["bytes"]] = value.to_bytes(row["bytes"], "little", signed=index == 1)
    path = root / f"invalid-{index}-{value}.bin"
    path.write_bytes(blob)
    done = subprocess.run([str(executable), str(path)], capture_output=True, timeout=5)
    assert done.returncode == 100 and done.stdout == b""


@pytest.mark.parametrize("member", ["model_owned.c", "invocation.bin", "signature.json", "driver.json", "extra"])
def test_driver_reopening_reconstructs_source_input_and_complete_member_inventory(compact, member):
    _root, contract, _executable, generated, _record, arrays = compact
    path = generated / member
    original = path.read_bytes() if path.exists() else None
    try:
        if member == "driver.json":
            data = json.loads(original)
            data["result_scope"] = "predicted-results"
            path.write_text(json.dumps(data))
        else:
            path.write_bytes((original or b"") + b"\n")
        with pytest.raises(ValueError, match="owned driver"):
            verify(COMPACT, contract, arrays, generated)
    finally:
        if original is None:
            path.unlink()
        else:
            path.write_bytes(original)


def test_false_rehashed_signature_cannot_generate_a_runtime(tmp_path):
    contract = signature(COMPACT, [{"dtype": "i32", "shape": ["n"]}])
    changed = deepcopy(contract)
    changed["outputs"][0]["shape"] = [7]
    with pytest.raises(ValueError, match="actual source"):
        generate(COMPACT, changed, [np.arange(6, dtype=np.int32)], tmp_path / "never")
    assert not (tmp_path / "never").exists()
