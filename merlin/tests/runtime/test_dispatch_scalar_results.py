"""By-value scalar results survive the ordinary kernel compiler and dispatcher."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import kernel_backend
from merlin.llvmlower.abi import HostModel
from merlin.runtime.dispatch_runtime import execute
from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program
from merlin.xdsl_dialects.lowering.outline import DispatchInfo, OutlineResult


def _outline(source):
    module = parse_mlir_text(source)
    module.verify()
    driver = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    calls = [op for op in driver.body.block.ops if op.name == "func.call"]
    return OutlineResult(
        module,
        [
            DispatchInfo(
                index,
                call.callee.string_value(),
                "func.func",
                len(call.operands),
                [str(result.type) for result in call.results],
            )
            for index, call in enumerate(calls)
        ],
    )


def _native_tools():
    # Native lowering requires the explicitly selected compiler interpreter;
    # a capture interpreter or a successful host C probe is not a substitute.
    from merlin.llvmlower.toolchain import clang

    selected = os.environ.get("MERLIN_COMPILER_PYTHON")
    if not selected or not Path(selected).is_file() or not clang().is_file():
        pytest.skip("explicit LLVM compiler Python and native clang required")


def _scalar_source(dtype):
    add = "arith.addi" if dtype.startswith("i") else "arith.addf"
    return f"""builtin.module {{
      func.func @forward(%x: {dtype}) -> {dtype} {{
        %y = func.call @forward$kernel_0(%x, %x) : ({dtype}, {dtype}) -> {dtype}
        func.return %y : {dtype}
      }}
      func.func private @forward$kernel_0(%a: {dtype}, %b: {dtype}) -> {dtype} {{
        %y = {add} %a, %b : {dtype}
        func.return %y : {dtype}
      }}
    }}"""


def _record_outputs(root, actual, expected):
    (root / "complete-output.json").write_text(
        json.dumps(
            {
                "actual": [
                    {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
                    for value in actual
                ],
                "original_expected": expected,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize(
    "dtype,value,npdtype",
    [
        ("i32", 7, np.int32),
        ("i64", -(1 << 40) + 13, np.int64),
        ("f32", 1.25, np.float32),
        ("f64", -1.125, np.float64),
    ],
)
def test_ordinary_native_scalar_result_retains_original_value_type_and_repeated_arguments(
    tmp_path,
    dtype,
    value,
    npdtype,
):
    _native_tools()
    source = _scalar_source(dtype)
    (tmp_path / "original.mlir").write_text(source)
    outlined = _outline(source)
    program = build_dispatch_program(outlined)
    assert program.nodes[0].inputs == ["b0", "b0"]
    counters = {}
    actual = execute(outlined, [npdtype(value)], tmp_path / "ordinary", counters=counters)
    _record_outputs(tmp_path, actual, [value * 2])
    assert len(actual) == 1 and actual[0].shape == () and actual[0].dtype == np.dtype(npdtype)
    np.testing.assert_array_equal(actual[0], npdtype(value * 2))
    assert counters["host_kernels_ran"] == 1
    product = tmp_path / "ordinary/forward_kernel_0"
    assert all((product / name).is_file() for name in ("model.ll", "model_host.o", "model_host.so"))


def test_scalar_result_type_is_preserved_when_loading_an_existing_native_kernel_cache(tmp_path, monkeypatch):
    _native_tools()
    monkeypatch.setenv("MERLIN_COMPILE_WORKERS", "1")
    outlined = _outline(_scalar_source("i64"))
    cache = tmp_path / "cache"
    first = execute(outlined, [np.int64(-19)], tmp_path / "first", cache_dir=cache)
    np.testing.assert_array_equal(first[0], np.int64(-38))

    def forbidden_recompile(*_args, **_kwargs):
        pytest.fail("the already compiled original kernel must load from the selected cache")

    monkeypatch.setattr(kernel_backend, "compile_host", forbidden_recompile)
    second = execute(outlined, [np.int64(1 << 40)], tmp_path / "second", cache_dir=cache)
    _record_outputs(tmp_path, [*first, *second], [-38, 1 << 41])
    np.testing.assert_array_equal(second[0], np.int64(1 << 41))
    assert len(list(cache.glob("*.so"))) == 1


def test_native_scalar_results_keep_distinct_ordered_driver_returns_and_repeated_slots(tmp_path):
    _native_tools()
    outlined = _outline("""builtin.module {
      func.func @forward(%x: i32, %z: i32) -> (i32, i32, i32) {
        %a = func.call @forward$kernel_0(%x, %x) : (i32, i32) -> i32
        %b = func.call @forward$kernel_1(%z, %a, %z) : (i32, i32, i32) -> i32
        func.return %b, %a, %b : i32, i32, i32
      }
      func.func private @forward$kernel_0(%a: i32, %b: i32) -> i32 {
        %r = arith.addi %a, %b : i32
        func.return %r : i32
      }
      func.func private @forward$kernel_1(%a: i32, %b: i32, %c: i32) -> i32 {
        %r = arith.subi %a, %b : i32
        %s = arith.addi %r, %c : i32
        func.return %s : i32
      }
    }""")
    program = build_dispatch_program(outlined)
    assert program.nodes[1].inputs == ["b1", "b2", "b1"]
    assert program.results == ["b3", "b2", "b3"]
    actual = execute(outlined, [np.int32(7), np.int32(19)], tmp_path / "ordinary")
    _record_outputs(tmp_path, actual, [24, 14, 24])
    assert len(actual) == 3
    for value, expected in zip(actual, (24, 14, 24)):
        assert value.shape == () and value.dtype == np.dtype("int32")
        np.testing.assert_array_equal(value, np.int32(expected))


def test_output_only_scalar_entry_uses_no_dummy_input_or_output_descriptor(tmp_path):
    _native_tools()
    outlined = _outline("""builtin.module {
      func.func @forward() -> i32 {
        %r = func.call @forward$kernel_0() : () -> i32
        func.return %r : i32
      }
      func.func private @forward$kernel_0() -> i32 {
        %r = arith.constant 17 : i32
        func.return %r : i32
      }
    }""")
    actual = execute(outlined, [], tmp_path / "ordinary")
    _record_outputs(tmp_path, actual, [17])
    assert len(actual) == 1 and actual[0].shape == ()
    np.testing.assert_array_equal(actual[0], np.int32(17))


def test_original_rank_zero_tensor_remains_an_output_descriptor():
    source = """builtin.module {
      func.func @forward(%x: tensor<i32>) -> tensor<i32> {
        func.return %x : tensor<i32>
      }
    }"""
    assert kernel_backend.host_scalar_result_dtype(parse_mlir_text(source)) is None


@pytest.mark.parametrize("results", ["(i32, i32)", "index", "f16", "vector<2xi32>"])
def test_unsupported_scalar_return_abi_refuses_before_any_build(tmp_path, monkeypatch, results):
    from merlin.llvmlower import lower

    source = f"builtin.module {{ func.func private @forward() -> {results} }}"

    def unexpected_build(*_args, **_kwargs):
        pytest.fail("unsupported original return ABI must not reach a native call or build")

    monkeypatch.setattr(lower, "lower_model", unexpected_build)
    with pytest.raises((ValueError, kernel_backend.KernelBackendError), match="scalar result"):
        kernel_backend.compile_host(parse_mlir_text(source), tmp_path)


@pytest.mark.parametrize("dtype", ["index", "f16", "vector<2xi32>", [], False])
def test_invalid_selected_return_type_refuses_before_shared_image_loading(dtype):
    with pytest.raises(ValueError, match="scalar result dtype"):
        HostModel.load("/not-selected/missing.so", scalar_result_dtype=dtype)
