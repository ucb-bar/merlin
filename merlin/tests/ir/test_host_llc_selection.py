"""A selected LLVM object producer composes with ordinary host dispatch builds."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import codegen, toolchain
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
    selected = toolchain.host_llc()
    compiler = os.environ.get("MERLIN_COMPILER_PYTHON")
    if selected is None or not selected.is_file() or not compiler or not Path(compiler).is_file():
        pytest.skip("explicit llc and LLVM compiler Python required")
    return selected


def _record(root, actual, expected):
    (root / "complete-output.json").write_text(
        json.dumps(
            {
                "actual": [
                    {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.tolist()}
                    for value in actual
                ],
                "original_expected": [value.tolist() for value in expected],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert len(actual) == len(expected)
    for value, want in zip(actual, expected, strict=True):
        assert value.shape == want.shape and value.dtype == want.dtype
        np.testing.assert_array_equal(value, want)


def test_absent_explicit_selection_preserves_historical_host_clang_commands(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(toolchain, "_env", lambda key: None)
    monkeypatch.setattr(codegen, "clang", lambda: Path("selected-clang"))
    monkeypatch.setattr(codegen, "_run", lambda argv, **kwargs: commands.append((argv, kwargs)))
    codegen.build_host_shared(tmp_path / "source.ll", tmp_path / "model.so")
    assert commands[0] == (
        [Path("selected-clang"), "-O2", "-fPIC", "-c", tmp_path / "source.ll", "-o", tmp_path / "model.o"],
        {"inputs": (tmp_path / "source.ll",), "outputs": (tmp_path / "model.o",)},
    )
    assert len(commands) == 3 and commands[1][0][0] == commands[2][0][0] == "cc"


def test_selected_missing_object_compiler_never_falls_back_to_clang(tmp_path, monkeypatch):
    source = tmp_path / "source.ll"
    source.write_text("define void @forward() { ret void }\n")
    monkeypatch.setenv("MERLIN_LLVM_LLC", str(tmp_path / "absent-llc"))

    def unselected_clang():
        pytest.fail("a missing explicitly selected producer must not discover clang")

    monkeypatch.setattr(codegen, "clang", unselected_clang)
    with pytest.raises((FileNotFoundError, codegen.CodegenError)):
        codegen.build_host_shared(source, tmp_path / "model.so")
    assert not (tmp_path / "model.o").exists() and not (tmp_path / "model.so").exists()


def test_actual_selected_producer_refuses_invalid_llvm_before_runtime_compile_or_link(tmp_path, monkeypatch):
    _native_tools()
    source = tmp_path / "invalid.ll"
    source.write_text("define i32 @forward() { ret i64 13 }\n")

    def later_work():
        pytest.fail("invalid original LLVM must stop before C-runtime compilation and linking")

    monkeypatch.setattr(codegen, "mlir_runtime_c", later_work)
    with pytest.raises(codegen.CodegenError):
        codegen.build_host_shared(source, tmp_path / "model.so")
    assert not (tmp_path / "model.so").exists()


def test_ordinary_repeated_tensor_arguments_and_tail_shape_execute_all_original_values(tmp_path):
    selected = _native_tools()
    source = """builtin.module {
      func.func @forward(%x: tensor<3x5xi32>) -> tensor<3x5xi32> {
        %r = func.call @forward$kernel_0(%x, %x)
          : (tensor<3x5xi32>, tensor<3x5xi32>) -> tensor<3x5xi32>
        func.return %r : tensor<3x5xi32>
      }
      func.func private @forward$kernel_0(%a: tensor<3x5xi32>, %b: tensor<3x5xi32>) -> tensor<3x5xi32> {
        %e = tensor.empty() : tensor<3x5xi32>
        %r = linalg.add ins(%a, %b : tensor<3x5xi32>, tensor<3x5xi32>)
          outs(%e : tensor<3x5xi32>) -> tensor<3x5xi32>
        func.return %r : tensor<3x5xi32>
      }
    }"""
    (tmp_path / "original.mlir").write_text(source)
    outlined = _outline(source)
    assert build_dispatch_program(outlined).nodes[0].inputs == ["b0", "b0"]
    inputs = np.array([2, -3, 5, 7, -11, 13, 17, -19, 23, 29, -31, 37, 41, -43, 47], np.int32).reshape(3, 5)
    actual = execute(outlined, [inputs], tmp_path / "ordinary")
    _record(tmp_path, actual, [inputs * 2])
    records = list((tmp_path / "ordinary/forward_kernel_0/invocations").glob("*/invocation.json"))
    assert len(records) == 3
    documents = [json.loads(path.read_text()) for path in records]
    assert {row["stage"] for row in documents} == {"object", "runtime_object", "link"}
    record = next(row for row in documents if row["stage"] == "object")
    assert Path(record["argv"][0]).resolve() == selected.resolve()
    assert record["status"] == "completed" and record["returncode"] == 0
    assert record["inputs"][0]["path"].endswith("model.ll")
    assert record["outputs"][0]["path"].endswith("model_host.o")


def test_ordinary_mixed_tensor_scalar_result_slots_are_not_reordered_or_wrapped_as_descriptors(tmp_path):
    _native_tools()
    source = """builtin.module {
      func.func @forward(%x: i32) -> (i32, tensor<2x3xi32>, i32) {
        %r:2 = func.call @forward$kernel_0(%x, %x)
          : (i32, i32) -> (tensor<2x3xi32>, i32)
        func.return %r#1, %r#0, %r#1 : i32, tensor<2x3xi32>, i32
      }
      func.func private @forward$kernel_0(%a: i32, %b: i32) -> (tensor<2x3xi32>, i32) {
        %sum = arith.addi %a, %b : i32
        %e = tensor.empty() : tensor<2x3xi32>
        %r = linalg.fill ins(%sum : i32) outs(%e : tensor<2x3xi32>) -> tensor<2x3xi32>
        %three = arith.constant 3 : i32
        %scalar = arith.subi %sum, %three : i32
        func.return %r, %scalar : tensor<2x3xi32>, i32
      }
    }"""
    (tmp_path / "original.mlir").write_text(source)
    outlined = _outline(source)
    program = build_dispatch_program(outlined)
    assert program.results == ["b2", "b1", "b2"]
    actual = execute(outlined, [np.int32(7)], tmp_path / "ordinary")
    _record(tmp_path, actual, [np.array(11, np.int32), np.full((2, 3), 14, np.int32), np.array(11, np.int32)])


def test_ordinary_multi_result_kernel_preserves_distinct_results_and_repeated_call_result_slots(tmp_path):
    _native_tools()
    source = """builtin.module {
      func.func @forward(%x: tensor<2x3xi32>, %z: tensor<2x3xi32>)
          -> (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>) {
        %a:2 = func.call @forward$kernel_0(%x, %x, %z)
          : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>) -> (tensor<2x3xi32>, tensor<2x3xi32>)
        %b = func.call @forward$kernel_1(%a#1, %a#0, %a#1)
          : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
        func.return %a#1, %b, %a#1 : tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32>
      }
      func.func private @forward$kernel_0(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>, %z: tensor<2x3xi32>)
          -> (tensor<2x3xi32>, tensor<2x3xi32>) {
        %e = tensor.empty() : tensor<2x3xi32>
        %sum = linalg.add ins(%a, %b : tensor<2x3xi32>, tensor<2x3xi32>)
          outs(%e : tensor<2x3xi32>) -> tensor<2x3xi32>
        %f = tensor.empty() : tensor<2x3xi32>
        %difference = linalg.sub ins(%z, %a : tensor<2x3xi32>, tensor<2x3xi32>)
          outs(%f : tensor<2x3xi32>) -> tensor<2x3xi32>
        func.return %sum, %difference : tensor<2x3xi32>, tensor<2x3xi32>
      }
      func.func private @forward$kernel_1(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>, %c: tensor<2x3xi32>)
          -> tensor<2x3xi32> {
        %e = tensor.empty() : tensor<2x3xi32>
        %ab = linalg.add ins(%a, %b : tensor<2x3xi32>, tensor<2x3xi32>)
          outs(%e : tensor<2x3xi32>) -> tensor<2x3xi32>
        %f = tensor.empty() : tensor<2x3xi32>
        %abc = linalg.add ins(%ab, %c : tensor<2x3xi32>, tensor<2x3xi32>)
          outs(%f : tensor<2x3xi32>) -> tensor<2x3xi32>
        func.return %abc : tensor<2x3xi32>
      }
    }"""
    (tmp_path / "original.mlir").write_text(source)
    outlined = _outline(source)
    program = build_dispatch_program(outlined)
    assert program.nodes[1].inputs == ["b3", "b2", "b3"]
    assert program.results == ["b3", "b4", "b3"]
    x = np.array([2, -3, 5, 7, -11, 13], np.int32).reshape(2, 3)
    z = np.array([17, 19, -23, 29, 31, -37], np.int32).reshape(2, 3)
    actual = execute(outlined, [x, z], tmp_path / "ordinary")
    _record(tmp_path, actual, [z - x, (z - x) * 2 + x * 2, z - x])
