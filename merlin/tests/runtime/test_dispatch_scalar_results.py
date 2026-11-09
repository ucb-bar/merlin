from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.runtime.dispatch_runtime import execute
from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program
from merlin.xdsl_dialects.lowering.outline import DispatchInfo, OutlineResult


def _outline(source):
    module = parse_mlir_text(source)
    module.verify()
    driver = next((op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward"))
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
    from merlin.llvmlower.toolchain import clang

    selected = os.environ.get("MERLIN_COMPILER_PYTHON")
    if not selected or not Path(selected).is_file() or (not clang().is_file()):
        pytest.skip("explicit LLVM compiler Python and native clang required")


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


def test_native_scalar_results_keep_distinct_ordered_driver_returns_and_repeated_slots(tmp_path):
    _native_tools()
    outlined = _outline(
        "builtin.module {\n      func.func @forward(%x: i32, %z: i32) -> (i32, i32, i32) {\n        %a = func.call @forward$kernel_0(%x, %x) : (i32, i32) -> i32\n        %b = func.call @forward$kernel_1(%z, %a, %z) : (i32, i32, i32) -> i32\n        func.return %b, %a, %b : i32, i32, i32\n      }\n      func.func private @forward$kernel_0(%a: i32, %b: i32) -> i32 {\n        %r = arith.addi %a, %b : i32\n        func.return %r : i32\n      }\n      func.func private @forward$kernel_1(%a: i32, %b: i32, %c: i32) -> i32 {\n        %r = arith.subi %a, %b : i32\n        %s = arith.addi %r, %c : i32\n        func.return %s : i32\n      }\n    }"
    )
    program = build_dispatch_program(outlined)
    assert program.nodes[1].inputs == ["b1", "b2", "b1"]
    assert program.results == ["b3", "b2", "b3"]
    actual = execute(outlined, [np.int32(7), np.int32(19)], tmp_path / "ordinary")
    _record_outputs(tmp_path, actual, [24, 14, 24])
    assert len(actual) == 3
    for value, expected in zip(actual, (24, 14, 24)):
        assert value.shape == () and value.dtype == np.dtype("int32")
        np.testing.assert_array_equal(value, np.int32(expected))
