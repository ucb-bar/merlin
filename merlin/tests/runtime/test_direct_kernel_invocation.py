"""Actual repeated ordinary pointer calls; no runtime/effect qualification.

The original copies preserve every input container word. A real second-call
defect is hidden by correct final outputs but exposed by complete histories.
"""

import base64
import json
import os
import shutil
import struct
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.runtime.direct_kernel_harness import DirectKernelAbi, render_direct_kernel
from merlin.runtime.direct_kernel_invocation import DirectKernelInvocationPlan
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, FULL_VALUES_B64, ReadbackPolicy


@pytest.fixture
def program():
    original = """module { func.func @main(%X: tensor<2x2xf32>, %I: tensor<4xi8>)
      -> (tensor<2x2xf32>, tensor<4xi8>) {
      %y = tensor.empty() : tensor<2x2xf32>
      %j = tensor.empty() : tensor<4xi8>
      %Y = linalg.copy ins(%X : tensor<2x2xf32>) outs(%y : tensor<2x2xf32>) -> tensor<2x2xf32>
      %J = linalg.copy ins(%I : tensor<4xi8>) outs(%j : tensor<4xi8>) -> tensor<4xi8>
      func.return %Y, %J : tensor<2x2xf32>, tensor<4xi8>
    } }"""
    parsed = parse_linalg_mlir(original)
    abi = CompileOnlySourceAbi(
        tuple(
            CompileOnlyTensor(name, tuple(row["shape"]), row["dtype"])
            for name, row in zip(("X", "I"), parsed["args"], strict=True)
        ),
        tuple(
            CompileOnlyTensor(name, tuple(row["shape"]), row["dtype"])
            for name, row in zip(("Y", "J"), parsed["results"], strict=True)
        ),
    )
    floats = struct.pack("<4I", 0x80000000, 0x7F800000, 0x7FA54321, 0xFF800000)
    integers = bytes((128, 255, 0, 127))
    cb = {
        "tensors": {
            "X": {"dtype": "f32", "shape": [2, 2], "preload_b64": base64.b64encode(floats).decode()},
            "I": {"dtype": "i8", "shape": [4], "preload_b64": base64.b64encode(integers).decode()},
            "Y": {"dtype": "f32", "shape": [2, 2]},
            "J": {"dtype": "i8", "shape": [4]},
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [
                {"tensor": name, "access": access}
                for name, access in (("X", "read"), ("I", "read"), ("Y", "write"), ("J", "write"))
            ],
            "outputs": ["Y", "J"],
        },
    }
    return original, cb, {"X": {}, "I": {}}, DirectKernelInvocationPlan(abi, 3, "history", "completed_count")


@pytest.mark.parametrize("byte_order", ["little", "big"])
@pytest.mark.parametrize("defect", [False, True])
def test_actual_native_full_histories_expose_hidden_middle_call(tmp_path, program, byte_order, defect):
    original, cb, inputs, plan = program
    selected = os.environ.get("MERLIN_TEST_CLANG") or os.environ.get("MERLIN_CLANG") or shutil.which("cc")
    if selected is None:
        pytest.skip("requires an explicitly selected or local native C compiler")
    compiler = Path(selected).resolve(strict=True)
    abi = DirectKernelAbi("copy_control", "complete_control", 8, byte_order, "void")
    harness = tmp_path / "harness.c"
    harness.write_text(
        render_direct_kernel(
            cb, inputs=inputs, readback_policy=ReadbackPolicy(COHERENT_DUMP_V1), abi=abi, invocation_plan=plan
        )
    )
    source = tmp_path / "original.mlir"
    source.write_text(original)
    header = tmp_path / "htif.h"
    header.write_text(
        "void console_init(void);void htif_puts(const char*);void htif_exit(int) __attribute__((noreturn));\n"
    )
    control = tmp_path / "control.c"
    control.write_text(
        "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
        "extern unsigned char history_0[48],history_1[12];extern volatile unsigned char completed_count[8];\n"
        "static unsigned calls,completed;static unsigned char final_y[16],final_j[4];\n"
        "void console_init(void){}\n"
        'void htif_puts(const char*s){if(completed!=3 || strcmp(s,"DONE\\n"))exit(4);fputs(s,stderr);}\n'
        "void copy_control(void*x,void*i,void*y,void*j){calls++;memcpy(y,x,16);memcpy(j,i,4);\n"
        "#ifdef BREAK_SECOND_OUTPUT\n"
        "if(calls==2)((unsigned char*)j)[3]=126;\n"
        "#endif\n"
        "memcpy(final_y,y,16);memcpy(final_j,j,4);}\n"
        "void complete_control(void){completed++;}\n"
        "void htif_exit(int code){if(code)exit(code);"
        "fwrite(history_0,1,48,stdout);fwrite(history_1,1,12,stdout);"
        "for(unsigned i=0;i<8;i++)fputc(completed_count[i],stdout);"
        "fwrite(final_y,1,16,stdout);fwrite(final_j,1,4,stdout);exit(0);}\n"
        "int harness_main(void);int main(void){return harness_main();}\n"
    )
    obj, executable = tmp_path / "harness.o", tmp_path / "control.elf"
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    commands = (
        (
            [str(compiler), "-std=c11", "-O2", "-Dmain=harness_main", "-c", str(harness), "-o", str(obj)],
            (harness, header, source),
            (obj,),
            "repeated_harness_object",
        ),
        (
            [
                str(compiler),
                "-std=c11",
                "-O2",
                *(["-DBREAK_SECOND_OUTPUT=1"] if defect else []),
                str(control),
                str(obj),
                "-o",
                str(executable),
            ],
            (control, obj),
            (executable,),
            "repeated_native_link",
        ),
        ([str(executable)], (executable,), (), "repeated_native_execution"),
    )
    for argv, input_files, output_files, stage in commands:
        result = I.run(
            argv,
            directory=tmp_path / "evidence",
            stage=stage,
            inputs=input_files,
            outputs=output_files,
            dependencies=(Path(__file__).resolve(),),
            env=environment,
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
    assert result.stderr == b"DONE\n" and len(result.stdout) == 88
    raw = result.stdout
    objects = {"history_0": raw[:48], "history_1": raw[48:60], "completed_count": raw[60:68]}
    observed = plan.decode(
        objects, cb=cb, entry_symbol=abi.entry_symbol, completion_symbol=abi.completion_symbol, byte_order=byte_order
    )
    expected_floats = base64.b64decode(cb["tensors"]["X"]["preload_b64"])
    if byte_order == "big":
        expected_floats = b"".join(expected_floats[i : i + 4][::-1] for i in range(0, 16, 4))
    expected_integers = base64.b64decode(cb["tensors"]["I"]["preload_b64"])
    assert observed.observed_count == 3
    assert dict(observed.output_bytes)["Y"] == (expected_floats,) * 3
    assert raw[68:] == expected_floats + expected_integers  # Final outputs conceal the genuine middle defect.
    integer_history = dict(observed.output_bytes)["J"]
    assert integer_history[0] == integer_history[2] == expected_integers
    assert (integer_history[1] == expected_integers) is not defect
    if defect:
        assert integer_history[1] == expected_integers[:3] + b"\x7e"
    for record in (tmp_path / "evidence").rglob("invocation.json"):
        I.require_environment(record, environment=environment)


@pytest.mark.parametrize("defect", ["missing", "extra", "short_history", "short_count", "wrong_count", "text_count"])
def test_changed_or_partial_history_roster_refuses(program, defect):
    _, cb, _, plan = program
    objects = {"history_0": bytes(48), "history_1": bytes(12), "completed_count": (3).to_bytes(8, "little")}
    if defect == "missing":
        objects.pop("history_1")
    elif defect == "extra":
        objects["unselected"] = b""
    elif defect == "short_history":
        objects["history_0"] = bytes(47)
    elif defect == "short_count":
        objects["completed_count"] = bytes(7)
    elif defect == "wrong_count":
        objects["completed_count"] = (2).to_bytes(8, "little")
    else:
        objects["completed_count"] = "3"
    with pytest.raises(ValueError):
        plan.decode(
            objects, cb=cb, entry_symbol="copy_control", completion_symbol="complete_control", byte_order="little"
        )


@pytest.mark.parametrize(
    "defect",
    ["outputs", "arguments", "shape", "dtype", "partial", "access", "collision", "bool_count", "missing_original"],
)
def test_original_roster_and_explicit_plan_are_required_before_render(program, defect):
    _, cb, inputs, plan = program
    cb = json.loads(json.dumps(cb))
    if defect == "outputs":
        cb["kernel_abi"]["outputs"].reverse()
    elif defect == "arguments":
        cb["kernel_abi"]["args"].reverse()
    elif defect == "shape":
        cb["tensors"]["Y"]["shape"] = [3]
    elif defect == "dtype":
        cb["tensors"]["Y"]["dtype"] = "i32"
    elif defect == "partial":
        cb["kernel_abi"]["outputs"].pop()
    elif defect == "access":
        cb["kernel_abi"]["args"][0]["access"] = "readwrite"
    elif defect == "collision":
        plan = DirectKernelInvocationPlan(plan.original_abi, 3, "tensor", "completed_count")
    elif defect == "bool_count":
        plan = DirectKernelInvocationPlan(plan.original_abi, True, "history", "completed_count")
    else:
        plan = DirectKernelInvocationPlan(None, 3, "history", "completed_count")
    with pytest.raises(ValueError):
        render_direct_kernel(
            cb,
            inputs=inputs,
            readback_policy=ReadbackPolicy(COHERENT_DUMP_V1),
            abi=DirectKernelAbi("copy_control", "complete_control", 8, "little", "void"),
            invocation_plan=plan,
        )


def test_unselected_default_and_serial_roster_remain_unchanged(program):
    _, cb, inputs, plan = program
    abi = DirectKernelAbi("copy_control", "complete_control", 8, "little", "void")
    default = render_direct_kernel(cb, inputs=inputs, readback_policy=ReadbackPolicy(COHERENT_DUMP_V1), abi=abi)
    assert default.count("copy_control(tensor_0, tensor_1, tensor_2, tensor_3);") == 1
    assert "history_" not in default and "invocation_completed" not in default
    with pytest.raises(ValueError, match="coherent history reader"):
        render_direct_kernel(
            cb, inputs=inputs, readback_policy=ReadbackPolicy(FULL_VALUES_B64), abi=abi, invocation_plan=plan
        )


def test_output_only_original_function_retains_its_complete_history():
    parsed = parse_linalg_mlir("""module { func.func @main() -> tensor<2xi8> {
      %empty = tensor.empty() : tensor<2xi8>
      %zero = arith.constant 0 : i8
      %filled = linalg.fill ins(%zero : i8) outs(%empty : tensor<2xi8>) -> tensor<2xi8>
      func.return %filled : tensor<2xi8>
    } }""")
    (row,) = parsed["results"]
    original = CompileOnlySourceAbi((), (CompileOnlyTensor("filled", tuple(row["shape"]), row["dtype"]),))
    plan = DirectKernelInvocationPlan(original, 2, "history", "completed_count")
    cb = {
        "operand_naming": "positional",
        "tensors": {"result": {"shape": [2], "dtype": "i8"}},
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "result", "access": "write"}],
            "outputs": ["result"],
        },
    }
    rendered = render_direct_kernel(
        cb,
        inputs={},
        readback_policy=ReadbackPolicy(COHERENT_DUMP_V1),
        abi=DirectKernelAbi("output_only", None, 8, "little", "void"),
        invocation_plan=plan,
    )
    assert "output_only(tensor_0);" in rendered
    observed = plan.decode(
        {"history_0": bytes(4), "completed_count": (2).to_bytes(8, "little")},
        cb=cb,
        entry_symbol="output_only",
        byte_order="little",
    )
    assert observed.output_bytes == (("filled", (bytes(2), bytes(2))),)


def test_original_history_extent_overflow_refuses_before_c_render(program):
    _, cb, inputs, original = program
    plan = DirectKernelInvocationPlan(original.original_abi, 1 << 60, "history", "completed_count")
    with pytest.raises(ValueError, match="history offset"):
        render_direct_kernel(
            cb,
            inputs=inputs,
            readback_policy=ReadbackPolicy(COHERENT_DUMP_V1),
            abi=DirectKernelAbi("copy_control", None, 8, "little", "void"),
            invocation_plan=plan,
        )
