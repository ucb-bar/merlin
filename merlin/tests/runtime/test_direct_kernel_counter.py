"""Actual native call boundaries retain full raw samples without timer grants."""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.runtime.direct_kernel_counter import DirectKernelCounterPlan
from merlin.runtime.direct_kernel_harness import DirectKernelAbi, render_direct_kernel
from merlin.runtime.direct_kernel_invocation import DirectKernelInvocationPlan
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, FULL_VALUES_B64, ReadbackPolicy


def program(extent=3, *, repeated=True):
    original = CompileOnlySourceAbi(
        (CompileOnlyTensor("X", (extent,), "i8"),), (CompileOnlyTensor("Y", (extent,), "i8"),)
    )
    cb = {
        "tensors": {name: {"shape": [extent], "dtype": "i8"} for name in ("X", "Y")},
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "X", "access": "read"}, {"tensor": "Y", "access": "write"}],
            "outputs": ["Y"],
        },
    }
    inputs = {"X": {"shape": [extent], "values": [index % 251 - 125 for index in range(extent)]}}
    invocations = DirectKernelInvocationPlan(original, 2, "history", "invocations") if repeated else None
    counters = DirectKernelCounterPlan("raw_counter", "raw_control", "samples", 2, 256)
    return cb, inputs, invocations, counters


@pytest.mark.parametrize("extent", [3, 257])
@pytest.mark.parametrize("byte_order", ["little", "big"])
@pytest.mark.parametrize("completion", [False, True])
def test_actual_native_counter_brackets_include_completion_and_preserve_every_output(
    tmp_path, extent, byte_order, completion
):
    _native(tmp_path, extent, byte_order, completion, repeated=True)


def test_actual_native_single_call_has_complete_samples_without_a_repeat_plan(tmp_path):
    _native(tmp_path, 3, "little", True, repeated=False)


def _native(tmp_path, extent, byte_order, completion, *, repeated):
    selected = os.environ.get("MERLIN_TEST_CLANG") or shutil.which("cc")
    if not selected:
        pytest.skip("raw counter controls require an explicit or local native C compiler")
    compiler = Path(selected).resolve(strict=True)
    cb, inputs, invocation_plan, counter_plan = program(extent, repeated=repeated)
    abi = DirectKernelAbi("original_copy", "original_completion" if completion else None, 8, byte_order, "void")
    roster = counter_plan.bind(cb, abi=abi, invocation_plan=invocation_plan)
    harness = tmp_path / "harness.c"
    harness.write_text(
        render_direct_kernel(
            cb,
            inputs=inputs,
            readback_policy=ReadbackPolicy(COHERENT_DUMP_V1),
            abi=abi,
            invocation_plan=invocation_plan,
            counter_plan=counter_plan,
        )
    )
    header = tmp_path / "htif.h"
    header.write_text("void console_init(void);void htif_puts(const char*);void htif_exit(int);\n")
    count = 2 if repeated else 1
    writer = "".join(f"for(unsigned i=0;i<{size};i++)fputc({name}[i],stdout);" for name, size in roster.items())
    declared = "".join(f"extern volatile unsigned char {name}[{size}];\n" for name, size in roster.items())
    history = f"extern unsigned char history_0[{extent * count}];\n" if repeated else ""
    history_write = f"fwrite(history_0,1,{extent * count},stdout);" if repeated else ""
    control = tmp_path / "control.c"
    control.write_text(
        "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
        + declared
        + history
        + "static uint64_t tick=UINT64_MAX-2,state=1234;static unsigned calls,completed;\n"
        + f"static unsigned char final_output[{extent}];\n"
        + "uint64_t raw_counter(void){return tick++;}\nuint64_t raw_control(void){return state;}\n"
        + "void console_init(void){}\n"
        + 'void htif_puts(const char*s){if(strcmp(s,"DONE\\n"))exit(8);}\n'
        + f"void original_copy(void*x,void*y){{calls++;memcpy(y,x,{extent});tick+=17;"
        + f"for(unsigned i=0;i<{extent};i++)if(((unsigned char*)y)[i]!=(unsigned char)((i%251)-125))exit(9);"
        + f"memcpy(final_output,y,{extent});}}\n"
        + "void original_completion(void){completed++;state++;tick+=9;}\n"
        + f"void htif_exit(int code){{if(code || calls!={count} || completed!={count if completion else 0})exit(10);"
        + writer
        + history_write
        + f"fwrite(final_output,1,{extent},stdout);exit(0);}}\n"
        + "int harness_main(void);int main(void){return harness_main();}\n"
    )
    env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    harness_object, control_object, executable = (tmp_path / name for name in ("harness.o", "control.o", "control"))
    for source, product, flags in ((harness, harness_object, ("-Dmain=harness_main",)), (control, control_object, ())):
        result = I.run(
            [str(compiler), "-O2", "-I", str(tmp_path), *flags, "-c", str(source), "-o", str(product)],
            directory=tmp_path,
            stage="native_raw_counter_object",
            inputs=(source, header),
            outputs=(product,),
            env=env,
            capture_output=True,
            timeout=30,
        )
        result.check_returncode()
    result = I.run(
        [str(compiler), str(harness_object), str(control_object), "-o", str(executable)],
        directory=tmp_path,
        stage="native_raw_counter_link",
        inputs=(harness_object, control_object),
        outputs=(executable,),
        env=env,
        capture_output=True,
        timeout=30,
    )
    result.check_returncode()
    result = I.run(
        [str(executable)],
        directory=tmp_path,
        stage="native_raw_counter_execute",
        env=env,
        capture_output=True,
        timeout=10,
    )
    result.check_returncode()
    offset, objects = 0, {}
    for name, extent_bytes in roster.items():
        objects[name] = result.stdout[offset : offset + extent_bytes]
        offset += extent_bytes
    observation = counter_plan.decode(objects, cb=cb, abi=abi, invocation_plan=invocation_plan)
    assert observation.completed_count == count
    assert observation.calibration_samples == ((2**64 - 3, 2**64 - 2, 1234, 1234), (2**64 - 1, 0, 1234, 1234))
    step = 27 if completion else 18
    expected = tuple(
        (
            1 + index * (step + 1),
            1 + index * (step + 1) + step,
            1234 + index * completion,
            1234 + (index + 1) * completion,
        )
        for index in range(count)
    )
    assert observation.call_samples == expected
    original = bytes((index % 251 - 125) % 256 for index in range(extent))
    assert result.stdout[offset:] == original * ((count + 1) if repeated else 1)
    assert "no units, integrity or complete-stage costs" in observation.scope
    for path in tmp_path.glob("invocations/*/invocation.json"):
        I.require_environment(path, environment=env)


@pytest.mark.parametrize("defect", ["missing", "extra", "partial", "count"])
def test_changed_or_partial_counter_objects_refuse(defect):
    cb, _, invocations, counters = program()
    abi = DirectKernelAbi("kernel", None, 8, "little", "void")
    roster = counters.bind(cb, abi=abi, invocation_plan=invocations)
    objects = {name: bytes(size) for name, size in roster.items()}
    objects["samples_completed"] = (2).to_bytes(8, "little")
    if defect == "missing":
        objects.pop("samples_call_state_after")
    elif defect == "extra":
        objects["other"] = b""
    elif defect == "partial":
        objects["samples_call_end"] = b"\0"
    else:
        objects["samples_completed"] = (1).to_bytes(8, "little")
    with pytest.raises(ValueError):
        counters.decode(objects, cb=cb, abi=abi, invocation_plan=invocations)


@pytest.mark.parametrize(
    "field,value",
    [
        ("counter_symbol", "kernel"),
        ("state_symbol", "tensor_0"),
        ("counter_symbol", "samples_call_start"),
        ("state_symbol", "history_0"),
        ("counter_symbol", "invocations"),
        ("counter_symbol", "counter_end"),
        ("calibration_count", True),
        ("calibration_count", 0),
        ("max_observation_bytes", True),
        ("max_observation_bytes", 32),
    ],
)
def test_ambiguous_names_or_unbounded_storage_refuse_before_render(field, value):
    cb, inputs, invocations, counters = program()
    abi = DirectKernelAbi("kernel", None, 8, "little", "void")
    with pytest.raises(ValueError):
        render_direct_kernel(
            cb,
            inputs=inputs,
            readback_policy=ReadbackPolicy(COHERENT_DUMP_V1),
            abi=abi,
            invocation_plan=invocations,
            counter_plan=replace(counters, **{field: value}),
        )


def test_counter_storage_cannot_be_omitted_by_selecting_console_only_readback():
    cb, inputs, _, counters = program()
    with pytest.raises(ValueError, match="complete coherent"):
        render_direct_kernel(
            cb,
            inputs=inputs,
            readback_policy=ReadbackPolicy(FULL_VALUES_B64),
            abi=DirectKernelAbi("kernel", None, 8, "little", "void"),
            counter_plan=counters,
        )


def test_saved_plan_metadata_has_no_counter_or_cost_qualification():
    _, _, _, counters = program()
    record = json.loads(json.dumps(counters.record()))
    assert record["counter_symbol"] == "raw_counter"
    assert "no counter units, integrity, cold/warm or stage-cost authority" in record["scope"]
