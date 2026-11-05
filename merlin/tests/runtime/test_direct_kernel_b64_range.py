"""Compile the actual renderer and codec; compare complete logical values."""

import copy
import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.direct_kernel_harness import DirectKernelAbi, render_direct_kernel
from merlin.runtime.out_b64 import OutB64Decoder, OutB64Error
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, ReadbackPolicy, require_full_value_roster


def _execute(tmp_path, values, *, dtype="i32", order="little", after_scan=None):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    width = int(dtype[1:]) // 8
    signed = dtype.startswith("i")
    produced = b"".join(v.to_bytes(width, order, signed=signed) for v in values)
    # Deliberately unrelated inputs: transport selection must inspect the actual
    # returned storage, not the input values or a Python/reference-side bound.
    cb = {
        "tensors": {
            "X": {"shape": [1, len(values)], "dtype": dtype},
            "Y": {"shape": [1, len(values)], "dtype": dtype},
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "X", "access": "read"}, {"tensor": "Y", "access": "write"}],
            "outputs": ["Y"],
        },
    }
    inputs = {"X": {"shape": [1, len(values)], "values": [(1 << (width * 8 - 1)) - 1] * len(values)}}
    before = copy.deepcopy((cb, inputs))
    source = render_direct_kernel(
        cb,
        inputs=inputs,
        readback_policy=ReadbackPolicy(FULL_VALUES_B64),
        abi=DirectKernelAbi("actual_entry", "actual_completion", 8, order, "void"),
    )
    assert (cb, inputs) == before
    harness = tmp_path / "harness.c"
    harness.write_text(source)
    (tmp_path / "htif.h").write_text(
        "void console_init(void);void htif_puts(const char*);void htif_exit(int) __attribute__((noreturn));\n"
    )
    (tmp_path / "out_b64.h").write_bytes((runtime_dir() / "baremetal/out_b64.h").read_bytes())
    mutation = ""
    if after_scan is not None:
        changed = after_scan.to_bytes(width, order, signed=signed)
        mutation = (
            'if(!strncmp(s,"OUT_B64_BEGIN ",14)) {'
            + "const unsigned char changed[]={"
            + ",".join(str(b) for b in changed)
            + "};"
            + "memcpy(returned,changed,sizeof(changed));}"
        )
    wrapper = tmp_path / "control.c"
    wrapper.write_text(
        "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
        "static unsigned char *returned;static int completed;\n"
        "void console_init(void){}\n"
        "void htif_puts(const char*s){if(!completed)exit(90);" + mutation + "fputs(s,stdout);}\n"
        "void htif_exit(int code){exit(code);}\n"
        "void actual_entry(void*x,void*y){(void)x;returned=y;"
        "const unsigned char data[]={" + ",".join(str(b) for b in produced) + "};"
        "memcpy(y,data,sizeof(data));}\n"
        "void actual_completion(void){completed=1;}\n"
        "int harness_main(void);int main(void){return harness_main();}\n"
    )
    obj, elf = tmp_path / "harness.o", tmp_path / "control.elf"
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Dmain=harness_main", "-c", str(harness), "-o", str(obj)],
        capture_output=True,
        check=True,
        timeout=10,
    )
    subprocess.run(
        [compiler, "-std=c11", "-O2", str(wrapper), str(obj), "-o", str(elf)],
        capture_output=True,
        check=True,
        timeout=10,
    )
    result = subprocess.run([str(elf)], capture_output=True, text=True, timeout=10)
    return result, cb


@pytest.mark.parametrize("order", ["little", "big"])
@pytest.mark.parametrize(
    "dtype,values,wire_width,wire_sign",
    [
        ("i32", [-128, -1, 0, 127], 1, "s"),
        ("i32", [0, 128, 255], 1, "u"),
        ("i32", [-129, 128], 2, "s"),
        ("i32", [0, 256, 65535], 2, "u"),
        ("i32", [-32769, 32768], 4, "s"),
        ("i32", [-(1 << 31), (1 << 31) - 1], 4, "s"),
        ("i64", [65536, (1 << 32) - 1], 4, "u"),
        ("i64", [-(1 << 63), (1 << 63) - 1], 8, "s"),
        ("i64", [0, (1 << 63) - 1], 8, "u"),
    ],
)
def test_actual_completed_values_choose_lossless_wire_width(tmp_path, order, dtype, values, wire_width, wire_sign):
    execution, cb = _execute(tmp_path, values, dtype=dtype, order=order)
    assert execution.returncode == 0
    lines = execution.stdout.splitlines()
    assert lines[0] == f"OUT_B64_BEGIN v1 Y 1 {len(values)} {wire_width} {wire_sign}"
    assert lines[-2:] == ["OUT_B64_END", "DONE"]
    outputs, decoder = {}, OutB64Decoder()
    for line in lines:
        decoder.consume(line.split(), outputs)
    decoder.require_closed()
    assert outputs == {"Y": [values]}
    require_full_value_roster(cb, execution.stdout, outputs, policy=ReadbackPolicy(FULL_VALUES_B64))
    assert cb["tensors"]["Y"]["dtype"] == dtype


def test_last_value_of_tail_forces_full_width_and_complete_chunks(tmp_path):
    values = [1] * 384 + [(1 << 31) - 1]
    execution, cb = _execute(tmp_path, values)
    assert execution.returncode == 0
    assert execution.stdout.splitlines()[0].endswith(" 4 u")
    outputs, decoder = {}, OutB64Decoder()
    for line in execution.stdout.splitlines():
        decoder.consume(line.split(), outputs)
    decoder.require_closed()
    assert outputs == {"Y": [values]}
    require_full_value_roster(cb, execution.stdout, outputs, policy=ReadbackPolicy(FULL_VALUES_B64))


@pytest.mark.parametrize("changed", [256, -129])
def test_value_changed_after_scan_cannot_be_truncated_to_wire_width(tmp_path, changed):
    execution, _ = _execute(tmp_path, [1, 2, 3], after_scan=changed)
    assert execution.returncode == 2
    assert "OUT_B64_END" not in execution.stdout and "DONE" not in execution.stdout
    outputs, decoder = {}, OutB64Decoder()
    for line in execution.stdout.splitlines():
        decoder.consume(line.split(), outputs)
    with pytest.raises(OutB64Error, match="without END"):
        decoder.require_closed()
