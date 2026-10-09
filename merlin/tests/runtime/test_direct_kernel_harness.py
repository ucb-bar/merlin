"""Execute the shared pointer harness and check complete raw output containers."""

import base64
import json
import shutil
import struct
import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.direct_kernel_harness import DirectKernelAbi, render_direct_kernel
from merlin.runtime.out_b64 import OutB64Decoder
from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, FULL_VALUES_B64, ReadbackPolicy


@pytest.fixture
def program():
    floats = struct.pack("<4I", 0x80000000, 0x7F800000, 0x7FA54321, 0xFF800000)
    integers = bytes((128, 255, 0, 127))
    specs = {
        "X": {"dtype": "f32", "shape": [2, 2], "preload_b64": base64.b64encode(floats).decode()},
        "I": {"dtype": "i8", "shape": [4], "preload_b64": base64.b64encode(integers).decode()},
        "Y": {"dtype": "f32", "shape": [2, 2]},
        "J": {"dtype": "i8", "shape": [4]},
    }
    cb = {
        "tensors": specs,
        "kernel_abi": {
            "args": [
                {"tensor": name, "access": access}
                for name, access in (("X", "read"), ("I", "read"), ("Y", "write"), ("J", "write"))
            ],
            "outputs": ["Y", "J"],
        },
    }
    return cb, {"X": {}, "I": {}}


@pytest.mark.parametrize("byte_order", ["little", "big"])
@pytest.mark.parametrize("main_convention", ["void", "primary_context_id"])
def test_native_harness_preserves_every_float_bit_and_signed_container(tmp_path, program, byte_order, main_convention):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    cb, inputs = program
    abi = DirectKernelAbi("copy_control", "complete_control", 8, byte_order, main_convention)
    harness = tmp_path / "harness.c"
    harness.write_text(
        render_direct_kernel(cb, inputs=inputs, readback_policy=ReadbackPolicy(FULL_VALUES_B64), abi=abi)
    )
    (tmp_path / "htif.h").write_text(
        "void console_init(void); void htif_puts(const char*); void htif_exit(int) __attribute__((noreturn));\n"
    )
    codec = runtime_dir() / "baremetal/out_b64.h"
    (tmp_path / "out_b64.h").write_bytes(codec.read_bytes())
    wrapper = tmp_path / "control.c"
    declaration = "unsigned long" if main_convention == "primary_context_id" else "void"
    call = "0" if main_convention == "primary_context_id" else ""
    wrapper.write_text(
        "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
        "static int completed;\nvoid console_init(void){}\n"
        "void htif_puts(const char *s){if(!completed)exit(4);fputs(s,stdout);}\n"
        "void htif_exit(int code){exit(code);}\n"
        "void copy_control(void*x,void*i,void*y,void*j){memcpy(y,x,16);memcpy(j,i,4);}\n"
        "void complete_control(void){completed=1;}\n"
        f"int harness_main({declaration});int main(void){{return harness_main({call});}}\n"
    )
    obj = tmp_path / "harness.o"
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Dmain=harness_main", "-c", str(harness), "-o", str(obj)],
        capture_output=True,
        check=True,
    )
    elf = tmp_path / "control.elf"
    subprocess.run(
        [compiler, "-std=c11", "-O2", str(wrapper), str(obj), "-o", str(elf)], capture_output=True, check=True
    )
    execution = subprocess.run([str(elf)], capture_output=True, text=True, timeout=10, check=True)
    outputs, decoder = {}, OutB64Decoder()
    for line in execution.stdout.splitlines():
        decoder.consume(line.split(), outputs)
    assert execution.stdout.splitlines()[-1] == "DONE"
    assert outputs == {"Y": [[0x80000000, 0x7F800000], [0x7FA54321, 0xFF800000]], "J": [[-128, -1, 0, 127]]}


@pytest.mark.parametrize("defect", ["short_input", "missing_output", "duplicate_slot", "numeric_float"])
def test_incomplete_or_lossy_bindings_refuse_before_compilation(program, defect):
    cb, inputs = json.loads(json.dumps(program))
    if defect == "short_input":
        cb["tensors"]["X"]["preload_b64"] = base64.b64encode(b"\x00").decode()
    elif defect == "missing_output":
        cb["kernel_abi"]["outputs"].pop()
    elif defect == "duplicate_slot":
        cb["kernel_abi"]["args"].append(dict(cb["kernel_abi"]["args"][0]))
    else:
        cb["tensors"]["X"].pop("preload_b64")
        inputs["X"] = {"shape": [2, 2], "values": [-0.0, float("inf"), float("nan"), float("-inf")]}
    with pytest.raises(ValueError):
        render_direct_kernel(
            cb,
            inputs=inputs,
            readback_policy=ReadbackPolicy(FULL_VALUES_B64),
            abi=DirectKernelAbi("copy_control", None, 8, "little", "void"),
        )


def test_explicit_coherent_harness_keeps_original_storage_call_and_completion(program):
    cb, inputs = program
    abi = DirectKernelAbi("copy_control", "complete_control", 8, "little", "void")
    serial = render_direct_kernel(cb, inputs=inputs, readback_policy=ReadbackPolicy(FULL_VALUES_B64), abi=abi)
    memory = render_direct_kernel(cb, inputs=inputs, readback_policy=ReadbackPolicy(COHERENT_DUMP_V1), abi=abi)
    for line in serial.splitlines():
        if line.startswith("static unsigned char tensor_"):
            assert line in memory
    assert "copy_control(tensor_0, tensor_1, tensor_2, tensor_3);" in memory
    assert memory.index("complete_control();") < memory.index('htif_puts("DONE')
    assert "out_b64.h" not in memory and "OUT_" not in memory
    assert memory.count('htif_puts("DONE\\n");') == 1
    for defect in ("coherent_packet_v1", None, {"transport": COHERENT_DUMP_V1}):
        policy = ReadbackPolicy(defect) if isinstance(defect, str) else defect
        with pytest.raises(ValueError, match="complete B64 or coherent"):
            render_direct_kernel(cb, inputs=inputs, readback_policy=policy, abi=abi)
