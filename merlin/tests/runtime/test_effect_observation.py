"""Source outcomes need real invocation markers and complete byte observations."""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower.c_runtime import generate
from merlin.runtime.effect_observation import decode
from merlin.runtime.out_bin import parse_binary_console_details
from merlin.runtime.whole_model_readback import decode_outputs, render_outputs_header
from test_whole_model_readback import _frame


def wire(payload=b"", *, fail=False):
    return (b"MERLIN_EFFECT_BEGIN v1\n" + _frame("source_stdout", np.frombuffer(payload, np.uint8).reshape(1, -1), 1, False)
        + b"MERLIN_EFFECT_END v1 " + (b"assertion_failure\n" if fail else b"return\nDONE\n"))


@pytest.mark.parametrize("fail", (False, True))
def test_source_bytes_cannot_impersonate_driver_markers(fail):
    payload = b"\0before\nDONE\nMERLIN_EFFECT_END v1 return\nOUT_BIN_BEGIN v1 \xff\0after"
    observed = decode(wire(payload, fail=fail), 134 if fail else 0, max_bytes=1024)
    assert observed["stdout"] == payload and observed["outputs"] == {}
    assert observed["termination"] == ("assertion_failure" if fail else "return")


@pytest.mark.parametrize("damage", ("no-start", "false-start", "no-end", "wrong-exit", "truncated", "early-output", "late-output", "fake-done", "overflow"))
def test_incomplete_crashed_or_rebound_outcomes_refuse(damage):
    data, code = wire(b"assertion diagnostic\n", fail=True), 134
    if damage == "no-start": data = data.removeprefix(b"MERLIN_EFFECT_BEGIN v1\n")
    elif damage == "false-start": data = b"pre-call failure\n" + data
    elif damage == "no-end": data = data.replace(b"MERLIN_EFFECT_END v1 assertion_failure\n", b"")
    elif damage == "wrong-exit": code = 17
    elif damage == "truncated": data = data[:-12]
    elif damage == "early-output": data = data.replace(b"OUT_BIN_BEGIN v1 source_stdout", b"OUT_BIN_BEGIN v1 out0")
    elif damage == "late-output": data += _frame("out0", np.asarray([[1]], np.int32), 4, True)
    elif damage == "fake-done": data += b"DONE\n"
    else: data = data.replace(b"MERLIN_EFFECT_END v1 assertion_failure\n", b"MERLIN_EFFECT_ERROR v1 diagnostic-capacity\n")
    with pytest.raises(ValueError): decode(data, code, max_bytes=1024)


def test_incomplete_readback_still_requires_done_without_explicit_outcome_selection():
    with pytest.raises(ValueError, match="did not reach DONE"):
        parse_binary_console_details(wire(fail=True))


@pytest.mark.parametrize("scalar", (False, True))
def test_void_runtime_generation_keeps_empty_result_roster_and_actual_call(tmp_path, scalar):
    source, out = tmp_path / "source", tmp_path / "generated"
    source.mkdir()
    (source / "model.mlir").write_text("module {func.func @forward(" + ("%n:i32" if scalar else "") + ") {return}}")
    Q = {"schema": "merlin.compile-signature.v1", "entry": "forward", "inputs": [
        {"shape": [], "dtype": "i32", "kind": "scalar"}] if scalar else [], "outputs": []}
    (source / "signature.json").write_text(json.dumps(Q))
    (source / "weights.safetensors.manifest.json").write_text(json.dumps({"0": {"kind": "input", "name": "in0"}} if scalar else {}))
    np.savez(source / "inputs.npz", **({"in0": np.asarray(3, np.int32)} if scalar else {}))
    info = generate(source, out, source / "inputs.npz", runtime_signature=source / "signature.json", observe_effects=True)
    assert info["outputs"] == [] and info["n_outputs"] == 0 and info["out_dt"] == "void"
    assert "#define MERLIN_OUT_ELEMS 0" in (out / "model_gen.h").read_text()
    call = (out / "model_call.c").read_text()
    assert call.index("  merlin_effect_begin();") < call.index("  _mlir_ciface_forward(") < call.index("  merlin_effect_return();")
    assert "MERLIN_OUTPUT_NAMES[0]" in render_outputs_header([])
    assert decode_outputs({}, {}, []) == []


def test_actual_C_observer_keeps_integer_words_literals_and_refuses_unknown_formats(tmp_path):
    cc = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not cc: pytest.skip("existing C compiler required")
    harness = runtime_dir() / "baremetal/spike"
    source = tmp_path / "main.c"
    source.write_text('''#include <unistd.h>
      #include <stdlib.h>
      #include "effect_observation.h"
      int printf(const char *,...);int putchar(int);
      void htif_putc(char c) {(void)write(1,&c,1);}
      void htif_puts(const char *s) {while(*s)htif_putc(*s++);}
      void htif_line_flush(int enabled) {(void)enabled;}
      void htif_exit(int code) {_Exit(code);}
      void merlin_effect_failure_inputs(void) {}
      int main(int argc,char **argv) {merlin_effect_begin();
        if(argc>1 && argv[1][0]=='f')printf("%q",1.5);
        if(argc>1 && argv[1][0]=='o')for(int i=0;i<1025;i++)putchar('x');
        printf("%lld %llu %x",(-9223372036854775807LL-1),18446744073709551615ULL,255u);
        putchar(0);printf("\\nDONE\\n");
        if(argc>1)merlin_effect_abort();
        merlin_effect_return();htif_puts("DONE\\n");return 0;}''')
    command = [cc, "-std=c11", "-O1", "-ffreestanding", "-fno-builtin", "-DMERLIN_EFFECT_OBSERVATION",
        "-DMERLIN_PRINTF_FORMAT_ONLY", "-DMERLIN_EFFECT_BYTES=1024", "-I", str(harness), "-I", str(runtime_dir() / "c"), str(source),
        str(harness / "printf_min.c"), str(harness / "effect_observation.c"), "-o", str(tmp_path / "observer")]
    built = subprocess.run(command, capture_output=True, timeout=30)
    assert built.returncode == 0, built.stderr.decode()
    for options, code in (([], 0), (["abort"], 134)):
        process = subprocess.run([str(tmp_path / "observer"), *options], capture_output=True, timeout=5)
        assert process.returncode == code
        observed = decode(process.stdout, code, max_bytes=1024)
        assert observed["stdout"] == b"-9223372036854775808 18446744073709551615 ff\0\nDONE\n"
    for option in ("format", "overflow"):
        process = subprocess.run([str(tmp_path / "observer"), option], capture_output=True, timeout=5)
        assert process.returncode == 125
        with pytest.raises(ValueError): decode(process.stdout, 125, max_bytes=1024)
