"""Finite independent state machine behind the ordinary C Kernel emitter.

These examples test real emitted execution and failure-state observations. They
are not independent qualification of a hardware instruction implementation.
Setter invocation counts are test-only cost instrumentation, excluded from the
logical observations because the setter contract declares no such observer.
"""

import ctypes
import os

from merlin.common.owned_process import run_owned
from merlin.sched.codegen import emit_c_function
from merlin.sched.ir import parse_kernel
from merlin.sched.isa import InstrDef, InstructionSet, Operand


def library(text, directory):
    directory.mkdir(parents=True)
    kernel = parse_kernel(text)
    operands = tuple(Operand(key, "int") for key in ("slot", "a", "b"))
    declarations = {
        "set": InstrDef("set", operands, lambda args: "set(" + ",".join(args) + ");"),
        "pair": InstrDef("pair", operands, lambda args: "pair(" + ",".join(args) + ");"),
        "pulse": InstrDef("pulse", operands, lambda args: "pulse(" + ",".join(args) + ");"),
        "work": InstrDef("work", operands, lambda args: "if (work(" + ",".join(args) + ")) return;"),
    }
    iset = InstructionSet("configuration_fixture", declarations, {}, "/* synchronous */")
    emitted = emit_c_function(kernel, iset, symbol="actual_kernel", index_bits=64)
    source = (
        r"""
#include <stdint.h>
static int64_t fields[4], other, trace_value, works, failed, setters;
static void set(int64_t slot, int64_t a, int64_t b) {
  (void)b; fields[slot] = a; setters++;
}
static void pair(int64_t slot, int64_t a, int64_t b) {
  (void)slot; fields[0] = a; fields[1] = b; setters++;
}
static void pulse(int64_t slot, int64_t a, int64_t b) {
  fields[slot] = a; other += a+b;
}
static int64_t fail_at;
static int work(int64_t slot, int64_t a, int64_t b) {
  (void)slot; other += a+b+1;
  trace_value += fields[0]*7 + fields[1]*11 + a + works;
  if (works++ == fail_at) { other += 17; failed = 1; return 1; }
  return 0;
}
"""
        + emitted
        + r"""
void execute(int64_t x, int64_t y, int64_t failure, int64_t *out) {
  fields[0]=x; fields[1]=y; fields[2]=-31; fields[3]=99;
  other=0; trace_value=0; works=0; failed=0; setters=0; fail_at=failure;
  actual_kernel();
  out[0]=fields[0]; out[1]=fields[1]; out[2]=fields[2]; out[3]=other;
  out[4]=trace_value; out[5]=works; out[6]=failed; out[7]=setters;
}
"""
    )
    c_path = directory / "kernel.c"
    c_path.write_text(source)
    shared = directory / "execution.so"
    process = run_owned(
        [os.environ["MERLIN_CLANG"], "-O0", "-shared", "-fPIC", str(c_path), "-o", str(shared)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (directory / "compile.stdout").write_text(process.stdout)
    (directory / "compile.stderr").write_text(process.stderr)
    if process.returncode:
        raise AssertionError(process.stderr)
    handle = ctypes.CDLL(str(shared.resolve()))
    execute = handle.execute
    execute.argtypes = [ctypes.c_int64] * 3 + [ctypes.POINTER(ctypes.c_int64)]
    execute.restype = None
    return handle, execute


def compare(source, delivered, directory, *, setters=1, body_work=True):
    _, before = library(source, directory / "original")
    _, after = library(delivered, directory / "delivered")
    rows = []
    for x, y in ((-9, 17), (0, 0), (31, -11)):
        for fail_at in (-1, 0, 1, 3, 6):
            left, right = (ctypes.c_int64 * 8)(), (ctypes.c_int64 * 8)()
            before(x, y, fail_at, left)
            after(x, y, fail_at, right)
            assert list(left)[:7] == list(right)[:7]
            assert right[7] == setters and left[7] >= right[7]
            if fail_at == -1:
                assert left[7] == 3 * setters and left[6] == 0
            elif body_work:
                assert left[6] == 1
            else:
                assert left[7] == 3 * setters and left[6] == int(fail_at == 0)
            rows.append(
                {
                    "initial": [x, y],
                    "fail_at": fail_at,
                    "observation": list(left)[:7],
                    "setters_original": left[7],
                    "setters_delivered": right[7],
                }
            )
    return rows
