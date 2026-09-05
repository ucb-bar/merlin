"""END TO END on the RTL-bootstrap oracle: a float host-lane result comes back as its VALUE.

The text-level tests (``test_host_lane_harness``, ``test_float_readback_decode``) pin the two halves
separately. This one runs the whole path the capsule grader runs -- lower a CPU-lane kernel to an object,
render the runner-owned harness, link, execute on the gemmini spike oracle, parse the console, decode --
and checks the numbers that come out. Nothing is stubbed, so a break anywhere in that chain fails here.

Both mutations are exercised against the SAME executed console, which is what makes the passing case
evidence rather than decoration: an output DECLARED as an integer must come back as the raw container
word, and one declared in the wrong FLOAT format must come back as the wrong numbers.

Heavy (compiles + runs a simulator); marked ``slow``.
"""
from __future__ import annotations

import pytest

from merlin.runtime.backends import base as bk

pytestmark = pytest.mark.slow

gem = bk.get_backend("gemmini")

ROWS, COLS = 2, 4
PITCH = 16                                  # the harness pads every row to the tile edge
VALUES = [-1.5, 0.25, 3.0, -0.125, 7.5, 2.0, -0.5, 1.0]   # exact in f32 AND bf16
EXPECTED = [[v * 2 for v in VALUES[r * COLS:(r + 1) * COLS]] for r in range(ROWS)]


def _f32_kernel() -> str:
    """A CPU-lane kernel: load f32, double it, store f32. No accelerator instruction."""
    out = ["builtin.module {",
           "  llvm.func @gemmini_kernel(%src: !llvm.ptr, %dst: !llvm.ptr) {",
           "    %two = llvm.mlir.constant(2.000000e+00 : f32) : f32"]
    for r in range(ROWS):
        for c in range(COLS):
            i = r * PITCH + c
            out += [f"    %k{i} = llvm.mlir.constant({i} : i64) : i64",
                    f"    %ps{i} = llvm.getelementptr %src[%k{i}] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
                    f"    %v{i} = llvm.load %ps{i} : !llvm.ptr -> f32",
                    f"    %m{i} = llvm.fmul %v{i}, %two : f32",
                    f"    %pd{i} = llvm.getelementptr %dst[%k{i}] : (!llvm.ptr, i64) -> !llvm.ptr, f32",
                    f"    llvm.store %m{i}, %pd{i} : f32, !llvm.ptr"]
    return "\n".join(out + ["    llvm.return", "  }", "}"])


def _bf16_kernel() -> str:
    """The same in bf16, done in the BIT domain (widen by <<16, narrow by >>16) because the baremetal
    toolchain carries no ``__truncsfbf2`` -- which is what a real host-lane lowering also finds."""
    out = ["builtin.module {",
           "  llvm.func @gemmini_kernel(%src: !llvm.ptr, %dst: !llvm.ptr) {",
           "    %two = llvm.mlir.constant(2.000000e+00 : f32) : f32",
           "    %sh = llvm.mlir.constant(16 : i32) : i32"]
    for r in range(ROWS):
        for c in range(COLS):
            i = r * PITCH + c
            out += [f"    %k{i} = llvm.mlir.constant({i} : i64) : i64",
                    f"    %ps{i} = llvm.getelementptr %src[%k{i}] : (!llvm.ptr, i64) -> !llvm.ptr, i16",
                    f"    %v{i} = llvm.load %ps{i} : !llvm.ptr -> i16",
                    f"    %z{i} = llvm.zext %v{i} : i16 to i32",
                    f"    %s{i} = llvm.shl %z{i}, %sh : i32",
                    f"    %f{i} = llvm.bitcast %s{i} : i32 to f32",
                    f"    %m{i} = llvm.fmul %f{i}, %two : f32",
                    f"    %b{i} = llvm.bitcast %m{i} : f32 to i32",
                    f"    %r{i} = llvm.lshr %b{i}, %sh : i32",
                    f"    %n{i} = llvm.trunc %r{i} : i32 to i16",
                    f"    %pd{i} = llvm.getelementptr %dst[%k{i}] : (!llvm.ptr, i64) -> !llvm.ptr, i16",
                    f"    llvm.store %n{i}, %pd{i} : i16, !llvm.ptr"]
    return "\n".join(out + ["    llvm.return", "  }", "}"])


def _cb(dtype: str) -> dict:
    """A host-lane command buffer: declared tensors, NO accelerator command, and the canonical operands
    the (hypothetical) independent golden was computed on -- exactly the shape the runner builds."""
    return {"abi_version": "0.1", "target": "gemmini", "backend": "test",
            "tensors": {"arg0": {"shape": [ROWS, COLS], "dtype": dtype, "role": "input"},
                        "Y0": {"shape": [ROWS, COLS], "dtype": dtype, "role": "output"}},
            "commands": [], "params": {},
            # what the runner attaches for a capsule graded against an independent golden: the
            # operands that golden was computed on, which the device must therefore run on too.
            "canonical_inputs": {"arg0": {"shape": [ROWS, COLS], "values": VALUES}}}


@pytest.mark.skipif(not gem.available("spike"), reason="spike-gemmini oracle unavailable")
@pytest.mark.parametrize("dtype,kernel", [("f32", _f32_kernel), ("bf16", _bf16_kernel)])
def test_a_float_host_lane_result_reaches_the_grader_as_values(dtype, kernel, tmp_path):
    from merlin.targetgen.contract import compile as oot

    res = oot.run_on_oracle(_cb(dtype), kernel(), simulator="spike", target="gemmini",
                            workdir=str(tmp_path), timeout=900)
    assert res["outputs"]["Y0"] == EXPECTED, res["console"][-800:]

    # The device computed on the operands the BUFFER carried, not on a name-materialized fill: the
    # deterministic fill is 0..3 integers, so doubling it could not produce these values.
    assert any(v not in (0, 2, 4, 6) for row in EXPECTED for v in row)

    # MUTATION 1 -- declare the output as an INTEGER of the same width. The same console must then read
    # back as raw container words, not values. Without this the passing case above proves nothing about
    # whether the decode is keyed on the declaration at all.
    from merlin.runtime import fp8_formats as ff
    ints = {"i32": "f32", "i16": "bf16"}
    wrong_int = next(k for k, v in ints.items() if v == dtype)
    outputs, _raw = gem.parse_output(res["console"])
    as_int = bk.decode_float_readback(outputs, {"Y0": wrong_int})
    assert as_int["Y0"] != EXPECTED
    assert all(isinstance(v, int) for row in as_int["Y0"] for v in row)

    # MUTATION 2 -- declare the WRONG float format. The pattern is the same bits; the values are not.
    other = "bf16" if dtype == "f32" else "f32"
    assert bk.decode_float_readback(outputs, {"Y0": other})["Y0"] != EXPECTED

    # and the codes the console actually carried ARE the codes the declared format encodes the answer
    # in -- i.e. the harness printed the stored pattern losslessly.
    flat = [v for row in outputs["Y0"] for v in row]
    want = [int(c) for c in ff.float_to_codes([v for row in EXPECTED for v in row], dtype)]
    assert [c & ((1 << ff.storage_bits(dtype)) - 1) for c in flat] == want
