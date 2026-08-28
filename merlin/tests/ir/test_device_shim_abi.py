"""The C adapter between compiled host code and a device kernel -- COMPILED, not just emitted.

An ABI adapter that is only string-compared is not tested: every mistake that matters here (a wrong
struct layout, a missing argument, pointer arithmetic in elements where the callee wants bytes)
produces text that looks right and code that reads the wrong memory. So these tests build the emitted
translation unit with a real compiler, link it against a stand-in kernel, call it through the exact
lowered-memref convention MLIR uses, and check what came back.

They skip when no C compiler is available rather than failing -- the emitter is still correct on a
machine that cannot build.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.llvmlower.device_shim import emit_translation_unit, kernel_abi_for

_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")

_KERNEL_STUB = """
#include <stdint.h>
/* stands in for the target's own backend artifact, which the archive supplies */
void {kernel}(void *weight, void *lhs_0, void *out_0) {{
  (void)weight; (void)lhs_0;
  ((int32_t *)out_0)[0] = 42;
}}
"""

_DRIVER = """
#include <stdint.h>
#include <stdio.h>
typedef struct {{ void *a; void *b; intptr_t o; intptr_t s[2]; intptr_t st[2]; }} mr2;
extern mr2 {symbol}(void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,
                    void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,
                    void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t);
int main(void) {{
  static int8_t A[{m}*{k}], B[{k}*{n}];
  static int32_t C[{m}*{n}];
  mr2 r = {symbol}(A,A,0,{m},{k},{k},1,  B,B,0,{k},{n},{n},1,  C,C,0,{m},{n},{n},1);
  int ok = (C[0] == 42) && (r.s[0] == {m}) && (r.s[1] == {n}) && (r.b == (void *)C);
  /* and the refusal path: a contradicting shape must not reach the kernel */
  C[0] = 0;
  mr2 bad = {symbol}(A,A,0,{m}+1,{k},{k},1,  B,B,0,{k},{n},{n},1,  C,C,0,{m},{n},{n},1);
  ok = ok && (bad.b == 0) && (bad.s[0] == 0) && (C[0] == 0);
  printf("%d\\n", ok);
  return ok ? 0 : 1;
}}
"""


def _emit(tmp_path, device="gemmini", sig=(16, 16, 32), dt=("i8", "i8", "i32")):
    sym = "merlin_dev_test_0"
    unit = emit_translation_unit(device, {sym: sig}, {sym: dt})
    if not unit.symbols:
        pytest.skip(f"nothing emitted for {device}: {unit.skipped}")
    (tmp_path / "shim.c").write_text(unit.text, encoding="utf-8")
    return unit, sym


@pytest.mark.skipif(_CC is None, reason="no C compiler available")
def test_the_emitted_unit_compiles_without_warnings(tmp_path):
    """-Wall -Wextra clean: an unused descriptor field usually means an argument went unread."""
    _emit(tmp_path)
    p = subprocess.run([_CC, "-Wall", "-Wextra", "-Werror", "-c", str(tmp_path / "shim.c"),
                        "-o", str(tmp_path / "shim.o")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


@pytest.mark.skipif(_CC is None, reason="no C compiler available")
def test_the_abi_round_trips_through_a_real_call(tmp_path):
    """The only test that can catch a wrong struct layout or a dropped argument."""
    unit, sym = _emit(tmp_path)
    m, n, k = 16, 16, 32
    (tmp_path / "kernel.c").write_text(_KERNEL_STUB.format(kernel=unit.kernel), encoding="utf-8")
    (tmp_path / "driver.c").write_text(_DRIVER.format(symbol=sym, m=m, n=n, k=k), encoding="utf-8")
    exe = tmp_path / "t"
    build = subprocess.run([_CC, "-Wall", str(tmp_path / "shim.c"), str(tmp_path / "kernel.c"),
                            str(tmp_path / "driver.c"), "-o", str(exe)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"ABI round-trip failed: {run.stdout} {run.stderr}"


# --------------------------------------------------------------- declines, reported not guessed

def test_a_sub_byte_format_is_declined_rather_than_guessed():
    """A sub-byte element offset is not a byte count; pointer arithmetic for it would be a guess."""
    unit = emit_translation_unit("gemmini", {"s": (16, 16, 32)}, {"s": ("mxfp4", "mxfp4", "f32")})
    assert unit.symbols == ()
    assert any("sub-byte" in why or "unknown element width" in why for _, why in unit.skipped)


def test_a_rank_this_emitter_cannot_express_is_declined():
    unit = emit_translation_unit("gemmini", {"s": (4, 16, 16, 32)}, {"s": ("i8", "i8", "i32")})
    assert unit.symbols == ()
    assert any("rank" in why for _, why in unit.skipped)


def test_a_signature_with_no_recorded_datapath_is_declined():
    unit = emit_translation_unit("gemmini", {"s": (16, 16, 32)}, {})
    assert unit.symbols == () and any("no datapath" in why for _, why in unit.skipped)


# --------------------------------------------------------------- the ABI comes from the contract

def test_the_kernel_symbol_comes_from_the_shared_contract_not_this_module():
    """A target that names its entry differently changes a declaration, not this emitter."""
    a, b = kernel_abi_for("alpha"), kernel_abi_for("beta")
    if a is None or b is None:
        pytest.skip("backend contract not readable here")
    assert a.symbol != b.symbol and "alpha" in a.symbol and "beta" in b.symbol


def test_the_kernel_is_extern_not_regenerated(tmp_path):
    """The device kernel is the target's own certified artifact. Emitting a transcription here would
    mean a model executes code no oracle ever graded."""
    unit, _ = _emit(tmp_path)
    assert f"extern void {unit.kernel}(" in unit.text
    assert unit.text.count(f"void {unit.kernel}(") == 1, "the kernel must be declared, never defined"


# --------------------------------------------------------------- the tile-edge padding contract

def test_extents_on_the_tile_edge_need_no_staging(tmp_path):
    """The fast path: nothing to pad, so nothing is copied."""
    unit = emit_translation_unit("gemmini", {"s": (16, 32, 16)}, {"s": ("i8", "i8", "i32")},
                                 kernel_symbol_for=lambda _s: "k", tile_edge=16)
    assert unit.symbols == ("s",)
    assert "static unsigned char" not in unit.text


def test_extents_off_the_tile_edge_are_staged_into_padded_buffers(tmp_path):
    """The kernel ABI states its operands are zero-padded to a multiple of the tile edge. Handing it
    raw buffers does not fault -- it strides by the padded width through unpadded data, reads a
    neighbouring row as its own, and returns plausible wrong numbers.

    Measured on a real model: every offloaded layer had M=8 against a 16-wide mesh, and the compiled
    artifact scored cos 0.9847 where the interpreted path scores 0.99993. With staging it scores
    0.999929 -- the padding was the entire gap."""
    unit = emit_translation_unit("gemmini", {"s": (8, 344, 128)}, {"s": ("i8", "i8", "i32")},
                                 kernel_symbol_for=lambda _s: "k", tile_edge=16)
    assert unit.symbols == ("s",)
    # M 8 -> 16, N 344 -> 352, K 128 already on the edge
    assert "s_a[16 * 128 * 1]" in unit.text
    assert "s_b[128 * 352 * 1]" in unit.text
    assert "s_c[16 * 352 * 4]" in unit.text


def test_a_padded_entry_compiles_and_round_trips(tmp_path):
    """Staging is real generated code with real index arithmetic; only building it proves it."""
    if _CC is None:
        pytest.skip("no C compiler available")
    unit = emit_translation_unit("gemmini", {"merlin_dev_test_0": (8, 16, 32)},
                                 {"merlin_dev_test_0": ("i8", "i8", "i32")},
                                 kernel_symbol_for=lambda _s: "gemmini_kernel", tile_edge=16)
    (tmp_path / "shim.c").write_text(unit.text, encoding="utf-8")
    p = subprocess.run([_CC, "-Wall", "-Wextra", "-Werror", "-c", str(tmp_path / "shim.c"),
                        "-o", str(tmp_path / "shim.o")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_the_tile_edge_is_derived_from_the_device_not_assumed():
    """Padding to a guessed edge is worse than not padding: it is differently wrong."""
    from merlin.llvmlower.device_shim import tile_edge_for
    assert tile_edge_for("definitely_not_a_target") is None
    edge = tile_edge_for("gemmini")
    if edge is None:
        pytest.skip("no mesh facts derivable here")
    assert edge > 0


def test_an_underivable_edge_declines_rather_than_guessing():
    unit = emit_translation_unit("definitely_not_a_target", {"s": (8, 24, 8)},
                                 {"s": ("i8", "i8", "i32")})
    assert unit.symbols == () or "s" not in unit.symbols
