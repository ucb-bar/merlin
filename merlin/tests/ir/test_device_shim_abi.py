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

from merlin.llvmlower.device_shim import KernelAbi, emit_translation_unit, kernel_abi_for

pytestmark = pytest.mark.target("gemmini")


@pytest.fixture(autouse=True)
def _legacy_resident_abi(monkeypatch):
    """These tests pin the version-1 resident ABI (weight first, edge tiles staged), which a support
    selects explicitly with ``harness_abi.kernel_abi_version: 1``; the logical ABI's shim is covered by
    ``test_device_shim_logical_abi``."""
    from merlin.targetgen.contract import harness_abi

    monkeypatch.setattr(harness_abi, "kernel_abi_version_for", lambda _device: harness_abi.LEGACY_KERNEL_ABI_VERSION)

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
    p = subprocess.run(
        [_CC, "-Wall", "-Wextra", "-Werror", "-c", str(tmp_path / "shim.c"), "-o", str(tmp_path / "shim.o")],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr


@pytest.mark.skipif(_CC is None, reason="no C compiler available")
def test_the_abi_round_trips_through_a_real_call(tmp_path):
    """The only test that can catch a wrong struct layout or a dropped argument."""
    unit, sym = _emit(tmp_path)
    m, n, k = 16, 16, 32
    (tmp_path / "kernel.c").write_text(_KERNEL_STUB.format(kernel=unit.kernel), encoding="utf-8")
    (tmp_path / "driver.c").write_text(_DRIVER.format(symbol=sym, m=m, n=n, k=k), encoding="utf-8")
    exe = tmp_path / "t"
    build = subprocess.run(
        [
            _CC,
            "-Wall",
            str(tmp_path / "shim.c"),
            str(tmp_path / "kernel.c"),
            str(tmp_path / "driver.c"),
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"ABI round-trip failed: {run.stdout} {run.stderr}"


@pytest.mark.skipif(_CC is None, reason="no C compiler available")
def test_padded_rank2_bridge_is_numerical_and_refuses_bad_pointer_descriptors(tmp_path):
    """The generated bridge passes B,A,C to a kernel and copies valid windows exactly.

    The C kernel is a stand-in for the OOT target artifact, so this proves the
    host pointer/transfer bridge, not accelerator execution or model equivalence.
    """
    unit = emit_translation_unit(
        "gemmini",
        {"selected": (3, 7, 5)},
        {"selected": ("i8", "i8", "i32")},
        kernel_symbol_for=lambda _sym: "selected_kernel",
        tile_edge=16,
    )
    (tmp_path / "shim.c").write_text(unit.text, encoding="utf-8")
    (tmp_path / "kernel.c").write_text(
        """
#include <stdint.h>
void selected_kernel(void *weight, void *lhs, void *out) {
  const int8_t *b = weight, *a = lhs;
  int32_t *c = out;
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j) {
      int32_t sum = 0;
      for (int p = 0; p < 16; ++p) sum += (int32_t)a[i*16+p] * b[p*16+j];
      c[i*16+j] = sum;
    }
}
""",
        encoding="utf-8",
    )
    (tmp_path / "driver.c").write_text(
        """
#include <stdint.h>
typedef struct { void *alloc, *aligned; intptr_t off, size[2], stride[2]; } mr2;
extern mr2 selected(void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,
                    void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,
                    void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t);
int main(int argc, char **argv) {
  (void)argv;
  static int8_t a[3*5], b[5*7];
  static int32_t c[3*7];
  for (int i = 0; i < 3*5; ++i) a[i] = (int8_t)((i*3)%11-5);
  for (int i = 0; i < 5*7; ++i) b[i] = (int8_t)((i*7)%13-6);
  int bad_stride = argc == 2;
  intptr_t bad_offset = argc == 4 ? INTPTR_MAX : 0;
  void *out = argc == 3 ? (void*)a : (void*)c;
  mr2 r = selected(a,a,0,3,5,5+bad_stride,1,
                   b,b,0,5,7,7,1, out,out,bad_offset,3,7,7,1);
  if (r.aligned != out) return 1;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 7; ++j) {
      int32_t want = 0;
      for (int p = 0; p < 5; ++p) want += (int32_t)a[i*5+p] * b[p*7+j];
      if (c[i*7+j] != want) return 2;
    }
  return 0;
}
""",
        encoding="utf-8",
    )
    exe = tmp_path / "bridge"
    build = subprocess.run(
        [
            _CC,
            "-Wall",
            "-Wextra",
            "-Werror",
            str(tmp_path / "shim.c"),
            str(tmp_path / "kernel.c"),
            str(tmp_path / "driver.c"),
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    assert subprocess.run([str(exe)], capture_output=True).returncode == 0
    assert subprocess.run([str(exe), "bad_stride"], capture_output=True).returncode != 0
    assert subprocess.run([str(exe), "bad", "alias"], capture_output=True).returncode != 0
    assert subprocess.run([str(exe), "bad", "offset", "overflow"], capture_output=True).returncode != 0


# --------------------------------------------------------------- declines, reported not guessed


def test_a_sub_byte_format_is_declined_rather_than_guessed():
    """A sub-byte element offset is not a byte count; pointer arithmetic for it would be a guess."""
    unit = emit_translation_unit("gemmini", {"s": (16, 16, 32)}, {"s": ("mxfp4", "mxfp4", "f32")})
    assert unit.symbols == ()
    assert any("sub-byte" in why or "unknown element width" in why for _, why in unit.skipped)


def test_a_shape_this_emitter_cannot_express_is_declined():
    """Rank-2 and batched rank-3 both have an entry shape. Anything else is declined and reported --
    emitting an entry whose arity does not match the caller's descriptors would fail at the call, far
    from here."""
    unit = emit_translation_unit("gemmini", {"s": (2, 3, 16, 16, 32)}, {"s": ("i8", "i8", "i32")})
    assert unit.symbols == ()
    assert any("extents" in why for _, why in unit.skipped)


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
    unit = emit_translation_unit(
        "gemmini", {"s": (16, 32, 16)}, {"s": ("i8", "i8", "i32")}, kernel_symbol_for=lambda _s: "k", tile_edge=16
    )
    assert unit.symbols == ("s",)
    assert "static unsigned char" not in unit.text
    assert "merlin_span(a_aligned" in unit.text
    if _CC is not None:
        (tmp_path / "direct.c").write_text(unit.text, encoding="utf-8")
        built = subprocess.run(
            [_CC, "-Wall", "-Wextra", "-Werror", "-c", str(tmp_path / "direct.c"), "-o", str(tmp_path / "direct.o")],
            capture_output=True,
            text=True,
        )
        assert built.returncode == 0, built.stderr


def test_extents_off_the_tile_edge_are_staged_into_padded_buffers(tmp_path):
    """The kernel ABI states its operands are zero-padded to a multiple of the tile edge. Handing it
    raw buffers does not fault -- it strides by the padded width through unpadded data, reads a
    neighbouring row as its own, and returns plausible wrong numbers.

    Measured on a real model: every offloaded layer had M=8 against a 16-wide mesh, and the compiled
    artifact scored cos 0.9847 where the interpreted path scores 0.99993. With staging it scores
    0.999929 -- the padding was the entire gap."""
    unit = emit_translation_unit(
        "gemmini", {"s": (8, 344, 128)}, {"s": ("i8", "i8", "i32")}, kernel_symbol_for=lambda _s: "k", tile_edge=16
    )
    assert unit.symbols == ("s",)
    # M 8 -> 16, N 344 -> 352, K 128 already on the edge
    assert "s_a[16 * 128 * 1]" in unit.text
    assert "s_b[128 * 352 * 1]" in unit.text
    assert "s_c[16 * 352 * 4]" in unit.text


def test_a_padded_entry_compiles_and_round_trips(tmp_path):
    """Staging is real generated code with real index arithmetic; only building it proves it."""
    if _CC is None:
        pytest.skip("no C compiler available")
    unit = emit_translation_unit(
        "gemmini",
        {"merlin_dev_test_0": (8, 16, 32)},
        {"merlin_dev_test_0": ("i8", "i8", "i32")},
        kernel_symbol_for=lambda _s: "gemmini_kernel",
        tile_edge=16,
    )
    (tmp_path / "shim.c").write_text(unit.text, encoding="utf-8")
    p = subprocess.run(
        [_CC, "-Wall", "-Wextra", "-Werror", "-c", str(tmp_path / "shim.c"), "-o", str(tmp_path / "shim.o")],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr


def test_the_tile_edge_is_derived_from_the_device_not_assumed(monkeypatch):
    """Padding to a guessed edge is worse than not padding: it is differently wrong."""
    from merlin.llvmlower.device_shim import tile_edge_for
    from merlin.targetgen.rtl import facts

    # An explicit MERLIN_RTL_FACTS selection is not scoped by the requested name.
    # State the missing evidence this negative case tests without discarding the
    # caller's selection for the positive provider case.
    selected_body = facts.body_if_present
    monkeypatch.setattr(
        facts, "body_if_present", lambda target: {} if target == "definitely_not_a_target" else selected_body(target)
    )

    assert tile_edge_for("definitely_not_a_target") is None
    edge = tile_edge_for("gemmini")
    if edge is None:
        pytest.skip("no mesh facts derivable here")
    assert edge > 0


def test_rectangular_or_ambiguous_mesh_does_not_mint_a_square_shim_edge(monkeypatch):
    from merlin.llvmlower.device_shim import tile_edge_for
    from merlin.targetgen.rtl import facts

    monkeypatch.setattr(
        facts,
        "body_if_present",
        lambda _target: {
            "arrays": [{"rows": 16, "cols": 32}],
        },
    )
    assert tile_edge_for("example") is None
    monkeypatch.setattr(
        facts,
        "body_if_present",
        lambda _target: {
            "arrays": [{"rows": 16, "cols": 16}, {"rows": 32, "cols": 32}],
        },
    )
    assert tile_edge_for("example") is None


def test_an_underivable_edge_declines_rather_than_guessing(monkeypatch):
    from merlin.targetgen.rtl import facts

    monkeypatch.setattr(facts, "body_if_present", lambda _target: {})
    unit = emit_translation_unit("definitely_not_a_target", {"s": (8, 24, 8)}, {"s": ("i8", "i8", "i32")})
    assert unit.symbols == () or "s" not in unit.symbols


# --------------------------------------------------------------- batched contractions

_K3_STUB = """
#include <stdint.h>
static int calls = 0;
void {kernel}(void *weight, void *lhs_0, void *out_0) {{
  (void)weight; (void)lhs_0; ((int32_t *)out_0)[0] = ++calls;
}}
"""

_D3 = """
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef struct {{ void *a; void *b; intptr_t o; intptr_t s[3]; intptr_t st[3]; }} mr3;
extern mr3 {symbol}(void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,
                    void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,
                    void*,void*,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t,intptr_t);
int main(void) {{
  enum {{ B={b}, M={m}, N={n}, K={k} }};
  static int8_t A[B*M*K], W[B*K*N]; static int32_t C[B*M*N];
  memset(C, 0, sizeof C);
  mr3 r = {symbol}(A,A,0,B,M,K, M*K,K,1,  W,W,0,B,K,N, K*N,N,1,  C,C,0,B,M,N, M*N,N,1);
  int ok = (r.s[0]==B && r.s[1]==M && r.s[2]==N && r.b==(void*)C);
  for (int i = 0; i < B; ++i) ok = ok && (C[i*M*N] == i+1);
  printf("%d\\n", ok);
  return ok ? 0 : 1;
}}
"""


@pytest.mark.parametrize("sig,staged", [((3, 16, 16, 32), False), ((3, 8, 344, 128), True)])
def test_a_batched_signature_emits_one_entry(sig, staged):
    """A batch is a LOOP over disjoint slices, not a third tile axis, so it reuses the same kernel."""
    unit = emit_translation_unit(
        "gemmini", {"s": sig}, {"s": ("i8", "i8", "i32")}, kernel_symbol_for=lambda _s: "k", tile_edge=16
    )
    assert unit.symbols == ("s",), unit.skipped
    assert ("static unsigned char" in unit.text) is staged


@pytest.mark.skipif(_CC is None, reason="no C compiler available")
def test_each_batch_slice_gets_its_own_call_at_its_own_offset(tmp_path):
    """The failure this catches: a loop that recomputes the same slice, or writes every result to
    slice 0. Both produce a full-looking output tensor whose later slices are wrong."""
    b, m, n, k = 3, 8, 16, 32
    unit = emit_translation_unit(
        "gemmini",
        {"dev3": (b, m, n, k)},
        {"dev3": ("i8", "i8", "i32")},
        kernel_symbol_for=lambda _s: "gk",
        tile_edge=16,
    )
    (tmp_path / "shim.c").write_text(unit.text, encoding="utf-8")
    (tmp_path / "k.c").write_text(_K3_STUB.format(kernel="gk"), encoding="utf-8")
    (tmp_path / "d.c").write_text(_D3.format(symbol="dev3", b=b, m=m, n=n, k=k), encoding="utf-8")
    exe = tmp_path / "t"
    build = subprocess.run(
        [
            _CC,
            "-Wall",
            "-Wextra",
            "-Werror",
            str(tmp_path / "shim.c"),
            str(tmp_path / "k.c"),
            str(tmp_path / "d.c"),
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"batch loop wrong: {run.stdout} {run.stderr}"


@pytest.mark.skipif(_CC is None, reason="no C compiler available")
@pytest.mark.parametrize("rank,padded", [(2, False), (2, True), (3, False), (3, True)])
def test_mlir_c_interface_uses_result_and_operand_descriptor_pointers(tmp_path, monkeypatch, rank, padded):
    """Call the ABI emitted by llvm.emit_c_interface, not the shim's flat helper ABI.

    The lowered host function calls ``void _mlir_ciface_<name>(result*, A*, B*, C*)``.
    A native call checks the symbol, descriptor order, returned output ownership and
    actual matrix numerics through both direct and zero-padded batch/ordinary paths.
    """
    m = n = k = 3 if padded else 2
    batch = 2 if rank == 3 else 1
    monkeypatch.setattr(
        "merlin.llvmlower.device_shim.kernel_abi_for", lambda _device: KernelAbi(symbol="unused_kernel")
    )
    sym = f"selected_{rank}_{int(padded)}"
    kernel = f"kernel_{rank}_{int(padded)}"
    signature = (batch, m, n, k) if rank == 3 else (m, n, k)
    unit = emit_translation_unit(
        "gemmini",
        {sym: signature},
        {sym: ("i8", "i8", "i32")},
        kernel_symbol_for=lambda _sym: kernel,
        tile_edge=2,
    )
    assert unit.symbols == (sym,), unit.skipped
    (tmp_path / "shim.c").write_text(unit.text, encoding="utf-8")
    edge = 4 if padded else 2
    (tmp_path / "kernel.c").write_text(
        """
#include <stdint.h>
#define EDGE %EDGE%
void %KERNEL%(void *weight, void *lhs, void *out) {
  const int8_t *b = weight, *a = lhs;
  int32_t *c = out;
  for (int i = 0; i < EDGE; ++i)
    for (int j = 0; j < EDGE; ++j) {
      int32_t sum = 0;
      for (int p = 0; p < EDGE; ++p) sum += a[i*EDGE+p] * b[p*EDGE+j];
      c[i*EDGE+j] = sum;
    }
}
""".replace("%EDGE%", str(edge)).replace("%KERNEL%", kernel),
        encoding="utf-8",
    )
    (tmp_path / "driver.c").write_text(
        """
#include <stdint.h>
#define BATCH %BATCH%
#define M %M%
#define N %N%
#define K %K%
#define RANK %RANK%
#if RANK == 2
typedef struct { void *allocated, *aligned; intptr_t offset, sizes[2], strides[2]; } mr;
#else
typedef struct { void *allocated, *aligned; intptr_t offset, sizes[3], strides[3]; } mr;
#endif
extern void _mlir_ciface_%SYMBOL%(mr *, const mr *, const mr *, const mr *);
int main(void) {
  int8_t a[1+BATCH*M*K], b[1+BATCH*K*N];
  int32_t c[1+BATCH*M*N];
  for (int i = 0; i < 1+BATCH*M*K; ++i) a[i] = (int8_t)(i % 7 - 3);
  for (int i = 0; i < 1+BATCH*K*N; ++i) b[i] = (int8_t)(i % 5 - 2);
  for (int i = 0; i < 1+BATCH*M*N; ++i) c[i] = -999;
#if RANK == 2
  mr aa = {a, a, 1, {M,K}, {K,1}};
  mr bb = {b, b, 1, {K,N}, {N,1}};
  mr cc = {c, c, 1, {M,N}, {N,1}};
#else
  mr aa = {a, a, 1, {BATCH,M,K}, {M*K,K,1}};
  mr bb = {b, b, 1, {BATCH,K,N}, {K*N,N,1}};
  mr cc = {c, c, 1, {BATCH,M,N}, {M*N,N,1}};
#endif
  mr result = {0};
  _mlir_ciface_%SYMBOL%(&result, &aa, &bb, &cc);
  if (result.allocated != c || result.aligned != c || result.offset != 1) return 1;
  for (int axis = 0; axis < RANK; ++axis)
    if (result.sizes[axis] != cc.sizes[axis] || result.strides[axis] != cc.strides[axis]) return 5;
  for (int slice = 0; slice < BATCH; ++slice)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        int32_t want = 0;
        for (int p = 0; p < K; ++p)
          want += a[1+slice*M*K+i*K+p] * b[1+slice*K*N+p*N+j];
        if (c[1+slice*M*N+i*N+j] != want) return 2;
      }
  if (c[0] != -999) return 3;
  aa.sizes[0] += 1;
  result = cc;
  _mlir_ciface_%SYMBOL%(&result, &aa, &bb, &cc);
  if (result.allocated || result.aligned || result.sizes[0]) return 4;
  return 0;
}
""".replace("%BATCH%", str(batch))
        .replace("%M%", str(m))
        .replace("%N%", str(n))
        .replace("%K%", str(k))
        .replace("%RANK%", str(rank))
        .replace("%SYMBOL%", sym),
        encoding="utf-8",
    )
    exe = tmp_path / "ciface"
    built = subprocess.run(
        [
            _CC,
            "-Wall",
            "-Wextra",
            "-Werror",
            str(tmp_path / "shim.c"),
            str(tmp_path / "kernel.c"),
            str(tmp_path / "driver.c"),
            "-o",
            str(exe),
        ],
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"C-interface bridge failed: {run.stdout} {run.stderr}"
