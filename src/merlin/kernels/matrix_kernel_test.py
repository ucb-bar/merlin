"""``kernel-test``: check one matrix-unit kernel in seconds, on the unit's simulator, against the reference.

    python -m merlin.kernels.matrix_kernel_test [KERNEL] --support-target T --unit U --config C --board B
                                                [--quick | --full] [--list] [--timeout S]

KERNEL is a name the support package registers in ``kernels/kernels.yaml`` or a path to a C file
(``file.c`` or ``file.c:function``); omitted, the package's default kernel. What it does:

1. **Builds one bare-metal image** (Merlin's spike harness, no Zephyr, no model): the kernel exactly as a
   model build compiles it (same compiler, flags and generated header), Merlin's scalar reference
   (:func:`merlin.kernels.matrix_kernels.reference_source`), and a driver that runs a sweep of shapes.
2. **Audits** that the image contains the unit's instructions (a kernel that never touches the unit would
   pass every numerical check on the host core).
3. **Runs it on the simulator the package declares for its unit** (a spike extension model, or a simulator
   binary of its own) at the board's vector length.
4. **Judges every shape three ways**: the kernel's output against the in-image reference element by
   element (the first difference is reported as ``(slice, row, col)``, with both values), the hash of the
   kernel's output against one computed host-side with numpy, and the words around each output: C is
   pre-filled with a poison pattern, so an element the kernel never wrote and a write past the end of C
   are both reported as such.

The sweep is derived from the configuration's tile edge (``te``): partial M and N around one and two tiles
(1, te-1, te+1, 2te+1, ...), K tails (1..9, 15..17, 31..33, 63..65, 100, 129) and deep K, multi-tile shapes,
a bias, full-range operands, back-to-back calls on several slices (what a batched contraction does), and
any extra certification shapes supplied by the selected package. A shape outside the
kernel's declared constraints is skipped with the rule it breaks; one over the simulation budget is
skipped with that reason (``--full`` raises the budget).

Writes ``out/artifacts/probes/kernel-test/<support target>/<kernel>/`` (report.json, the image, its
sources and the generated header) and exits non-zero unless every shape it ran was bit-exact. No target,
unit or instruction is named here: they are the arguments and the package's own declarations.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import matrix_kernels as MK

__all__ = ["Shape", "build_image", "expected_digest", "main", "operands", "run", "sweep"]

#: Pre-fill of every output word; an element still holding it after the call was never written.
POISON = 0x5A5A5A5A
#: Words past each output slice that must still hold the poison after the call (an out-of-bounds write).
GUARD_WORDS = 16
#: Per-shape budget of multiply-accumulates (the in-image reference is a scalar loop on a functional sim).
QUICK_MACS = 2_000_000
DEFAULT_MACS = 24_000_000
FULL_MACS = 400_000_000

_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class Shape:
    """One call shape of the sweep, with why it is there."""

    name: str
    m: int
    n: int
    k: int
    bias: bool = False
    batch: int = 1
    amplitude: int = 8
    seed: int = 1
    why: str = ""

    @property
    def macs(self) -> int:
        return self.batch * self.m * self.n * self.k


# ---------------------------------------------------------------------------------------------------------
# Operands: a counter-based generator the image and numpy compute identically (nothing expected is embedded)
# ---------------------------------------------------------------------------------------------------------


def _splitmix64(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _draw(seed: int, stream: int, count: int, span: int, low: int) -> np.ndarray:
    idx = np.arange(count, dtype=np.uint64)
    base = np.uint64(((int(seed) & 0xFFFFFF) << 40) | ((int(stream) & 0xFF) << 32))
    z = _splitmix64(idx ^ base)
    return ((z >> np.uint64(32)) % np.uint64(span)).astype(np.int64) + int(low)


def operands(shape: Shape, slice_: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``(at (k, m), b (k, n), bias (n,) or None)`` for one slice, exactly as the image generates them."""
    seed = shape.seed + slice_
    amp = int(shape.amplitude)
    at = _draw(seed, 1, shape.k * shape.m, 2 * amp + 1, -amp).astype(np.int8).reshape(shape.k, shape.m)
    b = _draw(seed, 2, shape.k * shape.n, 2 * amp + 1, -amp).astype(np.int8).reshape(shape.k, shape.n)
    if shape.name == "signed_endpoints":
        # The normal amplitude-127 draw cannot produce -128. Put all four endpoint
        # products in the first 2x2 output block without changing the random body.
        at[0, :2] = (-128, 127)
        b[0, :2] = (-128, 127)
    bias = _draw(seed, 3, shape.n, 2001, -1000).astype(np.int32) if shape.bias else None
    return at, b, bias


def _reference(at: np.ndarray, b: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    """Reference the shared K-major int8 by int8 to int32 kernel ABI."""
    lhs = np.asarray(at, dtype=np.int64)
    rhs = np.asarray(b, dtype=np.int64)
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[0] != rhs.shape[0]:
        raise ValueError(f"expected K-major operands, got {lhs.shape} and {rhs.shape}")
    acc = lhs.T @ rhs
    if bias is not None:
        acc = acc + np.asarray(bias, dtype=np.int64)[None, :]
    return acc.astype(np.int32)


def _fnv1a64(data: bytes, h: int = _FNV64_OFFSET) -> int:
    for byte in data:
        h ^= byte
        h = (h * _FNV64_PRIME) & _MASK64
    return h


def expected_digest(shape: Shape) -> int:
    """FNV-1a 64 over every slice's int32 output, little-endian, in slice order -- what the image prints."""
    h = _FNV64_OFFSET
    for s in range(shape.batch):
        h = _fnv1a64(np.ascontiguousarray(_reference(*operands(shape, s)), dtype="<i4").tobytes(), h)
    return h


# ---------------------------------------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------------------------------------


def sweep(
    tile_edge: int,
    constraints: MK.Constraints,
    *,
    budget: int = DEFAULT_MACS,
    quick: bool = False,
    certification_cases: Sequence[Any] = (),
) -> tuple[list[Shape], list[tuple[Shape, str]]]:
    """``(shapes to run, [(shape, why skipped)])`` for a unit whose tile edge is ``tile_edge``."""
    te = int(tile_edge)
    cand: list[Shape] = [Shape("one_tile", te, te, 64, why="exactly one tile")]
    for m in (1, 3, te - 1, te + 1, 2 * te - 1, 2 * te + 1):
        cand.append(Shape(f"m{m}", m, te, 32, why="partial / multiple M tiles"))
    for n in (1, 3, te - 1, te + 1, 2 * te - 1, 2 * te + 1):
        cand.append(Shape(f"n{n}", te, n, 32, why="partial / multiple N tiles"))
    for k in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 129):
        cand.append(Shape(f"k{k}", te, te, k, why="K tail"))
    cand += [
        Shape("tiny", 1, 1, 1, why="the smallest call"),
        Shape("odd_small", 5, 7, 3, why="smaller than a tile in every extent"),
        Shape("edges", te + 1, te - 1, 33, why="partial M and N with a K tail"),
        Shape("multi", 2 * te, 3 * te, 64, why="several tiles each way"),
        Shape("multi_edges", 3 * te + 5, 2 * te + 7, 129, why="several tiles with partial edges and a K tail"),
        Shape("deep_k", te, te, 512, why="a long reduction"),
        Shape("deeper_k", 2 * te, te, 1000, why="a long reduction with a K tail"),
        Shape("full_range", te, te, 256, amplitude=127, why="largest operand magnitudes"),
        Shape("signed_endpoints", te, te, 33, amplitude=127, why="all four signed-int8 endpoint products and K tail"),
        Shape("bias", te, te, 64, bias=True, why="non-NULL bias"),
        Shape("bias_edges", te + 3, 2 * te - 1, 33, bias=True, why="bias with partial tiles"),
        Shape("bias_narrow", 1, te + 1, 7, bias=True, why="bias, one row"),
        Shape("batch3", te, te + 1, 17, batch=3, why="three back-to-back calls (a batched contraction)"),
        Shape("batch2_multi", 2 * te, 2 * te, 64, batch=2, why="back-to-back multi-tile calls"),
        Shape("batch4_small", te - 1, 5, 9, batch=4, why="back-to-back small calls"),
    ]
    if not quick:
        for case in certification_cases:
            cand.append(
                Shape(
                    f"corpus_{case.name}",
                    int(case.m),
                    int(case.n),
                    int(case.k),
                    bias=bool(case.bias),
                    why="package certification corpus: " + " ".join(str(case.why).split())[:80],
                )
            )
    run_: list[Shape] = []
    skipped: list[tuple[Shape, str]] = []
    seen: set[tuple] = set()
    for i, shape in enumerate(cand):
        key = (shape.m, shape.n, shape.k, shape.bias, shape.batch, shape.amplitude)
        if key in seen or min(shape.m, shape.n, shape.k) < 1:
            continue
        seen.add(key)
        shape = Shape(**{**asdict(shape), "seed": 1 + 8 * i})
        why = constraints.violations(shape.m, shape.n, shape.k, tile_edge=te)
        if shape.bias and not constraints.bias:
            why.append("the kernel declares bias: false")
        if why:
            skipped.append((shape, "outside the declared constraints: " + "; ".join(why)))
        elif shape.macs > budget:
            skipped.append((shape, f"{shape.macs:,} MACs is over this run's budget of {budget:,} (--full)"))
        else:
            run_.append(shape)
    return run_, skipped


# ---------------------------------------------------------------------------------------------------------
# The image
# ---------------------------------------------------------------------------------------------------------


def _round_up(v: int, to: int) -> int:
    return -(-int(v) // int(to)) * int(to)


def emit_driver(shapes: Sequence[Shape], *, kernel_func: str, ref_func: str, align: int) -> str:
    """The C driver: generate each shape's operands, poison C, call the kernel, check it every way."""
    if not shapes:
        raise ValueError("no shapes to run: an image that runs nothing would report a pass")
    words = max(1, align // 4)
    a_max = max(s.batch * _round_up(s.k * s.m, align) for s in shapes)
    b_max = max(s.batch * _round_up(s.k * s.n, align) for s in shapes)
    bias_max = max(s.batch * _round_up(s.n, words) for s in shapes)
    c_max = max(s.batch * _round_up(s.m * s.n + GUARD_WORDS, words) for s in shapes)
    rows = "\n".join(
        f'  {{"{s.name}", {s.m}u, {s.n}u, {s.k}u, '
        f"{int(s.bias)}, {s.batch}u, {s.amplitude}, {s.seed}u, {int(s.name == 'signed_endpoints')}}},"
        for s in shapes
    )
    return f"""\
/* GENERATED by merlin.kernels.matrix_kernel_test -- the kernel-test driver. Nothing expected is embedded:
 * operands are generated here and on the host by the same counter-based function. */
#include <stdint.h>
#include <stddef.h>
#include "htif.h"

void {kernel_func}(int32_t *, const int8_t *, const int8_t *, const int32_t *, size_t, size_t, size_t);
void {ref_func}(int32_t *, const int8_t *, const int8_t *, const int32_t *, size_t, size_t, size_t);

struct kt_shape {{ const char *name; size_t m, n, k; int bias; size_t batch; long amp; uint64_t seed; int endpoints; }};
static const struct kt_shape SHAPES[] = {{
{rows}
}};
#define N_SHAPES ((int)(sizeof(SHAPES) / sizeof(SHAPES[0])))
#define ALIGN {align}
#define POISON ((int32_t){POISON:#x})
#define GUARD_WORDS {GUARD_WORDS}

static int8_t A[{a_max}] __attribute__((aligned(ALIGN)));
static int8_t B[{b_max}] __attribute__((aligned(ALIGN)));
static int32_t BIAS[{bias_max}] __attribute__((aligned(ALIGN)));
static int32_t C_DEV[{c_max}] __attribute__((aligned(ALIGN)));
static int32_t C_REF[{c_max}] __attribute__((aligned(ALIGN)));

static uint64_t splitmix64(uint64_t z)
{{
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}}
static long draw(uint64_t seed, uint64_t stream, uint64_t i, uint64_t span, long low)
{{
  const uint64_t base = ((seed & 0xffffffull) << 40) | ((stream & 0xffull) << 32);
  return (long)((splitmix64(i ^ base) >> 32) % span) + low;
}}
static size_t round_up(size_t v, size_t to) {{ return (v + to - 1) / to * to; }}

static uint64_t fnv1a64(uint64_t h, const void *p, size_t n)
{{
  const uint8_t *bytes = (const uint8_t *)p;
  for (size_t i = 0; i < n; ++i) {{ h ^= (uint64_t)bytes[i]; h *= {_FNV64_PRIME:#x}ull; }}
  return h;
}}
static void put_hex64(uint64_t v)
{{
  static const char digits[] = "0123456789abcdef";
  htif_puts("0x");
  for (int s = 60; s >= 0; s -= 4) htif_putc(digits[(v >> s) & 0xf]);
}}
static void put_u(unsigned long v) {{ htif_putd((long)v); }}

int main(long hart)
{{
  if (hart != 0)
    for (;;) asm volatile("wfi");
  console_init();
  for (int si = 0; si < N_SHAPES; ++si) {{
    const struct kt_shape *s = &SHAPES[si];
    const size_t as = round_up(s->k * s->m, ALIGN), bs = round_up(s->k * s->n, ALIGN);
    const size_t ns = round_up(s->n, ALIGN / 4), cs = round_up(s->m * s->n + GUARD_WORDS, ALIGN / 4);
    htif_puts("KT BEGIN "); htif_puts(s->name); htif_putc('\\n');
    for (size_t sl = 0; sl < s->batch; ++sl) {{
      const uint64_t seed = s->seed + sl;
      for (size_t i = 0; i < s->k * s->m; ++i) A[sl * as + i] = (int8_t)draw(seed, 1, i, 2 * s->amp + 1, -s->amp);
      for (size_t i = 0; i < s->k * s->n; ++i) B[sl * bs + i] = (int8_t)draw(seed, 2, i, 2 * s->amp + 1, -s->amp);
      if (s->endpoints) {{
        A[sl * as] = -128; A[sl * as + 1] = 127;
        B[sl * bs] = -128; B[sl * bs + 1] = 127;
      }}
      if (s->bias)
        for (size_t i = 0; i < s->n; ++i) BIAS[sl * ns + i] = (int32_t)draw(seed, 3, i, 2001, -1000);
    }}
    for (size_t i = 0; i < s->batch * cs; ++i) {{ C_DEV[i] = POISON; C_REF[i] = 0; }}
    asm volatile("fence rw, rw" ::: "memory");

    uint64_t cyc0, cyc1;
    asm volatile("csrr %0, mcycle" : "=r"(cyc0));
    for (size_t sl = 0; sl < s->batch; ++sl)
      {kernel_func}(C_DEV + sl * cs, A + sl * as, B + sl * bs, s->bias ? BIAS + sl * ns : (const int32_t *)0,
                    s->m, s->n, s->k);
    asm volatile("fence rw, rw" ::: "memory");
    asm volatile("csrr %0, mcycle" : "=r"(cyc1));

    long guard_bad = 0;
    for (size_t sl = 0; sl < s->batch; ++sl)
      for (size_t i = s->m * s->n; i < cs; ++i)
        if (C_DEV[sl * cs + i] != POISON) ++guard_bad;
    for (size_t sl = 0; sl < s->batch; ++sl)
      {ref_func}(C_REF + sl * cs, A + sl * as, B + sl * bs, s->bias ? BIAS + sl * ns : (const int32_t *)0,
                 s->m, s->n, s->k);
    long mism = 0, first_slice = -1, first = -1, unwritten = 0;
    int32_t got = 0, want = 0;
    uint64_t h = {_FNV64_OFFSET:#x}ull;
    for (size_t sl = 0; sl < s->batch; ++sl) {{
      for (size_t i = 0; i < s->m * s->n; ++i) {{
        const int32_t d = C_DEV[sl * cs + i], r = C_REF[sl * cs + i];
        if (d != r) {{
          if (first < 0) {{ first_slice = (long)sl; first = (long)i; got = d; want = r; }}
          ++mism;
          if (d == POISON) ++unwritten;
        }}
      }}
      h = fnv1a64(h, C_DEV + sl * cs, s->m * s->n * sizeof(int32_t));
    }}
    htif_puts("KT CASE "); htif_puts(s->name);
    htif_putc(' '); htif_putd(mism); htif_putc(' '); htif_putd(unwritten);
    htif_putc(' '); htif_putd(first_slice); htif_putc(' '); htif_putd(first);
    htif_putc(' '); htif_putd((long)got); htif_putc(' '); htif_putd((long)want);
    htif_putc(' '); htif_putd(guard_bad); htif_putc(' '); put_hex64(h);
    htif_putc(' '); put_u((unsigned long)(cyc1 - cyc0)); htif_putc('\\n');
  }}
  htif_puts("KT DONE\\n");
  htif_exit(0);
  return 0;
}}
"""


@dataclass
class Image:
    elf: Path
    work: Path
    kernel_function: str
    files: dict[str, str]


def build_image(
    kernel_source: str,
    kernel_function: str,
    shapes: Sequence[Shape],
    work: Path,
    *,
    vlen: int | None,
    align: int,
) -> Image:
    """Compile the kernel as a model build does (clang, the RVV flags pinned to the board's VLEN), and the
    driver and reference with the bare-metal spike harness; link one ELF."""
    from ..llvmlower import toolchain
    from ..runtime.backends import spike as spike_backend
    from ..runtime.backends import zephyr_model as zm

    work.mkdir(parents=True, exist_ok=True)
    (work / "kernel_tu.c").write_text(kernel_source, encoding="utf-8")
    (work / "reference.c").write_text(MK.reference_source(MK.REFERENCE_FUNCTION), encoding="utf-8")
    (work / "driver.c").write_text(
        emit_driver(shapes, kernel_func=kernel_function, ref_func=MK.REFERENCE_FUNCTION, align=align), encoding="utf-8"
    )
    kcmd = [
        str(toolchain.clang()),
        "--target=riscv64-unknown-elf",
        *zm.march_with_vlen(list(zm.RVV_CFLAGS), vlen),
        "-c",
        str(work / "kernel_tu.c"),
        "-o",
        str(work / "kernel.o"),
    ]
    got = subprocess.run(kcmd, capture_output=True, text=True, cwd=work)
    if got.returncode != 0:
        raise KernelBuildError("the kernel did not compile", kcmd, got.stderr)
    harness = spike_backend.harness_dir()
    elf = work / "kernel_test.elf"
    lcmd = [
        str(spike_backend.gcc_path()),
        "-march=rv64gcv",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-fno-tree-vectorize",
        "-ffreestanding",
        "-nostdlib",
        "-nostartfiles",
        "-I",
        str(harness),
        "-T",
        str(harness / "link.ld"),
        *(str(harness / f) for f in ("crt.S", "htif.c", "libc_min.c")),
        str(work / "driver.c"),
        str(work / "reference.c"),
        str(work / "kernel.o"),
        "-o",
        str(elf),
    ]
    got = subprocess.run(lcmd, capture_output=True, text=True, cwd=work)
    if got.returncode != 0:
        raise KernelBuildError("the kernel-test image did not link", lcmd, got.stderr)
    return Image(elf=elf, work=work, kernel_function=kernel_function, files={})


class KernelBuildError(RuntimeError):
    def __init__(self, what: str, cmd: Sequence[str], stderr: str) -> None:
        self.what, self.cmd, self.stderr = what, list(cmd), stderr
        super().__init__(f"{what}:\n$ {' '.join(cmd)}\n{stderr[-4000:]}")


# ---------------------------------------------------------------------------------------------------------
# The simulator that implements the unit
# ---------------------------------------------------------------------------------------------------------


@dataclass
class UnitSimulator:
    spike: Path
    isa: str
    flags: tuple[str, ...]
    env: dict[str, str] | None
    describe: str


def unit_simulator(provider: Any, *, support_target: str, unit: str, config: str, vlen: int | None) -> UnitSimulator:
    """The simulator the package declares for its unit -- the same resolution the shipped-image leg uses
    (``device_simulator`` in the plugin, else ``runner.spike_extension`` in the contract). Raises with the
    reason when none is declared or it is not usable; never falls back to a simulator without the unit."""
    from ..runtime.backends import spike as spike_backend
    from ..runtime.backends import zephyr_model as zm
    from ..targetgen import spike_extension as SE

    declare = getattr(provider, "device_simulator", None)
    if callable(declare):
        sim = dict(declare(unit=unit, config=config) or {})
        if not sim.get("available"):
            raise RuntimeError(f"the unit's simulator is not usable: {sim.get('reason')}")
        isa = zm.spike_isa(vlen) + "".join(f"_{e}" for e in sim.get("isa_extensions") or ())
        return UnitSimulator(Path(sim["binary"]), isa, (), None, f"{sim['binary']} --isa={isa}")
    if SE.declared_extension(support_target) is None:
        raise RuntimeError(
            f"{support_target} declares no simulator for {unit} (no device_simulator in its plugin, no "
            "runner.spike_extension in its contract), so a kernel for it cannot be run here"
        )
    ext = SE.resolve(support_target, default_library_dir=spike_backend.spike_path().parent, default_extension_name="")
    flags, env = SE.flags_and_env(ext)
    isa = zm.spike_isa(vlen)
    return UnitSimulator(
        spike_backend.spike_path(),
        isa,
        tuple(flags),
        env,
        f"spike --isa={isa} with the unit model {ext.extension_name!r} ({ext.extlib}, sha256 {ext.sha256[:16]})",
    )


# ---------------------------------------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------------------------------------


def parse(console: str) -> dict[str, Any]:
    began: list[str] = []
    cases: dict[str, dict[str, Any]] = {}
    malformed: list[str] = []
    done = False
    for raw in console.splitlines():
        parts = raw.strip().split()
        if len(parts) >= 3 and parts[:2] == ["KT", "BEGIN"]:
            began.append(parts[2])
        elif len(parts) >= 2 and parts[:2] == ["KT", "CASE"]:
            if len(parts) != 12:
                malformed.append(raw)
                continue
            try:
                vals = [int(v) for v in parts[3:10]]
                cases[parts[2]] = {
                    "mismatches": vals[0],
                    "unwritten": vals[1],
                    "first_slice": vals[2],
                    "first_index": vals[3],
                    "got": vals[4],
                    "want": vals[5],
                    "guard_bad": vals[6],
                    "digest": int(parts[10], 16),
                    "cycles": int(parts[11]),
                }
            except ValueError:
                malformed.append(raw)
        elif parts == ["KT", "DONE"]:
            done = True
    return {"began": began, "cases": cases, "done": done, "malformed": malformed}


def judge(shapes: Sequence[Shape], parsed: Mapping[str, Any], console: str) -> list[dict[str, Any]]:
    rows = []
    trapped_in = None
    if not parsed["done"] and parsed["began"]:
        last = parsed["began"][-1]
        if last not in parsed["cases"]:
            trapped_in = last
    for shape in shapes:
        row: dict[str, Any] = {"shape": asdict(shape)}
        got = parsed["cases"].get(shape.name)
        if got is None:
            if shape.name == trapped_in:
                tail = " ".join(console.strip().splitlines()[-3:])[-300:]
                row.update(ok=False, verdict=f"TRAP or hang inside the kernel ({tail})")
            else:
                row.update(ok=False, verdict="not run (the image stopped before this shape)")
            rows.append(row)
            continue
        row["cycles"] = got["cycles"]
        problems = []
        if got["mismatches"]:
            idx = got["first_index"]
            i, j = divmod(idx, shape.n)
            where = f"slice {got['first_slice']} " if shape.batch > 1 else ""
            first = (
                f"first at {where}C[{i}][{j}]: got {got['got']}"
                + (" (never written)" if got["got"] == POISON else "")
                + f", reference {got['want']}"
            )
            total = shape.batch * shape.m * shape.n
            problems.append(
                f"{got['mismatches']}/{total} elements differ"
                + (f" ({got['unwritten']} never written)" if got["unwritten"] else "")
                + f"; {first}"
            )
        if got["guard_bad"]:
            problems.append(f"wrote {got['guard_bad']} word(s) past the end of C")
        expect = expected_digest(shape)
        if not got["mismatches"] and got["digest"] != expect:
            problems.append("output hash differs from the host's numpy reference (the in-image reference agrees)")
        row.update(ok=not problems, verdict="ok" if not problems else "FAIL: " + "; ".join(problems))
        rows.append(row)
    return rows


def table(rows: Sequence[Mapping[str, Any]], skipped: Sequence[tuple[Shape, str]]) -> str:
    head = f"{'shape':<22} {'M':>5} {'N':>5} {'K':>5} {'bias':>4} {'calls':>5} {'cycles':>10}  result"
    out = [head, "-" * len(head)]
    for r in rows:
        s = r["shape"]
        out.append(
            f"{s['name']:<22} {s['m']:>5} {s['n']:>5} {s['k']:>5} {'yes' if s['bias'] else '':>4} "
            f"{s['batch']:>5} {r.get('cycles', ''):>10}  {r['verdict']}"
        )
    for shape, why in skipped:
        out.append(
            f"{shape.name:<22} {shape.m:>5} {shape.n:>5} {shape.k:>5} {'yes' if shape.bias else '':>4} "
            f"{shape.batch:>5} {'':>10}  skipped: {why}"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------------------------------------


def run(
    spec: str | None,
    *,
    support_target: str,
    unit: str,
    config: str,
    board: str,
    budget: int = DEFAULT_MACS,
    quick: bool = False,
    timeout: int = 900,
    out: Path | None = None,
) -> dict[str, Any]:
    """Build, audit, run and judge one kernel; returns the report (also written to ``out``)."""
    from ..common.paths import artifacts_dir
    from ..runtime.backends import zephyr_model as zm
    from ..runtime.boards import board as board_desc

    routing = zm.MatrixRouting(unit=unit, config=config, support_target=support_target, kernel=spec)
    provider = routing.provider()
    selected = MK.for_routing(routing)
    tile_edge, align = provider.geometry(unit=unit, config=config)
    vlen = board_desc(board).vlen
    label = "".join(c if c.isalnum() or c in "-_." else "_" for c in selected.name)
    out = out or artifacts_dir() / "probes" / "kernel-test" / support_target / label
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    corpus = getattr(provider, "certification_cases", None)
    certification_cases = corpus(tile_edge) if callable(corpus) and not quick else ()
    shapes, skipped = sweep(
        tile_edge,
        selected.entry.constraints,
        budget=budget,
        quick=quick,
        certification_cases=certification_cases,
    )
    header_name = selected.header or f"{unit}.h"
    header = MK.unit_header(
        provider, unit=unit, config=config, header=header_name, prefix=selected.header_prefix or "MATRIX"
    )
    (out / header_name).write_text(header, encoding="utf-8")
    if selected.generated:
        gen = getattr(provider, "generated_kernel", None)
        if not callable(gen):
            raise RuntimeError(
                f"{support_target}'s plugin exposes no generated_kernel(), so its default cannot be tested"
            )
        source, function = gen(unit=unit, config=config, max_k=max((s.k for s in shapes), default=1))
        files: dict[str, str] = {}
    else:
        prepared = MK.prepare(selected, out, header_text=header, scalar_tile=False)
        source, function, files = prepared.source, prepared.function, prepared.files
    report: dict[str, Any] = {
        "schema": "merlin.kernel_test.v1",
        "kernel": selected.record(),
        "support_target": support_target,
        "unit": unit,
        "config": config,
        "board": board,
        "vlen": vlen,
        "tile_edge": int(tile_edge),
        "alignment_bytes": int(align),
        "snapshot": files,
        "header": str(out / header_name),
    }
    started = time.monotonic()
    try:
        image = build_image(source, function, shapes, out, vlen=vlen, align=max(int(align), 64))
    except KernelBuildError as exc:
        report.update(status="build_failed", error=str(exc))
        _write(out, report)
        return report
    report["elf"] = str(image.elf)

    from .decode import opu as unit_audit

    derived = provider.derive_encodings(provider.load_contract(unit))
    counts = {
        k: int(v) for k, v in sorted((unit_audit.audit_object(image.elf, derived.encodings).counts or {}).items()) if v
    }
    report["unit_instruction_counts"] = counts
    if not counts:
        report.update(
            status="no_unit_instructions",
            error=f"the image contains none of {unit}'s instructions: the kernel computes on the host core and "
            "would prove nothing about the unit",
        )
        _write(out, report)
        return report

    try:
        sim = unit_simulator(provider, support_target=support_target, unit=unit, config=config, vlen=vlen)
    except Exception as exc:  # noqa: BLE001 -- reported, never substituted
        report.update(status="not_run", error=f"{type(exc).__name__}: {exc}")
        _write(out, report)
        return report
    report["simulator"] = sim.describe
    cmd = [str(sim.spike), f"--isa={sim.isa}", "-p1", *sim.flags, str(image.elf)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=sim.env)
        console = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        console = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        console += f"\n[kernel-test] the simulation exceeded {timeout} s"
    (out / "console.txt").write_text(console, encoding="utf-8")
    report["run_seconds"] = round(time.monotonic() - started, 1)
    parsed = parse(console)
    rows = judge(shapes, parsed, console)
    report["rows"] = rows
    report["skipped"] = [{"shape": asdict(s), "why": why} for s, why in skipped]
    passed = sum(1 for r in rows if r["ok"])
    report["passed"], report["ran"] = passed, len(rows)
    report["status"] = "pass" if rows and passed == len(rows) and parsed["done"] else "fail"
    if parsed["malformed"]:
        report["malformed_console_lines"] = parsed["malformed"]
    _write(out, report)
    return report


def _write(out: Path, report: Mapping[str, Any]) -> None:
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def _print_report(report: Mapping[str, Any]) -> None:
    k = report["kernel"]
    what = "generated by the package" if k.get("generated") else f"{k.get('file')} (sha256 {str(k.get('sha256'))[:16]})"
    print(f"\n== kernel-test: {k['name']} -- {what}")
    print(
        f"   unit {report['unit']} ({report['config']}, tile edge {report['tile_edge']}) on board {report['board']} "
        f"(VLEN {report['vlen']})"
    )
    if report.get("unit_instruction_counts"):
        print(f"   unit instructions in the image: {report['unit_instruction_counts']}")
    if report.get("simulator"):
        print(f"   simulator: {report['simulator']}")
    print(f"   generated header: {report.get('header')}")
    if report.get("rows"):
        print()
        print(table(report["rows"], [(Shape(**s["shape"]), s["why"]) for s in report.get("skipped") or []]))
    print()
    if report["status"] == "pass":
        print(
            f"PASS: {report['passed']}/{report['ran']} shapes bit-exact against Merlin's reference "
            f"({len(report.get('skipped') or [])} skipped, {report.get('run_seconds')} s)"
        )
    else:
        print(
            f"FAIL ({report['status']}): "
            + (report.get("error") or f"{report.get('passed')}/{report.get('ran')} shapes passed")
        )
    print(f"   report: {Path(str(report.get('header'))).parent / 'report.json'}")


def _list(support_target: str, unit: str) -> int:
    root = MK.package_root(support_target)
    reg = MK.load_registry(root)
    if reg is None:
        print(f"{support_target} ({root}) registers no kernels ({MK.REGISTRY_FILE} absent)")
        return 1
    print(f"kernels for {unit} in {reg.path} (header {reg.header}, macro prefix {reg.header_prefix}):")
    for e in reg.for_unit(unit):
        tag = " [default]" if reg.defaults.get(unit) == e.name else ""
        src = "generated" if e.generated else f"{Path(str(e.file)).name}:{e.function}"
        cons = "any shape" if e.constraints.any_shape else json.dumps(e.constraints.to_dict())
        print(f"  {e.name}{tag}: {src}; {cons}")
        if e.description:
            print(f"      {e.description}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "kernel", nargs="?", help="registered kernel name or path/to/file.c[:function] (default: the package default)"
    )
    ap.add_argument("--support-target", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--board", required=True, help="the board whose VLEN the kernel is compiled and simulated at")
    ap.add_argument("--quick", action="store_true", help="the core sweep only, small shapes (seconds)")
    ap.add_argument("--full", action="store_true", help="also the large corpus shapes (minutes)")
    ap.add_argument("--timeout", type=int, default=1800, help="simulation limit, s")
    ap.add_argument("--list", action="store_true", help="list the package's registered kernels and exit")
    a = ap.parse_args(argv)
    if a.list:
        return _list(a.support_target, a.unit)
    budget = QUICK_MACS if a.quick else FULL_MACS if a.full else DEFAULT_MACS
    try:
        report = run(
            a.kernel,
            support_target=a.support_target,
            unit=a.unit,
            config=a.config,
            board=a.board,
            budget=budget,
            quick=a.quick,
            timeout=a.timeout,
        )
    except (MK.KernelSelectionError, MK.KernelConstraintError) as exc:
        print(f"kernel-test: {exc}", file=sys.stderr)
        return 2
    _print_report(report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
