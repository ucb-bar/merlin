"""Certify the matrix-extension microkernel against the frozen corpus, on the hardware's own RTL.

This is the step that turns the kernel from "compiles and audits clean" into "computes the right numbers
on this unit". Everything before it reads the instruction stream; nothing before it ran the datapath.

**Three independent implementations, and the answer key is not in the image.** A case is certified only
when the device kernel agrees with a scalar C reference *inside* the image AND the hash of its output
matches one computed host-side by numpy. The in-image comparison is what localises a failure (it knows
which element and can say so); the host-side hash is what makes the verdict trustworthy, because the
in-image reference could itself be wrong and two agreeing implementations in one image would still look
like a pass. Nothing expected is embedded: an image carrying its own goldens can pass by copying them,
and that failure mode is indistinguishable from success in the console output.

**The geometry is derived, then cross-checked against the running hardware.** The host derives the vector
length from the config's own Scala declaration (:func:`opu_isa.vector_unit_params`) and turns it into the
logical tile edge, which decides which corpus cases are in reach and how large the operand buffers are.
The image then reports the edge the hardware actually gives it, and a mismatch is a hard failure rather
than a warning -- a corpus selected for a 32-lane tile and run on a 16-lane one would test shapes it does
not claim to, and would do it quietly.

**The scalar build is the pre-flight.** Compiled with ``OPU_SCALAR_TILE``, the same image runs the same
corpus with a scalar stand-in for the unit, on spike, in seconds. It proves the *plumbing* -- operand
embedding, layout, comparison, hashing, console framing -- so that a failure on the RTL run is
attributable to the datapath instead of to the harness. It also must contain none of the unit's
instructions, which is checked, so it cannot be mistaken for a device run.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import opu_corpus
from .opu_kernel import KernelSpec, emit_microkernel, emit_reference_c

__all__ = ["CaseResult", "CertReport", "IMAGE_MAIN", "REPORT_VERSION", "build_image", "certify",
           "emit_image_c", "expected_digests", "fnv1a64", "logical_tile_edge",
           "operand_alignment_for_config", "parse_console", "provenance_stamp", "tile_edge_for_config", "verdict"]

#: Bumped when the report's shape changes, so an old artifact is never read as a new one.
REPORT_VERSION = 1

#: FNV-1a, 64-bit. A sum or an XOR would let two compensating errors cancel; this will not. It is used
#: only to carry an exact integer result across a console cheaply -- the comparison it stands in for is
#: exact equality, so a hash collision is the single way a wrong result could pass, and at 2**-64 per case
#: that is not the risk worth engineering around.
_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1

#: The generated image's file name, and the console tokens it frames results with.
IMAGE_MAIN = "opu_corpus_main.c"
_TILE_LINE = "TILE"
_CASE_LINE = "CASE"
_DONE_LINE = "DONE"
#: Per-case device-kernel cycle count. Its own line so an older reader ignores it.
_CYCLES_LINE = "CYCLES"
_PROV_LINE = "PROV"

#: Element width the kernel runs its operands at. Not a target fact -- it is what an int8 kernel *is*,
#: and it is the SEW the emitted kernel configures.
_OPERAND_SEW_BITS = 8


def fnv1a64(data: bytes) -> int:
    """FNV-1a over raw bytes. Mirrored byte-for-byte by ``_fnv1a64`` in the emitted image."""
    h = _FNV64_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV64_PRIME) & _MASK64
    return h


def logical_tile_edge(vlen_bits: int, *, sew_bits: int = _OPERAND_SEW_BITS, lmul: int = 1) -> int:
    """The unit's logical tile edge: ``VLMAX`` for the element width the kernel runs at.

    This is the RVV definition (``VLEN * LMUL / SEW``) rather than a fact about any one unit, which is
    why it is computed instead of tabulated -- and the image re-derives the same number from the hardware
    at run time, so a wrong vector length here is caught rather than assumed.
    """
    if vlen_bits <= 0 or sew_bits <= 0 or lmul <= 0:
        raise ValueError(f"vlen={vlen_bits} sew={sew_bits} lmul={lmul} must all be positive")
    return (vlen_bits * lmul) // sew_bits


def _joined_text(paths: "str | Path | Sequence[str | Path]") -> str:
    """The concatenated text of one or several Scala sources.

    Several, because a unit's configurations do not all live in one file: the extension's own generator
    declares the standalone ones, while an integrating SoC declares the heterogeneous ones that put the
    unit on one tile beside something else. Both are legitimate places to look for a named config, and
    which one a given config lives in is not something this can know -- so the caller names every place
    and this searches all of them. Concatenating is sound because a lookup is by CLASS NAME, which is
    unique across the files by Scala's own rules.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return "\n".join(Path(p).read_text(encoding="utf-8") for p in paths)


def operand_alignment_for_config(config: str, *,
                                 config_scala: "str | Path | Sequence[str | Path]",
                                 mixin_scala: Sequence["str | Path"]) -> int:
    """Byte alignment the unit's operand panels require, derived from the datapath width.

    The vector load moves ``dLen`` bits per beat, so a panel that does not start on a ``dLen / 8`` byte
    boundary has its beats split across boundaries. MEASURED on this unit's RTL, that is not merely slower:
    the SECOND beat of an operand load comes back wrong, so a contraction whose operand base was 8 bytes
    off a 16-byte boundary produced correct values in columns 0..15 and garbage from column 16 on. Because
    the address depends on whatever else the image contains, the same shape with the same data passed or
    failed depending on which other cases were compiled beside it -- which is what made this look like
    non-determinism rather than an alignment rule.

    This is a constraint on the unit's CALLERS, so a packing pass that produces operand panels has to
    honour it, and the alignment is part of what packing costs.
    """
    cfg_text = _joined_text(config_scala)
    mixin_text = "\n".join(Path(p).read_text(encoding="utf-8") for p in mixin_scala)
    from ..targetgen.rtl import opu_isa
    params = opu_isa.vector_unit_params(cfg_text, config, mixin_text=mixin_text)
    for key in ("dLen", "dlen", "DLen"):
        if key in params:
            return max(1, int(params[key]) // 8)
    raise ValueError(
        f"could not derive a datapath width for {config!r} from {config_scala} (bound: {params}); "
        "refusing to guess an operand alignment, because an under-aligned panel returns wrong data "
        "rather than failing")


def tile_edge_for_config(config: str, *,
                         config_scala: "str | Path | Sequence[str | Path]",
                         mixin_scala: Sequence["str | Path"],
                         sew_bits: int = _OPERAND_SEW_BITS) -> int:
    """The tile edge for a named hardware config, read from that config's own declaration.

    Raises when the vector length cannot be grounded. That is deliberate: a defaulted vector length gives
    a plausible tile edge, a corpus selected against the wrong geometry, and a certification report that
    claims shapes it never ran.
    """
    cfg_text = _joined_text(config_scala)
    mixin_text = "\n".join(Path(p).read_text(encoding="utf-8") for p in mixin_scala)
    from ..targetgen.rtl import opu_isa
    params = opu_isa.vector_unit_params(cfg_text, config, mixin_text=mixin_text)
    for key in ("vLen", "vlen", "VLen"):
        if key in params:
            return logical_tile_edge(int(params[key]), sew_bits=sew_bits)
    raise ValueError(
        f"could not derive a vector length for {config!r} from {config_scala} (bound: {params}); "
        "refusing to guess one, because the tile edge decides which corpus cases are in reach")


# ---------------------------------------------------------------------------------------------
# Host-side truth
# ---------------------------------------------------------------------------------------------


def expected_digests(cases: Sequence[opu_corpus.Case]) -> dict[str, dict[str, Any]]:
    """``{case: {digest, shape, ...}}`` — the numpy-computed truth each case is judged against.

    The digest is over the int32 result in little-endian order, which is the byte order the image hashes
    its own output buffer in. Nothing here is written into the image.
    """
    out: dict[str, dict[str, Any]] = {}
    for case in cases:
        lhs, rhs, bias = case.operands()
        ref = opu_corpus.reference(lhs, rhs, bias)
        payload = np.ascontiguousarray(ref, dtype="<i4").tobytes()
        out[case.name] = {
            "digest": fnv1a64(payload),
            "m": int(case.m), "n": int(case.n), "k": int(case.k),
            "bias": bool(case.bias), "elements": int(ref.size),
        }
    return out


# ---------------------------------------------------------------------------------------------
# The image
# ---------------------------------------------------------------------------------------------


def _c_array(name: str, values: np.ndarray, ctype: str, per_line: int = 24,
             align: int | None = None) -> str:
    flat = np.asarray(values).reshape(-1)
    body = []
    for start in range(0, flat.size, per_line):
        body.append("  " + ", ".join(str(int(v)) for v in flat[start:start + per_line]) + ",")
    attr = f" __attribute__((aligned({int(align)})))" if align else ""
    return (f"static const {ctype} {name}[{flat.size}]{attr} = {{\n"
            + "\n".join(body) + "\n};\n")


def _ident(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)


def emit_image_c(cases: Sequence[opu_corpus.Case], *, kernel_func: str = "opu_gemm_i8",
                 ref_func: str = "opu_gemm_i8_ref", operand_align: int | None = None,
                 provenance_stamp: str | None = None) -> str:
    """The C source of the corpus image: operands as data, both implementations called, results framed.

    The operands are embedded rather than generated on the device because the corpus draws them from
    numpy's generator, which cannot be reproduced in C -- and embedding them means the bytes the image
    computes on are provably the same ones the host derived its expected digests from, instead of two
    generators that are supposed to agree.
    """
    if not cases:
        raise ValueError("refusing to emit an image with no cases; an image that runs nothing "
                         "reports DONE and would read as a pass")
    blocks, table = [], []
    max_out = 0
    for case in cases:
        lhs, rhs, bias = case.operands()
        tag = _ident(case.name)
        blocks.append(_c_array(f"at_{tag}", lhs, "int8_t", align=operand_align))
        blocks.append(_c_array(f"b_{tag}", rhs, "int8_t", align=operand_align))
        bias_expr = "0"
        if bias is not None:
            blocks.append(_c_array(f"bias_{tag}", bias, "int32_t", align=operand_align))
            bias_expr = f"bias_{tag}"
        max_out = max(max_out, int(case.m) * int(case.n))
        table.append(f'  {{ "{case.name}", at_{tag}, b_{tag}, {bias_expr}, '
                     f'{int(case.m)}, {int(case.n)}, {int(case.k)} }},')

    return f"""\
/* GENERATED by merlin.kernels.opu_cert — do not edit.
 *
 * Runs the frozen corpus on the unit and reports, per case, how many elements disagreed with an
 * in-image scalar reference and a hash of the kernel's own output. The EXPECTED values are deliberately
 * absent: an image that carries its goldens can pass by copying them.
 */
#include <stdint.h>
#include <stddef.h>
#include "htif.h"
{('#define PROV_STAMP "' + provenance_stamp + '"' + chr(10)) if provenance_stamp else ""}
void {kernel_func}(int32_t *, const int8_t *, const int8_t *, const int32_t *, size_t, size_t, size_t);
void {ref_func}(int32_t *, const int8_t *, const int8_t *, const int32_t *, size_t, size_t, size_t);

{"".join(blocks)}
struct opu_case {{
  const char *name;
  const int8_t *at;
  const int8_t *b;
  const int32_t *bias;
  size_t m, n, k;
}};

static const struct opu_case CASES[] = {{
{chr(10).join(table)}
}};
#define N_CASES ((int)(sizeof(CASES) / sizeof(CASES[0])))

/* Sized to the largest case the host selected, so no case can overrun and none of this is dynamic. */
static int32_t c_dev[{max_out}];
static int32_t c_ref[{max_out}];

/* Mirrors kernels.opu_cert.fnv1a64 byte for byte. */
static uint64_t fnv1a64(const void *p, size_t n)
{{
  const uint8_t *bytes = (const uint8_t *)p;
  uint64_t h = {_FNV64_OFFSET:#x}ULL;
  for (size_t i = 0; i < n; ++i) {{
    h ^= (uint64_t)bytes[i];
    h *= {_FNV64_PRIME:#x}ULL;
  }}
  return h;
}}

/* The console ABI has no unsigned/hex printer, and a 64-bit digest through a signed decimal one comes
 * back negative half the time. */
static void put_hex64(uint64_t v)
{{
  static const char digits[] = "0123456789abcdef";
  htif_puts("0x");
  for (int shift = 60; shift >= 0; shift -= 4)
    htif_putc(digits[(v >> shift) & 0xf]);
}}

int main(long hart)
{{
  /* Every hart enters main; only one runs the corpus. A second hart writing the shared result buffers
   * would corrupt the comparison in a way that looks like a datapath error. */
  if (hart != 0)
    for (;;)
      asm volatile("wfi");

  console_init();

  /* The image declares what it was generated from, and prints it, so a binary found on disk carries its
   * own provenance and the host can confirm that the image which RAN is the one it built. A path and a
   * timestamp do not establish that; a stamp compared against the expected one does. */

  /* Ask the hardware for its own tile edge and report it, so the host can check that the corpus it
   * selected matches the geometry that actually ran. Reporting only; the kernel establishes its own
   * vector length inside each fused block and does not inherit this one. */
  size_t tile = 0;
#ifdef OPU_SCALAR_TILE
  tile = (size_t)OPU_TILE_EDGE;
#else
  asm volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(tile));
#endif
{('  htif_puts("' + _PROV_LINE + ' " PROV_STAMP "\\n");' + chr(10)) if provenance_stamp else ""}  htif_puts("{_TILE_LINE} ");
  htif_putd((long)tile);
  htif_putc('\\n');

  for (int ci = 0; ci < N_CASES; ++ci) {{
    const struct opu_case *c = &CASES[ci];
    const size_t elems = c->m * c->n;

    for (size_t i = 0; i < elems; ++i) {{
      c_dev[i] = 0;
      c_ref[i] = 0;
    }}

    /* Cycles for the DEVICE kernel alone. This is what makes a certification run a MEASUREMENT run:
     * `routing.MeasuredCost` declines a unit absent from its throughput table, so without a measured
     * figure the router refuses to move any work onto the unit -- correctly, but permanently. Reading
     * `mcycle` around the call is the cheapest honest way to produce that figure, on the same RTL and
     * the same shapes the verdict is about.
     *
     * The in-image reference is deliberately OUTSIDE the bracket: it is a scalar triple loop costing far
     * more than the kernel it checks, and including it would report a number about the harness. The
     * buffer zeroing is outside too, for the same reason. */
    uint64_t cyc0, cyc1;
    __asm__ volatile("csrr %0, mcycle" : "=r"(cyc0));
    {kernel_func}(c_dev, c->at, c->b, c->bias, c->m, c->n, c->k);
    __asm__ volatile("csrr %0, mcycle" : "=r"(cyc1));

    /* -1 means NOT COMPUTED, and the host must not read it as agreement. The in-image reference is a
     * scalar triple loop, so on cycle-accurate RTL it costs far more than the kernel it is checking;
     * OPU_SCREEN_ONLY skips it to make a re-run cheap. That weakens the run to a SCREENING one -- the
     * host-side digest still catches a wrong result, but nothing in the image cross-checks which buffer
     * was hashed -- so the verdict refuses to certify it. */
    long mismatches = -1;
    long first_bad = -1;
#ifndef OPU_SCREEN_ONLY
    {ref_func}(c_ref, c->at, c->b, c->bias, c->m, c->n, c->k);
    mismatches = 0;
    for (size_t i = 0; i < elems; ++i) {{
      if (c_dev[i] != c_ref[i]) {{
        if (first_bad < 0)
          first_bad = (long)i;
        ++mismatches;
      }}
    }}
#endif

#ifdef OPU_DUMP_MISMATCHES
    /* Debug aid: print the first few disagreeing elements as (index, device, reference). A count and a
     * digest say THAT a case is wrong; localising the cause needs the values -- zero means the tile was
     * never written, a plausible-but-wrong sum means the arithmetic ran on the wrong operands. */
    {{
      long shown = 0;
      for (size_t i = 0; i < elems && shown < (OPU_DUMP_MISMATCHES); ++i) {{
        if (c_dev[i] != c_ref[i]) {{
          htif_puts("DIFF ");
          htif_puts(c->name);
          htif_putc(' ');
          htif_putd((long)i);
          htif_putc(' ');
          htif_putd((long)c_dev[i]);
          htif_putc(' ');
          htif_putd((long)c_ref[i]);
          htif_putc('\\n');
          ++shown;
        }}
      }}
    }}
#endif
    htif_puts("{_CASE_LINE} ");
    htif_puts(c->name);
    htif_putc(' ');
    htif_putd((long)c->m);
    htif_putc(' ');
    htif_putd((long)c->n);
    htif_putc(' ');
    htif_putd((long)c->k);
    htif_putc(' ');
    htif_putd(mismatches);
    htif_putc(' ');
    htif_putd(first_bad);
    htif_putc(' ');
    put_hex64(fnv1a64(c_dev, elems * sizeof(int32_t)));
    htif_putc('\\n');

    /* A SEPARATE line rather than another CASE field: a parser that does not know about cycles ignores an
     * unknown line, while a widened CASE line would make every existing reader mis-split. */
    htif_puts("{_CYCLES_LINE} ");
    htif_puts(c->name);
    htif_putc(' ');
    htif_putd((long)(cyc1 - cyc0));
    htif_putc('\\n');
  }}

  htif_puts("{_DONE_LINE}\\n");
  htif_exit(0);
  return 0;
}}
"""


def provenance_stamp(provenance: Mapping[str, Any]) -> str:
    """A short, printable identity for what an image was generated from.

    Deliberately compact: it travels through a bare-metal console one character at a time. It carries the
    pinned revision, the digest of the sources actually read, and our own commit -- enough to tell a stale
    binary from a fresh one, which is the question a path and a timestamp cannot answer.
    """
    pins = provenance.get("hardware_pins") or {}
    parts = []
    for name in sorted(pins):
        got = (pins[name].get("observed") or {}).get("commit", "UNKNOWN")
        parts.append(f"{name}={got[:12]}")
    src = str(provenance.get("source_digest") or "UNKNOWN")[:12]
    mine = str((provenance.get("merlin") or {}).get("commit") or "UNKNOWN")[:12]
    return " ".join([*parts, f"src={src}", f"merlin={mine}"])


# ---------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------

#: Harness sources the corpus image needs. `rvv_matmul_i8.S` is deliberately absent -- it is another
#: kernel entirely and linking it would put unrelated vector instructions in the image the audit reads.
_HARNESS_FILES = ("crt.S", "htif.c", "libc_min.c")


def build_image(cases: Sequence[opu_corpus.Case], encodings: Mapping[str, Any], spec: KernelSpec,
                workdir: "str | Path", *, derivation_ok: bool = True, scalar_tile: int | None = None,
                screen_only: bool = False, operand_align: int | None = None,
                dump_mismatches: int = 0, provenance_stamp: str | None = None,
                extra_cflags: Sequence[str] = ()) -> Path:
    """Compile the corpus image and return the ELF.

    ``scalar_tile`` selects the pre-flight build: the unit is replaced by a scalar stand-in and the tile
    edge is supplied rather than probed, which is how the tiling loop gets exercised at edges no available
    part has. A device build passes None.
    """
    from ..runtime.backends import spike as spike_backend

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    kernel_c = work / "opu_kernel_gen.c"
    kernel_c.write_text(emit_microkernel(encodings, spec, derivation_ok=derivation_ok)
                        + "\n" + emit_reference_c(), encoding="utf-8")
    main_c = work / IMAGE_MAIN
    main_c.write_text(emit_image_c(cases, kernel_func=spec.func_name,
                                   operand_align=operand_align,
                                   provenance_stamp=provenance_stamp), encoding="utf-8")

    harness = spike_backend.harness_dir()
    stem = "opu_corpus_scalar" if scalar_tile is not None else "opu_corpus"
    elf = work / f"{stem}{'_screen' if screen_only else ''}.elf"
    cmd = [
        str(spike_backend.gcc_path()),
        "-march=rv64gcv", "-mabi=lp64d", "-mcmodel=medany",
        # -fno-tree-vectorize keeps the in-image reference a genuinely independent scalar implementation
        # rather than a second vectorised one, which is the whole point of having it.
        "-O2", "-fno-tree-vectorize", "-ffreestanding", "-nostdlib", "-nostartfiles",
        "-I", str(harness),
        "-T", str(harness / "link.ld"),
        *(str(harness / f) for f in _HARNESS_FILES),
        str(kernel_c), str(main_c),
        *(["-DOPU_SCALAR_TILE", f"-DOPU_TILE_EDGE={int(scalar_tile)}"] if scalar_tile is not None
          else []),
        *(["-DOPU_SCREEN_ONLY"] if screen_only else []),
        *([f"-DOPU_DUMP_MISMATCHES={int(dump_mismatches)}"] if dump_mismatches > 0 else []),
        *extra_cflags,
        "-o", str(elf),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"building the corpus image failed:\n{proc.stderr[-4000:]}")
    return elf


# ---------------------------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------------------------


def parse_console(text: str) -> dict[str, Any]:
    """``{tile, done, cases: {name: {...}}}`` from the image's console.

    Structural field-splitting, and an unparseable CASE line is collected into ``malformed`` rather than
    skipped -- a dropped line would shrink the set of cases the report believes ran.
    """
    tile: int | None = None
    done = False
    stamp: str | None = None
    cases: dict[str, dict[str, Any]] = {}
    cycles: dict[str, int] = {}
    malformed: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(f"{_TILE_LINE} "):
            token = line.split()[1] if len(line.split()) > 1 else ""
            try:
                tile = int(token)
            except ValueError:
                malformed.append(line)
        elif line.startswith(f"{_CASE_LINE} "):
            parts = line.split()
            if len(parts) != 8:
                malformed.append(line)
                continue
            _, name, m, n, k, mismatches, first_bad, digest = parts
            try:
                cases[name] = {
                    "m": int(m), "n": int(n), "k": int(k),
                    "mismatches": int(mismatches), "first_bad": int(first_bad),
                    "digest": int(digest, 16),
                }
            except ValueError:
                malformed.append(line)
        elif line.startswith(f"{_CYCLES_LINE} "):
            parts = line.split()
            if len(parts) != 3:
                malformed.append(line)
                continue
            try:
                cycles[parts[1]] = int(parts[2])
            except ValueError:
                malformed.append(line)
        elif line.startswith(f"{_PROV_LINE} "):
            stamp = line.split(None, 1)[1].strip()
        elif line == _DONE_LINE:
            done = True
    return {"tile": tile, "done": done, "cases": cases, "cycles": cycles, "stamp": stamp,
            "malformed": tuple(malformed)}


# ---------------------------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    """One case's verdict, with both checks kept separate so a disagreement between them is visible."""

    name: str
    m: int
    n: int
    k: int
    ran: bool
    in_image_agrees: bool          # device kernel == in-image scalar reference
    digest_agrees: bool            # device kernel == numpy, carried by hash
    mismatches: int = 0
    first_bad: int = -1
    #: False when the image skipped its own reference (a screening build). A case whose in-image
    #: cross-check never ran is NOT certified, however well its digest matched: nothing then confirms
    #: which buffer was hashed, and reporting agreement for a check that did not execute is the exact
    #: shape of a vacuous pass.
    in_image_checked: bool = True
    #: Cycles the DEVICE kernel took, measured on the substrate that ran it. -1 when the image did not
    #: report one (an older build). Not part of the verdict: a slow kernel is still a correct kernel, and
    #: conflating the two is how a performance regression gets recorded as a numerical failure.
    cycles: int = -1
    note: str = ""

    @property
    def certified(self) -> bool:
        return (self.ran and self.in_image_checked and self.in_image_agrees
                and self.digest_agrees)

    @property
    def macs(self) -> int:
        return int(self.m) * int(self.n) * int(self.k)

    @property
    def macs_per_cycle(self) -> float | None:
        """Measured throughput for this shape, or None when no cycle count was reported.

        This is the number ``routing.MeasuredCost`` declines a unit for lacking. It is per SHAPE rather
        than per unit on purpose: a tiled unit's throughput is a function of how well the shape fills the
        tile, so a single headline figure would flatter the narrow shapes and understate the wide ones --
        which is precisely the error that made a crude cost model route 89 of 90 contractions onto the
        matrix unit.
        """
        if self.cycles <= 0:
            return None
        return self.macs / float(self.cycles)


@dataclass(frozen=True)
class CertReport:
    """The whole verdict. ``certified`` is true only when every case in reach passed BOTH checks."""

    config: str
    tile_edge: int
    results: tuple[CaseResult, ...] = ()
    deferred: tuple[tuple[str, str], ...] = ()
    uses_unit: bool | None = None
    unit_instruction_counts: dict[str, int] = field(default_factory=dict)
    tile_edge_reported: int | None = None
    #: Which external hardware revision this result belongs to (merlin.common.provenance.record). A
    #: certification without it is a number nobody can attribute to a device later.
    provenance: dict[str, Any] = field(default_factory=dict)
    gaps: tuple[str, ...] = ()
    version: int = REPORT_VERSION

    @property
    def certified(self) -> bool:
        return (not self.gaps and bool(self.results)
                and all(r.certified for r in self.results))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "config": self.config,
            "tile_edge": self.tile_edge,
            "tile_edge_reported": self.tile_edge_reported,
            "provenance": dict(self.provenance),
            "certified": self.certified,
            "uses_unit": self.uses_unit,
            "unit_instruction_counts": dict(self.unit_instruction_counts),
            "gaps": list(self.gaps),
            "n_certified": sum(1 for r in self.results if r.certified),
            "n_cases": len(self.results),
            "results": [
                {"name": r.name, "m": r.m, "n": r.n, "k": r.k, "ran": r.ran,
                 "in_image_agrees": r.in_image_agrees, "digest_agrees": r.digest_agrees,
                 "mismatches": r.mismatches, "first_bad": r.first_bad,
                 "in_image_checked": r.in_image_checked,
                 "cycles": r.cycles, "macs": r.macs, "macs_per_cycle": r.macs_per_cycle,
                 "certified": r.certified, "note": r.note}
                for r in self.results],
            "deferred": [{"name": n, "reason": why} for n, why in self.deferred],
        }

    def write(self, path: "str | Path") -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")
        return p


def certify(cases: Sequence[opu_corpus.Case], encodings: Mapping[str, Any], spec: KernelSpec,
            workdir: "str | Path", *, config: str, tile_edge: int, run,
            deferred: Sequence[tuple[opu_corpus.Case, str]] = (), derivation_ok: bool = True,
            scalar_tile: int | None = None, screen_only: bool = False,
            operand_align: int | None = None,
            provenance: Mapping[str, Any] | None = None) -> CertReport:
    """Build the image, check it actually uses the unit, run it, and judge the console.

    ``run`` is a callable taking the ELF path and returning console text, so the substrate is the
    caller's choice (spike for the scalar pre-flight, the Verilator sim for the real thing) and this
    function needs no hardware to be tested.

    The audit between build and run is the anti-cheat: a device image that contains none of the unit's
    instructions would compute correct answers on the host core and pass every numerical check, which is
    the most comfortable way for this whole exercise to be wrong.
    """
    stamp = provenance_stamp(provenance) if provenance else None
    elf = build_image(cases, encodings, spec, workdir, derivation_ok=derivation_ok,
                      scalar_tile=scalar_tile, screen_only=screen_only,
                      operand_align=operand_align, provenance_stamp=stamp)
    from .decode import opu as opu_audit
    audit = opu_audit.audit_object(elf, encodings)
    wanted = (spec.accumulate, spec.broadcast, spec.readout)
    counts = {name: int(audit.counts.get(name, 0)) for name in wanted}
    uses_unit = all(counts[name] > 0 for name in (spec.accumulate, spec.readout))

    report = verdict(run(elf), cases, config=config, tile_edge=tile_edge, deferred=deferred,
                     uses_unit=uses_unit, unit_counts=counts, provenance=provenance,
                     expected_stamp=stamp)
    if scalar_tile is not None:
        # The pre-flight must NOT use the unit; if it does, the stand-in was not selected and this run
        # says nothing about the tiling loop it was built to exercise.
        extra = () if not uses_unit else (
            "the scalar pre-flight image contains the unit's instructions, so OPU_SCALAR_TILE did not "
            "take effect and this run is not the host-checkable build it claims to be",)
        report = CertReport(config=report.config, tile_edge=report.tile_edge, results=report.results,
                            deferred=report.deferred, uses_unit=uses_unit,
                            unit_instruction_counts=counts,
                            tile_edge_reported=report.tile_edge_reported,
                            provenance=report.provenance,
                            gaps=tuple(g for g in report.gaps
                                       if not g.startswith("the image contains none")) + extra)
    return report


def verdict(console: str, cases: Sequence[opu_corpus.Case], *, config: str, tile_edge: int,
            deferred: Sequence[tuple[opu_corpus.Case, str]] = (),
            uses_unit: bool | None = None,
            unit_counts: Mapping[str, int] | None = None,
            provenance: Mapping[str, Any] | None = None,
            expected_stamp: str | None = None) -> CertReport:
    """Judge a console against host-derived truth. Absent and malformed cases are gaps, not passes."""
    parsed = parse_console(console)
    expected = expected_digests(cases)
    gaps: list[str] = []
    if not parsed["done"]:
        gaps.append("the image did not report DONE, so the run did not complete")
    if parsed["malformed"]:
        gaps.append(f"{len(parsed['malformed'])} unparseable console line(s): "
                    f"{list(parsed['malformed'])[:3]}")
    reported = parsed["tile"]
    if reported is not None and int(reported) != int(tile_edge):
        gaps.append(f"the hardware reported a tile edge of {reported} but the corpus was selected for "
                    f"{tile_edge}; the run did not test the shapes this report names")
    if uses_unit is False:
        gaps.append("the image contains none of the unit's instructions, so whatever it computed, it "
                    "did not use the unit")
    if expected_stamp is not None:
        got_stamp = parsed.get("stamp")
        if got_stamp is None:
            gaps.append("the image reported no provenance stamp, so the result cannot be attributed to a "
                        "hardware revision")
        elif got_stamp != expected_stamp:
            gaps.append(f"the image reported provenance {got_stamp!r} but this build expects "
                        f"{expected_stamp!r}; a stale binary ran")
    prov = dict(provenance or {})
    for name, entry in (prov.get("hardware_pins") or {}).items():
        # Material drift only. A dirty tree elsewhere in the checkout is recorded but not a gap, because
        # `source_digest` already pins the bytes that were actually read; a wrong revision or missing
        # hardware is a different matter and cannot be reconciled after the fact.
        if entry.get("missing_paths"):
            gaps.append(f"pin {name!r} is missing {entry['missing_paths']}: this checkout does not "
                        "contain the hardware this result claims to be about")
        if entry.get("forbidden_present"):
            gaps.append(f"pin {name!r} declares {entry['forbidden_present']} absent but they are "
                        "present, so this is not the revision it claims to be")
        for d in entry.get("drift", ()):
            if d.startswith("commit is") or d.startswith("no checkout"):
                gaps.append(f"pin {name!r}: {d}")

    results: list[CaseResult] = []
    for case in cases:
        exp = expected[case.name]
        got = parsed["cases"].get(case.name)
        if got is None:
            results.append(CaseResult(name=case.name, m=int(case.m), n=int(case.n), k=int(case.k),
                                      ran=False, in_image_agrees=False, digest_agrees=False,
                                      note="no CASE line for this case in the console"))
            continue
        shape_ok = (got["m"], got["n"], got["k"]) == (exp["m"], exp["n"], exp["k"])
        notes = []
        if not shape_ok:
            notes.append(f"the image ran m={got['m']} n={got['n']} k={got['k']}, not the shape this "
                         "case names")
        checked = int(got["mismatches"]) >= 0
        if not checked:
            notes.append("the image skipped its own reference (screening build), so nothing in it "
                         "cross-checked the device result; this case is screened, not certified")
        results.append(CaseResult(
            name=case.name, m=int(case.m), n=int(case.n), k=int(case.k), ran=True,
            in_image_agrees=(checked and int(got["mismatches"]) == 0),
            digest_agrees=(got["digest"] == exp["digest"] and shape_ok),
            mismatches=int(got["mismatches"]), first_bad=int(got["first_bad"]),
            in_image_checked=checked,
            # -1 when the image reported none, which an older build will not. Absent, not zero: a zero
            # cycle count would compute an infinite throughput.
            cycles=int(parsed.get("cycles", {}).get(case.name, -1)),
            note="; ".join(notes)))

    return CertReport(config=config, tile_edge=int(tile_edge), results=tuple(results),
                      deferred=tuple((c.name, why) for c, why in deferred),
                      uses_unit=uses_unit, unit_instruction_counts=dict(unit_counts or {}),
                      tile_edge_reported=None if reported is None else int(reported),
                      provenance=prov, gaps=tuple(gaps))
