"""Emit the C microkernel for a matrix extension, with every instruction word from a derived table.

This is the golden manual delta — the smallest correct integer path, deliberately easy to reason about,
and explicitly NOT the research claim. Its value is as an oracle: a hand implementation whose behaviour
is known lets the generated path be judged against something other than its own output.

Two properties are the reason this is generated rather than written by hand once.

**Every encoding comes from :mod:`targetgen.rtl.opu_isa`, so a hardware revision cannot silently
mis-encode the kernel.** The instructions occupy reserved slots and no assembler knows their mnemonics,
so a wrong field does not fail to build — it emits a neighbouring instruction. The funct6 values are
``ChiselEnum`` ordinals, i.e. a function of declaration order in an RTL source file, so a literal typed
here would rot the first time an upstream edit inserts a slot.

**The operand setup and the accumulate are fused into one ``asm volatile``.** This is the historical bug,
not a hypothetical: LLVM's vector-length insertion pass treats a standalone ``asm volatile("vsetvli
…")`` as opaque and does not track the length it establishes, so it may leave a later load running on a
length set for something else. In the failure that motivated all of this, both operand loads silently
used a length of 16 and one read past the end of its panel. Keeping the whole sequence — configure,
load, configure, load, accumulate — inside a single asm block means the pass has no separate
instructions to reason about, because it cannot see inside.

The generated loop also pre-zeroes the row operand and loads it tail-undisturbed. The accumulate's row
count is whatever the row vector holds, so lanes past the real panel would otherwise multiply undefined
register contents into the accumulator; zeroed lanes contribute nothing, which makes a short panel
harmless instead of merely unlikely to be read.

Layout is a derived requirement rather than a choice: the hardware indexes both operands K-major (the
expert kernel reads ``at[k*M + i]`` and ``b[k*N + j]``), so the left operand arrives transposed and the
packing that produces it is a real cost term the routing decision has to price.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["KernelSpec", "emit_microkernel", "emit_reference_c"]


@dataclass(frozen=True)
class KernelSpec:
    """What the emitted kernel needs to know, all of it either derived or a caller's choice.

    ``matrix_reg`` / ``row_vreg`` / ``col_vreg`` are register NUMBERS. The assembler will not accept the
    matrix-register spellings (they do not exist in its tables), so the convention every real binary for
    this unit uses is to write integer-register names in the ``.insn`` operand slots and let the field
    encode the number. That is a property of there being no assembler support, not a claim that these are
    integer registers.
    """

    accumulate: str                # the encoding-table name of the accumulate
    broadcast: str                 # the encoding-table name of the bias/zero broadcast
    readout: str                   # the encoding-table name of the row readout
    matrix_reg: int = 1
    row_vreg: int = 5              # holds the LHS row operand
    col_vreg: int = 4              # holds the RHS column operand
    out_vreg: int = 0              # readout destination, and the broadcast source
    func_name: str = "opu_gemm_i8"
    #: A SECOND register for the left operand, rotated with ``row_vreg`` across reduction steps. None
    #: keeps the single-register form.
    #:
    #: This exists because of a hazard hole in the unit's own sequencer, read from its RTL: for an
    #: accumulate the LHS read-intent (``rvs1_mask``) is cleared on EVERY iteration, while the RHS
    #: (``rvs2_mask``) is gated on ``row_idx_tail``. One accumulate iterates ``(vLen/dLen)²`` subtiles and
    #: the LHS element group is indexed by ``row_idx``, so the two column iterations of a given row read
    #: the SAME group -- and the first of them clears the intent bit. For the remaining iteration the
    #: hardware no longer advertises that it is reading the operand, so a younger write to it is not
    #: blocked by the ``war_hazard`` check. Rotating the register means the write that follows an
    #: accumulate targets a register that accumulate is not reading.
    #: Defaults ON: the hazard below is present in the unit's shipped RTL, so the single-register
    #: form emits a kernel that is wrong on this hardware. None opts out, for A/B testing only.
    row_vreg_alt: int | None = 6
    #: Operand and accumulator element widths in bits. Not target facts — they are what an int8 kernel
    #: accumulating in int32 *is*, and the ratio between them is load-bearing (see :attr:`acc_lmul`).
    operand_bits: int = 8
    acc_bits: int = 32

    @property
    def acc_lmul(self) -> int:
        """The register-group multiplier a full tile row of accumulator needs — a DERIVED ratio.

        The tile edge is ``VLEN / operand_bits`` lanes, and each lane's accumulator is ``acc_bits`` wide,
        so holding one row takes ``tile_edge * acc_bits / VLEN = acc_bits / operand_bits`` registers. The
        ``VLEN`` cancels, which is why the same multiplier is right for every configuration of the unit
        and why this is a ratio rather than a constant someone measured on one part.

        Getting it wrong does not fail loudly. At ``LMUL = 1`` and ``VLEN = 256`` the readout's ``vsetvli``
        clamps ``vl`` to ``256/32 = 8``, so a 32-wide tile row stores its first 8 elements and silently
        drops the rest.
        """
        return max(1, int(self.acc_bits) // int(self.operand_bits))

    def __post_init__(self) -> None:
        # A vector register group of N registers must start at a multiple of N. The accumulator is read
        # and written as such a group (`vle32.v`/`vse32.v` under the operand vtype have EMUL = acc_lmul),
        # so a misaligned base is an illegal encoding rather than a slower one.
        if int(self.out_vreg) % self.acc_lmul:
            raise ValueError(
                f"out_vreg={self.out_vreg} must be a multiple of {self.acc_lmul} (the accumulator is a "
                f"{self.acc_lmul}-register group, and a group must be aligned to its size)")
        span = range(int(self.out_vreg), int(self.out_vreg) + self.acc_lmul)
        pairs = [("row_vreg", self.row_vreg), ("col_vreg", self.col_vreg)]
        if self.row_vreg_alt is not None:
            pairs.append(("row_vreg_alt", self.row_vreg_alt))
            if int(self.row_vreg_alt) == int(self.row_vreg):
                raise ValueError("row_vreg_alt must differ from row_vreg, or it rotates nothing and the "
                                 "write still targets the register the accumulate is reading")
        for name, reg in pairs:
            if int(reg) in span:
                raise ValueError(
                    f"{name}=v{reg} falls inside the accumulator group "
                    f"v{self.out_vreg}..v{self.out_vreg + self.acc_lmul - 1}; the operand loads would "
                    "overwrite the accumulator between the broadcast and the readout")


def _x(n: int) -> str:
    """An operand slot spelling the assembler accepts. See :class:`KernelSpec`."""
    return f"x{int(n)}"


def _require(encodings: Mapping[str, Any], *names: str) -> None:
    missing = [n for n in names if n not in encodings]
    if missing:
        raise ValueError(f"the derived encoding table is missing {missing}; refusing to emit a kernel "
                         "with a guessed instruction")


def emit_microkernel(encodings: Mapping[str, Any], spec: KernelSpec, *,
                     derivation_ok: bool = True) -> str:
    """The C source for one tile's worth of GEMM on the extension.

    ``encodings`` is :attr:`targetgen.rtl.opu_isa.IsaDerivation.encodings`. ``derivation_ok`` is that
    derivation's own verdict: emitting from a derivation whose RTL and cross-check source disagreed would
    bake in whichever side happened to be read, so this refuses rather than picking one.
    """
    if not derivation_ok:
        raise ValueError("the encoding derivation did not agree with its cross-check source; refusing "
                         "to emit a kernel from an unresolved encoding")
    _require(encodings, spec.accumulate, spec.broadcast, spec.readout)
    acc, bcast, out = (encodings[spec.accumulate], encodings[spec.broadcast],
                       encodings[spec.readout])
    md, vr, vc, vo = _x(spec.matrix_reg), _x(spec.row_vreg), _x(spec.col_vreg), _x(spec.out_vreg)

    def _fused(row_reg: int) -> str:
        """One reduction step, with the left operand in ``row_reg``."""
        return "\\n\\t".join([
            # Zero every row lane at the maximum length, so lanes past the real panel hold 0 rather than
            # whatever the register happened to contain.
            "vsetvli t0, zero, e8, m1, ta, ma",
            f"vmv.v.i v{row_reg}, 0",
            # Row operand: `ml` lanes, tail UNDISTURBED so the zeros above survive.
            "vsetvli zero, %[ml], e8, m1, tu, ma",
            f"vle8.v v{row_reg}, (%[ap])",
            # Column operand: `nl` lanes. This is also the length the accumulate runs at.
            "vsetvli zero, %[nl], e8, m1, ta, ma",
            f"vle8.v v{spec.col_vreg}, (%[bp])",
            acc.insn_r(md, _x(row_reg), vc),
        ])

    # Bound for the partial-N tail scratch below. It is a CAP, not an assumption: the emitted guard falls
    # back to the direct (unpadded) call when k exceeds it, so a deeper contraction still compiles and runs
    # -- it just does not get the tail workaround. 9216 covers the deepest reduction we lower today
    # (Gemma's ffn down, k=9216) at 9216*64 = 576 KiB of .bss, which fits the 512 MB part comfortably.
    pad_k_max = 9216

    fused = _fused(spec.row_vreg)
    fused_alt = _fused(spec.row_vreg_alt) if spec.row_vreg_alt is not None else None

    return f"""\
/* GENERATED by merlin.kernels.opu_kernel — do not edit.
 *
 * Every .insn word below is derived from the hardware's own sources and cross-checked against the
 * expert header; see merlin.targetgen.rtl.opu_isa. Both operands are K-major, which the hardware
 * requires: the left operand arrives already transposed.
 *
 *   accumulate : {spec.accumulate:<12} opcode={acc.opcode:#04x} funct3={acc.funct3} funct6={acc.funct6}
 *   broadcast  : {spec.broadcast:<12} opcode={bcast.opcode:#04x} funct3={bcast.funct3} funct6={bcast.funct6}
 *   readout    : {spec.readout:<12} opcode={out.opcode:#04x} funct3={out.funct3} funct6={out.funct6}
 */
#include <stdint.h>
#include <stddef.h>

/* One TILE: C[i0:i0+ml, j0:j0+nl] += sum_k A[k, i0:i0+ml] * B[k, j0:j0+nl], in int32.
 * `at` is K-major (K x M), `b` is K-major (K x N), `bias` is per output column or NULL.
 * `ml`/`nl` must be <= the unit's tile edge; {spec.func_name} below tiles a full contraction into
 * calls to this. */
static void {spec.func_name}_tile(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,
                                  size_t m, size_t n, size_t k, size_t ml, size_t nl)
{{
#ifdef OPU_SCALAR_TILE
  /* A scalar stand-in for the unit, selected at COMPILE time. Its only purpose is to let the tiling
   * loop below -- the tail bounds, the pointer arithmetic, the bias column offset -- be checked
   * numerically on a host with no such hardware. There is one copy of that loop, so what the host
   * validates is the same code the device build runs, not a re-implementation of it. */
  for (size_t i = 0; i < ml; ++i)
    for (size_t j = 0; j < nl; ++j) {{
      int32_t sum = bias ? bias[j] : 0;
      for (size_t kk = 0; kk < k; ++kk)
        sum += (int32_t)at[kk * m + i] * (int32_t)b[kk * n + j];
      c[i * n + j] = sum;
    }}
  return;
#else
  /* Initialise the accumulator: broadcast the bias row down every row of the tile, or zero.
   *
   * THE RULE, and it was learned the hard way twice: every instruction runs under the vtype of the data
   * IT moves, with LMUL sized so `vl` spans a full tile row. The accumulate carries
   * {spec.operand_bits}-bit operands and runs at e{spec.operand_bits}/m1; the broadcast and the readout
   * carry {spec.acc_bits}-bit accumulator rows and run at e{spec.acc_bits}/m{spec.acc_lmul}.
   *
   * Both failure modes were MEASURED on this unit's RTL, and neither announces itself:
   *
   *   - e{spec.acc_bits}/m1 HANGS the core. No trap, no retire, the simulation spins forever, because
   *     that vl cannot span a tile row.
   *   - e{spec.operand_bits}/m1 -- which is what this code used, and what the unit's own expert kernel
   *     does -- SILENTLY UNDER-INITIALISES. One register at that vtype is
   *     {spec.operand_bits} * tile_edge bits = tile_edge/{spec.acc_lmul} accumulator elements, so only
   *     the first quarter-row of the tile is written and the rest keeps whatever the matrix register held.
   *     The symptom is a wrong answer whose FIRST BAD COLUMN is exactly that boundary, and whose
   *     mismatch count changes with unrelated contents of the same image -- a case passed or failed
   *     depending on which OTHER cases were compiled beside it. The expert kernel gets away with it
   *     because it is validated on square tiles, where its `vl` (the row count) happens to equal the
   *     column count.
   *
   * At e{spec.acc_bits}/m{spec.acc_lmul} the `vle32.v` has EMUL = LMUL, so it fills the whole accumulator
   * group v{spec.out_vreg}..v{spec.out_vreg + spec.acc_lmul - 1}, and `vmv.v.i` zeroes all of it. */
  if (bias) {{
    asm volatile("vsetvli zero, %[nl], e{spec.acc_bits}, m{spec.acc_lmul}, ta, ma\\n\\t"
                 "vle32.v v{spec.out_vreg}, (%[bp])\\n\\t"
                 "{bcast.insn_r(md, _x(0), vo)}"
                 :: [nl] "r"(nl), [bp] "r"(bias)
                 : "memory");
  }} else {{
    asm volatile("vsetvli zero, %[nl], e{spec.acc_bits}, m{spec.acc_lmul}, ta, ma\\n\\t"
                 "vmv.v.i v{spec.out_vreg}, 0\\n\\t"
                 "{bcast.insn_r(md, _x(0), vo)}"
                 :: [nl] "r"(nl));
  }}

  /* One fused block per reduction step. Configure, load, configure, load, accumulate -- all inside a
   * single asm volatile, so the vector-length insertion pass cannot reinterpret either length. */
{_reduction_loop(fused, fused_alt, spec)}

  /* Readout is row-serial and it is the only way out of the accumulator.
   *
   * LMUL here is {spec.acc_lmul} = acc_bits / operand_bits, DERIVED rather than chosen: one tile row is
   * `VLEN / operand_bits` lanes of `acc_bits` each, so it needs that many registers and the VLEN cancels.
   *
   * An under-provisioned LMUL is FATAL, not merely lossy. MEASURED on the unit's RTL: at e{spec.acc_bits}
   * with m1 the readout HANGS the core -- vl clamps to VLEN/acc_bits (8 at VLEN=256), which cannot cover a
   * 32-lane tile row, and the unit stalls without trapping or retiring. The same vtype at m{spec.acc_lmul}
   * completes, and so does the operand vtype e{spec.operand_bits}/m1. So the constraint is not "avoid the
   * accumulator width" but "vl must be able to span a tile row". */
  asm volatile("vsetvli zero, %[nl], e{spec.acc_bits}, m{spec.acc_lmul}, ta, ma" :: [nl] "r"(nl));
  for (size_t r = 0; r < ml; ++r) {{
    asm volatile("{out.insn_r(vo, '%[r]', md)}\\n\\t"
                 "vse32.v v{spec.out_vreg}, (%[cp])"
                 :: [r] "r"(r), [cp] "r"(c + r * n)
                 : "memory");
  }}
#endif
}}

/* A full contraction: C[0:m, 0:n] = sum_k A[k, :] * B[k, :], tiled over M and N.
 *
 * The tile edge is READ FROM THE HARDWARE at run time rather than compiled in: `vsetvli` with an
 * unbounded requested length returns the maximum lane count for the element width, which is the tile
 * edge for this unit. Baking a constant here would tie the object to one configuration of the unit --
 * the same mistake the acceptance corpus made by holding an extent at a literal -- and would be wrong
 * silently, since a too-large tile reads past a panel and a too-small one just leaves the unit idle.
 *
 * Tails are the interesting part: the last row and column tile are short, and a short tile is exactly
 * where the historical failure lived. They go through the SAME tile routine with a smaller length, so
 * there is no separate tail path that could be right in the full case and wrong at the edge.
 *
 * THE TILE LOOP IS THE PARALLEL ONE, and on a chip with a unit per core it is the only place the extra
 * units can be reached from: a routed contraction is one opaque call by the time the parallel transform
 * schedule runs, so that schedule cannot split it. Distinct tiles write disjoint spans of `c` and read
 * `at`/`b` without writing them, so the iterations are independent by construction; each core
 * accumulates in ITS OWN matrix register file, so there is no shared accumulator to race on either.
 *
 * TWO loop forms are emitted rather than one parallel form that degenerates. `collapse(2)` needs a
 * perfect nest, which means computing `ml` inside the inner loop -- and MEASURED on the RTL that costs
 * the serial build 1.8% overall and 2.8-3.9% on the workload-scale shapes, because the row length stops
 * being hoisted out of the column loop. The certified configuration should not pay for a capability it
 * does not use, so `#ifndef _OPENMP` keeps exactly the loop the corpus certified 47/47 and the collapsed
 * form appears only under -fopenmp. Both loops are collapsed there because splitting only the outer one
 * hands out `ceil(m/tile)` pieces -- 4 for a 196-row activation, which divides badly over 2 or 3 cores --
 * while the collapsed space is `ceil(m/tile) * ceil(n/tile)`, e.g. 64 for 196x1024. */

/* PARTIAL-N-TILE WORKAROUND for the taped-out unit.
 *
 * MEASURED on the part with the frozen 47-case corpus: every case with `n % tile == 0` passes (16 of 16)
 * and every case with a partial N tile fails (0 of 31) -- a perfect predictor over all 47 -- and the first
 * wrong output is exactly the first element of the tail (workload_im2col n=196, first_bad=192 = 3*64,
 * reproduced independently at k=32 and k=768). M tails are unaffected. The identical images are bit-exact
 * on the FPGA revision, whose bitstream is 128 commits ahead of the taped-out one, so the two revisions
 * disagree about what a short VL means for the matrix ops. Which revisions those are is recorded in the
 * pin registry and in the measurement artifact, not named here: this file must not carry a target
 * literal. The silicon is not ours to fix.
 *
 * So never hand the unit a short N: copy the tail's B panel into a tile-wide ZERO-PADDED scratch, run the
 * tile at full width, and copy back only the `nl` columns that exist. Padding columns accumulate against
 * zeroed weights so they cannot perturb the real ones, and the copy is paid on the tail tile only.
 *
 * OPU_SCALAR_TILE keeps the direct path: that build has no such unit, it is what the unit tests compile,
 * and correctness there is defined by the reference rather than by this erratum. */
/* Default: ON for a real unit, OFF for the scalar stand-in, whose reference has no such erratum.
 * But it must be POSSIBLE to force it on together with the stand-in. Coupling the two meant the padded
 * path -- and therefore its concurrency -- was compiled by no test on any host, which is exactly how a
 * data race in it survived review: `TestTheTileLoopIsParallel` built the parallel loop with the padded
 * branch preprocessed away. */
#ifndef MERLIN_OPU_PAD_PARTIAL_N
# ifndef OPU_SCALAR_TILE
#  define MERLIN_OPU_PAD_PARTIAL_N 1
# endif
#endif

#ifdef MERLIN_OPU_PAD_PARTIAL_N
/* The cap is DERIVED, not written down. A literal here would be a VLEN-512 fact baked into a kernel
 * whose whole point is that it reads its tile edge from the hardware at run time: the buffers below are
 * sized by it, so a wider unit would silently stop being padded and fall back to the direct path that
 * the partial-N erratum makes wrong. `__riscv_v_min_vlen` is defined by the compiler from the `-march`
 * the caller passed (`rv64gcv_zvl512b` here, from the board descriptor's vlen), so this tracks the
 * target with no literal and no plumbing. The scalar stand-in has no V at all and supplies its edge
 * directly, which is also what lets the fixtures sweep the tail alignments. */
#ifndef MERLIN_OPU_TILE_CAP
# if defined(__riscv_v_min_vlen)
#  define MERLIN_OPU_TILE_CAP (__riscv_v_min_vlen / {spec.operand_bits})
# elif defined(OPU_TILE_EDGE)
#  define MERLIN_OPU_TILE_CAP OPU_TILE_EDGE
# else
#  error "cannot derive MERLIN_OPU_TILE_CAP: no __riscv_v_min_vlen and no OPU_TILE_EDGE"
# endif
#endif
/* The pad buffers are split by WHO WRITES THEM, because the previous arrangement raced.
 *
 * B and bias padding depend only on the TAIL COLUMN BLOCK, which is identical for every row block.
 * They are therefore padded ONCE, before the parallel region, and are read-only inside it. That is a
 * correctness fix first: these were being written from inside an `omp parallel for collapse(2)`
 * region, where every row block `i` shares the same tail column `j`, so concurrent threads
 * overwrote each other's pad and the result was silently wrong rather than a crash. It is also less
 * work -- the old placement re-padded the identical B panel ceil(m/tile) times.
 *
 * The C staging buffer cannot be shared: it is a per-tile OUTPUT. It is a STACK LOCAL, which makes it
 * per-thread by construction with no bound to get wrong. A slab indexed by `omp_get_thread_num()` was
 * tried and is a trap: it needs a compile-time thread bound, and any clamp on that index SILENTLY
 * ALIASES two threads onto one buffer -- measured, 59 of 60 runs wrong at 64 threads, every wrong
 * element in the tail column block. 16 KB of frame is affordable on both paths (bare-metal reserves
 * 256 KB, a Zephyr OMP worker ~8 MB) and it is entered on the tail column only. */
static int8_t  merlin_opu_bpad[{pad_k_max} * MERLIN_OPU_TILE_CAP] __attribute__((aligned(64)));
static int32_t merlin_opu_biaspad[MERLIN_OPU_TILE_CAP] __attribute__((aligned(64)));

/* Pad the single tail column block. Returns 0 if it cannot be padded, in which case the caller runs
 * every tile direct -- the pre-padding behaviour, preserved exactly. Called OUTSIDE the region. */
static int {spec.func_name}_pad_tail(const int8_t *b, const int32_t *bias,
                                     size_t n, size_t k, size_t nl, size_t tile)
{{
  if (tile > MERLIN_OPU_TILE_CAP || k > {pad_k_max}) return 0;
  /* Plain copy/zero loops. clang's loop-idiom pass would turn these into memcpy/memset CALLS, which
   * this freestanding object cannot link -- but the shipping build already compiles with
   * `-ffreestanding -fno-builtin` (zephyr_model RVV_CFLAGS), under which no libcall is emitted.
   * Verified both ways: at plain `-O2` the object needs memcpy+memset, with the real flags it needs
   * nothing. So this is the caller's flags to get right, not something to obfuscate here. */
  for (size_t kk = 0; kk < k; ++kk) {{
    const int8_t *src = b + kk * n;
    int8_t *dst = merlin_opu_bpad + kk * tile;
    for (size_t j = 0; j < nl; ++j) dst[j] = src[j];
    for (size_t j = nl; j < tile; ++j) dst[j] = 0;
  }}
  if (bias) {{
    for (size_t j = 0; j < nl; ++j) merlin_opu_biaspad[j] = bias[j];
    for (size_t j = nl; j < tile; ++j) merlin_opu_biaspad[j] = 0;
  }}
  return 1;
}}

static void {spec.func_name}_tile_padded(int32_t *c, const int8_t *at, const int32_t *bias,
                                         size_t m, size_t n, size_t k,
                                         size_t ml, size_t nl, size_t tile)
{{
  int32_t cpad[MERLIN_OPU_TILE_CAP * MERLIN_OPU_TILE_CAP] __attribute__((aligned(64)));
  {spec.func_name}_tile(cpad, at, merlin_opu_bpad,
                        bias ? merlin_opu_biaspad : (const int32_t *)0,
                        m, tile, k, ml, tile);
  for (size_t r = 0; r < ml; ++r)
    for (size_t j = 0; j < nl; ++j) c[r * n + j] = cpad[r * tile + j];
}}
#endif

void {spec.func_name}(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,
                      size_t m, size_t n, size_t k)
{{
#ifdef OPU_SCALAR_TILE
  /* The host has no vector unit to ask, so the edge is supplied. Sweeping it is how the tail paths get
   * exercised at every alignment rather than only at the one the device happens to have. */
  const size_t tile = (size_t)OPU_TILE_EDGE;
#else
  size_t tile;
  asm volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(tile));
#endif
  if (tile == 0) return;                      /* no vector unit: nothing this kernel can do */

#ifdef MERLIN_OPU_PAD_PARTIAL_N
  /* There is at most ONE partial column block and every row block shares it, so it is padded here --
   * before any parallel region -- and read-only inside. `padded == 0` means it could not be padded, in
   * which case every tile runs direct, exactly as it did before padding existed. */
  const size_t tail = n % tile;
  int padded = 0;
  if (tail) {{
    padded = {spec.func_name}_pad_tail(b + (n - tail),
                                       bias ? bias + (n - tail) : (const int32_t *)0,
                                       n, k, tail, tile);
  }}
#endif

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < m; i += tile) {{
    for (size_t j = 0; j < n; j += tile) {{
      const size_t ml = (m - i) < tile ? (m - i) : tile;
      const size_t nl = (n - j) < tile ? (n - j) : tile;
#ifdef MERLIN_OPU_PAD_PARTIAL_N
      if (nl < tile && padded) {{
        {spec.func_name}_tile_padded(c + i * n + j, at + i, bias, m, n, k, ml, nl, tile);
        continue;
      }}
#endif
      {spec.func_name}_tile(c + i * n + j, at + i, b + j, bias ? bias + j : (const int32_t *)0,
                            m, n, k, ml, nl);
    }}
  }}
#else
  for (size_t i = 0; i < m; i += tile) {{
    const size_t ml = (m - i) < tile ? (m - i) : tile;
    for (size_t j = 0; j < n; j += tile) {{
      const size_t nl = (n - j) < tile ? (n - j) : tile;
#ifdef MERLIN_OPU_PAD_PARTIAL_N
      if (nl < tile && padded) {{
        {spec.func_name}_tile_padded(c + i * n + j, at + i, bias, m, n, k, ml, nl, tile);
        continue;
      }}
#endif
      {spec.func_name}_tile(c + i * n + j, at + i, b + j, bias ? bias + j : (const int32_t *)0,
                            m, n, k, ml, nl);
    }}
  }}
#endif
}}
"""


def _reduction_loop(fused: str, fused_alt: "str | None", spec: KernelSpec) -> str:
    """The k-loop, rotating the left-operand register across steps when a second one is given.

    Unrolled by two so consecutive steps use different left-operand registers. The point is WHICH
    REGISTER the write following an accumulate targets: the unit releases the accumulate's LHS read-intent
    one iteration before its last read of that element group (see :attr:`KernelSpec.row_vreg_alt`), so a
    write landing in that window is not blocked and the accumulate finishes on clobbered data. Rotating
    means the next step writes a register the in-flight accumulate is not reading.

    The odd step is peeled rather than handled by a predicated tail, so the loop body stays exactly the
    fused block and nothing new appears between an accumulate and the next write.
    """
    operands = ('                 :: [ml] "r"(ml), [nl] "r"(nl), [ap] "r"(ap), [bp] "r"(bp)\n'
                '                 : "t0", "memory");')
    if fused_alt is None:
        return f"""  for (size_t kk = 0; kk < k; ++kk) {{{{
    const int8_t *ap = at + kk * m;
    const int8_t *bp = b + kk * n;
    asm volatile("{fused}"
{operands}
  }}}}"""
    return f"""  size_t kk = 0;
  for (; kk + 2 <= k; kk += 2) {{{{
    {{{{
      const int8_t *ap = at + kk * m;
      const int8_t *bp = b + kk * n;
      asm volatile("{fused}"
{operands}
    }}}}
    {{{{
      const int8_t *ap = at + (kk + 1) * m;
      const int8_t *bp = b + (kk + 1) * n;
      asm volatile("{fused_alt}"
{operands}
    }}}}
  }}}}
  if (kk < k) {{{{                              /* odd reduction length: one step, first register */
    const int8_t *ap = at + kk * m;
    const int8_t *bp = b + kk * n;
    asm volatile("{fused}"
{operands}
  }}}}"""

def emit_reference_c(func_name: str = "opu_gemm_i8_ref") -> str:
    """A scalar C reference with the SAME signature, for a host-side or in-image comparison.

    Accumulates in ``int32_t`` and lets it wrap, matching hardware with no saturation logic. On the int8
    datapath the wrap is unreachable for any realistic reduction length (see
    :mod:`kernels.opu_corpus`), so this agrees with a saturating implementation on every real input — it
    is written this way so it would still agree if that ever stopped being true.
    """
    return f"""\
/* GENERATED by merlin.kernels.opu_kernel — scalar reference, same signature as the kernel. */
#include <stdint.h>
#include <stddef.h>

void {func_name}(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,
                 size_t m, size_t n, size_t k)
{{
  for (size_t i = 0; i < m; ++i) {{
    for (size_t j = 0; j < n; ++j) {{
      int32_t sum = bias ? bias[j] : 0;              /* wraps, as the accumulator does */
      for (size_t kk = 0; kk < k; ++kk)
        sum += (int32_t)at[kk * m + i] * (int32_t)b[kk * n + j];
      c[i * n + j] = sum;
    }}
  }}
}}
"""
