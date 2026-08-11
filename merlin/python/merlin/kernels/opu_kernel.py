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
        for name, reg in (("row_vreg", self.row_vreg), ("col_vreg", self.col_vreg)):
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

    # The fused block, one instruction per line so the emitted asm is readable in a disassembly diff.
    fused = "\\n\\t".join([
        # Zero every row lane at the maximum length, so lanes past the real panel hold 0 rather than
        # whatever the register happened to contain.
        "vsetvli t0, zero, e8, m1, ta, ma",
        f"vmv.v.i v{spec.row_vreg}, 0",
        # Row operand: `ml` lanes, tail UNDISTURBED so the zeros above survive.
        "vsetvli zero, %[ml], e8, m1, tu, ma",
        f"vle8.v v{spec.row_vreg}, (%[ap])",
        # Column operand: `nl` lanes. This is also the length the accumulate runs at.
        "vsetvli zero, %[nl], e8, m1, ta, ma",
        f"vle8.v v{spec.col_vreg}, (%[bp])",
        acc.insn_r(md, vr, vc),
    ])

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
   * The broadcast is issued under the OPERAND vtype (e{spec.operand_bits}/m1), which is what the unit's
   * own expert kernel does. This is not a style choice: MEASURED on the RTL, the same instruction under
   * e{spec.acc_bits}/m1 HANGS the core -- no trap, no retire, the simulation spins forever -- because that
   * vl cannot span a tile row. Under e{spec.acc_bits}/m{spec.acc_lmul} it completes, so either the operand
   * vtype or the correctly-grouped accumulator vtype is safe, and only the under-provisioned one is not.
   *
   * The int32 data still moves with an explicit-width `vle32.v`, which takes its element width from the
   * opcode rather than from vtype, so the load is unaffected by the surrounding e{spec.operand_bits}.
   * Its EMUL is {spec.acc_lmul}, so it writes the whole accumulator group
   * v{spec.out_vreg}..v{spec.out_vreg + spec.acc_lmul - 1} -- which is why the zero-init below has to
   * cover the group and not just its first register. */
  if (bias) {{
    asm volatile("vsetvli zero, %[nl], e{spec.operand_bits}, m1, ta, ma\\n\\t"
                 "vle32.v v{spec.out_vreg}, (%[bp])\\n\\t"
                 "{bcast.insn_r(md, _x(0), vo)}"
                 :: [nl] "r"(nl), [bp] "r"(bias)
                 : "memory");
  }} else {{
    /* Zero the whole group at the accumulator vtype, then drop back to the operand vtype to broadcast. */
    asm volatile("vsetvli zero, %[nl], e{spec.acc_bits}, m{spec.acc_lmul}, ta, ma\\n\\t"
                 "vmv.v.i v{spec.out_vreg}, 0\\n\\t"
                 "vsetvli zero, %[nl], e{spec.operand_bits}, m1, ta, ma\\n\\t"
                 "{bcast.insn_r(md, _x(0), vo)}"
                 :: [nl] "r"(nl));
  }}

  /* One fused block per reduction step. Configure, load, configure, load, accumulate -- all inside a
   * single asm volatile, so the vector-length insertion pass cannot reinterpret either length. */
  for (size_t kk = 0; kk < k; ++kk) {{
    const int8_t *ap = at + kk * m;
    const int8_t *bp = b + kk * n;
    asm volatile("{fused}"
                 :: [ml] "r"(ml), [nl] "r"(nl), [ap] "r"(ap), [bp] "r"(bp)
                 : "t0", "memory");
  }}

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
 * there is no separate tail path that could be right in the full case and wrong at the edge. */
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

  for (size_t i = 0; i < m; i += tile) {{
    const size_t ml = (m - i) < tile ? (m - i) : tile;
    for (size_t j = 0; j < n; j += tile) {{
      const size_t nl = (n - j) < tile ? (n - j) : tile;
      {spec.func_name}_tile(c + i * n + j, at + i, b + j, bias ? bias + j : (const int32_t *)0,
                            m, n, k, ml, nl);
    }}
  }}
}}
"""


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
