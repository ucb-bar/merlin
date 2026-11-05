"""Emit Iris's int8 GEMM microkernel for the VME/Xsfmm matrix unit, every instruction from a derived table.

Modeled on Miles Rusch's kernels in saturn ``vme`` (``benchmarks/common/vme_gemm.h``: ``vme_gemm_2x2`` and
``vme_gemm_tiled``; ``benchmarks/vme-fused-lm2-gemm-transpose``): a 2x2 super-tile of four TE x TE int32
tiles accumulated in parallel, the k loop consuming ``kmax`` reduction rows per ``sf.mm``, the tiles
zeroed with ``sf.vtzero.t`` and read row by row with ``sf.vtmv.v.t``, the optional per-column bias added on
the way out. Both operands are K-major (``at[k*M + i]``, ``b[k*N + j]``), which is the unit's native
layout: ``sf.mm`` multiplies A^T by B from rows held in vector registers.

What is derived here and what is a choice:

* every instruction word comes from :class:`merlin.targetgen.rtl.matrix_isa.MaskedEncoding` (the unit's
  RTL decode table and the host core's ``BitPat`` declarations, cross-checked against ``vme.h``), passed
  in by the caller. Nothing below spells an opcode or a funct field;
* ``kmax`` (reduction rows per ``sf.mm``) and the two matrix configurations (operand SEW x widen, readout
  SEW x widen) come from the RTL too -- the caller reads them -- and the kernel re-reads ``kmax`` and the
  tile edge from the hardware at run time and takes a scalar path if they disagree;
* the operand LAYOUTS (where the widen code and altfmt sit in ``sf.msetmtype``'s operands, the tile and
  A-sign bits of ``rd``, the tile-subset specifier of ``sf.vtmv.v.t``) are the Zvt draft's operand
  formats, declared in ``contracts/matrix_units.yaml`` with the source each is read from;
* the register allocation (A at v0/v8, B at v16/v24, rows ``8 / kmax`` registers apart) is the kernel's
  choice, as in ``vme_gemm.h``, and is checked against the unit's alignment rule.

Partial tiles: a 2TE x 2TE super-tile that the matrix covers runs the 2x2 path; anything else is split into
TE x TE tiles with ``tm``/``tn`` clamped to what remains (``sf.msettm`` / ``sf.msettn``), and a final reduction
step shorter than ``kmax`` sets ``sf.msettk`` to what remains. Operand rows are loaded at exactly ``tm`` /
``tn`` lanes, so nothing past a panel is read. No shape is refused; a configuration the hardware does not
report as expected takes the scalar path rather than computing with the wrong geometry.

The stand-in twin (``-DMERLIN_MATRIX_SCALAR_TILE -DMERLIN_MATRIX_TILE_EDGE=<te>``) keeps the same tiling
loop and replaces each block body by scalar arithmetic, so it carries none of the unit's instructions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["KERNEL_FUNC", "OperandLayout", "VmeKernelSpec", "emit_kernel"]

#: The kernel's C symbol, with the signature merlin.kernels.matrix_shim expects.
KERNEL_FUNC = "merlin_vme_gemm_i8"

#: The instruction roles the kernel emits, by the contract's role names.
ROLES = ("matmul", "zero", "readout", "release", "set_type", "set_n", "set_m", "set_k")


@dataclass(frozen=True)
class OperandLayout:
    """Where values sit inside the unit's register operands (declared; see contracts/matrix_units.yaml)."""

    mtype_widen_lsb: int
    vtype_sew_lsb: int
    vtype_altfmt_bit: int
    tile_rd_shift: int
    a_signed_rd_bit: int
    tss_tile_lsb: int
    tss_tile_scale: int
    tss_pattern_lsb: int
    tss_row_pattern: int
    #: The matrix-state enable in mstatus (Zvt's MS, like FS/VS for F/V). Software sets it before the first
    #: matrix instruction, as saturn's own crt.S does; a core without the field ignores the write.
    mstatus_ms_lsb: int
    mstatus_ms_width: int


@dataclass(frozen=True)
class VmeKernelSpec:
    """What the emitted kernel needs: the role -> instruction map, the derived configuration, a layout.

    ``operand_type`` / ``readout_type`` are ``(sew_bits, widen)`` pairs read from the RTL's list of matrix
    configurations; their product must be the accumulator width. ``kstep`` is the RTL's ``kmax`` for the
    operand configuration.
    """

    roles: Mapping[str, str]
    operand_type: tuple[int, int]
    readout_type: tuple[int, int]
    kstep: int
    layout: OperandLayout
    acc_bits: int = 32
    operand_bits: int = 8
    a_regs: tuple[int, int] = (0, 8)
    b_regs: tuple[int, int] = (16, 24)
    out_reg: int = 16
    bias_reg: int = 24

    def __post_init__(self) -> None:
        missing = [r for r in ROLES if r not in self.roles]
        if missing:
            raise ValueError(f"kernel roles {missing} are not mapped to instructions")
        sew, widen = self.operand_type
        if sew != self.operand_bits or sew * widen != self.acc_bits:
            raise ValueError(
                f"operand configuration e{sew}w{widen} is not {self.operand_bits}-bit operands accumulating "
                f"in {self.acc_bits} bits"
            )
        rsew, rwiden = self.readout_type
        if rsew * rwiden != self.acc_bits or rwiden != 1:
            raise ValueError(f"readout configuration e{rsew}w{rwiden} does not move {self.acc_bits}-bit rows")
        if self.kstep < 1 or 8 % self.kstep:
            raise ValueError(f"kmax={self.kstep} does not divide a group of 8 registers")
        for base in (*self.a_regs, *self.b_regs):
            # Rows of one sf.mm operand live at base + i * (8 / kmax): the base must open an 8-register
            # window (the unit requires rs % 8 < 8 / kmax, and the rows fill the window).
            if base % 8:
                raise ValueError(f"operand base v{base} does not open an 8-register window")
        if len({*self.a_regs, *self.b_regs}) != 4:
            raise ValueError("the four operand windows must be distinct")

    @property
    def row_stride(self) -> int:
        return 8 // self.kstep

    @property
    def sew_code(self) -> int:
        return _log2(self.operand_type[0] // 8)

    @property
    def readout_sew_code(self) -> int:
        return _log2(self.readout_type[0] // 8)

    @staticmethod
    def widen_code(widen: int) -> int:
        # MatrixConfig.vtwidenEncoding in the host core's Matrix.scala: log2Ceil(twiden) + 1.
        return _log2(widen) + 1


def _log2(v: int) -> int:
    if v < 1 or v & (v - 1):
        raise ValueError(f"{v} is not a power of two")
    return v.bit_length() - 1


def _require(encodings: Mapping[str, Any], spec: VmeKernelSpec) -> dict[str, Any]:
    out = {}
    for role in ROLES:
        name = spec.roles[role]
        if name not in encodings:
            raise ValueError(f"the derived encoding table has no {name!r} (role {role}); refusing to guess it")
        out[role] = encodings[name]
    return out


def _q(lines: list[str]) -> str:
    """One C string literal holding ``lines`` as an asm block."""
    return '"' + "\\n\\t".join(lines) + '"'


def _vclobbers(regs: list[int]) -> str:
    return ", ".join(f'"v{r}"' for r in sorted(set(regs)))


def emit_kernel(encodings: Mapping[str, Any], spec: VmeKernelSpec, *, derivation_ok: bool) -> str:
    """The C source of ``merlin_vme_gemm_i8``. Refuses to emit from a derivation that did not cross-check."""
    if not derivation_ok:
        raise ValueError("the VME encoding derivation did not agree with its cross-check source; refusing to emit")
    enc = _require(encodings, spec)
    lay = spec.layout
    ks, stride = spec.kstep, spec.row_stride
    a0, a1 = spec.a_regs
    b0, b1 = spec.b_regs

    def mm(tile: int, a_base: int, b_base: int) -> str:
        rd = (tile << lay.tile_rd_shift) | (1 << lay.a_signed_rd_bit)
        return enc["matmul"].insn_r(f"x{rd}", f"x{b_base}", f"x{a_base}")

    def zero(tile: int) -> str:
        return enc["zero"].insn_r(f"x{tile << lay.tile_rd_shift}", None, None)

    set_k = enc["set_k"].insn_r("t4", "%[kk]", None)

    def loads(base: int, ptr: str, ld: str, rows: int, tmp: str) -> list[str]:
        out = [f"vle8.v v{base}, ({ptr})"]
        if rows > 1:
            out.append(f"mv {tmp}, {ptr}")
        for i in range(1, rows):
            out += [f"add {tmp}, {tmp}, {ld}", f"vle8.v v{base + i * stride}, ({tmp})"]
        return out

    # 2x2 super-tile reduction steps, one per possible row count (the last step may be short).
    steps_2x2 = []
    for kk in range(1, ks + 1):
        body = []
        body += loads(a0, "%[pa0]", "%[lda]", kk, "t0")
        body += loads(a1, "%[pa1]", "%[lda]", kk, "t1")
        body += loads(b0, "%[pb0]", "%[ldb]", kk, "t2")
        body += loads(b1, "%[pb1]", "%[ldb]", kk, "t3")
        if kk != ks:
            body.append(set_k)
        body += [mm(0, a0, b0), mm(1, a0, b1), mm(2, a1, b0), mm(3, a1, b1)]
        regs = [base + i * stride for base in (a0, a1, b0, b1) for i in range(kk)]
        steps_2x2.append(
            f"""    case {kk}:
      asm volatile({_q(body)}
                   :: [pa0] "r"(pa0), [pa1] "r"(pa1), [pb0] "r"(pb0), [pb1] "r"(pb1),
                      [lda] "r"(lda), [ldb] "r"(ldb), [kk] "r"(kk)
                   : "t0", "t1", "t2", "t3", "t4", {_vclobbers(regs)}, "memory");
      break;"""
        )

    set_n_tm = enc["set_n"].insn_r("t4", "%[tm]", None)
    set_n_tn = enc["set_n"].insn_r("t4", "%[tn]", None)
    steps_1x1 = []
    for kk in range(1, ks + 1):
        body = [set_n_tm, *loads(a0, "%[pa]", "%[lda]", kk, "t0"), set_n_tn, *loads(b0, "%[pb]", "%[ldb]", kk, "t1")]
        body += [set_k, mm(0, a0, b0)]
        regs = [base + i * stride for base in (a0, b0) for i in range(kk)]
        steps_1x1.append(
            f"""    case {kk}:
      asm volatile({_q(body)}
                   :: [pa] "r"(pa), [pb] "r"(pb), [lda] "r"(lda), [ldb] "r"(ldb),
                      [tm] "r"(tm), [tn] "r"(tn), [kk] "r"(kk)
                   : "t0", "t1", "t4", {_vclobbers(regs)}, "memory");
      break;"""
        )

    set_type = enc["set_type"].insn_r("t5", "%[mt]", "%[vt]")
    set_n = enc["set_n"].insn_r("%[r]", "%[n]", None)
    set_m = enc["set_m"].insn_r("%[r]", "%[m]", None)
    set_k_fn = enc["set_k"].insn_r("%[r]", "%[k]", None)
    readout = enc["readout"].insn_r(f"x{spec.out_reg}", "%[tss]", None)
    release = enc["release"].insn_r(None, None, None)
    zeros4 = _q([zero(t) for t in range(4)])
    zero1 = _q([zero(0)])
    out_regs = _vclobbers(list(range(spec.out_reg, spec.out_reg + 8)) + list(range(spec.bias_reg, spec.bias_reg + 8)))

    out_r, bias_r = spec.out_reg, spec.bias_reg
    readout_bias = _q(
        [readout, f"vle32.v v{bias_r}, (%[bp])", f"vadd.vv v{out_r}, v{out_r}, v{bias_r}", f"vse32.v v{out_r}, (%[cp])"]
    )
    readout_plain = _q([readout, f"vse32.v v{out_r}, (%[cp])"])
    op_sew, op_widen = spec.operand_type
    rd_sew, rd_widen = spec.readout_type

    header = "\n".join(
        f" *   {role:<9}: {spec.roles[role]:<13} match={enc[role].match:#010x} mask={enc[role].mask:#010x}"
        + (f" (+{enc[role].fill_mask:#010x} from {enc[role].fill_source})" if enc[role].fill_mask else "")
        for role in ROLES
    )

    return f"""\
/* GENERATED by iris-mlir vme_kernel.py -- do not edit.
 *
 * int8 x int8 -> int32 GEMM on the VME/Xsfmm matrix unit, modeled on saturn vme benchmarks/common/vme_gemm.h.
 * Every .insn word is derived from the unit's RTL decode table and the host core's BitPat declarations and
 * cross-checked against benchmarks/common/vme.h (merlin.targetgen.rtl.matrix_isa):
 *
{header}
 *
 * C[i][j] = bias[j] + sum_k at[k*m + i] * b[k*n + j]   (both operands K-major; bias may be NULL)
 */
#include <stdint.h>
#include <stddef.h>

/* kmax at e{op_sew}w{op_widen}: reduction rows per sf.mm (the RTL's MatrixConfig.kmax). */
#define MERLIN_VME_KSTEP {ks}

/* mtype / vtype operands of sf.msetmtype for the two configurations the RTL accepts here. */
#define MERLIN_VME_OPERAND_MTYPE ((size_t){spec.widen_code(op_widen)} << {lay.mtype_widen_lsb})
/* altfmt set: the B operand is signed */
#define MERLIN_VME_OPERAND_VTYPE \\
  (((size_t){spec.sew_code} << {lay.vtype_sew_lsb}) | ((size_t)1 << {lay.vtype_altfmt_bit}))
#define MERLIN_VME_READOUT_MTYPE ((size_t){spec.widen_code(rd_widen)} << {lay.mtype_widen_lsb})
#define MERLIN_VME_READOUT_VTYPE ((size_t){spec.readout_sew_code} << {lay.vtype_sew_lsb})
/* mstatus.MS: matrix state enabled. Set per hart before the unit is touched (OpenMP workers included). */
#define MERLIN_VME_MSTATUS_MS ((((size_t)1 << {lay.mstatus_ms_width}) - 1) << {lay.mstatus_ms_lsb})

static void merlin_vme_scalar(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,
                              size_t m, size_t n, size_t k, size_t i0, size_t j0, size_t tm, size_t tn)
{{
  for (size_t i = i0; i < i0 + tm; ++i)
    for (size_t j = j0; j < j0 + tn; ++j) {{
      int32_t sum = bias ? bias[j] : 0;
      for (size_t kk = 0; kk < k; ++kk)
        sum += (int32_t)at[kk * m + i] * (int32_t)b[kk * n + j];
      c[i * n + j] = sum;
    }}
}}

#ifndef MERLIN_MATRIX_SCALAR_TILE
static inline void merlin_vme_enable(void)
{{
  asm volatile("csrs mstatus, %0" :: "r"(MERLIN_VME_MSTATUS_MS) : "memory");
}}

/* sf.msetmtype twice, then sf.msettn. The configuration's LMUL is derived from the CURRENT element width,
 * so crossing a width change takes a first (sacrificial) msetmtype to make the new width current
 * (vme.h's vme_vsettnt_*). Returns tn, which is also vl. */
static inline size_t merlin_vme_config(size_t mt, size_t vt, size_t n)
{{
  size_t r;
  asm volatile({_q([set_type, set_type, set_n])}
               : [r] "=&r"(r) : [mt] "r"(mt), [vt] "r"(vt), [n] "r"(n) : "t5", "memory");
  return r;
}}
static inline size_t merlin_vme_set_m(size_t m)
{{
  size_t r;
  asm volatile({_q([set_m])} : [r] "=r"(r) : [m] "r"(m) : "memory");
  return r;
}}
static inline size_t merlin_vme_set_k(size_t k)
{{
  size_t r;
  asm volatile({_q([set_k_fn])} : [r] "=r"(r) : [k] "r"(k) : "memory");
  return r;
}}
static inline void merlin_vme_release(void)
{{
  asm volatile({_q([release])} ::: "memory");
}}

/* Read one tile row out (sf.vtmv.v.t), add the bias slice if any, store. Runs in the readout config. */
static inline void merlin_vme_store_row(int32_t *dst, const int32_t *bias, size_t tile, size_t r)
{{
  const size_t tss = ((tile * {lay.tss_tile_scale}) << {lay.tss_tile_lsb})
                   | ((size_t){lay.tss_row_pattern} << {lay.tss_pattern_lsb}) | r;
  if (bias) {{
    asm volatile({readout_bias}
                 :: [tss] "r"(tss), [bp] "r"(bias), [cp] "r"(dst) : {out_regs}, "memory");
  }} else {{
    asm volatile({readout_plain}
                 :: [tss] "r"(tss), [cp] "r"(dst) : {out_regs}, "memory");
  }}
}}

/* One 2TE x 2TE super-tile: four TE x TE tiles (mt0..mt3) share every A and B row load. */
static void merlin_vme_block_2x2(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,
                                 size_t m, size_t n, size_t k, size_t m0, size_t n0, size_t te)
{{
  merlin_vme_enable();
  merlin_vme_config(MERLIN_VME_OPERAND_MTYPE, MERLIN_VME_OPERAND_VTYPE, te);
  merlin_vme_set_m(te);
  asm volatile({zeros4} ::: "memory");
  merlin_vme_set_k(MERLIN_VME_KSTEP);
  const size_t lda = m, ldb = n;
  for (size_t k0 = 0; k0 < k; k0 += MERLIN_VME_KSTEP) {{
    const size_t kk = (k - k0) < MERLIN_VME_KSTEP ? (k - k0) : MERLIN_VME_KSTEP;
    const int8_t *pa0 = at + k0 * m + m0, *pa1 = pa0 + te;
    const int8_t *pb0 = b + k0 * n + n0, *pb1 = pb0 + te;
    switch (kk) {{
{chr(10).join(steps_2x2)}
    }}
  }}
  merlin_vme_config(MERLIN_VME_READOUT_MTYPE, MERLIN_VME_READOUT_VTYPE, te);
  for (size_t r = 0; r < te; ++r)
    for (size_t t = 0; t < 4; ++t) {{
      const size_t row = m0 + (t >> 1) * te + r, col = n0 + (t & 1) * te;
      merlin_vme_store_row(c + row * n + col, bias ? bias + col : (const int32_t *)0, t, r);
    }}
  merlin_vme_release();
}}

/* One tile of tm x tn (each <= TE): the clamped path for whatever a super-tile does not cover. */
static void merlin_vme_tile_1x1(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,
                                size_t m, size_t n, size_t k, size_t m0, size_t n0, size_t tm, size_t tn)
{{
  merlin_vme_enable();
  merlin_vme_config(MERLIN_VME_OPERAND_MTYPE, MERLIN_VME_OPERAND_VTYPE, tn);
  merlin_vme_set_m(tm);
  asm volatile({zero1} ::: "memory");
  const size_t lda = m, ldb = n;
  for (size_t k0 = 0; k0 < k; k0 += MERLIN_VME_KSTEP) {{
    const size_t kk = (k - k0) < MERLIN_VME_KSTEP ? (k - k0) : MERLIN_VME_KSTEP;
    const int8_t *pa = at + k0 * m + m0, *pb = b + k0 * n + n0;
    switch (kk) {{
{chr(10).join(steps_1x1)}
    }}
  }}
  merlin_vme_config(MERLIN_VME_READOUT_MTYPE, MERLIN_VME_READOUT_VTYPE, tn);
  for (size_t r = 0; r < tm; ++r)
    merlin_vme_store_row(c + (m0 + r) * n + n0, bias ? bias + n0 : (const int32_t *)0, 0, r);
  merlin_vme_release();
}}
#endif

/* The region a super-tile at (m0, n0) owns: full 2x2 when the matrix covers it, else clamped tiles. */
static void merlin_vme_region(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,
                              size_t m, size_t n, size_t k, size_t m0, size_t n0, size_t te)
{{
#ifdef MERLIN_MATRIX_SCALAR_TILE
  const size_t tm = (m - m0) < 2 * te ? (m - m0) : 2 * te;
  const size_t tn = (n - n0) < 2 * te ? (n - n0) : 2 * te;
  merlin_vme_scalar(c, at, b, bias, m, n, k, m0, n0, tm, tn);
#else
  if (m - m0 >= 2 * te && n - n0 >= 2 * te) {{
    merlin_vme_block_2x2(c, at, b, bias, m, n, k, m0, n0, te);
    return;
  }}
  for (size_t i = m0; i < m && i < m0 + 2 * te; i += te)
    for (size_t j = n0; j < n && j < n0 + 2 * te; j += te)
      merlin_vme_tile_1x1(c, at, b, bias, m, n, k, i, j,
                          (m - i) < te ? (m - i) : te, (n - j) < te ? (n - j) : te);
#endif
}}

void {KERNEL_FUNC}(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,
                   size_t m, size_t n, size_t k)
{{
  if (m == 0 || n == 0) return;
#ifdef MERLIN_MATRIX_SCALAR_TILE
  const size_t te = (size_t)MERLIN_MATRIX_TILE_EDGE;
#else
  /* The tile edge and kmax are READ FROM THE HARDWARE, and a unit that does not report the kmax this
   * kernel's register layout was generated for takes the scalar path rather than a wrong layout. */
  merlin_vme_enable();
  const size_t te = merlin_vme_config(MERLIN_VME_OPERAND_MTYPE, MERLIN_VME_OPERAND_VTYPE, ~(size_t)0);
  if (te == 0 || k == 0 || merlin_vme_set_k(MERLIN_VME_KSTEP) != MERLIN_VME_KSTEP) {{
    merlin_vme_scalar(c, at, b, bias, m, n, k, 0, 0, m, n);
    return;
  }}
#endif
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (size_t m0 = 0; m0 < m; m0 += 2 * te)
    for (size_t n0 = 0; n0 < n; n0 += 2 * te)
      merlin_vme_region(c, at, b, bias, m, n, k, m0, n0, te);
}}
"""
