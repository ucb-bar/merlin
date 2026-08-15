"""Self-contained MX reference-kernel emission (fp8 / fp6 / fp4).

Bake a block-scaled MX matmul's operand codes + E8M0 block scales into a self-contained C++ kernel that
drives cyclotron's MX-Gemmini co-model through ``mxgemm_lib.hpp`` and prints the result via the OUT
protocol (:func:`muon.parse_output`). The tile is padded to 128x128x128 with the real operands in the
top-left corner (the co-model mishandles a lone sub-byte PE tile); padding contributes ~0 to the graded
top-left output (padding codes = 0, K-group padding scales = e8m0 code 0 -> 2**-127).

IMPORTANT provenance note: the operand CODES and the E8M0 block SCALES are corpus-seeded (the scales are a
deterministic function of the capsule-name salt, NOT amax-derived from the operand values -- see
``corpus_operands.e8m0_scale_codes``), so they live only in the capsule golden and cannot be reconstructed
from the decoded-float workload a general backend receives. This emitter therefore reads them from the
golden's ``operand_codes`` bundle (attached to the cb as ``mx_operands`` by the grading runner). That is a
PUBLIC-capsule reference path (the golden is masked for hidden capsules); it is the known-good baseline, not
a general compiler capability. See memory ``radiance-launch-tooling-gap``.
"""
from __future__ import annotations

from typing import Any

_PADW = 128
_GROUP = 32
# golden ``fmt`` token -> the ``GemmDatatype`` enumerator the co-model config takes.
_DTYPE = {"fp8": "FP8", "fp6": "FP6", "fp4": "FP4"}


def is_mx_cb(cb: dict) -> bool:
    """True when any operand tensor is a microscaling float (fp8 E4M3/E5M2, fp6 E3M2, fp4 E2M1) — the block
    scaled MX-PE datapath, which needs the co-model kernel rather than the fp32 LLVM-dialect nest."""
    for t in (cb.get("tensors") or {}).values():
        dt = str(t.get("dtype", ""))
        if dt.startswith("f8E") or dt.startswith("f6E") or dt.startswith("f4E"):
            return True
    return False


def _short_fmt(fmt: str) -> str:
    """golden ``fmt`` (``fp8_e4m3`` / ``fp6_e3m2`` / ``fp4_e2m1``) -> the ``fp8``/``fp6``/``fp4`` short key."""
    for k in _DTYPE:
        if fmt.startswith(k):
            return k
    raise MxCodegenError(f"unsupported MX operand format {fmt!r}")


class MxCodegenError(Exception):
    """An MX capsule that the reference emitter cannot bake (missing operand codes / unknown format)."""


def _emit2d(ctype: str, name: str, rows: int, cols: int, data) -> str:
    lines = [f"static const {ctype} {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        lines.append("  {" + ",".join(str(int(x)) for x in data[r]) + "},")
    lines.append("};\n")
    return "\n".join(lines)


def _pack96(codes16) -> list[int]:
    """16 six-bit fp6 palette codes -> three LE uint32 (96 bits), matching the co-model ``unpack_lut_96bit``."""
    v = 0
    for i, c in enumerate(codes16):
        v |= (int(c) & 0x3F) << (6 * i)
    return [v & 0xFFFFFFFF, (v >> 32) & 0xFFFFFFFF, (v >> 64) & 0xFFFFFFFF]


def _data_header(mx: dict) -> str:
    """The operand-data header: A/B codes + E8M0 scales (+ fp6 LUTs) padded into a 128-tile, top-left real."""
    fmt = _short_fmt(mx["fmt"])
    is_fp8, is_fp6 = fmt == "fp8", fmt == "fp6"
    m, n, k = int(mx["M"]), int(mx["N"]), int(mx["K"])
    a, b = mx["A_bytes"], mx["B_bytes"]
    sa, sb = mx["SA"], mx["SB"]
    gk = _PADW // _GROUP
    parts = [f"#define MATMUL_M {_PADW}", f"#define MATMUL_K {_PADW}", f"#define MATMUL_N {_PADW}",
             f"#define MATMUL_GK {gk}", f"#define MATMUL_GN {gk}", ""]
    if is_fp8:                                             # one byte per element: A[M,K], B[K,N]
        ain = [[0] * _PADW for _ in range(_PADW)]
        for i in range(m):
            for j in range(k):
                ain[i][j] = a[i * k + j]
        bin_ = [[0] * _PADW for _ in range(_PADW)]
        for kk in range(k):
            for j in range(n):
                bin_[kk][j] = b[kk * n + j]
        parts.append(_emit2d("uint8_t", "A_in", _PADW, _PADW, ain))
        parts.append(_emit2d("uint8_t", "B_in", _PADW, _PADW, bin_))
    else:                                                 # sub-byte: A packed [M/2,K] (nibble along M), B [K,N/2]
        ah = [[0] * _PADW for _ in range(_PADW // 2)]
        for r in range(m // 2):
            for kk in range(k):
                ah[r][kk] = a[r * k + kk]
        bh = [[0] * (_PADW // 2) for _ in range(_PADW)]
        for kk in range(k):
            for c in range(n // 2):
                bh[kk][c] = b[kk * (n // 2) + c]
        parts.append(_emit2d("uint8_t", "A_in_hw", _PADW // 2, _PADW, ah))
        parts.append(_emit2d("uint8_t", "B_in", _PADW, _PADW // 2, bh))
    # scales [GK][PADW]: real K-group 0 in cols 0..M/N-1, 0x7f (scale 1.0) in the discarded M/N padding;
    # padding K-groups 1..GK-1 stay code 0 -> 2**-127 so their (zero-code) contribution cannot perturb.
    as_ = [[0] * _PADW for _ in range(gk)]
    bs = [[0] * _PADW for _ in range(gk)]
    for i in range(_PADW):
        as_[0][i] = sa[0][i] if i < m else 0x7F
    for j in range(_PADW):
        bs[0][j] = sb[0][j] if j < n else 0x7F
    parts.append(_emit2d("uint8_t", "A_scales_row", gk, _PADW, as_))
    parts.append(_emit2d("uint8_t", "B_scales_col", gk, _PADW, bs))
    if is_fp6:                                            # one 16-entry palette replicated to every LUT slot
        pa, pb = _pack96(mx["lutA"][0]), _pack96(mx["lutB"][0])
        n_lut = _PADW // 2
        parts.append(_emit2d("uint32_t", "A_lut", n_lut, 3, [pa] * n_lut))
        parts.append(_emit2d("uint32_t", "B_lut", n_lut, 3, [pb] * n_lut))
        parts.append(_emit2d("uint32_t", "C_lut", n_lut, 3, [[0, 0, 0]] * n_lut))
    return "\n".join(parts)


def _putchars(s: str) -> str:
    return "".join(f"vx_putchar({ord(c)}); " for c in s)


def emit_mx_kernel(mx: dict, out_name: str) -> str:
    """Render the self-contained MX kernel: the data header + the ``mxgemm<CFG>`` driver + OUT-protocol print
    of the ``M x N`` top-left result. ``mx`` is the golden operand bundle (fmt / M / N / K / A_bytes / B_bytes
    / SA / SB / lutA / lutB)."""
    fmt = _short_fmt(mx["fmt"])
    is_fp6 = fmt == "fp6"
    rows, cols = int(mx["M"]), int(mx["N"])
    data = _data_header(mx)
    # fp8/fp4 leave the LUTs unused -> zeroed placeholders; fp6's header already defines them (uint32[.][3]).
    lut_decls = "" if is_fp6 else (
        "static const uint8_t A_lut[64][16] = {0};\n"
        "static const uint8_t B_lut[64][16] = {0};\n"
        "static const uint8_t C_lut[64][16] = {0};\n")
    unify = "static const uint8_t *A_in = &A_in_hw[0][0];" if fmt != "fp8" else ""
    prefix = _putchars(f"OUT {out_name} {rows} {cols}")
    return f"""// @generated self-contained MX reference kernel ({fmt}); drives cyclotron MX-Gemmini co-model.
#include <stdint.h>
#include <mu_schedule.h>
#include <mu_intrinsics.h>
#include "gemmini_abs_shim.h"

{data}

// unify naming for A_in (sub-byte headers expose A_in_hw)
{unify}
{lut_decls}#include "mxgemm_lib.hpp"

extern "C" void vx_putchar(int c);
namespace {{
inline void put_u32(unsigned v) {{
  char b[12]; int n = 0;
  if (!v) {{ vx_putchar('0'); return; }}
  while (v) {{ b[n++] = (char)('0' + (v % 10u)); v /= 10u; }}
  while (n) vx_putchar(b[--n]);
}}
inline void put_f32(float f) {{
  if (f != f) {{ vx_putchar('n'); vx_putchar('a'); vx_putchar('n'); return; }}
  if (f < 0) {{ vx_putchar('-'); f = -f; }}
  unsigned ip = (unsigned)f; float rem = f - (float)ip;
  put_u32(ip); vx_putchar('.');
  for (int i = 0; i < 6; i++) {{ rem *= 10.0f; unsigned d = (unsigned)rem; vx_putchar((int)('0' + (d % 10u))); rem -= (float)d; }}
}}
inline float bf16f(unsigned bf) {{ union {{ unsigned u; float f; }} x; x.u = (bf & 0xffffu) << 16; return x.f; }}
}}  // namespace

constexpr GemmConfig CFG{{
    .TILE_M = {_PADW}, .TILE_N = {_PADW}, .TILE_K = {_PADW},
    .DATATYPE = GemmDatatype::{_DTYPE[fmt]}, .QUANT_OUTPUT = false,
}};

void mxgemm_entry(void *a, uint32_t tid, uint32_t th, uint32_t tb) {{
    auto Cg = reinterpret_cast<uint8_t *>(0x40000000);
    mxgemm<CFG>(CFG.TILE_M, CFG.TILE_N, CFG.TILE_K, Cg, tid, th, tb);
    if (tid == 0 && tb == 0) {{
        gemmini_fence();
        volatile uint32_t *C32 = reinterpret_cast<volatile uint32_t *>(0x40000000);
        {prefix}
        for (int i = 0; i < {rows}; i++)
            for (int j = 0; j < {cols}; j++) {{
                unsigned k = (unsigned)(i * {_PADW} + j);
                unsigned w = C32[k >> 1];
                unsigned bf = (k & 1) ? (w >> 16) : (w & 0xffffu);
                vx_putchar(' '); put_f32(bf16f(bf));
            }}
        vx_putchar('\\n');
        vx_putchar('D'); vx_putchar('O'); vx_putchar('N'); vx_putchar('E'); vx_putchar('\\n');
    }}
}}

int main() {{ mu_schedule(mxgemm_entry, nullptr, 2); return 0; }}
"""


def mx_output_name(cb: dict) -> str:
    """The graded output tensor name — the COMMIT dst, else a declared output-role tensor, else ``Y0``."""
    for c in cb.get("commands", []):
        if (c.get("opcode") or "").upper() == "COMMIT":
            dst = (c.get("operands") or {}).get("dst")
            if dst:
                return dst
    for name, spec in (cb.get("tensors") or {}).items():
        if spec.get("role") == "output":
            return name
    outs = cb.get("outputs")
    return outs[0] if outs else "Y0"
