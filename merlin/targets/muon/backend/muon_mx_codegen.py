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
    from merlin.common import quant_formats as _qf
    for t in (cb.get("tensors") or {}).values():
        dt = str(t.get("dtype", ""))
        # DERIVED: the format registry knows which formats are block-scaled (kind "mx_block", scale
        # "block_e8m0"). The prefix test below misses every `mxfp8`/`mxfp6`/`mxfp4` spelling -- the one
        # the capsules themselves are named after -- and matches `f8E4M3FN`, which the registry classes as
        # per-tensor OCP fp8 and not block-scaled at all. Kept as a union so nothing that resolved before
        # stops resolving; the registry is what actually decides.
        try:
            _f = _qf.get(dt)
        except Exception:  # noqa: BLE001 — unknown spelling: fall through to the legacy prefixes
            _f = None
        if _f is not None and (getattr(_f, "kind", None) == "mx_block"
                               or getattr(getattr(_f, "scale", None), "kind", None) == "block_e8m0"):
            return True
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
    # scales [GK][PADW]: the ``gk_real = K/32`` real K-groups carry the golden scales (one row of ``sa``/``sb``
    # per group) in cols 0..M/N-1, 0x7f (scale 1.0) in the discarded M/N padding; the K-padding groups
    # ``gk_real..GK-1`` stay code 0 -> 2**-127 so their (zero-code) contribution cannot perturb the result.
    gk_real = max(1, k // _GROUP)
    as_ = [[0] * _PADW for _ in range(gk)]
    bs = [[0] * _PADW for _ in range(gk)]
    for g in range(min(gk_real, len(sa))):
        for i in range(_PADW):
            as_[g][i] = sa[g][i] if i < m else 0x7F
    for g in range(min(gk_real, len(sb))):
        for j in range(_PADW):
            bs[g][j] = sb[g][j] if j < n else 0x7F
    parts.append(_emit2d("uint8_t", "A_scales_row", gk, _PADW, as_))
    parts.append(_emit2d("uint8_t", "B_scales_col", gk, _PADW, bs))
    if is_fp6:                                            # one 16-entry palette replicated to every LUT slot
        pa, pb = _pack96(mx["lutA"][0]), _pack96(mx["lutB"][0])
        n_lut = _PADW // 2
        parts.append(_emit2d("uint32_t", "A_lut", n_lut, 3, [pa] * n_lut))
        parts.append(_emit2d("uint32_t", "B_lut", n_lut, 3, [pb] * n_lut))
        parts.append(_emit2d("uint32_t", "C_lut", n_lut, 3, [[0, 0, 0]] * n_lut))
    return "\n".join(parts)


def _assemble_batched(mx: dict) -> dict:
    """Pack a ``B``-way batched MX matmul (each batch ``[M,H]@[H,N]``, stacked to ``[B*M, N]``) into ONE
    block-diagonal MX tile: batch ``b`` occupies rows ``b*M..`` and K-group ``b`` (cols ``b*H..``), off-blocks
    zero. A single tile then reproduces every batch (batch ``b``'s output rows read only its own K-group; the
    zero off-blocks contribute nothing), so the existing single-tile path emits it. fp8 only (byte/element)."""
    if _short_fmt(mx["fmt"]) != "fp8":
        raise MxCodegenError("batched MX packing is implemented for fp8 operands only")
    B, m, h, n = int(mx["B"]), int(mx["M"]), int(mx["H"]), int(mx["N"])
    md, kd = B * m, B * h                                  # block-diagonal tile dims
    a_bd = [0] * (md * kd)
    b_bd = [0] * (kd * n)
    sa_bd = [[0x7F] * md for _ in range(B)]
    sb_bd = [[0x7F] * n for _ in range(B)]
    for bi, batch in enumerate(mx["batches"]):
        a, w, sa, sb = batch["A_bytes"], batch["W_bytes"], batch["SA"], batch["SB"]
        for i in range(m):                                # A block at rows bi*m.., cols bi*h..
            for j in range(h):
                a_bd[(bi * m + i) * kd + (bi * h + j)] = a[i * h + j]
        for kk in range(h):                               # W block stacked at rows bi*h..
            for j in range(n):
                b_bd[(bi * h + kk) * n + j] = w[kk * n + j]
        for i in range(m):                                # row scales live in this batch's K-group only
            sa_bd[bi][bi * m + i] = int(sa[i])
        for j in range(n):
            sb_bd[bi][j] = int(sb[j])
    return {"fmt": mx["fmt"], "M": md, "N": n, "K": kd, "G": 0,
            "A_bytes": a_bd, "B_bytes": b_bd, "SA": sa_bd, "SB": sb_bd, "lutA": None, "lutB": None}


def _putchars(s: str) -> str:
    return "".join(f"vx_putchar({ord(c)}); " for c in s)


# --------------------------------------------------------------------------------------------------
# Fused MX flash-attention: wrap the proven radiance-kernels fused kernel (kernels/
# flash_attention_mx_stable/kernel.cpp, ``FULL_ATTN2``) by baking an ``fa_data.h`` from the golden's
# attention_codes. The kernel runs QK (block-scaled MX) -> on-device softmax (the ``fexp.h`` hardware
# exp, which cyclotron models as ``bf16(f32(bf16(x)).exp())`` == the golden's numpy bf16 row-softmax)
# -> per-row E8M0 requant of P -> PV (block-scaled MX) -> finalize. Two fixes vs the raw kernel:
#   (1) STRUCTURAL: at these small (sub-tuned) shapes the ``PVF`` matmul accumulates onto the stale QK
#       result still resident at ``SPAD_DEST`` (finalize_O reads O_unnorm there). We zero the
#       ``[FA_SQ][FA_D]`` C-region between the scale-pack barrier and ``mxgemm_compute_tile<PVF>``.
#   (2) OUTPUT: replace ``main`` with an OUT-protocol print of the first ``Dv`` columns of ``O_GMEM``.
# fp8 track only (byte-per-element operands); fp6/fp4 need the sub-byte sibling kernels.
# --------------------------------------------------------------------------------------------------
_FLASH_ANCHOR = "mu_fence_smem(); BAR_PAD3(); mu_barrier(3, wpb); BAR_PAD3(); MARK();  // 6: pack+bar3"


def _bf16_code(x: float) -> int:
    import struct
    return (struct.unpack("<I", struct.pack("<f", float(x)))[0] >> 16) & 0xFFFF


def _emit2d_hex(ctype: str, name: str, dims: str, rows) -> str:
    w = 2 if ctype == "uint8_t" else 4
    body = ",\n".join(
        "  { " + ", ".join(f"0x{int(v) & (0xFF if w == 2 else 0xFFFF):0{w}x}" for v in r) + " }"
        for r in rows)
    return f"static const {ctype} {name}{dims} = {{\n{body}\n}};\n\n"


def _flash_fp8_fa_data(mx: dict) -> str:
    """Render ``fa_data.h`` contents (byte-per-element fp8) from the golden's attention_codes, in the
    radiance-kernels ``gen_data.py`` layout. FA_D = H (QK contraction); V's value dim ``Dv`` is padded
    with zero columns to FA_D so the single-FA_D kernel expresses ``Dv != H`` (only cols 0..Dv-1 graded)."""
    m, h, skv, dv = int(mx["M"]), int(mx["H"]), int(mx["Skv"]), int(mx["Dv"])
    qk, pv = mx["qk_stage"], mx["pv_stage"]
    a, b, vb = qk["A_bytes"], qk["B_bytes"], pv["B_bytes"]        # Q[M,H], K^T[H,Skv], V[Skv,Dv]
    sa_q, sb_k, sb_v = mx["SA_q"], mx["SB_k"], mx["SB_v"]
    fa_sq, fa_sk, fa_d = m, skv, h
    fa_gk, fa_gkv = h // _GROUP, skv // _GROUP
    a_in = [[a[i * h + j] for j in range(h)] for i in range(m)]
    b_in = [[b[k * skv + n] for n in range(skv)] for k in range(h)]
    a_scales = [[sa_q[g * m + i] for i in range(m)] for g in range(fa_gk)]      # SA_q laid [H/32][M]
    b_scales = [[sb_k[g * skv + n] for n in range(skv)] for g in range(fa_gk)]  # SB_k laid [H/32][Skv]
    v_in = [[(vb[k * dv + j] if j < dv else 0) for j in range(fa_d)] for k in range(skv)]
    v_scales = [[(sb_v[g * dv + j] if j < dv else 0x7F) for j in range(fa_d)] for g in range(fa_gkv)]
    softcap = mx.get("softcap")
    # softcap capsules apply cap*tanh(bf16(S*att_scale)/cap) in an injected pre-softmax loop, so the
    # softmax's own scale is set to 1.0 (the att_scale is consumed inside the softcap). Non-softcap
    # capsules keep the softmax scale = bf16(att_scale).
    softmax_scale = 1.0 if softcap is not None else mx["att_scale"]
    parts = ["#ifndef FA_DATA_H", "#define FA_DATA_H", "#include <stdint.h>", "",
             f"#define FA_SQ {fa_sq}", f"#define FA_SK {fa_sk}", f"#define FA_D {fa_d}",
             f"#define FA_GK {fa_gk}", f"#define FA_GKV {fa_gkv}",
             f"#define FA_BK {fa_sk}", "#define FA_NBLK 1", f"#define FA_GKB {fa_sk // _GROUP}",
             f"#define FA_SOFTMAX_SCALE_BF16 0x{_bf16_code(softmax_scale):04x}",
             f"#define FA_DV_REAL {dv}"]
    if softcap is not None:
        cap = float(softcap)
        parts += [f"#define FA_SOFTCAP_ATT_BF16 0x{_bf16_code(mx['att_scale']):04x}",
                  f"#define FA_SOFTCAP_2OVERCAP_BF16 0x{_bf16_code(2.0 / cap):04x}",
                  f"#define FA_SOFTCAP_CAP_BF16 0x{_bf16_code(cap):04x}"]
    parts.append("")
    parts.append(_emit2d_hex("uint8_t", "QK_A_in", "[FA_SQ][FA_D]", a_in))
    parts.append(_emit2d_hex("uint8_t", "QK_B_in", "[FA_D][FA_SK]", b_in))
    parts.append(_emit2d_hex("uint8_t", "QK_A_scales_row", "[FA_GK][FA_SQ]", a_scales))
    parts.append(_emit2d_hex("uint8_t", "QK_B_scales_col", "[FA_GK][FA_SK]", b_scales))
    parts.append(_emit2d_hex("uint8_t", "V_in", "[FA_SK][FA_D]", v_in))
    parts.append(_emit2d_hex("uint8_t", "V_scales", "[FA_GKV][FA_D]", v_scales))
    parts.append(_emit2d_hex("uint16_t", "QK_S_bf16", "[FA_SQ][FA_SK]", [[0] * fa_sk for _ in range(fa_sq)]))
    parts.append(_emit2d_hex("uint16_t", "O_ref_bf16", "[FA_SQ][FA_D]", [[0] * fa_d for _ in range(fa_sq)]))
    parts.append(_emit2d_hex("uint16_t", "O_mx_bf16", "[FA_SQ][FA_D]", [[0] * fa_d for _ in range(fa_sq)]))
    parts.append(_emit2d_hex("uint8_t", "QK_B_blocks", "[FA_NBLK*FA_D][FA_BK]", b_in))
    parts.append(_emit2d_hex("uint8_t", "QK_B_scales_blocks", "[FA_NBLK*FA_GK][FA_BK]", b_scales))
    parts.append("#endif  // FA_DATA_H\n")
    return "\n".join(parts)


def _flash_kernel_dir(subdir: str = "flash_attention_mx_stable"):
    """The on-disk radiance-kernels flash kernel dir (the fused reference we wrap). Fail closed if the
    kernels repo is not reachable in this environment (masked/agentic path) — never a baked default."""
    from . import muon
    kd = muon.radiance_kernels_root() / "kernels" / subdir
    if not (kd / "kernel.cpp").is_file():
        raise MxCodegenError(f"flash reference kernel not found: {kd / 'kernel.cpp'}")
    return kd


def _flash_out_main(out_name: str) -> str:
    """OUT-protocol ``main``: run ``fa_entry`` (3 warps) then thread-0 prints the first ``FA_DV_REAL``
    columns of each ``O_GMEM`` row as ``OUT <name> FA_SQ FA_DV_REAL <vals...>`` so the grader compares an
    ``[M][Dv]`` tensor (the padded value columns are skipped)."""
    prefix = _putchars(f"OUT {out_name}")
    return r'''
extern "C" void vx_putchar(int c);
namespace {
inline void _pu(unsigned v){char b[12];int n=0;if(!v){vx_putchar('0');return;}while(v){b[n++]=(char)('0'+v%10u);v/=10u;}while(n)vx_putchar(b[--n]);}
inline void _pf(float f){if(f!=f){vx_putchar('n');vx_putchar('a');vx_putchar('n');return;}if(f<0){vx_putchar('-');f=-f;}unsigned ip=(unsigned)f;float r=f-(float)ip;_pu(ip);vx_putchar('.');for(int i=0;i<6;i++){r*=10.f;unsigned d=(unsigned)r;vx_putchar((int)('0'+d%10u));r-=(float)d;}}
inline float _bf(unsigned bf){union{unsigned u;float f;}x;x.u=(bf&0xffffu)<<16;return x.f;}
}
static void _flash_print(void*, uint32_t tid, uint32_t th, uint32_t tb){
  if(tid==0 && tb==0){
    volatile uint32_t *O32 = reinterpret_cast<volatile uint32_t*>(O_GMEM);
    ''' + prefix + r'''
    vx_putchar(' ');_pu(FA_SQ);vx_putchar(' ');_pu(FA_DV_REAL);
    for(int i=0;i<FA_SQ;i++)
      for(int j=0;j<FA_DV_REAL;j++){
        unsigned k=(unsigned)(i*FA_D+j);
        unsigned w=O32[k>>1];
        unsigned bf=(k&1)?(w>>16):(w&0xffffu);
        vx_putchar(' ');_pf(_bf(bf));
      }
    vx_putchar('\n');
    vx_putchar('D');vx_putchar('O');vx_putchar('N');vx_putchar('E');vx_putchar('\n');
  }
}
int main(){ mu_schedule(fa_entry, nullptr, 3); mu_schedule(_flash_print, nullptr, 1); return 0; }
'''


def _emit_flash_kernel(mx: dict, out_name: str) -> str:
    """Bake the fused MX flash-attention reference kernel for the fp8 golden bundle ``mx``. Returns a
    self-contained C++ TU carrying a ``// mu-extra-include: <dir>`` directive (:func:`muon.compile_kernel`
    honors it) so the wrapped kernel's ``mxgemm_core.hpp`` / ``flash_mx_impl.hpp`` resolve."""
    if _short_fmt(mx["fmt"]) != "fp8":
        # The fp8 wrap needs a sub-byte-packed fused flash kernel for fp6/fp4, and none exists in
        # radiance-kernels: flash_attention_mx_fp6 is a single-matmul activation-quantizer DEMO (no
        # QK/softmax/PV chain), and flash_attention_mx_gemma is itself fp8 (Gemma-2 softcap+window),
        # not fp4. Fail closed rather than mis-emit a byte-per-element kernel over sub-byte operands.
        raise MxCodegenError(
            f"flash-attention MX wrap is fp8-only; {mx['fmt']!r} (sub-byte fp6/fp4) has no fused "
            f"flash sibling kernel (flash_attention_mx_fp6 is a matmul-only demo; ...gemma is fp8)")
    kd = _flash_kernel_dir()
    src = (kd / "kernel.cpp").read_text(encoding="utf-8")
    inc = '#include "include/fa_data.h"'
    if inc not in src:
        raise MxCodegenError("flash kernel.cpp layout changed: fa_data include anchor missing")
    src = src.replace(inc, _flash_fp8_fa_data(mx))
    if _FLASH_ANCHOR not in src:
        raise MxCodegenError("flash kernel.cpp layout changed: scale-pack barrier anchor missing")
    # STRUCTURAL FIX: zero the [FA_SQ][FA_D] SPAD_DEST C-region (== S_SMEM) before the PVF matmul so
    # finalize_O reads (P@V), not (stale_S + P@V). See the fused-flash note above.
    zero_c = (_FLASH_ANCHOR +
              "\n    for (uint32_t _zz = tid; _zz < (FA_SQ*FA_D)/2u; _zz += thr)"
              " reinterpret_cast<__shared uint32_t*>(S_SMEM)[_zz] = 0u;"
              "\n    mu_fence_smem(); mu_barrier(3, wpb);")
    src = src.replace(_FLASH_ANCHOR, zero_c, 1)
    # SOFTCAP: inject cap*tanh(bf16(S*att_scale)/cap) over S_SMEM right after the post-QK barrier and
    # before online_softmax_block (which then runs with scale 1.0). tanh via fexp only (no HW tanh):
    #   t = (fexp(2y)-1)/(fexp(2y)+1),  y = St/cap.  Mirrors kernel_model._softcap exactly.
    if mx.get("softcap") is not None:
        bar2 = "mu_fence_smem(); BAR_PAD2(); mu_barrier(2, wpb); BAR_PAD2(); MARK();  // 3: bar2"
        if bar2 not in src:
            raise MxCodegenError("flash kernel.cpp layout changed: post-QK bar2 anchor missing (softcap)")
        softcap_loop = bar2 + r"""
    {   // @softcap: S <- cap*tanh(bf16(S*att_scale)/cap); online_softmax then uses scale 1.0.
        // EXACT transcription of radiance-kernels flash_attention_mx_gemma bf16_softcap (the validated
        // Gemma-2 primitive): x=s/cap; e=exp(-2|x|) (nonpositive fexp arg -> stable/precise);
        // t=(1-e)/(1+e); cap*sign(x)*t.  All bf16 (fexp + a few bf16 ops + one bounded bf16 divide).
        volatile __shared uint16_t *_Sh = reinterpret_cast<volatile __shared uint16_t *>(S_SMEM);
        const _Float16 _att = as_bf16(FA_SOFTCAP_ATT_BF16);
        const _Float16 _inv2 = as_bf16(FA_SOFTCAP_2OVERCAP_BF16);
        const _Float16 _cap = as_bf16(FA_SOFTCAP_CAP_BF16);
        for (uint32_t _i = tid; _i < (uint32_t)(FA_SQ*FA_SK); _i += thr) {
            _Float16 _x = (_Float16)(as_bf16(_Sh[_i]) * _att);            // scaled score
            _Float16 _p = mu_fexp((_Float16)(_x * _inv2));                // e^{2y}, y=s/cap, bf16
            // tanh(y)=(p-1)/(p+1) then *cap, in fp32 (bf16 divide + abs/sign-branches miscompile on this
            // SIMT target; the plain (p-1)/(p+1) needs no sign handling). RNE bf16 at the end.
            float _pf = (float)_p;
            float _tf = (_pf - 1.0f) / (_pf + 1.0f);
            _Sh[_i] = __builtin_bit_cast(uint16_t, (_Float16)((float)_cap * _tf));
        }
        mu_fence_smem(); mu_barrier(2, wpb);
    }"""
        src = src.replace(bar2, softcap_loop, 1)
    idx = src.rfind("int main()")
    if idx <= 0:
        raise MxCodegenError("flash kernel.cpp layout changed: main() not found")
    src = src[:idx] + _flash_out_main(out_name)
    return (f"// mu-extra-include: {kd}\n"
            f"// @generated fused MX flash-attention reference kernel (fp8); wraps FULL_ATTN2.\n"
            f"#define FULL_ATTN2\n{src}")


def emit_mx_kernel(mx: dict, out_name: str) -> str:
    """Render the self-contained MX kernel: the data header + the ``mxgemm<CFG>`` driver + OUT-protocol print
    of the ``M x N`` top-left result. ``mx`` is the golden operand bundle (fmt / M / N / K / A_bytes / B_bytes
    / SA / SB / lutA / lutB)."""
    if mx.get("flash"):
        # Fused MX flash-attention: wrap the proven radiance-kernels FULL_ATTN2 kernel, baking an fa_data.h
        # from these codes + the structural SPAD_DEST-zeroing fix + an OUT-print of O. The on-device softmax
        # uses the fexp.h hardware exp, which the L2 reference (cyclotron) models as bf16(f32(bf16(x)).exp())
        # — the SAME bf16 row-softmax the golden's numpy reference computes — so the wrapped kernel reaches a
        # strict-tolerance pass. fp8 track (R8/R9/R10 softcap/RH4); fp6/fp4 fail closed inside the emitter.
        return _emit_flash_kernel(mx, out_name)
    if mx.get("batched"):
        mx = _assemble_batched(mx)
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
