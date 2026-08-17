"""Datapath-faithful MX flash-attention reference (the fused flash kernel's stages 2-3).

This module is the SINGLE source of truth for the microscaling (MX, e4m3 operands + E8M0 block scale)
flash-attention golden numerics. It reproduces, bit-for-bit, the arithmetic the fused flash-attention
kernel executes on the block-scaled MX datapath AFTER the QK score matmul:

  1. per-row bf16 softmax with the kernel's fexp semantics
       St = bf16(S * bf16(att_scale));   [optional Gemma-2 bf16 logit soft-cap]
       rowmax = max(St);   shifted = bf16(St - rowmax);   u = bf16(exp(f32(shifted)))
  2. the online-softmax row denominator ``l`` in the EXACT reduction order the kernel uses (16 strided
     lanes, each owning the words ``{j*16 + lane}`` and folding its two elements per word with a bf16
     add chain, then a balanced 16-leaf tree reduce)
  3. a per-32-element-block requant of the UNNORMALIZED ``u`` (never the normalized softmax): the block
     E8M0 scale is ``se = exp_field(bf16(blockmax)) - 127`` (E8M0 code ``se + 127``) and each element is
     truncatingly quantized to e4m3 via ``bf16_to_e4m3_scaled``
  4. the PV MX matmul over the unnormalized e4m3 P codes
  5. finalize ``O = O_unnorm * bf16(1 / bf16(l))``

Steps 1-5 are ported verbatim from the independent numpy model that is validated bit-exact (0 element
mismatches) against the cyclotron RTL run for the fp8 flash capsules. The tracked golden generator
(``merlin/contract/capsules/generate_corpus.py``) calls this, so the corpus it emits and the kernel the
grader runs share ONE arithmetic definition — there is no separate "normalize then requant" order that
could silently diverge from what the kernel computes.

TARGET-AGNOSTIC: the subject here is the MX *format* (a datapath fact the derived MX reference exposes),
never a target name. The caller supplies the validated ``mx_ref`` module (the derived reference), so this
code depends on no particular accelerator and bakes in no opcode/encoding constant.
"""
from __future__ import annotations


def bf16_to_e4m3_scaled(b: int, se: int) -> int:
    """Quantize a bf16 value (given as its 16-bit pattern ``b``) divided by ``2**se`` to an e4m3 code,
    TRUNCATING (RNE disabled) exactly as the kernel's ``bf16_to_e4m3_scaled`` does. ``emax=8``/``emin=-6``
    are the e4m3 exponent bounds; a subnormal/underflowing input maps to code 0 (fail-closed, no default
    substituted)."""
    emax, emin = 8, -6
    exp = (b >> 7) & 0xFF
    m3 = (b >> 4) & 0x7
    E = exp - 127 - se
    if E > emax:
        E, m3 = emax, 6
    if E == emax and m3 > 6:
        m3 = 6
    code = ((b >> 8) & 0x80) | ((E + 7) << 3) | m3
    if exp == 0 or E < emin:
        return 0
    return code & 0xFF


def e4m3_code_table(mx) -> dict:
    """``value(float) -> canonical e4m3 code`` map, DERIVED by decoding every code with the reference's own
    decoder (no baked table); the lowest code is kept per finite value. Mirrors the encoder the corpus
    operands are packed with, so a value that is exactly e4m3-representable round-trips to one byte."""
    table: dict[float, int] = {}
    for c in range(256):
        v = mx.fp8_e4m3_decode(c)
        if v == v and abs(v) != float("inf"):            # finite; keep the FIRST (lowest) code per value
            table.setdefault(float(v), c)
    return table


def flash_attention_fp8(mx, S_scores, V, SB_v, *, M, Skv, Dv, att_scale, softcap=None):
    """Datapath-faithful mxfp8 flash-attention stages 2-5.

    Inputs (decoded operands + scales + shapes, all target-agnostic):
      ``mx``        the derived MX reference module (``mx_ref``) supplying the bf16/e4m3/E8M0 primitives
      ``S_scores``  raw QK scores ``[M, Skv]`` as the QK MX matmul produced them (bf16-decoded, UNSCALED)
      ``V``         decoded value operand ``[Skv, Dv]`` (exactly e4m3-representable)
      ``SB_v``      E8M0 block-scale codes for V, shape ``[Skv/32, Dv]``
      ``M/Skv/Dv``  attention shapes;   ``att_scale`` the softmax logit scale;   ``softcap`` the optional
                    Gemma-2 logit soft-cap (``None`` to disable)

    Returns ``(O, P_codes, SA_p, l, P_dec, pv_art)``:
      ``O``        finalized output ``[M, Dv]`` (bf16-valued float64)
      ``P_codes``  the UNNORMALIZED e4m3 P codes ``[M, Skv]`` fed to the PV matmul
      ``SA_p``     the per-(32-block, row) E8M0 scale codes ``[Skv/32, M]``
      ``l``        the online-softmax row denominators ``[M]``
      ``P_dec``    the P codes decoded (per-element, pre-E8M0-scale) ``[M, Skv]`` for provenance
      ``pv_art``   PV-stage packing artifacts (raw P/V code bytes + shapes) for provenance
    """
    import numpy as np

    def bf16(x):
        return mx.bf16_round(float(x))                    # f32 -> bf16 RNE, returns float

    def bf16_bits(x):
        return mx.f32_to_bf16_rne(float(x))               # f32 -> 16-bit bf16 pattern

    S = np.asarray(S_scores, dtype=np.float64).reshape(M, Skv)
    scale = bf16(att_scale)
    NBLK = Skv // 32
    P_codes = np.zeros((M, Skv), np.uint8)
    P_dec = np.zeros((M, Skv), np.float64)
    SA_p = np.zeros((NBLK, M), np.uint8)
    l = np.zeros(M)

    _softcap = None
    if softcap is not None:
        # EXACT transcription of the kernel's bf16_softcap (Gemma-2): x=s/cap; e=exp(-2|x|);
        # t=(1-e)/(1+e); cap*sign(x)*t. All bf16 (bounded bf16 divide via f32). Mirrors the kernel.
        cap = float(softcap)
        two_over_cap = bf16(2.0 / cap)
        capv = bf16(cap)

        def _softcap(s):
            p = bf16(np.exp(np.float32(bf16(s * two_over_cap))))     # e^{2y}, bf16
            pf = np.float32(p)                                       # tanh=(p-1)/(p+1), fp32
            tf = np.float32((pf - np.float32(1)) / (pf + np.float32(1)))
            return bf16(np.float32(np.float32(capv) * tf))

    for m in range(M):
        St = np.array([bf16(S[m, n] * scale) for n in range(Skv)])
        if _softcap is not None:
            St = np.array([_softcap(v) for v in St])
        rowmax = float(np.max(St))
        shifted = np.array([bf16(v - rowmax) for v in St])
        # fexp: bf16(f32_exp(bf16(shifted)))
        u = np.array([bf16(np.exp(np.float32(v))) for v in shifted])
        # row denom l — EXACT online_softmax_block order: per-lane strided chain then 16-leaf tree.
        # lane owns words {j*NT+lane}; word w = elements (2w, 2w+1); lloc_lane = ((0+a)+b) over its
        # words (bf16 each step); then the balanced 16-leaf pairing.
        NT = 16
        WPL = Skv // (2 * NT)                             # words-per-lane-pair = Skv/32
        leaves = []
        for lane in range(NT):
            lloc = 0.0
            for j in range(WPL):
                w = j * NT + lane
                lloc = bf16(bf16(lloc + float(u[2 * w])) + float(u[2 * w + 1]))
            leaves.append(lloc)
        vals = leaves
        while len(vals) > 1:
            vals = [bf16(vals[i] + vals[i + 1]) for i in range(0, len(vals), 2)]
        l[m] = vals[0]
        # requant the UNNORMALIZED u per 32-block (default requant_P_to_spad_tiled path):
        #   se = bf16_floor_log2(blockmax) = exp_field(bf16(blockmax)) - 127; E8M0 code = se+127;
        #   elems = bf16_to_e4m3_scaled(u, se) (truncating)
        for b in range(NBLK):
            blk = u[b * 32:(b + 1) * 32]
            bmax_bits = bf16_bits(float(np.max(blk)))
            se = ((bmax_bits >> 7) & 0xFF) - 127
            SA_p[b, m] = (se + 127) & 0xFF
            blk_scale = 2.0 ** se                        # E8M0 block scale = 2**(code-127), code=se+127
            for j in range(32):
                code = bf16_to_e4m3_scaled(bf16_bits(float(blk[j])), se)
                P_codes[m, b * 32 + j] = code
                # decoded P for provenance = the ACTUAL represented value (e4m3 code decode * block scale)
                P_dec[m, b * 32 + j] = float(mx.fp8_e4m3_decode(int(code))) * blk_scale

    # PV MX matmul (bf16 hardware accumulate) over the UNNORMALIZED e4m3 P codes.
    table = e4m3_code_table(mx)
    Varr = np.asarray(V, dtype=np.float64).reshape(Skv, Dv)
    V_codes = np.array([[table[float(np.float32(Varr[i, j]))] for j in range(Dv)]
                        for i in range(Skv)], dtype=np.uint8)
    SBv = np.asarray(SB_v, dtype=np.uint8)
    Cbits = np.asarray(mx.mx_matmul(P_codes.reshape(-1), V_codes.reshape(-1),
                                    SA_p, SBv, M, Dv, Skv, fmt=mx.FMT_FP8))
    O_un = np.array([[float(mx.bf16_to_f32(int(Cbits[m, j]))) for j in range(Dv)] for m in range(M)])

    # finalize: O = O_unnorm * bf16(1 / bf16(l))
    O = np.zeros((M, Dv))
    for m in range(M):
        inv_l = bf16(1.0 / bf16(l[m]))
        for j in range(Dv):
            O[m, j] = bf16(float(O_un[m, j]) * inv_l)

    pv_art = {"A_bytes": P_codes.reshape(-1).tolist(), "A_shape": [M, Skv],
              "B_bytes": V_codes.reshape(-1).tolist(), "B_shape": [Skv, Dv], "G": 0,
              "lutA": None, "lutB": None}
    return O, P_codes, SA_p, l, P_dec, pv_art
