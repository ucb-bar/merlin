"""Host-owned Phase 0 numerics implementation."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
import os
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

from merlin.common.paths import _dotenv  # noqa: E402
from merlin.development.targetgen.corpus import binding as CS  # noqa: E402

# ---- numerics ----------------------------------------------------------------------------------


def _mx_golden(entry, binding):
    """MX matmul golden (bf16 output) + provenance, computed by mx_ref in hardware semantics. Operands are

    format-derived + rigor-gated; the E8M0 block-scale streams are rigor-gated too (a mis-indexed per-lane

    scale must change the output)."""

    from merlin.common.fp8_formats import canonical_float, e8m0_decode
    from merlin.development.targetgen.corpus import operands as CO
    from merlin.development.testgen.generation.goldens import _mx_ref, _mx_value_codes, _salt

    mx = _mx_ref(entry.get("numerical_semantics"))

    tok = canonical_float(binding.operand_dtype)  # fp8_e4m3 / fp6_e3m2 / fp4_e2m1

    op = entry.get("op", "matmul")

    if op not in ("matmul", "linear"):
        raise ValueError(f"MX regime supports matmul/linear only (got op {op!r} in {entry['name']!r})")

    dim = binding.tile_dim

    M = entry.get("M", entry.get("M_tiles", 1) * dim)

    K = entry.get("K", entry.get("K_tiles", 1) * dim)

    N = entry.get("N", entry.get("N_tiles", 1) * dim)

    if tok == "fp8_e4m3":
        fmt, max_alpha, G = mx.FMT_FP8, None, 0

    elif tok == "fp4_e2m1":
        fmt, max_alpha, G = mx.FMT_FP4, None, 0

    elif tok == "fp6_e3m2":
        fmt, max_alpha, G = mx.FMT_FP6, 16, 5  # single 16-entry LUT (fp6 is LUT-indexed)

    else:
        raise ValueError(f"unsupported MX operand dtype {tok!r}")

    codes = _mx_value_codes(mx, tok)

    lhs, weight, out = entry.get("lhs", "A0"), entry.get("weight", "W"), entry.get("out", "Y0")

    def synth(name, shape):

        # MX operands are kept small (|v| <= 4): a wide E8M0 block scale over a long-K bf16 accumulate would

        # otherwise saturate to inf (a golden any broken kernel matches). See rand_fp8 in lib/golden.

        vals = CO.operand_values(shape, tok, _salt(entry["name"], name), max_alphabet=max_alpha, mag_cap=4.0)

        problems = CO.rigor_findings(vals, shape)

        if problems:
            raise AssertionError(f"non-rigorous MX operand {name}{shape}: {problems}")

        return np.array(vals, dtype=np.float64).reshape(shape)

    A = synth(lhs, (M, K))

    W = synth(weight, (K, N))

    def enc(v):

        c = codes.get(float(np.float32(v)))

        if c is None:  # exactly-representable palette -> exact hit expected
            raise AssertionError(f"MX value {v!r} not exactly representable in {tok}")

        return c

    A_codes = np.vectorize(enc)(A).astype(np.uint8)

    B_codes = np.vectorize(enc)(W).astype(np.uint8)

    # Same partial-block refusal as _mx_requant_blocks: one E8M0 scale per WHOLE group, so a K with a

    # remainder would emit scale streams covering only K - (K % GROUP) elements while the operand codes

    # cover all K. The mismatch is silent -- the scales simply stop early.

    if K % mx.GROUP:
        raise ValueError(
            f"MX golden needs K to be a whole multiple of the {mx.GROUP}-element block-scale group; got "
            f"K={K} for capsule {entry.get('name')!r} ({K % mx.GROUP} element(s) in a partial final "
            f"group). One E8M0 scale is emitted per whole group, so the scale stream would cover only "
            f"{mx.GROUP * (K // mx.GROUP)} of {K} K elements."
        )

    GK = K // mx.GROUP

    SA = np.array(CO.e8m0_scale_codes((GK, M), _salt(entry["name"], "SA")), dtype=np.uint8)

    SB = np.array(CO.e8m0_scale_codes((GK, N), _salt(entry["name"], "SB")), dtype=np.uint8)

    for nm, sc in (("SA", SA), ("SB", SB)):
        prob = CO.scale_rigor_findings(sc.tolist())

        if prob:
            raise AssertionError(f"non-rigorous E8M0 scale stream {nm}{sc.shape}: {prob}")

    lutA = lutB = None

    if fmt == mx.FMT_FP8:
        Ab, Bb = A_codes, B_codes

    else:
        if fmt == mx.FMT_FP6:  # nibbles index a shared 16-entry LUT of e3m2 codes
            lut = np.array(
                sorted({int(c) for c in A_codes.reshape(-1)} | {int(c) for c in B_codes.reshape(-1)}), dtype=np.uint8
            )

            assert lut.size <= 16, f"fp6 LUT overflow ({lut.size} > 16)"

            lut = np.pad(lut, (0, 16 - lut.size))[:16]

            idx = {int(v): i for i, v in enumerate(lut)}

            A_nib = np.vectorize(lambda c: idx[int(c)])(A_codes).astype(np.uint8)

            B_nib = np.vectorize(lambda c: idx[int(c)])(B_codes).astype(np.uint8)

            # mx_ref indexes the LUT as ``L[(row_or_col >> G) * 16 + nib]`` — ONE 16-entry block per

            # ``1<<G`` rows (A) / cols (B). Supply exactly that many blocks (a single global palette shared

            # by all groups is replicated: every block is identical, so ``(g)*16 + nib`` always resolves to

            # lut[nib]). Prior code shipped a lone block, so any fp6 capsule with M or N > 1<<G (e.g. N=64)

            # indexed past it and crashed.

            grp = 1 << G

            nblk_A = (A_codes.shape[0] + grp - 1) // grp  # blocks along A rows (M)

            nblk_B = (B_codes.shape[1] + grp - 1) // grp  # blocks along B cols (N)

            lutA = np.tile(lut.reshape(1, 16), (nblk_A, 1))

            lutB = np.tile(lut.reshape(1, 16), (nblk_B, 1))

        else:
            A_nib, B_nib = A_codes, B_codes  # fp4 nibble == code

        Ab = ((A_nib[1::2, :] << 4) | (A_nib[0::2, :] & 0xF)).astype(np.uint8)  # pack along M

        Bb = ((B_nib[:, 1::2] << 4) | (B_nib[:, 0::2] & 0xF)).astype(np.uint8)  # pack along N

    C = mx.mx_matmul(Ab, Bb, SA, SB, M, N, K, fmt=fmt, lutA=lutA, lutB=lutB, G=G)

    y = [[float(mx.bf16_to_f32(int(C[i, j]))) for j in range(N)] for i in range(M)]

    prov = {
        lhs: {"shape": [M, K], "decoded": A.reshape(-1).tolist()},
        weight: {"shape": [K, N], "decoded": W.reshape(-1).tolist()},
        "SA_e8m0_codes": SA.tolist(),
        "SB_e8m0_codes": SB.tolist(),
        # The SAME scales again, keyed by the operand names the capsule DECLARES, as ordinary per-tensor
        # specs. The two lists above are non-tensor provenance that `canonical_input_raws` skips by
        # design, so before this the scales reached the reference kernel (which bakes them) and no one
        # else -- a submitted backend was handed block-scaled element bytes and no scales, which is half
        # a number. Recorded additively: the oracle keeps reading the lists above.
        f"{lhs}_scale": {
            "shape": [GK, M],
            "decoded": SA.reshape(-1).tolist(),
            "note": "E8M0 exponent codes, one per block of K elements per lhs row",
        },
        f"{weight}_scale": {
            "shape": [GK, N],
            "decoded": SB.reshape(-1).tolist(),
            "note": "E8M0 exponent codes, one per block of K elements per weight column",
        },
        "scale_example": {"SA[0][0]": int(SA[0, 0]), "as_scale": e8m0_decode(int(SA[0, 0]))},
        # RAW device operand bytes exactly as mx_ref consumed them (fp8: one byte/elt; fp4/fp6: packed) —
        # the ``decoded`` floats above lose precision through YAML, so a bit-exact grade re-runs the MX
        # datapath oracle over THESE codes, not the decoded values. fmt/dims/LUTs ride along so the grade
        # is self-contained and reproduces the golden exactly.
        "operand_codes": {
            "lhs": lhs,
            "weight": weight,
            "fmt": tok,
            "M": M,
            "N": N,
            "K": K,
            "G": G,
            "A_bytes": Ab.reshape(-1).tolist(),
            "A_shape": list(Ab.shape),
            "B_bytes": Bb.reshape(-1).tolist(),
            "B_shape": list(Bb.shape),
            "lutA": lutA.tolist() if lutA is not None else None,
            "lutB": lutB.tolist() if lutB is not None else None,
        },
    }

    return {out: y}, prov


def _mx_safe_palette(mx, tok: str) -> list:
    """The requant candidate values for ``tok``: exactly-representable decoded values in the |v|<=4 window

    the operand synthesizer uses (so a requant code never overflows the 4-bit column accumulator). fp6 is

    capped to the SAME 16-value pool the synthesizer draws from (``derive_palette(...,16)``), so the fused

    PV union LUT (P-codes ∪ V-codes) stays within the 16-entry fp6 LUT."""

    from merlin.development.targetgen.corpus import operands as CO

    if tok == "fp6_e3m2":
        return sorted(CO.derive_palette("fp6_e3m2", 16, mag_cap=4.0))

    if tok == "fp8_e4m3":
        vals = {float(mx.fp8_e4m3_decode(c)) for c in range(256)}

    elif tok == "fp4_e2m1":
        vals = {float(mx.fp4_e2m1_decode(c)) for c in range(16)}

    else:
        raise ValueError(f"no MX palette for {tok!r}")

    return sorted(v for v in vals if not math.isnan(v) and abs(v) <= 4.0)


def _mx_requant_blocks(P, palette, *, group: int, target: float = 2.0):
    """Requantize float ``P[M,K]`` to (DECODED values ``[M,K]`` drawn from ``palette``, E8M0 scale codes

    ``[K/group, M]``) as the MX PE does before a GEMM: one shared power-of-two E8M0 scale per (K-group, row)

    — the (group, lane) granularity ``mx_matmul`` indexes SA with — chosen so the block max maps to

    ~``target``, then nearest-palette rounding of each scaled element. Returns DECODED values (not codes) so

    the caller re-encodes them through the SAME codec as the synth operands (fp8 byte / fp4 nibble / fp6

    LUT). The block scale is applied back by mx_matmul via the E8M0 code (2**(code-127))."""

    import math

    import numpy as np

    M, K = P.shape

    # FAIL CLOSED ON A PARTIAL BLOCK. `K // group` silently drops the elements past the last whole group,

    # and every array here is zero-initialised, so the tail comes back as zeros and the golden simply does

    # not depend on that part of its own input. MEASURED: at K=33 one column is dropped; at K=48 sixteen of

    # forty-eight are -- a THIRD of the reduction -- and perturbing A[0,32] with K=33 leaves the result

    # bit-identical. That is a silently wrong golden, which is worse than no golden: it would certify a

    # backend that also ignored the tail and fail one that did not.

    #

    # No capsule on disk trips this (every MX K is 32 or 64), so refusing here changes nothing today and

    # turns the trap into a message. It is also the reason MX coverage is aligned-only: a non-aligned MX

    # capsule cannot be minted, so the tail path has never been exercised. Supporting it means giving the

    # tail group its own E8M0 scale over a short block -- a real change to this reference, not a relaxation

    # of this guard.

    if K % group:
        raise ValueError(
            f"MX requant needs K to be a whole multiple of the {group}-element block-scale group; got "
            f"K={K} ({K % group} element(s) in a partial final group). The reference assigns one E8M0 "
            f"scale per whole group and would silently zero the tail, producing a golden that ignores "
            f"{K % group} of its own K elements. Use a K that is a multiple of {group}, or extend this "
            f"reference to scale a partial final group."
        )

    G = K // group

    pv = sorted(palette)

    dec = np.zeros((M, K), dtype=np.float64)

    scodes = np.zeros((G, M), dtype=np.uint8)

    for m in range(M):
        for g in range(G):
            blk = P[m, g * group : (g + 1) * group]

            mabs = float(np.max(np.abs(blk)))

            e = 0 if mabs == 0.0 else int(round(math.log2(mabs / target)))

            scodes[g, m] = max(0, min(254, e + 127))

            s = 2.0 ** (int(scodes[g, m]) - 127)

            for j in range(group):
                t = float(blk[j]) / s

                dec[m, g * group + j] = min(pv, key=lambda val: abs(val - t))

    return dec, scodes


def _mx_stage_matmul(mx, A_dec, B_dec, SA, SB, M, N, K, tok):
    """ONE MX GEMM over DECODED operands + E8M0 scales at ``tok`` (mxfp8 / mxfp6 / mxfp4), using the SAME

    codec as the R6/R7 tiles: fp8 = one code byte per element; fp4 = e2m1 nibble packed (A along rows, B

    along cols); fp6 = e3m2 nibble packed indexing a per-group union 16-entry LUT (G=log2 rows/cols per LUT

    block). Returns (C_float[M][N] bf16-decoded, packing-artifacts dict for provenance). A_dec/B_dec must be

    exactly representable in ``tok`` (synth operands + palette-requantized P both are)."""

    import numpy as np

    from merlin.development.testgen.generation.goldens import _mx_value_codes

    codes = _mx_value_codes(mx, tok)

    def enc(X):

        return np.vectorize(lambda v: codes[float(np.float32(v))])(X).astype(np.uint8)

    A_codes, B_codes = enc(A_dec), enc(B_dec)

    lutA = lutB = None

    if tok == "fp8_e4m3":
        fmt, G = mx.FMT_FP8, 0

        Ab, Bb = A_codes, B_codes

    else:
        if tok == "fp6_e3m2":
            fmt, G = mx.FMT_FP6, 5

            lut = np.array(
                sorted({int(c) for c in A_codes.reshape(-1)} | {int(c) for c in B_codes.reshape(-1)}), dtype=np.uint8
            )

            assert lut.size <= 16, f"fp6 LUT overflow ({lut.size} > 16) — requant/synth palette too wide"

            lut = np.pad(lut, (0, 16 - lut.size))[:16]

            idx = {int(v): i for i, v in enumerate(lut)}

            A_nib = np.vectorize(lambda c: idx[int(c)])(A_codes).astype(np.uint8)

            B_nib = np.vectorize(lambda c: idx[int(c)])(B_codes).astype(np.uint8)

            grp = 1 << G

            lutA = np.tile(lut.reshape(1, 16), ((A_codes.shape[0] + grp - 1) // grp, 1))

            lutB = np.tile(lut.reshape(1, 16), ((B_codes.shape[1] + grp - 1) // grp, 1))

        else:  # fp4: nibble == code
            fmt, G = mx.FMT_FP4, 0

            A_nib, B_nib = A_codes, B_codes

        Ab = ((A_nib[1::2, :] << 4) | (A_nib[0::2, :] & 0xF)).astype(np.uint8)  # pack along M (rows)

        Bb = ((B_nib[:, 1::2] << 4) | (B_nib[:, 0::2] & 0xF)).astype(np.uint8)  # pack along N (cols)

    C = np.asarray(mx.mx_matmul(Ab, Bb, SA, SB, M, N, K, fmt=fmt, lutA=lutA, lutB=lutB, G=G))

    Cf = [[float(mx.bf16_to_f32(int(C[i, j]))) for j in range(N)] for i in range(M)]

    art = {
        "A_bytes": Ab.reshape(-1).tolist(),
        "A_shape": list(Ab.shape),
        "B_bytes": Bb.reshape(-1).tolist(),
        "B_shape": list(Bb.shape),
        "G": G,
        "lutA": lutA.tolist() if lutA is not None else None,
        "lutB": lutB.tolist() if lutB is not None else None,
    }

    return Cf, art


def _mx_attention_golden(entry, binding):
    """Fused MX flash-attention golden (bf16 output) + provenance, composed from mx_ref (QK & PV, at the

    entry's operand format) + a numpy bf16 row-softmax + a per-(K-group,row) E8M0 requant of P. Shapes:

    M queries, H head dim (K of QK), Skv keys, Dv value dim. fp8 tiles by DIM=16; fp6/fp4 tile by 32, so

    for a sub-format M, Skv, Dv must all be multiples of 32 (a smaller tile yields a degenerate all-zero

    GEMM). Optional Gemma-2 logit soft-cap via ``softcap``."""

    import math

    import numpy as np

    from merlin.common.fp8_formats import canonical_float, e8m0_decode
    from merlin.development.targetgen.corpus import operands as CO
    from merlin.development.testgen.generation.goldens import _mx_ref, _salt

    mx = _mx_ref(entry.get("numerical_semantics"))

    tok = canonical_float(binding.operand_dtype)  # fp8_e4m3 / fp6_e3m2 / fp4_e2m1

    if tok not in ("fp8_e4m3", "fp6_e3m2", "fp4_e2m1"):
        raise ValueError(f"MX attention operand dtype {binding.operand_dtype!r} -> {tok!r} unsupported")

    sub = tok != "fp8_e4m3"

    max_alpha = 16 if tok == "fp6_e3m2" else None  # fp6 draws from a 16-value LUT pool

    dim = binding.tile_dim

    # ONE definition of this datapath's shape granularity, shared with the requirement and the

    # synthesizer (corpus_spec.shape_quantum). The local `32 if sub else DIM` said the same thing for

    # this format and said it in a second place, so the DEFAULTS below -- which are what an attention

    # entry actually gets, since a cell carries only M/K/N -- kept spelling `dim` and `2*dim` and landed

    # under the row tile for every sub-byte format.

    _q = CS.shape_quantum(binding.operand_dtype, tile_dim=dim, scale_block=(binding.scale_block or mx.GROUP))

    row_tile, red_q = int(_q["row"]), int(_q["reduction"])

    def _up(n: int, q: int) -> int:

        return -(-int(n) // int(q)) * int(q)

    M = entry.get("M", entry.get("M_tiles", 1) * dim)

    H = entry.get("H", entry.get("head_dim", _up(2 * dim, red_q)))  # QK contraction (head dim), %GROUP

    Skv = entry.get("Skv", entry.get("keys", _up(2 * dim, red_q)))  # key positions

    Dv = entry.get("Dv", _up(dim, row_tile))  # value dim

    if H % mx.GROUP or Skv % mx.GROUP or M % row_tile or Dv % row_tile:
        raise ValueError(
            f"MX attention dims must satisfy H%{mx.GROUP}=Skv%{mx.GROUP}=0 and "
            f"M%{row_tile}=Dv%{row_tile}=0 for {tok} (got M={M} H={H} Skv={Skv} Dv={Dv})"
        )

    att_scale = float(entry.get("scale", 1.0 / math.sqrt(H)))

    softcap = entry.get("softcap")  # Gemma-2 logit soft-cap (None to disable)

    q, k, v, out = entry.get("q", "Q"), entry.get("k", "K"), entry.get("v", "V"), entry.get("out", "Y0")

    def synth(name, shape):

        vals = CO.operand_values(shape, tok, _salt(entry["name"], name), max_alphabet=max_alpha, mag_cap=4.0)

        prob = CO.rigor_findings(vals, shape)

        if prob:
            raise AssertionError(f"non-rigorous MX attention operand {name}{shape}: {prob}")

        return np.array(vals, dtype=np.float64).reshape(shape)

    def scales(name, shape):

        sc = np.array(CO.e8m0_scale_codes(shape, _salt(entry["name"], name)), dtype=np.uint8)

        prob = CO.scale_rigor_findings(sc.tolist())

        if prob:
            raise AssertionError(f"non-rigorous E8M0 stream {name}{shape}: {prob}")

        return sc

    def bf16_round(a):

        u = np.asarray(a, dtype=np.float32).view(np.uint32)

        return ((u >> 16) << 16).view(np.float32).astype(np.float64)

    Q = synth(q, (M, H))

    K = synth(k, (Skv, H))

    V = synth(v, (Skv, Dv))

    Kt = np.ascontiguousarray(K.T)  # device consumes K pre-transposed (K^T)

    # stage 1: S = mx_matmul(Q[M,H], K^T[H,Skv]) -> bf16 scores [M, Skv] (UNSCALED; the logit scale +

    # optional soft-cap are applied inside stage 2, in the datapath-faithful order the kernel uses).

    SA_q = scales("SA_q", (H // mx.GROUP, M))

    SB_k = scales("SB_k", (H // mx.GROUP, Skv))

    S_rows, qk_art = _mx_stage_matmul(mx, Q, Kt, SA_q, SB_k, M, Skv, H, tok)

    SB_v = scales("SB_v", (Skv // mx.GROUP, Dv))

    if not sub:
        # stages 2-5, DATAPATH-FAITHFUL (mxfp8): the EXACT flash-kernel order — a bf16 softmax over the

        # UNNORMALIZED exp-P, the online-softmax row denominator l (kernel reduction order), a per-32-block

        # e4m3 requant of the UNNORMALIZED P, the PV MX matmul, then finalize O = O_unnorm * bf16(1/bf16(l)).

        # The reference (mx_flash_ref) is validated bit-exact vs the cyclotron RTL, so the generator and the

        # kernel share ONE arithmetic — a regeneration reproduces exactly what the kernel computes.

        from merlin.verification.references import microscaling_flash as MXF

        O_arr, _P_codes, SA_p, _l, P_dec, pv_art = MXF.flash_attention_fp8(
            mx, S_rows, V, SB_v, M=M, Skv=Skv, Dv=Dv, att_scale=att_scale, softcap=softcap
        )

        O_rows = [[float(O_arr[m, j]) for j in range(Dv)] for m in range(M)]

    else:
        # sub-formats (mxfp6/mxfp4): the flash kernel's e4m3 requant is not defined for these, so keep the

        # palette-requant composition unchanged (these goldens fail closed at grade time and stay identical).

        S = bf16_round(np.array(S_rows) * att_scale)

        if softcap is not None:
            cap = float(softcap)

            S = bf16_round(cap * np.tanh(S / cap))

        # bf16 row-softmax (numerically stable: subtract row max) -> P [M, Skv]

        P = np.zeros((M, Skv), dtype=np.float64)

        for m in range(M):
            r = bf16_round(S[m] - float(np.max(S[m])))

            e = bf16_round(np.exp(r))

            P[m] = bf16_round(e / float(np.sum(e)))

        # requant P into the format palette (per (K-group,row) E8M0), then O = mx_matmul(P, V)

        palette = _mx_safe_palette(mx, tok)

        P_dec, SA_p = _mx_requant_blocks(P, palette, group=mx.GROUP)  # SA_p shape [Skv/32, M]

        O_rows, pv_art = _mx_stage_matmul(mx, P_dec, V, SA_p, SB_v, M, Dv, Skv, tok)

    prov = {
        q: {"shape": [M, H], "decoded": Q.reshape(-1).tolist()},
        k: {"shape": [Skv, H], "decoded": K.reshape(-1).tolist()},
        v: {"shape": [Skv, Dv], "decoded": V.reshape(-1).tolist()},
        "SA_q_e8m0_codes": SA_q.tolist(),
        "SB_k_e8m0_codes": SB_k.tolist(),
        "SB_v_e8m0_codes": SB_v.tolist(),
        "scale_example": {"SA_q[0][0]": int(SA_q[0, 0]), "as_scale": e8m0_decode(int(SA_q[0, 0]))},
        # The four scale streams under the operand names the capsule declares, so a submitted backend is
        # handed them alongside the elements. P_scale is the exponent the softmax intermediate is
        # requantized against: chosen HERE when the golden was built, so the kernel cannot derive it.
        f"{q}_scale": {
            "shape": [H // mx.GROUP, M],
            "decoded": SA_q.reshape(-1).tolist(),
            "note": "E8M0 codes, one per block of H per query row",
        },
        f"{k}_scale": {
            "shape": [H // mx.GROUP, Skv],
            "decoded": SB_k.reshape(-1).tolist(),
            "note": "E8M0 codes, one per block of H per key row",
        },
        f"{v}_scale": {
            "shape": [Skv // mx.GROUP, Dv],
            "decoded": SB_v.reshape(-1).tolist(),
            "note": "E8M0 codes, one per block of Skv per value column",
        },
        "P_scale": {
            "shape": [Skv // mx.GROUP, M],
            "decoded": SA_p.reshape(-1).tolist(),
            "note": "E8M0 codes the softmax intermediate P is requantized against",
        },
        # RAW device operand bytes exactly as mx_ref consumed them (per stage, format-packed) + LUTs, so a
        # bit-exact grade re-runs the two MX GEMMs + the pinned softmax/requant over THESE codes.
        "attention_codes": {
            "q": q,
            "k": k,
            "v": v,
            "fmt": tok,
            "M": M,
            "H": H,
            "Skv": Skv,
            "Dv": Dv,
            "att_scale": att_scale,
            "softcap": (None if softcap is None else float(softcap)),
            "SA_q": SA_q.reshape(-1).tolist(),
            "SB_k": SB_k.reshape(-1).tolist(),
            "SB_v": SB_v.reshape(-1).tolist(),
            "SA_p": SA_p.reshape(-1).tolist(),
            "qk_stage": qk_art,
            "pv_stage": pv_art,
            # the requantized P intermediate DECODED values (derived from the softmax; NOT an input operand).
            "P_decoded": P_dec.reshape(-1).tolist(),
        },
    }

    return {out: O_rows}, prov


def _mx_gemv_batched_golden(entry, binding):
    """Batched MX matmul golden (the MX host golden's decode-time gemv_batched, MX regime): ``B`` independent

    MX GEMMs ``A_b[M,H] @ W_b[H,N]`` on the block-scaled mx_pe, returned as ``[B,M,N]`` bf16.

    (The MX PE tiles N by ``DIM``=16, so N must be a multiple of 16 — a literal N=1 gemv is not expressible

    on the mx_ref datapath; this is the faithful batched analog.) mxfp8 only; golden from mlc mx_ref."""

    import numpy as np

    from merlin.common.fp8_formats import canonical_float, e8m0_decode
    from merlin.development.targetgen.corpus import operands as CO
    from merlin.development.testgen.generation.goldens import _mx_ref, _mx_value_codes, _salt

    mx = _mx_ref(entry.get("numerical_semantics"))

    tok = canonical_float(binding.operand_dtype)

    if tok != "fp8_e4m3":
        raise ValueError(f"MX gemv_batched supports mxfp8 only (got {binding.operand_dtype!r} -> {tok!r})")

    dim = binding.tile_dim

    B = int(entry.get("B", 2))

    M = entry.get("M", entry.get("M_tiles", 1) * dim)

    H = entry.get("H", entry.get("K", 2 * dim))  # contraction dim, %32

    N = entry.get("N", dim)  # %16

    if H % mx.GROUP or M % mx.DIM or N % mx.DIM:
        raise ValueError(f"MX gemv_batched dims: H%{mx.GROUP}=0, M%{mx.DIM}=N%{mx.DIM}=0 (got B={B} M={M} H={H} N={N})")

    codes = _mx_value_codes(mx, tok)

    lhs, weight, out = entry.get("lhs", "A0"), entry.get("weight", "W"), entry.get("out", "Y0")

    def synth(name, shape):

        vals = CO.operand_values(shape, tok, _salt(entry["name"], name), mag_cap=4.0)

        prob = CO.rigor_findings(vals, shape)

        if prob:
            raise AssertionError(f"non-rigorous MX gemv operand {name}{shape}: {prob}")

        return np.array(vals, dtype=np.float64).reshape(shape)

    def enc(A):

        return np.vectorize(lambda x: codes[float(np.float32(x))])(A).astype(np.uint8)

    rows_out: list = []

    A_dec, W_dec, batches = [], [], []

    for b in range(B):
        A = synth(f"{lhs}{b}", (M, H))

        W = synth(f"{weight}{b}", (H, N))

        SA = np.array(CO.e8m0_scale_codes((H // mx.GROUP, M), _salt(entry["name"], f"SA{b}")), dtype=np.uint8)

        SB = np.array(CO.e8m0_scale_codes((H // mx.GROUP, N), _salt(entry["name"], f"SB{b}")), dtype=np.uint8)

        for nm, sc in ((f"SA{b}", SA), (f"SB{b}", SB)):
            prob = CO.scale_rigor_findings(sc.tolist())

            if prob:
                raise AssertionError(f"non-rigorous E8M0 stream {nm}{sc.shape}: {prob}")

        C = np.asarray(mx.mx_matmul(enc(A), enc(W), SA, SB, M, N, H, fmt=mx.FMT_FP8))

        rows_out.append([[float(mx.bf16_to_f32(int(C[i, j]))) for j in range(N)] for i in range(M)])

        A_dec.append(A.reshape(-1).tolist())

        W_dec.append(W.reshape(-1).tolist())

        batches.append(
            {
                "A_bytes": enc(A).reshape(-1).tolist(),
                "W_bytes": enc(W).reshape(-1).tolist(),
                "SA": SA.reshape(-1).tolist(),
                "SB": SB.reshape(-1).tolist(),
            }
        )

    prov = {
        lhs: {"shape": [B, M, H], "decoded": A_dec},
        weight: {"shape": [B, H, N], "decoded": W_dec},
        "batched_codes": {
            "lhs": lhs,
            "weight": weight,
            "fmt": tok,
            "B": B,
            "M": M,
            "H": H,
            "N": N,
            # The MX reference emitter retains a flattened physical backing buffer, but
            # that layout is not the logical result type exposed by the capsule.
            "stacked_out_shape": [B * M, N],
            "logical_output_shape": [B, M, N],
            "batches": batches,
        },
        "scale_example": {"SA0[0][0]": batches[0]["SA"][0], "as_scale": e8m0_decode(int(batches[0]["SA"][0]))},
        # The per-batch scale streams under the operand names the capsule declares, so a submitted
        # backend is handed them the same way it is handed the elements (see the single-GEMM path).
        f"{lhs}_scale": {
            "shape": [B, H // mx.GROUP, M],
            "decoded": [c for bt in batches for c in bt["SA"]],
            "note": "E8M0 exponent codes per batch, one per block of H per lhs row",
        },
        f"{weight}_scale": {
            "shape": [B, H // mx.GROUP, N],
            "decoded": [c for bt in batches for c in bt["SB"]],
            "note": "E8M0 exponent codes per batch, one per block of H per weight column",
        },
    }

    return {out: rows_out}, prov
