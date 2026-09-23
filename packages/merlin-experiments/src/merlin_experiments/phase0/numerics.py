"""Host-owned Phase 0 numerics implementation."""

from __future__ import annotations

import importlib.util
import os
from fractions import Fraction
from pathlib import Path

import numpy as np

from merlin.common.paths import _dotenv  # noqa: E402
from merlin.targetgen import corpus_spec as CS  # noqa: E402


# ------------------------------------------------------------------------------------------------
# float golden engine (generation-time only; needs the external specir refmodel)
# ------------------------------------------------------------------------------------------------
def _specir():
    from merlin.integrations.specir import importable

    root = os.environ.get("SPECIR_ROOT") or _dotenv().get("SPECIR_ROOT")
    with importable(root):
        from specir.oracle import dtypes as D
        from specir.oracle.refmodel import fp_reduce

    return D, fp_reduce


# specir fp8 format handle per canonical operand dtype token (fail closed if the refmodel lacks it).
_SPECIR_FP8_ATTR = {"fp8_e4m3": "FP8_E4M3", "fp8_e5m2": "FP8_E5M2"}


def _specir_fp8(D, fmt_token: str):
    attr = _SPECIR_FP8_ATTR.get(fmt_token)
    if attr is None or not hasattr(D, attr):
        raise ValueError(
            f"specir refmodel has no fp8 format for operand dtype {fmt_token!r} (known: {sorted(_SPECIR_FP8_ATTR)})"
        )
    return getattr(D, attr)


def _det_fp8(D, name, shape, salt, fmt_token, d_fp8):
    """Structured, format-DERIVED operand bytes: distinct rows AND columns + asymmetric (so a wrong row
    stride / base offset / transposed load changes the output), spanning the fp8 format's representable
    range. Replaces the old 11-magnitude flat-hash fill (~6 distinct values, ~11/32 distinct rows) that hid
    those bug classes. See merlin.targetgen.corpus_operands."""
    from merlin.targetgen import corpus_operands as CO

    salt_int = sum((i + 1) * ord(c) for i, c in enumerate(f"{salt}|{name}")) or 1
    vals = CO.operand_values(tuple(shape), fmt_token, salt_int)
    raw = [D.encode_float(v, d_fp8) for v in vals]
    # Self-enforcing rigor: fail generation loudly if the ENCODED bytes are not distinct-per-row/col +
    # asymmetric (e.g. a future palette/fill change, or an encode that collapsed distinct values). A weak
    # operand silently hides addressing/stride/transpose bugs — never let a regeneration ship one.
    if len(shape) == 2:
        problems = CO.rigor_findings([float(b) for b in raw], tuple(shape))
        if problems:
            raise AssertionError(f"non-rigorous operand {name}{tuple(shape)}: {problems}")
    return raw, vals


def _operand_decoder(D, fmt, *, flush_subnormals: bool):
    """Decode a raw operand code the way the DATAPATH decodes it, exactly.

    By default that is the format's own exact value. A datapath that admits only NORMAL operands sees
    zero wherever the operand's exponent field is zero, and a reference model that decodes those codes
    to their tiny nonzero value is modelling different hardware — every later add carries the
    difference. Whether the target does that is a measured property of its compute unit, declared in
    its profile's ``datapath`` block (``subnormal_operand_flush``); nothing here assumes it.

    The subnormal test is DERIVED from the format descriptor the refmodel hands back (the exponent
    field, located by the format's own ``mant_bits``/``exp_bits``), so it holds for any exponent /
    mantissa split rather than one hardcoded byte layout.
    """
    if not flush_subnormals:
        return lambda raw: D.decode_float_exact(int(raw), fmt)
    exp_mask = (1 << fmt.exp_bits) - 1

    def decode(raw):
        raw = int(raw)
        if ((raw >> fmt.mant_bits) & exp_mask) == 0:  # exponent field zero => subnormal (or zero)
            return Fraction(0)
        return D.decode_float_exact(raw, fmt)

    return decode


def _float_golden(entry, binding):
    """A capsule's fp8->bf16 golden + input provenance from the specir refmodel (independent of the RTL)."""
    D, fp_reduce = _specir()
    fmt_token = binding.operand_dtype  # e.g. "fp8_e4m3" — DERIVED, not assumed
    FP8, BF16 = _specir_fp8(D, fmt_token), D.BF16
    dec = _operand_decoder(D, FP8, flush_subnormals=binding.subnormal_operand_flush)
    salt, dim = entry["name"], binding.tile_dim
    prov, outputs = {}, {}

    def reg(name, shape, *, declared_shape=None):
        """Register one input operand. ``shape`` is the 2-D shape the STIMULUS is built (and rigor-checked)
        at; ``declared_shape`` is the shape the capsule declares for the same flat row-major bytes, when the
        two differ. They differ for a rank-4 activation: `operand_values` builds a matrix, and an NHWC
        image with N=1 IS the matrix [H*W, Ci] in row-major order -- so the rigor guarantee (distinct rows,
        distinct columns, asymmetric) lands on exactly the axes a conv can get wrong, spatial position and
        channel, instead of being skipped because the declared rank is not two."""
        raw, vals = _det_fp8(D, name, shape, salt, fmt_token, FP8)
        prov[name] = {
            "shape": list(declared_shape or shape),
            "fp8_raw_hex": [f"0x{r:02x}" for r in raw],
            "decoded": vals,
        }
        return raw

    def reg_acc(name, shape):
        """An operand that lives in the ACCUMULATOR's format, not the input format.

        A bias is added to the accumulator, so it is declared and generated there. Its VALUES still come
        from the operand format's palette, and that is the load-bearing part: the accumulator format's
        own palette spans its entire exponent range, which for bf16 reaches ~1e-31, and a bias that
        small added to a matmul output of order one rounds away to nothing. The golden would then be
        byte-identical to the unfused matmul's, so a backend that DROPPED the bias entirely would pass
        the fused capsule -- the failure mode where a gate is satisfied by arithmetic nobody performed.
        Drawing from the operand palette keeps the addend on the same scale as the sum it lands on.

        Returns the decoded values (not raw codes): everything downstream of an accumulator-format
        operand is float arithmetic, and the byte-level palette-preload path is for input operands.
        """
        from merlin.targetgen import corpus_operands as CO

        if len(shape) == 2:
            _, vals = _det_fp8(D, name, shape, salt, fmt_token, BF16)
        else:
            # A bias is a VECTOR, and `operand_values` shapes a matrix. Ask it for one row and flatten.
            # The per-row/per-column rigor `_det_fp8` enforces is not the right check for a vector --
            # it has one row -- but the part that still matters is: a constant bias is satisfied by a
            # kernel that adds any single number, and would not detect a broadcast along the wrong
            # axis. So distinctness ALONG the vector is asserted here instead of skipped.
            (n,) = shape
            salt_int = sum((i + 1) * ord(c) for i, c in enumerate(f"{salt}|{name}")) or 1
            vals = list(CO.operand_values((1, n), fmt_token, salt_int))
            if n > 1 and len(set(vals)) < 2:
                raise AssertionError(
                    f"non-rigorous bias {name}({n},): every element is {vals[0]!r}, so a kernel that "
                    f"broadcast one value along the wrong axis would still match the golden"
                )
        prov[name] = {"shape": list(shape), "decoded": vals}
        return vals

    def rnd(x):
        return D.round_to_format(x, BF16, "rne")

    #: MEMOIZED PRODUCT. ``rnd(dec(a) * dec(b))`` is a pure function of the OPERAND CODE PAIR, so
    #: caching it on that pair is bit-identical by construction -- same inputs, same function, same
    #: answer -- rather than an approximation traded for speed. It is worth doing because the operand
    #: fill draws from a small deterministic alphabet, so a deep-K contraction re-derives the same few
    #: products millions of times: measured, this engine ran ~33.7 ms per unit of K, which is ~37
    #: minutes for the single k65536 residency member and ~86 minutes across the four deep-K ones.
    #:
    #: THE REDUCTION IS DELIBERATELY NOT TOUCHED. ``fp_reduce`` accumulates in the device's own order,
    #: one step at a time, and that sequencing is the whole reason this engine is not a numpy dot
    #: product. Only the per-element product -- which carries no order -- is cached.
    _prod_cache: dict = {}

    def _prod(a_code, b_code):
        key = (a_code, b_code)
        hit = _prod_cache.get(key)
        if hit is None:  # `is None` not truthiness: a rounded product may be 0
            hit = rnd(dec(a_code) * dec(b_code))
            _prod_cache[key] = hit
        return hit

    def mm(a_raw, ashape, w_raw, wshape):
        m, k = ashape
        _, n = wshape
        out = [[0] * n for _ in range(m)]
        for i in range(m):
            a_row = a_raw[i * k : (i + 1) * k]
            for j in range(n):
                prods = [_prod(a_row[p], w_raw[p * n + j]) for p in range(k)]
                out[i][j] = fp_reduce(prods, BF16, order="index_sequential", cadence="per_step", rm="rne")
        return out

    def floats(y):
        return [[D.decode_float(v, BF16) for v in row] for row in y]

    op = entry.get("op", "matmul")
    if op in ("matmul", "linear"):
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        a = reg(entry.get("lhs", "A0"), (M, K))
        w = reg(entry.get("weight", "W"), (K, N))
        y = mm(a, (M, K), w, (K, N))
        epi = entry.get("epilogue", [])
        if "acc_scale" in epi:
            s = Fraction(entry["acc_scale"]).limit_denominator(1 << 20)
            y = [[rnd(D.decode_float_exact(v, BF16) * s) for v in row] for row in y]
        if "relu" in epi:
            y = [[v if D.decode_float(v, BF16) > 0 else 0 for v in row] for row in y]
        outputs[entry.get("out", "Y0")] = floats(y)
    elif op == "fused_matmul_bias":
        # The matmul branch above with the bias stage, which is where the op name says it happens. The
        # addend is in the accumulator format (see `reg_acc`) because that is where it lands.
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        a = reg(entry.get("lhs", "A0"), (M, K))
        w = reg(entry.get("weight", "W"), (K, N))
        y = mm(a, (M, K), w, (K, N))
        # EXACT rationals, like the acc_scale stage above: `round_to_format` needs a Fraction, and a
        # bf16 palette value is a dyadic rational, so Fraction(v) is exact -- no limit_denominator,
        # which would perturb the very addend whose effect the golden has to record.
        b = [Fraction(v) for v in reg_acc(entry.get("bias", "B"), (N,))]
        y = [[rnd(D.decode_float_exact(v, BF16) + b[j]) for j, v in enumerate(row)] for row in y]
        if "relu" in entry.get("epilogue", []):
            y = [[v if D.decode_float(v, BF16) > 0 else 0 for v in row] for row in y]
        outputs[entry.get("out", "Y0")] = floats(y)
    elif op == "bias_add":
        # The same addition standing alone. Both operands are in the accumulator format, because this op
        # IS the fused capsule's bias stage lifted out of it -- so the two members add the same numbers.
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        x = [Fraction(v) for v in reg_acc(entry.get("src", "X"), (M, N))]
        b = [Fraction(v) for v in reg_acc(entry.get("bias", "B"), (N,))]
        outputs[entry.get("out", "Y0")] = floats([[rnd(x[i * N + j] + b[j]) for j in range(N)] for i in range(M)])
    elif op in ("gemv_batched", "batch_matmul"):
        # A BATCHED CONTRACTION IS B INDEPENDENT ONES, and that is the whole of it: the device's shim
        # loops over B calling the same (M,N,K) kernel per slice, so the golden is the same `mm` per
        # slice with source-visible shape [B,M,N]. It exists because the corpus could not
        # express a rank-3 region on this datapath at all -- the only batched golden was block-scaled,
        # so a target whose contract admits batching had its `contraction.batched` requirement reported
        # as "no builder materializes a rank-3 region" while the rewrite, the device kernel and the
        # shim's B loop were all already there.
        B = int(entry.get("B", 2))
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("H", entry.get("K_tiles", 2) * dim))
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        lhs, weight = entry.get("lhs", "A0"), entry.get("weight", "W")
        batches: list = []
        for b in range(B):
            # One operand PER SLICE, salted by the slice index. Reusing one operand across B would make
            # every slice's output identical, and a kernel that computed one slice and broadcast it
            # would match the golden exactly -- the degeneracy this corpus already refuses elsewhere.
            a = reg(f"{lhs}_b{b}", (M, K))
            w = reg(f"{weight}_b{b}", (K, N))
            batches.append(floats(mm(a, (M, K), w, (K, N))))
        outputs[entry.get("out", "Y0")] = batches
    elif op == "movement":
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        x = reg(entry.get("src", "X"), (M, N))
        outputs[entry.get("out", "Y0")] = floats([[rnd(dec(x[i * N + j])) for j in range(N)] for i in range(M)])
    elif op == "resident_reuse":
        K = entry.get("K_tiles", 1) * dim
        N = entry.get("N_tiles", 1) * dim
        w = reg(entry["weight"], (K, N))
        for m in entry["matmuls"]:
            M = m.get("M_tiles", 1) * dim
            a = reg(m["lhs"], (M, K))
            outputs[m["out"]] = floats(mm(a, (M, K), w, (K, N)))
    elif op == "attention_qk":
        M = entry.get("M_tiles", 1) * dim
        Kd = entry.get("K_tiles", 1) * dim
        q = reg(entry.get("q", "Q"), (M, Kd))
        k = reg(entry.get("k", "K"), (M, Kd))
        kt = [0] * (M * Kd)
        for i in range(M):
            for j in range(Kd):
                kt[j * M + i] = k[i * Kd + j]
        outputs[entry.get("out", "Y0")] = floats(mm(q, (M, Kd), kt, (Kd, M)))
    elif op == "conv2d":
        # AN IM2COL CONV IS A CONTRACTION OVER GATHERED WINDOWS, and that is the whole of it: the device
        # gathers [Ho*Wo, Kh*Kw*Ci] out of the NHWC activation and runs the same reduction the matmul
        # branch runs. So the golden reuses `mm` over the runtime's OWN gather (`conv_im2col`, the single
        # source of truth shared with the runner harness and the integer engine) rather than a second
        # transcription of the window arithmetic -- a second transcription is how a golden and a harness
        # come to disagree about which tap a window reads.
        #
        # The gather is a permutation-with-zero-fill of raw operand CODES, so it is dtype-blind: an
        # out-of-bounds tap contributes code 0, which every float format here decodes to +0.0, exactly the
        # zero-pad the integer engine applies. Nothing about this branch is target-specific; the geometry
        # comes from the entry and the formats from the binding.
        from merlin.runtime.commandbuffer import conv_im2col, conv_out_dims
        from merlin.runtime.tensor import Tensor

        if "maxpool" in [str(s) for s in (entry.get("epilogue") or [])]:
            # Fail closed rather than emit an unpooled reference for a capsule that declares pooling: a
            # golden that silently skipped the stage would agree with a backend that skipped it too.
            raise ValueError(
                f"float conv2d golden: capsule {entry['name']!r} declares a maxpool "
                f"epilogue, which this engine does not model"
            )
        ci = int(entry.get("ci", entry.get("Cin", 4)))
        cout = int(entry.get("N", entry.get("Cout", dim)))
        Himg, Wimg = int(entry.get("Himg", 8)), int(entry.get("Wimg", 8))
        kh, kw = int(entry.get("kh", 3)), int(entry.get("kw", 3))
        stride = tuple(entry.get("stride", [1, 1]))
        padding = tuple(entry.get("padding", [0, 0, 0, 0]))
        dilation = tuple(entry.get("dilation", [1, 1]))
        layout = entry.get("layout", "nhwc")
        Ho, Wo = conv_out_dims(Himg, Wimg, kh, kw, stride, padding, dilation)
        Kdim = kh * kw * ci
        ifm_name, w_name = entry.get("ifm", "IFM"), entry.get("weight", "W")
        # Stimulus built as the [H*W, Ci] matrix the NHWC image is in row-major order (see `reg`), so the
        # operand rigor gate applies; declared to the capsule at its rank-4 shape.
        ifm = reg(ifm_name, (Himg * Wimg, ci), declared_shape=(1, Himg, Wimg, ci))
        w = reg(w_name, (Kdim, cout))
        cols = conv_im2col(
            Tensor((1, Himg, Wimg, ci), list(ifm), "u8"),  # raw operand CODES, gathered
            kh=kh,
            kw=kw,
            ci=ci,
            stride=stride,
            padding=padding,
            dilation=dilation,
            layout=layout,
        )
        y = mm(cols.data, (Ho * Wo, Kdim), w, (Kdim, cout))
        epi = entry.get("epilogue", [])
        if "acc_scale" in epi:
            s = Fraction(entry["acc_scale"]).limit_denominator(1 << 20)
            y = [[rnd(D.decode_float_exact(v, BF16) * s) for v in row] for row in y]
        if "relu" in epi:
            y = [[v if D.decode_float(v, BF16) > 0 else 0 for v in row] for row in y]
        outputs[entry.get("out", "Y0")] = floats(y)
    else:
        raise ValueError(f"no float golden for op {op!r}")
    return outputs, prov


# ------------------------------------------------------------------------------------------------
# MX (microscaling block-scaled FP) golden engine — HARDWARE semantics via mlc's mx_ref, NOT specir
# (specir is the atlas fp8 refmodel; MX is a different datapath: 16-deep systolic per-column accumulate
# schedule + one E8M0 scale per 32-element K group). mx_ref is transcribed bit-exactly from the target's
# own reference (radiance-kernels lib/golden/{mx_fp_math.h,mx_golden.cpp}, mirroring the RTL).
# ------------------------------------------------------------------------------------------------
def _mx_ref():
    """Import mlc's ``validate/mx_ref.py`` BY FILE PATH (like the specir import) so we do NOT trigger
    ``mlc/validate/__init__.py`` (which carries concurrent work and heavy imports)."""
    root = os.environ.get("MERLIN_MLC_DIR") or _dotenv().get("MERLIN_MLC_DIR")
    path = Path(root) / "mlc" / "validate" / "mx_ref.py"
    if not path.exists():
        raise FileNotFoundError(f"mx_ref not found at {path} (set MERLIN_MLC_DIR to the mlc modeling root)")
    spec = importlib.util.spec_from_file_location("merlin_mx_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _salt(name: str, tensor: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(f"{name}|{tensor}")) or 1


def _mx_value_codes(mx, fmt_token: str):
    """value(float) -> device code, DERIVED by decoding every code with mx_ref's own decoder (no baked
    table). fp8: 8-bit e4m3 code; fp4: 4-bit e2m1 nibble; fp6: 6-bit e3m2 code."""
    if fmt_token == "fp8_e4m3":
        rng, dec = range(256), mx.fp8_e4m3_decode
    elif fmt_token == "fp4_e2m1":
        rng, dec = range(16), mx.fp4_e2m1_decode
    elif fmt_token == "fp6_e3m2":
        rng, dec = range(64), mx.fp6_e3m2_decode
    else:
        raise ValueError(f"no MX code table for {fmt_token!r}")
    table: dict[float, int] = {}
    for c in rng:
        v = dec(c)
        if v == v and abs(v) != float("inf"):  # finite; keep the FIRST (lowest) code per value
            table.setdefault(float(v), c)
    return table


def _mx_golden(entry, binding):
    """MX matmul golden (bf16 output) + provenance, computed by mx_ref in hardware semantics. Operands are
    format-derived + rigor-gated; the E8M0 block-scale streams are rigor-gated too (a mis-indexed per-lane
    scale must change the output)."""
    from merlin.runtime.fp8_formats import canonical_float, e8m0_decode
    from merlin.targetgen import corpus_operands as CO

    mx = _mx_ref()
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


# ------------------------------------------------------------------------------------------------
# MX FUSED FLASH-ATTENTION golden — COMPOSED from the SAME validated mx_ref engine used above:
#   S = mx_matmul(Q, K^T)   (block-scaled E8M0, bf16)  -> scaled by 1/sqrt(head) [+ optional soft-cap]
#   P = bf16 row-softmax(S)                             (numpy, bf16-rounded)
#   O = mx_matmul(P_requant, V)  (block-scaled E8M0, bf16)
# The intermediate P is requantized to the MX code space exactly as the MX PE does before the second GEMM:
# a per-(K-group,row) E8M0 scale brings each block to O(1) mantissas (the mx_ref accumulator carries only
# 4-bit column exponents, so DECODED codes must stay small and the E8M0 scale carry the magnitude — same
# convention as the synthesized Q/K/V operands), then each element rounds to the nearest representable value
# in that format's palette. NOTHING is fabricated: both matmuls are the mlc mx_ref hardware datapath (same
# codec as the R6/R7 fp6/fp4 tiles: fp8 = one byte/code, fp4 = e2m1 nibble, fp6 = e3m2 nibble + per-group
# 16-entry LUT); only the softmax + the standard MX requant of P are numpy. Parameterized by operand format
# (mxfp8 / mxfp6 / mxfp4).
# ------------------------------------------------------------------------------------------------
def _mx_safe_palette(mx, tok: str) -> list:
    """The requant candidate values for ``tok``: exactly-representable decoded values in the |v|<=4 window
    the operand synthesizer uses (so a requant code never overflows the 4-bit column accumulator). fp6 is
    capped to the SAME 16-value pool the synthesizer draws from (``derive_palette(...,16)``), so the fused
    PV union LUT (P-codes ∪ V-codes) stays within the 16-entry fp6 LUT."""
    from merlin.targetgen import corpus_operands as CO

    if tok == "fp6_e3m2":
        return sorted(CO.derive_palette("fp6_e3m2", 16, mag_cap=4.0))
    if tok == "fp8_e4m3":
        vals = {float(mx.fp8_e4m3_decode(c)) for c in range(256)}
    elif tok == "fp4_e2m1":
        vals = {float(mx.fp4_e2m1_decode(c)) for c in range(16)}
    else:
        raise ValueError(f"no MX palette for {tok!r}")
    return sorted(v for v in vals if v == v and abs(v) <= 4.0)


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

    from merlin.runtime.fp8_formats import canonical_float, e8m0_decode
    from merlin.targetgen import corpus_operands as CO

    mx = _mx_ref()
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
        from merlin.targetgen import mx_flash_ref as MXF

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
    """Batched MX matmul golden (radiance-kernels decode-time gemv_batched, MX regime): ``B`` independent
    MX GEMMs ``A_b[M,H] @ W_b[H,N]`` on the block-scaled mx_pe, returned as ``[B,M,N]`` bf16.
    (The MX PE tiles N by ``DIM``=16, so N must be a multiple of 16 — a literal N=1 gemv is not expressible
    on the mx_ref datapath; this is the faithful batched analog.) mxfp8 only; golden from mlc mx_ref."""
    import numpy as np

    from merlin.runtime.fp8_formats import canonical_float, e8m0_decode
    from merlin.targetgen import corpus_operands as CO

    mx = _mx_ref()
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


def _simt_golden(entry, binding):
    """SIMT (CVFPU) golden in ordinary IEEE float — fp32 accumulate, format-rounded operands. Covers the
    matmul / attention / rmsnorm shapes; independent of any accelerator model (the SIMT cores do plain IEEE
    math). Operands are format-derived + rigor-gated."""
    from merlin.runtime.fp8_formats import canonical_float
    from merlin.targetgen import corpus_operands as CO

    tok = canonical_float(binding.operand_dtype)  # fp16 / bf16 / f32
    dim = binding.tile_dim
    op = entry.get("op", "matmul")

    def q(arr):
        a = np.asarray(arr, dtype=np.float64)
        if tok == "fp16":
            return a.astype(np.float16).astype(np.float64)
        if tok == "bf16":  # operands are exact bf16 already; identity round
            u = a.astype(np.float32).view(np.uint32)
            return ((u >> 16) << 16).view(np.float32).astype(np.float64)
        return a.astype(np.float32).astype(np.float64)

    def synth(name, shape):
        vals = CO.operand_values(shape, tok, _salt(entry["name"], name))
        problems = CO.rigor_findings(vals, shape)
        if problems:
            raise AssertionError(f"non-rigorous SIMT operand {name}{shape}: {problems}")
        return q(np.array(vals, dtype=np.float64).reshape(shape))

    def rnd_out(y):
        return [[float(np.float32(v)) for v in row] for row in np.asarray(y)]

    prov, outputs = {}, {}
    if op in ("matmul", "linear"):
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        A = synth(entry.get("lhs", "A0"), (M, K))
        W = synth(entry.get("weight", "W"), (K, N))
        y = (A.astype(np.float32) @ W.astype(np.float32)).astype(np.float64)
        epi = entry.get("epilogue", [])
        if "acc_scale" in epi:
            y = y * float(entry["acc_scale"])
        if "relu" in epi:
            y = np.maximum(y, 0.0)
        prov[entry.get("lhs", "A0")] = {"shape": [M, K], "decoded": A.reshape(-1).tolist()}
        prov[entry.get("weight", "W")] = {"shape": [K, N], "decoded": W.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    elif op in ("gemv_batched", "batch_matmul"):
        # B independent GEMMs with source-visible output [B,M,N] -- the same decomposition the integer and
        # specir engines make, because it is the one the device's shim performs: a loop over B calling
        # the (M,N,K) kernel once per slice. One operand PER SLICE, salted by index, so a kernel that
        # computed one slice and broadcast it does not match.
        B = int(entry.get("B", 2))
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("H", entry.get("K_tiles", 2) * dim))
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        lhs, weight = entry.get("lhs", "A0"), entry.get("weight", "W")
        A = np.stack([synth(f"{lhs}_b{b}", (M, K)) for b in range(B)])
        W = np.stack([synth(f"{weight}_b{b}", (K, N)) for b in range(B)])
        y = [(A[b].astype(np.float32) @ W[b].astype(np.float32)).astype(np.float64) for b in range(B)]
        prov[lhs] = {"shape": [B, M, K], "decoded": A.reshape(-1).tolist()}
        prov[weight] = {"shape": [B, K, N], "decoded": W.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = [rnd_out(batch) for batch in y]
    elif op == "movement":
        # A load->store movement (mvin/mvout) moves data and computes nothing, so the reference is the
        # operand itself at the OUTPUT format. Its value as a capsule is that it exercises the movement
        # family the contract declares, on a datapath whose only job is to not corrupt what it carries.
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        X = synth(entry.get("src", "X"), (M, N))
        prov[entry.get("src", "X")] = {"shape": [M, N], "decoded": X.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(X)
    elif op == "attention_qk":
        M = entry.get("M_tiles", 1) * dim
        Kd = entry.get("K_tiles", 1) * dim
        Q = synth(entry.get("q", "Q"), (M, Kd))
        Kk = synth(entry.get("k", "K"), (M, Kd))
        y = (Q.astype(np.float32) @ Kk.astype(np.float32).T).astype(np.float64)
        prov[entry.get("q", "Q")] = {"shape": [M, Kd], "decoded": Q.reshape(-1).tolist()}
        prov[entry.get("k", "K")] = {"shape": [M, Kd], "decoded": Kk.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    elif op == "rmsnorm":
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        eps = float(entry.get("eps", 1.0 / 65536.0))
        X = synth(entry.get("src", "X"), (M, K))
        gamma = synth(entry.get("gamma", "G"), (1, K))[0]
        y = np.empty((M, K), dtype=np.float64)
        for m in range(M):
            row = X[m].astype(np.float32)
            ss = np.float32(0.0)
            for k in range(K):
                ss = np.float32(ss + np.float32(row[k] * row[k]))
            mean = np.float32(ss / np.float32(K))
            rms = np.float32(1.0) / np.float32(np.sqrt(np.float32(mean + np.float32(eps))))
            for k in range(K):
                y[m, k] = float(np.float32(np.float32(row[k] * rms) * np.float32(gamma[k])))
        prov[entry.get("src", "X")] = {"shape": [M, K], "decoded": X.reshape(-1).tolist()}
        prov[entry.get("gamma", "G")] = {"shape": [1, K], "decoded": gamma.tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    elif op == "rmsnorm_qkv":
        # fused pre-norm QKV projection: H = rmsnorm(X, gamma); Y = H @ Wqkv. Both stages IEEE fp32-accum.
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        eps = float(entry.get("eps", 1.0 / 65536.0))
        X = synth(entry.get("src", "X"), (M, K))
        gamma = synth(entry.get("gamma", "G"), (1, K))[0]
        Wqkv = synth(entry.get("weight", "Wqkv"), (K, N))
        Hn = np.empty((M, K), dtype=np.float64)
        for m in range(M):
            row = X[m].astype(np.float32)
            ss = np.float32(0.0)
            for k in range(K):
                ss = np.float32(ss + np.float32(row[k] * row[k]))
            rms = np.float32(1.0) / np.float32(np.sqrt(np.float32(np.float32(ss / np.float32(K)) + np.float32(eps))))
            for k in range(K):
                Hn[m, k] = float(np.float32(np.float32(row[k] * rms) * np.float32(gamma[k])))
        y = (Hn.astype(np.float32) @ Wqkv.astype(np.float32)).astype(np.float64)
        prov[entry.get("src", "X")] = {"shape": [M, K], "decoded": X.reshape(-1).tolist()}
        prov[entry.get("gamma", "G")] = {"shape": [1, K], "decoded": gamma.tolist()}
        prov[entry.get("weight", "Wqkv")] = {"shape": [K, N], "decoded": Wqkv.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    elif op == "rope_qkv":
        # fused QKV projection + RoPE: H = X @ Wqkv; Y = rope(H). GPT-NeoX/Llama rotation (theta=10000),
        # position = row index, identical convention to the pytorch RP8 rope (capsule_source._rope).
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        X = synth(entry.get("src", "X"), (M, K))
        Wqkv = synth(entry.get("weight", "Wqkv"), (K, N))
        H = (X.astype(np.float32) @ Wqkv.astype(np.float32)).astype(np.float64)
        half = N // 2
        theta = float(entry.get("rope_theta", 10000.0))
        freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
        pos = np.arange(M, dtype=np.float64)
        ang = pos[:, None] * freq[None, :]
        cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=1)
        sin = np.concatenate([np.sin(ang), np.sin(ang)], axis=1)
        x1, x2 = H[:, :half], H[:, half:]
        rot = np.concatenate([-x2, x1], axis=1)
        y = (H.astype(np.float32) * cos.astype(np.float32) + rot.astype(np.float32) * sin.astype(np.float32)).astype(
            np.float64
        )
        prov[entry.get("src", "X")] = {"shape": [M, K], "decoded": X.reshape(-1).tolist()}
        prov[entry.get("weight", "Wqkv")] = {"shape": [K, N], "decoded": Wqkv.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    elif op == "conv2d":
        # AN IM2COL CONV IS A CONTRACTION OVER GATHERED WINDOWS, exactly as in the float engine, and it
        # reuses the runtime's OWN gather (`conv_im2col`) for the same reason that one does: a second
        # transcription of the window arithmetic is how a golden and a harness come to disagree about
        # which tap a window reads. The gather is a permutation-with-zero-fill and never arithmetic, so
        # it carries already-format-rounded IEEE values through untouched and its out-of-bounds tap is a
        # real +0.0 -- the same zero pad the integer and float engines apply.
        from merlin.runtime.commandbuffer import conv_im2col, conv_out_dims
        from merlin.runtime.tensor import Tensor

        if "maxpool" in [str(s) for s in (entry.get("epilogue") or [])]:
            # Fail closed rather than emit an unpooled reference for a capsule that declares pooling: a
            # golden that silently skipped the stage would agree with a backend that skipped it too.
            raise ValueError(
                f"SIMT conv2d golden: capsule {entry['name']!r} declares a maxpool "
                f"epilogue, which this engine does not model"
            )
        ci = int(entry.get("ci", entry.get("Cin", 4)))
        cout = int(entry.get("N", entry.get("Cout", dim)))
        Himg, Wimg = int(entry.get("Himg", 8)), int(entry.get("Wimg", 8))
        kh, kw = int(entry.get("kh", 3)), int(entry.get("kw", 3))
        stride = tuple(entry.get("stride", [1, 1]))
        padding = tuple(entry.get("padding", [0, 0, 0, 0]))
        dilation = tuple(entry.get("dilation", [1, 1]))
        layout = entry.get("layout", "nhwc")
        Ho, Wo = conv_out_dims(Himg, Wimg, kh, kw, stride, padding, dilation)
        Kdim = kh * kw * ci
        ifm_name, w_name = entry.get("ifm", "IFM"), entry.get("weight", "W")
        # Built as the [H*W, Ci] matrix the NHWC image is in row-major order (so the operand rigor gate
        # applies at a 2-D shape it understands); declared to the capsule at its rank-4 shape.
        ifm = synth(ifm_name, (Himg * Wimg, ci))
        Wt = synth(w_name, (Kdim, cout))
        cols = conv_im2col(
            Tensor((1, Himg, Wimg, ci), ifm.reshape(-1).tolist(), tok),
            kh=kh,
            kw=kw,
            ci=ci,
            stride=stride,
            padding=padding,
            dilation=dilation,
            layout=layout,
        )
        A = np.array(cols.data, dtype=np.float64).reshape(Ho * Wo, Kdim)
        y = (A.astype(np.float32) @ Wt.astype(np.float32)).astype(np.float64)
        epi = entry.get("epilogue", [])
        if "acc_scale" in epi:
            y = y * float(entry["acc_scale"])
        if "relu" in epi:
            y = np.maximum(y, 0.0)
        prov[ifm_name] = {"shape": [1, Himg, Wimg, ci], "decoded": ifm.reshape(-1).tolist()}
        prov[w_name] = {"shape": [Kdim, cout], "decoded": Wt.reshape(-1).tolist()}
        outputs[entry.get("out", "Y0")] = rnd_out(y)
    else:
        raise ValueError(f"no SIMT golden for op {op!r}")
    return outputs, prov
