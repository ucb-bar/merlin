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


def _specir(*, root: str | Path | None = None, use_legacy_environment: bool = True):
    from merlin.common.external_tools import specir_importable

    if root is None and use_legacy_environment:
        root = os.environ.get("SPECIR_ROOT") or _dotenv().get("SPECIR_ROOT")
    with specir_importable(root):
        from specir.oracle import dtypes as D
        from specir.oracle.refmodel import fp_reduce

    return D, fp_reduce


def _selected_specir_root(selected: dict) -> tuple[str | Path | None, bool]:
    """Resolve the same external oracle checkout for identity and evaluation."""
    model = selected["model"]
    root = model.get("source_root_path")
    from merlin.experiments.frozen_python import active_source_identity

    frozen_root = os.environ.get("MERLIN_PHASE0_NUMERICAL_MODEL_ROOT") if active_source_identity() else None
    if frozen_root:
        root = frozen_root
    elif "source_root_env" in model:
        root = os.environ.get(model["source_root_env"])
        if not root:
            raise ValueError(f"selected independent model requires {model['source_root_env']}")
    return root, selected["selection_status"] == "legacy_compatibility"


def specir_oracle_source_identity(selected: dict) -> dict:
    """Bind float goldens to the *external* numerical code actually selected.

    The Merlin writer and cache sources do not own SpecIR's arithmetic. A changed
    ``fp_reduce`` or codec must invalidate a golden even when every Merlin byte
    and every target fact is unchanged. Unreadable/non-source modules are fatal,
    never an excuse to serve an earlier cached numerical answer.
    """
    root, legacy = _selected_specir_root(selected)
    D, fp_reduce = _specir(root=root, use_legacy_environment=legacy)
    paths = {
        "specir.oracle.dtypes": Path(inspect.getfile(D)),
        "specir.oracle.refmodel": Path(inspect.getfile(fp_reduce)),
    }
    modules = {}
    for name, path in sorted(paths.items()):
        if path.suffix != ".py" or path.is_symlink() or not path.is_file():
            raise OSError(f"selected SpecIR oracle source for {name} is unavailable")
        modules[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"schema": "merlin.specir_oracle_source.v1", "modules": modules}


_SPECIR_FP8_ATTR = {"fp8_e4m3": "FP8_E4M3", "fp8_e5m2": "FP8_E5M2"}


def _specir_fp8(D, fmt_token: str):
    attr = _SPECIR_FP8_ATTR.get(fmt_token)
    if attr is None or not hasattr(D, attr):
        raise ValueError(
            f"specir refmodel has no fp8 format for operand dtype {fmt_token!r} (known: {sorted(_SPECIR_FP8_ATTR)})"
        )
    return getattr(D, attr)


def _det_fp8(D, name, shape, salt, fmt_token, d_fp8, *, palette=None, palette_dtype=None):
    """Structured, format-DERIVED operand bytes: distinct rows AND columns + asymmetric (so a wrong row
    stride / base offset / transposed load changes the output), spanning the fp8 format's representable
    range. Replaces the old 11-magnitude flat-hash fill (~6 distinct values, ~11/32 distinct rows) that hid
    those bug classes. See merlin.development.targetgen.corpus.operands."""
    from merlin.development.targetgen.corpus import operands as CO

    salt_int = sum((i + 1) * ord(c) for i, c in enumerate(f"{salt}|{name}")) or 1
    if palette is not None:
        from merlin.development.targetgen.corpus.operands import realize

        vals = realize(palette, name=name, shape=shape, dtype=palette_dtype or fmt_token)
    else:
        vals = None
    explicit = vals is not None
    if vals is None:
        vals = CO.operand_values(tuple(shape), fmt_token, salt_int)
    raw = [D.encode_float(v, d_fp8) for v in vals]
    # Self-enforcing rigor: fail generation loudly if the ENCODED bytes are not distinct-per-row/col +
    # asymmetric (e.g. a future palette/fill change, or an encode that collapsed distinct values). A weak
    # operand silently hides addressing/stride/transpose bugs — never let a regeneration ship one.
    if len(shape) == 2 and not explicit:
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


def float_semantics(entry, binding, *, semantics=None):
    """Resolve an explicit selection, or label the preserved historical arithmetic.

    Legacy callers retain their old answer, not an implied newly reviewed spec.
    New recipes carry this mapping on the entry, so cache identities bind it.
    """
    from merlin.target.spec.software import validate_numerical_semantics

    selected = semantics if semantics is not None else entry.get("numerical_semantics")
    if selected is None:
        return {
            "model": {"engine": "specir_fp_reduce"},
            "operand_dtype": binding.operand_dtype,
            "accumulator_dtype": "bf16",
            "readout_dtype": "bf16",
            "product_rounding": "accumulator_format",
            "rounding": "rne",
            "reduction_order": "index_sequential",
            "reduction_cadence": "per_step",
            "subnormal_operand_flush": binding.subnormal_operand_flush,
            "selection_status": "legacy_compatibility",
        }
    resolved = validate_numerical_semantics(selected)
    if resolved["model"]["engine"] != "specir_fp_reduce":
        raise ValueError("float golden requires the selected independent float model")
    if resolved["operand_dtype"] != binding.operand_dtype or resolved["accumulator_dtype"] != binding.accum_dtype:
        raise ValueError("selected float semantics disagree with the corpus operand/accumulator binding")
    if resolved["subnormal_operand_flush"] != binding.subnormal_operand_flush:
        raise ValueError("selected float semantics disagree with the corpus subnormal handling")
    resolved["selection_status"] = "explicit"
    return resolved


def _float_reducer(fp_reduce, fmt, *, order: str, cadence: str, rm: str):
    """Return an exact reducer and its bounded, capsule-local cache cleanup.

    Sequential per-step reduction is a fold over the *same* two-term SpecIR
    operation. Memoizing that operation on raw accumulator/addend bits preserves
    order, rounding, signed zero, NaN payload handling, and infinities; skipping
    zero or using a vector dot product would not. Other declared schedules go
    straight to SpecIR unchanged. The cache is scoped to one golden, not shared
    across different formats or oracle implementations.
    """

    def original(addends):
        return fp_reduce(addends, fmt, order=order, cadence=cadence, rm=rm)

    if order != "index_sequential" or cadence != "per_step":
        return original, lambda: None

    @lru_cache(maxsize=1 << 20)
    def step(acc: int, addend: int) -> int:
        return fp_reduce([acc, addend], fmt, order=order, cadence=cadence, rm=rm)

    def reduce(addends):
        if not addends:
            return original(addends)
        acc = addends[0]
        for addend in addends[1:]:
            acc = step(acc, addend)
        return acc

    # The two-term transition, exposed so a contraction can run the SAME ordered fold over every
    # output lane at once (see `ordered_fold`); the schedules above without one keep the scalar path.
    reduce.step = step
    return reduce, step.cache_clear


_CODE_LIMIT = 1 << 31


def ordered_fold(table, lhs_index, rhs_index, step):
    """``out[i][j] = fold(step, [table[lhs[i,p], rhs[p,j]] for p in range(K)])`` for every lane at once.

    Identical, lane by lane, to the scalar fold: the first product seeds the accumulator and each
    later product is folded in by the same two-term ``step`` in index order -- the reduction order
    is not touched, only the bookkeeping around it. Each step evaluates ``step`` once per DISTINCT
    (accumulator, addend) pair across the lanes, because the operand alphabet is small.
    ``table`` holds the rounded products of the distinct operand codes (``lhs_index``/``rhs_index``
    index into it); every value is a raw format code.
    """
    table = np.asarray(table, dtype=np.int64)
    if table.size and (table.min() < 0 or table.max() >= _CODE_LIMIT):
        raise ValueError("ordered_fold needs non-negative raw codes below 2**31")
    k = lhs_index.shape[1]
    acc = table[lhs_index[:, 0][:, None], rhs_index[0][None, :]]
    for p in range(1, k):
        addend = table[lhs_index[:, p][:, None], rhs_index[p][None, :]]
        keys = (acc << 32) | addend
        unique, inverse = np.unique(keys, return_inverse=True)
        folded = np.fromiter(
            (step(int(key >> 32), int(key & 0xFFFFFFFF)) for key in unique), dtype=np.int64, count=unique.size
        )
        if folded.size and (folded.min() < 0 or folded.max() >= _CODE_LIMIT):
            raise ValueError("ordered_fold produced a code outside the packable range")
        acc = folded[inverse.reshape(acc.shape)]
    return acc


def _float_golden(entry, binding, *, semantics=None):
    """Independent float golden in the explicitly selected numerical format and order."""
    selected = float_semantics(entry, binding, semantics=semantics)
    root, legacy = _selected_specir_root(selected)
    D, fp_reduce = _specir(root=root, use_legacy_environment=legacy)
    fmt_token = binding.operand_dtype  # e.g. "fp8_e4m3" — DERIVED, not assumed
    attr = {"bf16": "BF16", "fp16": "FP16", "f16": "FP16", "f32": "FP32", "fp32": "FP32"}.get(
        selected["accumulator_dtype"]
    )
    if attr is None or not hasattr(D, attr):
        raise ValueError(f"selected independent model lacks accumulator format {selected['accumulator_dtype']!r}")
    FP8, ACC = _specir_fp8(D, fmt_token), getattr(D, attr)
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
        raw, vals = _det_fp8(D, name, shape, salt, fmt_token, FP8, palette=entry.get("input_palette"))
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
        from merlin.development.targetgen.corpus import operands as CO

        if len(shape) == 2:
            _, vals = _det_fp8(
                D,
                name,
                shape,
                salt,
                fmt_token,
                ACC,
                palette=entry.get("input_palette"),
                palette_dtype=selected["accumulator_dtype"],
            )
        else:
            # A bias is a VECTOR, and `operand_values` shapes a matrix. Ask it for one row and flatten.
            # The per-row/per-column rigor `_det_fp8` enforces is not the right check for a vector --
            # it has one row -- but the part that still matters is: a constant bias is satisfied by a
            # kernel that adds any single number, and would not detect a broadcast along the wrong
            # axis. So distinctness ALONG the vector is asserted here instead of skipped.
            (n,) = shape
            salt_int = sum((i + 1) * ord(c) for i, c in enumerate(f"{salt}|{name}")) or 1
            from merlin.development.targetgen.corpus.operands import realize

            explicit = (
                realize(entry["input_palette"], name=name, shape=shape, dtype=selected["accumulator_dtype"])
                if entry.get("input_palette") is not None
                else None
            )
            vals = explicit if explicit is not None else list(CO.operand_values((1, n), fmt_token, salt_int))
            if explicit is None and n > 1 and len(set(vals)) < 2:
                raise AssertionError(
                    f"non-rigorous bias {name}({n},): every element is {vals[0]!r}, so a kernel that "
                    f"broadcast one value along the wrong axis would still match the golden"
                )
        prov[name] = {"shape": list(shape), "decoded": vals}
        return vals

    def rnd(x):
        return D.round_to_format(x, ACC, selected["rounding"])

    #: MEMOIZED PRODUCT. ``rnd(dec(a) * dec(b))`` is a pure function of the OPERAND CODE PAIR, so
    #: caching it on that pair is bit-identical by construction -- same inputs, same function, same
    #: answer -- rather than an approximation traded for speed. It is worth doing because the operand
    #: fill draws from a small deterministic alphabet, so a deep-K contraction re-derives the same few
    #: products millions of times.
    #:
    #: THE REDUCTION ORDER IS DELIBERATELY NOT TOUCHED. The selected sequential/per-step schedule
    #: remains an ordered fold of SpecIR's own two-term transition, memoized on raw bits. A numpy dot
    #: product or a skipped zero would change IEEE behavior; other schedules use SpecIR unchanged.
    _prod_cache: dict = {}

    reduce, clear_reduce_cache = _float_reducer(
        fp_reduce,
        ACC,
        order=selected["reduction_order"],
        cadence=selected["reduction_cadence"],
        rm=selected["rounding"],
    )

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
        step = getattr(reduce, "step", None)
        if step is not None and k > 0:
            lhs = np.asarray(a_raw, dtype=np.int64).reshape(m, k)
            rhs = np.asarray(w_raw, dtype=np.int64).reshape(k, n)
            lhs_codes, lhs_index = np.unique(lhs, return_inverse=True)
            rhs_codes, rhs_index = np.unique(rhs, return_inverse=True)
            table = [[_prod(int(a), int(b)) for b in rhs_codes] for a in lhs_codes]
            return ordered_fold(table, lhs_index.reshape(m, k), rhs_index.reshape(k, n), step).tolist()
        out = [[0] * n for _ in range(m)]
        for i in range(m):
            a_row = a_raw[i * k : (i + 1) * k]
            for j in range(n):
                prods = [_prod(a_row[p], w_raw[p * n + j]) for p in range(k)]
                out[i][j] = reduce(prods)
        return out

    def floats(y):
        return [[D.decode_float(v, ACC) for v in row] for row in y]

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
            y = [[rnd(D.decode_float_exact(v, ACC) * s) for v in row] for row in y]
        if "relu" in epi:
            y = [[v if D.decode_float(v, ACC) > 0 else 0 for v in row] for row in y]
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
        y = [[rnd(D.decode_float_exact(v, ACC) + b[j]) for j, v in enumerate(row)] for row in y]
        if "relu" in entry.get("epilogue", []):
            y = [[v if D.decode_float(v, ACC) > 0 else 0 for v in row] for row in y]
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
        K = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry.get("N_tiles", 1) * dim)
        w = reg(entry["weight"], (K, N))
        for m in entry["matmuls"]:
            M = m.get("M", m.get("M_tiles", 1) * dim)
            a = reg(m["lhs"], (M, K))
            outputs[m["out"]] = floats(mm(a, (M, K), w, (K, N)))
    elif op == "attention_qk":
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        Kd = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry["N_tiles"] * dim if "N_tiles" in entry else M)
        q = reg(entry.get("q", "Q"), (M, Kd))
        k = reg(entry.get("k", "K"), (N, Kd))
        kt = [0] * (N * Kd)
        for i in range(N):
            for j in range(Kd):
                kt[j * N + i] = k[i * Kd + j]
        outputs[entry.get("out", "Y0")] = floats(mm(q, (M, Kd), kt, (Kd, N)))
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
        from merlin.runtime.engine.command_buffer import Tensor, conv_im2col, conv_out_dims

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
            y = [[rnd(D.decode_float_exact(v, ACC) * s) for v in row] for row in y]
        if "relu" in epi:
            y = [[v if D.decode_float(v, ACC) > 0 else 0 for v in row] for row in y]
        outputs[entry.get("out", "Y0")] = floats(y)
    else:
        raise ValueError(f"no float golden for op {op!r}")
    clear_reduce_cache()
    return outputs, prov


def _mx_ref(selected: dict | None = None):
    """Import mlc's ``validate/mx_ref.py`` BY FILE PATH (like the specir import) so we do NOT trigger
    ``mlc/validate/__init__.py`` (which carries concurrent work and heavy imports)."""
    if selected is not None:
        from merlin.target.spec.software import validate_numerical_semantics

        selected = validate_numerical_semantics(selected)
        if selected["model"]["engine"] != "mx_block_reference":
            raise ValueError("microscaling golden requires the selected independent block model")
        root, _ = _selected_specir_root({**selected, "selection_status": "explicit"})
    else:
        # Preserve the legacy selection while honoring an existing frozen source
        # projection. The explicit recipe path above never guesses another root.
        root, _ = _selected_specir_root({"model": {}, "selection_status": "legacy_compatibility"})
        root = root or os.environ.get("MERLIN_MLC_DIR") or _dotenv().get("MERLIN_MLC_DIR")
    if not root:
        raise FileNotFoundError("the independent block model requires an explicitly selected source root")
    path = Path(root) / "mlc" / "validate" / "mx_ref.py"
    if not path.exists():
        raise FileNotFoundError(f"mx_ref not found at {path} (set MERLIN_MLC_DIR to the mlc modeling root)")
    spec = importlib.util.spec_from_file_location("merlin_mx_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if selected is not None and getattr(mod, "GROUP", None) != selected["block_size"]:
        raise ValueError("selected numerical block_size disagrees with the independent block model")
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


def _simt_golden(entry, binding):
    """SIMT (CVFPU) golden in ordinary IEEE float — fp32 accumulate, format-rounded operands. Covers the
    matmul / attention / rmsnorm shapes; independent of any accelerator model (the SIMT cores do plain IEEE
    math). Operands are format-derived + rigor-gated."""
    from merlin.common.fp8_formats import canonical_float
    from merlin.development.targetgen.corpus import operands as CO

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
        from merlin.development.targetgen.corpus.operands import realize

        explicit = (
            realize(entry["input_palette"], name=name, shape=shape, dtype=tok)
            if entry.get("input_palette") is not None
            else None
        )
        vals = explicit if explicit is not None else CO.operand_values(shape, tok, _salt(entry["name"], name))
        problems = CO.rigor_findings(vals, shape) if explicit is None else []
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
        M = entry.get("M", entry.get("M_tiles", 1) * dim)
        Kd = entry.get("K", entry.get("K_tiles", 1) * dim)
        N = entry.get("N", entry["N_tiles"] * dim if "N_tiles" in entry else M)
        Q = synth(entry.get("q", "Q"), (M, Kd))
        Kk = synth(entry.get("k", "K"), (N, Kd))
        y = (Q.astype(np.float32) @ Kk.astype(np.float32).T).astype(np.float64)
        prov[entry.get("q", "Q")] = {"shape": [M, Kd], "decoded": Q.reshape(-1).tolist()}
        prov[entry.get("k", "K")] = {"shape": [N, Kd], "decoded": Kk.reshape(-1).tolist()}
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
        from merlin.runtime.engine.command_buffer import Tensor, conv_im2col, conv_out_dims

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
