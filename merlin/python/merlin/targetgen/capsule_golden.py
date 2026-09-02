"""Independent numeric golden for capsules + exact-equality comparison.

The golden is computed from the **capsule's declared operation** (not from the package's emitted
command buffer), so a wrong command buffer is caught by ``golden != reference(cb)``. The arithmetic
reuses the authoritative, dependency-free :class:`~merlin.runtime.tensor.Tensor` primitives (exact
int matmul, round-half-even f32 ``acc_scale``, saturating i8 cast) so the golden is bit-identical in
*semantics* to the reference/simulate/oracle paths while being structurally independent.

Leaves are materialized via :meth:`Tensor.deterministic` — the SAME function the command-buffer
materializer and the device harness use — so L0 (this golden) cannot silently diverge from L2/L3
on leaf data (the single-source-of-truth rule).

torch is not available in this environment; goldens are labeled ``merlin_tensor_int`` (numpy is used
only for the im2col gather). A capsule may cross-check against torch in ``model_slice_export`` when
torch is present.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from merlin.runtime.tensor import Tensor


# --------------------------------------------------------------------------------------------
# leaf materialization (single source of truth)
# --------------------------------------------------------------------------------------------
def materialize_capsule_leaves(capsule: dict) -> dict[str, Tensor]:
    """Materialize the capsule's declared leaf tensors deterministically by name."""
    env: dict[str, Tensor] = {}
    for spec in capsule.get("inputs", []):
        if spec.get("role") in ("input", "weight", "bias"):
            env[spec["name"]] = Tensor.deterministic(
                spec["name"], tuple(spec["shape"]), spec.get("dtype", "i8"))
    return env


# --------------------------------------------------------------------------------------------
# im2col (shared by golden + runner harness for conv2d)
# --------------------------------------------------------------------------------------------
from merlin.runtime.commandbuffer import (  # noqa: E402  (single source of truth)
    apply_pool_stage, conv_im2col, conv_out_dims)


def im2col(ifm: Tensor, ci: int, kh: int, kw: int, *, stride, padding, dilation,
           layout: str = "nhwc") -> Tensor:
    """Thin wrapper over the runtime's canonical :func:`conv_im2col` (shared with the harness)."""
    return conv_im2col(ifm, kh=kh, kw=kw, ci=ci, stride=stride, padding=padding,
                       dilation=dilation, layout=layout)


# --------------------------------------------------------------------------------------------
# epilogue application (matches runtime/reference.py)
# --------------------------------------------------------------------------------------------
# Back-compat requant DEFAULTS — the numerics a capsule inherits when it declares no override. These
# are DEFAULTS, not the only path: a capsule that declares a different integer shift, a different
# acc_scale rounding mode, or a different narrow output dtype gets THAT value instead. Keeping them as
# named, overridable parameters (sourced from the capsule's declared operation attributes) rather than
# inline literals is what lets a differently-rounding / differently-narrowing integer target be
# expressed without editing this function — the point of the de-overfit. A capsule that omits them
# reproduces the historical behavior byte-for-byte.
_DEFAULT_REQUANT_SHIFT = 4              # round-half-up integer shift when a ``requant`` stage omits it
_DEFAULT_ACC_SCALE_ROUND = "half_even"  # gemmini ACC_SCALE / ROUND_NEAR_EVEN float readout rounding


def _int_dtype_bits(dtype: str) -> tuple[int, bool] | None:
    """Parse an integer dtype name (``i8`` / ``i16`` / ``i4`` / ``u8`` ...) to ``(bitwidth, signed)``.

    Returns ``None`` for a non-integer dtype (``f32`` / ``bf16``) or an unparseable name, so the caller
    leaves the tensor un-narrowed. Parsed structurally (leading kind char + digit suffix), never by a
    literal dtype table."""
    if not dtype:
        return None
    kind, digits = dtype[0], dtype[1:]
    if kind not in ("i", "u") or not digits.isdigit():
        return None
    return int(digits), kind == "i"


def _narrow_to_dtype(t: Tensor, dtype: str) -> Tensor:
    """Saturating cast of the i32 accumulator to the capsule's DECLARED narrow integer output dtype.

    The narrow width (hence the saturation range) is DERIVED from ``output_dtype`` — not assumed to be
    i8. ``i8`` is routed through :meth:`Tensor.to_i8` so its result is byte-identical to the historical
    path; any other narrow integer width (``i16`` / ``i4`` / ``u8`` ...) saturates to the range derived
    from its bitwidth. A wide accumulator dtype (``i32`` / ``i64``) or a non-integer dtype passes
    through unchanged — there is nothing to narrow."""
    parsed = _int_dtype_bits(dtype)
    if parsed is None:
        return t
    bits, signed = parsed
    if bits >= 32:                      # wide accumulator width — no narrowing
        return t
    if dtype == "i8":                   # byte-identical to the historical to_i8() path
        return t.to_i8()
    if signed:
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        lo, hi = 0, (1 << bits) - 1
    out = [lo if x < lo else (hi if x > hi else x) for x in t.data]
    return Tensor(t.shape, out, dtype)


def _apply_epilogue(t: Tensor, attrs: dict, env: dict[str, Tensor]) -> Tensor:
    for stage in attrs.get("epilogue", []):
        if stage in ("bias_add", "bias"):
            bname = attrs.get("bias")
            if bname:
                t = t.add_bias(env[bname])
        elif stage == "requant":
            t = t.requant(int(attrs.get("requant_shift", _DEFAULT_REQUANT_SHIFT)))
        elif stage == "acc_scale":
            # The acc_scale readout rounding mode is a named, overridable parameter (default the gemmini
            # round-half-even). The Tensor engine reproduces half-even exactly; a capsule that declares a
            # different mode is a datapath this integer engine cannot reproduce, so fail CLOSED (surface
            # it) rather than silently applying half-even to a target that rounds otherwise.
            mode = attrs.get("requant_round", _DEFAULT_ACC_SCALE_ROUND)
            if mode != "half_even":
                raise ValueError(
                    f"acc_scale requant_round={mode!r} is declared but the integer Tensor engine only "
                    f"reproduces 'half_even'; grade this datapath against an independent golden.yaml "
                    f"instead of silently rounding half-even")
            t = t.requant_acc_scale(float(attrs.get("acc_scale", 1.0)))
        elif stage == "relu":
            t = t.relu()
        elif stage == "maxpool":
            # The pooling epilogue this target FUSES into the store/conv path. The geometry rides on the
            # operation's attributes (``pool_in_dims``/``pool_size``/``pool_stride``/``pool_padding``,
            # the ABI's ``orows``/``ocols``/``pool_size``/``pool_stride``/``upad``/``lpad``) rather than
            # being inferred, because by this point ``t`` is 2-D for EVERY op that reaches here: a
            # matmul commits ``[M, N]`` and the conv branch above has already contracted its im2col
            # matrix down to ``[N*Ho*Wo, Co]``. Neither shape says what spatial extent its rows
            # unflatten to. Parsed by the runtime's one ``pool_params``, so this golden and the
            # reference/simulator it is compared against cannot disagree about the window.
            t = apply_pool_stage(
                t, stage, attrs,
                op=f"golden epilogue of {attrs.get('out', 'the output')!r}")
        else:
            # FAIL CLOSED ON AN EPILOGUE STAGE THIS ENGINE CANNOT APPLY. There was no terminal branch
            # here, so an unrecognised stage was silently skipped and the capsule shipped a golden that
            # did not match the semantics it declared -- a wrong answer, produced quietly, and one the
            # numeric gate then enforced. Concretely: a capsule declaring a pooling epilogue would ship
            # an UNPOOLED golden while its cover credited it for the pooled family. Refusing names the
            # stage and the stages that exist, so the fix is to implement it here rather than to
            # discover months later that a cell was covered by arithmetic nobody performed.
            raise ValueError(
                f"epilogue stage {stage!r} is declared but the integer Tensor engine does not implement "
                f"it (implemented: bias_add/bias, requant, acc_scale, relu, maxpool). Implement it here -- "
                f"and in "
                f"the reference and simulator, which grade against this -- rather than shipping a golden "
                f"that skips it")
    return _narrow_to_dtype(t, attrs.get("output_dtype", "i32"))


def _rmsnorm(x: Tensor, gamma: Tensor, eps: float) -> Tensor:
    """Row RMSNorm: ``x * rsqrt(mean(x^2, -1) + eps) * gamma``, gamma broadcast over rows.

    One implementation serves EVERY capsule that declares ``rmsnorm`` (and composes into the fused
    norm+projection ops) — the oracle stays a general definition of the op rather than a per-capsule
    answer file. Accumulated in double and returned f32, so the reference is not tied to one device
    accumulation order (the capsule's tolerance policy absorbs that difference)."""
    rows, cols = x.shape
    g = [float(v) for v in gamma.data]
    if len(g) != cols:
        raise ValueError(f"rmsnorm: gamma has {len(g)} element(s), expected {cols}")
    out: list[float] = []
    for r in range(rows):
        row = [float(v) for v in x.data[r * cols:(r + 1) * cols]]
        inv = 1.0 / math.sqrt(sum(v * v for v in row) / cols + eps)
        out.extend(v * inv * g[c] for c, v in enumerate(row))
    return Tensor((rows, cols), out, "f32")


def _rope_rotate_half(t: Tensor, theta: float) -> Tensor:
    """Rotary position embedding, HALF-SPLIT ("rotate_half") convention:

        half = d/2;  freq = theta ** -(arange(half)/half);  ang = pos[:, None] * freq[None, :]
        cos = cat(cos(ang), cos(ang));  sin = cat(sin(ang), sin(ang))
        out = x * cos + cat(-x[..., half:], x[..., :half]) * sin

    The convention is not arbitrary and must not be guessed: it is the one the corpus's own PyTorch
    reference slices use, so a fused rope capsule graded here agrees with the independent torch golden
    of the standalone rope capsule. Row index is the position."""
    rows, d = t.shape
    if d % 2:
        raise ValueError(f"rope: head dim {d} is odd; the half-split convention needs an even width")
    half = d // 2
    freq = [theta ** (-(i / half)) for i in range(half)]
    out: list[float] = []
    for p in range(rows):
        row = [float(v) for v in t.data[p * d:(p + 1) * d]]
        cos = [math.cos(p * f) for f in freq]
        sin = [math.sin(p * f) for f in freq]
        for c in range(d):
            # cos/sin are the half-length tables duplicated across both halves.
            cs, sn = cos[c % half], sin[c % half]
            partner = -row[c + half] if c < half else row[c - half]
            out.append(row[c] * cs + partner * sn)
    return Tensor((rows, d), out, "f32")


def _softmax_rows(t: Tensor) -> Tensor:
    """Row-wise softmax, max-subtracted (the standard numerically-stable form)."""
    rows, cols = t.shape
    out: list[float] = []
    for r in range(rows):
        row = [float(v) for v in t.data[r * cols:(r + 1) * cols]]
        mx = max(row)
        ex = [math.exp(v - mx) for v in row]
        s = sum(ex)
        out.extend(e / s for e in ex)
    return Tensor((rows, cols), out, "f32")


def _attention(q: Tensor, k: Tensor, v: Tensor, *, scale: float | None = None,
               causal: bool = False, softcap: float | None = None) -> Tensor:
    """Scaled dot-product attention: ``softmax(softcap?(Q@K^T * scale) + causal_mask) @ V``.

    ONE definition covering the plain, causal, soft-capped and block-scaled (MX) attention capsules —
    an MX capsule differs only in how its operands were decoded upstream, not in the math, so routing
    them through the same function keeps a fused/quantized variant from silently disagreeing with the
    plain one. ``K`` is stored row-per-key ([n, d]), so the contraction is over the trailing dim of
    both operands. Default scale is ``1/sqrt(d)``; ``causal`` masks strictly-above-diagonal.
    """
    m, d = q.shape
    n, d2 = k.shape
    if d != d2:
        raise ValueError(f"attention: head-dim mismatch q{q.shape} vs k{k.shape}")
    sc = (1.0 / math.sqrt(d)) if scale is None else float(scale)
    scores: list[float] = []
    for i in range(m):
        qi = [float(x) for x in q.data[i * d:(i + 1) * d]]
        for j in range(n):
            kj = k.data[j * d:(j + 1) * d]
            s = sum(a * float(b) for a, b in zip(qi, kj)) * sc
            if softcap is not None:
                s = float(softcap) * math.tanh(s / float(softcap))
            if causal and j > i:
                s = -math.inf          # masked BEFORE softmax; exp(-inf) == 0 contributes nothing
            scores.append(s)
    p = _softmax_rows(Tensor((m, n), scores, "f32"))
    return p.matmul(v)


def _nested_list(t: Tensor) -> Any:
    """Row-major nested lists for a 2-D or 3-D tensor. :meth:`Tensor.to_list` handles only rank 2
    (``rows, cols = self.shape``), so a batched result must be shaped here rather than crashing the
    oracle on a conformant 3-D output."""
    if len(t.shape) == 2:
        return t.to_list()
    if len(t.shape) == 3:
        b, m, n = t.shape
        return [[[t.data[x * m * n + i * n + j] for j in range(n)] for i in range(m)]
                for x in range(b)]
    raise ValueError(f"golden: cannot shape a rank-{len(t.shape)} result into nested lists")


def _batched_matmul(a: Tensor, w: Tensor) -> Tensor:
    """``O[b] = A[b] @ W[b]`` over the leading batch dim (independent per-batch 2-D contractions)."""
    if len(a.shape) != 3 or len(w.shape) != 3:
        raise ValueError(f"batched matmul needs 3-D operands, got {a.shape} and {w.shape}")
    batch, m, k = a.shape
    wb, k2, n = w.shape
    if batch != wb or k != k2:
        raise ValueError(f"batched matmul shape mismatch: {a.shape} @ {w.shape}")
    out: list[float] = []
    for b in range(batch):
        asl = Tensor((m, k), a.data[b * m * k:(b + 1) * m * k], a.dtype)
        wsl = Tensor((k, n), w.data[b * k * n:(b + 1) * k * n], w.dtype)
        out.extend(asl.matmul(wsl).data)
    return Tensor((batch, m, n), out, "f32")


def _transpose2d(t: Tensor) -> Tensor:
    r, c = t.shape
    out = [0] * (r * c)
    for i in range(r):
        for j in range(c):
            out[j * r + i] = t.data[i * c + j]
    return Tensor((c, r), out, t.dtype)


# --------------------------------------------------------------------------------------------
# golden source resolution (recompute vs read an INDEPENDENT golden)
# --------------------------------------------------------------------------------------------
def _load_golden_yaml(capsule_dir: str | Path | None) -> dict | None:
    """Read the capsule's ``golden.yaml`` (the independent oracle's masked answer key), if present."""
    if not capsule_dir:
        return None
    import yaml
    gy = Path(capsule_dir) / "golden.yaml"
    if not gy.is_file():
        return None
    return yaml.safe_load(gy.read_text(encoding="utf-8"))


def canonical_input_raws(capsule: dict, capsule_dir: str | Path | None = None) -> dict[str, bytes]:
    """The EXACT per-leaf input bytes the independent float golden was computed with, keyed by tensor
    name — read from ``golden.yaml`` ``oracle_provenance.inputs[name].fp8_raw_hex`` (a flat row-major
    list of per-element raw hex). This is the canonical device preload for a float target's program
    oracle: it must run on the SAME operands the golden used (the atlas exact-fp8 palette), NOT the
    integer-engine ``Tensor.deterministic`` 0..3 fill (whose bytes-as-fp8 collapse to subnormal/zero).

    When a spec records no raw hex but DOES record decoded values, the bytes are ENCODED from those values
    into the tensor's own declared dtype (:func:`merlin.runtime.fp8_formats.encode_bytes`). Only the fp8
    palette was ever written as raw hex, so every bf16/fp16/f32 capsule returned nothing here and its
    kernel was preloaded with NOTHING — it ran on empty DRAM and its output was graded as though the
    kernel had failed to store. The encoding is the inverse of the shared decoder, so a value that came
    out of the golden's own dtype round-trips exactly; a dtype the format table does not know, or a
    sub-byte format whose packing is the caller's choice, yields nothing (fail closed) rather than a
    guessed byte image.

    Empty for integer capsules (which record no raws and are reproduced on the Tensor engine)."""
    gy = _load_golden_yaml(capsule_dir)
    if not gy:
        return {}
    ins = ((gy.get("oracle_provenance", {}) or {}).get("inputs", {})) or {}
    out: dict[str, bytes] = {}
    for name, spec in ins.items():
        # An ``inputs`` entry is a per-tensor spec dict; a block-scaled datapath (mxfp8) also records
        # NON-tensor provenance under the same map (E8M0 block-scale code arrays as lists, a scale_example
        # dict without raw bytes). Only real tensor specs carry raw device bytes — skip the rest, never
        # ``.get`` on a non-dict (that raised ``AttributeError: 'list' object has no attribute 'get'``).
        if not isinstance(spec, dict):
            continue
        raws = spec.get("fp8_raw_hex") or spec.get("raw_hex")
        if raws:
            out[name] = bytes(int(x, 16) & 0xFF for x in raws)
    for name, spec in _decoded_inputs(ins).items():
        if name in out:
            continue                                   # recorded device bytes always win over re-encoding
        enc = _encode_leaf(capsule, name, spec)
        if enc is not None:
            out[name] = enc
    return out


def _leaf_dtype(capsule: dict, name: str) -> str | None:
    """The declared dtype of leaf ``name``, read off the capsule's own input list (never inferred)."""
    for t in (capsule.get("inputs") or []):
        if t.get("name") == name:
            return t.get("dtype")
    return None


def _encode_leaf(capsule: dict, name: str, values: list) -> bytes | None:
    """Device bytes for one leaf's decoded values, or None when this capsule cannot supply them: no
    declared dtype, a dtype outside the shared float table (integer capsules included — they are
    reproduced on the Tensor engine, not preloaded), or a sub-byte format. Never guesses a width."""
    dtype = _leaf_dtype(capsule, name)
    if not dtype:
        return None
    import numpy as _np

    from merlin.runtime import fp8_formats as _ff
    try:
        raw = _ff.encode_bytes(values, dtype)
        width = _ff.storage_bits(dtype) // 8
        codes = _np.frombuffer(raw, dtype=f"<u{width}").astype(_np.uint32)
        back = _ff._decode(codes, dtype)
    except (KeyError, ValueError):
        return None
    want = _np.asarray(values, dtype=_np.float32).ravel()
    # REFUSE a lossy re-encoding. The oracle must run on the SAME operands the golden used; if the
    # recorded values do not sit exactly on this dtype's grid (a golden that stored pre-quantization
    # floats for a narrow format), encoding them hands the device operands the golden never saw and
    # grades the kernel against the wrong reference. No preload is the honest answer there.
    if back.shape != want.shape or not _np.array_equal(back, want):
        return None
    return raw


def _decoded_inputs(ins: dict) -> dict[str, list]:
    """``{name: flat row-major values}`` for the tensor specs that record decoded values."""
    out: dict[str, list] = {}
    for name, spec in ins.items():
        if not isinstance(spec, dict):
            continue
        decoded = spec.get("decoded")
        if decoded is None:
            continue
        flat: list = []
        stack = [decoded]
        while stack:                                   # flatten any nesting to row-major order
            cur = stack.pop(0)
            if isinstance(cur, list):
                stack = list(cur) + stack
            else:
                flat.append(cur)
        out[name] = flat
    return out


def canonical_input_values(capsule: dict, capsule_dir: str | Path | None = None) -> dict[str, dict]:
    """The DECODED per-leaf operand values the independent float golden was computed with, keyed by tensor
    name — read from ``golden.yaml`` ``oracle_provenance.inputs[name]`` (``decoded`` = a flat row-major list
    of numbers, plus ``shape``). Unlike :func:`canonical_input_raws` (byte-level ``fp8_raw_hex`` for the
    palette-preload program oracle), this returns the actual numeric operands a self-contained kernel harness
    embeds. Each value is ``{"shape": [r, c], "values": [...]}``. Empty when the golden records no decoded
    inputs (e.g. an integer capsule reproduced on the Tensor engine)."""
    gy = _load_golden_yaml(capsule_dir)
    if not gy:
        return {}
    ins = ((gy.get("oracle_provenance", {}) or {}).get("inputs", {})) or {}
    out: dict[str, dict] = {}
    for name, spec in ins.items():
        if not isinstance(spec, dict):   # skip non-tensor provenance (mxfp8 block-scale code arrays, examples)
            continue
        decoded = spec.get("decoded")
        if decoded is not None:
            # Normalize to the documented FLAT row-major list. Most goldens store ``decoded`` flat, but a
            # specir golden stores it as a 2D nested list (rows) — leaving it nested makes every consumer
            # (each does ``float(x)`` per element) crash on a row. Flatten so the contract holds regardless
            # of the golden generator.
            out[name] = {"shape": list(spec.get("shape") or []), "values": _flatten_row_major(decoded)}
    return out


def materialized_input_values(capsule: dict) -> dict[str, dict]:
    """The capsule's DETERMINISTIC leaf operands, in the :func:`canonical_input_values` shape.

    The stimulus a RECOMPUTED golden was evaluated on. A capsule that ships no ``golden.yaml`` has no
    recorded operands, so :func:`canonical_input_values` is empty and the runner previously attached
    nothing — leaving a program-oracle target with no operands to build its kernel harness from, which
    fails a CORRECT backend on an output it was never given the inputs to compute. The golden and the
    device must run on ONE stimulus; this exposes the recompute path's own so the runner can attach it.

    Element type follows the declared leaf dtype (an integer leaf stays integral) so a consumer that
    embeds these operands emits the same literals the Tensor engine reduced over.
    """
    out: dict[str, dict] = {}
    for name, t in materialize_capsule_leaves(capsule).items():
        integral = str(t.dtype).startswith(("i", "u"))
        vals = [int(v) if integral else float(v) for v in t.data]
        out[name] = {"shape": list(t.shape), "values": vals}
    return out


def _flatten_row_major(x: object) -> list:
    """Flatten an arbitrarily-nested list to a single row-major list of scalars; a non-list passes through
    as a 1-element list. Idempotent on an already-flat list."""
    if not isinstance(x, (list, tuple)):
        return [x]
    flat: list = []
    for e in x:
        if isinstance(e, (list, tuple)):
            flat.extend(_flatten_row_major(e))
        else:
            flat.append(e)
    return flat


def mx_scale_codes(capsule: dict, capsule_dir: str | Path | None = None) -> dict[str, list[int]]:
    """The E8M0 per-block scale codes a microscaling (mxfp8) golden used, keyed by their provenance name
    (e.g. ``SA_e8m0_codes`` / ``SB_e8m0_codes``) — read from ``golden.yaml`` ``oracle_provenance.inputs``,
    where they sit alongside the tensor operands as a list (NOT a per-tensor dict). Each is flattened to a
    row-major list of int codes (one exponent per K group). Empty for a non-block-scaled capsule. The block
    scale is a device operand the accelerator kernel stages into its scale SRAM, separate from the fp8
    element bytes — the two together reproduce the block-scaled matmul the golden records."""
    gy = _load_golden_yaml(capsule_dir)
    if not gy:
        return {}
    ins = ((gy.get("oracle_provenance", {}) or {}).get("inputs", {})) or {}
    out: dict[str, list[int]] = {}
    for name, spec in ins.items():
        if not isinstance(spec, list):   # scale-code arrays are lists; tensor specs are dicts (skipped here)
            continue
        flat: list[int] = []
        for row in spec:
            if isinstance(row, list):
                flat.extend(int(x) & 0xFF for x in row)
            else:
                flat.append(int(row) & 0xFF)
        out[name] = flat
    return out


def mx_operands(capsule: dict, capsule_dir: str | Path | None = None) -> dict | None:
    """The block-scaled MX matmul operand bundle a microscaling golden records — the quantized element
    ``operand_codes`` (A/B device bytes, shapes, fmt, M/N/K/G, fp6 LUTs) PLUS the ``SA``/``SB`` E8M0 block
    scales — read from ``golden.yaml`` ``oracle_provenance.inputs``. Returns ``None`` for a non-MX capsule.

    These operands are corpus-seeded (the E8M0 scales are a function of the capsule-name salt, not the
    operand values — :func:`corpus_operands.e8m0_scale_codes`), so they exist ONLY in the golden and cannot
    be reconstructed from the decoded-float workload. The reference MX kernel bakes them; a general backend
    could not (this is the public-capsule known-good baseline, masked for hidden capsules)."""
    gy = _load_golden_yaml(capsule_dir)
    if not gy:
        return None
    ins = ((gy.get("oracle_provenance", {}) or {}).get("inputs", {})) or {}
    # Batched MX matmul (B independent tiles stacked): pass the per-batch codes through; the reference
    # emitter packs them block-diagonally into a single MX tile.
    bc = ins.get("batched_codes")
    if isinstance(bc, dict) and bc.get("batches"):
        return {"fmt": bc.get("fmt", "fp8_e4m3"), "batched": True,
                "B": bc["B"], "M": bc["M"], "H": bc["H"], "N": bc["N"],
                "stacked_out_shape": bc.get("stacked_out_shape"), "batches": bc["batches"]}
    # Flash attention (fused MX): O = mx_matmul(softmax(mx_matmul(Q,Kᵀ)/scale), V). The golden decomposes it
    # into the two MX matmul stages (each with its own operand codes + E8M0 scales) plus the softmax scale +
    # the P (softmax output) requant scales, so the reference kernel can chain two mxgemm calls with an
    # on-device softmax+requant between them. Passed through as ``flash``; the emitter builds the fused kernel.
    ac = ins.get("attention_codes")
    if isinstance(ac, dict) and ac.get("qk_stage") and ac.get("pv_stage"):
        return {"fmt": ac.get("fmt", "fp8_e4m3"), "flash": True,
                "M": ac["M"], "H": ac["H"], "Skv": ac["Skv"], "Dv": ac["Dv"],
                "att_scale": ac["att_scale"], "softcap": ac.get("softcap"),
                "qk_stage": ac["qk_stage"], "pv_stage": ac["pv_stage"],
                "SA_q": ac["SA_q"], "SB_k": ac["SB_k"], "SB_v": ac["SB_v"], "SA_p": ac.get("SA_p"),
                "P_decoded": ac.get("P_decoded")}
    oc = ins.get("operand_codes")
    if not isinstance(oc, dict) or "A_bytes" not in oc:
        return None
    scales = mx_scale_codes(capsule, capsule_dir)
    # SA/SB laid out [K/32, lane]; take them as [groups][lanes] rows straight from the golden lists.
    sa = ins.get("SA_e8m0_codes") or ins.get("SA_q_e8m0_codes")
    sb = ins.get("SB_e8m0_codes") or ins.get("SB_k_e8m0_codes")
    if sa is None or sb is None:
        return None
    return {
        "fmt": oc["fmt"], "M": oc["M"], "N": oc["N"], "K": oc["K"], "G": oc.get("G", 0),
        "A_bytes": oc["A_bytes"], "B_bytes": oc["B_bytes"],
        "A_shape": oc.get("A_shape"), "B_shape": oc.get("B_shape"),
        "SA": sa, "SB": sb, "lutA": oc.get("lutA"), "lutB": oc.get("lutB"),
        "_scale_codes": scales,
    }


def golden_source(capsule: dict, capsule_dir: str | Path | None = None) -> str:
    """The golden's PROVENANCE: ``merlin_tensor_int`` when it is (re)computed on the integer
    :class:`~merlin.runtime.tensor.Tensor` engine, or the INDEPENDENT source declared in the capsule's
    ``golden.yaml`` (e.g. ``specir_refmodel_fp8_bf16`` for the atlas fp8-e4m3 -> bf16 path). Defaults to
    ``merlin_tensor_int`` when no ``golden.yaml`` / source is present, so integer capsules keep recomputing."""
    if capsule_dir is None:
        capsule_dir = capsule.get("__dir__")
    src = (_load_golden_yaml(capsule_dir) or {}).get("golden_source")
    return src if (src and src != "merlin_tensor_int") else "merlin_tensor_int"


def is_independent_float_golden(capsule: dict, capsule_dir: str | Path | None = None) -> bool:
    """True iff the capsule is graded against an INDEPENDENT golden under a FLOAT compare policy — the
    atlas fp8/bf16 case: the integer Tensor engine cannot recompute the float datapath, so the golden is
    READ from ``golden.yaml`` and the integer reference/simulate tiers do not apply. False for every
    integer capsule (gemmini / ``exact_int`` / ``golden_source: merlin_tensor_int``) and for a float
    capsule that ships no independent ``golden.yaml`` (e.g. muon), which keep the recompute path."""
    compare = (capsule.get("numeric_policy") or {}).get("compare", "exact_int")
    float_policy = compare not in ("exact_int", "exact")
    return float_policy and golden_source(capsule, capsule_dir) != "merlin_tensor_int"


# --------------------------------------------------------------------------------------------
# golden dispatch
# --------------------------------------------------------------------------------------------
def golden(capsule: dict, capsule_dir: str | Path | None = None) -> dict[str, list]:
    """Return the capsule's expected outputs (name -> nested list).

    For an INDEPENDENT float golden (float compare policy + ``golden.yaml`` ``golden_source`` !=
    ``merlin_tensor_int``, e.g. atlas fp8-e4m3 -> bf16) the golden is READ from ``golden.yaml`` — the
    integer Tensor engine cannot reproduce the float datapath, and ``golden.yaml`` is the answer key the
    independent oracle already produced. For every other capsule (gemmini / ``exact_int``) the golden is
    RECOMPUTED on the Tensor engine exactly as before (byte-identical integer path)."""
    if capsule_dir is None:
        capsule_dir = capsule.get("__dir__")
    if is_independent_float_golden(capsule, capsule_dir):
        outs = (_load_golden_yaml(capsule_dir) or {}).get("outputs")
        if not outs:
            raise ValueError(
                f"independent float golden declared (golden_source="
                f"{golden_source(capsule, capsule_dir)!r}) but golden.yaml has no 'outputs' "
                f"({Path(capsule_dir) / 'golden.yaml' if capsule_dir else '<no dir>'})")
        return outs
    return _recompute_golden(capsule)


def _recompute_golden(capsule: dict) -> dict[str, list]:
    """Compute the capsule's expected outputs on the integer Tensor engine (the gemmini path)."""
    env = materialize_capsule_leaves(capsule)
    op = capsule["operation"]["op"]
    attrs = capsule["operation"].get("attributes", {})
    out_name = attrs.get("out", "Y0")

    def _pick(role: str) -> str:
        for s in capsule["inputs"]:
            if s.get("role") == role:
                return s["name"]
        raise KeyError(f"no input with role {role!r}")

    if op in ("matmul", "linear"):
        lhs = env[attrs.get("lhs", _pick("input"))]
        w = env[attrs.get("weight", _pick("weight"))]
        t = lhs.matmul(w)
        t = _apply_epilogue(t, attrs, env)
        return {out_name: t.to_list()}

    if op == "movement":
        src = env[attrs.get("src", _pick("input"))]
        return {out_name: src.to_list()}

    if op == "fused_matmul_bias":
        # The matmul whose epilogue is the bias, so it is the matmul branch with the bias stage forced
        # on. Forced rather than read: the op NAME is the declaration that the bias happens, and a
        # capsule that named this op but omitted the stage from `epilogue` would otherwise ship a golden
        # with no bias in it -- agreeing with a backend that dropped the bias, and both wrong.
        lhs = env[attrs.get("lhs", _pick("input"))]
        w = env[attrs.get("weight", _pick("weight"))]
        stages = list(attrs.get("epilogue") or [])
        if not any(s in ("bias_add", "bias") for s in stages):
            stages.insert(0, "bias_add")
        t = _apply_epilogue(lhs.matmul(w), {**attrs, "epilogue": stages}, env)
        return {out_name: t.to_list()}

    if op == "bias_add":
        # The same stage standing alone -- the unfused half of the pair above. It reads the SAME
        # `_apply_epilogue` implementation, so the fused capsule and the part it is compared against
        # cannot disagree about what adding a bias means, which is the whole basis of the comparison.
        src = env[attrs.get("src", _pick("input"))]
        stages = list(attrs.get("epilogue") or []) or ["bias_add"]
        return {out_name: _apply_epilogue(src, {**attrs, "epilogue": stages}, env).to_list()}

    if op == "conv2d":
        ifm = env[attrs["ifm"]]
        w = env[attrs["weight"]]              # packed [Kh*Kw*Ci, Co]
        ci = int(attrs["ci"]); kh = int(attrs["kh"]); kw = int(attrs["kw"])
        cols = im2col(ifm, ci, kh, kw, stride=tuple(attrs.get("stride", [1, 1])),
                      padding=tuple(attrs.get("padding", [0, 0, 0, 0])),
                      dilation=tuple(attrs.get("dilation", [1, 1])),
                      layout=attrs.get("layout", "nhwc"))
        t = cols.matmul(w)
        t = _apply_epilogue(t, attrs, env)
        return {out_name: t.to_list()}

    if op == "attention_qk":
        q = env[attrs["q"]]; k = env[attrs["k"]]
        t = q.matmul(_transpose2d(k))         # Q @ K^T
        t = _apply_epilogue(t, attrs, env)
        return {out_name: t.to_list()}

    if op == "attention_pv":
        p = env[attrs["p"]]; v = env[attrs["v"]]
        t = p.matmul(v)
        t = _apply_epilogue(t, attrs, env)
        return {out_name: t.to_list()}

    if op == "resident_reuse":
        # one resident weight, multiple matmuls (each with its own lhs/epilogue/out)
        w = env[attrs["weight"]]
        outs: dict[str, list] = {}
        for spec in attrs["matmuls"]:
            t = env[spec["lhs"]].matmul(w)
            sub = {"epilogue": spec.get("epilogue", []),
                   "output_dtype": spec.get("output_dtype", "i32"),
                   "acc_scale": spec.get("acc_scale", attrs.get("acc_scale", 1.0))}
            t = _apply_epilogue(t, sub, env)
            outs[spec["out"]] = t.to_list()
        return outs

    # --- float op family -------------------------------------------------------------------------
    # These are DEFINITIONS of the declared op, not per-capsule answers: one implementation grades
    # every capsule (present or future) that declares the op, on any target. Each composition reuses
    # the same primitives, so a fused capsule cannot disagree with its unfused counterpart.
    if op == "rmsnorm":
        t = _rmsnorm(env[attrs.get("src", _pick("input"))],
                     env[attrs.get("gamma", _pick("weight"))],
                     float(attrs.get("eps", 1e-5)))
        return {out_name: _apply_epilogue(t, attrs, env).to_list()}

    if op == "rmsnorm_qkv":
        # Fused norm -> QKV projection: rmsnorm(x, gamma) @ Wqkv.
        t = _rmsnorm(env[attrs["src"]], env[attrs["gamma"]], float(attrs.get("eps", 1e-5)))
        t = t.matmul(env[attrs["weight"]])
        return {out_name: _apply_epilogue(t, attrs, env).to_list()}

    if op == "rope_qkv":
        # Fused QKV projection -> rotary embedding, positions along the projected rows.
        t = env[attrs["src"]].matmul(env[attrs["weight"]])
        t = _rope_rotate_half(t, float(attrs.get("rope_theta", 10000.0)))
        return {out_name: _apply_epilogue(t, attrs, env).to_list()}

    if op in ("attention_mx", "attention_full", "attention"):
        # Operand names come from explicit q/k/v attrs when present, else the declared arg_order, else
        # the input order — never a positional guess when the capsule says otherwise.
        order = attrs.get("arg_order") or [s["name"] for s in capsule["inputs"]]
        qn = attrs.get("q", order[0])
        kn = attrs.get("k", order[1] if len(order) > 1 else None)
        vn = attrs.get("v", order[2] if len(order) > 2 else None)
        if not (kn and vn):
            raise ValueError(f"golden: {op} needs q/k/v operands (got {order!r})")
        cap_v = attrs.get("softcap")
        t = _attention(env[qn], env[kn], env[vn],
                       scale=attrs.get("scale"),
                       causal=bool(attrs.get("causal", False)),
                       softcap=float(cap_v) if cap_v is not None else None)
        return {out_name: _apply_epilogue(t, attrs, env).to_list()}

    if op in ("gemv_batched", "batched_matmul"):
        t = _batched_matmul(env[attrs.get("lhs", _pick("input"))],
                            env[attrs.get("weight", _pick("weight"))])
        return {out_name: _nested_list(_apply_epilogue(t, attrs, env))}

    raise ValueError(f"golden: unsupported operation {op!r}")


# --------------------------------------------------------------------------------------------
# comparison + numeric report
# --------------------------------------------------------------------------------------------
def _flat(nested) -> list:
    """Deep-flatten to a row-major value list (handles rank >= 3, e.g. a batched matmul output whose
    golden is (batch, m, n) while the kernel readback is a flat (batch*m, n))."""
    out: list = []

    def _rec(x):
        if isinstance(x, list):
            for e in x:
                _rec(e)
        else:
            out.append(x)

    _rec(nested)
    return out


#: How many diverging element indices to record per output. Enough to see whether the failure is
#: clustered (a row/column/tile — a scale, stride or tail bug) or scattered (rounding); bounded so a
#: saturated output cannot balloon the record. `mismatch_count` always carries the true total, and
#: `mismatch_indices_truncated` says when the list is partial.
_MISMATCH_INDEX_CAP = 64


def compare(expected: dict[str, list], observed: dict[str, list], policy: dict,
            *, golden_source: str = "merlin_tensor_int") -> dict:
    """Exact-int (or tolerance-float) comparison; returns a numeric_report dict. ``golden_source`` is
    stamped into the report so provenance is honest — ``merlin_tensor_int`` for a recomputed integer
    golden, or the INDEPENDENT source (e.g. ``specir_refmodel_fp8_bf16``) when it was read from
    ``golden.yaml`` rather than recomputed."""
    mode = policy.get("compare", "exact_int")
    rep: dict[str, Any] = {"policy": mode, "golden_source": golden_source,
                           "status": "pass", "mismatch_count": 0,
                           "max_abs_error": 0, "max_rel_error": 0.0,
                           "first_mismatch": None, "per_output": {}}
    total_mismatch = 0
    for name, exp in expected.items():
        ef = _flat(exp)
        if name not in observed:
            rep["status"] = "fail"
            rep["per_output"][name] = {"status": "fail", "reason": "missing from observed"}
            total_mismatch += len(ef)
            continue
        of = _flat(observed[name])
        if len(ef) != len(of):
            # A SHAPE error, not a value error -- and the two must not be conflated in one number. The
            # count below is |len delta| + 1, which is not a count of diverging elements at all: an
            # 8-element output emitted as 512 reads as "505 mismatches", indistinguishable from 505 wrong
            # values, and a later round that FIXES the shape and still has every value wrong reads as
            # "5 mismatches" -- looking like near-success when it is 5 of 8 wrong. Both were misread that
            # way on a real run. Name the class and carry both lengths so the number is interpretable.
            rep["status"] = "fail"
            rep["per_output"][name] = {"status": "fail",
                                       "reason": f"length {len(of)} != {len(ef)}",
                                       "failure_class": "output_shape_mismatch",
                                       "n_expected": len(ef), "n_observed": len(of)}
            rep.setdefault("outputs_wrong_shape", []).append(name)
            total_mismatch += abs(len(ef) - len(of)) + 1
            continue
        mism = 0
        maxabs = 0
        maxrel = 0.0
        first = None
        # WHERE the divergences are, not just how many. A count plus one index cannot distinguish a
        # CLUSTERED failure (a whole row/column/tile wrong -> a scale, stride or tail-handling bug) from a
        # SCATTERED one (a few elements a couple of ULP out -> rounding or accumulation order), and those
        # need completely different fixes. Measured: a capsule diverged on 8 of 256 elements by exactly
        # 3/32 and 5/32 -- even multiples of the bf16 ULP and sub-ULP in the operand format -- and the
        # record could not say whether the 8 shared a row, so the reading stayed a guess.
        #
        # Bounded so a saturated output cannot balloon the record: the first `_MISMATCH_INDEX_CAP`
        # indices are enough to see clustering, and `mismatch_count` already carries the total.
        bad_idx: list[int] = []
        for idx, (a, b) in enumerate(zip(ef, of)):
            if mode == "exact_int":
                bad = int(a) != int(b)
                d = abs(int(a) - int(b))
            else:
                rtol = float(policy.get("rtol", 0.0)); atol = float(policy.get("atol", 0.0))
                d = abs(float(a) - float(b))
                bad = d > (atol + rtol * abs(float(a)))
            if bad:
                mism += 1
                maxabs = max(maxabs, d)
                # max relative error over the DIVERGING elements. Undefined when the expected value is 0
                # (division by zero) — max_abs_error covers that case; we simply don't fold it into maxrel.
                den = abs(float(a))
                if den > 0.0:
                    maxrel = max(maxrel, d / den)
                if first is None:
                    first = {"output": name, "index": idx, "expected": a, "observed": b}
                if len(bad_idx) < _MISMATCH_INDEX_CAP:
                    bad_idx.append(idx)
        rep["per_output"][name] = {"status": "pass" if mism == 0 else "fail",
                                   "mismatch_count": mism, "max_abs_error": maxabs,
                                   "max_rel_error": maxrel,
                                   "n_elements": len(ef),
                                   "saturated": bool(mism and mism == len(ef))}
        if mism:
            rep["per_output"][name]["mismatch_indices"] = bad_idx
            rep["per_output"][name]["mismatch_indices_truncated"] = mism > len(bad_idx)
        if mism:
            # DISTINCT FAILURE CLASS: the kernel never wrote this output, so what was compared is the
            # buffer's initial fill, not a computed result. Calling that a numeric mismatch is actively
            # misleading -- a measured run spent six rounds chasing "functional_mismatch" on 12 capsules
            # whose observed output was uniformly 0.0 while the emitted artifact changed underneath, and
            # the mismatch COUNT could not move because it was a function of the golden's zero
            # distribution rather than of the kernel. Detected from the observed values alone: no target
            # fact, no dtype assumption, no fill constant baked in -- "every observed element is the same
            # value, the expected values are not, and that value is what an untouched buffer holds."
            uniq = {float(x) for x in of}
            if len(uniq) == 1 and len({float(x) for x in ef}) > 1:
                rep["per_output"][name]["failure_class"] = "output_never_written"
                rep["per_output"][name]["observed_constant"] = next(iter(uniq))
                rep.setdefault("outputs_never_written", []).append(name)
        if mism:
            rep["status"] = "fail"
            rep["max_abs_error"] = max(rep["max_abs_error"], maxabs)
            rep["max_rel_error"] = max(rep["max_rel_error"], maxrel)
            if rep["first_mismatch"] is None:
                rep["first_mismatch"] = first
        total_mismatch += mism
    rep["mismatch_count"] = total_mismatch
    # Surface the distinct class at the TOP level too, so a caller reading only the summary sees "the
    # kernel wrote nothing" rather than a large mismatch count that looks like ordinary numeric drift.
    if rep.get("outputs_never_written"):
        rep["failure_class"] = "output_never_written"
    return rep


def write_numeric_report(path: str | Path, report: dict) -> None:
    import yaml
    Path(path).write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
