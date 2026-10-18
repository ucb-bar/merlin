"""One fusion pass: match a contraction's source epilogue against the DECLARED output path, exactly.

A captured integer contraction is followed by a float chain -- convert, scale, bias, residual,
rectify, pool, quantize -- and a target's output path (:mod:`merlin.targetgen.output_path`) is an
ordered sequence of stages that can perform part of it. This pass is the only place that decides how
much: it reads the source chain by structure, tries the longest CUT the declared path allows, and
absorbs a cut only when a proof says the hardware computes exactly the source's integers.

Cuts, longest first:

``pool``
    The whole chain, its window max included, leaves the unit requantized and pooled.
``requant``
    The whole chain up to the quantize leaves the unit requantized.
``accumulator``
    The unit adds the bias (preloaded) and residual (accumulated) into the accumulator and hands the
    raw accumulator out; the host applies the rest of the chain, as written, without those adds.
``bare``
    The contraction alone.

THE PROOF. Every source stage and every declared stage is monotone non-decreasing in the accumulator
when its scales are positive and finite (convert, multiply, add, max with zero, round, clamp). Two
monotone step functions with outputs in ``[qmin, qmax]`` are equal on an interval exactly when every
output level ``q`` is first reached at the same point. So for each channel and each ``q`` the pass
finds the source's threshold and the device's, and an integer offset (the preloaded bias) is
admitted only when one offset makes every threshold coincide over the whole accumulator range the
stored weights and the operand formats allow. That is a complete check, not a sample. A residual
adds one dimension: the same offset must hold for every value the residual operand can take.

Everything target-dependent is a field of the declared path: stage order (an activation before or
after the scale), number formats, rounding, clamp, scale granularity and the columns one scale
covers. A field the path does not state, or a format or rounding this module has no exact model of,
refuses the cut with that reason. Nothing here names a target.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq

from . import compute_groups as CG

SCHEMA = "output_fusion_v1"
CUTS = ("pool", "requant", "accumulator", "bare")
#: Refusal codes, each a reason a cut was not taken. Closed: a new check adds a name.
UNDECLARED = "stage_undeclared"  # the path states no stage that performs a source stage
UNMODELED = "format_unmodeled"  # a format or rounding this module has no exact model of
GRANULARITY = "scale_granularity"  # the device scale cannot vary as the source scale does
INEXACT = "inexact"  # the proof found an output level no offset can align
UNREAD = "source_unread"  # the chain's numbers or channel axis could not be read
PADDING = "pool_padding_unknown"  # a padded pool whose device padding is not declared
NO_EXIT = "exit_undeclared"  # the path declares no exit at this cut
REFUSALS = (UNDECLARED, UNMODELED, GRANULARITY, INEXACT, UNREAD, PADDING, NO_EXIT)
#: The owner class of each refusal (``merlin.perf.placement_census`` vocabulary): a missing declaration
#: is OG1, a limit of the design or of the numbers is OG0, an unread source is OG7, and a format or
#: rounding this pass has no exact model of is OG6 -- ours to add, not the hardware's.
GAP_CLASS = {
    UNDECLARED: "OG1",
    PADDING: "OG1",
    NO_EXIT: "OG1",
    GRANULARITY: "OG0",
    INEXACT: "OG0",
    UNREAD: "OG7",
    UNMODELED: "OG6",
}
#: Rounding modes this module models exactly (numpy's round-half-to-even and IEEE arithmetic).
_ROUNDINGS = ("half_even",)


# ----------------------------------------------------------------------------------- source values


class Values:
    """The numbers behind SSA values: dense constants in the module, stored tensors beside a capture."""

    def __init__(self, capture: str | Path | None = None):
        self.manifest, self.weights = None, None
        if capture is not None:
            beside = Path(capture)
            beside = beside if beside.is_dir() else beside.parent
            manifest = next(iter(sorted(beside.glob("*.manifest.json"))), None)
            weights = next(iter(sorted(beside.glob("*.safetensors"))), None)
            if manifest is not None and weights is not None:
                self.manifest, self.weights = json.loads(manifest.read_text(encoding="utf-8")), weights

    def array(self, value):
        """A numpy array of ``value``'s elements (looking back through views and splats), or None."""
        import numpy as np

        from . import group_numerics as GN

        for _ in range(16):
            index = GN._arg_index(value)  # noqa: SLF001 -- the shared argument reader
            if index is not None:
                if self.manifest is None:
                    return None
                from . import group_prepack as GP

                try:
                    _name, raw, _dtype = GP._stored_raw(index, self.manifest, self.weights)  # noqa: SLF001
                except (GN.GroupNumericsError, KeyError, OSError, ValueError):
                    return None
                return np.asarray(raw)
            owner = getattr(value, "owner", None)
            if owner is None or not hasattr(owner, "operands"):
                return None
            name = mq.op_name(owner)
            if name == "arith.constant":
                attribute = owner.properties.get("value")
                if attribute is None:
                    attribute = owner.attributes.get("value")
                data = getattr(getattr(attribute, "value", None), "data", None)
                if isinstance(data, (int, float)) and not isinstance(data, bool):
                    return np.asarray(data)
                values = getattr(attribute, "get_values", None)
                if not callable(values):
                    return None
                listed = list(values())
                shape, _ = mq.type_shape_dtype(owner.results[0].type)
                count = 1
                for extent in shape:
                    count *= int(extent)
                return np.asarray(listed * count if len(listed) == 1 and count > 1 else listed)
            stage = CG.classify(owner)
            if name in ("tensor.splat", "linalg.fill") or (stage is not None and stage.kind in (CG.VIEW, CG.MOVEMENT)):
                if not owner.operands:
                    return None
                value = owner.operands[0]
                continue
            return None
        return None


# ----------------------------------------------------------------------------------- source chain


@dataclass
class Stage:
    kind: str  # convert | scale | bias | residual | relu | pad | pool | quantize
    op: Any
    #: Per-channel numbers (length = channels), a scalar, or None for a stage that carries none.
    numbers: Any = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chain:
    root: Any
    channels: int | None
    stages: list[Stage]
    #: Every op from the root's consumer through the last stage, views included, in program order.
    ops: list[Any]
    float_dtype: str | None = None
    unread: str | None = None


def _follow(axis_key, op):
    """``(extent, inner)`` of the channel axis after ``op``, recomputed for its result; None if lost."""
    extent, inner = axis_key
    in_shape = [int(v) for v in mq.type_shape_dtype(op.operands[0].type)[0]]
    out_shape = [int(v) for v in mq.type_shape_dtype(op.results[0].type)[0]]
    name = mq.op_name(op)
    if name == "linalg.transpose":
        here = _resolve(axis_key, in_shape)
        permutation = CG._int_array(op, "permutation") or []
        if here is None or here not in permutation:
            return None
        there = permutation.index(here)
        return (out_shape[there], CG._product(out_shape[there + 1 :]))
    return axis_key if CG._product(in_shape) == CG._product(out_shape) else None


def _resolve(axis_key, shape) -> int | None:
    extent, inner = axis_key
    hits = [i for i, v in enumerate(shape) if int(v) == extent and CG._product(shape[i + 1 :]) == inner]
    return hits[0] if len(hits) == 1 else None


def _channel_numbers(op, data, axis_key, channels: int, values: Values):
    """The per-channel (or scalar) numbers an elementwise stage applies, or a reason they are unread."""
    import numpy as np

    stage = CG.classify(op)
    operands = [operand for operand in list(op.operands)[: CG._n_inputs(op)] if operand is not data]
    if len(operands) != 1:
        return None, "the stage does not combine its input with exactly one other operand"
    numbers = values.array(operands[0])
    if numbers is None:
        return None, f"the operand of {mq.op_name(op)} is neither a constant nor a stored tensor"
    numbers = np.asarray(numbers, dtype=np.float64).reshape(-1)
    shape = [int(v) for v in mq.type_shape_dtype(op.results[0].type)[0]]
    varies = set(stage.varies_over or ()) if stage is not None and stage.varies_over is not None else None
    if numbers.size == 1:
        return float(numbers[0]), None
    axis = _resolve(axis_key, shape)
    if axis is None or varies is None:
        return None, "the channel axis of a per-channel stage could not be followed"
    if {d for d in varies if shape[d] != 1} != {axis} or numbers.size != channels:
        return None, "a stage varies over something other than the contraction's output channels"
    return numbers, None


def source_chain(root, values: Values) -> Chain:
    """The chain of single-use consumers after ``root``, read by structure up to its quantize."""
    from . import contraction_coverage as CC

    shape = [int(v) for v in mq.type_shape_dtype(root.results[0].type)[0]] if root.results else []
    axes = CG._root_axes(root)
    column = sorted(axes[1]) if axes else []
    if len(column) != 1 or not shape or column[0] >= len(shape):
        return Chain(root, None, [], [], unread="the contraction's output-channel axis is not one dimension")
    channels = shape[column[0]]
    axis_key = (channels, CG._product(shape[column[0] + 1 :]))
    chain = Chain(root, channels, [], [])
    value = root.results[0]
    while value is not None:
        users = CG._users(value)
        if len(users) != 1:
            break
        op = users[0]
        name = mq.op_name(op)
        with CC.select_max_is_max():  # a converted integer accumulator holds no NaN
            stage = CG.classify(op)
        if stage is None:
            break
        if stage.kind == CG.VIEW or name == "linalg.transpose":
            axis_key = _follow(axis_key, op) if axis_key is not None else None
            chain.ops.append(op)
            value = op.results[0]
            continue
        data = value
        if stage.kind == CG.CAST:
            _, dtype = mq.type_shape_dtype(op.results[0].type)
            chain.float_dtype = chain.float_dtype or dtype
            chain.stages.append(Stage("convert", op))
        elif stage.kind in (CG.SCALE, CG.BIAS_ADD):
            numbers, why = _channel_numbers(op, data, axis_key, channels, values)
            if why:
                chain.unread = why
                break
            chain.stages.append(Stage("scale" if stage.kind == CG.SCALE else "bias", op, numbers))
        elif stage.kind == CG.RESIDUAL_ADD:
            other = [operand for operand in list(op.operands)[:2] if operand is not data]
            chain.stages.append(Stage("residual", op, None, {"operand": other[0] if other else None}))
        elif stage.kind == CG.RELU:
            chain.stages.append(Stage("relu", op))
        elif stage.kind == CG.PAD:
            chain.stages.append(Stage("pad", op))
        elif stage.kind in (CG.POOL, CG.REDUCTION):
            if name.startswith("linalg.pooling"):
                maximum = "max" in name.removeprefix("linalg.").split("_")
            else:
                with CC.select_max_is_max():
                    maximum = name == "linalg.generic" and CC.classify_generic(op) == "max-reduction"
            if not maximum:
                break
            chain.stages.append(Stage("pool", op, None, {"reduction": "max"}))
        elif stage.kind == CG.QUANTIZE:
            from . import group_numerics as GN

            try:
                source = GN._scale_source(op)  # noqa: SLF001 -- the shared quantize reader
            except GN.GroupNumericsError as exc:
                chain.unread = str(exc)
                break
            _, out_dtype = mq.type_shape_dtype(op.results[0].type)
            chain.stages.append(
                Stage(
                    "quantize",
                    op,
                    source.value,
                    {
                        "zero_point": source.zero_point,
                        "min": GN._int_attr(op, "quant_min"),  # noqa: SLF001
                        "max": GN._int_attr(op, "quant_max"),  # noqa: SLF001
                        "dtype": out_dtype,
                    },
                )
            )
            chain.ops.append(op)
            break
        else:
            break
        chain.ops.append(op)
        value = op.results[0] if op.results else None
    return chain


# ---------------------------------------------------------------------------------------- formats


def _numpy_float(name: str | None):
    """The numpy type that IS ``name``'s IEEE arithmetic, or None when there is no exact model."""
    import numpy as np

    from merlin.common import quant_formats as qf

    if not name or not qf.has(str(name)):
        return None
    fmt = qf.get(str(name))
    for candidate in (np.float16, np.float32, np.float64):
        info = np.finfo(candidate)
        if fmt.kind == "float_ieee" and fmt.exp_bits == info.nexp and fmt.mant_bits == info.nmant:
            return candidate
    return None


def _int_range(name: str | None) -> tuple[int, int] | None:
    from merlin.common import quant_formats as qf

    if not name or not qf.has(str(name)) or qf.get(str(name)).exp_bits is not None:
        return None
    bits = qf.machine_bits(str(name))
    if not bits:
        return None
    signed = qf.get(str(name)).signed
    return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)


# ------------------------------------------------------------------------------------------ proof


def _threshold(fn, q, lo, hi, channels):
    """Per channel, the least integer in ``[lo, hi + 1]`` where ``fn`` reaches ``q`` (``hi + 1``: never)."""
    import numpy as np

    left = np.full(channels, lo, dtype=np.int64)
    right = np.full(channels, hi + 1, dtype=np.int64)
    for _ in range(int(hi - lo + 1).bit_length() + 1):
        active = left < right
        if not active.any():
            break
        mid = (left + right) // 2
        reached = fn(mid) >= q
        right = np.where(active & reached, mid, right)
        left = np.where(active & ~reached, mid + 1, left)
    return left


def offset_interval(source, target, channels, lo, hi, levels, target_range):
    """``(lower, upper, conflict)``: the integer offsets ``b`` with ``source(x) == target(x + b)`` on
    ``[lo, hi]``, per channel, for monotone step functions over the output ``levels``.

    ``target`` is evaluated on ``target_range`` (the accumulator's own range). ``conflict`` is the first
    level at which a channel's interval became empty, or ``None``.
    """
    import numpy as np

    ylo, yhi = target_range
    lower = np.full(channels, ylo - lo, dtype=np.int64)
    upper = np.full(channels, yhi - hi, dtype=np.int64)
    conflict = np.full(channels, np.iinfo(np.int64).min, dtype=np.int64)
    for q in levels:
        s = _threshold(source, q, lo, hi, channels)
        t = _threshold(target, q, ylo, yhi, channels)
        interior = (s > lo) & (s <= hi)
        lower = np.maximum(lower, np.where(s == lo, t - lo, np.where(interior, t - s, lower)))
        upper = np.minimum(upper, np.where(s == hi + 1, t - hi - 1, np.where(interior, t - s, upper)))
        conflict = np.where((conflict == np.iinfo(np.int64).min) & (lower > upper), q, conflict)
    return lower, upper, conflict


# ------------------------------------------------------------------------------------- the decision


@dataclass
class Decision:
    cut: str
    members: list[Any]
    stages: list[str]
    refusals: list[dict[str, Any]]
    proof: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "cut": self.cut,
            "stages": list(self.stages),
            "refusals": [dict(r) for r in self.refusals],
            "proof": dict(self.proof),
        }

    @property
    def first_refusal(self) -> dict[str, Any] | None:
        return self.refusals[0] if self.refusals else None


class _Refused(Exception):
    def __init__(self, code: str, reason: str):
        super().__init__(reason)
        self.code, self.reason = code, reason


def _evaluate(stages: Sequence[Stage], fdt, x, *, skip=(), residual=None):
    """The source chain, as the host computes it, on accumulator values ``x`` (one per channel)."""
    import numpy as np

    v = x.astype(fdt)
    out = None
    for stage in stages:
        if stage.kind in skip:
            continue
        if stage.kind == "scale":
            v = np.multiply(v, np.asarray(stage.numbers, dtype=fdt), dtype=fdt)
        elif stage.kind == "bias":
            v = np.add(v, np.asarray(stage.numbers, dtype=fdt), dtype=fdt)
        elif stage.kind == "residual":
            v = np.add(v, residual, dtype=fdt)
        elif stage.kind == "relu":
            v = np.maximum(v, fdt(0))
        elif stage.kind == "quantize":
            inv = fdt(1) / fdt(stage.numbers)
            t = np.rint(np.multiply(v, inv, dtype=fdt)) + stage.detail["zero_point"]
            out = np.clip(t, stage.detail["min"], stage.detail["max"]).astype(np.int64)
    return out


def _device(path_readout: Sequence[Mapping[str, Any]], scale, y, *, relu: bool):
    """The declared readout, in its declared order and formats, on accumulator values ``y``."""
    import numpy as np

    v, dt = y, None
    for stage in path_readout:
        kind = stage["kind"]
        if kind == "activation":
            if relu:
                v = np.maximum(v, 0) if dt is None else np.maximum(v, dt(0))
        elif kind == "convert":
            dt = _numpy_float(stage.get("to"))
            with np.errstate(over="ignore"):  # IEEE overflow to infinity is the modelled result
                v = v.astype(dt)
        elif kind == "scale":
            v = np.multiply(v, scale.astype(dt), dtype=dt)
        elif kind == "requant":
            low, high = stage["clamp"]
            return np.clip(np.rint(v), low, high).astype(np.int64)
    raise _Refused(UNDECLARED, "the declared readout ends without a requant")


def _readout(path: Mapping[str, Any], cut: str) -> list[Mapping[str, Any]]:
    stages = [dict(s) for s in path.get("readout") or ()]
    return [s for s in stages if s["kind"] != "pool"] if cut == "requant" else stages


def _require_models(readout, chain: Chain) -> None:
    for stage in readout:
        if stage["kind"] in ("convert", "scale", "requant") and stage.get("rounding") not in _ROUNDINGS:
            raise _Refused(
                UNMODELED, f"the {stage['kind']} rounds {stage.get('rounding')!r}, which has no exact model here"
            )
        if stage["kind"] in ("convert", "scale") and _numpy_float(stage.get("to") or stage.get("dtype")) is None:
            raise _Refused(
                UNMODELED,
                f"the {stage['kind']} format {stage.get('to') or stage.get('dtype')!r} has no exact model here",
            )
    if _numpy_float(chain.float_dtype) is None:
        raise _Refused(UNMODELED, f"the source computes in {chain.float_dtype!r}, which has no exact model here")


def _device_scale(chain: Chain, readout, channels: int):
    """The one multiplier per channel the device applies: the source scales and output reciprocal."""
    import numpy as np

    scale_stage = next((s for s in readout if s["kind"] == "scale"), None)
    if scale_stage is None:
        raise _Refused(UNDECLARED, "the output path applies no scale")
    dt = _numpy_float(scale_stage["dtype"])
    fdt = _numpy_float(chain.float_dtype)
    quantize = chain.stages[-1]
    total = np.ones(channels, dtype=dt)
    for stage in chain.stages:
        if stage.kind == "scale":
            total = np.multiply(
                total, np.broadcast_to(np.asarray(stage.numbers, dtype=fdt), (channels,)).astype(dt), dtype=dt
            )
    total = np.multiply(total, (fdt(1) / fdt(quantize.numbers)).astype(dt), dtype=dt)
    return total, scale_stage


def _granularity(total, scale_stage, channels: int) -> dict[str, Any]:
    """Direct gain blocks, or a bound obligation for independent logical columns."""
    import numpy as np

    granularity = scale_stage.get("granularity")
    if granularity == "per_channel":
        return {"granularity": "per_channel", "commands": 1}
    width = scale_stage.get("columns_per_scale") if granularity == "per_command" else None
    if granularity == "per_command":
        if type(width) is not int or width < 1:
            raise _Refused(UNDECLARED, "the physical scale-vector width is not a positive integer")
    else:
        width = channels
    blocks = [total[i : i + width] for i in range(0, channels, width)]
    if any(np.unique(block).size != 1 for block in blocks):
        if granularity == "per_command":
            from merlin.targetgen.output_scale_partition import obligation

            try:
                partition = obligation([float(value) for value in total], width)
            except ValueError as error:
                raise _Refused(UNMODELED, str(error)) from error
            # Numeric equivalence is still proved over the complete source bound.
            # No command count or physical placement follows from this record.
            return {"granularity": "per_command", "columns_per_scale": width,
                    "output_column_partition": partition}
        raise _Refused(
            GRANULARITY,
            f"the scale differs within a group of {width} column(s) one readout scale covers",
        )
    distinct = [float(block[0]) for block in blocks]
    reconfigurations = sum(1 for a, b in zip(distinct, distinct[1:]) if a != b)
    return {"granularity": "per_command", "columns_per_scale": width, "commands": len(blocks),
            "scale_reconfigurations": reconfigurations}  # fmt: skip


def _offset_range(path: Mapping[str, Any], has_bias: bool) -> tuple[int, int]:
    preload = next((s for s in path.get("accumulator_stages") or () if s.get("kind") == "preload"), None)
    if preload is None or not isinstance(preload.get("range"), list):
        return (0, 0)  # without a preload the accumulator starts at zero: the offset is zero
    return int(preload["range"][0]), int(preload["range"][1])


def _accumulator_range(path: Mapping[str, Any]) -> tuple[int, int] | None:
    return _int_range((path.get("accumulator") or {}).get("dtype"))


def _prove(chain: Chain, path, target, bound, *, residual_values=None, residual_on_device=True) -> dict[str, Any]:
    """Solve the offsets for ``target(y, residual)`` against the source; refuse unless every channel aligns.

    With a residual, the same offset must hold for every value the residual operand can take: on the
    device it is the integer ``r`` added into the accumulator, on the host the float it dequantizes to.
    """
    import numpy as np

    fdt = _numpy_float(chain.float_dtype)
    quantize = chain.stages[-1]
    # Equal reachable thresholds do not forbid a wider device clamp from
    # producing values ABOVE the source maximum. Its first unreachable level
    # must also remain unreachable throughout the proved accumulator domain.
    levels = list(range(int(quantize.detail["min"]), int(quantize.detail["max"]) + 2))
    acc = _accumulator_range(path)
    if acc is None:
        raise _Refused(UNMODELED, "the accumulator format is not an integer the proof can range over")
    lo, hi = bound
    channels = int(chain.channels)
    has_bias = any(stage.kind == "bias" for stage in chain.stages)
    allowed = _offset_range(path, has_bias)
    lower = np.full(channels, allowed[0], dtype=np.int64)
    upper = np.full(channels, allowed[1], dtype=np.int64)
    conflict = None
    for r in residual_values if residual_values is not None else (None,):
        residual = None if r is None else residual_values[r]

        def source(x, residual=residual):
            return _evaluate(chain.stages, fdt, x, residual=residual)

        shift = int(r) if r is not None and residual_on_device else 0
        host_residual = None if residual_on_device else residual

        def device(y, shift=shift, host_residual=host_residual):
            return target(y + shift, host_residual)

        found_lo, found_hi, found_conflict = offset_interval(source, device, channels, lo, hi, levels, acc)
        if shift:
            found_hi = np.minimum(found_hi, acc[1] - hi - max(shift, 0))
            found_lo = np.maximum(found_lo, acc[0] - lo - min(shift, 0))
        lower, upper = np.maximum(lower, found_lo), np.minimum(upper, found_hi)
        if conflict is None and (found_lo > found_hi).any():
            conflict = int(found_conflict[found_lo > found_hi][0])
    accepted = lower <= upper
    if not accepted.all():
        raise _Refused(
            INEXACT,
            f"{int((~accepted).sum())} of {channels} channel(s) have no integer accumulator offset that "
            f"reproduces every output level over [{lo}, {hi}]"
            + (f" (first conflict at level {conflict})" if conflict is not None else ""),
        )
    offsets = np.clip(np.zeros(channels, dtype=np.int64), lower, upper)
    return {"offsets": [int(v) for v in offsets], "accumulator_bound": [int(lo), int(hi)], "channels": channels}


def _positive(chain: Chain) -> None:
    """Every scale positive and finite: the condition under which each stage is monotone."""
    import numpy as np

    for stage in chain.stages:
        if stage.kind in ("scale", "quantize"):
            numbers = np.asarray(stage.numbers, dtype=np.float64)
            if not (np.isfinite(numbers).all() and (numbers > 0).all()):
                raise _Refused(INEXACT, f"a {stage.kind} is not positive and finite, so the chain is not monotone")


def _bound(chain: Chain, values: Values, extents: Mapping[int, Any]) -> tuple[int, int]:
    """Every accumulator value the contraction can produce: ``K * max|w| * max|x|`` from formats and weights."""
    root = chain.root
    extent = extents.get(id(root))
    if not extent or extent[1] is None:
        raise _Refused(UNREAD, "the contraction's reduction extent is not static")
    magnitudes = []
    for operand in list(root.operands)[:2]:
        _, dtype = mq.type_shape_dtype(operand.type)
        span = _int_range(CG._dtype_token(dtype))
        if span is None:
            raise _Refused(UNREAD, f"a contraction operand of {dtype!r} has no integer range")
        stored = values.array(operand)
        # abs(minimum signed code) overflows in that tensor's integer dtype.
        # Read extrema as Python integers before taking their magnitudes.
        magnitudes.append(
            max(abs(int(stored.min())), abs(int(stored.max())))
            if stored is not None and stored.size else max(-span[0], span[1])
        )
    bound = int(extent[1]) * magnitudes[0] * magnitudes[1]
    return -bound, bound


def _residual_values(chain: Chain, stage: Stage, values: Values, fdt):
    """``{r: the residual's float value}`` for every integer ``r`` of a dequantized residual operand."""
    import numpy as np

    from . import group_numerics as GN

    operand = stage.detail.get("operand")
    owner = getattr(operand, "owner", None)
    for _ in range(8):  # through views to the dequantize
        if owner is None or not hasattr(owner, "operands"):
            break
        kind = CG.classify(owner)
        if kind is None or kind.kind not in (CG.VIEW, CG.MOVEMENT):
            break
        owner = getattr(owner.operands[0], "owner", None)
    kind = CG.classify(owner) if owner is not None and hasattr(owner, "operands") else None
    if kind is None or kind.kind != CG.DEQUANTIZE or "per_tensor" not in kind.detail:
        raise _Refused(UNREAD, "the residual operand is not a per-tensor dequantized integer tensor")
    source = GN._scale_source(owner)  # noqa: SLF001
    if source.value is None or source.zero_point is None:
        raise _Refused(UNREAD, "the residual's scale or zero point is not a compile-time number")
    _, dtype = mq.type_shape_dtype(owner.operands[0].type)
    span = _int_range(CG._dtype_token(dtype))
    if span is None:
        raise _Refused(UNREAD, "the residual's integer format has no range")
    table = {}
    for r in range(span[0], span[1] + 1):
        diff = np.subtract(fdt(r), fdt(source.zero_point), dtype=fdt)
        table[r - int(source.zero_point)] = np.multiply(diff, fdt(source.value), dtype=fdt)
    return table, float(source.value)


def _accumulator_unit(chain: Chain, upto: Stage):
    """The source scales applied before ``upto``, per channel, as exact rationals."""
    from fractions import Fraction

    unit = [Fraction(1)] * int(chain.channels)
    for stage in chain.stages:
        if stage is upto:
            break
        if stage.kind == "scale":
            numbers = stage.numbers if not isinstance(stage.numbers, float) else [stage.numbers] * len(unit)
            unit = [u * Fraction(float(n)) for u, n in zip(unit, numbers)]
    return unit


def _residual(chain: Chain, path, values: Values, absorbing: bool):
    """The residual table, when the residual is IN the chain and the cut absorbs it; else None."""
    from fractions import Fraction

    stage = next((s for s in chain.stages if s.kind == "residual"), None)
    if stage is None:
        return None
    fdt = _numpy_float(chain.float_dtype)
    table, scale = _residual_values(chain, stage, values, fdt)
    if absorbing:
        accumulate = next((s for s in path.get("accumulator_stages") or () if s.get("kind") == "accumulate"), None)
        if accumulate is None:
            raise _Refused(UNDECLARED, "the output path accumulates no operand, so the residual stays on the host")
        unit = _accumulator_unit(chain, stage)
        if any(Fraction(scale) != u for u in unit):
            raise _Refused(
                INEXACT,
                "the residual is not at the accumulator's scale (its scale differs from the product of the "
                "scales before it), so it cannot be added in the accumulator unchanged",
            )
    return table


def decide(chain: Chain, path: Mapping[str, Any], values: Values, extents: Mapping[int, Any]) -> Decision:
    """The longest cut of ``chain`` the declared ``path`` performs exactly, with every refusal."""
    refusals: list[dict[str, Any]] = []
    kinds = [stage.kind for stage in chain.stages]
    closed = bool(kinds) and kinds[-1] == "quantize" and "convert" in kinds and chain.unread is None
    for cut in CUTS:
        try:
            members, proof = _attempt(cut, chain, path, values, extents, kinds, closed)
        except _Refused as refused:
            refusals.append({"cut": cut, "code": refused.code, "reason": refused.reason})
            continue
        absorbed = {id(op) for op in members}
        stages = [stage.kind for stage in chain.stages if id(stage.op) in absorbed]
        if cut != "bare":
            # Consumers must preserve the formats the theorem actually modeled.
            # A vector of Python floats alone cannot distinguish f16 from f32.
            proof["accumulator"] = dict(path["accumulator"])
            proof["preload"] = next(
                (dict(s) for s in path.get("accumulator_stages") or () if s["kind"] == "preload"), None
            )
            proof["readout_stages"] = _readout(path, cut) if cut != "accumulator" else []
        return Decision(cut, members, stages, refusals, proof)
    return Decision("bare", [], [], refusals)


def _attempt(cut, chain, path, values, extents, kinds, closed):
    import numpy as np

    exits = path.get("exits") or {}
    if cut == "bare":
        return [], {}
    if chain.unread:
        raise _Refused(UNREAD, chain.unread)
    if not closed:
        raise _Refused(UNREAD, "the chain after the contraction does not end in a quantize")
    readout = _readout(path, cut)
    if cut == "accumulator":
        if not exits.get("accumulator"):
            raise _Refused(NO_EXIT, "the output path declares no full-width accumulator exit")
        absorbable = [stage for stage in chain.stages if stage.kind in ("bias", "residual")]
        if not absorbable:
            raise _Refused(UNDECLARED, "nothing in the chain is an accumulator-domain stage")
        if any(stage.kind == "bias" for stage in absorbable) and not any(
            s.get("kind") == "preload" for s in path.get("accumulator_stages") or ()
        ):
            raise _Refused(UNDECLARED, "the output path preloads no bias")
        fdt = _numpy_float(chain.float_dtype)
        if fdt is None:
            raise _Refused(UNMODELED, f"the source computes in {chain.float_dtype!r}, which has no exact model here")
        on_device, host_note = True, None
        table = None
        if "residual" in kinds:
            try:
                table = _residual(chain, path, values, absorbing=True)
            except _Refused as refused:
                # The bias can still be preloaded; the residual is then added on the host, as written.
                table, on_device = _residual(chain, path, values, absorbing=False), False
                absorbable = [stage for stage in absorbable if stage.kind != "residual"]
                host_note = {"code": refused.code, "reason": refused.reason}
                if not absorbable:
                    raise
        bound = _bound(chain, values, extents)
        skipped = tuple(sorted({stage.kind for stage in absorbable}))

        def target(y, residual):
            return _evaluate(chain.stages, fdt, y, skip=skipped, residual=residual)

        _positive(chain)
        proof = _prove(chain, path, target, bound, residual_values=table, residual_on_device=on_device)
        if host_note:
            proof["residual_on_host"] = host_note
        return [stage.op for stage in absorbable], {**proof, "readout": "full_width", "host_skips": list(skipped)}
    if not exits.get("requant"):
        raise _Refused(NO_EXIT, "the output path declares no requantizing exit")
    pool_stage = next((s for s in path.get("readout") or () if s.get("kind") == "pool"), None)
    has_pool = "pool" in kinds
    if cut == "pool":
        if not has_pool:
            raise _Refused(UNDECLARED, "the chain has no pool to absorb")
        if pool_stage is None or "max" not in (pool_stage.get("reductions") or ()):
            raise _Refused(UNDECLARED, "the output path pools no window maximum")
        if "pad" in kinds and pool_stage.get("padding") in (None, "unknown"):
            raise _Refused(PADDING, "the source pads its pool and the output path's padding value is not derived")
    elif has_pool:
        raise _Refused(UNDECLARED, "the chain pools before it quantizes and this cut does not pool")
    _require_models(readout, chain)
    relu = "relu" in kinds
    if relu and not any(s["kind"] == "activation" and s.get("function") == "relu" for s in readout):
        raise _Refused(UNDECLARED, "the output path applies no rectifier")
    quantize = chain.stages[-1]
    requant = next(s for s in readout if s["kind"] == "requant")
    if quantize.detail.get("zero_point") not in (0,):
        raise _Refused(UNDECLARED, "the source quantizes with a zero point the output path does not apply")
    if CG._dtype_token(quantize.detail.get("dtype")) != CG._dtype_token(requant.get("to")):
        raise _Refused(
            UNDECLARED,
            f"the output path narrows to {requant.get('to')!r}, the source to {quantize.detail.get('dtype')!r}",
        )
    channels = int(chain.channels)
    total, scale_stage = _device_scale(chain, readout, channels)
    split = _granularity(total, scale_stage, channels)
    table = _residual(chain, path, values, absorbing=True) if "residual" in kinds else None
    bound = _bound(chain, values, extents)
    scale = np.asarray(total)

    def target(y, _residual):
        return _device(readout, scale, y, relu=relu)

    _positive(chain)
    proof = _prove(chain, path, target, bound, residual_values=table)
    members = list(chain.ops)
    return members, {**proof, **split, "readout": "requant", "device_scale": [float(v) for v in scale]}


# ----------------------------------------------------------------------------------- the pass driver


def declared_path(target: str) -> dict[str, Any] | None:
    """The selected contract's derived output path, or None when it declares none."""
    from merlin.targetgen import output_path as OP
    from merlin.targetgen.target_registry import TargetContractMissing, load_contract

    try:
        contract = load_contract(target) or {}
    except TargetContractMissing:
        return None  # no selected contract declares anything: grouping keeps its own rules
    record = contract.get("output_epilogue") or {}
    path = record.get("output_path")
    return OP.validate(path) if path else None


def decisions(groups, module, path: Mapping[str, Any], values: Values) -> dict[int, Decision]:
    """``id(root) -> Decision`` for every accelerator contraction group, from ONE matching rule."""
    from merlin.targetgen import model_coverage as MC

    extents = MC._contraction_extents(module)  # noqa: SLF001 -- the shared extent reader
    out: dict[int, Decision] = {}
    for group in groups:
        if group.root is None or group.placement == CG.HOST or CG.classify(group.root).kind != CG.CONTRACTION:
            continue
        out[id(group.root)] = decide(source_chain(group.root, values), path, values, extents)
    return out
