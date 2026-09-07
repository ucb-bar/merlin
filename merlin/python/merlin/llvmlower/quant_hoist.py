"""Hoist WEIGHT-INVARIANT dynamic quantization out of `@forward` and into the build.

THE WORK THIS REMOVES. `quant_passes.apply_quant` lowers a W8A8 capture by quantizing BOTH operands
of every contraction at run time: an amax reduce (``math.absf`` + ``arith.maximumf``) finds the
per-row maximum, a divide turns it into a scale, and a ``roundeven -> clamp -> fptosi`` map applies
it. For the ACTIVATION operand that is correct and unavoidable -- the activation is different on
every inference. For the WEIGHT operand it is redundant: a weight does not change between
inferences, so its amax, its scale and its int8 form are functions of the weight ALONE and are
recomputed identically every time.

MEASURED on the prepared modules (`build_tools/scripts/model_math_census.py`, elements processed per
inference)::

    resnet50_v1_5_int8_w8a8   weight-invariant  25,289,472 absf + 25,289,472 roundeven  = 55.9%
    lstmnetvit_int8_w8a8      weight-invariant     941,281 absf +    941,281 roundeven  = 21.3%

of the capture's whole ``math.*`` traffic, in 53 and 21 chains respectively.

WHY THE CHAIN IS NOT IN THE BUNDLE, AND WHAT THAT IMPLIES. The same census over the module prepared
WITHOUT the int8 datapath finds no weight-invariant ``absf``/``roundeven`` at all -- the 53 chains
are created by OUR OWN lowering, not by the capture. So this is not a missing constant-fold of
something the model asked for; it is our int8 pass choosing run time for work whose inputs are known
at build time. It also means a bundle rewrite (`weight_prepack.prepacked_bundle`'s shape) CANNOT
implement it: the chain does not exist in the raw ``model.mlir`` that a bundle rewrite edits.

HOW IT IS DONE. Each hoisted result becomes a TRAILING `@forward` argument whose bytes are computed
here and appended to the weight blob -- deliberately the same channel `qinner` uses, so there is one
ABI shape rather than two. The alternative (materialising ``arith.constant dense<...>``) was rejected
for the int8 weights: the prepared module is round-tripped through TEXT several times, and resnet50's
hoisted weights are ~25 MB of elements, which is the 770 MB-header / 7 GB-RSS pathology `c_runtime`
already documents for embedded C literals. Scales are small, but splitting the two halves across two
mechanisms would mean two ABIs to keep in step, which is the skew this module exists to avoid.

FAIL CLOSED, EVERYWHERE. A chain whose root cannot be resolved to a weight, whose view chain
contains an op this module cannot reproduce exactly, or whose recomputed value disagrees in shape or
dtype with the IR, is REFUSED (the whole pass declines and reports why) rather than partially
applied. Substituting a guess would reproduce the silent-garbage failure mode `qinner` was written
to remove -- and unlike a numerical bug it would not show up as a wrong answer until the weights
changed.

DEFAULT OFF. With the feature absent the module is not walked and the prepared IR is byte-identical.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The impr-feature name. Registered EAGERLY by :func:`ensure_registered` and additionally reachable
#: through `impr_features._try_lazy_register`, because a package that names it in `compiler_features`
#: arrives via `k1.build_k1_binary -> normalize` (which imports no proposer) and again inside the
#: lowering SUBPROCESS, which re-imports `impr_features` fresh and sees no run-time registration the
#: parent made. Both raise "unknown impr feature" without that hook.
FEATURE = "hoist_weight_invariant_quantize"

#: Written beside `model.prepared.mlir` so `c_runtime.generate` -- which re-parses the RAW bundle and
#: therefore cannot rederive a chain that only exists after `apply_quant` -- appends matching rows.
PLAN_FILE = "quant_hoist_args.json"
#: The recomputed bytes, keyed by the plan's ``key``.
VALUES_FILE = "quant_hoist_values.npz"

#: Ops that only RE-VIEW a tensor. A weight reaches its quantize chain through these, and each has
#: an exact numpy equivalent (a reshape to the result type's own shape), so the value computed here
#: is the value the IR would have computed. An op outside this set is a refusal, never a guess.
VIEW_OPS: frozenset[str] = frozenset({
    "tensor.collapse_shape", "tensor.expand_shape", "tensor.cast", "tensor.reshape",
})


class QuantHoistRefused(RuntimeError):
    """The hoist cannot be applied soundly. Raised, never downgraded to a partial rewrite."""


@dataclass(frozen=True)
class HoistedArg:
    """One recomputed tensor lifted to a trailing `@forward` argument."""

    key: str                      #: stable name; also the npz key holding the bytes
    shape: tuple[int, ...]
    dtype: str                    #: MLIR element-type spelling (``i8``, ``f32``, ...)

    def as_json(self) -> dict[str, Any]:
        return {"key": self.key, "shape": list(self.shape), "dtype": self.dtype}

    @staticmethod
    def from_json(o: dict[str, Any]) -> "HoistedArg":
        return HoistedArg(str(o["key"]), tuple(int(d) for d in o["shape"]), str(o["dtype"]))


def write_plan(dirpath: str | Path, args: list[HoistedArg]) -> Path:
    p = Path(dirpath) / PLAN_FILE
    p.write_text(json.dumps([a.as_json() for a in args], indent=2))
    return p


def read_plan(dirpath: str | Path) -> list[HoistedArg]:
    """The plan, or ``[]`` when the feature was off. An unreadable plan is an ERROR: a build that
    wrote one and a consumer that silently read none is exactly the ABI skew this avoids."""
    p = Path(dirpath) / PLAN_FILE
    if not p.is_file():
        return []
    return [HoistedArg.from_json(o) for o in json.loads(p.read_text())]


def write_values(dirpath: str | Path, values: dict) -> Path:
    """The recomputed bytes, as an npz keyed by the plan's ``key``. Written beside the plan so the
    two always travel together; `c_runtime` refuses a plan whose values are missing."""
    import numpy as np
    p = Path(dirpath) / VALUES_FILE
    np.savez(p, **{k: np.ascontiguousarray(v) for k, v in values.items()})
    return p.with_suffix(".npz") if p.suffix != ".npz" else p


def read_values(dirpath: str | Path):
    import numpy as np
    p = Path(dirpath) / VALUES_FILE
    if not p.is_file():
        return {}
    return np.load(p)


def weight_loader(bundle_dir: str | Path, manifest: dict):
    """``load_arg(i) -> numpy array | None`` reading the bundle's safetensors payload directly.

    Returns None for an argument the manifest does not describe as stored data (an input, or a
    lifted quant-inner tensor that lives in ``extra.npz``); the caller REFUSES on None rather than
    substituting anything, so a chain rooted in something this cannot read is not hoisted.
    """
    import numpy as np
    from .weights_pack import load_safetensors_header

    bundle_dir = Path(bundle_dir)
    st = bundle_dir / "weights.safetensors"
    hdr, payload_off = load_safetensors_header(st)
    raw = st.read_bytes()
    # safetensors' own dtype spelling -> numpy. An unknown spelling is a refusal, not a default.
    dt = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": np.float32,
          "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8}

    def load_arg(i: int):
        meta = manifest.get(str(i))
        if not meta or meta.get("kind") not in ("param", "buffer"):
            return None
        name = meta.get("weight")
        if name not in hdr:
            return None
        h = hdr[name]
        spelling = h["dtype"]
        if spelling not in dt:
            raise QuantHoistRefused(f"safetensors dtype {spelling!r} has no numpy mapping")
        if spelling == "BF16":
            raise QuantHoistRefused("bf16 payload needs an explicit widening; refusing to guess")
        begin, end = h["data_offsets"]
        npdt = dt[spelling]
        count = (end - begin) // np.dtype(npdt).itemsize
        return np.frombuffer(raw, dtype=npdt, count=count,
                             offset=payload_off + begin).reshape(h["shape"])

    return load_arg


def _elem_str(t) -> str:
    from xdsl.dialects.builtin import TensorType
    return str(t.element_type if isinstance(t, TensorType) else t)


def _shape_of(t) -> tuple[int, ...]:
    return tuple(int(d) for d in t.get_shape())


def _body_op_names(op) -> list[str]:
    if not op.regions or not op.regions[0].blocks:
        return []
    return [i.name for i in op.regions[0].blocks[0].ops if i.name != "linalg.yield"]


def _iterator_kinds(op) -> list[str]:
    its = op.properties.get("iterator_types")
    return [str(e) for e in its.data] if its is not None else []


def _forward_func(module):
    for op in module.walk():
        if op.name in ("func.func", "builtin.func") and \
                "forward" in str(op.properties.get("sym_name", "")):
            return op
    raise QuantHoistRefused("no @forward function in the module")


def activation_arg_indices(manifest: dict) -> frozenset[int]:
    """The `@forward` argument positions that carry a real INPUT, read off the bundle manifest's own
    ``kind`` field.

    THIS IS NOT ``range(len(input_order.json))``, AND THE DIFFERENCE INVERTS THE ANSWER. That
    spelling -- "the first n arguments are the inputs, the rest are weights" -- is what
    `build_tools/scripts/model_math_census.py` assumed, and it is false for both captures measured
    here: resnet50's single activation is argument **320** (arguments 0-160 are params and 161-319
    are buffers), and lstmnetvit's five are **98-102**. Under the positional assumption the census
    labelled `model.conv1.weight` "the activation" and the image "weight-invariant", so its
    ACTIVATION and WEIGHT_INVARIANT populations were the wrong way round -- the aggregate share
    barely moved (the two sides of a contraction are similar in size), but every op it NAMED as
    hoistable was an activation-side chain, which is exactly the work that cannot be hoisted.
    ``input_order.json`` maps a name to its position in the INPUTS NPZ, not to an argument index.

    The manifest states the kind per argument, so this is read rather than inferred.
    """
    idx = frozenset(int(k) for k, v in manifest.items() if v.get("kind") == "input")
    if not idx:
        raise QuantHoistRefused(
            "no @forward argument is declared kind='input' in the manifest; refusing to guess "
            "which arguments are activations (a wrong guess inverts the whole analysis)")
    return idx


def activation_reach(module, activation_args):
    """Predicate: does this value's def-chain reach one of `@forward`'s ACTIVATION arguments?

    ``activation_args`` is the SET of argument positions carrying a real input -- see
    :func:`activation_arg_indices` for why it must be a set read from the manifest and never a
    ``range`` over a count. A value reaching none of them is identical on every inference, which is
    the whole predicate this pass acts on.
    """
    block = _forward_func(module).regions[0].blocks[0]
    args = list(block.args)
    activation_args = frozenset(int(i) for i in activation_args)
    bad = [i for i in activation_args if i < 0 or i >= len(args)]
    if not activation_args or bad:
        raise QuantHoistRefused(
            f"activation argument positions {sorted(activation_args)} do not fit @forward's arity "
            f"{len(args)} (out of range: {bad}); refusing to guess")
    activation = {id(args[i]) for i in activation_args}
    memo: dict[int, bool] = {}

    def reaches(value) -> bool:
        k = id(value)
        if k in memo:
            return memo[k]
        memo[k] = False                      # cycle guard; an unresolved value is not an activation
        if k in activation:
            memo[k] = True
            return True
        owner = getattr(value, "owner", None)
        if owner is None or not hasattr(owner, "operands"):
            return False
        r = any(reaches(o) for o in owner.operands)
        memo[k] = r
        return r

    return reaches


def _arg_index(module, value) -> "int | None":
    """The `@forward` argument position of ``value``, or None when it is not a block argument."""
    block = _forward_func(module).regions[0].blocks[0]
    for i, a in enumerate(block.args):
        if a is value:
            return i
    return None


def _resolve_source(module, value, load_arg):
    """The numpy VALUE of ``value``, following only :data:`VIEW_OPS` back to a `@forward` argument.

    Every step is exact: a collapse/expand/cast/reshape is a pure re-view, so the numpy equivalent is
    a reshape to the step's own result type. An op outside the set, or a chain that does not end at
    an argument, is REFUSED -- the point of this pass is that the hoisted bytes are the bytes the IR
    would have produced, and an approximation of that is worse than not hoisting.
    """
    import numpy as np

    seen = 0
    cur = value
    while True:
        seen += 1
        if seen > 64:
            raise QuantHoistRefused("view chain longer than 64 hops; refusing to keep walking")
        idx = _arg_index(module, cur)
        if idx is not None:
            arr = load_arg(idx)
            if arr is None:
                raise QuantHoistRefused(f"@forward argument {idx} has no weight bytes")
            return np.asarray(arr).reshape(_shape_of(value.type))
        owner = getattr(cur, "owner", None)
        name = getattr(owner, "name", None)
        if name not in VIEW_OPS:
            raise QuantHoistRefused(
                f"chain reaches {name!r}, which is not a pure re-view; refusing to reproduce it")
        cur = owner.operands[0]


# ---------------------------------------------------------------------------------------------
# The three chain shapes, recognised STRUCTURALLY (body op names + iterator kinds), never by a
# constant, a dtype, a rank or a tensor name. A capture that spells the same quantization with the
# operands in another order, or over another float width, matches the same way.
# ---------------------------------------------------------------------------------------------

def _is_amax(op) -> bool:
    """A single-result reduce whose body is |x| folded with a running max."""
    return (op.name == "linalg.generic"
            and any("reduction" in k for k in _iterator_kinds(op))
            and sorted(_body_op_names(op)) == ["arith.maximumf", "math.absf"])


def _is_scale(op) -> bool:
    """An all-parallel map that divides its single input by a loop-invariant value."""
    return (op.name == "linalg.generic"
            and not any("reduction" in k for k in _iterator_kinds(op))
            and _body_op_names(op) == ["arith.divf"]
            and len(op.operands) >= 2)


def _is_quantize(op) -> bool:
    """An all-parallel map spelling divide -> round -> clamp -> narrow."""
    body = sorted(_body_op_names(op))
    return (op.name == "linalg.generic"
            and not any("reduction" in k for k in _iterator_kinds(op))
            and body == ["arith.divf", "arith.fptosi", "arith.maximumf", "arith.minimumf",
                         "math.roundeven"])


def _yielded_const(op, which: str):
    """The loop-invariant float an ``arith.constant`` in the body supplies, or None.

    Read off the body's own constant ops rather than assumed, so a capture that clamps to a
    different range (a 7-bit weight, an unsigned scheme) is either read correctly or refused.
    """
    from xdsl.dialects.builtin import FloatAttr
    # The clamp bounds are FUNCTION-SCOPE constants captured into the region (xDSL prints them
    # before the generic and refers to them from the body), not ops inside the body -- so the search
    # is over the body ops' OPERANDS, following each to its defining op wherever that op lives.
    vals = []
    for inner in op.regions[0].blocks[0].ops:
        for operand in getattr(inner, "operands", ()):
            owner = getattr(operand, "owner", None)
            if getattr(owner, "name", None) != "arith.constant":
                continue
            v = owner.properties.get("value")
            if isinstance(v, FloatAttr):
                vals.append(float(v.value.data))
    if not vals:
        return None
    return max(vals) if which == "max" else min(vals)


def plan(module, activation_args, load_arg):
    """Find every weight-invariant quantize chain and RECOMPUTE its results. No mutation.

    Returns ``(chains, values)`` where ``chains`` is a list of ``(amax_op, scale_op, quant_op,
    args)`` and ``values`` maps a :class:`HoistedArg` key to its numpy array. The same walk drives
    :func:`apply`, so the IR rewrite and the emitted bytes are one function of one input -- the
    property that keeps the object and the ABI table from describing different arguments.
    """
    import numpy as np

    reaches = activation_reach(module, activation_args)
    chains, values = [], {}
    n = 0
    for op in module.walk():
        if not _is_quantize(op):
            continue
        if any(reaches(o) for o in op.operands):
            continue                                    # activation side: correctly per-inference
        # ins = (source, scale); outs = the init. Take the operand order from the op, not a rule.
        ins = list(op.operands)
        if len(ins) < 3:
            raise QuantHoistRefused("quantize generic with fewer than 3 operands; unexpected shape")
        src, scale_v = ins[0], ins[1]
        scale_op = getattr(scale_v, "owner", None)
        if not (scale_op is not None and _is_scale(scale_op)):
            raise QuantHoistRefused("quantize's scale operand is not a divide map; refusing")
        amax_v = list(scale_op.operands)[0]
        amax_op = getattr(amax_v, "owner", None)
        if not (amax_op is not None and _is_amax(amax_op)):
            raise QuantHoistRefused("scale's input is not an amax reduce; refusing")

        qmax = _yielded_const(scale_op, "max")
        qhi = _yielded_const(op, "max")
        qlo = _yielded_const(op, "min")
        if qmax is None or qhi is None or qlo is None:
            raise QuantHoistRefused("quantization range not readable from the body constants")

        w = _resolve_source(module, list(amax_op.operands)[0], load_arg)
        wq_src = _resolve_source(module, src, load_arg)
        if w.shape != wq_src.shape:
            raise QuantHoistRefused("amax and quantize read differently shaped views of the weight")

        # The reduction axis is the one the OUTPUT map drops, derived from the op's own maps.
        red = _reduced_axes(amax_op)
        amax = np.max(np.abs(w.astype(np.float32)), axis=red)
        want = _shape_of(amax_op.results[0].type)
        if tuple(amax.shape) != want:
            raise QuantHoistRefused(
                f"recomputed amax shape {tuple(amax.shape)} != IR's {want}; refusing")
        scale = (amax / np.float32(qmax)).astype(np.float32)
        b = _broadcast_for(op, scale, wq_src.shape)
        q = np.clip(np.rint(wq_src.astype(np.float32) / b), qlo, qhi)
        qt = _np_of(_elem_str(op.results[0].type))
        qv = q.astype(qt)
        if tuple(qv.shape) != _shape_of(op.results[0].type):
            raise QuantHoistRefused("recomputed quantized weight disagrees with the IR's shape")

        sa = HoistedArg(f"qhoist::scale::{n}", tuple(scale.shape), "f32")
        qa = HoistedArg(f"qhoist::weight::{n}", tuple(qv.shape), _elem_str(op.results[0].type))
        values[sa.key], values[qa.key] = scale, qv
        chains.append((amax_op, scale_op, op, (sa, qa)))
        n += 1
    return chains, values


def _reduced_axes(op) -> tuple[int, ...]:
    """The iteration dims that do NOT appear in the op's OUTPUT indexing map -- the reduce axes,
    derived from the op's own maps rather than assumed to be the innermost."""
    from xdsl.ir.affine import AffineDimExpr
    maps = op.properties.get("indexing_maps")
    if maps is None:
        raise QuantHoistRefused("reduce has no indexing maps")
    out_map = maps.data[-1]
    kept = {r.position for r in out_map.data.results if isinstance(r, AffineDimExpr)}
    n_loops = len(_iterator_kinds(op))
    axes = tuple(i for i in range(n_loops) if i not in kept)
    if not axes:
        raise QuantHoistRefused("reduce whose output map keeps every dim; refusing")
    return axes


def _broadcast_for(op, scale, src_shape):
    """``scale`` reshaped so numpy broadcasts it against ``src_shape`` exactly as the op's own
    indexing map broadcasts it. Derived from the map; never positional."""
    import numpy as np
    from xdsl.ir.affine import AffineDimExpr
    maps = op.properties.get("indexing_maps")
    src_map, scale_map = maps.data[0], maps.data[1]
    src_dims = [r.position for r in src_map.data.results if isinstance(r, AffineDimExpr)]
    scale_dims = [r.position for r in scale_map.data.results if isinstance(r, AffineDimExpr)]
    if len(src_dims) != len(src_shape):
        raise QuantHoistRefused("quantize's source map is not a plain permutation; refusing")
    shape = [scale.shape[scale_dims.index(d)] if d in scale_dims else 1 for d in src_dims]
    return np.asarray(scale).reshape(shape)


_NP_OF = {"f32": "float32", "f16": "float16", "f64": "float64",
          "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64"}


def _np_of(mlir_dtype: str):
    import numpy as np
    if mlir_dtype not in _NP_OF:
        raise QuantHoistRefused(f"element type {mlir_dtype!r} has no numpy mapping; refusing")
    return getattr(np, _NP_OF[mlir_dtype])


def apply(module, activation_args, load_arg):
    """Rewrite the module in place; return ``(args, values, n_chains)``.

    Each chain's quantized weight and its scale become TRAILING `@forward` arguments (appended after
    any `qinner` args, which `qinner.lift` has already added by the time this runs), and the three
    generics that computed them are erased. Nothing else moves: the contraction still reads a
    quantized weight and a scale, they simply arrive instead of being recomputed.
    """
    from xdsl.dialects.builtin import FunctionType

    chains, values = plan(module, activation_args, load_arg)
    if not chains:
        return [], {}, 0

    func = _forward_func(module)
    block = func.regions[0].blocks[0]
    args: list[HoistedArg] = []
    for amax_op, scale_op, quant_op, (sa, qa) in chains:
        s_arg = block.insert_arg(scale_op.results[0].type, len(block.args))
        q_arg = block.insert_arg(quant_op.results[0].type, len(block.args))
        quant_op.results[0].replace_all_uses_with(q_arg)
        scale_op.results[0].replace_all_uses_with(s_arg)
        for dead in (quant_op, scale_op, amax_op):
            if not any(True for _ in dead.results[0].uses):
                dead.detach()
                dead.erase()
        args.extend((sa, qa))

    func.properties["function_type"] = FunctionType.from_lists(
        [a.type for a in block.args], list(func.function_type.outputs.data))
    return args, values, len(chains)


def ensure_registered() -> str:
    """Register the feature (idempotent). Eager at import of `impr_features`, and reachable from its
    `_try_lazy_register` name hook so the lowering SUBPROCESS resolves it too."""
    from . import impr_features as F
    if FEATURE not in F._REGISTRY:
        F.register(F.ImprFeature(
            name=FEATURE, action_class="PASS",
            description="recompute weight-invariant dynamic-quantization chains at build time and "
                        "bind the results as trailing @forward arguments"))
    return FEATURE
