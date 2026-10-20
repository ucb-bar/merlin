"""Restrict integer lowering to contractions the captured graph marks as quantized.

``int8_compute`` historically meant "run every integer approximation pass Merlin has", even when
the capture's quantization recipe had rewritten only Linear weights.  That changes the program:
attention BMMs, convolutions and nonlinearities become quantized despite carrying no quantized
operand in the source graph.  This module provides a default-off, frontend-independent policy that
uses the IR's own evidence instead: a contraction is eligible only when an input traces through
layout-only views to a ``quant_ext.dequantize*`` producer.

The rule contains no framework op names, model names, shapes, or target facts.  It is usable by any
backend consuming the normalized linalg contract.  Unsupported/opaque producer chains are refused,
never guessed to be quantized.
"""

from __future__ import annotations

FEATURE = "respect_captured_quantization_scope"

_LAYOUT_ONLY = frozenset(
    {
        "linalg.transpose",
        "tensor.cast",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.extract_slice",
    }
)


def _owner(value):
    return getattr(value, "owner", None)


def _name(op) -> str:
    # xDSL represents dialects not registered in its context as ``builtin.unregistered`` and
    # preserves the real spelling in ``op_name__``.  Looking only at ``.name`` made every captured
    # quant_ext producer invisible while its printed IR still plainly said dequantize.
    attrs = getattr(op, "attributes", {}) or {}
    original = attrs.get("op_name__") if hasattr(attrs, "get") else None
    data = getattr(original, "data", None)
    if isinstance(data, str):
        return data
    return str(getattr(op, "name", getattr(getattr(op, "operation", None), "name", "")))


def _identity_map(attr) -> bool:
    amap = getattr(attr, "data", attr)
    results = list(getattr(amap, "results", ()))
    if len(results) != getattr(amap, "num_dims", -1) or getattr(amap, "num_symbols", 0):
        return False
    return all(getattr(expr, "position", None) == index for index, expr in enumerate(results))


def value_preserving_input(op):
    """The input that ``op`` returns unchanged in VALUE, or None.

    Two forms qualify, both a single-input single-result ``linalg.generic`` with identity indexing maps
    and only parallel iterators: a copy (the body yields its input) and an exact float widening (the
    body is one ``arith.extf`` of its input). ``extf`` always widens, and widening a float is exact, so
    the result holds the same numbers as the input in a wider type. A frontend that computes a
    half-precision matmul in f32 emits exactly this in front of each operand; the dequantized values
    it widens are the captured ones.
    """
    if op is None or _name(op) != "linalg.generic":
        return None
    inputs, outputs = list(getattr(op, "inputs", ())), list(getattr(op, "outputs", ()))
    if len(inputs) != 1 or len(outputs) != 1 or len(op.results) != 1:
        return None
    iterators = getattr(op, "iterator_types", None)
    kinds = [str(getattr(it, "data", it)).lower() for it in getattr(iterators, "data", ())]
    if not kinds or any("parallel" not in kind for kind in kinds):
        return None
    maps = getattr(getattr(op, "indexing_maps", None), "data", ())
    if len(maps) != 2 or not all(_identity_map(m) for m in maps):
        return None
    block = op.regions[0].blocks[0] if op.regions and op.regions[0].blocks else None
    if block is None or not block.args:
        return None
    body = list(block.ops)
    if not body or body[-1].name != "linalg.yield" or len(body[-1].operands) != 1:
        return None
    yielded = body[-1].operands[0]
    if len(body) == 1 and yielded is block.args[0]:
        return inputs[0]
    if (
        len(body) == 2
        and body[0].name == "arith.extf"
        and body[0].operands[0] is block.args[0]
        and yielded is body[0].results[0]
    ):
        return inputs[0]
    return None


def traces_to_prequantized(value, *, max_layout_hops: int = 4) -> bool:
    """Whether ``value`` comes from a captured dequant op through layout-only or value-preserving ops."""
    current = value
    seen: set[int] = set()
    for _ in range(max_layout_hops + 1):
        op = _owner(current)
        if op is None:
            return False
        name = _name(op)
        if name.startswith("quant_ext.dequantize"):
            return True
        preserved = value_preserving_input(op) if name not in _LAYOUT_ONLY else None
        if name not in _LAYOUT_ONLY and preserved is None:
            return False
        key = id(op)
        if key in seen:
            return False
        seen.add(key)
        if preserved is not None:
            current = preserved
            continue
        operands = list(getattr(op, "operands", ()))
        if not operands:
            return False
        current = operands[0]
    return False


def prequantized_source(value, *, max_hops: int = 4):
    """``value`` without its leading copies and exact widenings, when what remains traces to a captured
    dequantize; otherwise ``value`` itself.

    Integer lowering reuses a captured operand's calibrated bytes only through layouts it recognizes
    (a transpose, a reshape, the dequantize itself). Peeling the value-preserving generics a frontend
    put in front of them exposes those layouts, so the contraction reuses the captured static scale
    instead of re-quantizing the same numbers dynamically.
    """
    current = value
    for _ in range(max_hops):
        source = value_preserving_input(_owner(current))
        if source is None:
            break
        current = source
    if current is not value and traces_to_prequantized(current):
        return current
    return value


def captured_quantized_contraction(op) -> bool:
    """Selection predicate for ``quant_passes.apply_quant``.

    Only input operands are inspected.  The final DPS operand is an output/init tensor and may
    itself be reached through a fill or cast; treating it as quantization evidence would admit an
    unrelated contraction.
    """
    operands = list(getattr(op, "operands", ()))
    inputs = operands[:-1] if len(operands) >= 3 else operands
    return any(traces_to_prequantized(value) for value in inputs)


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="HEURISTIC",
                description=(
                    "Honor the capture's quantization reach: run integer lowering only for linalg "
                    "contractions whose input traces through layout-only views to a "
                    "quant_ext.dequantize producer, and leave conv/nonlinear approximation passes off. "
                    "This prevents the compiler from silently widening a Linear-only TorchAO recipe "
                    "to BMMs, convs and activations. Structure-driven, frontend/target/model independent; "
                    "default-off and full-output gated."
                ),
            )
        )
    return FEATURE
