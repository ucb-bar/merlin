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

_LAYOUT_ONLY = frozenset({
    "linalg.transpose",
    "tensor.cast",
    "tensor.collapse_shape",
    "tensor.expand_shape",
    "tensor.extract_slice",
})


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


def traces_to_prequantized(value, *, max_layout_hops: int = 4) -> bool:
    """Whether ``value`` comes from a captured dequant op through layout-only operations."""
    current = value
    seen: set[int] = set()
    for _ in range(max_layout_hops + 1):
        op = _owner(current)
        if op is None:
            return False
        name = _name(op)
        if name.startswith("quant_ext.dequantize"):
            return True
        if name not in _LAYOUT_ONLY:
            return False
        key = id(op)
        if key in seen:
            return False
        seen.add(key)
        operands = list(getattr(op, "operands", ()))
        if not operands:
            return False
        current = operands[0]
    return False


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
        register(ImprFeature(
            name=FEATURE,
            action_class="HEURISTIC",
            description=(
                "Honor the capture's quantization reach: run integer lowering only for linalg "
                "contractions whose input traces through layout-only views to a "
                "quant_ext.dequantize producer, and leave conv/nonlinear approximation passes off. "
                "This prevents the compiler from silently widening a Linear-only TorchAO recipe "
                "to BMMs, convs and activations. Structure-driven, frontend/target/model independent; "
                "default-off and full-output gated."),
        ))
    return FEATURE
