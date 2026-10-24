"""Choose the independent golden reference for a converted PT2E Q/DQ graph, per graph.

Two references exist, with different arithmetic:

* the exact-rational oracle (``exact_quantized_reference``): every value between Q/DQ
  boundaries is an exact rational, rounded once where it leaves (a quantize or an output);
* model2MLIR's ``pt2e_integer_reference``: integer contractions over float32-dequantized
  operands, with every other operation evaluated in ordinary framework floating point.

They agree with a program that rounds after every floating operation only when no rounded
floating result feeds another rounding operation, every contraction reads float32-dequantized
operands, and every operation has an exact rational model. That is decided here from the
graph's structure before anything is evaluated. A graph classified closed-rational is
evaluated by the exact oracle only: a refusal inside it still fails the reference closed.
"""

from __future__ import annotations

import hashlib
import operator
from pathlib import Path

EXACT_ENGINE = "exact_rational_oracle"
FRAMEWORK_ENGINE = "pt2e_integer_reference"

#: Operations whose result is a rearrangement of their operands' elements.
_MOVEMENT = frozenset(
    {
        "view",
        "reshape",
        "_unsafe_view",
        "flatten",
        "clone",
        "detach",
        "contiguous",
        "alias",
        "permute",
        "transpose",
        "t",
        "squeeze",
        "unsqueeze",
        "expand",
        "broadcast_to",
        "slice",
        "select",
    }
)

#: Operations with an exact model that commute with rounding: each result element is an
#: operand element, a constant, or a selection among them, so no new rounding is introduced.
_ROUNDING_FREE = _MOVEMENT | frozenset(
    {
        "repeat",
        "cat",
        "stack",
        "constant_pad_nd",
        "slice_scatter",
        "select_scatter",
        "relu",
        "neg",
        "abs",
        "clamp",
        "clamp_min",
        "clamp_max",
        "hardtanh",
        "amax",
        "amin",
        "where",
        "isnan",
        "native_dropout",
        "_assert_tensor_metadata",
        "fill",
        "full",
        "full_like",
        "zeros",
        "zeros_like",
        "ones",
        "ones_like",
        "zero",
    }
)

#: Constants the oracle carries as exact rationals even without a rational operand.
_CONSTANTS = frozenset({"fill", "full", "full_like", "zeros", "zeros_like", "ones", "ones_like"})

#: Operations whose result is a framework Boolean tensor even from rational operands.
_BOOLEAN = frozenset({"isnan", "_assert_tensor_metadata"})

#: Operations with an exact model whose floating result the program rounds.
_ROUNDING = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "true_divide",
        "fmod",
        "remainder",
        "cumsum",
        "sum",
        "mean",
        "prod",
        "elu",
        "col2im",
    }
)

#: Contractions and the argument positions of their quantized operands.
_CONTRACTION_OPERANDS = {
    "mm": (0, 1),
    "bmm": (0, 1),
    "matmul": (0, 1),
    "linear": (0, 1),
    "addmm": (1, 2),
    "baddbmm": (1, 2),
    "conv1d": (0, 1),
    "conv2d": (0, 1),
    "conv3d": (0, 1),
    "convolution": (0, 1),
    "convolution_backward": (0, 1, 2),
}

_POOL_PREFIXES = ("avg_pool", "adaptive_avg_pool", "_adaptive_avg_pool")
_PAD_PREFIXES = ("reflection_pad", "replication_pad")
_CASTS = frozenset({"_to_copy", "to"})


def _operation(node):
    """``(op, overload)`` of an ``aten`` call, in-place ``op_`` folded onto ``op``."""
    name = str(node.target)
    if not name.startswith("aten."):
        return None, None
    parts = name.split(".")
    op = parts[1]
    if op.endswith("_") and not op.startswith("_"):
        op = op[:-1]
    return op, parts[2] if len(parts) > 2 else "default"


def _is_quantizer(node, kind):
    return str(node.target).startswith(f"quantized_decomposed.{kind}_per_")


def _dequantize_dtype(node):
    import torch

    channel = "per_channel" in str(node.target)
    position = 7 if channel else 6
    declared = node.kwargs.get("out_dtype", node.args[position] if len(node.args) > position else None)
    return declared or torch.float32


def _value_dtype(node, dtypes):
    value = node.meta.get("val", node.meta.get("tensor_meta"))
    dtype = getattr(value, "dtype", None)
    return dtype if dtype is not None else dtypes.get(node)


def _cast_dtypes(node, dtypes):
    """``(source, target)`` floating dtypes of a conversion, or ``None`` when underivable."""
    from merlin.targetgen.exact_quantized_reference import _cast_target

    source_node = node.args[0] if node.args else None
    source = dtypes.get(source_node) if hasattr(source_node, "op") else None
    if source is None:
        return None
    try:
        return source, _cast_target(str(node.target), node.args, node.kwargs, source)
    except ValueError:
        return None


def _float32_dequantized(value):
    """Whether ``value`` reaches a float32 dequantize through movement operations only."""
    import torch

    while hasattr(value, "op") and value.op == "call_function":
        if _is_quantizer(value, "dequantize"):
            return _dequantize_dtype(value) == torch.float32
        op, _ = _operation(value)
        if op not in _MOVEMENT or not value.args:
            return False
        value = value.args[0]
    return False


def _input_guard(graph_module, node):
    """``torch.export``'s unused input-guard call: it checks inputs and yields no value."""
    import importlib

    try:
        guards = getattr(importlib.import_module("torch.export._unlift"), "GuardsFn", None)
    except ImportError:
        guards = None
    if node.op != "call_module" or node.users or guards is None:
        return False
    return isinstance(graph_module.get_submodule(node.target), guards)


def classify(graph_module):
    """Structural closed-rational classification; never evaluates the graph."""
    import torch

    from merlin.targetgen.exact_quantized_reference import _float_formats

    contains = _float_formats()[1]
    rounded = {}
    dtypes = {}

    def inputs(node):
        return [x for x in torch.utils._pytree.tree_leaves((node.args, node.kwargs)) if hasattr(x, "op")]

    def refuse(reason):
        return {"closed_rational": False, "reason": reason}

    # ``exact`` holds the values the oracle carries as rationals: dequantize results, floating
    # constants, and whatever an operation computes from them. An operation with no exact
    # operand is evaluated by the framework in the oracle exactly as in the program, so it is
    # neither modeled nor rounded here; its result enters the rational domain as an exact input.
    exact = set()
    for node in graph_module.graph.nodes:
        operands = inputs(node)
        carried = next((rounded[x] for x in operands if rounded.get(x)), None)
        dtypes[node] = _value_dtype(node, dtypes)
        rounded[node] = None
        if node.op in {"placeholder", "get_attr", "output"} or _input_guard(graph_module, node):
            continue
        rational = any(x in exact for x in operands)
        if node.op != "call_function":
            if rational:
                return refuse(f"{node.op} {node.target} has no exact rational model")
            continue
        if node.target is operator.getitem or getattr(node.target, "_m2m_tuple_selection", False):
            rounded[node] = carried
            if rational:
                exact.add(node)
            continue
        if _is_quantizer(node, "quantize"):
            continue
        if _is_quantizer(node, "dequantize"):
            dtypes[node] = _dequantize_dtype(node)
            exact.add(node)
            continue
        op, _ = _operation(node)
        if not rational:
            requested = node.kwargs.get("dtype")
            if op in _CONSTANTS and (requested is None or requested.is_floating_point):
                exact.add(node)
            continue
        if op is None:
            return refuse(f"{node.target} has no exact rational model")
        if dtypes[node] is None:
            dtypes[node] = next((dtypes[x] for x in operands if dtypes.get(x) is not None), None)
        if op not in _BOOLEAN:
            exact.add(node)
        if op in _ROUNDING_FREE or op.startswith(_PAD_PREFIXES):
            rounded[node] = carried
            continue
        if op in _CASTS:
            pair = _cast_dtypes(node, dtypes)
            if pair is not None and pair[0] in contains.get(pair[1], ()):
                rounded[node] = carried
                continue
        elif op in _CONTRACTION_OPERANDS:
            positions = _CONTRACTION_OPERANDS[op]
            if any(len(node.args) <= i or not _float32_dequantized(node.args[i]) for i in positions):
                return refuse(f"{node.name} ({node.target}) contracts operands that are not float32-dequantized")
        elif op not in _ROUNDING and not op.startswith(_POOL_PREFIXES):
            return refuse(f"{node.name} ({node.target}) has no exact rational model")
        if carried is not None:
            return refuse(f"{node.name} ({node.target}) rounds the already rounded floating result of {carried}")
        rounded[node] = node.name
    return {"closed_rational": True, "reason": None}


def _identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def reference_selection(model):
    """The engine a converted graph's golden reference must use, and why."""
    selection = dict(classify(model), classifier=_identity(__file__))
    selection["engine"] = EXACT_ENGINE if selection["closed_rational"] else FRAMEWORK_ENGINE
    return selection


def run_selected_reference(selection, model, inputs, *, declared_readout=False, numerics="integer_readout"):
    """Run exactly the selected reference: ``(result, source identity of the engine that ran)``.

    The framework reference records no rational sites or post-call inputs; a caller that
    consumes them must check the selected engine (the capture worker writes the declared-graph
    reference artifact for the exact oracle only).
    """
    import copy

    if selection["engine"] == EXACT_ENGINE:
        from merlin.targetgen.core_aten_w8a8_reference import selected_pt2e_reference

        return selected_pt2e_reference(model, inputs, declared_readout=declared_readout, numerics=numerics)
    if selection["engine"] != FRAMEWORK_ENGINE:
        raise ValueError(f"unknown PT2E reference engine {selection['engine']!r}")
    from m2m.capture import pt2e_integer_reference

    identity = _identity(pt2e_integer_reference.__file__)
    result = pt2e_integer_reference.run_pt2e_integer_reference(model, copy.deepcopy(tuple(inputs)), numerics=numerics)
    return result, identity
