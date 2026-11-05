"""Layout, tail, broadcast and alias OBSERVATIONS of one structured operation, read from the graph.

A software declaration admits an operation only for the layouts, tail handling, broadcasting and
aliasing it names, and the screen treats an absent observation as unresolved -- never as a match. These
four observations are therefore derived here from the operation itself, in the declaration's own
vocabulary, and each is ``None`` (unknown) whenever the graph does not establish it:

* ``layout``: ``row_major_contiguous`` when every shaped operand and result is a tensor with no
  encoding, or a memref with no explicit layout. An encoding or a strided layout is not assumed away.
* ``tails``: ``zero_pad_valid_window`` for a contraction whose body is only operand extension, a
  product and a sum (so zero is the additive identity of every padded reduction element and padded
  result rows and columns fall outside the valid window) over a static iteration space.
* ``broadcasting``: from the indexing maps. ``none`` when no parallel dimension is shared by every
  input and the result and each input contributes at most one exclusive result dimension;
  ``independent_batches`` when the shared (batch) dimensions index every input; ``operand_broadcast``
  otherwise (an input omits a dimension it would need to be independent).
* ``aliasing``: ``disjoint_inputs_outputs`` when no output (init) operand is the same SSA value as an
  input operand; ``in_place`` otherwise.

Nothing here names a target or an operation: it reads types, indexing maps, iterator types and the
body's operation names.
"""

from __future__ import annotations

from typing import Any

#: Body operations that keep zero the additive identity of a padded reduction element.
_ZERO_PAD_EXACT_BODY = frozenset(
    {
        "arith.extsi",
        "arith.extui",
        "arith.extf",
        "arith.sitofp",
        "arith.uitofp",
        "arith.muli",
        "arith.mulf",
        "arith.addi",
        "arith.addf",
        "linalg.yield",
    }
)
_NAMED_CONTRACTIONS = frozenset({"linalg.matmul", "linalg.batch_matmul", "linalg.matvec", "linalg.vecmat"})


def type_layout(value_type) -> str | None:
    """``row_major_contiguous`` for an unencoded tensor or identity-layout memref, else ``None``.

    The same rule judges one SSA value (a transfer edge) and every operand of an operation, so a
    transfer and the operations on either side of it are described in one vocabulary.
    """
    kind = type(value_type).__name__
    if kind == "TensorType":
        encoding = getattr(value_type, "encoding", None)
        return None if encoding is not None and type(encoding).__name__ != "NoneAttr" else "row_major_contiguous"
    if kind == "MemRefType":
        layout = getattr(value_type, "layout", None)
        return None if layout is not None and type(layout).__name__ != "NoneAttr" else "row_major_contiguous"
    return None


def _layout(op) -> str | None:
    values = [*op.operands, *op.results]
    shaped = [value.type for value in values if hasattr(value.type, "get_shape") or hasattr(value.type, "shape")]
    if not shaped:
        return None
    layouts = {type_layout(value_type) for value_type in shaped}
    return "row_major_contiguous" if layouts == {"row_major_contiguous"} else None


def _n_inputs(op) -> int:
    return len(getattr(op, "inputs", ())) or max(len(op.operands) - len(op.results), 0)


def _maps(op):
    from merlin.compile.analysis import contraction_shapes as KS

    maps, kinds = KS.indexing_maps(op), KS._iterator_types(op)  # noqa: SLF001
    if not maps or not kinds:
        return None, None
    positions = []
    for results in maps:
        dims = []
        for expr in results:
            position = KS._dim_position(expr)  # noqa: SLF001
            if position is None:
                return None, None
            dims.append(position)
        positions.append(dims)
    return positions, kinds


def _tails(op, family: str | None) -> str | None:
    from merlin.common import mlir_query as mq

    if family != "contraction":
        return None
    name = mq.op_name(op)
    if name not in _NAMED_CONTRACTIONS:
        body = [mq.op_name(inner) for inner in mq.walk(op)][1:]
        if not body or any(inner not in _ZERO_PAD_EXACT_BODY for inner in body):
            return None
    for value in [*op.operands, *op.results]:
        shape, _dtype = mq.type_shape_dtype(value.type)
        if shape and any(not isinstance(extent, int) or extent < 0 for extent in shape):
            return None
    return "zero_pad_valid_window"


def _broadcasting(op) -> str | None:
    positions, kinds = _maps(op)
    if positions is None:
        return None
    n_in = _n_inputs(op)
    inputs, result = positions[:n_in], positions[-1] if len(positions) > n_in else None
    if not inputs or result is None:
        return None
    parallel = {dim for dim, kind in enumerate(kinds) if kind == "parallel"}
    shared = {dim for dim in parallel if dim in result and all(dim in dims for dims in inputs)}
    exclusive = [len({dim for dim in dims if dim in parallel and dim in result} - shared) for dims in inputs]
    if any(count > 1 for count in exclusive) or any(dim not in set().union(*map(set, inputs)) for dim in result):
        return "operand_broadcast"
    return "independent_batches" if shared and len(inputs) > 1 else "none"


def _aliasing(op) -> str | None:
    n_in = _n_inputs(op)
    if n_in <= 0 or len(op.operands) <= n_in:
        return None
    inputs, outputs = list(op.operands[:n_in]), list(op.operands[n_in:])
    return "in_place" if any(output is source for output in outputs for source in inputs) else "disjoint_inputs_outputs"


def observe(op, family: str | None) -> dict[str, Any]:
    """The four observations for a structured (``linalg.*``) operation; every key ``None`` otherwise."""
    from merlin.common import mlir_query as mq

    if not mq.op_name(op).startswith("linalg."):
        return {"layout": None, "tails": None, "broadcasting": None, "aliasing": None}
    return {
        "layout": _layout(op),
        "tails": _tails(op, family),
        "broadcasting": _broadcasting(op) if family == "contraction" else None,
        "aliasing": _aliasing(op),
    }
