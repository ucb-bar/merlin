"""Operation rows OBSERVED from a written accelerator-interface command buffer.

A capsule written in the shared interface grammar (``merlin_iface``) states its operations as
commands over declared tensors. Screening such a capsule from its entry left every constraint the
software declarations name (rank, layout, tails, broadcasting, aliasing, composition) unresolved,
because the entry carries none of them. They are properties of the written program, so they are read
from it here, into the same row vocabulary a captured application's operations use
(``operation_accounting.admit_operation_row``):

* ``layout``: ``row_major_contiguous`` for operand and result tensors the command buffer declares
  densely (the grammar's tensors carry a shape and an element type and nothing else); a command whose
  tensors are not all declared leaves it unknown.
* ``tails``: ``zero_pad_valid_window`` for an integer multiply-accumulate contraction over static
  extents (the same rule ``access_observations`` applies to a named integer contraction).
* ``broadcasting``: ``none`` for a single contraction; for a batched one, ``independent_batches``
  when both operands carry the batch extent and ``operand_broadcast`` when one does not.
* ``aliasing``: ``disjoint_inputs_outputs`` when the destination names no operand.
* epilogue stages on a commit or a named command become fused ``elementwise_map`` rows observed as
  ``composed_with`` the carrier's family, with ``scale_granularity: tensor`` for a scalar scale.

Residency and readout bookkeeping (pack, evict, a stage-less commit) are support rows. Nothing here
names a target or an opcode's hardware meaning beyond the grammar's own families.
"""

from __future__ import annotations

import math
from typing import Any

_ROW_MAJOR = "row_major_contiguous"
_INTEGER_PREFIX = "i"


def epilogue_carrier(operation: str, family: str | None) -> str | None:
    """Preserve a named operand sum as the stage's producer, not any map."""
    return operation if operation == "residual_add" else family


def _is_integer(dtype: str | None) -> bool:
    return isinstance(dtype, str) and dtype.startswith(_INTEGER_PREFIX) and dtype[1:].isdigit()


def _typed(tensors: dict, name: str | None) -> dict | None:
    spec = tensors.get(name) if name is not None else None
    if not isinstance(spec, dict) or not isinstance(spec.get("shape"), list):
        return None
    return {"dtype": spec.get("dtype"), "shape": list(spec["shape"])}


def _static(*typed: dict | None) -> bool:
    return all(
        item is not None and all(isinstance(extent, int) and extent > 0 for extent in item["shape"]) for item in typed
    )


#: Interface movement forms whose single-source work is stated by the form itself.
_VIEW_FORM_WORK = {"permute": "permutation", "reshape": "copy", "upsample": "copy", "as_strided": "copy"}


def _strided_read_class(attributes: dict, source: dict, result: dict) -> str | None:
    """Broadcast class of an ``as_strided`` read, from its static sizes, strides and storage offset.

    Every result element is one source element at ``offset + sum(i_k * stride_k)``. Distinct addresses
    (no element read twice) is ``none``; a repeated read through a contiguous innermost run (stride 1: the
    DMA reads whole rows at an arbitrary row pitch) is ``outer``; anything else is not stated.
    """
    sizes, strides, offset = attributes.get("sizes"), attributes.get("strides"), attributes.get("storage_offset")
    if (
        not isinstance(sizes, list) or not isinstance(strides, list) or len(sizes) != len(strides) or not sizes
        or any(type(v) is not int or v < 1 for v in [*sizes, *strides]) or type(offset) is not int or offset < 0
        or result["shape"] != sizes or math.prod(sizes) > (1 << 22)
        or offset + sum((n - 1) * s for n, s in zip(sizes, strides, strict=True)) >= math.prod(source["shape"])
    ):
        return None
    addresses = {offset}
    for n, s in zip(sizes, strides, strict=True):
        addresses = {a + i * s for a in addresses for i in range(n)}
    if len(addresses) == math.prod(sizes):
        return "none"
    return "outer" if strides[-1] == 1 else None


def _movement_broadcasting(attributes: dict, source: dict | None, result: dict | None) -> str | None:
    """Broadcast class of a shape-changing single-source movement, from its form's own static parameters.

    A ``slice`` or ``pad`` form reads each result element from one selected source element (or a pad fill):
    no operand is broadcast, whatever the result extent. A ``broadcast`` or ``repeat`` form replays source
    rows: ``outer`` when the innermost axis is replayed unchanged, ``inner`` when it is expanded, and no
    observation (unresolved) when the form's parameters do not state the replay exactly.
    """
    if not _static(source, result):
        return None
    form = attributes.get("form")
    if form in {"slice", "pad"}:
        return "none"
    if form == "permute":
        perm = attributes.get("perm")
        # An axis permutation reads each result element from exactly one source element: nothing is
        # broadcast. The parameter must state a true permutation of the source's axes.
        if isinstance(perm, list) and sorted(perm) == list(range(len(source["shape"]))) and len(result["shape"]) == len(perm):
            return "none" if all(result["shape"][i] == source["shape"][p] for i, p in enumerate(perm)) else None
        return None
    if form == "as_strided":
        return _strided_read_class(attributes, source, result)
    if form == "upsample":
        # Nearest upsample by integer factors: each result element replays one source element along the
        # resampled axes. The class is the broadcast class of the replay: `outer` while the innermost axis
        # is not resampled (scale 1: whole contiguous rows are replayed), `inner` otherwise.
        scales = attributes.get("scales")
        if (
            not isinstance(scales, list) or len(scales) != len(source["shape"]) or len(result["shape"]) != len(scales)
            or any(type(v) is not int or v < 1 for v in scales)
            or any(out != n * s for out, n, s in zip(result["shape"], source["shape"], scales, strict=True))
        ):
            return None
        return "inner" if scales[-1] != 1 else "outer"
    if form == "reshape":
        # A row-major reshape (view, squeeze, unsqueeze, flatten) relabels the same elements in the same
        # order: no element is replayed and none is dropped.
        return "none" if math.prod(source["shape"]) == math.prod(result["shape"]) else None
    src, out = source["shape"], result["shape"]
    if form == "broadcast":
        if len(out) < len(src):
            return None
        padded = [1] * (len(out) - len(src)) + list(src)
        if any(have not in (1, want) for have, want in zip(padded, out)):
            return None
        return "inner" if padded[-1] != out[-1] else "outer"
    if form == "repeat":
        repeats = attributes.get("repeats")
        if (
            not isinstance(repeats, list)
            or len(repeats) != len(src)
            or len(out) != len(src)
            or any(type(count) is not int or count < 1 for count in repeats)
            or any(extent * count != want for extent, count, want in zip(src, repeats, out))
        ):
            return None
        return "inner" if repeats[-1] != 1 else "outer"
    return None


def _joined_movement_broadcasting(operation: str, attributes: dict, sources: list, result: dict | None) -> str | None:
    """Broadcast class of a multi-source movement (``concat`` / ``insert_slice``), from its static geometry.

    ``concat`` writes each source into its own disjoint extent of the result along ``axis``; ``insert_slice``
    writes ``src`` at ``offsets`` with ``strides`` into a copy of ``dest``. Each result element is read from
    exactly one source element: ``none``. Stated only when the typed geometry proves it (ranks equal, every
    other extent equal, the written extents inside the result); otherwise no observation (unresolved).
    """
    if not sources or not _static(*sources, result):
        return None
    out = result["shape"]
    shapes = [item["shape"] for item in sources]
    if any(len(shape) != len(out) for shape in shapes):
        return None
    if operation == "concat":
        axis = attributes.get("axis")
        if type(axis) is not int or not -len(out) <= axis < len(out):
            return None
        axis %= len(out)
        if sum(shape[axis] for shape in shapes) != out[axis]:
            return None
        if any(shape[i] != out[i] for shape in shapes for i in range(len(out)) if i != axis):
            return None
        return "none"
    if operation == "insert_slice":
        offsets, strides = attributes.get("offsets"), attributes.get("strides")
        if len(shapes) != 2 or shapes[0] != out:
            return None
        for stated in (offsets, strides):
            if not isinstance(stated, list) or len(stated) != len(out) or any(type(x) is not int for x in stated):
                return None
        for extent, size, offset, stride in zip(out, shapes[1], offsets, strides):
            if offset < 0 or stride < 1 or offset + (size - 1) * stride >= extent:
                return None
        return "none"
    return None


#: The static window parameters each ``window_sum`` mode states (``WINDOW_SUM`` in the command-buffer ABI);
#: the adaptive modes state none -- their windows follow from the source and result extents alone.
_WINDOW_SUM_GEOMETRY = {
    "window": ("kernel", "stride", "padding"),
    "window_transpose": ("kernel", "stride", "padding"),
    "fold": ("kernel", "stride", "padding", "dilation"),
    "reduce": ("dims",),
    "adaptive": (),
    "adaptive_transpose": (),
}


def _window_sum_tails(attributes: dict, source: dict | None, result: dict | None) -> str | None:
    """``zero_pad_valid_window`` for an integer window sum whose geometry the command states exactly.

    The sums are exact integer accumulations over windows whose out-of-range taps are zero (padding) or
    absent (a clipped window), the same tail rule an integer contraction over static extents is observed
    by. Nothing is observed unless the operand is an integer, every extent is static and the mode's own
    geometry parameters are all stated as integer lists: a geometry left implicit stays unresolved.
    """
    geometry = _WINDOW_SUM_GEOMETRY.get(attributes.get("mode"))
    if geometry is None or source is None or not _is_integer(source["dtype"]) or not _static(source, result):
        return None
    for key in geometry:
        values = attributes.get(key)
        if not isinstance(values, list) or not values or any(type(value) is not int for value in values):
            return None
    return "zero_pad_valid_window"


def _row(index: int, opcode: str, operation: str, family: str | None, **fields) -> dict:
    return {
        "ordinals": [index],
        "count": 1,
        "operation": operation,
        "mlir_operation": f"merlin_iface.{operation}",
        "command_opcode": opcode,
        "semantic_family": family,
        "frontend_op": None,
        **fields,
    }


#: Command attributes that state GEOMETRY (the command-buffer ABI's window and store-path fields).
#: They are integer lists by definition and say nothing about how a readout scales its values.
_GEOMETRY_ATTRIBUTES = frozenset(
    {"kernel", "stride", "padding", "dilation", "pool_in_dims", "pool_size", "pool_stride", "pool_padding"}
)


def _per_tensor_scales(attributes: dict) -> bool:
    """Whether every multiplier a command states is one value for the whole tensor: no attribute other
    than its stage list and its geometry is an array or a mapping (a per-channel scale would be one)."""
    return all(
        not isinstance(value, (list, dict))
        for key, value in attributes.items()
        if key != "epilogue" and key not in _GEOMETRY_ATTRIBUTES and value is not None
    )


def _stage_rows(
    *,
    index: int,
    opcode: str,
    stages: list[str],
    producer: str | None,
    operand_dtype: str | None,
    result_dtype: str | None,
    accumulator: str,
    destination_declared: bool,
    attributes: dict,
    column_source: dict | None = None,
    result_type: dict | None = None,
) -> list[dict]:
    from merlin.targetgen.semantic_families import from_op

    scalar_scales = _per_tensor_scales(attributes)
    return [
        _row(
            index,
            opcode,
            stage,
            from_op(stage),
            disposition="unclassified",
            operand_dtypes=[operand_dtype] if operand_dtype else [],
            ordered_result_types=[result_type] if result_type is not None else [],
            **({"reduction_kind": "max"} if stage == "maxpool" else {}),
            accumulator_dtypes=[accumulator],
            result_dtypes=[result_dtype] if result_dtype else [],
            layout=_ROW_MAJOR if destination_declared else None,
            aliasing="disjoint_inputs_outputs",
            **(
                {"logical_output_column_source": {**column_source, "attributes": attributes}}
                if column_source is not None and "acc_scale_by_column" in attributes
                else {}
            ),
            composed_observation={
                "epilogues": [stage],
                "composed_with": [producer] if producer else [],
                "scale_granularity": "tensor"
                if scalar_scales
                else "channel"
                if "acc_scale_by_column" in attributes
                else None,
            },
        )
        for stage in stages
    ]


def command_rows(cb: dict[str, Any]) -> list[dict]:
    """One row per command and per declared epilogue stage, in program order."""
    from merlin.runtime.commandbuffer import bias_tensor_name
    from merlin.targetgen.contract.interface_emit import _NAMED_OPCODE_TO_OP, _OPCODE_TO_OP, _acc_dtype
    from merlin.targetgen.semantic_families import from_op, reduction_kind_of_op

    tensors = dict(cb.get("tensors") or {})
    accumulator = _acc_dtype(cb)
    resident: dict[str, str] = {}
    produced_by: dict[str, tuple[str | None, str | None]] = {}
    column_sources: dict[str, dict] = {}
    rows: list[dict] = []
    for index, command in enumerate(cb.get("commands") or []):
        opcode = str(command.get("opcode"))
        operands = dict(command.get("operands") or {})
        attributes = dict(command.get("attributes") or {})
        operation = _OPCODE_TO_OP.get(opcode) or _NAMED_OPCODE_TO_OP.get(opcode) or opcode.lower()
        destination = operands.get("dst")
        sources = [value for key, value in operands.items() if key != "dst"]
        if opcode == "RES_PACK":
            resident[str(destination)] = str(operands.get("src"))
            rows.append(_row(index, opcode, operation, None, disposition="support_required"))
            continue
        if opcode == "EVICT":
            rows.append(_row(index, opcode, operation, None, disposition="support_required"))
            continue
        if opcode == "COMMIT":
            stages = [str(stage) for stage in attributes.get("epilogue") or []]
            producer, producer_dtype = produced_by.get(str(operands.get("src")), (None, None))
            rows.append(_row(index, opcode, operation, None, disposition="support_required"))
            rows.extend(
                _stage_rows(
                    index=index,
                    opcode=opcode,
                    stages=stages,
                    producer=producer,
                    operand_dtype=producer_dtype,
                    result_dtype=attributes.get("output_dtype") or accumulator,
                    accumulator=accumulator,
                    destination_declared=_typed(tensors, destination) is not None,
                    result_type=_typed(tensors, destination),
                    attributes=attributes,
                    column_source={
                        **column_sources[str(operands.get("src"))],
                        "disjoint": column_sources[str(operands.get("src"))]["disjoint"]
                        and destination not in column_sources[str(operands.get("src"))]["operand_names"]
                        and destination != attributes.get("bias"),
                        "result": _typed(tensors, destination),
                        "bias": _typed(tensors, bias_tensor_name(operands, attributes, op=operation))
                        if any(stage in {"bias", "bias_add"} for stage in stages)
                        else None,
                    }
                    if str(operands.get("src")) in column_sources
                    else None,
                )
            )
            continue
        family = from_op(operation)
        typed_sources = [_typed(tensors, resident.get(str(name), str(name))) for name in sources]
        result = _typed(tensors, destination)
        if result is None and destination is not None:
            # A contraction commits into an accumulator handle, whose extent the command implies.
            lhs, rhs = (typed_sources + [None, None])[:2]
            if lhs is not None and rhs is not None and len(lhs["shape"]) >= 2 and len(rhs["shape"]) >= 2:
                result = {"dtype": accumulator, "shape": [*lhs["shape"][:-1], rhs["shape"][-1]]}
        operand_dtype = next((item["dtype"] for item in typed_sources if item is not None and item["dtype"]), None)
        # A standalone operand sum is an elementwise operation, but its readout is not attached
        # to every possible elementwise map. Preserve its specific carrier so a SW declaration
        # cannot accidentally license ReLU after an unrelated map on the same family.
        carrier = epilogue_carrier(operation, family)
        produced_by[str(destination)] = (carrier, operand_dtype)
        if operation in {"matmul", "conv2d"} and len(typed_sources) >= 2:
            column_sources[str(destination)] = {
                "operation": operation,
                "lhs": typed_sources[0],
                "rhs": typed_sources[1],
                "operand_names": [resident.get(str(name), str(name)) for name in sources],
                "disjoint": destination not in sources and destination != attributes.get("bias"),
                "result": _typed(tensors, destination),
                "bias": _typed(tensors, bias_tensor_name(operands, attributes, op=operation))
                if any(stage in {"bias", "bias_add"} for stage in attributes.get("epilogue") or ())
                else None,
            }
        row = _row(
            index,
            opcode,
            operation,
            family,
            disposition="unclassified",
            ordered_operand_types=[item for item in typed_sources if item is not None],
            ordered_result_types=[result] if result is not None else [],
            operand_dtypes=sorted({item["dtype"] for item in typed_sources if item is not None and item["dtype"]}),
            result_dtypes=[attributes.get("output_dtype") or (result or {}).get("dtype")],
            layout=_ROW_MAJOR if all(item is not None for item in [*typed_sources, result]) else None,
            aliasing="disjoint_inputs_outputs" if destination not in sources else "in_place",
        )
        # Only equal, statically declared tensor extents prove no broadcasting.
        if family in {"elementwise_map", "movement"} and typed_sources and _static(*typed_sources, result):
            if all(item["shape"] == result["shape"] for item in typed_sources):
                row["broadcasting"] = "none"
        if family == "movement" and "broadcasting" not in row and len(typed_sources) == 1:
            observed = _movement_broadcasting(attributes, typed_sources[0], result)
            if observed is not None:
                row["broadcasting"] = observed
            # The work a view-like form performs, stated so a form-constrained capability applies to it:
            # an axis permutation needs the transposing mechanism; a reshape relabels row-major bytes (a copy).
            if row.get("broadcasting") == "none" and attributes.get("form") in _VIEW_FORM_WORK:
                row["movement_form"] = _VIEW_FORM_WORK[attributes["form"]]
        if family == "movement" and operation == "fill" and not typed_sources and _static(result):
            # A fill reads no operand: its element type is the result's, and it writes one quantized code
            # everywhere. Code 0 is the zero_fill form (a load from DRAM address 0: nothing is read or
            # replayed, class none); any other code is one constant row replayed (class outer).
            row["operand_dtypes"] = [result["dtype"]]
            code = attributes.get("value_code")
            if type(code) is int:
                row["broadcasting"] = "none" if code == 0 else "outer"
                row["movement_form"] = "zero_fill" if code == 0 else "broadcast"
        if family == "movement" and "broadcasting" not in row and operation in {"concat", "insert_slice"}:
            observed = _joined_movement_broadcasting(operation, attributes, typed_sources, result)
            if observed is not None:
                row["broadcasting"] = observed
        if family == "reduction":
            # The kind a NAMED reduction computes (a mean or an average pool reduces by sum; the division is a
            # readout scale applied after it). An op without a known kind observes none, never a guess.
            kind = reduction_kind_of_op(operation)
            if kind is not None:
                row["reduction_kind"] = kind
            if operation == "window_sum" and len(typed_sources) == 1:
                tails = _window_sum_tails(attributes, typed_sources[0], result)
                if tails is not None:
                    row["tails"] = tails
        if opcode == "RESIDUAL_ADD" and len(typed_sources) == 2 and all(item is not None for item in typed_sources):
            row["tensor_operands"] = 2
            row["expression"] = "affine"
        if family == "elementwise_map":
            # A standalone command consumes declared tensors, not a producer's accumulator: it is
            # observed as composed with nothing and as no carrier's epilogue. Its operands are tensors
            # only, so a multiplier it carries is a scalar attribute: when it carries one and no
            # attribute is an array, each operand is scaled by one value for the whole tensor -- the
            # rule a commit's stages are observed by. A command with no multiplier states none.
            stated = {key: value for key, value in attributes.items() if key != "epilogue" and value is not None}
            multiplied = any(isinstance(value, float) for value in stated.values())
            scalar = _per_tensor_scales(attributes)
            row["composed_observation"] = {
                "epilogues": [],
                "composed_with": [],
                "scale_granularity": "tensor" if multiplied and scalar else None,
            }
        if family == "contraction" and len(typed_sources) >= 2 and None not in typed_sources[:2] and result:
            lhs, rhs = typed_sources[0], typed_sources[1]
            row["accumulator_dtypes"] = [accumulator]
            row["result_dtypes"] = [attributes.get("output_dtype") or accumulator]
            out_shape = result["shape"]
            row["contraction_shape"] = {"M": out_shape[-2], "K": rhs["shape"][-2], "N": out_shape[-1]}
            row["tails"] = (
                "zero_pad_valid_window"
                if _is_integer(lhs["dtype"]) and _is_integer(rhs["dtype"]) and _static(lhs, rhs, result)
                else None
            )
            batched = len(out_shape) > 2
            row["broadcasting"] = (
                "none"
                if not batched
                else "independent_batches"
                if len(lhs["shape"]) == len(rhs["shape"]) == len(out_shape)
                else "operand_broadcast"
            )
        rows.append(row)
        stages = [str(stage) for stage in attributes.get("epilogue") or []]
        if stages:
            stage_operand_dtype = operand_dtype if family == "contraction" else (result or {}).get("dtype")
            rows.extend(
                _stage_rows(
                    index=index,
                    opcode=opcode,
                    stages=stages,
                    producer=carrier,
                    operand_dtype=stage_operand_dtype,
                    result_dtype=attributes.get("output_dtype") or (result or {}).get("dtype"),
                    accumulator=accumulator,
                    destination_declared=_typed(tensors, destination) is not None,
                    result_type=_typed(tensors, destination),
                    attributes=attributes,
                    column_source=column_sources.get(str(destination)),
                )
            )
    return rows
