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
            accumulator_dtypes=[accumulator],
            result_dtypes=[result_dtype] if result_dtype else [],
            layout=_ROW_MAJOR if destination_declared else None,
            aliasing="disjoint_inputs_outputs",
            composed_observation={
                "epilogues": [stage],
                "composed_with": [producer] if producer else [],
                "scale_granularity": "tensor" if scalar_scales else None,
            },
        )
        for stage in stages
    ]


def command_rows(cb: dict[str, Any]) -> list[dict]:
    """One row per command and per declared epilogue stage, in program order."""
    from merlin.targetgen.contract.interface_emit import _NAMED_OPCODE_TO_OP, _OPCODE_TO_OP, _acc_dtype
    from merlin.targetgen.semantic_families import from_op

    tensors = dict(cb.get("tensors") or {})
    accumulator = _acc_dtype(cb)
    resident: dict[str, str] = {}
    produced_by: dict[str, tuple[str | None, str | None]] = {}
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
                    attributes=attributes,
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
                    attributes=attributes,
                )
            )
    return rows
