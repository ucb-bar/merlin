"""Original-source ownership for scratch passed through a logical pointer ABI.

A scratch grant only changes which buffers are initialized/published by the
neutral caller. It proves no emitted dataflow, numerical or device execution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar

from merlin.targetgen.oot_starterkit.plan import source_operation_inventory, validate_mixed_program_plan


class SourceOwnershipError(ValueError):
    def __init__(self, message, *, code=None):
        super().__init__(message)
        self.code = code


def _source_owned_mutables(command_buffer: Mapping, inventory: Mapping) -> tuple[str, ...]:
    """Check initial-byte ownership after the caller validates the complete source/plan join."""
    abi = command_buffer["kernel_abi"]["args"]
    read = {str(arg["tensor"]) for arg in abi if arg["access"] == "read"}
    entry = command_buffer["params"]["global_program_plan"]["entry_bindings"]
    if set(entry) != read or len(entry) != len(read):
        raise SourceOwnershipError("source entry bindings differ from exact read-only pointer ABI")
    tensors = command_buffer["tensors"]
    mutable = {
        str(arg["tensor"])
        for arg in abi
        if arg["access"] in ("write", "readwrite") and tensors[arg["tensor"]]["role"] == "intermediate"
    }
    readwrite = {str(arg["tensor"]) for arg in abi if arg["access"] == "readwrite"}
    if readwrite - mutable:
        raise SourceOwnershipError("readwrite entry/output requires an explicit carried-state seed")
    outputs = set(command_buffer["kernel_abi"]["outputs"])
    if (
        len(outputs) != len(command_buffer["kernel_abi"]["outputs"])
        or set(tensors) != read | mutable | outputs
        or any(tensors[name]["role"] != "output" for name in outputs)
    ):
        raise SourceOwnershipError("one-shot model ABI has an unsupported writable pointer role")
    plan = command_buffer["params"]["global_program_plan"]
    source_values = {row["tensor"]: (row["op_index"], row["result_index"]) for row in plan["source_values"]}
    if len(source_values) != len(plan["source_values"]):
        raise SourceOwnershipError("mutable source values alias one pointer without a carry contract")
    temporary_owners = {row["tensor"]: row["source_op_index"] for row in plan["compiler_temporaries"]}
    if len(temporary_owners) != len(plan["compiler_temporaries"]):
        raise SourceOwnershipError("compiler temporaries have duplicate pointer identity")
    if read.intersection(set(source_values) | set(temporary_owners)):
        raise SourceOwnershipError("source-produced value aliases a captured read-only entry")
    # This one-shot runner has no input/output alias or epilogue-copy contract.
    # A named output must be written once by the task that owns its exact
    # source result; merely listing it in the ABI proves no produced bytes.
    for name in command_buffer["kernel_abi"]["outputs"]:
        producer = source_values.get(name)
        writers = [task for task in plan["tasks"] if name in task["writes"]]
        if (
            not any(arg["tensor"] == name and arg["access"] == "write" for arg in abi)
            or producer is None
            or len(writers) != 1
            or producer[0] not in writers[0]["source_op_indices"]
        ):
            raise SourceOwnershipError(
                f"output writer lacks exact source-result task ownership: {name!r}",
                code="output_writer_source_result_ownership_unverified",
            )

    def has_source_initializer(name: str, task: Mapping[str, Any]) -> bool:
        binding = source_values.get(name)
        if binding is None or binding[0] not in task["source_op_indices"]:
            return False
        operation = inventory["operations"][binding[0]]
        result_index = binding[1]
        results = operation["results"]
        operands = operation["operands"]
        init_index = len(operands) - len(results) + result_index
        if operation["operation"] not in ("linalg.generic", "linalg.reduce") or not 0 <= init_index < len(operands):
            return False
        init = operands[init_index]
        origin = init.get("source") or {}
        producer = origin.get("op_index")
        if (
            type(producer) is not int
            or not 0 <= producer < binding[0]
            or origin.get("result_index") != 0
            or init.get("shape") != results[result_index].get("shape")
            or init.get("dtype") != results[result_index].get("dtype")
            or inventory["operations"][producer]["operation"] != "tensor.splat"
            or any(row["op_index"] == producer for row in plan["source_values"])
        ):
            return False
        splat = inventory["operations"][producer]
        if len(splat["operands"]) != 1:
            return False
        scalar_origin = splat["operands"][0].get("source") or {}
        constant = scalar_origin.get("op_index")
        return (
            type(constant) is int
            and 0 <= constant < producer
            and scalar_origin.get("result_index") == 0
            and splat["operands"][0].get("type") == results[result_index].get("dtype")
            and inventory["operations"][constant]["operation"] == "arith.constant"
        )

    initialized: set[str] = set()
    for task in plan["tasks"]:
        if read.intersection(task["writes"]):
            raise SourceOwnershipError("task writes a captured read-only entry")
        for name in task["writes"]:
            if name in mutable:
                owner = source_values[name][0] if name in source_values else temporary_owners.get(name)
                if owner not in task["source_op_indices"]:
                    raise SourceOwnershipError(f"mutable intermediate {name!r} has no source-owned writer")
        for name in task["reads"]:
            if (
                name in mutable
                and name not in initialized
                and not (name in task["writes"] and has_source_initializer(name, task))
            ):
                raise SourceOwnershipError(f"mutable intermediate {name!r} is read before a source-owned write")
        initialized.update(name for name in task["writes"] if name in mutable)
    if initialized != mutable:
        raise SourceOwnershipError("mutable intermediate lacks a source-owned writer")
    return tuple(sorted(mutable))


def source_owned_mutables(source: bytes | str, command_buffer: Mapping, *, lowered_mlir=None) -> tuple[str, ...]:
    """Derive scratch only from the caller's original source and a completely matching plan."""
    validation = validate_mixed_program_plan(source, command_buffer, lowered_mlir)
    if not validation.get("ok"):
        raise SourceOwnershipError(f"candidate source/entry plan is incomplete: {validation.get('findings', [])[:2]}")
    return _source_owned_mutables(command_buffer, source_operation_inventory(source))


_ACTIVE = ContextVar("original_source_owned_mutables", default=None)


def _binding(command_buffer):
    # Operands and oracle answers do not confer scratch authority. Freeze the
    # complete logical storage/plan boundary, which must survive adapter copies.
    raw = {key: command_buffer.get(key) for key in ("tensors", "kernel_abi")}
    raw["plan"] = (command_buffer.get("params") or {}).get("global_program_plan")
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).digest()


@contextmanager
def source_mutable_scope(source, command_buffer, *, lowered_mlir=None):
    """Trusted caller's live scope; candidate metadata cannot reconstruct this grant."""
    mutables = source_owned_mutables(source, command_buffer, lowered_mlir=lowered_mlir)
    binding = _binding(command_buffer)
    token = _ACTIVE.set((binding, mutables))
    try:
        yield mutables
        if _binding(command_buffer) != binding:
            raise SourceOwnershipError("source-owned scratch boundary changed during execution")
    finally:
        _ACTIVE.reset(token)


def active_source_owned_mutables(command_buffer) -> tuple[str, ...]:
    grant = _ACTIVE.get()
    if grant is None:
        return ()
    if _binding(command_buffer) != grant[0]:
        raise SourceOwnershipError("source-owned scratch boundary differs from original live selection")
    return grant[1]


def accepts_keyword(callable_object, name: str) -> bool:
    """Whether a renderer explicitly accepts ``name`` or a generic keyword set."""
    import inspect

    parameters = inspect.signature(callable_object).parameters
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def renderer_scratch_options(command_buffer, renderer, *, alternate_caller=False):
    """Forward only the exact live original-source selection to the neutral caller."""
    mutables = active_source_owned_mutables(command_buffer)
    if not mutables:
        return {}
    if alternate_caller:
        raise ValueError("original source scratch scope cannot mix alternate caller authority")
    if not accepts_keyword(renderer, "source_owned_mutables"):
        raise NotImplementedError("selected harness cannot consume source-owned scratch")
    return {"source_owned_mutables": mutables}
