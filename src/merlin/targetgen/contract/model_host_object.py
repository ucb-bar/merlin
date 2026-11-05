"""Check a compiled host object against one source-bound native model probe.

This stage checks source identity, outlined entry wiring and retained Linalg bodies,
initialization,
preprocessing result types, and actual object symbols. It does not qualify
host arithmetic, native accumulator reset, device wrappers, linking, or
execution; required-native regions retained as host bodies remain failures.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from xdsl.dialects import func
from xdsl.dialects.builtin import FloatAttr, TensorType
from xdsl.ir.affine.affine_expr import AffineDimExpr

from merlin.frontends.compile_inputs import load_compile_inputs
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.typed_preprocessing_correspondence import check_result_types
from merlin.targetgen.contract.model_host_bridge import diagnostic_host_bridge


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _llvm_ciface_pointer_parameters(llvm_ir: bytes, entry: str) -> list[str]:
    """Read one exact void C-interface definition from LLVM's line grammar."""
    symbol = "_mlir_ciface_" + entry
    wrappers = []
    for line in llvm_ir.decode().splitlines():
        if not line.startswith("define void @") or not line.endswith(") {"):
            continue
        head, separator, tail = line.partition("(")
        if separator and head.removeprefix("define void @") in {symbol, f'"{symbol}"'}:
            wrappers.append(tail[:-3])
    if len(wrappers) != 1:
        raise ValueError("LLVM C-interface wrapper has no unique definition")
    return [parameter.strip() for parameter in wrappers[0].split(",")]


def _symbols(text: str) -> dict[str, str]:
    """Read POSIX nm output; only external names prove a call binding."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2 or len(fields[1]) != 1:
            raise ValueError("nm produced a malformed symbol row")
        name, kind = fields[:2]
        if kind.islower():
            # LLVM may repeat temporary local labels such as .L0 across
            # sections. They cannot satisfy an external C-interface call.
            continue
        if name in result:
            raise ValueError(f"nm produced a duplicate symbol: {name}")
        result[name] = kind
    if not result:
        raise ValueError("nm produced no symbols")
    return result


def _check_call_symbols(bridge: dict, symbols: dict[str, str]) -> tuple[list[str], list[str]]:
    bindings = bridge.get("bindings")
    selected = bridge.get("external_symbols")
    if (
        not isinstance(bindings, list)
        or not isinstance(selected, list)
        or len(selected) != len(set(selected))
        or selected != bridge.get("c_interface_symbols")
    ):
        raise ValueError("host bridge has an invalid selected call roster")
    expected = [
        row["symbol"] for row in bindings
        if row.get("selected_route") in {"direct_native", "tiled_native"}
    ]
    if expected != selected or len({row["symbol"] for row in bindings}) != len(bindings):
        raise ValueError("host bridge calls differ from materializable native regions")
    entry = "_mlir_ciface_" + bridge["entry_c_interface_symbol"]
    if symbols.get(entry) not in {"T", "W"}:
        raise ValueError("compiled host object lacks a defined C-interface entry")
    expected_undefined = {"_mlir_ciface_" + symbol for symbol in selected}
    actual_undefined = {
        name for name, kind in symbols.items() if kind == "U" and name.startswith("_mlir_ciface_")
    }
    if actual_undefined != expected_undefined:
        raise ValueError("compiled host object has missing or unexpected device calls")
    if any(symbols.get(symbol) in {"T", "W"} or symbols.get("_mlir_ciface_" + symbol) in {"T", "W"}
           for symbol in selected):
        raise ValueError("selected native region was defined in the host object")
    unmet = [row for row in bindings if row["symbol"] not in selected]
    if [row["region_id"] for row in unmet] != bridge.get("required_native_unmet"):
        raise ValueError("host bridge lost a required-native failure")
    if any(symbols.get(row["symbol"]) not in {"T", "W"} for row in unmet):
        raise ValueError("compiled diagnostic host object lacks a retained region body")
    other_undefined = sorted(
        name for name, kind in symbols.items() if kind == "U" and name not in expected_undefined
    )
    return [row["region_id"] for row in unmet], other_undefined


def _check_object_receipt(lowered_dir: Path, ll_path: Path, object_path: Path) -> str:
    """Bind the checked object bytes to the LLVM IR used by the recorded codegen call."""
    receipts = list((lowered_dir / "invocations").glob("*/invocation.json"))
    matching = []
    for path in receipts:
        record = json.loads(path.read_text())
        if record.get("stage") != "object":
            continue
        if any(row.get("path") == str(object_path) for row in record.get("outputs", [])):
            matching.append((path, record))
    if len(matching) != 1:
        raise ValueError("host object needs one recorded codegen invocation")
    path, record = matching[0]
    if (record.get("schema") != "merlin.invocation_record.v1"
            or record.get("status") != "completed" or record.get("returncode") != 0
            or record.get("inputs_unchanged") is not True
            or record.get("executable_unchanged") is not True
            or record.get("inputs") != [{"path": str(ll_path), "sha256": _digest(ll_path.read_bytes())}]
            or record.get("outputs") != [{"path": str(object_path), "sha256": _digest(object_path.read_bytes())}]):
        raise ValueError("host object codegen receipt differs from LLVM IR or object bytes")
    argv = record.get("argv")
    if not isinstance(argv, list) or argv[-4:] != ["-c", str(ll_path), "-o", str(object_path)]:
        raise ValueError("host object codegen receipt has an unexpected compile command")
    return _digest(path.read_bytes())


def _descriptor_positions(
    source_inputs: tuple[str, ...], source_outputs: tuple[str, ...],
    lowered_inputs: tuple[str, ...],
) -> dict:
    """Check the selected buffer-results pass's appended out-parameter ABI."""
    def memref(value: str) -> str:
        if not value.startswith("tensor<") or not value.endswith(">"):
            raise ValueError("model entry has an unsupported non-tensor ABI type")
        return "memref<" + value[len("tensor<"):]

    expected = tuple(memref(value) for value in (*source_inputs, *source_outputs))
    if not source_outputs or lowered_inputs != expected:
        raise ValueError("buffered model entry does not append its result descriptors")
    return {
        "source_input_count": len(source_inputs),
        "source_result_count": len(source_outputs),
        "input_descriptor_positions": list(range(len(source_inputs))),
        "result_descriptor_positions": list(range(len(source_inputs), len(expected))),
        "descriptor_types": list(expected),
    }


def _positive_zero_fill(value) -> bool:
    fill = value.owner
    kind = getattr(fill, "name", None)
    if kind not in {"linalg.fill", "tensor.splat"} or len(fill.operands) != (2 if kind == "linalg.fill" else 1):
        return False
    constant = fill.operands[0].owner
    literal = getattr(constant, "properties", {}).get("value")
    if kind == "tensor.splat":
        storage_valid = (
            isinstance(value.type, TensorType)
            and all(dim >= 0 for dim in value.type.get_shape())
        )
    else:
        storage_valid = (
            getattr(fill.operands[1].owner, "name", None) == "tensor.empty"
            and not fill.operands[1].owner.operands
            and isinstance(fill.operands[1].type, TensorType)
            and all(dim >= 0 for dim in fill.operands[1].type.get_shape())
        )
    return (
        storage_valid
        and getattr(constant, "name", None) == "arith.constant"
        and isinstance(literal, FloatAttr)
        and literal.value.data == 0.0
        and math.copysign(1.0, literal.value.data) == 1.0
    )


def _overwrites_empty_output(operation, value) -> bool:
    empty = value.owner
    if getattr(empty, "name", None) != "tensor.empty" or empty.operands:
        return False
    if not isinstance(value.type, TensorType) or any(dim < 0 for dim in value.type.get_shape()):
        return False
    if operation.name == "linalg.transpose":
        permutation = tuple(operation.properties["permutation"].get_values())
        source_shape = operation.inputs[0].type.get_shape()
        return (
            len(permutation) == len(source_shape)
            and sorted(permutation) == list(range(len(source_shape)))
            and tuple(source_shape[index] for index in permutation) == value.type.get_shape()
        )
    if operation.name == "linalg.generic" and len(operation.outputs) == 1:
        iterators = operation.properties["iterator_types"].data
        mapping = operation.properties["indexing_maps"].data[-1].data
        positions = tuple(
            expression.position for expression in mapping.results
            if isinstance(expression, AffineDimExpr)
        )
        return (
            len(operation.regions) == 1
            and not any(operation.regions[0].block.args[-1].uses)
            and all(iterator.data.value == "parallel" for iterator in iterators)
            and mapping.num_dims == len(iterators) == len(value.type.get_shape())
            and len(positions) == len(mapping.results)
            and sorted(positions) == list(range(mapping.num_dims))
        )
    return False


def _check_omitted_initializer(original, definition, selected: bool) -> str:
    """Check the only currently supported ways to drop an outlined tensor out operand.

    A selected native matmul still needs its own reset/accumulation check. This
    check establishes only that the source requested a positive-zero start.
    """
    outputs = tuple(original.outputs)
    if len(outputs) != 1 or len(original.results) != 1:
        raise ValueError("outlined call drops an unsupported output initializer")
    source_init = outputs[0]
    if original.name == "linalg.matmul":
        if not _positive_zero_fill(source_init):
            raise ValueError("outlined matmul drops a nonzero or unknown initializer")
        kind = "positive_zero_matmul"
    elif (original.name == "linalg.generic"
          and isinstance(original.results[0].type, TensorType)
          and len(original.results[0].type.get_shape()) >= 3):
        from merlin.semantic_compiler.linalg_bridge import _source_f32_batched_matmul
        from merlin.semantic_compiler.precision_intake import _check_positive_zero_init

        _, _, checked_init, _, _ = _source_f32_batched_matmul(original)
        if checked_init is not source_init or not _positive_zero_fill(source_init):
            raise ValueError("outlined batched matmul drops a nonzero or unknown initializer")
        _check_positive_zero_init(source_init, rank=len(original.results[0].type.get_shape()))
        kind = "positive_zero_matmul"
    elif getattr(source_init.owner, "name", None) == "tensor.splat" and not selected:
        kind = "retained_splat_initializer"
    elif _overwrites_empty_output(original, source_init):
        kind = "full_overwrite_empty"
    else:
        raise ValueError("outlined call drops an unproved output initializer")
    if selected:
        return kind
    matching = [operation for operation in definition.body.block.ops
                if operation.name == original.name]
    if len(matching) != 1 or len(matching[0].outputs) != 1:
        raise ValueError("outlined host body has no matching initialized operation")
    host_init = matching[0].outputs[0]
    if host_init.type != source_init.type:
        raise ValueError("outlined host body changed initializer type")
    if kind == "positive_zero_matmul":
        if not _positive_zero_fill(host_init):
            raise ValueError("outlined host body lost positive-zero initialization")
    elif kind == "retained_splat_initializer":
        if getattr(host_init.owner, "name", None) != "tensor.splat":
            raise ValueError("outlined host body lost splat initialization")
    elif not _overwrites_empty_output(matching[0], host_init):
        raise ValueError("outlined host body lost full-overwrite initialization")
    return kind


def _distinct_ssa_values(values: tuple) -> tuple:
    """Match the outliner's one argument per free SSA value, in first-use order."""
    ordered = []
    seen: set[int] = set()
    for value in values:
        identity = id(value)
        if identity not in seen:
            ordered.append(value)
            seen.add(identity)
    return tuple(ordered)


def _outlined_source_operands(original) -> tuple:
    """Reconstruct the source values passed to a single-op dispatch.

    Linalg bodies may read values captured from the enclosing function. The
    operation's explicit inputs alone do not describe those dependencies.
    Initializer tensors are handled separately below and cannot become an
    implicit host argument for a selected device call.
    """
    source_inputs = (
        tuple(original.operands)
        if original.name == "linalg.reduce"
        else tuple(getattr(original, "inputs", original.operands))
    )
    internal = {id(result) for sub in original.walk() for result in sub.results}
    for sub in original.walk():
        for region in sub.regions:
            for block in region.blocks:
                internal.update(id(argument) for argument in block.args)
    captured = []
    for sub in original.walk():
        if sub is original:
            continue
        for operand in sub.operands:
            if id(operand) not in internal:
                captured.append(operand)
    initializer_inputs = ()
    if original.name != "linalg.reduce" and len(original.outputs) == 1:
        initializer = original.outputs[0].owner
        if getattr(initializer, "name", None) == "tensor.splat":
            seed = initializer.operands[0]
            if getattr(seed.owner, "name", None) != "arith.constant":
                initializer_inputs = (seed,)
    return _distinct_ssa_values((*source_inputs, *initializer_inputs, *captured))


def _region_matches_source(source_region, host_region, outer_values: dict[int, object]) -> bool:
    """Compare a cloned Linalg body with captured values mapped to kernel args."""
    if len(source_region.blocks) != 1 or len(host_region.blocks) != 1:
        return False
    for source_block, host_block in zip(source_region.blocks, host_region.blocks, strict=True):
        source_ops, host_ops = tuple(source_block.ops), tuple(host_block.ops)
        if (len(source_block.args) != len(host_block.args)
                or len(source_ops) != len(host_ops)
                or tuple(value.type for value in source_block.args)
                != tuple(value.type for value in host_block.args)):
            return False
        values = dict(outer_values)
        values.update((id(source), host) for source, host in zip(
            source_block.args, host_block.args, strict=True,
        ))
        for source_op, host_op in zip(source_ops, host_ops, strict=True):
            if (source_op.name != host_op.name
                    or source_op.properties != host_op.properties
                    or source_op.attributes != host_op.attributes
                    or tuple(value.type for value in source_op.results)
                    != tuple(value.type for value in host_op.results)
                    or len(source_op.operands) != len(host_op.operands)
                    or any(values.get(id(source)) is not host for source, host in zip(
                        source_op.operands, host_op.operands, strict=True,
                    ))
                    or len(source_op.regions) != len(host_op.regions)):
                return False
            values.update((id(source), host) for source, host in zip(
                source_op.results, host_op.results, strict=True,
            ))
            if any(not _region_matches_source(source, host, values)
                   for source, host in zip(source_op.regions, host_op.regions, strict=True)):
                return False
    return True


def _check_host_linalg_body(original, definition, expected_operands: tuple,
                            initializer_kind: str | None) -> None:
    """Match a retained outlined computation to its parsed source operation."""
    if len(definition.body.blocks) != 1:
        raise ValueError("outlined host body has unsupported control flow")
    block = definition.body.block
    if tuple(value.type for value in block.args) != tuple(
        value.type for value in expected_operands
    ):
        raise ValueError("outlined host body arguments differ from source operands")

    source_prefix = ()
    if initializer_kind in {"positive_zero_matmul", "retained_splat_initializer"}:
        initializer = original.outputs[0].owner
        if initializer.name == "linalg.fill":
            source_prefix = (
                initializer.operands[0].owner,
                initializer.operands[1].owner,
                initializer,
            )
        elif getattr(initializer.operands[0].owner, "name", None) == "arith.constant":
            source_prefix = (initializer.operands[0].owner, initializer)
        else:
            source_prefix = (initializer,)
    elif initializer_kind == "full_overwrite_empty":
        source_prefix = (original.outputs[0].owner,)
    operations = tuple(block.ops)
    if len(operations) != len(source_prefix) + 2:
        raise ValueError("outlined host body changed its operation sequence")
    for source_op, host_op in zip(source_prefix, operations[:len(source_prefix)], strict=True):
        if (source_op.name != host_op.name
                or source_op.properties != host_op.properties
                or source_op.attributes != host_op.attributes
                or tuple(value.type for value in source_op.results)
                != tuple(value.type for value in host_op.results)):
            raise ValueError("outlined host body changed its initializer")
    if initializer_kind in {"positive_zero_matmul", "retained_splat_initializer"}:
        if len(source_prefix) == 3:
            constant, empty, fill = operations[:3]
            if (constant.operands or empty.operands
                    or tuple(fill.operands) != (constant.results[0], empty.results[0])):
                raise ValueError("outlined host body changed its initializer wiring")
        elif len(source_prefix) == 2:
            constant, splat = operations[:2]
            if constant.operands or tuple(splat.operands) != (constant.results[0],):
                raise ValueError("outlined host body changed its initializer wiring")
        else:
            (splat,) = operations[:1]
            seed = original.outputs[0].owner.operands[0]
            if tuple(splat.operands) != (block.args[expected_operands.index(seed)],):
                raise ValueError("outlined host body changed its initializer wiring")
    host_op, returned = operations[-2:]
    host_args = {id(source): argument for source, argument in zip(
        expected_operands, block.args, strict=True
    )}
    if (host_op.name != original.name
            or host_op.properties != original.properties
            or host_op.attributes != original.attributes
            or tuple(value.type for value in host_op.results)
            != tuple(value.type for value in original.results)
            or len(host_op.regions) != len(original.regions)
            or any(not _region_matches_source(source_region, host_region, host_args)
                   for source_region, host_region in zip(
                       original.regions, host_op.regions, strict=True
                   ))):
        raise ValueError("outlined host body changed its Linalg computation")
    source_operands = _outlined_source_operands(original)
    if source_operands != expected_operands:
        raise ValueError("outlined host body arguments differ from source operands")
    explicit_operands = (
        tuple(original.operands)
        if original.name == "linalg.reduce"
        else tuple(getattr(original, "inputs", original.operands))
    )
    expected_host_operands = tuple(host_args[id(value)] for value in explicit_operands)
    if source_prefix:
        expected_host_operands += (operations[-3].results[0],)
    if tuple(host_op.operands) != expected_host_operands:
        raise ValueError("outlined host body changed its Linalg operands")
    if returned.name != "func.return" or tuple(returned.operands) != tuple(host_op.results):
        raise ValueError("outlined host body changed its returned results")


def _reduction_initializer_storage(original) -> list[str]:
    """Record whether a host reduction may consume private initializer storage.

    Bufferization can reuse the initializer as its output. This source check
    does not assert that the compiled body actually did so, or prove its
    numerical result. An external, shared, or otherwise unknown initializer
    needs an independent alias/effect check before model execution is admitted.
    """
    if original.name != "linalg.reduce" or len(original.operands) != 2 * len(original.results):
        raise ValueError("host reduction has an unsupported initializer signature")
    effects = []
    for initializer in original.operands[len(original.results):]:
        owner = initializer.owner
        fresh_splat = (
            getattr(owner, "name", None) == "tensor.splat"
            and sum(1 for _ in initializer.uses) == 1
        )
        effects.append("fresh_single_use_splat" if fresh_splat else "requires_storage_check")
    return effects


def _check_entry_dataflow(source_text: str, bridge_text: str, entry: str,
                          selected_symbols: list[str]) -> dict:
    """Check source-to-host SSA wiring and retained parsed Linalg bodies."""
    source = parse_mlir_text(source_text)
    bridge = parse_mlir_text(bridge_text)

    def functions(module) -> dict:
        result = {}
        for operation in module.body.block.ops:
            if isinstance(operation, func.FuncOp):
                name = operation.sym_name.data
                if name in result:
                    raise ValueError("outlined host module has duplicate function symbols")
                result[name] = operation
        return result

    source_functions, bridge_functions = functions(source), functions(bridge)
    if entry not in source_functions or entry not in bridge_functions:
        raise ValueError("outlined host entry is missing")
    before, after = source_functions[entry], bridge_functions[entry]
    if (before.function_type != after.function_type
            or len(before.body.blocks) != 1 or len(after.body.blocks) != 1):
        raise ValueError("outlined host entry type or control flow differs")
    source_block, bridge_block = before.body.block, after.body.block
    if (tuple(value.type for value in source_block.args)
            != tuple(value.type for value in bridge_block.args)
            or tuple(value.type for value in source_block.args)
            != tuple(before.function_type.inputs.data)):
        raise ValueError("outlined host entry block arguments differ")
    source_ops, bridge_ops = tuple(source_block.ops), tuple(bridge_block.ops)
    if len(source_ops) != len(bridge_ops):
        raise ValueError("outlined host changed the top-level operation count")

    def positions(block, operations) -> dict:
        result = {value: ("argument", index) for index, value in enumerate(block.args)}
        for ordinal, operation in enumerate(operations):
            for index, value in enumerate(operation.results):
                result[value] = ("result", ordinal, index)
        return result

    source_values = positions(source_block, source_ops)
    bridge_values = positions(bridge_block, bridge_ops)
    selected = set(selected_symbols)
    if len(selected) != len(selected_symbols):
        raise ValueError("outlined host has duplicate selected calls")
    seen_selected: set[str] = set()
    native_calls = host_calls = edges = omitted_initializers = 0
    host_initializers = native_zero_initializers = checked_host_bodies = 0
    operation_routes = []
    host_reduction_initializers = []
    for ordinal, (original, lowered) in enumerate(zip(source_ops, bridge_ops, strict=True)):
        initializer_storage = None
        if tuple(str(value.type) for value in original.results) != tuple(
            str(value.type) for value in lowered.results
        ):
            raise ValueError(f"outlined host changed result types at operation {ordinal}")
        if original.name.startswith("linalg.") and isinstance(lowered, func.CallOp):
            callee = lowered.callee
            if callee.nested_references.data:
                raise ValueError("outlined host call has a nested symbol reference")
            symbol = callee.root_reference.data
            definition = bridge_functions.get(symbol)
            if definition is None or (
                tuple(definition.function_type.inputs.data)
                != tuple(value.type for value in lowered.operands)
            ) or (
                tuple(definition.function_type.outputs.data)
                != tuple(value.type for value in lowered.results)
            ):
                raise ValueError("outlined host call has no matching typed function")
            if symbol in selected:
                if definition.body.blocks or symbol in seen_selected:
                    raise ValueError("selected device call was defined or duplicated in the host")
                seen_selected.add(symbol)
                native_calls += 1
                route = "native_external_call"
            else:
                if not definition.body.blocks:
                    raise ValueError("unselected host call has no compiled body")
                host_calls += 1
                route = "retained_host_call"
            # Tensor-output destinations are omitted by the outlined call.
            # Their initialization semantics remain a separate obligation.
            explicit_operands = (
                tuple(original.operands)
                if original.name == "linalg.reduce"
                else tuple(getattr(original, "inputs", original.operands))
            )
            expected_operands = _outlined_source_operands(original)
            omitted = len(original.operands) - len(explicit_operands)
            initializer_kind = None
            if omitted:
                initializer_kind = _check_omitted_initializer(
                    original, definition, symbol in selected
                )
                omitted_initializers += omitted
                if symbol in selected:
                    native_zero_initializers += 1
                else:
                    host_initializers += 1
            if symbol not in selected:
                _check_host_linalg_body(original, definition, expected_operands,
                                        initializer_kind)
                checked_host_bodies += 1
                if original.name == "linalg.reduce":
                    initializer_storage = _reduction_initializer_storage(original)
                    host_reduction_initializers.extend(initializer_storage)
        else:
            if original.name != lowered.name:
                raise ValueError(f"outlined host changed operation {ordinal} without a call")
            if original.properties != lowered.properties or original.attributes != lowered.attributes:
                raise ValueError(f"outlined host changed unchanged-operation attributes at {ordinal}")
            expected_operands = tuple(original.operands)
            route = "retained_entry_operation"
            symbol = None
        try:
            original_edges = tuple(source_values[value] for value in expected_operands)
            lowered_edges = tuple(bridge_values[value] for value in lowered.operands)
        except KeyError as exc:
            raise ValueError("outlined host operand escapes the entry block") from exc
        if original_edges != lowered_edges:
            raise ValueError(f"outlined host changed SSA operands at operation {ordinal}")
        edges += len(original_edges)
        operation_route = {
            "operation_ordinal": ordinal,
            "mlir_operation": original.name,
            "bridge_route": route,
            "callee": symbol,
        }
        if initializer_storage is not None:
            operation_route["host_reduction_initializer_storage"] = initializer_storage
        operation_routes.append(operation_route)
    if seen_selected != selected:
        raise ValueError("outlined host lost a selected device call")
    return {
        "schema": "merlin.host_entry_dataflow_check.v2",
        "status": "source_to_host_body_checked",
        "operation_count": len(source_ops),
        "checked_operand_edges": edges,
        "native_external_call_count": native_calls,
        "host_defined_call_count": host_calls,
        "host_linalg_body_correspondence_checked_count": checked_host_bodies,
        "omitted_initializer_operand_count": omitted_initializers,
        "host_initializer_check_count": host_initializers,
        "native_zero_initialization_obligation_count": native_zero_initializers,
        "initialization_semantics": "source_drop_and_host_initializer_checked",
        "native_reset_semantics": "not_checked",
        "host_numerics": "not_checked",
        "host_reduction_initializer_storage": {
            "fresh_single_use_splat": host_reduction_initializers.count("fresh_single_use_splat"),
            "requires_storage_check": host_reduction_initializers.count("requires_storage_check"),
        },
        "operation_routes": operation_routes,
    }


def _source_route_accounting(probe: dict, bridge: dict, entry_dataflow: dict) -> dict:
    """Join the native probe to checked outlined calls without granting execution."""
    source_ops = probe.get("source_operations")
    routes = entry_dataflow.get("operation_routes")
    bindings = bridge.get("bindings")
    if (not isinstance(source_ops, list) or not isinstance(routes, list)
            or not isinstance(bindings, list)
            or len(source_ops) != len(routes) or len(source_ops) != probe.get("source_operation_count")):
        raise ValueError("source route inventory differs from the checked host entry")
    by_operation = {row["operation_id"]: row for row in bindings}
    if len(by_operation) != len(bindings):
        raise ValueError("source route inventory has duplicate native operation IDs")
    selected = set(bridge["external_symbols"])
    unmet = set(bridge["required_native_unmet"])
    seen_bindings: set[str] = set()
    states = []
    for ordinal, (source, route) in enumerate(zip(source_ops, routes, strict=True)):
        if (source.get("operation_ordinal") != ordinal
                or route.get("operation_ordinal") != ordinal
                or source.get("mlir_operation") != route.get("mlir_operation")):
            raise ValueError("source route inventory changed operation order or names")
        kind = route.get("bridge_route")
        region_id = source.get("region_id")
        if region_id is not None:
            binding = by_operation.get(source.get("operation_id"))
            if (binding is None or binding.get("region_id") != region_id
                    or binding.get("symbol") != route.get("callee")
                    or source.get("route") != "native_region_candidate"):
                raise ValueError("source route inventory changed a native region binding")
            seen_bindings.add(source["operation_id"])
            if binding["symbol"] in selected and kind == "native_external_call":
                state = "native_external_unexecuted"
            elif region_id in unmet and kind == "retained_host_call":
                state = "required_native_retained_host"
            else:
                raise ValueError("source route inventory silently changed native placement")
        elif (kind == "retained_host_call" and source.get("route") == "unrouted"
              and route.get("callee") is not None):
            state = "host_body_numerics_unqualified"
        elif kind == "retained_entry_operation" and route.get("callee") is None:
            if source.get("route") == "model_output" and source["mlir_operation"] == "func.return":
                state = "model_output_unexecuted"
            elif source.get("route") == "unrouted" and source["mlir_operation"] != "func.return":
                state = "entry_support_numerics_unqualified"
            else:
                raise ValueError("source route inventory has an unsupported entry operation")
        else:
            raise ValueError("source route inventory has an unsupported host route")
        states.append({"operation_ordinal": ordinal, "operation_id": source["operation_id"],
                       "state": state})
    if seen_bindings != set(by_operation):
        raise ValueError("source route inventory omitted a native region binding")
    counts = {state: sum(row["state"] == state for row in states)
              for state in sorted({row["state"] for row in states})}
    return {
        "schema": "merlin.model_source_route_accounting.v1",
        "scope": "parsed source-to-outlined boundary; no numerical or execution qualification",
        "operation_count": len(states),
        "state_counts": counts,
        "operations": states,
    }


def _check_entry_abi(
    source_text: str, entry: str, audit_dir: Path, audit_index: Path,
    llvm_ir: bytes, mlir_opt: Path,
) -> dict:
    source = parse_mlir_text(source_text)
    source_entries = [operation for operation in source.body.block.ops
                      if isinstance(operation, func.FuncOp) and operation.sym_name.data == entry]
    if len(source_entries) != 1:
        raise ValueError("model entry has no unique parsed source signature")
    source_type = source_entries[0].function_type
    index = json.loads(audit_index.read_text())
    files = index.get("pass_inspection", {}).get("files", [])
    candidates = [row for row in files if isinstance(row, dict)
                  and row.get("file", "").endswith("_buffer-results-to-out-params.mlir")]
    after = []
    for row in candidates:
        relative = row["file"]
        path = audit_dir / relative
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(audit_dir.resolve())
                or _digest(path.read_bytes()) != row.get("sha256")):
            raise ValueError("buffer-results ABI pass audit changed")
        if path.read_bytes().startswith(b"// -----// IR Dump After BufferResultsToOutParamsPass:"):
            after.append((path, row["sha256"]))
    if len(after) != 1:
        raise ValueError("model entry needs one audited buffer-results ABI pass")
    mlir_opt = mlir_opt.resolve(strict=True)
    if not mlir_opt.is_file() or not os.access(mlir_opt, os.X_OK):
        raise ValueError("selected mlir-opt is unavailable")
    process = subprocess.run(
        [str(mlir_opt), "--allow-unregistered-dialect", "--mlir-print-op-generic", str(after[0][0])],
        capture_output=True, text=True, check=False, timeout=30,
    )
    if process.returncode:
        raise ValueError("selected mlir-opt could not parse audited buffer-results IR")
    lowered = parse_mlir_text(process.stdout)
    lowered_entries = [operation for operation in lowered.body.block.ops
                       if isinstance(operation, func.FuncOp) and operation.sym_name.data == entry]
    if len(lowered_entries) != 1 or lowered_entries[0].function_type.outputs.data:
        raise ValueError("buffered model entry has no unique void signature")
    positions = _descriptor_positions(
        tuple(str(value) for value in source_type.inputs.data),
        tuple(str(value) for value in source_type.outputs.data),
        tuple(str(value) for value in lowered_entries[0].function_type.inputs.data),
    )
    parameters = _llvm_ciface_pointer_parameters(llvm_ir, entry)
    if len(parameters) != len(positions["descriptor_types"]):
        raise ValueError("LLVM C-interface wrapper has the wrong descriptor arity")
    if any(not parameter.startswith("ptr ") for parameter in parameters):
        raise ValueError("LLVM C-interface wrapper has a non-pointer argument")
    return {
        **positions,
        "scope": "audited buffer-results signature and LLVM C-interface pointer arity",
        "buffer_results_pass_sha256": after[0][1],
        "mlir_opt_sha256": _digest(mlir_opt.read_bytes()),
    }


def check_compiled_host_object(
    *,
    probe_dir: Path,
    compile_inputs: Path,
    lowered_dir: Path,
    object_path: Path,
    nm: Path,
    mlir_opt: Path,
) -> dict:
    """Reopen exact source and lowering bytes, then inspect the object roster."""
    probe = json.loads((probe_dir / "manifest.json").read_text())
    if (probe.get("schema") != "merlin.native_model_region_probe.v1"
            or probe.get("engine") != "merlin_native" or probe.get("status") != "incomplete"):
        raise ValueError("host object needs an incomplete native model probe")
    staged = load_compile_inputs(compile_inputs)
    if (staged.original_source_sha256 != probe.get("original_source_sha256")
            or staged.program_sha256 != probe.get("staged_program_sha256")
            or staged.precision_policy_sha256 != probe.get("precision_policy_sha256")):
        raise ValueError("host object probe differs from staged model source or precision policy")
    bridge = probe.get("host_bridge")
    if not isinstance(bridge, dict) or bridge.get("path") != "host-bridge.mlir":
        raise ValueError("native model probe has no checked host bridge")
    bridge_bytes = (probe_dir / "host-bridge.mlir").read_bytes()
    if _digest(bridge_bytes) != bridge.get("sha256"):
        raise ValueError("host bridge bytes differ from native model probe")
    expected_bridge, expected_text = diagnostic_host_bridge(
        staged.program, staged.entry, probe["regions"], probe.get("named_tiles", []),
        batched_regions=probe.get("batched_contractions", []),
        batched_tiles=probe.get("batched_tiles", []),
    )
    if (bridge_bytes != expected_text.encode()
            or {key: value for key, value in bridge.items() if key not in {"path", "sha256"}} != expected_bridge):
        raise ValueError("host bridge no longer matches the exact native region decisions")
    entry_dataflow = _check_entry_dataflow(
        staged.program, bridge_bytes.decode(), staged.entry, bridge["external_symbols"],
    )
    route_accounting = _source_route_accounting(probe, bridge, entry_dataflow)

    lowered_dir = lowered_dir.resolve(strict=True)
    object_path = object_path.resolve(strict=True)
    if not object_path.is_relative_to(lowered_dir) or not object_path.is_file():
        raise ValueError("compiled host object must belong to the selected lowering")
    object_bytes = object_path.read_bytes()
    if (len(object_bytes) < 20 or object_bytes[:4] != b"\x7fELF"
            or object_bytes[4] != 2 or object_bytes[5] != 1
            or int.from_bytes(object_bytes[16:18], "little") != 1
            or int.from_bytes(object_bytes[18:20], "little") != 243):
        raise ValueError("compiled host object is not a RISC-V ELF64 relocatable object")
    source_mlir = (lowered_dir / "model.upstream.mlir").read_bytes()
    audit_indices = list(lowered_dir.glob("ir-audit-*/index.json"))
    if len(audit_indices) != 1:
        raise ValueError("host lowering needs one complete source correspondence audit")
    correspondence = check_result_types(bridge_bytes, source_mlir, audit_indices[0])
    ll_path = lowered_dir / "model.ll"
    ll_bytes = ll_path.read_bytes()
    if not ll_bytes:
        raise ValueError("host lowering has no LLVM IR")
    entry_abi = _check_entry_abi(
        staged.program, bridge["entry"], audit_indices[0].parent, audit_indices[0],
        ll_bytes, mlir_opt,
    )
    object_receipt_sha256 = _check_object_receipt(lowered_dir, ll_path, object_path)
    nm = nm.resolve(strict=True)
    if not nm.is_file() or not os.access(nm, os.X_OK):
        raise ValueError("selected nm tool is unavailable")
    process = subprocess.run([str(nm), "-P", str(object_path)], capture_output=True, text=True, timeout=15)
    if process.returncode != 0:
        raise ValueError("selected nm tool could not inspect the compiled host object")
    symbols = _symbols(process.stdout)
    unmet, runtime_symbols = _check_call_symbols(bridge, symbols)
    return {
        "schema": "merlin.native_host_object_check.v1",
        "status": "object_symbols_checked",
        "scope": "exact source, preprocessing result types, and ELF object symbols only",
        "source_sha256": staged.program_sha256,
        "probe_manifest_sha256": _digest((probe_dir / "manifest.json").read_bytes()),
        "host_bridge_sha256": _digest(bridge_bytes),
        "preprocessed_sha256": _digest(source_mlir),
        "llvm_ir_sha256": _digest(ll_bytes),
        "host_object_sha256": _digest(object_bytes),
        "host_object_elf_machine": 243,
        "object_codegen_receipt_sha256": object_receipt_sha256,
        "nm_sha256": _digest(nm.read_bytes()),
        "selected_device_call_count": len(bridge["external_symbols"]),
        "selected_device_symbols": bridge["external_symbols"],
        "required_native_unmet": unmet,
        "other_unresolved_symbols": runtime_symbols,
        "unrouted_source_operation_count": probe["unrouted_operation_count"],
        "typed_preprocessing": correspondence,
        "entry_dataflow": entry_dataflow,
        "source_route_accounting": route_accounting,
        "entry_abi": entry_abi,
        "host_operation_numerics_qualified": False,
        "device_wrappers_linked": False,
        "executed": False,
    }
