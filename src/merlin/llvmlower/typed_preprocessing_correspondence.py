"""Check result types through the audited, top-level xDSL preprocessing map.

This checks a structural type invariant only. A result-index map and equal MLIR
types cannot prove equal values, later LLVM lowering, or operation coverage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _top_level_operations(raw: bytes):
    _, functions = _function_inventory(raw)
    if len(functions) != 1:
        raise ValueError("typed correspondence requires one defined, single-block function")
    return next(iter(functions.items()))


def _function_inventory(raw: bytes):
    from ..frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(raw.decode("utf-8"))
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    names = [op.sym_name.data for op in functions]
    if len(names) != len(set(names)):
        raise ValueError("typed correspondence has duplicate function symbols")
    signatures = [
        {
            "symbol": op.sym_name.data,
            "defined": bool(op.body.blocks),
            "inputs": [str(value) for value in op.function_type.inputs.data],
            "outputs": [str(value) for value in op.function_type.outputs.data],
        }
        for op in functions
    ]
    defined = {}
    for op in functions:
        if not op.body.blocks:
            continue
        if len(op.body.blocks) != 1:
            raise ValueError("typed correspondence requires single-block functions")
        defined[op.sym_name.data] = [item for item in op.body.block.ops if item.name != "func.return"]
    if not defined:
        raise ValueError("typed correspondence requires a defined function")
    return signatures, defined


def _operation_name(operation) -> str:
    return operation.op_name.data if operation.name == "builtin.unregistered" else operation.name


def _may_elide(operation) -> bool:
    name = _operation_name(operation)
    return bool(operation.results) and name.startswith(
        ("arith.", "linalg.", "math.", "tensor.", "quant_ext.")
    )


def _check_operation_rows(source_ops, processed_ops, receipt: dict) -> tuple[int, int, int]:
    rows = receipt.get("operations")
    if (
        not isinstance(rows, list)
        or receipt.get("source_op_count") != len(source_ops)
        or receipt.get("preprocessed_op_count") != len(processed_ops)
        or len(rows) != len(source_ops)
    ):
        raise ValueError("source-transform map operation denominator differs from exact IR")

    emitted_indices = []
    checked_results = 0
    elided_results = 0
    elided_operations = 0
    source_index_by_id = {id(op): index for index, op in enumerate(source_ops)}
    for source_index, (source, row) in enumerate(zip(source_ops, rows)):
        if not isinstance(row, dict) or row.get("source_op_index") != source_index:
            raise ValueError("source-transform map source indices are incomplete or reordered")
        if row.get("source_op_name") != _operation_name(source):
            raise ValueError("source-transform map operation name differs from exact IR")
        owned = row.get("preprocessed_op_indices")
        if not isinstance(owned, list) or any(type(index) is not int for index in owned):
            raise ValueError("source-transform map has invalid preprocessed indices")
        emitted_indices.extend(owned)
        mappings = row.get("result_map")
        if not isinstance(mappings, list):
            raise ValueError("source-transform map has incomplete result correspondence")
        if not owned and not source.results:
            raise ValueError("source-transform map elided an effectful or resultless operation")
        if source.results and not mappings:
            if owned:
                raise ValueError("elided source result still owns emitted operations")
            if not _may_elide(source):
                raise ValueError("source-transform map elided an operation outside the pure pruning set")
            for result in source.results:
                for use in result.uses:
                    user = source_index_by_id.get(id(use.operation))
                    if (
                        user is None
                        or not isinstance(rows[user], dict)
                        or rows[user].get("preprocessed_op_indices") != []
                    ):
                        raise ValueError("elided source result still has a live source user")
            elided_results += len(source.results)
            elided_operations += 1
            continue
        if len(mappings) != len(source.results):
            raise ValueError("source-transform map has incomplete result correspondence")
        if any(not isinstance(mapping, dict) for mapping in mappings):
            raise ValueError("source-transform map has malformed result correspondence")
        if [mapping.get("source_result_index") for mapping in mappings] != list(range(len(source.results))):
            raise ValueError("source-transform map result indices are incomplete or reordered")
        for source_result, mapping in zip(source.results, mappings):
            target_index = mapping.get("preprocessed_op_index")
            result_index = mapping.get("preprocessed_result_index")
            if (
                type(target_index) is not int
                or target_index not in owned
                or not 0 <= target_index < len(processed_ops)
            ):
                raise ValueError("mapped result is not owned by its source operation")
            target_results = processed_ops[target_index].results
            if type(result_index) is not int or not 0 <= result_index < len(target_results):
                raise ValueError("mapped result index is absent from exact preprocessed IR")
            if source_result.type != target_results[result_index].type:
                raise ValueError("mapped result type or shape differs across preprocessing")
            checked_results += 1
    if sorted(emitted_indices) != list(range(len(processed_ops))):
        raise ValueError("source-transform map does not own every preprocessed operation exactly once")
    return checked_results, elided_results, elided_operations


def check_result_types(source_mlir: bytes, preprocessed_mlir: bytes, audit_index: Path) -> dict:
    """Reparse both exact IR byte streams and check mapped result type/shape equality.

    The audit may retain only compact inspection views. Callers therefore supply
    the exact source and preprocessed bytes; their digests must match the audit's
    hash-bound source-transform map and terminal xDSL preprocessing stage.
    Nothing here grants Phase 0/1 admission or certifies operation semantics.
    """

    audit_index = Path(audit_index)
    if audit_index.is_symlink() or not audit_index.is_file():
        raise ValueError("audit index is absent or indirect")
    index = json.loads(audit_index.read_bytes())
    if index.get("outcome") != "completed":
        raise ValueError("audit did not complete")
    selected = index.get("source_transform_map") or {}
    receipts = [
        item
        for item in index.get("accounting_receipts") or []
        if item.get("name") == "source-transform-map"
    ]
    if len(receipts) != 1:
        raise ValueError("audit has no unique source-transform map receipt")
    descriptor = receipts[0]
    if (
        selected.get("status") != "recorded"
        or selected.get("receipt") != "source-transform-map.json"
        or descriptor.get("file") != "source-transform-map.json"
        or descriptor.get("schema") not in {"source_transform_map_v1", "source_transform_map_v2"}
        or descriptor.get("representation") != "accounting-only"
        or descriptor.get("executable") is not False
        or selected.get("sha256") != descriptor.get("sha256")
    ):
        raise ValueError("source-transform map is not the selected accounting receipt")
    receipt_path = audit_index.parent / "source-transform-map.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("source-transform map receipt is absent or indirect")
    receipt_bytes = receipt_path.read_bytes()
    if _digest(receipt_bytes) != descriptor["sha256"]:
        raise ValueError("source-transform map receipt digest differs from the audit")
    receipt = json.loads(receipt_bytes)
    if receipt.get("schema") != descriptor["schema"]:
        raise ValueError("unsupported source-transform map schema")
    if receipt.get("source_sha256") != _digest(source_mlir):
        raise ValueError("source digest differs from the audited map")
    if receipt.get("preprocessed_sha256") != _digest(preprocessed_mlir):
        raise ValueError("preprocessed digest differs from the audited map")
    stages = [stage for stage in index.get("stages") or [] if stage.get("name") == "xdsl-c-interface"]
    if len(stages) != 1 or stages[0].get("sha256") != _digest(preprocessed_mlir):
        raise ValueError("preprocessed bytes differ from the audited terminal xDSL stage")

    if receipt["schema"] == "source_transform_map_v1":
        source_entry, source_ops = _top_level_operations(source_mlir)
        processed_entry, processed_ops = _top_level_operations(preprocessed_mlir)
        if source_entry != processed_entry or receipt.get("entry") != source_entry:
            raise ValueError("entry identity differs across the audited preprocessing boundary")
        source_count = len(source_ops)
        checked_results, elided_results, elided_operations = _check_operation_rows(
            source_ops, processed_ops, receipt)
        function_count = 1
    else:
        source_signatures, source_functions = _function_inventory(source_mlir)
        processed_signatures, processed_functions = _function_inventory(preprocessed_mlir)
        if source_signatures != processed_signatures or receipt.get("function_signatures") != source_signatures:
            raise ValueError("function symbols, definitions, or types differ across preprocessing")
        if list(source_functions) != list(processed_functions):
            raise ValueError("defined function order differs across preprocessing")
        parts = receipt.get("functions")
        if (not isinstance(parts, list) or len(parts) != len(source_functions)
                or receipt.get("defined_function_count") != len(parts)):
            raise ValueError("source-transform map function denominator differs from exact IR")
        source_count = 0
        processed_count = 0
        checked_results = 0
        elided_results = 0
        elided_operations = 0
        for entry, part in zip(source_functions, parts, strict=True):
            if not isinstance(part, dict) or part.get("entry") != entry:
                raise ValueError("source-transform map function identity differs from exact IR")
            source_ops, processed_ops = source_functions[entry], processed_functions[entry]
            source_count += len(source_ops)
            processed_count += len(processed_ops)
            checked, elided, elided_ops = _check_operation_rows(source_ops, processed_ops, part)
            checked_results += checked
            elided_results += elided
            elided_operations += elided_ops
        if (receipt.get("source_op_count") != source_count
                or receipt.get("preprocessed_op_count") != processed_count):
            raise ValueError("source-transform map aggregate operation denominator differs from exact IR")
        function_count = len(parts)
    result = {
        "schema": "merlin.typed_preprocessing_correspondence.v1",
        "claim": "preprocessing_result_types_only",
        "source_sha256": receipt["source_sha256"],
        "preprocessed_sha256": receipt["preprocessed_sha256"],
        "source_transform_map_sha256": descriptor["sha256"],
        "source_operation_count": source_count,
        "checked_result_count": checked_results,
        "elided_source_result_count": elided_results,
        "elided_source_operation_count": elided_operations,
        "value_preservation": "not_checked",
        "terminal_lowering": "not_checked",
    }
    if function_count > 1:
        result["defined_function_count"] = function_count
    return result
