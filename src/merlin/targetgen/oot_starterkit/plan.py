"""Public, target-neutral authoring checks for ``mixed_program_plan_v1``.

The inventory counts every direct operation in the unique model entry block except
``func.return``. In particular constants, empty tensors, splats and fills have
indices: the payload-only linalg reader deliberately omits those operations.
These checks establish syntax and source/plan consistency, not device execution,
semantic equivalence, eligibility, or numerical correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.common.ir_lock import IR_LOCK
from merlin.common.paths import data_path
from merlin.targetgen.contract.linalg_iface import make_linalg_context

OUTPUT_WRITER_OWNERSHIP_FINDING = "output writer lacks exact source-result task ownership"


def _attribute_text(op, name: str) -> str | None:
    from xdsl.dialects.builtin import StringAttr

    for table in (op.attributes, getattr(op, "properties", {}) or {}):
        value = table.get(name)
        if isinstance(value, StringAttr):
            return value.data
    return None


def _parsed_source(source: bytes):
    from xdsl.parser import Parser

    text = source.decode("utf-8")
    with IR_LOCK:
        module = Parser(make_linalg_context(), text).parse_module()
    functions = [op for op in module.body.block.ops if op.name == "func.func" and op.body.blocks]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("source requires one unique single-block function definition")
    entry = _attribute_text(functions[0], "sym_name")
    if not entry:
        raise ValueError("source function has no symbol name")
    block = functions[0].body.blocks[0]
    operations = [op for op in block.ops if op.name != "func.return"]
    returns = [op for op in block.ops if op.name == "func.return"]
    if len(returns) != 1 or block.last_op is not returns[0]:
        raise ValueError("source requires one terminating func.return")
    # Parsing alone accepts a return whose count or types disagree with the
    # declared function result ABI. Verify that exact boundary before exposing
    # its ordered value identities; this grants no body/effect equivalence.
    returns[0].verify()
    return entry, block, operations, returns[0]


def _tensor_type(value) -> dict[str, Any]:
    from xdsl.dialects.builtin import TensorType

    typ = value.type
    if isinstance(typ, TensorType):
        return {"shape": [int(d) for d in typ.get_shape()], "dtype": str(typ.get_element_type())}
    return {"type": str(typ)}


def source_operation_inventory(source: bytes | str) -> dict[str, Any]:
    """Inventory exact UTF-8 source bytes, including direct initialization operations."""
    raw = source.encode("utf-8") if isinstance(source, str) else source
    entry, block, operations, returned = _parsed_source(raw)
    owners = {
        value: {"op_index": i, "result_index": j}
        for i, op in enumerate(operations)
        for j, value in enumerate(op.results)
    }
    owners.update({value: {"arg_index": i} for i, value in enumerate(block.args)})

    def value_record(value):
        return {**_tensor_type(value), "source": owners.get(value)}

    rows = []
    for i, op in enumerate(operations):
        rows.append(
            {
                "source_op_index": i,
                "operation": op.name
                if op.name != "builtin.unregistered"
                else (_attribute_text(op, "op_name__") or op.name),
                "region_id": _attribute_text(op, "prov.region_id"),
                "operands": [value_record(value) for value in op.operands],
                "results": [_tensor_type(value) for value in op.results],
            }
        )
    return {
        "schema": "mixed_source_operation_inventory_v1",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "entry": entry,
        "source_op_count": len(rows),
        "arguments": [_tensor_type(value) for value in block.args],
        "operations": rows,
        "returns": [value_record(value) for value in returned.operands],
    }


def _plan_shape_problems(plan: Mapping[str, Any]) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("mixed-program plan validation requires jsonschema") from exc
    schema = json.loads(data_path("contract", "schemas", "mixed_program_plan.schema.json").read_text())
    return [
        f"plan schema {e.json_path}: {e.message}"
        for e in sorted(Draft202012Validator(schema).iter_errors(plan), key=lambda e: e.json_path)
    ]


def _source_plan_problems(
    inventory: Mapping[str, Any], command_buffer: Mapping[str, Any], plan: Mapping[str, Any]
) -> list[str]:
    problems: list[str] = []
    if plan["source_sha256"] != inventory["source_sha256"]:
        problems.append("plan source_sha256 differs from exact source bytes")
    nops = inventory["source_op_count"]
    if plan["source_op_count"] != nops:
        problems.append("plan source_op_count differs from direct source operations")
    tasks = plan["tasks"]
    if [row["task_index"] for row in tasks] != list(range(len(tasks))):
        problems.append("task_index must enumerate emission order exactly")
    owned: dict[int, int] = {}
    ranges = []
    for row in tasks:
        ident = row["task_index"]
        for index in row["source_op_indices"]:
            if index >= nops:
                problems.append(f"task {ident} references absent source operation {index}")
            elif index in owned:
                problems.append(f"source operation {index} is owned more than once")
            else:
                owned[index] = ident
        if "source_region_ids" in row:
            actual = sorted(
                {
                    inventory["operations"][index]["region_id"]
                    for index in row["source_op_indices"]
                    if index < nops and inventory["operations"][index]["region_id"] is not None
                }
            )
            if sorted(row["source_region_ids"]) != actual:
                problems.append(f"task {ident} region IDs differ from owned source operations")
        start, end = row["instruction_start"], row["instruction_end"]
        if end <= start:
            problems.append(f"task {ident} has empty or reversed instruction range")
        ranges.append((start, end))
    if set(owned) != set(range(nops)):
        problems.append(f"source operations lack unique task ownership: {sorted(set(range(nops)) - owned.keys())}")
    for index, operation in enumerate(inventory["operations"]):
        for operand in operation["operands"]:
            origin = operand["source"]
            if origin is not None and "op_index" in origin:
                producer = origin["op_index"]
                if producer in owned and index in owned and owned[producer] > owned[index]:
                    problems.append(f"task order violates source dependency {producer}->{index}")
    for key in ("prologue_instruction_range", "epilogue_instruction_range"):
        start, end = plan[key]
        if end < start:
            problems.append(f"{key} is reversed")
        elif end > start:
            ranges.append((start, end))
    ranges.sort()
    if (
        not ranges
        or ranges[0][0] != 0
        or ranges[-1][1] != plan["schedule_instruction_count"]
        or any(left[1] != right[0] for left, right in zip(ranges, ranges[1:]))
    ):
        problems.append("task and wrapper ranges do not partition scheduled instructions")

    tensors = command_buffer.get("tensors")
    if not isinstance(tensors, Mapping):
        return problems + ["whole-program tensors must be a mapping"]
    names = set(tensors)
    params = command_buffer.get("params")
    encodings = params.get("storage_encodings", {}) if isinstance(params, Mapping) else {}
    if not isinstance(encodings, Mapping) or not set(encodings).issubset(names):
        problems.append("storage_encodings must name declared tensors")
        encodings = {}
    elif encodings and set(encodings) != names:
        problems.append("explicit storage_encodings must cover every materialized tensor")
    abi = command_buffer.get("kernel_abi")
    if not isinstance(abi, Mapping) or abi.get("kind") != "whole_program":
        return problems + ["kernel_abi.kind must be whole_program"]
    abi_args = abi.get("args")
    if not isinstance(abi_args, list) or any(not isinstance(arg, Mapping) for arg in abi_args):
        return problems + ["kernel_abi.args must name tensor records"]
    abi_names = [arg.get("tensor") for arg in abi_args]
    if any(not isinstance(name, str) for name in abi_names):
        return problems + ["kernel_abi.args must contain tensor names"]
    if len(set(abi_names)) != len(abi_names) or set(abi_names) != names:
        problems.append("kernel_abi.args must cover each materialized tensor exactly once")
    if abi.get("outputs") != plan["output_bindings"]:
        problems.append("kernel_abi.outputs differ from source return bindings")
    if len(plan["entry_bindings"]) != len(inventory["arguments"]):
        problems.append("entry_bindings do not cover the source signature")

    values: dict[tuple[Any, ...], str] = {}

    def bind(key, name, typ):
        if key in values:
            problems.append(f"duplicate source binding {key}")
            return
        if name not in names:
            problems.append(f"source value names absent tensor {name!r}")
            return
        spec = tensors[name]
        encoding = encodings.get(name)
        if encoding is not None:
            shape_ok = (
                isinstance(encoding, Mapping)
                and encoding.get("schema") == "grouped_axes_storage_v1"
                and encoding.get("logical_shape") == typ.get("shape")
                and isinstance(spec, Mapping)
                and encoding.get("physical_shape") == spec.get("shape")
                and encoding.get("dtype") == typ.get("dtype") == spec.get("dtype")
            )
        else:
            shape_ok = (
                "shape" in typ
                and isinstance(spec, Mapping)
                and spec.get("shape") == typ["shape"]
                and spec.get("dtype") == typ["dtype"]
            )
        if not shape_ok:
            problems.append(f"tensor {name!r} changes source shape or dtype")
        values[key] = name

    for i, name in enumerate(plan["entry_bindings"][: len(inventory["arguments"])]):
        bind(("arg", i), name, inventory["arguments"][i])
    for row in plan["source_values"]:
        i, j = row["op_index"], row["result_index"]
        if i >= nops or j >= len(inventory["operations"][i]["results"]):
            problems.append("source_values references absent operation result")
        else:
            bind(("op", i, j), row["tensor"], inventory["operations"][i]["results"][j])
    if len(plan["output_bindings"]) != len(inventory["returns"]):
        problems.append("output_bindings do not cover source returns")
    else:
        for value, name in zip(inventory["returns"], plan["output_bindings"], strict=True):
            source = value["source"]
            key = (
                ("arg", source["arg_index"])
                if source and "arg_index" in source
                else ("op", source["op_index"], source["result_index"])
                if source
                else None
            )
            if key is None or values.get(key) != name:
                problems.append("output_bindings differ from actual source return order")
    # This one-shot whole-program ABI has no input/output alias or epilogue-copy
    # contract. A source return bound to an output must have one materialized
    # source-result owner and exactly one writer in that same source-owning task.
    # The public check is structural only; the grader independently rechecks it.
    for name in plan["output_bindings"]:
        source_results = [row for row in plan["source_values"] if row["tensor"] == name]
        writers = [row for row in tasks if name in row["writes"]]
        if (
            not any(arg.get("tensor") == name and arg.get("access") == "write" for arg in abi_args)
            or len(source_results) != 1
            or len(writers) != 1
            or source_results[0]["op_index"] not in writers[0]["source_op_indices"]
        ):
            problems.append(OUTPUT_WRITER_OWNERSHIP_FINDING)
    temporary_names = set()
    for row in plan.get("compiler_temporaries", []):
        name, i, j = row["tensor"], row["source_op_index"], row["source_result_index"]
        if (
            name in values.values()
            or name in temporary_names
            or i >= nops
            or j >= len(inventory["operations"][i]["results"])
        ):
            problems.append("compiler temporary lacks unique source-result provenance")
        elif (
            name not in names
            or not isinstance(tensors[name], Mapping)
            or (
                tensors[name].get("role") != "intermediate"
                or tensors[name].get("shape") != inventory["operations"][i]["results"][j].get("shape")
                or tensors[name].get("dtype") != inventory["operations"][i]["results"][j].get("dtype")
            )
        ):
            problems.append(f"compiler temporary {name!r} lacks matching intermediate shape/dtype storage")
        temporary_names.add(name)
    if names != set(values.values()) | temporary_names:
        problems.append("materialized tensors lack source or temporary provenance")
    initialized = set()
    for row in tasks:
        for access in ("reads", "writes"):
            for name in row[access]:
                if name not in names:
                    problems.append(f"task {row['task_index']} crosses absent tensor {name!r}")
                elif access == "reads" and name in temporary_names and name not in initialized:
                    problems.append(f"temporary {name!r} read before a task writes it")
                elif access == "writes" and name in temporary_names:
                    initialized.add(name)
    if temporary_names != initialized:
        problems.append("declared compiler temporary has no owning writer")
    return problems


def _lowered_task_problems(text: str, plan: Mapping[str, Any], command_buffer: Mapping[str, Any]) -> list[str]:
    from xdsl.parser import Parser

    from merlin.perf.task_cfg_evidence import analyze_task_cfg
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    with IR_LOCK:
        module = Parser(make_llvm_context(), text).parse_module()
        module.verify()
    functions = [op for op in module.body.block.ops if op.name == "llvm.func" and op.body.blocks]
    if len(functions) != 1:
        return ["lowered artifact requires one defined llvm.func kernel"]
    fn = functions[0]
    if len(fn.body.blocks[0].args) != len(command_buffer["kernel_abi"]["args"]):
        return ["lowered kernel pointer arity differs from kernel_abi.args"]
    task_ids = {row["task_index"] for row in plan["tasks"]}
    source_owners = {index: row["task_index"] for row in plan["tasks"] for index in row["source_op_indices"]}
    seen = set()
    problems = []
    for op in fn.walk():
        attr = op.attributes.get("merlin.global_task")
        if attr is None:
            continue
        ident = getattr(attr, "value", None)
        ident = getattr(ident, "data", ident)
        if type(ident) is not int or ident not in task_ids | {-1, -2}:
            problems.append("lowered operation names an absent global task")
            continue
        if ident in task_ids:
            seen.add(ident)
        source = op.attributes.get("merlin.source_op_index")
        if source is not None:
            index = getattr(source, "value", None)
            index = getattr(index, "data", index)
            if not isinstance(index, int) or source_owners.get(index) != ident:
                problems.append("lowered source_op_index differs from owning global task")
    if seen != task_ids:
        problems.append(f"planned tasks have no tagged lowered operation: {sorted(task_ids - seen)}")
    # Reuse only generic structural checks, never the private grading pipeline.
    # Tagged constants and optional branches cannot stand in for executable tasks.
    problems.extend(analyze_task_cfg(fn, sorted(task_ids))["problems"])
    return problems


def validate_mixed_program_plan(
    source: bytes | str, command_buffer: Mapping[str, Any], lowered_mlir: str | None = None
) -> dict[str, Any]:
    """Public preflight only; a successful result is never a grading certificate."""
    from xdsl.utils.exceptions import ParseError, VerifyException

    try:
        if not isinstance(command_buffer, Mapping):
            return {"ok": False, "findings": ["command buffer must be an object"]}
        inventory = source_operation_inventory(source)
        params = command_buffer.get("params")
        plan = params.get("global_program_plan") if isinstance(params, Mapping) else None
        if not isinstance(plan, Mapping):
            return {"ok": False, "findings": ["params.global_program_plan is required"]}
        findings = _plan_shape_problems(plan)
        if not findings:
            findings = _source_plan_problems(inventory, command_buffer, plan)
            abi = command_buffer.get("kernel_abi")
            if lowered_mlir is not None and isinstance(abi, Mapping) and abi.get("kind") == "whole_program":
                findings += _lowered_task_problems(lowered_mlir, plan, command_buffer)
        return {
            "ok": not findings,
            "findings": findings,
            "source_sha256": inventory["source_sha256"],
            "source_op_count": inventory["source_op_count"],
            "scope": "public structural preflight only",
        }
    except (ValueError, UnicodeError, ParseError, VerifyException) as exc:
        return {"ok": False, "findings": [f"invalid source or plan: {exc}"]}


def _authoring_guidance(findings: list[str]) -> list[str]:
    """Explain public CFG refusals without relaxing the structural preflight."""
    explanations = {
        "planned task emits no owned computation or control flow": (
            "A hoisted constant or tag-only marker does not establish source-owned task work. "
            "Bind actual materialization/computation to the task, or use an explicitly supported "
            "constant-folding proof; relabelling unrelated operations is insufficient."
        ),
        "a returning CFG path bypasses an entire planned task": (
            "Every planned task must have owned executable work on each returning CFG path. "
            "A task present only on an optional path cannot prove mandatory source coverage."
        ),
        "kernel control-flow edge reverses scheduled task order": (
            "A cross-task reverse CFG edge, including a fused-loop backedge, contradicts the "
            "declared linear task order. A genuine fused task must own the exact direct source "
            "indices it represents and retain independent transformation evidence; changing "
            "task tags alone is not proof."
        ),
    }
    return [explanations[problem] for problem in explanations if problem in findings]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("inventory", "validate"):
        command = sub.add_parser(action)
        command.add_argument("--source", type=Path, required=True)
        if action == "validate":
            command.add_argument("--command-buffer", type=Path, required=True)
            command.add_argument("--lowered-mlir", type=Path)
    args = parser.parse_args(argv)
    source = args.source.read_bytes()
    if args.action == "inventory":
        result = source_operation_inventory(source)
    else:
        result = validate_mixed_program_plan(
            source,
            json.loads(args.command_buffer.read_text()),
            args.lowered_mlir.read_text() if args.lowered_mlir else None,
        )
        result = {**result, "authoring_guidance": _authoring_guidance(result["findings"])}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.action == "inventory" or result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
