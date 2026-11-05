"""Rich observations of a candidate's own public artifacts; no ranking authority."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from merlin.perf.candidate_source import (
    MAX_OPERATIONS,
    MAX_TEXT,
    digest,
    placement_observation,
    public_projection,
    source_observation,
)

SCHEMA = "merlin.candidate_diagnostics.v1"


def _unknown(error):
    # Parser/provider messages may contain private paths or copied source text.
    return {"status": "UNKNOWN", "reason": "observation unavailable: " + type(error).__name__}


def _host(text, entry_symbol):
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.parser import Parser

    from merlin.common.ir_lock import IR_LOCK
    from merlin.perf.emitted_work import assembly_site_kind, summarize_emitted_work
    from merlin.perf.host_cfg_activity import analyze_host_cfg_activity

    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, llvm.LLVM):
        context.load_dialect(dialect)
    with IR_LOCK:
        module = Parser(context, text).parse_module()
    if sum(1 for _ in module.walk()) > MAX_OPERATIONS:
        raise ValueError("host operation limit exceeded")
    rows = []
    for op in module.walk():
        if not isinstance(op, llvm.FuncOp) or not op.body.blocks:
            continue
        row = {"symbol": op.sym_name.data, "selected_entry": op.sym_name.data == entry_symbol,
               "activity": analyze_host_cfg_activity(op)}
        if row["selected_entry"]:
            try:
                row["emitted_work"] = {"status": "observed", **summarize_emitted_work(
                    op, command_kind=assembly_site_kind, max_operations=MAX_OPERATIONS,
                )}
            except (ValueError, TypeError, KeyError) as error:
                row["emitted_work"] = _unknown(error)
        rows.append(row)
    return {"status": "observed" if rows else "UNKNOWN", "functions": rows,
            "scope": "emitted LLVM before native optimization; supported constant CFG execution counts only",
            "cpu_cycles": None, "physical_traffic_bytes": None}


def _binary(data):
    from merlin.perf.debug_companion import _read

    _, sections = _read(data)
    rows = [{"name": row.name, "bytes": row.size, "address": row.address,
             "allocated": bool(row.flags & 2), "executable": bool(row.flags & 4),
             "writable": bool(row.flags & 1), "file_backed": row.kind != 8}
            for row in sections]
    return {"status": "observed", "sections": rows,
            "executable_bytes": sum(row["bytes"] for row in rows if row["executable"]),
            "readonly_allocated_data_bytes": sum(row["bytes"] for row in rows
                                                 if row["allocated"] and not row["executable"] and not row["writable"]),
            "constant_table_semantics": "UNKNOWN", "peak_stack_bytes": None,
            "scope": "linked section extents; readonly data is not necessarily a constant table"}


def observe_candidate_artifacts(*, source_mlir=None, emitted_mlir=None, command_buffer=None,
                                artifacts=None, compiler_options=None, numeric_context=None,
                                elf_bytes=None, stack_usage_text=None, entry_symbol=None,
                                candidate_sha256=None):
    """Pure bounded census; caller owns original compilation/consumption and public selection.

    ``artifacts`` maps semantic roles to raw own-artifact bytes. Only identities
    are returned. Options and numeric context are public declarations, never
    evidence that a transform executed or that numerical/physical gates passed.
    """
    started = time.monotonic()
    artifacts = artifacts or {}
    if (len(artifacts) > 32 or any(type(raw) is not bytes or len(raw) > 64 << 20 for raw in artifacts.values())
            or sum(len(raw) for raw in artifacts.values()) > 128 << 20):
        raise ValueError("candidate artifact roster exceeds limit")
    for text in (source_mlir, emitted_mlir, stack_usage_text):
        if text is not None and (type(text) is not str or len(text.encode()) > MAX_TEXT):
            raise ValueError("candidate text exceeds limit")
    if elf_bytes is not None and (type(elf_bytes) is not bytes or len(elf_bytes) > 64 << 20):
        raise ValueError("candidate ELF exceeds limit")
    result = {"schema": SCHEMA, "authority": "none", "scope": "own public source and emitted artifacts",
              "artifacts": {role: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                            for role, raw in artifacts.items()},
              "compiler_options": {"status": "declared" if compiler_options is not None else "UNKNOWN",
                                   "selection": compiler_options, "applied_to_codegen": "UNKNOWN"},
              "numerical_context": {"status": "declared" if numeric_context is not None else "UNKNOWN",
                                    "selection": numeric_context, "numerical_verdict": "UNKNOWN"},
              "dynamic_execution": "UNKNOWN", "physical_memory_traffic": None,
              "timing_cycles": None, "array_utilization": None}
    for field, call in (
        ("source", lambda: source_observation(source_mlir) if source_mlir is not None else {"status": "UNKNOWN"}),
        ("host", lambda: _host(emitted_mlir, entry_symbol) if emitted_mlir is not None else {"status": "UNKNOWN"}),
        ("linked_image", lambda: _binary(elf_bytes) if elf_bytes is not None else {"status": "UNKNOWN"}),
    ):
        try:
            result[field] = call()
        except Exception as error:  # noqa: BLE001 -- retain partial diagnostics, never infer zero
            result[field] = _unknown(error)
    try:
        from merlin.perf.command_buffer_diagnostics import representation_activity

        result["placement"] = placement_observation(result["source"], command_buffer or {})
        result["representations"] = representation_activity(command_buffer or {})
    except Exception as error:  # noqa: BLE001 -- malformed declarations remain UNKNOWN
        result["placement"] = result["representations"] = _unknown(error)
    result["source_emission_join"] = {"status": "UNKNOWN"}
    params = (command_buffer or {}).get("params")
    plan = params.get("global_program_plan") if isinstance(params, dict) else None
    if source_mlir and emitted_mlir and candidate_sha256 and plan:
        try:
            from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan

            result["source_emission_join"] = verify_compiler_global_plan(
                source_text=source_mlir, lowered_text=emitted_mlir, command_buffer=command_buffer,
                candidate_sha256=candidate_sha256,
            )
        except Exception as error:  # noqa: BLE001 -- source/CFG binding never impersonates execution
            result["source_emission_join"] = _unknown(error)
    result["dependency_stages"] = {
        "status": "verified_source_emission"
                  if result["source_emission_join"].get("status") == "verified" else "UNKNOWN",
        "stages": [{key: row.get(key) for key in ("task_index", "source_op_indices", "reads", "writes",
                                                 "instruction_start", "instruction_end")}
                   for row in plan.get("tasks", [])] if isinstance(plan, dict)
                   and result["source_emission_join"].get("status") == "verified" else [],
        "scope": "original complete source and CFG ownership and declared tensor dependencies",
        "resource_intervals": "UNKNOWN", "physical_overlap": "UNKNOWN",
    }
    try:
        from merlin.targetgen.contract.stack_usage import parse_stack_usage

        rows = parse_stack_usage(stack_usage_text) if stack_usage_text is not None else ()
        result["stack_frames"] = {"status": "recorded" if rows else "UNKNOWN",
                                  "rows": [{"function": row.function, "frame_bytes": row.frame_bytes,
                                            "allocation": row.allocation} for row in rows],
                                  "source_and_tool_consumption_join": "UNKNOWN", "peak_stack_bytes": None}
    except Exception as error:  # noqa: BLE001 -- no ambiguous frame may be credited
        result["stack_frames"] = _unknown(error)
    result["completeness"] = {"status": "partial", "missing": [
        "verified source operation ownership and emitted numerical equivalence",
        "post-optimization host attribution and opaque assembly effects",
        "dynamic multiplicity, physical traffic, service prices and overlap",
        "interprocedural peak stack and constant-table semantics",
    ]}
    from merlin.perf import candidate_features as feature_owner
    from merlin.perf import candidate_source as source_owner

    result["provenance"] = {"observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "source_observer_sha256": hashlib.sha256(
                                Path(source_owner.__file__).read_bytes()).hexdigest(),
                            "feature_observer_sha256": hashlib.sha256(
                                Path(feature_owner.__file__).read_bytes()).hexdigest(),
                            "source_scope": "direct observer owners; original caller retains its full source inventory",
                            "input_basis": "caller-selected original public artifact bytes"}
    result["collection_cost"] = {"wall_seconds": time.monotonic() - started, "engine_execution": False,
                                 "source_bytes": sum(len(raw) for raw in artifacts.values())}
    from merlin.perf.candidate_features import candidate_features

    result["candidate_features"] = candidate_features(result)
    from merlin.perf.candidate_cost_graph import observe_candidate_cost_graph

    result["cost_graph"] = observe_candidate_cost_graph(result, command_buffer=command_buffer)
    projected = public_projection(result)
    projected["observation_sha256"] = digest(projected)
    return projected
