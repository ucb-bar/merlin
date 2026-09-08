"""Host-owned reduced source-chain numerical qualification for a global compiler edit.

No simulation, candidate Python import, or native candidate execution in the host process.
The existing experiment compiler/native sandbox hooks own every submitted executable.
The receipt does not upgrade a reduced check into a full-shape equivalence theorem.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping

from .host_source_witness import (evaluate_pointwise_source, extract_pointwise_chain,
                                  extract_dequant_contraction, extract_pointwise_reduction,
                                  extract_bounded_gather, extract_generic_reduction,
                                  extract_named_reduction, extract_insert_slice,
                                  extract_pointwise_concat)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cached_host_task_activity(plan: Mapping[str, Any], *, lowered_sha256: str) -> dict[str, Mapping[str, Any]]:
    """Use complete host-retained analysis, never reparse a full model in a probe.

    The controller supplies its verified immutable analysis record. Artifact
    binding and completeness are required; legacy top-N reports fail closed.
    """
    activity = plan.get("host_activity") or {}
    rows = activity.get("tasks")
    if (plan.get("status") != "verified" or not lowered_sha256
            or plan.get("candidate_lowered_sha256") != lowered_sha256
            or activity.get("artifact_sha256") != lowered_sha256
            or activity.get("status") != "derived"
            or activity.get("task_activity_coverage") != "complete"
            or not isinstance(rows, list)):
        raise ValueError("complete artifact-bound cached host task activity unavailable")
    table = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("task"), str) or row["task"] in table:
            raise ValueError("cached host task activity has ambiguous task rows")
        if any(type(row.get(key)) is not int or row[key] < 0
               for key in ("load_payload_bytes", "store_payload_bytes")):
            raise ValueError("cached host task payload is unknown")
        table[row["task"]] = row
    count = plan.get("tasks")
    if type(count) is not int or count < 0 or not {str(index) for index in range(count)} <= table.keys():
        raise ValueError("cached host task activity omits a planned task")
    return table


def changed_host_task_candidates(before_tasks, after_tasks, tables):
    """Rank same-source host tasks; bytes and operations remain separate units.

    Task indices are compiler-local and may change. Several complete preceding
    host tasks may merge into one current task. Partial overlap or mixed-lane
    ownership needs another attribution proof, never matching by task index.
    """
    candidates, unresolved = [], []
    for task in after_tasks:
        if task["kind"] != "host":
            continue
        wanted = set(task["source_op_indices"])
        prior = [old for old in before_tasks if wanted.intersection(old["source_op_indices"])]
        covered = [index for old in prior for index in old["source_op_indices"]]
        current_key = str(task["task_index"])
        if (not prior or any(old["kind"] != "host" for old in prior)
                or set(covered) != wanted or len(set(covered)) != len(covered)
                or any(str(old["task_index"]) not in tables[0] for old in prior)
                or current_key not in tables[1]):
            unresolved.append({"task_index": task["task_index"],
                "source_op_indices": task["source_op_indices"],
                "reason": "same-source host task accounting unavailable across revisions"})
            continue
        row_groups = [[tables[0][str(old["task_index"])] for old in prior], [tables[1][current_key]]]
        payload = [sum(row[name] for row in group for name in ("load_payload_bytes", "store_payload_bytes"))
                   for group in row_groups]
        integer = []
        for group in row_groups:
            values = [(row.get("dynamic_operations") or {}).get("integer_arithmetic") for row in group]
            integer.append(sum(values) if all(type(value) is int and value >= 0 for value in values) else None)
        integer_delta = (abs(integer[1] - integer[0]) if all(type(value) is int and value >= 0
                         for value in integer) else None)
        if payload[0] != payload[1] or (integer_delta is not None and integer_delta > 0):
            # Lexicographic priority, not addition of bytes and operations.
            selected = {**task, "comparison_prior_task_indices": [old["task_index"] for old in prior]}
            candidates.append(((abs(payload[1] - payload[0]), integer_delta is not None, integer_delta), selected, payload))
    return candidates, unresolved


_SOURCE_WITNESS_KINDS = (
    "pointwise_concat", "insert_slice", "bounded_gather", "named_reduction", "generic_reduction",
    "pointwise_reduction", "dequant_contraction", "fanout", "chain",
)


def _ranked_source_witness_options(candidates):
    """Try every supported mechanism on the best changed task before the next task.

    Each extractor currently parses the source module. Grouping by mechanism made a large graph pay
    one full parse for every changed task before reaching its likely pointwise mechanism. Task-major
    order preserves the payload/integer-work ranking and bounds the common successful path to at most
    one pass through the extractor set.
    """
    return ((kind, row)
            for row in sorted(candidates, key=lambda candidate: candidate[0], reverse=True)
            for kind in _SOURCE_WITNESS_KINDS)


def pointwise_concat_emission_evidence(
        extraction: Mapping[str, Any], before_activity: Mapping[str, Any],
        after_activity: Mapping[str, Any], *, emitted_operation_names,
        direct_output: bool, materialization_delta: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the reduced artifact computes the pointwise segment into concat output.

    Numerical equivalence is established separately by the independent source evaluator.  This
    check binds that result to the emitted mechanism: the source result scalar remains present,
    the producer tensor's exact payload disappears, and the reduced artifact writes directly to
    its sole result without another allocation.
    """
    lowered_scalar = {
        "arith.negf": "llvm.fneg", "arith.addf": "llvm.fadd",
        "arith.subf": "llvm.fsub", "arith.mulf": "llvm.fmul",
        "arith.divf": "llvm.fdiv", "arith.addi": "llvm.add",
        "arith.subi": "llvm.sub", "arith.muli": "llvm.mul",
        "arith.extsi": "llvm.sext", "arith.extui": "llvm.zext",
        "arith.trunci": "llvm.trunc", "arith.sitofp": "llvm.sitofp",
        "arith.uitofp": "llvm.uitofp", "arith.fptosi": "llvm.fptosi",
        "arith.fptoui": "llvm.fptoui",
    }
    source_scalar = extraction.get("producer_result_scalar_operation")
    emitted_scalar = lowered_scalar.get(source_scalar)
    names = list(emitted_operation_names)
    actual_scalar_operations = names.count(emitted_scalar) if emitted_scalar else 0
    counts_preserved = (
        all(arm.get("status") == "derived" for arm in (before_activity, after_activity))
        and all((before_activity.get("dynamic_operations") or {}).get(category)
                == (after_activity.get("dynamic_operations") or {}).get(category)
                for category in ("floating_arithmetic", "conversion"))
    )
    deleted = materialization_delta.get("deleted_payload_bytes") or {}
    expected = extraction.get("probe_intermediate_payload_bytes")
    operand_shapes = extraction.get("probe_operand_shapes")
    operand_count = extraction.get("concat_operand_count")
    result_shape = extraction.get("probe_result_shape")
    identity_contract = (
        operand_count != 1
        or (isinstance(operand_shapes, list) and len(operand_shapes) == 1
            and extraction.get("identity_concat") is True
            and operand_shapes[0] == result_shape)
    )
    source_contract = (
        extraction.get("schema") == "actual_source_pointwise_concat_witness_v1"
        and extraction.get("mechanism") == "pointwise_concat"
        and extraction.get("all_source_producer_uses_preserved") is True
        and type(extraction.get("concat_axis")) is int
        and isinstance(operand_shapes, list)
        and isinstance(result_shape, list)
        and type(operand_count) is int and operand_count >= 1
        and operand_count == len(operand_shapes)
        and identity_contract
    )
    exact_deletion = (
        materialization_delta.get("status") == "changed_reduced_materialization"
        and type(expected) is int and expected > 0
        and deleted.get("static_allocation_payload_bytes") == expected
        and type(deleted.get("load_payload_bytes")) is int
        and deleted["load_payload_bytes"] >= expected
        and type(deleted.get("store_payload_bytes")) is int
        and deleted["store_payload_bytes"] >= expected
    )
    demonstrated = bool(source_contract and direct_output and emitted_scalar
                        and actual_scalar_operations == 1 and counts_preserved
                        and exact_deletion)
    return {
        "status": ("demonstrated_changed_pointwise_in_concat" if demonstrated
                   else "NOT_DEMONSTRATED"),
        "kind": "pointwise_producer_direct_scalar_store_in_static_concat_segment",
        "source_result_scalar_operation": source_scalar,
        "emitted_result_scalar_operation": emitted_scalar,
        "actual_result_scalar_operations": actual_scalar_operations,
        "floating_arithmetic_and_conversion_counts_preserved": counts_preserved,
        "direct_output_without_intermediate_allocation": bool(direct_output),
        "materialization_delta": dict(materialization_delta),
        "source_scalar_region_sha256": extraction.get("scalar_region_sha256"),
        "concat_axis": extraction.get("concat_axis"),
        "concat_operand_count": operand_count,
        "identity_concat": extraction.get("identity_concat"),
        "probe_operand_shapes": extraction.get("probe_operand_shapes"),
        "probe_result_shape": extraction.get("probe_result_shape"),
        "all_source_producer_uses_preserved": extraction.get(
            "all_source_producer_uses_preserved"),
        "performance": "UNMEASURED",
    }


def reduced_materialization_change(before: Mapping[str, Any], after: Mapping[str, Any],
                                   extraction: Mapping[str, Any], *,
                                   before_lowered_sha256: str | None = None,
                                   after_lowered_sha256: str | None = None) -> dict[str, Any]:
    """An unchanged supported mechanism is not evidence for a different compiler edit.

    Most supported fusions delete a tensor materialization. Direct generic and named reductions
    instead move the accumulator from the result buffer to an ordered scalar recurrence. Their
    pre-optimization total traffic can therefore grow even though repeated result-buffer
    addressing disappeared; report that distinction rather than pretending it is deletion.
    """
    fields = ("static_allocation_payload_bytes", "load_payload_bytes", "store_payload_bytes")
    if any(arm.get("status") != "derived" or any(type(arm.get(key)) is not int for key in fields)
           for arm in (before, after)):
        return {"status": "UNKNOWN", "reason": "reduced before/after memory activity is not derived"}
    deleted = {key: before[key] - after[key] for key in fields}
    expected = extraction.get("probe_intermediate_payload_bytes")
    mechanism = extraction.get("mechanism")
    if mechanism in {"generic_reduction", "named_reduction"}:
        output_bytes = extraction.get("probe_output_payload_bytes")
        steps = extraction.get("probe_reduction_steps_per_output")
        before_buffers = before.get("buffer_payload") or {}
        after_buffers = after.get("buffer_payload") or {}
        before_accumulator = any(
            key.startswith("alloca:") and row.get("load_payload_bytes", 0) >= output_bytes * steps
            and row.get("store_payload_bytes", 0) >= output_bytes * steps
            for key, row in before_buffers.items()) if type(output_bytes) is int and type(steps) is int else False
        after_result_loads = output_bytes if mechanism == "named_reduction" else 0
        after_result = any(
            key.startswith("alloca:") and row.get("load_payload_bytes") == after_result_loads
            and row.get("store_payload_bytes") == output_bytes
            for key, row in after_buffers.items()) if type(output_bytes) is int else False
        scalar_counts_preserved = all(
            (before.get("dynamic_operations") or {}).get(category, 0)
            == (after.get("dynamic_operations") or {}).get(category, 0)
            for category in ("floating_arithmetic", "conversion"))
        lowered_changed = (bool(before_lowered_sha256) and bool(after_lowered_sha256)
                           and before_lowered_sha256 != after_lowered_sha256)
        changed = (type(output_bytes) is int and output_bytes > 0 and type(steps) is int and steps > 0
                   and lowered_changed and before_accumulator and after_result and scalar_counts_preserved)
        return {"status": "changed_reduced_codegen" if changed else "NO_RELEVANT_REDUCED_CHANGE",
                "deleted_payload_bytes": deleted,
                "result_buffer_accumulator_loads_removed": bool(before_accumulator and after_result),
                "floating_arithmetic_and_conversion_counts_preserved": scalar_counts_preserved,
                "lowered_artifact_changed": lowered_changed,
                "total_preoptimization_payload_reduced": all(value >= 0 for value in deleted.values())
                    and any(value > 0 for value in deleted.values()),
                "expected_output_payload_bytes": output_bytes,
                "reduction_steps_per_output": steps,
                "final_result_buffer_load_bytes": after_result_loads,
                "ordered_accumulator_representation":
                    "ssa" if mechanism == "named_reduction" else "scalar_slot",
                "machine_scalar_slot_promotion": "UNPROVEN_BY_PREOPT_CFG",
                "scope": f"same cloned {mechanism.replace('_', ' ')} under preceding/current compilers; proves "
                         "a result-buffer-to-scalar-accumulator codegen change, not machine traffic or performance"}
    if mechanism == "insert_slice":
        overwritten = extraction.get("probe_inserted_payload_bytes")
        changed = (type(overwritten) is int and overwritten > 0
                   and deleted["static_allocation_payload_bytes"] == 0
                   and 0 <= deleted["load_payload_bytes"] <= overwritten
                   and deleted["store_payload_bytes"] == overwritten)
        return {"status": "changed_reduced_materialization" if changed else "NO_RELEVANT_REDUCED_CHANGE",
                "deleted_payload_bytes": deleted,
                "deleted_overwritten_payload_bytes": overwritten,
                "deleted_destination_load_bytes": deleted["load_payload_bytes"],
                "scope": "same cloned insert slice under preceding/current compilers; proves "
                         "one source-box worth of overwritten destination traffic was removed"}
    if mechanism in {"dequant_contraction", "pointwise_reduction", "pointwise_concat"}:
        changed = (type(expected) is int and expected > 0
                   and deleted["static_allocation_payload_bytes"] == expected
                   and deleted["store_payload_bytes"] >= expected
                   and deleted["load_payload_bytes"] >= expected)
    else:
        changed = any(value > 0 for value in deleted.values()) and all(value >= 0 for value in deleted.values())
    if not changed and mechanism in {"chain", "fanout", "bounded_gather"} and all(
            value == 0 for value in deleted.values()):
        operations = [arm.get("dynamic_operations") for arm in (before, after)]
        if all(isinstance(row, Mapping) for row in operations):
            integer = [row.get("integer_arithmetic") for row in operations]
            unchanged_other = all(operations[0].get(key, 0) == operations[1].get(key, 0)
                                  for key in ("floating_arithmetic", "conversion"))
            if (all(type(value) is int and value >= 0 for value in integer)
                    and integer[0] != integer[1] and unchanged_other
                    and before_lowered_sha256 and after_lowered_sha256
                    and before_lowered_sha256 != after_lowered_sha256):
                return {"status": "changed_reduced_integer_codegen", "deleted_payload_bytes": deleted,
                    "integer_operations_before_after": integer,
                    "scope": "same cloned source under both compilers changes preoptimization integer work; "
                             "may be address or data arithmetic, not machine work or a performance verdict"}
    return {"status": "changed_reduced_materialization" if changed else "NO_RELEVANT_REDUCED_CHANGE",
            "deleted_payload_bytes": deleted, "expected_producer_payload_bytes": expected,
            "scope": "same actual source witness compiled by preceding and current bound compilers; "
                     "not full-shape changed-chain attribution or performance"}


class HostChangedRegionQualifier:
    def __init__(self, *, native_layout: Callable[[Mapping[str, Any]], Mapping[str, int]],
                 expected_symbol: str, abi_provenance: Mapping[str, Any], output: Path,
                 float_atol: float = 2e-6, float_rtol: float = 1e-5):
        if not expected_symbol or not abi_provenance:
            raise ValueError("native host qualifier needs an independently derived ABI contract")
        self.native_layout = native_layout
        self.expected_symbol = expected_symbol
        self.abi_provenance = dict(abi_provenance)
        self.output = Path(output)
        self.float_atol, self.float_rtol = float_atol, float_rtol

    def __call__(self, *, candidate: Path, experiment: Any, timeout_s: float,
                 portfolio_member: Mapping[str, Any] | None = None) -> dict[str, Any]:
        from tempfile import mkdtemp
        import numpy as np
        from xdsl.dialects.llvm import LLVM
        from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
        from merlin.llvmlower.toolchain import mlir_translate
        from merlin.runtime.commandbuffer import validate_command_buffer
        from . import native_host_witness_runner

        started = monotonic()
        budget = min(float(timeout_s), float(experiment.timeout_s), 60.0)
        if budget <= 0:
            raise ValueError("source qualifier requires a positive remaining budget")
        deadline = started + budget
        self.output.mkdir(parents=True, exist_ok=True)
        work = Path(mkdtemp(prefix="host_chain_", dir=self.output))
        record: dict[str, Any] = {"schema": "host_changed_source_qualification_v1",
            "status": "unresolved", "full_model_correctness": "UNPROVEN",
            "full_shape_backend_correspondence": "UNPROVEN",
            "full_model_changed_chain_attribution": "UNPROVEN",
            "full_model_executed": False, "simulator_executed": False, "timing_measured": False,
            "abi_provenance": self.abi_provenance, "proof_scope": "selected reduced source mechanism only"}

        def remaining():
            value = deadline - monotonic()
            if value <= 0:
                raise TimeoutError("changed-source qualification exhausted its wall-clock budget")
            return value

        try:
            selected_context = None
            if callable(getattr(experiment, "selected_changed_portfolio_context", None)):
                selected_context = experiment.selected_changed_portfolio_context(
                    candidate, portfolio_member)
                current_context = selected_context["current"]
                previous_context = selected_context["previous"]
                binding = current_context["probe_binding"].to_dict()
                preceding_binding = previous_context["probe_binding"].to_dict()
                current = current_context["artifacts"]
                previous = previous_context["artifacts"]
                before, after = (previous_context["analysis"], current_context["analysis"])
                record["portfolio_member_binding"] = {
                    "selection": selected_context["selection"],
                    "previous": previous_context["member_binding"],
                    "current": current_context["member_binding"],
                }
            else:
                if portfolio_member is not None:
                    raise ValueError("portfolio member selection requires member-aware experiment accessors")
                binding = experiment.current_probe_binding(candidate).to_dict()
                preceding_binding = experiment.previous_probe_binding(candidate).to_dict()
                current = experiment.current_artifacts(candidate)
                previous = experiment.previous_artifacts(candidate)
                before, after = (row["analysis"] for row in experiment.iterations[-2:])
            if not previous or len(experiment.iterations) < 2:
                raise ValueError("qualification needs a previous bound full-model candidate artifact")
            before_plan = before["diagnostics"]["verified_global_plan_emission"]
            after_plan = after["diagnostics"]["verified_global_plan_emission"]
            for plan, artifact in ((before_plan, previous), (after_plan, current)):
                if (plan.get("status") != "verified" or plan.get("candidate_lowered_sha256")
                        != _sha(artifact["lowered_text"].encode())):
                    raise ValueError("numerical qualification requires bound verified full-model artifacts")
            if before_plan["source_sha256"] != after_plan["source_sha256"]:
                raise ValueError("source mechanism changed between candidate revisions")
            source_text = Path(current["interface"]).read_text()
            if _sha(source_text.encode()) != after_plan["source_sha256"]:
                raise ValueError("current source does not match verified full-model plan")
            # Reuse complete task accounting from mandatory model compilation.
            # Old top-N-only evidence is unavailable, never a reason to parse or
            # re-analyze the complete model inside a short probe action.
            from .host_cfg_activity import analyze_host_cfg_activity
            tables = [cached_host_task_activity(plan, lowered_sha256=_sha(artifact["lowered_text"].encode()))
                      for plan, artifact in ((before_plan, previous), (after_plan, current))]
            record["full_model_activity_source"] = "cached_complete_verified_analysis"
            tasks = current["command_buffer"]["params"]["global_program_plan"]["tasks"]
            candidates, unresolved = changed_host_task_candidates(
                previous["command_buffer"]["params"]["global_program_plan"]["tasks"], tasks, tables)
            record["unresolved_host_task_attribution"] = unresolved
            options = _ranked_source_witness_options(candidates)
            record["probe_selection_attempts"] = []
            if not callable(getattr(experiment, "compile_previous_probe_candidate", None)):
                raise ValueError("changed-region qualification needs an exact preceding-compiler probe compilation hook")
            for kind, (_, task, payload) in options:
                # Extraction parses host-owned source and therefore consumes this same action's
                # budget. Check before every attempt so an unsupported mechanism on a large graph
                # cannot silently start another complete parse after the deadline has expired.
                remaining()
                try:
                    if kind == "pointwise_concat":
                        probe_text, extraction = extract_pointwise_concat(
                            source_text, task["source_op_indices"])
                    elif kind == "insert_slice":
                        probe_text, extraction = extract_insert_slice(source_text, task["source_op_indices"])
                    elif kind == "bounded_gather":
                        probe_text, extraction = extract_bounded_gather(source_text, task["source_op_indices"])
                    elif kind == "named_reduction":
                        probe_text, extraction = extract_named_reduction(source_text, task["source_op_indices"])
                    elif kind == "generic_reduction":
                        probe_text, extraction = extract_generic_reduction(source_text, task["source_op_indices"])
                    elif kind == "pointwise_reduction":
                        probe_text, extraction = extract_pointwise_reduction(source_text, task["source_op_indices"])
                    elif kind == "dequant_contraction":
                        probe_text, extraction = extract_dequant_contraction(source_text, task["source_op_indices"])
                    else:
                        probe_text, extraction = extract_pointwise_chain(
                            source_text, task["source_op_indices"], mechanism=kind)
                except ValueError:
                    continue
                if len(record["probe_selection_attempts"]) >= 4:
                    break
                attempt = work / f"selection_{len(record['probe_selection_attempts'])}"
                attempt.mkdir()
                source = attempt / "probe.mlir"
                source.write_text(probe_text)
                reduced_activities, compilations = [], []
                for arm, compile_method in (("before", experiment.compile_previous_probe_candidate),
                                            ("after", experiment.compile_probe_candidate)):
                    compilation = compile_method(candidate, source, attempt / arm,
                                                 timeout_s=remaining(), emit_command_buffer=True)
                    result, command_buffer = compilation["lowered"], compilation["command_buffer"]
                    if (result.returncode or not command_buffer or validate_command_buffer(command_buffer)
                            or command_buffer.get("declined") or command_buffer.get("commands")
                            or len(result.stdout.encode()) > 2_000_000):
                        raise ValueError(f"{arm} reduced source compilation is not a bounded pure-host artifact")
                    context = make_context()
                    context.load_dialect(LLVM)
                    parsed_probe = parse_mlir_text(result.stdout, context)
                    functions = [op for op in parsed_probe.walk() if op.name == "llvm.func"]
                    if len(functions) != 1:
                        raise ValueError("reduced before/after compilation has no single host function")
                    reduced_activities.append(analyze_host_cfg_activity(functions[0]))
                    compilations.append(compilation)
                    (attempt / f"{arm}.lowered.mlir").write_text(result.stdout)
                before_lowered_sha256 = _sha(compilations[0]["lowered"].stdout.encode())
                after_lowered_sha256 = _sha(compilations[1]["lowered"].stdout.encode())
                delta = reduced_materialization_change(*reduced_activities, extraction,
                    before_lowered_sha256=before_lowered_sha256,
                    after_lowered_sha256=after_lowered_sha256)
                selection = {"kind": kind, "source_indices": extraction["source_indices"],
                             "probe_source_sha256": extraction["probe_source_sha256"],
                             "before_lowered_sha256": before_lowered_sha256,
                             "after_lowered_sha256": after_lowered_sha256,
                             "materialization_delta": delta}
                record["probe_selection_attempts"].append(selection)
                if delta["status"] not in {"changed_reduced_materialization", "changed_reduced_codegen",
                                           "changed_reduced_integer_codegen"}:
                    continue
                compiled = compilations[1]
                record["changed_reduced_mechanism"] = selection
                break
            else:
                raise ValueError("no selected actual-source witness exercises a reduced host-codegen change")
            if "changed_reduced_mechanism" not in record:
                raise ValueError("bounded source selection found no changed reduced host codegen")
            record.update(binding=binding, extraction=extraction, task_index=task["task_index"],
                          comparison_prior_task_indices=task["comparison_prior_task_indices"],
                          preceding_binding=preceding_binding,
                          task_payload_before_after=payload,
                          selection_scope="supported actual source chain within a same-source host task with changed "
                                          "payload or integer work; bytes prioritized separately from operations",
                          full_artifact_before_sha256=_sha(previous["lowered_text"].encode()),
                          full_artifact_after_sha256=_sha(current["lowered_text"].encode()))
            if kind == "dequant_contraction":
                record["selection_scope"] = (
                    "largest exact eligible dequant-to-contraction pair within a changed host task; "
                    "task membership/size ranking is not proof of full-model changed-chain attribution")
            lowered = compiled["lowered"]
            cb = compiled["command_buffer"]
            record["probe_compilation"] = {
                "lowered_returncode": lowered.returncode,
                "lowered_stderr": (lowered.stderr or "")[-3000:],
                "command_buffer_returncode": compiled["command_buffer_emission"].returncode
                    if compiled["command_buffer_emission"] is not None else None,
                "command_buffer_stderr": (compiled["command_buffer_emission"].stderr or "")[-3000:]
                    if compiled["command_buffer_emission"] is not None else None,
            }
            if lowered.returncode or not cb or compiled["command_buffer_emission"].returncode:
                raise ValueError("sandboxed same-candidate source witness compilation failed")
            if validate_command_buffer(cb) or cb.get("declined") or cb.get("commands"):
                raise ValueError("native witness must be a non-declined pure host program")
            abi = cb.get("kernel_abi") or {}
            if abi.get("kind") != "whole_program":
                raise ValueError("source witness needs canonical whole_program ABI, never carrier commands")
            if len(lowered.stdout.encode()) > 2_000_000:
                raise ValueError("short source witness emitted an oversized native artifact")
            context = make_context()
            context.load_dialect(LLVM)
            module = parse_mlir_text(lowered.stdout, context)
            functions = [op for op in module.walk() if op.name == "llvm.func"]
            if (len(functions) != 1 or functions[0].sym_name.data != self.expected_symbol
                    or len(functions[0].body.blocks[0].args) != len(abi["args"])
                    or any(op.name in {"llvm.inline_asm", "llvm.call", "llvm.call_intrinsic"} for op in module.walk())):
                raise ValueError("source witness is not a closed native host function")
            activity = analyze_host_cfg_activity(functions[0])
            output_specs = extraction["outputs"]
            output_bytes = [int(np.prod(spec["shape"])) * ((int(spec["dtype"][1:])+7)//8)
                            for spec in output_specs]
            final_bytes = sum(output_bytes)
            output_indices = [i for i, arg in enumerate(abi["args"]) if arg["access"] == "write"]
            buffers = activity.get("buffer_payload") or {}
            direct_output = (
                activity.get("status") == "derived"
                and activity.get("static_allocation_payload_bytes") == 0
                and activity.get("store_payload_bytes") == final_bytes
                and len(output_indices) == len(output_specs)
                and all(buffers.get(f"arg:{index}", {}).get("store_payload_bytes") == size
                        and buffers.get(f"arg:{index}", {}).get("load_payload_bytes") == 0
                        for index, size in zip(output_indices, output_bytes))
            )
            record["emitted_mechanism"] = {
                "status": "demonstrated_reduced_direct_output" if direct_output else "NOT_DEMONSTRATED",
                "kind": "pointwise_chain_without_intermediate_or_boundary_materialization",
                "host_activity": activity, "source_final_output_payload_bytes": final_bytes,
                "scope": "actual reduced emitted CFG, not a full-shape backend equivalence claim",
            }
            if extraction.get("mechanism") == "pointwise_concat":
                concat_evidence = pointwise_concat_emission_evidence(
                    extraction, reduced_activities[0], activity,
                    emitted_operation_names=(op.name for op in module.walk()),
                    direct_output=direct_output,
                    materialization_delta=record[
                        "changed_reduced_mechanism"]["materialization_delta"])
                record["emitted_mechanism"].update(concat_evidence)
                if concat_evidence["status"] != "demonstrated_changed_pointwise_in_concat":
                    raise ValueError(
                        "changed reduced pointwise-to-concat mechanism not demonstrated")
            if extraction.get("mechanism") == "bounded_gather":
                record["emitted_mechanism"].update(
                    status="demonstrated_changed_bounded_gather_subgraph",
                    kind="source_gather_index_padding_views_pointwise_subgraph",
                    materialization_delta=record["changed_reduced_mechanism"]["materialization_delta"],
                    geometry=extraction["geometry"],
                    attribution="changed bounded source subgraph; not an individual instruction proof",
                    dynamic_operations_before=reduced_activities[0].get("dynamic_operations"),
                    dynamic_operations_after=activity.get("dynamic_operations"))
            if extraction.get("mechanism") == "insert_slice":
                delta = record["changed_reduced_mechanism"]["materialization_delta"]
                scalar_counts_preserved = all(
                    reduced_activities[0].get("dynamic_operations", {}).get(category, 0)
                    == activity.get("dynamic_operations", {}).get(category, 0)
                    for category in ("floating_arithmetic", "conversion"))
                insertion_demonstrated = (
                    activity.get("status") == "derived"
                    and delta.get("status") == "changed_reduced_materialization"
                    and scalar_counts_preserved)
                record["emitted_mechanism"].update(
                    status="demonstrated_changed_insert_slice" if insertion_demonstrated
                        else "NOT_DEMONSTRATED",
                    kind="static_insert_slice_without_overwritten_destination_traffic",
                    materialization_delta=delta,
                    floating_arithmetic_and_conversion_counts_preserved=scalar_counts_preserved,
                    geometry=extraction["geometry"])
                if not insertion_demonstrated:
                    raise ValueError("changed reduced insert-slice mechanism not demonstrated")
            if extraction.get("mechanism") == "pointwise_reduction":
                lowered_scalar = {"arith.addf": "llvm.fadd", "arith.subf": "llvm.fsub",
                                  "arith.mulf": "llvm.fmul", "arith.divf": "llvm.fdiv",
                                  "arith.addi": "llvm.add", "arith.subi": "llvm.sub", "arith.muli": "llvm.mul"}
                producer_name = lowered_scalar.get(extraction["producer_result_scalar_operation"])
                reduction_name = lowered_scalar.get(extraction["reduction_result_scalar_operation"])
                direct_edges = sum(1 for op in module.walk() if producer_name and op.name == producer_name
                    for use in op.results[0].uses if use.operation.name == reduction_name
                    and use.operation.parent_block() is op.parent_block())
                scalar_counts_preserved = all(
                    reduced_activities[0].get("dynamic_operations", {}).get(category, 0)
                    == activity.get("dynamic_operations", {}).get(category, 0)
                    for category in ("floating_arithmetic", "conversion"))
                reduction_demonstrated = (
                    activity.get("status") == "derived" and direct_edges == 1 and scalar_counts_preserved
                    and activity.get("static_allocation_payload_bytes") == final_bytes
                    and len(activity.get("allocations", [])) == 1)
                record["emitted_mechanism"].update(
                    status="demonstrated_changed_pointwise_in_reduction" if reduction_demonstrated else "NOT_DEMONSTRATED",
                    kind="pointwise_producer_direct_scalar_use_in_full_domain_reduction",
                    actual_direct_scalar_edges=direct_edges,
                    floating_arithmetic_and_conversion_counts_preserved=scalar_counts_preserved,
                    materialization_delta=record["changed_reduced_mechanism"]["materialization_delta"],
                    source_scalar_region_sha256=extraction["scalar_region_sha256"],
                    reduction_dimensions=extraction["reduction_dimensions"],
                    recomputation_factor=1,
                    allocation_scope="one reduced initialized reduction output; producer tensor removed")
                if not reduction_demonstrated:
                    raise ValueError("changed reduced pointwise-to-reduction scalar mechanism not demonstrated")
            if extraction.get("mechanism") in {"generic_reduction", "named_reduction"}:
                named = extraction.get("mechanism") == "named_reduction"
                output_spills = [key for key, row in buffers.items() if key.startswith("alloca:")
                                 and row.get("load_payload_bytes") == (final_bytes if named else 0)
                                 and row.get("store_payload_bytes") == final_bytes]
                scalar_slots = [row for row in activity.get("allocations", [])
                                if row.get("payload_bytes") == 4 and row.get("declared_payload_bytes") is None]
                scalar_counts_preserved = all(
                    reduced_activities[0].get("dynamic_operations", {}).get(category, 0)
                    == activity.get("dynamic_operations", {}).get(category, 0)
                    for category in ("floating_arithmetic", "conversion"))
                register_reduction_demonstrated = (
                    activity.get("status") == "derived" and len(output_spills) == 1
                    and (bool(scalar_slots) or extraction.get("mechanism") == "named_reduction")
                    and scalar_counts_preserved
                    and len(output_indices) == 1
                    and buffers.get(f"arg:{output_indices[0]}", {}).get("store_payload_bytes") == final_bytes
                    and buffers.get(f"arg:{output_indices[0]}", {}).get("load_payload_bytes") == 0)
                record["emitted_mechanism"].update(
                    status="demonstrated_reduced_register_accumulator" if register_reduction_demonstrated
                        else "NOT_DEMONSTRATED",
                    kind=("named_reduction_result_buffer_replaced_by_ordered_ssa_accumulator" if named
                          else "generic_reduction_result_buffer_replaced_by_ordered_scalar_accumulator"),
                    result_buffer_candidates=output_spills,
                    promotable_scalar_slot_count=len(scalar_slots),
                    floating_arithmetic_and_conversion_counts_preserved=scalar_counts_preserved,
                    materialization_delta=record["changed_reduced_mechanism"]["materialization_delta"],
                    source_scalar_region_sha256=extraction["scalar_region_sha256"],
                    reduction_dimensions=extraction["reduction_dimensions"],
                    scalar_iteration_order="source lexicographic order preserved",
                    ordered_accumulator_representation="ssa" if named else "scalar_slot",
                    machine_scalar_slot_promotion="not_required" if named else "UNPROVEN_BY_PREOPT_CFG",
                    performance="UNMEASURED")
                if not register_reduction_demonstrated:
                    raise ValueError("changed reduced register-accumulator mechanism not demonstrated")
            if extraction.get("mechanism") == "dequant_contraction":
                # Verify the actual reduced artifact computes dequantized values directly into
                # the reduction, not merely another unrelated fused chain in its host task.
                chains = []
                for subtract in module.walk():
                    if subtract.name != "llvm.fsub":
                        continue
                    if not all(getattr(value.owner, "name", None) == "llvm.sitofp"
                               for value in subtract.operands):
                        continue
                    for scale_use in subtract.results[0].uses:
                        scale = scale_use.operation
                        if scale.name != "llvm.fmul":
                            continue
                        for product_use in scale.results[0].uses:
                            product = product_use.operation
                            if product.name != "llvm.fmul":
                                continue
                            for sum_use in product.results[0].uses:
                                addition = sum_use.operation
                                if (addition.name == "llvm.fadd"
                                        and all(op.parent_block() is subtract.parent_block()
                                                for op in (scale, product, addition))
                                        and all(str(op.results[0].type) == "f32"
                                                for op in (subtract, scale, product, addition))):
                                    chains.append((subtract, scale, product, addition))
                contraction_demonstrated = (
                    activity.get("status") == "derived" and len(chains) == 1
                    and activity.get("static_allocation_payload_bytes") == final_bytes
                    and len(activity.get("allocations", [])) == 1
                    and len(output_indices) == 1
                    and buffers.get(f"arg:{output_indices[0]}", {}).get("store_payload_bytes") == final_bytes
                    and buffers.get(f"arg:{output_indices[0]}", {}).get("load_payload_bytes") == 0)
                record["emitted_mechanism"].update(
                    status="demonstrated_reduced_dequant_in_contraction" if contraction_demonstrated else "NOT_DEMONSTRATED",
                    kind="dequantized_operand_direct_scalar_use_in_contraction_without_intermediate",
                    actual_scalar_chain_count=len(chains),
                    exact_scalar_chain="sitofp,sitofp -> fsub -> fmul(scale) -> fmul(other operand) -> fadd(initialized reduction)",
                    scalar_intermediate_spill_between_chain_operations=False if chains else None,
                    allocation_scope="one reduced output accumulator only; no dequant tensor materialization",
                    source_geometry_mkn=extraction["source_geometry_mkn"],
                    probe_geometry_mkn=extraction["probe_geometry_mkn"],
                    recomputation_factor=extraction["recomputation_factor"])
                if not contraction_demonstrated:
                    raise ValueError("selected source dequant-to-contraction mechanism absent from actual reduced emitted artifact")
            fanout = extraction.get("fanout")
            if fanout:
                root_bytes = int(np.prod(fanout["probe_shape"])) * ((int(fanout["root_dtype"][1:])+7)//8)
                allocated = [value for key, value in buffers.items() if key.startswith("alloca:")]
                materialized_once = (
                    activity.get("status") == "derived" and len(allocated) == 1
                    and activity.get("static_allocation_payload_bytes") == root_bytes
                    and allocated[0]["store_payload_bytes"] == root_bytes
                    and allocated[0]["load_payload_bytes"] >= root_bytes * len(fanout["uses"])
                    and activity.get("store_payload_bytes") == root_bytes + final_bytes)
                record["emitted_mechanism"].update(
                    status="demonstrated_reduced_dead_initializer_removal" if materialized_once else "NOT_DEMONSTRATED",
                    kind="materialized_pointwise_fanout_without_dead_initializer_stores",
                    source_materialized_root_payload_bytes=root_bytes,
                    actual_source_root_use_count=len(fanout["uses"]))
            executable = work / "native"
            executable.mkdir()
            lowered_path = executable / "probe.mlir"
            lowered_path.write_text(lowered.stdout)
            llvm_path, library_path = executable / "probe.ll", executable / "probe.so"
            translator = mlir_translate()
            for argv in ([str(translator), "--mlir-to-llvmir", str(lowered_path), "-o", str(llvm_path)],
                         [str(translator.with_name("clang")), "-shared", "-fPIC", "-O2", str(llvm_path), "-o", str(library_path)]):
                result = experiment.run_native_probe(candidate, executable, argv, timeout_s=remaining())
                if result.returncode:
                    raise ValueError("sandboxed native witness build failed: " + (result.stderr or "")[-2000:])
            runner = executable / "runner.py"
            runner.write_bytes(Path(native_host_witness_runner.__file__).read_bytes())
            record.update(probe_lowered_sha256=_sha(lowered.stdout.encode()),
                          probe_command_buffer_sha256=_sha(json.dumps(cb, sort_keys=True).encode()),
                          native_binary_sha256=_sha(library_path.read_bytes()),
                          native_runner_sha256=_sha(runner.read_bytes()),
                          reference_sha256=_sha(Path(__file__).with_name("host_source_witness.py").read_bytes()))
            input_specs = extraction["inputs"]
            if len(abi["args"]) != len(input_specs)+len(output_specs) or len(abi["outputs"]) != len(output_specs):
                raise ValueError("native source witness ABI contains undeclared carrier or intermediate pointers")
            cases = []
            for seed in range(3):
                inputs = []
                for i, spec in enumerate(input_specs):
                    dtype = np.dtype("float32" if spec["dtype"] == "f32" else "int"+spec["dtype"][1:])
                    pool = ([-1.0, -0.5, -2**-24, 0.0, 2**-24, 0.5, 1.0]
                            if spec["dtype"] == "f32" else [-127, -1, 0, 1, 126, 127])
                    data = [pool[(j+i+seed)%len(pool)] for j in range(int(np.prod(spec["shape"])))]
                    inputs.append(np.asarray(data, dtype=dtype).reshape(spec["shape"]))
                expected = evaluate_pointwise_source(probe_text, inputs)
                arguments = []
                input_index = 0
                output_index = 0
                for arg in abi["args"]:
                    spec = cb["tensors"][arg["tensor"]]
                    layout = dict(self.native_layout(spec))
                    if (not 0 < layout["rows"]*layout["cols"] <= 4096
                            or not layout["cols"] <= layout["row_stride"]
                            or not layout["rows"]*layout["row_stride"] <= layout["storage_elements"] <= 1_000_000):
                        raise ValueError("native target ABI layout exceeds the short witness budget")
                    row = {**layout, "dtype": spec["dtype"], "access": arg["access"]}
                    if arg["access"] == "read":
                        wanted = input_specs[input_index]
                        if spec["shape"] != wanted["shape"] or spec["dtype"] != wanted["dtype"]:
                            raise ValueError("candidate pointer ABI differs from source boundary argument order")
                        row["values"] = inputs[input_index].reshape(-1).tolist()
                        input_index += 1
                    else:
                        wanted = output_specs[output_index]
                        if (arg["access"] != "write" or spec["shape"] != wanted["shape"]
                                or spec["dtype"] != wanted["dtype"]
                                or arg["tensor"] != abi["outputs"][output_index]):
                            raise ValueError("source witness has an unsupported output ABI")
                        output_index += 1
                    arguments.append(row)
                request = executable / "inputs.json"
                request.write_text(json.dumps({"library": str(library_path), "symbol": self.expected_symbol,
                                                "arguments": arguments}))
                observed = experiment.run_native_probe(candidate, executable,
                    ["python3", str(runner), str(request)], timeout_s=remaining())
                if observed.returncode:
                    raise ValueError("sandboxed native source witness failed: " + (observed.stderr or "")[-2000:])
                decoded = json.loads(observed.stdout)
                if len(decoded["outputs"]) != len(expected):
                    raise ValueError("native source witness output count differs from source")
                correct_outputs = []
                for output, wanted in zip(decoded["outputs"], expected):
                    actual = np.asarray(output).reshape(wanted.shape)
                    if extraction.get("mechanism") in {
                            "dequant_contraction", "pointwise_reduction", "generic_reduction",
                            "named_reduction", "insert_slice", "pointwise_concat"} and wanted.dtype.kind == "f":
                        correct_outputs.append(actual.astype(np.float32).tobytes() == wanted.tobytes())
                        continue
                    correct_outputs.append(bool(np.allclose(actual, wanted, atol=self.float_atol,
                        rtol=self.float_rtol, equal_nan=False) if wanted.dtype.kind == "f"
                        else np.array_equal(actual, wanted)))
                correct = all(correct_outputs)
                cases.append({"seed": seed, "correct": bool(correct),
                              "correct_outputs": correct_outputs,
                              "inputs_sha256": _sha(request.read_bytes()),
                              "observed_sha256": _sha(observed.stdout.encode()),
                              "expected_sha256": [_sha(value.tobytes()) for value in expected]})
                if not correct:
                    raise ValueError("candidate differs from independent typed source reference")
            if selected_context is not None:
                verified = experiment.selected_changed_portfolio_context(
                    candidate, selected_context["selection"])
                if (verified["current"]["probe_binding"].to_dict() != binding
                        or verified["previous"]["probe_binding"].to_dict()
                        != preceding_binding):
                    raise ValueError("compiler or selected portfolio evidence changed during qualification")
            elif experiment.current_probe_binding(candidate).to_dict() != binding:
                raise ValueError("compiler or whole-model evidence changed during qualification")
            record.update(status="passed_reduced_source_witness", cases=cases,
                          source_mechanism_correspondence="exact scalar regions/maps, reduced extents",
                          float_atol=self.float_atol, float_rtol=self.float_rtol)
            if extraction.get("mechanism") == "dequant_contraction":
                record.update(source_mechanism_correspondence=extraction["scalar_semantics"],
                              float_comparison="exact_f32_bits", float_atol=0, float_rtol=0)
            elif extraction.get("mechanism") == "pointwise_reduction":
                record.update(source_mechanism_correspondence=extraction["scope"],
                              float_comparison="exact_f32_bits", float_atol=0, float_rtol=0)
            elif extraction.get("mechanism") in {"generic_reduction", "named_reduction"}:
                record.update(source_mechanism_correspondence=extraction["scope"],
                              float_comparison="exact_f32_bits", float_atol=0, float_rtol=0)
            elif extraction.get("mechanism") == "bounded_gather":
                record.update(source_mechanism_correspondence=extraction["scope"],
                              geometry=extraction["geometry"])
            elif extraction.get("mechanism") == "pointwise_concat":
                record.update(source_mechanism_correspondence=extraction["scope"],
                              geometry=extraction["geometry"],
                              float_comparison="exact_f32_bits", float_atol=0, float_rtol=0)
            elif extraction.get("mechanism") == "insert_slice":
                record.update(source_mechanism_correspondence=extraction["scope"],
                              geometry=extraction["geometry"],
                              float_comparison="exact_f32_bits", float_atol=0, float_rtol=0)
        except Exception as exc:
            record["reason"] = f"{type(exc).__name__}: {str(exc)[:4000]}"
        finally:
            record["elapsed_seconds"] = monotonic()-started
            (work / "qualification.json").write_text(json.dumps(record, indent=2))
        return record
