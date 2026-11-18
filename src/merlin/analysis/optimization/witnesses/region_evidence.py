"""Host-owned reduced source-chain numerical qualification for a global compiler edit.

No simulation, candidate Python import, or native candidate execution in the host process.
The existing probes compiler/native sandbox hooks own every submitted executable.
The receipt does not upgrade a reduced check into a full-shape equivalence theorem.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping

from merlin.analysis.optimization.witnesses.gathers import (
    bf16_bits,
    bf16_values,
    extract_bounded_gather,
    extract_insert_slice,
)
from merlin.analysis.optimization.witnesses.pointwise_chains import (
    extract_generic_reduction,
    extract_named_reduction,
    extract_pointwise_chain,
)
from merlin.analysis.optimization.witnesses.pointwise_views import (
    extract_pointwise_concat,
    extract_pointwise_view_chain,
)
from merlin.analysis.optimization.witnesses.reductions import (
    evaluate_pointwise_source,
    extract_dequant_contraction,
    extract_pointwise_reduction,
)
from merlin.common import digest as _mdigest
from merlin.common.paths import module_source_path

# ---- host_region_qualifier ---------------------------------------------------------------------

_sha = _mdigest.sha256_bytes


def cached_host_task_activity(plan: Mapping[str, Any], *, lowered_sha256: str) -> dict[str, Mapping[str, Any]]:
    """Use complete host-retained analysis, never reparse a full model in a probe.

    The controller supplies its verified immutable analysis record. Artifact
    binding and completeness are required; legacy top-N reports fail closed.
    """
    activity = plan.get("host_activity") or {}
    rows = activity.get("tasks")
    if (
        plan.get("status") != "verified"
        or not lowered_sha256
        or plan.get("candidate_lowered_sha256") != lowered_sha256
        or activity.get("artifact_sha256") != lowered_sha256
        or activity.get("status") != "derived"
        or activity.get("task_activity_coverage") != "complete"
        or not isinstance(rows, list)
    ):
        raise ValueError("complete artifact-bound cached host task activity unavailable")
    table = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("task"), str) or row["task"] in table:
            raise ValueError("cached host task activity has ambiguous task rows")
        if any(type(row.get(key)) is not int or row[key] < 0 for key in ("load_payload_bytes", "store_payload_bytes")):
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
        if (
            not prior
            or any(old["kind"] != "host" for old in prior)
            or set(covered) != wanted
            or len(set(covered)) != len(covered)
            or any(str(old["task_index"]) not in tables[0] for old in prior)
            or current_key not in tables[1]
        ):
            unresolved.append(
                {
                    "task_index": task["task_index"],
                    "source_op_indices": task["source_op_indices"],
                    "reason": "same-source host task accounting unavailable across revisions",
                }
            )
            continue
        row_groups = [[tables[0][str(old["task_index"])] for old in prior], [tables[1][current_key]]]
        payload = [
            sum(row[name] for row in group for name in ("load_payload_bytes", "store_payload_bytes"))
            for group in row_groups
        ]
        integer = []
        for group in row_groups:
            values = [(row.get("dynamic_operations") or {}).get("integer_arithmetic") for row in group]
            integer.append(sum(values) if all(type(value) is int and value >= 0 for value in values) else None)
        integer_delta = (
            abs(integer[1] - integer[0]) if all(type(value) is int and value >= 0 for value in integer) else None
        )
        if payload[0] != payload[1] or (integer_delta is not None and integer_delta > 0):
            # Lexicographic priority, not addition of bytes and operations.
            selected = {**task, "comparison_prior_task_indices": [old["task_index"] for old in prior]}
            candidates.append(
                ((abs(payload[1] - payload[0]), integer_delta is not None, integer_delta), selected, payload)
            )
    return candidates, unresolved


_SOURCE_WITNESS_KINDS = (
    "pointwise_view_chain",
    "pointwise_concat",
    "insert_slice",
    "bounded_gather",
    "named_reduction",
    "generic_reduction",
    "pointwise_reduction",
    "dequant_contraction",
    "fanout",
    "chain",
)


def _bounded_source_witness_options(candidates, *, max_ranked_tasks: int = 1):
    """Bound expensive source parses to the highest-value changed host task.

    Each option reparses the complete source module today.  Trying every mechanism for every
    changed task can therefore outlive the short-probe budget without producing any evidence.
    The full-model accounting already ranks tasks by separately reported payload and integer-work
    deltas, so a failed top task is recorded as an explicit qualification refusal.  A later work
    order can bind another task instead of turning one action into an unbounded search.
    """
    if type(max_ranked_tasks) is not int or max_ranked_tasks < 1:
        raise ValueError("source witness selection requires a positive ranked-task budget")
    ranked = sorted(candidates, key=lambda candidate: candidate[0], reverse=True)
    return ((kind, row) for row in ranked[:max_ranked_tasks] for kind in _SOURCE_WITNESS_KINDS)


def pointwise_concat_emission_evidence(
    extraction: Mapping[str, Any],
    before_activity: Mapping[str, Any],
    after_activity: Mapping[str, Any],
    *,
    emitted_operation_names,
    direct_output: bool,
    materialization_delta: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the reduced artifact computes the pointwise segment into concat output.

    Numerical equivalence is established separately by the independent source evaluator.  This
    check binds that result to the emitted mechanism: the source result scalar remains present,
    the producer tensor's exact payload disappears, and the reduced artifact writes directly to
    its sole result without another allocation.
    """
    lowered_scalar = {
        "arith.negf": "llvm.fneg",
        "arith.addf": "llvm.fadd",
        "arith.subf": "llvm.fsub",
        "arith.mulf": "llvm.fmul",
        "arith.divf": "llvm.fdiv",
        "arith.addi": "llvm.add",
        "arith.subi": "llvm.sub",
        "arith.muli": "llvm.mul",
        "arith.extsi": "llvm.sext",
        "arith.extui": "llvm.zext",
        "arith.trunci": "llvm.trunc",
        "arith.sitofp": "llvm.sitofp",
        "arith.uitofp": "llvm.uitofp",
        "arith.fptosi": "llvm.fptosi",
        "arith.fptoui": "llvm.fptoui",
    }
    source_scalar = extraction.get("producer_result_scalar_operation")
    emitted_scalar = lowered_scalar.get(source_scalar)
    names = list(emitted_operation_names)
    actual_scalar_operations = names.count(emitted_scalar) if emitted_scalar else 0
    counts_preserved = all(arm.get("status") == "derived" for arm in (before_activity, after_activity)) and all(
        (before_activity.get("dynamic_operations") or {}).get(category)
        == (after_activity.get("dynamic_operations") or {}).get(category)
        for category in ("floating_arithmetic", "conversion")
    )
    deleted = materialization_delta.get("deleted_payload_bytes") or {}
    expected = extraction.get("probe_intermediate_payload_bytes")
    operand_shapes = extraction.get("probe_operand_shapes")
    operand_count = extraction.get("concat_operand_count")
    result_shape = extraction.get("probe_result_shape")
    identity_contract = operand_count != 1 or (
        isinstance(operand_shapes, list)
        and len(operand_shapes) == 1
        and extraction.get("identity_concat") is True
        and operand_shapes[0] == result_shape
    )
    source_contract = (
        extraction.get("schema") == "actual_source_pointwise_concat_witness_v1"
        and extraction.get("mechanism") == "pointwise_concat"
        and extraction.get("all_source_producer_uses_preserved") is True
        and type(extraction.get("concat_axis")) is int
        and isinstance(operand_shapes, list)
        and isinstance(result_shape, list)
        and type(operand_count) is int
        and operand_count >= 1
        and operand_count == len(operand_shapes)
        and identity_contract
    )
    exact_deletion = (
        materialization_delta.get("status") == "changed_reduced_materialization"
        and type(expected) is int
        and expected > 0
        and deleted.get("static_allocation_payload_bytes") == expected
        and type(deleted.get("load_payload_bytes")) is int
        and deleted["load_payload_bytes"] >= expected
        and type(deleted.get("store_payload_bytes")) is int
        and deleted["store_payload_bytes"] >= expected
    )
    demonstrated = bool(
        source_contract
        and direct_output
        and emitted_scalar
        and actual_scalar_operations == 1
        and counts_preserved
        and exact_deletion
    )
    return {
        "status": ("demonstrated_changed_pointwise_in_concat" if demonstrated else "NOT_DEMONSTRATED"),
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
        "all_source_producer_uses_preserved": extraction.get("all_source_producer_uses_preserved"),
        "performance": "UNMEASURED",
    }


def pointwise_view_chain_emission_evidence(
    extraction: Mapping[str, Any],
    before_activity: Mapping[str, Any],
    after_activity: Mapping[str, Any],
    *,
    emitted_operation_names,
    direct_output: bool,
    materialization_delta: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind exact reduced semantics to deletion across a source-cloned view chain."""
    lowered_scalar = {
        "arith.negf": "llvm.fneg",
        "arith.addf": "llvm.fadd",
        "arith.subf": "llvm.fsub",
        "arith.mulf": "llvm.fmul",
        "arith.divf": "llvm.fdiv",
        "arith.addi": "llvm.add",
        "arith.subi": "llvm.sub",
        "arith.muli": "llvm.mul",
        "arith.extsi": "llvm.sext",
        "arith.extui": "llvm.zext",
        "arith.trunci": "llvm.trunc",
        "arith.sitofp": "llvm.sitofp",
        "arith.uitofp": "llvm.uitofp",
        "arith.fptosi": "llvm.fptosi",
        "arith.fptoui": "llvm.fptoui",
    }
    source_scalar = extraction.get("producer_result_scalar_operation")
    emitted_scalar = lowered_scalar.get(source_scalar)
    names = list(emitted_operation_names)
    activities_derived = all(arm.get("status") == "derived" for arm in (before_activity, after_activity))
    operation_counts = {
        category: [(arm.get("dynamic_operations") or {}).get(category) for arm in (before_activity, after_activity)]
        for category in ("floating_arithmetic", "conversion")
    }
    expected = extraction.get("probe_intermediate_payload_bytes")
    deleted = materialization_delta.get("deleted_payload_bytes") or {}
    source_contract = (
        extraction.get("schema") == "actual_source_pointwise_view_chain_witness_v1"
        and extraction.get("mechanism") == "pointwise_view_chain"
        and extraction.get("all_source_producer_uses_preserved") is True
        and extraction.get("all_source_view_uses_preserved") is True
        and isinstance(extraction.get("view_operations"), list)
        and bool(extraction["view_operations"])
        and set(extraction["view_operations"]) <= {"tensor.collapse_shape", "tensor.expand_shape"}
    )
    exact_deletion = (
        materialization_delta.get("status") == "changed_reduced_materialization"
        and type(expected) is int
        and expected > 0
        and deleted.get("static_allocation_payload_bytes") == expected
        and type(deleted.get("load_payload_bytes")) is int
        and deleted["load_payload_bytes"] >= expected
        and type(deleted.get("store_payload_bytes")) is int
        and deleted["store_payload_bytes"] >= expected
    )
    demonstrated = bool(
        source_contract
        and activities_derived
        and direct_output
        and emitted_scalar
        and names.count(emitted_scalar) == 1
        and exact_deletion
    )
    return {
        "status": ("demonstrated_changed_pointwise_through_view_chain" if demonstrated else "NOT_DEMONSTRATED"),
        "kind": "pointwise_producer_direct_scalar_store_through_static_view_chain",
        "source_result_scalar_operation": source_scalar,
        "emitted_result_scalar_operation": emitted_scalar,
        "actual_result_scalar_operations": names.count(emitted_scalar) if emitted_scalar else 0,
        "dynamic_floating_arithmetic_and_conversion_before_after": operation_counts,
        "aggregate_conversion_reduction_allowed_for_deleted_view_addressing": True,
        "direct_output_without_intermediate_allocation": bool(direct_output),
        "materialization_delta": dict(materialization_delta),
        "source_scalar_region_sha256": extraction.get("scalar_region_sha256"),
        "view_operations": extraction.get("view_operations"),
        "source_view_shapes": extraction.get("source_view_shapes"),
        "probe_view_shapes": extraction.get("probe_view_shapes"),
        "all_source_producer_uses_preserved": extraction.get("all_source_producer_uses_preserved"),
        "all_source_view_uses_preserved": extraction.get("all_source_view_uses_preserved"),
        "performance": "UNMEASURED",
    }


def reduced_materialization_change(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    extraction: Mapping[str, Any],
    *,
    before_lowered_sha256: str | None = None,
    after_lowered_sha256: str | None = None,
) -> dict[str, Any]:
    """An unchanged supported mechanism is not evidence for a different compiler edit.

    Most supported fusions delete a tensor materialization. Direct generic and named reductions
    instead move the accumulator from the result buffer to an ordered scalar recurrence. Their
    pre-optimization total traffic can therefore grow even though repeated result-buffer
    addressing disappeared; report that distinction rather than pretending it is deletion.
    """
    fields = ("static_allocation_payload_bytes", "load_payload_bytes", "store_payload_bytes")
    if any(
        arm.get("status") != "derived" or any(type(arm.get(key)) is not int for key in fields)
        for arm in (before, after)
    ):
        return {"status": "UNKNOWN", "reason": "reduced before/after memory activity is not derived"}
    deleted = {key: before[key] - after[key] for key in fields}
    expected = extraction.get("probe_intermediate_payload_bytes")
    mechanism = extraction.get("mechanism")
    if mechanism in {"generic_reduction", "named_reduction"}:
        output_bytes = extraction.get("probe_output_payload_bytes")
        steps = extraction.get("probe_reduction_steps_per_output")
        before_buffers = before.get("buffer_payload") or {}
        after_buffers = after.get("buffer_payload") or {}
        before_accumulator = (
            any(
                key.startswith("alloca:")
                and row.get("load_payload_bytes", 0) >= output_bytes * steps
                and row.get("store_payload_bytes", 0) >= output_bytes * steps
                for key, row in before_buffers.items()
            )
            if type(output_bytes) is int and type(steps) is int
            else False
        )
        after_result_loads = output_bytes if mechanism == "named_reduction" else 0
        after_result = (
            any(
                key.startswith("alloca:")
                and row.get("load_payload_bytes") == after_result_loads
                and row.get("store_payload_bytes") == output_bytes
                for key, row in after_buffers.items()
            )
            if type(output_bytes) is int
            else False
        )
        scalar_counts_preserved = all(
            (before.get("dynamic_operations") or {}).get(category, 0)
            == (after.get("dynamic_operations") or {}).get(category, 0)
            for category in ("floating_arithmetic", "conversion")
        )
        lowered_changed = (
            bool(before_lowered_sha256) and bool(after_lowered_sha256) and before_lowered_sha256 != after_lowered_sha256
        )
        changed = (
            type(output_bytes) is int
            and output_bytes > 0
            and type(steps) is int
            and steps > 0
            and lowered_changed
            and before_accumulator
            and after_result
            and scalar_counts_preserved
        )
        return {
            "status": "changed_reduced_codegen" if changed else "NO_RELEVANT_REDUCED_CHANGE",
            "deleted_payload_bytes": deleted,
            "result_buffer_accumulator_loads_removed": bool(before_accumulator and after_result),
            "floating_arithmetic_and_conversion_counts_preserved": scalar_counts_preserved,
            "lowered_artifact_changed": lowered_changed,
            "total_preoptimization_payload_reduced": all(value >= 0 for value in deleted.values())
            and any(value > 0 for value in deleted.values()),
            "expected_output_payload_bytes": output_bytes,
            "reduction_steps_per_output": steps,
            "final_result_buffer_load_bytes": after_result_loads,
            "ordered_accumulator_representation": "ssa" if mechanism == "named_reduction" else "scalar_slot",
            "machine_scalar_slot_promotion": "UNPROVEN_BY_PREOPT_CFG",
            "scope": f"same cloned {mechanism.replace('_', ' ')} under preceding/current compilers; proves "
            "a result-buffer-to-scalar-accumulator codegen change, not machine traffic or performance",
        }
    if mechanism == "insert_slice":
        overwritten = extraction.get("probe_inserted_payload_bytes")
        changed = (
            type(overwritten) is int
            and overwritten > 0
            and deleted["static_allocation_payload_bytes"] == 0
            and 0 <= deleted["load_payload_bytes"] <= overwritten
            and deleted["store_payload_bytes"] == overwritten
        )
        return {
            "status": "changed_reduced_materialization" if changed else "NO_RELEVANT_REDUCED_CHANGE",
            "deleted_payload_bytes": deleted,
            "deleted_overwritten_payload_bytes": overwritten,
            "deleted_destination_load_bytes": deleted["load_payload_bytes"],
            "scope": "same cloned insert slice under preceding/current compilers; proves "
            "one source-box worth of overwritten destination traffic was removed",
        }
    if mechanism in {"dequant_contraction", "pointwise_reduction", "pointwise_concat", "pointwise_view_chain"}:
        changed = (
            type(expected) is int
            and expected > 0
            and deleted["static_allocation_payload_bytes"] == expected
            and deleted["store_payload_bytes"] >= expected
            and deleted["load_payload_bytes"] >= expected
        )
    else:
        changed = any(value > 0 for value in deleted.values()) and all(value >= 0 for value in deleted.values())
    if (
        not changed
        and mechanism in {"chain", "fanout", "bounded_gather"}
        and all(value == 0 for value in deleted.values())
    ):
        operations = [arm.get("dynamic_operations") for arm in (before, after)]
        if all(isinstance(row, Mapping) for row in operations):
            integer = [row.get("integer_arithmetic") for row in operations]
            unchanged_other = all(
                operations[0].get(key, 0) == operations[1].get(key, 0) for key in ("floating_arithmetic", "conversion")
            )
            if (
                all(type(value) is int and value >= 0 for value in integer)
                and integer[0] != integer[1]
                and unchanged_other
                and before_lowered_sha256
                and after_lowered_sha256
                and before_lowered_sha256 != after_lowered_sha256
            ):
                return {
                    "status": "changed_reduced_integer_codegen",
                    "deleted_payload_bytes": deleted,
                    "integer_operations_before_after": integer,
                    "scope": "same cloned source under both compilers changes preoptimization integer work; "
                    "may be address or data arithmetic, not machine work or a performance verdict",
                }
    return {
        "status": "changed_reduced_materialization" if changed else "NO_RELEVANT_REDUCED_CHANGE",
        "deleted_payload_bytes": deleted,
        "expected_producer_payload_bytes": expected,
        "scope": "same actual source witness compiled by preceding and current bound compilers; "
        "not full-shape changed-chain attribution or performance",
    }
