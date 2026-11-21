"""Keep a useful local calibration separate from evidence for the compiler's actual change.

An unchanged compute primitive cannot establish a benefit from deleting synchronization, changing
an encoding, or overlapping transfers. This audit compares host-retained emitted instructions with
the admitted probe's actual semantic signature. It does not deny isolated calibration, invent a
composition rule, or stop structural authoring when a relevant context probe is not yet available.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from time import monotonic
from typing import Any

from merlin.analysis.optimization.probes.mechanisms import MechanismSignature
from merlin.analysis.perf.dependencies.reorder_claim import permutation_of
from merlin.analysis.programs.cfg_activity import _category
from merlin.common import digest as _mjson

# ---- probe_relevance ---------------------------------------------------------------------------


def _canonical(value: Any) -> str:
    return _mjson.canonical_json(value).decode("ascii")


def _operand(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    if value.get("kind") == "const":
        return {"kind": "const", "value": value.get("value", value.get("raw"))}
    return {key: item for key, item in value.items() if key != "raw"}


def _atom(row: Mapping[str, Any]) -> str:
    item = {"class": row.get("class", "UNKNOWN")}
    selector = row.get("selector", row.get("funct"))
    if selector is not None:
        item.update(selector=selector, rs1=_operand(row.get("rs1")), rs2=_operand(row.get("rs2")))
    return _canonical(item)


def _signature_atoms(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        if "class" in value:
            return [_atom(value)]
        return [atom for item in value.values() for atom in _signature_atoms(item)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [atom for item in value for atom in _signature_atoms(item)]
    return []


def _bound_rows(analysis: Mapping[str, Any], artifacts: Mapping[str, Any]) -> tuple[str, list[str]]:
    text = artifacts.get("lowered_text")
    if not isinstance(text, str) or not text:
        raise ValueError("probe relevance requires actual retained lowered bytes")
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != artifacts.get("candidate_lowered_sha256") or digest != analysis.get("emission", {}).get(
        "candidate_lowered_sha256"
    ):
        raise ValueError("probe relevance artifacts do not match their whole-model analysis")
    rows = artifacts.get("decoded_trace", {}).get("instructions")
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise ValueError("probe relevance requires a host-decoded instruction stream")
    return digest, [_atom(row) for row in rows]


def classify_probe_relevance(
    *,
    previous_analysis: Mapping[str, Any] | None,
    current_analysis: Mapping[str, Any],
    previous_artifacts: Mapping[str, Any] | None,
    current_artifacts: Mapping[str, Any],
    signature: MechanismSignature,
) -> dict[str, Any]:
    """Report what the measured motif says about the current emitted change, not a user label.

    A class match is only a routing hint. Exact instruction payloads are also reported, but even
    matching payloads do not prove equal live state, neighboring traffic, or repeated scheduling.
    Those require a host-extracted changed-window signature and matched contextual observations.
    Therefore this function never promotes a local sample into a global performance verdict.
    """
    current_digest, current = _bound_rows(current_analysis, current_artifacts)
    facts = signature.to_dict()
    sampled = _signature_atoms(facts["instruction_semantics"])
    sampled_classes = {json.loads(atom)["class"] for atom in sampled}
    events = facts["events"]
    result = {
        "schema": "changed_mechanism_probe_relevance_v1",
        "current_artifact_sha256": current_digest,
        "probe_signature_sha256": signature.digest,
        "isolated_calibration_allowed": True,
        "global_cost_validated": False,
        "changed_context_validated": False,
        "full_model_cycles": None,
        "probe_instruction_classes": sorted(sampled_classes),
        "probe_event_evidence": {
            "event_kinds": sorted({event["kind"] for event in events}),
            "resources": sorted({event["resource"] for event in events}),
            "dependency_edges": sum(len(event["depends_on"]) for event in events),
            "serial_groups": sorted({group for event in events for group in event["serial_group"]}),
        },
    }
    if previous_analysis is None or previous_artifacts is None:
        return {
            **result,
            "status": "calibration_only_no_previous_artifact",
            "unresolved": ["no previous bound emitted artifact to identify the changed mechanism"],
        }
    previous_digest, previous = _bound_rows(previous_analysis, previous_artifacts)
    result["previous_artifact_sha256"] = previous_digest
    before_graph = previous_analysis.get("diagnostics", {}).get("captured_logical_graph", {})
    after_graph = current_analysis.get("diagnostics", {}).get("captured_logical_graph", {})
    if not before_graph.get("logical_dispatch_digest") or before_graph.get(
        "logical_dispatch_digest"
    ) != after_graph.get("logical_dispatch_digest"):
        return {
            **result,
            "status": "unresolved_logical_graph_identity",
            "unresolved": ["the compared artifacts do not bind the same captured logical graph"],
        }

    changes = []
    before_changed: list[str] = []
    after_changed: list[str] = []
    if len(previous) * len(current) > 4_000_000:
        return {
            **result,
            "status": "unresolved_instruction_alignment_budget",
            "unresolved": ["detailed instruction alignment exceeds the bounded host-work budget"],
            "selection": "retain calibration; compare smaller host-owned changed windows",
        }
    for kind, first, last, new_first, new_last in SequenceMatcher(a=previous, b=current, autojunk=False).get_opcodes():
        if kind == "equal":
            continue
        left, right = previous[first:last], current[new_first:new_last]
        before_changed.extend(left)
        after_changed.extend(right)
        changes.append(
            {
                "kind": kind,
                "previous_range": [first, last],
                "current_range": [new_first, new_last],
                "previous_classes": [json.loads(atom)["class"] for atom in left],
                "current_classes": [json.loads(atom)["class"] for atom in right],
            }
        )
    multiset_preserved = Counter(previous) == Counter(current)
    if changes and multiset_preserved:
        # Both endpoints participate in an order reversal. Sequence alignment alone can call
        # one endpoint "unchanged", which would misroute a probe of that interacting endpoint.
        permutation = permutation_of([(atom, {}) for atom in previous], [(atom, {}) for atom in current])
        participants = set()
        for index, left in enumerate(permutation):
            for right in permutation[index + 1 :]:
                if left > right:
                    participants.update((left, right))
        before_changed = [previous[index] for index in sorted(participants)]
        after_changed = list(before_changed)
    changed_classes = {json.loads(atom)["class"] for atom in before_changed + after_changed}
    covered_classes = changed_classes & sampled_classes
    result.update(
        alignment_scope="semantic sequence alignment; identical repeated commands may pair ambiguously",
        instruction_changes=changes,
        changed_instruction_classes=sorted(changed_classes),
        sampled_changed_classes=sorted(covered_classes),
        unsampled_changed_classes=sorted(changed_classes - sampled_classes),
        exact_previous_changed_payloads_in_probe=sum(
            count for atom, count in Counter(before_changed).items() if atom in sampled
        ),
        exact_current_changed_payloads_in_probe=sum(
            count for atom, count in Counter(after_changed).items() if atom in sampled
        ),
        instruction_multiset_preserved=multiset_preserved,
    )
    if not previous or not current or "UNKNOWN" in changed_classes:
        status, missing = "unresolved_decoding", ["complete decoded changed mechanism"]
    elif not changes:
        status = (
            "calibration_only_unchanged_artifact"
            if previous_digest == current_digest
            else "unresolved_host_or_control_flow_change"
        )
        missing = (
            ["no emitted change was observed"]
            if previous_digest == current_digest
            else ["host/control-flow/representation change outside the decoded device instruction stream"]
        )
    elif not covered_classes:
        status, missing = (
            "unrelated_to_changed_instructions",
            [
                "a short probe containing the changed instruction/state mechanism",
                "unchanged local compute timing cannot establish synchronization, encoding, or overlap benefit",
            ],
        )
    else:
        status, missing = (
            "unresolved_changed_context",
            [
                "host extraction of the changed window's dependency/resource/representation signature",
                "matching before/after contextual measurements; a shared instruction class is not equivalence",
                "explicit composition evidence for contention, latency hiding, and repetition",
            ],
        )
    return {
        **result,
        "status": status,
        "unresolved": missing,
        "selection": "continue structural authoring; request only the unresolved changed mechanism",
        "licence": "local calibration is retained; no unrelated global performance claim is admitted",
    }


# ---- completion_delta --------------------------------------------------------------------------
# Automatically qualify a narrow relative completion-deletion transformation.
#
# The target supplies its existing completion contract. The shared proof checks actual emitted LLVM
# and CFG, not a compiler's claim that a wait is redundant. Device completion and host-write visibility
# are separate obligations: distinct tensor names alone never establish physical non-aliasing.


@dataclass(frozen=True)
class CompletionContract:
    identity: str
    provenance: Mapping[str, Any]
    recognizes_decoded: Callable[[Mapping[str, Any]], bool]
    recognizes_operation: Callable[[Any, Mapping[str, Any]], bool]

    def __post_init__(self):
        if not self.identity or not self.provenance:
            raise ValueError("completion contract needs host-owned target provenance")


def _host_category(op):
    category = _category(op.name)
    if category in {"load", "store"}:
        ordering = getattr(getattr(op.properties.get("ordering"), "value", None), "data", 0)
        if ordering != 0 or any("volatile" in key for key in (*op.properties, *op.attributes)):
            return "other"
    return category


def qualify_relative_completion_delta(
    *,
    previous_analysis: Mapping[str, Any] | None,
    current_analysis: Mapping[str, Any],
    previous_artifacts: Mapping[str, Any] | None,
    current_artifacts: Mapping[str, Any],
    contract: CompletionContract,
    timeout_seconds: float = 30,
    max_artifact_bytes: int = 2_000_000,
) -> dict[str, Any]:
    """Reuse retained parsed IR, or parse only a small completion-only changed artifact pair.

    A terminal exit-wrapper deletion can establish only device-completion redundancy. This API
    has no bound caller contract, so it never qualifies that deletion's caller-visible ordering.
    Entry wrappers and nonterminal wrappers remain unsupported.

    This is a host-only diagnostic. It never invokes a compiler or simulator and never mutates a
    retained module. Unsupported changes return a scoped unresolved receipt, not an authoring gate.
    The caller charges elapsed time to the current experiment. The byte cap bounds parsing work;
    cooperative deadline checks also bound the CFG traversal and structural comparison.
    """
    from collections import Counter

    from xdsl.dialects.llvm import LLVM
    from xdsl.irdl.dominance import DominanceInfo

    from merlin.compiler.frontend.linalg_mlir import make_context, parse_mlir_text

    started = monotonic()
    result = {
        "schema": "relative_completion_delta_v1",
        "completion_contract": {"identity": contract.identity, "provenance": dict(contract.provenance)},
        "device_completion_redundancy": "UNRESOLVED",
        "host_visibility": "UNRESOLVED",
        "relative_synchronization_qualified": False,
        "numerical_equivalence": "NOT_ESTABLISHED",
        "performance_improvement": "UNMEASURED",
        "full_model_executed": False,
        "compiler_invoked": False,
        "scope_assumptions": [
            "ordinary captured tensor-buffer ABI",
            "no concurrent external accelerator command issuer",
        ],
        "scope": "relative completion deletion only; inherited arithmetic and target ABI not requalified",
    }

    def finish(status, reason=""):
        return {**result, "status": status, "reason": reason, "elapsed_seconds": monotonic() - started}

    def check_time():
        if monotonic() - started > timeout_seconds:
            raise TimeoutError("bounded completion-delta host analysis exhausted its deadline")

    if timeout_seconds <= 0 or timeout_seconds > 600 or max_artifact_bytes <= 0:
        raise ValueError("completion qualification needs a positive bounded host-work policy")
    if previous_analysis is None or previous_artifacts is None:
        return finish("not_applicable", "no previous bound candidate artifact")
    try:
        before_sha, before_atoms = _bound_rows(previous_analysis, previous_artifacts)
        after_sha, after_atoms = _bound_rows(current_analysis, current_artifacts)
        result.update(previous_artifact_sha256=before_sha, current_artifact_sha256=after_sha)
        if before_sha == after_sha:
            return finish("not_applicable", "emitted bytes are unchanged")
        before_rows = previous_artifacts["decoded_trace"]["instructions"]
        after_rows = current_artifacts["decoded_trace"]["instructions"]
        added = Counter(after_atoms) - Counter(before_atoms)
        removed = Counter(before_atoms) - Counter(after_atoms)
        if added or not removed:
            return finish("not_applicable", "change is not instruction deletion only")
        remaining = removed.copy()
        for row in before_rows:
            key = _atom(row)
            if remaining[key] and contract.recognizes_decoded(row):
                remaining[key] -= 1
        if +remaining:
            return finish("not_applicable", "deleted instructions include non-completion mechanisms")
        before_plan = previous_analysis.get("diagnostics", {}).get("verified_global_plan_emission", {})
        after_plan = current_analysis.get("diagnostics", {}).get("verified_global_plan_emission", {})
        plans_verified = all(
            plan.get("status") == "verified"
            and plan.get("candidate_lowered_sha256") == digest
            and plan.get("candidate_command_buffer_sha256") == artifacts.get("candidate_command_buffer_sha256")
            for plan, digest, artifacts in (
                (before_plan, before_sha, previous_artifacts),
                (after_plan, after_sha, current_artifacts),
            )
        )
        plans_verified = (
            plans_verified
            and bool(before_plan.get("source_sha256"))
            and (before_plan.get("source_sha256") == after_plan.get("source_sha256"))
        )
        result["verified_plan_abi_contract"] = {"bound": plans_verified, "before": before_plan, "after": after_plan}
        if not plans_verified:
            return finish("unresolved", "before/after verified plan and ABI contract is not bound")

        def module(artifacts):
            retained = artifacts.get("parsed_lowered_module")
            if retained is not None:
                return retained
            text = artifacts["lowered_text"]
            if len(text.encode()) > max_artifact_bytes:
                raise ValueError("artifact exceeds bounded parsing policy; retain parsed IR in the worker")
            context = make_context()
            context.load_dialect(LLVM)
            parsed = parse_mlir_text(text, context)
            check_time()
            return parsed

        before = module(previous_artifacts).clone()
        after = module(current_artifacts)
        before_commands = [op for op in before.walk() if op.name == "llvm.inline_asm"]
        after_commands = [op for op in after.walk() if op.name == "llvm.inline_asm"]
        if len(before_commands) != len(before_rows) or len(after_commands) != len(after_rows):
            raise ValueError("decoded command order does not cover the actual inline-assembly operations")

        def owner(op):
            value = getattr(getattr(op.attributes.get("merlin.global_task"), "value", None), "data", None)
            if not isinstance(value, int):
                raise ValueError("actual command has no global-task ownership")
            return value

        def key(row, op):
            return owner(op), _atom(row)

        functions = [op for op in before.body.block.ops if op.name == "llvm.func"]
        if len(functions) != 1:
            raise ValueError("completion proof requires one emitted kernel function")
        dominance = DominanceInfo(functions[0].body)
        returns = [op for op in functions[0].walk() if op.name == "llvm.return"]
        new_keys = [key(row, op) for row, op in zip(after_rows, after_commands)]
        command_rows = dict(zip(before_commands, before_rows))
        removed_ops = []
        relied_on_completions = []
        details = []
        cursor = 0
        for index, (row, op) in enumerate(zip(before_rows, before_commands)):
            check_time()
            if cursor < len(new_keys) and key(row, op) == new_keys[cursor]:
                cursor += 1
                continue
            if not contract.recognizes_operation(op, row) or index == 0:
                raise ValueError("relative change removes a non-contracted or wrapper operation")
            exit_wrapper = owner(op) == -2
            if owner(op) < 0 and not (
                exit_wrapper
                and index == len(before_commands) - 1
                and len(returns) == 1
                and op.next_op is returns[0]
                and returns[0].next_op is None
                and not returns[0].successors
            ):
                raise ValueError("deleted wrapper is not the terminal exit completion before the sole return")
            completion = before_commands[index - 1]
            if not contract.recognizes_operation(completion, before_rows[index - 1]):
                raise ValueError("no retained completion immediately precedes the deleted command")
            if not dominance.dominates(completion.parent, op.parent):
                raise ValueError("retained completion does not dominate the deleted operation")
            pending, seen, reached = [completion.next_op], set(), False
            host_categories = set()
            while pending:
                check_time()
                current = pending.pop()
                if current is None or current in seen:
                    continue
                if current is op:
                    reached = True
                    continue
                seen.add(current)
                category = _host_category(current)
                if category in {"other", "opaque_inline_asm"}:
                    raise ValueError("intervening CFG path has unknown or device-producing effects")
                host_categories.add(category)
                pending.extend(
                    [current.next_op]
                    if current.next_op is not None
                    else [block.first_op for block in current.successors]
                )
            if not reached:
                raise ValueError("deleted completion is not reachable from retained completion")
            removed_ops.append(op)
            relied_on_completions.append(completion)
            details.append(
                {
                    "previous_instruction_index": index,
                    "task_index": owner(op),
                    "deletion_scope": "terminal_exit_wrapper" if exit_wrapper else "internal_completion",
                    "retained_completion_index": index - 1,
                    "retained_completion_dominates": True,
                    "intervening_host_categories": sorted(host_categories),
                    "no_intervening_device_issue_on_any_cfg_path": True,
                }
            )
        if cursor != len(new_keys) or len(removed_ops) != sum(removed.values()):
            raise ValueError("before/after commands do not differ only by the identified deletions")
        removed_set = set(removed_ops)
        if removed_set.intersection(relied_on_completions):
            raise ValueError("completion proof relies on an operation also deleted by the candidate")
        for op, detail in zip(removed_ops, details):
            if detail["deletion_scope"] == "terminal_exit_wrapper":
                # Return ends this kernel's CFG, not the caller's memory-ordering obligations.
                # A renderer template or candidate annotation is not bound caller evidence.
                detail.update(
                    host_visibility="UNRESOLVED",
                    caller_visible_ordering="UNKNOWN",
                    visibility_reason="terminal exit deletion has no bound caller ordering contract",
                )
                result["caller_visible_ordering"] = "UNKNOWN"
                continue
            # A second fence also ordered host writes. Check the resulting program, skipping all
            # deleted operations; a barrier being deleted elsewhere cannot establish visibility.
            pending, seen, visibility = [op.next_op], set(), True
            while pending:
                check_time()
                current = pending.pop()
                if current is None or current in seen:
                    continue
                seen.add(current)
                if current in command_rows and current not in removed_set:
                    if not contract.recognizes_operation(current, command_rows[current]):
                        visibility = False
                    continue
                if current not in removed_set and _host_category(current) in {"other", "opaque_inline_asm"}:
                    visibility = False
                    continue
                pending.extend(
                    [current.next_op]
                    if current.next_op is not None
                    else [block.first_op for block in current.successors]
                )
            detail.update(
                host_visibility="verified_by_retained_barrier" if visibility else "UNRESOLVED",
                visibility_reason="retained barrier or return precedes later device issue"
                if visibility
                else "later device issue lacks intervening barrier; concrete alias independence is not proved",
            )
        for op in removed_ops:
            op.parent.erase_op(op)
        if not before.is_structurally_equivalent(after):
            raise ValueError("complete LLVM differs beyond identified completion deletions")
        check_time()
        result.update(
            deleted_completions=details,
            complete_ir_equal_after_only_identified_deletions=True,
            all_other_memory_ordering_operations_preserved=True,
            device_completion_redundancy="verified",
        )
        visible = all(row["host_visibility"] != "UNRESOLVED" for row in details)
        result["host_visibility"] = "verified" if visible else "UNRESOLVED"
        result["relative_synchronization_qualified"] = visible
        return finish(
            "relative_synchronization_qualified" if visible else "conditional_device_completion_delta_verified",
            "only the relative synchronization transformation is qualified"
            if visible
            else "device completion redundancy is proved; caller-visible exit ordering remains unknown"
            if "caller_visible_ordering" in result
            else "device completion redundancy is proved; missing concrete alias independence prevents full host-visibility qualification",
        )
    except Exception as error:  # a diagnostic proof failure never turns into implicit qualification
        return finish("unresolved", f"{type(error).__name__}: {error}")
