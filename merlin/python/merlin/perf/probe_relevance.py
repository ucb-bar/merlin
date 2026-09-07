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
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

from .mechanism_probe import MechanismSignature
from .reorder_claim import permutation_of


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    if (digest != artifacts.get("candidate_lowered_sha256")
            or digest != analysis.get("emission", {}).get("candidate_lowered_sha256")):
        raise ValueError("probe relevance artifacts do not match their whole-model analysis")
    rows = artifacts.get("decoded_trace", {}).get("instructions")
    if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
            or any(not isinstance(row, Mapping) for row in rows)):
        raise ValueError("probe relevance requires a host-decoded instruction stream")
    return digest, [_atom(row) for row in rows]


def classify_probe_relevance(*, previous_analysis: Mapping[str, Any] | None,
                             current_analysis: Mapping[str, Any],
                             previous_artifacts: Mapping[str, Any] | None,
                             current_artifacts: Mapping[str, Any],
                             signature: MechanismSignature) -> dict[str, Any]:
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
        return {**result, "status": "calibration_only_no_previous_artifact",
                "unresolved": ["no previous bound emitted artifact to identify the changed mechanism"]}
    previous_digest, previous = _bound_rows(previous_analysis, previous_artifacts)
    result["previous_artifact_sha256"] = previous_digest
    before_graph = previous_analysis.get("diagnostics", {}).get("captured_logical_graph", {})
    after_graph = current_analysis.get("diagnostics", {}).get("captured_logical_graph", {})
    if (not before_graph.get("logical_dispatch_digest")
            or before_graph.get("logical_dispatch_digest") != after_graph.get("logical_dispatch_digest")):
        return {**result, "status": "unresolved_logical_graph_identity",
                "unresolved": ["the compared artifacts do not bind the same captured logical graph"]}

    changes = []
    before_changed: list[str] = []
    after_changed: list[str] = []
    if len(previous) * len(current) > 4_000_000:
        return {**result, "status": "unresolved_instruction_alignment_budget",
                "unresolved": ["detailed instruction alignment exceeds the bounded host-work budget"],
                "selection": "retain calibration; compare smaller host-owned changed windows"}
    for kind, first, last, new_first, new_last in SequenceMatcher(
            a=previous, b=current, autojunk=False).get_opcodes():
        if kind == "equal":
            continue
        left, right = previous[first:last], current[new_first:new_last]
        before_changed.extend(left)
        after_changed.extend(right)
        changes.append({"kind": kind, "previous_range": [first, last],
                        "current_range": [new_first, new_last],
                        "previous_classes": [json.loads(atom)["class"] for atom in left],
                        "current_classes": [json.loads(atom)["class"] for atom in right]})
    multiset_preserved = Counter(previous) == Counter(current)
    if changes and multiset_preserved:
        # Both endpoints participate in an order reversal. Sequence alignment alone can call
        # one endpoint "unchanged", which would misroute a probe of that interacting endpoint.
        permutation = permutation_of([(atom, {}) for atom in previous],
                                     [(atom, {}) for atom in current])
        participants = set()
        for index, left in enumerate(permutation):
            for right in permutation[index + 1:]:
                if left > right:
                    participants.update((left, right))
        before_changed = [previous[index] for index in sorted(participants)]
        after_changed = list(before_changed)
    changed_classes = {json.loads(atom)["class"] for atom in before_changed + after_changed}
    covered_classes = changed_classes & sampled_classes
    result.update(
        alignment_scope="semantic sequence alignment; identical repeated commands may pair ambiguously",
        instruction_changes=changes, changed_instruction_classes=sorted(changed_classes),
        sampled_changed_classes=sorted(covered_classes),
        unsampled_changed_classes=sorted(changed_classes - sampled_classes),
        exact_previous_changed_payloads_in_probe=sum(
            count for atom, count in Counter(before_changed).items() if atom in sampled),
        exact_current_changed_payloads_in_probe=sum(
            count for atom, count in Counter(after_changed).items() if atom in sampled),
        instruction_multiset_preserved=multiset_preserved,
    )
    if not previous or not current or "UNKNOWN" in changed_classes:
        status, missing = "unresolved_decoding", ["complete decoded changed mechanism"]
    elif not changes:
        status = ("calibration_only_unchanged_artifact" if previous_digest == current_digest
                  else "unresolved_host_or_control_flow_change")
        missing = (["no emitted change was observed"] if previous_digest == current_digest else
                   ["host/control-flow/representation change outside the decoded device instruction stream"])
    elif not covered_classes:
        status, missing = "unrelated_to_changed_instructions", [
            "a short probe containing the changed instruction/state mechanism",
            "unchanged local compute timing cannot establish synchronization, encoding, or overlap benefit"]
    else:
        status, missing = "unresolved_changed_context", [
            "host extraction of the changed window's dependency/resource/representation signature",
            "matching before/after contextual measurements; a shared instruction class is not equivalence",
            "explicit composition evidence for contention, latency hiding, and repetition"]
    return {**result, "status": status, "unresolved": missing,
            "selection": "continue structural authoring; request only the unresolved changed mechanism",
            "licence": "local calibration is retained; no unrelated global performance claim is admitted"}
