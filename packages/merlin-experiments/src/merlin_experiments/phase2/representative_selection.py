"""Conservative, unmeasured Phase 2 development-subset selection.

The unit is a whole performance family: predictive fit points, paired arms and
negative controls must never be separated by a shape-only set cover.  This is
an iteration-cost choice, not a performance, source-equivalence or coverage
certificate.  Omitted levers and connected source obligations remain visible.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping

from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2.contracts import StageGateError

SCHEMA = "merlin.phase2.representative_selection.v1"


def derive(test_justification: Mapping, selected_scope: Mapping | None) -> dict:
    """Select a deterministic family-closed diagnostic subset from a full receipt."""
    if test_justification.get("schema") not in {
        "merlin.phase2.test_justification.v1",
        "merlin.phase2.test_justification.v2",
    }:
        raise StageGateError("representative selection requires the full Phase 2 test justification")
    members = test_justification.get("members")
    if not isinstance(members, list) or not members:
        raise StageGateError("representative selection has no generated members")
    by_family: dict[str, list[Mapping]] = defaultdict(list)
    for row in members:
        if not isinstance(row, Mapping) or not isinstance(row.get("family"), str):
            raise StageGateError("representative selection has a malformed member")
        by_family[row["family"]].append(row)
    keys = [(row.get("family"), row.get("capsule")) for row in members]
    if len(set(keys)) != len(keys):
        raise StageGateError("representative selection has duplicate members")

    axes_by_family: dict[str, set[tuple[str, str]]] = {}
    all_axes: set[tuple[str, str]] = set()
    refused: list[dict] = []
    for family, rows in sorted(by_family.items()):
        axes: set[tuple[str, str]] = set()
        for row in rows:
            purpose = row.get("purpose") or {}
            for field in ("lever", "level", "claim"):
                if purpose.get(field):
                    axes.add((field, str(purpose[field])))
            for trait in (row.get("hardware_basis") or {}).get("traits") or {}:
                axes.add(("rtl_trait", str(trait)))
            if test_justification.get("schema") == "merlin.phase2.test_justification.v2":
                source_basis = (row.get("source_execution_basis") or {}).get("selection") or {}
                for operation in source_basis.get("operations") or []:
                    axes.add(("source_operation", str(operation)))
                boundary = (row.get("curriculum") or {}).get("resource_boundary")
                if isinstance(boundary, Mapping):
                    axes.add(("source_resource_boundary", json.dumps(boundary, sort_keys=True, separators=(",", ":"))))
            for match in (row.get("workload_need") or {}).get("matched_demands") or []:
                if (match.get("form_match") or {}).get("status") == "exact_arithmetic_form":
                    axes.add(("exact_form", f"{match.get('application')}:{match.get('signature_index')}"))
            form = (row.get("workload_need") or {}).get("capsule_form")
            if isinstance(form, Mapping) and form.get("operation") and form.get("inputs"):
                axes.add(("capsule_form", json.dumps(form, sort_keys=True, separators=(",", ":"))))
        all_axes.update(axes)
        # Do not select an incomplete comparator and imply the missing control
        # was measured.  Every selected family retains its entire fit/arm grid.
        if any((row.get("matched_comparator") or {}).get("matching_status") == "incomplete_group" for row in rows):
            refused.append({"family": family, "reason": "incomplete_comparison_group"})
            continue
        if not axes:
            refused.append({"family": family, "reason": "no_declared_selection_axis"})
            continue
        axes_by_family[family] = axes

    uncovered = set(all_axes)
    selected: list[str] = []
    while uncovered:
        remaining = sorted(set(axes_by_family) - set(selected))
        if not remaining:
            break
        # Stable greedy selection, preferring fewer members on equal gain.  It
        # does not assert mathematical minimum-cardinality set cover.
        family = min(
            remaining,
            key=lambda name: (-len(axes_by_family[name] & uncovered), len(by_family[name]), name),
        )
        gain = axes_by_family[family] & uncovered
        if not gain:
            break
        selected.append(family)
        uncovered -= gain
    if not selected:
        raise StageGateError("no complete family covers a selected exact-form or RTL-trait axis")

    selected_set = set(selected)
    selected_members = [
        {"family": row["family"], "capsule": row["capsule"], "capsule_tree_sha256": row["capsule_tree_sha256"]}
        for row in sorted(members, key=lambda item: (item["family"], item["capsule"]))
        if row["family"] in selected_set
    ]
    unselected = [
        {
            "family": family,
            "reason": (
                "redundant_declared_axes"
                if family in axes_by_family
                else next(row["reason"] for row in refused if row["family"] == family)
            ),
            "levers": sorted({str(lever) for row in rows if (lever := (row.get("purpose") or {}).get("lever"))}),
            "capsules": sorted(str(row["capsule"]) for row in rows),
        }
        for family, rows in sorted(by_family.items())
        if family not in selected_set
    ]
    performance = (selected_scope or {}).get("performance") or {}
    if selected_scope is not None and not isinstance(performance, Mapping):
        raise StageGateError("representative selection has a malformed connected-slice scope")
    connected = {
        "status": performance.get("status", "unknown"),
        "required_unverified": [
            {"signature": row.get("signature"), "instance_ids": row.get("instance_ids")}
            for row in performance.get("required") or []
        ],
        "excluded": [
            {key: row.get(key) for key in ("instance_id", "signature", "status", "reason")}
            for row in performance.get("excluded") or []
        ],
        "unresolved": [
            {key: row.get(key) for key in ("instance_id", "signature", "status", "reason")}
            for row in performance.get("unresolved") or []
        ],
        "qualification": "no selected standalone family establishes source-connected placement or timing",
    }
    return {
        "schema": SCHEMA,
        "status": "diagnostic_subset_unmeasured",
        "target": test_justification.get("target"),
        "source": {
            "test_justification_sha256": C.document_sha256(test_justification),
            "selected_scope_sha256": (C.document_sha256(selected_scope) if selected_scope is not None else None),
        },
        "strategy": "greedy_static_axis_cover_with_indivisible_families",
        "selected_families": sorted(selected_set),
        "selected_members": selected_members,
        "unselected_families": unselected,
        "refused_families": refused,
        "uncovered_family_obligations": refused,
        "uncovered_static_axes": [{"kind": kind, "value": value} for kind, value in sorted(uncovered)],
        "uncovered_source_demands": (
            (test_justification.get("workload_accounting") or {}).get("uncovered_demands") or []
        ),
        "connected_slice": connected,
        "qualification": (
            "iteration subset only; static operation sites and RTL traits do not weight model time, "
            "establish placement, certify correctness, or predict speedup; omitted levers remain unmeasured"
        ),
    }
