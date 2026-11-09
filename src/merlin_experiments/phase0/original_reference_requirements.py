"""Join live finite original comparisons to unchanged source requirements.

The join credits only its checked comparison/ordered-ABI facets. Protected
semantic-owner correspondence, domain/stress/effect/resource premises and
candidate body/execution remain required. No saved rows mint evidence, and no
original coverage requirement is removed or made ready here.
"""

from __future__ import annotations

import copy
import hashlib
import json

from merlin.common.jsonio import canonical_json

from . import original_reference_flow as F
from . import original_reference_standard_ir as S
from .original_call_sources import required_source_cohorts

SCHEMA = "merlin.phase0.source_requirement_ledger.v2"
_CALL = ("original_member_id", "graph_path", "node", "target")


def _original_calls(coverage, references):
    rows = coverage["automatic_derivation"]["original_call_sources"]["members"]
    basis = json.loads(references.basis.declaration_json)["members"]
    if [row["graph_path"] for row in rows] != [row.path for row in references.basis.graph_sources]:
        raise ValueError("original reference join changed the complete original graph roster")
    wanted = [
        ((original["id"], row["graph_path"], call["node"], call["target"]), call)
        for original, row in zip(basis, rows, strict=True)
        for call in row["calls"]
    ]
    if len({key for key, _ in wanted}) != len(wanted):
        raise ValueError("original reference join duplicated an original typed call")
    return wanted


def join(document, *, coverage, hardware, software, standard_ir):
    """Reopen actual complete sources/references/processes before attribution."""
    if type(standard_ir) is not S.OriginalReferenceStandardIr:
        raise ValueError("original requirement join needs its actual live standard IR preparation")
    references = standard_ir.references
    if references.schema_intake.software is not software or software.hardware is not hardware:
        raise ValueError("original reference join needs the same live hardware and original software owners")
    record = standard_ir.record()  # Replays complete references, values, ABI and invocation/source pins.
    reference = references.record_without_verification()
    wanted = _original_calls(coverage, references)
    expected = [(*key, cohort, extent) for key, _ in wanted for cohort, extent in required_source_cohorts()]
    actual = [tuple(row[key] for key in (*_CALL, "cohort", "extent")) for row in reference["members"]]
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("original reference join lost exact ordered original call/cohort membership")
    calls = dict(wanted)
    witnesses, grouped = [], {}
    for index, (original, emitted) in enumerate(zip(reference["members"], record["members"], strict=True)):
        key = tuple(original[name] for name in _CALL)
        if canonical_json(original["call"]) != canonical_json(calls[key]):
            raise ValueError("original reference join changed original scalar/default/result ABI bindings")
        witness = {
            "source_slot": index,
            "original": emitted["original"],
            "original_graph_source": S.R._pin(original["graph_path"]),
            "original_typed_call_sha256": hashlib.sha256(canonical_json(original["call"])).hexdigest(),
            "reference_member_sha256": emitted["reference_member_sha256"],
            "reference_state": original["state"],
            "standard_ir_state": emitted["state"],
            "reference_products": copy.deepcopy(original.get("products", {})),
            "standard_ir_products": copy.deepcopy(emitted.get("products", {})),
            "reference_invocation": original.get("invocation"),
            "upstream_invocation": record["invocation"],
            "parse_invocation": emitted.get("parse_invocation"),
            "comparison": copy.deepcopy(emitted.get("comparison", original.get("comparison"))),
            "ordered_abi": copy.deepcopy(emitted.get("ordered_abi")),
            "required_unknowns": sorted(set(emitted["required_unknowns"])),
            "reason": emitted.get("reason", original.get("reason")),
        }
        witnesses.append(witness)
        grouped.setdefault((key[0], key[2], key[3]), []).append(witness)
    result = copy.deepcopy(document)
    if [row["original_id"] for row in result["requirements"]] != result["original_required_ids"]:
        raise ValueError("original reference ledger changed its full original requirement denominator")
    selected = {
        row["id"]: row
        for row in coverage["automatic_derivation"]["required_unknowns"]
        if row["kind"] in {"original_operator_admission", "original_operator_factory"}
    }
    seen = set()
    for row in result["requirements"]:
        original = selected.get(row["original_id"])
        if original is None:
            continue
        seen.add(row["original_id"])
        selector = original["selector"]
        slots = grouped.get(tuple(selector[name] for name in ("member", "node", "target")))
        if slots is None:
            raise ValueError("original reference join cannot bind the exact required original call selector")
        if original["kind"] == "original_operator_factory":
            slots = [
                slot for slot in slots if all(slot["original"][key] == selector[key] for key in ("cohort", "extent"))
            ]
            if len(slots) != 1:
                raise ValueError("original reference join cannot bind the required original source slot")
        row["original_reference_facets"] = {
            "required_source_slots": [slot["source_slot"] for slot in slots],
            "complete_comparison": "checked"
            if all(slot["reference_state"] == "reference_checked" for slot in slots)
            else "unavailable",
            "ordered_upstream_source_abi": "checked"
            if all(slot["standard_ir_state"] == "source_reference_ir_checked" for slot in slots)
            else "unavailable",
            "missing_original_premises": sorted({value for slot in slots for value in slot["required_unknowns"]}),
            "scope": "finite original source/reference facets only; whole original requirement remains unchanged",
        }
        # Original missing-source reasons and mandatory blockers deliberately
        # persist. Complete values/ABI do not establish semantic-owner or domain
        # premises, and successful source parsing is not compiled-body proof.
    if seen != set(selected):
        raise ValueError("original reference ledger lost an original admission/factory requirement")
    result.update(
        schema=SCHEMA,
        original_source_reference_witnesses=witnesses,
        original_reference_observations=F.summary(standard_ir),
    )
    return result
