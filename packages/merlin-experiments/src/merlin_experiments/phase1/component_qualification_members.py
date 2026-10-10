"""Join live original call/cohort inputs to the ordinary candidate denominator.

Original requirements and pending candidate predicates are never replaced. A
single actual original source case can serve several original requirement IDs;
it still executes once with its complete ordered input/output roster.
"""

from merlin.common.jsonio import canonical_json
from merlin_experiments.phase2.contracts import StageGateError

from . import component_original_members as M
from .component_qualification_domain import selected_obligations


def prepare_for_origin(preparation, destination):
    if preparation is None or preparation.semantic_cases is None:
        return None
    preparation.require_complete()
    owner = M.prepare(standard_ir=preparation.semantic_cases.standard_ir, destination=destination)
    owner.require_complete()
    return owner


def obligations(report, *, preparation=None, original_members=None):
    original = selected_obligations(report, preparation=preparation)
    if original_members is None:
        if preparation is not None and preparation.semantic_cases is not None:
            raise StageGateError("qualification omitted the live original source/reference candidate members")
        return original
    if (
        type(original_members) is not M.OriginalCandidateMembers
        or preparation is None
        or (
            preparation.semantic_cases is None
            or original_members.standard_ir is not preparation.semantic_cases.standard_ir
        )
    ):
        raise StageGateError("qualification original candidate members differ from the exact prepared source owner")
    prepared = preparation.require_complete()
    rows = original_members.require_complete()["members"]
    witnesses = prepared["original_source_reference_witnesses"]
    if len(rows) != len(witnesses):
        raise StageGateError("qualification lost an original call/cohort source witness")
    identities = {}
    for requirement in prepared["requirements"]:
        for slot in requirement.get("original_reference_facets", {}).get("required_source_slots", []):
            if type(slot) is not int or not 0 <= slot < len(rows):
                raise StageGateError("qualification has an invalid original source/requirement slot")
            identities.setdefault(slot, []).append(requirement["original_id"])
    result = list(original)
    for row, witness in zip(rows, witnesses, strict=True):
        slot = row["source_slot"]
        if (
            canonical_json(witness["source_slot"]) != canonical_json(slot)
            or canonical_json(witness["original"]) != canonical_json(row["original"])
            or witness["reference_member_sha256"] != row["reference_member_sha256"]
            or witness["standard_ir_products"]["source"]["sha256"] != row["program_sha256"]
            or slot not in identities
        ):
            raise StageGateError("qualification original source/reference witness membership changed")
        result.append(
            {
                "id": "candidate_" + row["name"],
                "mandatory": True,
                "cohort": row["original"]["cohort"],
                "expectation": "admitted_program",
                "members": [
                    {
                        "name": row["name"],
                        "member": row["capsule_root"],
                        "program_sha256": row["program_sha256"],
                        "sha256": M._sha(canonical_json(row["envelope"])),
                        "output_roster": row["output_roster"],
                        "original": row["original"],
                        "original_requirement_ids": identities[slot],
                    }
                ],
            }
        )
    return result


def record(original_members):
    return original_members.verify() if original_members is not None else None


def verify_record(document, original_members):
    if canonical_json(document.get("original_candidate_members")) != canonical_json(record(original_members)):
        raise StageGateError("qualification original candidate-member binding changed")


def numeric_failures(score, obligations, actual_results):
    """Preserve ordinary complete numeric/refusal checks for both member domains."""
    members = [member for row in obligations for member in row["members"]]
    rows = score.get("per_capsule") or []
    index = {row.get("capsule"): row for row in rows}
    failures = []
    if len(rows) != len(members) or set(index) != {member["name"] for member in members}:
        failures.append("mandatory declared-domain denominator did not execute completely")
    for obligation in obligations:
        for member in obligation["members"]:
            row = index.get(member["name"], {})
            if obligation["expectation"] == "unsupported_program":
                actual = actual_results.get(member["name"], {})
                refused = (
                    row.get("status") == "declined"
                    and actual.get("status") == "declined"
                    and (actual.get("failure") or {}).get("plane") == "backend_declined"
                    and isinstance((actual.get("declined") or {}).get("reason"), str)
                    and bool(actual["declined"]["reason"].strip())
                )
                if not refused:
                    failures.append(member["name"] + ": declared unsupported program was not refused")
            elif row.get("status") != "pass" or row.get("numeric") != "pass" or row.get("cert_verdict"):
                failures.append(member["name"] + ": complete numerical/executable certification failed")
    return failures
