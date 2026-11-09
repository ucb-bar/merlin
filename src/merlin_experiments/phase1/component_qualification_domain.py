"""Reopen the original input domain without replacing candidate qualification.

The source-preparation selection comes only from the issued compiler origin.
It preserves every original requirement and pending candidate predicate. Its
completion admits test inputs, never emitted semantics or hardware execution.
Historical qualifications retain their concrete coverage and member selection.
"""

from merlin.common.jsonio import canonical_json
from merlin.common.strict_json import loads
from merlin_experiments.phase2.contracts import StageGateError

LEGACY_RECEIPT_SCHEMA = "merlin.component_functional_qualification.v1"
SOURCE_RECEIPT_SCHEMA = "merlin.component_functional_qualification.v2"


def preparation_for_origin(origin):
    return getattr(getattr(origin, "inputs", None), "source_preparation", None)


def _source_record(preparation):
    from merlin_experiments.phase0.source_preparation_release import SourcePreparation

    if type(preparation) is not SourcePreparation:
        raise StageGateError("component qualification needs the actual live original source preparation")
    return loads(preparation.receipt_json)


def selected_obligations(report, *, preparation=None):
    """Select the original denominator, also used by actual evidence replay.

    This is a data selection after the ordinary domain replay, not an issuer.
    Explicit source preparation includes every mandatory development member;
    legacy qualification keeps its original guard/transfer selection.
    """
    if preparation is not None:
        original = _source_record(preparation)
        if original["coverage_sha256"] != report["sha256"] or original["mandatory_source_blockers"]:
            raise StageGateError("component qualification source denominator is incomplete or changed")
    return [
        row
        for row in report["obligations"]
        if row["mandatory"] and (preparation is not None or row["cohort"] != "development")
    ]


def reopen_domain(corpus_root, *, origin):
    """Require unchanged actual source owners before any candidate grading."""
    preparation = preparation_for_origin(origin)
    if preparation is None:
        from merlin_experiments.phase0.component_coverage import verify_report

        return verify_report(corpus_root), None
    from .component_generation_admission import verify_generation_inputs

    report = verify_generation_inputs(
        corpus_root, preparation=preparation, hardware=origin.inputs.hardware, software=origin.inputs.software
    )
    original = _source_record(preparation)
    rows = report["obligations"]
    from merlin_experiments.phase0.component_generation import digest

    if (
        original["original_required_ids"] != [row["id"] for row in rows]
        or original["original_mandatory_ids"] != [row["id"] for row in rows if row["mandatory"]]
        or len(original["candidate_predicates"]) != len(rows)
        or len(original["requirements"]) != len(rows)
        or any(
            requirement["original_id"] != row["id"] or requirement["original_requirement_sha256"] != digest(row)
            for row, requirement in zip(rows, original["requirements"], strict=True)
        )
        or any(
            predicate["original_id"] != row["id"]
            or type(predicate["candidate_verdict_phase"]) is not int
            or predicate["candidate_verdict_phase"] != 1
            or predicate["candidate_verdict"] != "not_evaluated"
            for row, predicate in zip(rows, original["candidate_predicates"], strict=True)
        )
    ):
        raise StageGateError("component qualification lost original requirements or pending candidate predicates")
    binding = {
        "source_preparation_sha256": preparation.sha256,
        "source_preparation_schema": original["schema"],
        "original_required_ids": original["original_required_ids"],
        "original_mandatory_ids": original["original_mandatory_ids"],
        "original_requirements": original["requirements"],
        "candidate_predicates": original["candidate_predicates"],
        "scope": "original test inputs only; every candidate execution/static/effect verdict remains required",
    }
    return report, binding


def receipt_domain(binding):
    if binding is None:
        return {"schema": LEGACY_RECEIPT_SCHEMA}
    return {"schema": SOURCE_RECEIPT_SCHEMA, "source_domain": binding}


def verify_receipt_domain(document, binding):
    selected = receipt_domain(binding)
    if any(canonical_json(document.get(key)) != canonical_json(value) for key, value in selected.items()) or (
        binding is None and "source_domain" in document
    ):
        raise StageGateError("component qualification original source domain selection changed")


def unchanged_domain(original, actual):
    return canonical_json(original) == canonical_json(actual)
