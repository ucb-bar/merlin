"""The author's top-down schedule stream in a reference comparison: summary first, detail on disk.

The author serializes a schedule stream in their own process (``python -m merlin.analysis.kernels.schedule_effects``);
this service only reads that document, refuses it unless it is an owned ``artifacts/`` or ``variants/``
file of bounded size with a matching stream digest, and decodes it with the comparison's own selected
decoder. No candidate code runs here. With the emitted binary as well, each expert difference is
attributed to the lowering (already in the schedule) or to emission (absent from the schedule).
"""

from __future__ import annotations

import hashlib
import json

MAX_BYTES = 256 << 20
_ROWS = 24
GAPS = {
    "candidate_schedule_outside_owned_artifacts": "Write the stream document under artifacts/ or variants/NAME/ "
    "with python -m merlin.analysis.kernels.schedule_effects MODULE:FUNCTION --arguments ... --out artifacts/FILE.json.",
    "candidate_schedule_exceeds_bound": "Record a smaller occurrence or pass --limit to the author tool.",
    "schedule_stream_schema_mismatch": "Pass the document python -m merlin.analysis.kernels.schedule_effects wrote, unedited.",
    "schedule_stream_unbounded_or_malformed": "Re-record the stream with the author tool; do not edit it.",
    "schedule_stream_entry_malformed": "Re-record the stream with the author tool; do not edit it.",
    "schedule_stream_digest_mismatch": "The stream changed after it was recorded; re-record it with the author tool.",
}


def validate_path(value):
    """The request field: one owned relative path under ``artifacts/`` or ``variants/NAME/``."""
    from merlin.development.backend.feedback.reference_analysis import _relative

    if value is None:
        return None
    parts = _relative(value).parts
    if not (parts[0] == "artifacts" and len(parts) > 1 or parts[0] == "variants" and len(parts) > 2):
        raise ValueError("candidate_schedule_outside_owned_artifacts")
    return value


def read(workspace, value):
    """The validated stream document and its file pin."""
    from merlin.analysis.kernels.schedule_effects import validate_stream_document
    from merlin.development.backend.feedback.reference_analysis import _path

    path, label = _path(workspace, validate_path(value), candidate=True)
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("candidate_schedule_exceeds_bound")
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except ValueError:
        raise ValueError("schedule_stream_schema_mismatch") from None
    validate_stream_document(document)
    return document, dict(path=label, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _brief(value, width=160):
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if k not in ("windows", "window_sequence_prefix", "shapes")}
    text = json.dumps(value, sort_keys=True, default=str)
    return value if len(text) <= width else text[: width - 3] + "..."


def summary(full):
    """Summary-first projection: order facts, then each difference's attribution; the rest is detail."""
    from merlin.analysis.kernels.schedule_effects import ORDER_FACTS

    attribution = full.get("attribution")
    if attribution is not None:
        rows = [
            dict(
                effect=r["effect"],
                role=r.get("role"),
                kind=r["kind"],
                attribution=r["attribution"],
                expert=_brief((r.get("expert_vs_emitted") or {}).get("reference")),
                schedule=_brief((r.get("expert_vs_schedule") or {}).get("candidate")),
                emitted=_brief((r.get("expert_vs_emitted") or {}).get("candidate")),
            )
            for r in attribution["attribution"]
        ]
        realized = attribution["emission_realized_schedule"]
        emission = dict(
            realized=realized["value"],
            compared=realized["compared"],
            changes=[dict(effect=c["effect"], role=c["role"]) for c in attribution["emission_changes"]],
        )
    else:
        rows = [
            dict(
                effect=r["effect"],
                role=r.get("role"),
                kind="order" if r["effect"] in ORDER_FACTS else "count_or_width",
                attribution="schedule_differs_from_expert",
                expert=_brief(r.get("reference")),
                schedule=_brief(r.get("candidate")),
            )
            for r in full["schedule_vs_expert"]
            if r.get("difference") == "differs"
        ]
        emission = dict(realized=None, reason="no emitted candidate supplied; emission is not compared")
    rows.sort(key=lambda r: r["kind"] != "order")
    order = {
        name: dict(status=fact.get("status"), **({"reason": fact.get("reason")} if fact.get("reason") else {}))
        for name, fact in (full["schedule_effects"]["facts"] or {}).items()
        if name in ORDER_FACTS
    }
    return dict(
        input=full["input"],
        status=full["status"],
        order_facts=order,
        emission=emission,
        differences=rows[:_ROWS],
        differences_total=len(rows),
        more="every comparison row, the ordered stream summary and the facts are in the detail file",
        performance="unknown",
        timing="not_claimed",
    )


def attribute_schedule(workspace, value, *, reference, emitted, decoding_context):
    """``(summary, full)`` for one request's ``candidate_schedule``; ``emitted`` may be ``None``."""
    from merlin.analysis.kernels.decoded_effects import selected_structure
    from merlin.analysis.kernels.schedule_effects import attribute, effects_record, schedule_effects, stream_differences
    from merlin.development.backend.feedback import reference_analysis as A
    from merlin.development.backend.feedback.reference_applicability import _contract_document

    document, pin = read(workspace, value)
    context = dict(decoding_context)
    decoder = A._operand_decoder(
        context["contract"],
        context["facts"],
        context["target"],
        descriptor=context.get("descriptor"),
        include_structure=True,
    )
    structure = (
        selected_structure(
            decoder, _contract_document(context["contract"], context["target"], context.get("descriptor"))
        )
        if decoder
        else {}
    )
    effects = schedule_effects(document, decoder=decoder, structure=structure, evidence_id="candidate_schedule")
    schedule = effects_record(effects, evidence_id="candidate_schedule")
    expert = dict(
        derived_effects=reference.get("derived_effects") or {},
        evidence=dict(evidence_id=(reference.get("evidence") or {}).get("evidence_id") or "reference"),
    )
    full = dict(input=pin, status=document["status"], schedule_effects=effects)
    if document["status"] != "derived":
        full["reason"] = document.get("reason")
    if emitted is not None:
        full["attribution"] = attribute(
            schedule=schedule,
            emitted=dict(derived_effects=emitted, evidence=dict(evidence_id="candidate")),
            expert=expert,
        )
    else:
        full["schedule_vs_expert"] = stream_differences(expert, schedule)
    return summary(full), full
