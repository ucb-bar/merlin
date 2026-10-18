"""Answer-free counts that distinguish execution from required structural evidence."""

from __future__ import annotations

from collections import Counter


def candidate_evidence_complete(summary: dict) -> bool:
    """Require the closed candidate's source, placement, dispatch and tier evidence."""
    tiers = summary.get("tiers")
    required = summary.get("required_tiers")
    source = summary.get("source_coverage")
    components = summary.get("components")
    return (
        summary.get("status") == "pass"
        and isinstance(tiers, dict)
        and isinstance(required, list)
        and bool(required)
        and all(isinstance(tier, str) and tiers.get(tier) == "pass" for tier in required)
        and isinstance(source, dict)
        and source.get("status") == "verified"
        and isinstance(components, dict)
        and all(
            components.get(key) == expected
            for key, expected in (
                ("emitted_host_compute", "clean"),
                ("source_placement", "clean"),
                ("completed_dispatch", "verified"),
                ("candidate_required_tiers", "pass"),
            )
        )
    )


def summarize(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        passed = row.get("status") == "pass"
        refusal = (row.get("expected_refusal") or {}).get("status") == "met" and passed
        counts["result_passed"] += int(passed)
        counts["execution_passed"] += int(passed and not refusal)
        native = row.get("candidate_native_verification")
        if isinstance(native, dict):
            complete = passed and candidate_evidence_complete(native)
            counts["model_programs_verified"] += int(complete)
            counts["model_programs_unverified"] += int(not complete)
        elif refusal:
            counts["expected_refusals_met"] += 1
        else:
            requirement = row.get("trace_requirement_status")
            complete = row.get("trace_status") == "pass" and requirement in (None, "pass")
            counts["execution_passes_with_trace_evidence"] += int(passed and complete)
            counts["execution_passes_without_trace_evidence"] += int(passed and not complete)
            if isinstance(requirement, str) and requirement.startswith("unavailable"):
                counts["trace_evidence_unavailable"] += 1
            elif row.get("trace_status") == "fail":
                counts["trace_evidence_failed"] += 1
    fields = (
        "result_passed",
        "execution_passed",
        "execution_passes_with_trace_evidence",
        "execution_passes_without_trace_evidence",
        "trace_evidence_unavailable",
        "trace_evidence_failed",
        "model_programs_verified",
        "model_programs_unverified",
        "expected_refusals_met",
    )
    return {
        "schema": "merlin.phase1.qualification_counts.v1",
        "n_completed": len(rows),
        **{name: counts[name] for name in fields},
        "scope": "execution and structural evidence counts; final source/cohort/tier gates remain separate",
    }
