"""Redacted QA gate for the capsule_bench_v0 raw_baseline pilot.

Grades a candidate ``submission/`` against the PUBLIC pilot capsules (A0/A2/A4/B0) through the
real ladder (L0 reference==simulate, spike, verilator + trace_check), then emits ONLY a redacted
verdict the agent is allowed to see:

    {all_pass, n_passed, n_capsules, integrity_status,
     per_capsule: [{capsule, status, numeric_status, mismatch_count, trace_status,
                    trace_violations:[class-name strings], tiers:{L*:status}, failure_plane,
                    failure_category, highest_tier}],
     first_failure_planes}

It DELIBERATELY omits every answer-bearing value: golden outputs, reference/oracle outputs,
numeric diffs (max_abs_diff / first_mismatch), command buffers, lowered MLIR. The full grading
work tree (which contains numeric_report.yaml etc.) is written under an OPERATOR-ONLY runs_root
that the agent never sees; only the scrubbed verdict crosses back.

The agent uses this as a pass/fail QA signal to iterate against — never as an answer key.

Usage:
  qa_check.py --submission <dir> --out <verdict.json> [--labels public,dev]
              [--runs-root <operator-only tmp>] [--no-oracle] [--timeout 900]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

from merlin.common.artifacts import cache_dir  # noqa: E402 — purgeable work trees
from merlin.development.backend.context import (
    InvocationContext,
    add_context_arguments,
    readback_kwargs,
    resolve_context,
)
from merlin.development.backend.feedback.certification import tier_policy as certification_gates
from merlin.development.backend.feedback.verdicts import (
    summarize as qualification_summary,
)
from merlin.development.targetgen.grading import model_checks as CG  # noqa: E402
from merlin.development.targetgen.oracles import tier_policy as _TI  # noqa: E402
from merlin.evaluation.capsule.evidence import certificates as capsule_adapters
from merlin.evaluation.capsule.evidence import certificates as capsule_availability

# ---- qa ----------------------------------------------------------------------------------------


def _per_capsule_from_results(runs_root: Path, *, only_capsule: str | None = None) -> dict[str, dict]:
    """Read each capsule_result.json from the operator-only work tree and redact it."""
    from merlin.development.backend.feedback.grading.qa import (
        _abandoned_on_budget,
        _emitted_cost,
        _execution_digest_from_result,
        _liveness_screen,
        _placement_coverage,
        _redact_detail,
    )

    out: dict[str, dict] = {}
    # The runner writes under <target>-capsule-bench (target-derived), NOT the one default baked into
    # CR.SUITE — keying off CR.SUITE dropped every per-capsule failure detail for every other target,
    # so the agent saw plane counts but never the reason. Glob any suite subdir so the reason always
    # surfaces (target-general).
    rr = runs_root / "runs"
    if not rr.exists():
        return out
    for cr in sorted(rr.glob("*/*/capsule_result.json")):
        if only_capsule is not None and cr.parent.name != only_capsule:
            continue
        try:
            r = json.loads(cr.read_text())
        except Exception:
            continue
        num = r.get("numeric") or {}
        fail = r.get("failure") or {}
        tiers = r.get("tiers") or {}
        candidate_feedback = _candidate_native_feedback(r)
        if candidate_feedback is not None:
            # The old runner's numerical/timing/tier evidence belongs to a
            # different program. Only the candidate record may populate these
            # fields; missing evidence remains missing, not a host-graph pass.
            native = r.get("candidate_native_execution")
            num = native.get("numeric") if isinstance(native, dict) else None
            num = num if isinstance(num, dict) else {}
            tiers = {tier: {"status": value} for tier, value in candidate_feedback.get("tiers", {}).items()}
            mandatory = candidate_feedback.get("required_tiers", [])
            missing = [tier for tier in mandatory if (tiers.get(tier) or {}).get("status") != "pass"]
            verified_mismatch = (
                candidate_feedback["status"] == "fail"
                and "candidate_verified_numeric_mismatch" in candidate_feedback["violations"]
            )
            fail = (
                {}
                if candidate_feedback["status"] == "pass"
                else {
                    "plane": "candidate_model_numeric" if verified_mismatch else "model_execution",
                    "category": (
                        "FUNCTIONAL_MISMATCH"
                        if verified_mismatch
                        else "PROTOCOL_VIOLATION"
                        if candidate_feedback["status"] == "fail"
                        else "NOT_RUN_IS_NOT_PASS"
                    ),
                    "tier": missing[0] if missing else None,
                    "detail": ", ".join(candidate_feedback.get("violations", [])) or candidate_feedback.get("reason"),
                }
            )
        out[r.get("capsule", cr.parent.name)] = {
            "status": r.get("status"),
            "expected_refusal": (
                {"status": "met"}
                if r.get("status") == "pass"
                and r.get("failure") is None
                and (r.get("expected_refusal") or {}).get("status") == "met"
                and isinstance(r.get("declined"), dict)
                and bool(r["declined"])
                else None
            ),
            # Opaque content address of the exact executable plus target/RTL identity. It reveals no
            # answer-bearing value; promotion uses it solely to retain a still-applicable certificate.
            "execution_digest": _execution_digest_from_result(cr) if candidate_feedback is None else None,
            # The backend's own STATED refusal. Redaction-safe: it is text the SUBMISSION wrote about
            # its own coverage, never corpus or golden data.
            "declined": r.get("declined"),
            "numeric_status": (
                num.get("status") if candidate_feedback is None or num.get("status") in ("pass", "fail") else None
            ),
            "mismatch_count": (
                num.get("mismatch_count")
                if candidate_feedback is None or (type(num.get("mismatch_count")) is int and num["mismatch_count"] >= 0)
                else None
            ),
            "trace_status": (r.get("trace_check") or {}).get("status") if candidate_feedback is None else None,
            "trace_requirement_status": (
                (r.get("trace_check") or {}).get("trace_requirement_status") if candidate_feedback is None else None
            ),
            "trace_unavailable_requirements": (
                list((r.get("trace_check") or {}).get("unavailable_requirements") or [])[:16]
                if candidate_feedback is None
                else []
            ),
            "trace_violations": (
                list((r.get("trace_check") or {}).get("violations") or []) if candidate_feedback is None else []
            ),
            "tiers": {t: (tiers.get(t) or {}).get("status") for t in tiers},
            # WHY EACH NON-PASSING TIER DID NOT PASS, redacted. The line above carries the bare status,
            # which is what left the agent (and every reader of a verdict) with `L3: "unavailable"` and
            # `failure_detail: null` on 24 of 29 rows. The tier's own sentence is the only actionable
            # half. Passed through the SAME scrub as `failure_detail` (`_redact_detail`) because a tier
            # reason carries absolute paths, engine command lines and console tails, and through
            # `tier_integrity` first so a tier that stated nothing yields an explicit UNKNOWN sentence
            # rather than a null.
            "tier_reasons": {t: _redact_detail(why) for t, why in _TI.not_certified(tiers).items()},
            # WHICH TIERS WERE ABANDONED ON TIME, not answered wrongly. A cert the engine ran out of
            # budget on is not evidence of incorrectness, but the redacted verdict carried only the tier
            # STATUS -- so "abandoned at 900s" and "produced wrong numbers" were the same `fail` string,
            # and a reader (and `agg_agentic_results._l3_evidence`) could not separate them. Measured:
            # six capsules in one batch were abandoned at exactly 900s and counted against the arm's
            # L3-clean headline as if their lowering were wrong. A BOOLEAN list, never the reason text:
            # the reason carries absolute paths and engine command lines, and nothing about the corpus
            # needs to ride here for the distinction to be usable.
            "tiers_abandoned": sorted(
                t
                for t in tiers
                if (tiers.get(t) or {}).get("status") == "fail"
                and _abandoned_on_budget((tiers.get(t) or {}).get("reason"))
            ),
            "tier_cycles": {
                t: (tiers.get(t) or {}).get("cycles") for t in tiers if (tiers.get(t) or {}).get("cycles") is not None
            },
            # WHICH SIMULATOR ANSWERED EACH TIER. Redaction-safe: an engine name is a harness constant
            # ("gsim", "verilator"), never capsule or golden data. It rides here because a cycle count is
            # only interpretable next to the instrument that produced it -- two engines at the same
            # fidelity differ by more than an order of magnitude in cost -- and because a cert that fell
            # back to the slow engine should be visible to whoever is reading the verdict, not only to
            # whoever ran readiness. Filtered to a short identifier so nothing else can ride the field.
            "tier_engines": {
                t: (tiers.get(t) or {}).get("engine")
                for t in tiers
                if isinstance((tiers.get(t) or {}).get("engine"), str) and len((tiers.get(t) or {}).get("engine")) <= 32
            },
            # WHICH TIERS THIS GRADE ACTUALLY EXECUTED, and which carried a verdict earned earlier by
            # the same executable on the same instrument. Rides the verdict because the alternative is a
            # cached grade that is indistinguishable from a fresh one: a reader comparing two rounds'
            # wall clocks, or asking "was L3 measured against these bytes today", has no other way to
            # tell. Redaction-safe -- tier labels and a boolean, never capsule or golden data.
            "tier_reuse": (
                lambda v: (
                    {
                        "executed": [t for t in (v.get("executed") or []) if isinstance(t, str)],
                        "carried": [t for t in (v.get("carried") or []) if isinstance(t, str)],
                    }
                    if isinstance(v, dict)
                    else None
                )
            )(r.get("tier_reuse") if candidate_feedback is None else None),
            # A seeded grade's pass carried in full from its seed run (merlin.development.targetgen.grading.seeded_grade):
            # "carried from <run>", an operator run name. Nothing else can ride the field.
            **(
                lambda v: (
                    {"carried_from": v}
                    if isinstance(v, str) and v.startswith("carried from ") and len(v) <= 160
                    else {}
                )
            )((r.get("tier_reuse") or {}).get("carried_from") if isinstance(r.get("tier_reuse"), dict) else None),
            "failure_plane": fail.get("plane"),
            "failure_category": fail.get("category"),
            # The tier LABEL survives redaction. It is a harness constant ("L2"), never capsule data, and
            # the blanket numeric scrub was rewriting it to "L#" -- so an agent could not even tell WHICH
            # tier blocked it. Passed through a strict shape check so only a tier label can ever ride here.
            "failure_tier": (
                lambda v: v if isinstance(v, str) and len(v) <= 4 and v[:1] == "L" and v[1:].isdigit() else None
            )(fail.get("tier")),
            "failure_detail": _redact_detail(fail.get("detail")),
            # The cost of the agent's OWN emitted program, available before the cert tier runs. See
            # _emitted_cost: an opaque L3 failure with a null plane was the only feedback on a lowering
            # that moved 2,000x the median capsule's DRAM traffic.
            "emitted_cost": _emitted_cost(cr) if candidate_feedback is None else None,
            # WHERE EACH CALL LANDED AND, OFF THE ACCELERATOR, WHY. See _placement_coverage: the
            # grader computes this census for every whole-model row and the agent was told only
            # `eligible_model_layer_fell_back_to_host` -- a violation string that names neither the
            # call nor the reason, so "the hardware cannot take this shape" and "the compiler never
            # tried" reached the agent as the same sentence.
            "placement_coverage": _placement_coverage(r),
            "candidate_native_verification": candidate_feedback,
            # WOULD THIS PROGRAM EVEN RUN ON SILICON. See _liveness_screen: the L2.5 screen's verdict was
            # computed for every capsule, written beside the result, and never shown to the agent -- 313
            # `stall` verdicts across 3446 reports on disk with no reader. SURFACED, NOT GATED (see
            # REFUSING_SEVERITIES): rule names, severities and counts only.
            "liveness": _liveness_screen(cr) if candidate_feedback is None else None,
        }
    return out


def _candidate_native_feedback(result: dict) -> dict | None:
    """Project native verification without consoles, goldens or private error text.

    Component statuses and closed failure codes concern the candidate's own
    artifact. Coverage counts are admitted only from the independently verified
    source/dispatch join, never from the legacy runner's distinct program.
    This projection does not grade or upgrade the underlying result.
    """
    owner = result
    owned = result.get("model_execution_check")
    if isinstance(owned, dict) and owned.get("kind") == "owned_host_model_execution":
        # Project only the independently reopened owned audit, never its raw
        # runtime receipt or the legacy host graph's numeric observations.
        source = owned.get("candidate_source_placement")
        dispatch = owned.get("candidate_completed_dispatch")
        owner = {
            "candidate_native_model_check": {
                "schema": "merlin_candidate_native_model_check_v1",
                "status": owned.get("status"),
                "violations": owned.get("violations"),
                "emitted_host_compute": (
                    {"status": "violation"}
                    if owned.get("status") == "fail"
                    and "candidate_host_tensor_compute_violation" in (owned.get("violations") or [])
                    else {"status": "clean"}
                    if isinstance(source, dict) and source.get("status") == "clean"
                    else {}
                ),
                "source_placement": source,
                "completed_dispatch": dispatch,
                "candidate_source_coverage": owned.get("candidate_source_coverage"),
                "candidate_required_tiers": owned.get("candidate_required_tiers"),
            }
        }
    top_level_candidate = "candidate_native_model_check" in owner
    if not top_level_candidate:
        owner = result.get("model_execution_check")
    if not isinstance(owner, dict) or "candidate_native_model_check" not in owner:
        return None
    check = owner["candidate_native_model_check"]
    model_check = result.get("model_execution_check")
    if (
        check is None
        and not top_level_candidate
        and "legacy_model_diagnostic" not in result
        and isinstance(model_check, dict)
        and model_check.get("kind") == "model_accelerator_execution"
    ):
        # Non-must-accelerate models use the legacy model grade. Its owner
        # serializes a null candidate slot; that is not a malformed candidate
        # record. The candidate branch marks its distinct legacy diagnostic.
        return None
    if not isinstance(check, dict) or check.get("schema") != "merlin_candidate_native_model_check_v1":
        return {"status": "unverified", "reason": "candidate_verification_record_unrecognized"}

    def status(record, allowed):
        value = record.get("status") if isinstance(record, dict) else None
        return value if isinstance(value, str) and value in allowed else "unverified"

    allowed_codes = {
        "candidate_host_tensor_compute_violation",
        "candidate_emitted_host_compute_unverified",
        "candidate_source_placement_violation",
        "candidate_source_placement_unverified",
        "candidate_full_model_native_unverified",
        "candidate_required_rtl_engine_mismatch",
        "candidate_completed_dispatch_unverified",
        "candidate_source_coverage_unverified",
        "candidate_required_tiers_unverified",
        "candidate_verified_numeric_mismatch",
    }
    violations = check.get("violations")
    codes = []
    for value in violations if isinstance(violations, list) else []:
        code = (
            value
            if isinstance(value, str) and value in allowed_codes
            else "candidate_verification_failure_unrecognized"
        )
        if code not in codes:
            codes.append(code)
    summary = {
        "status": status(check, {"pass", "fail", "incomplete", "unverified"}),
        "violations": codes,
        "scope": "candidate-only native verification; not general transformation equivalence",
        "components": {
            key: status(check.get(key), {"clean", "violation", "verified", "pass", "fail", "incomplete", "unverified"})
            for key in ("emitted_host_compute", "source_placement", "completed_dispatch", "candidate_required_tiers")
        },
    }
    native_status = check.get("native_status")
    if isinstance(native_status, str) and native_status in {
        "incomplete",
        "compiled_not_run",
        "numeric_match_diagnostic",
        "numeric_mismatch_diagnostic",
    }:
        summary["native_status"] = native_status
    native = result.get("candidate_native_execution")
    if (
        isinstance(native, dict)
        and native.get("schema") == "merlin_candidate_native_model_execution_v1"
        and native.get("status") == "incomplete"
    ):
        failure = native.get("failure")
        if (
            isinstance(failure, dict)
            and failure.get("type") == "NativeModelExecutionError"
            and failure.get("code") == "output_writer_source_result_ownership_unverified"
        ):
            # A closed refusal from the trusted binder, never its raw error text,
            # tensor names, source contents, or numerical expectations. This
            # diagnoses missing evidence without changing any grading decision.
            summary["native_failure"] = {
                "stage": "source_binding",
                "code": "output_writer_source_result_ownership_unverified",
            }
    coverage = check.get("candidate_source_coverage")
    counts = [
        coverage.get(key) if isinstance(coverage, dict) else None
        for key in ("n_source_operations", "n_eligible", "n_completed_eligible")
    ]
    if (
        isinstance(coverage, dict)
        and coverage.get("status") == "verified"
        and all(type(value) is int for value in counts)
        and counts[0] >= counts[1] == counts[2] > 0
    ):
        summary["source_coverage"] = {
            "status": "verified",
            "n_source_operations": counts[0],
            "n_eligible": counts[1],
            "n_completed_eligible": counts[2],
        }
    else:
        summary["source_coverage"] = {"status": "unverified"}
    tier_check = check.get("candidate_required_tiers")
    if isinstance(tier_check, dict):
        known_tiers = {"L0", "L1", "L2", "L3"}
        required = tier_check.get("required_tiers")
        if isinstance(required, list) and all(isinstance(tier, str) and tier in known_tiers for tier in required):
            summary["required_tiers"] = list(required)
        tiers = tier_check.get("tiers")
        if isinstance(tiers, dict):
            summary["tiers"] = {
                tier: status(tiers[tier], {"pass", "fail", "unverified", "unavailable", "incomplete", "skipped"})
                for tier in sorted(known_tiers & tiers.keys())
            }
    return summary
