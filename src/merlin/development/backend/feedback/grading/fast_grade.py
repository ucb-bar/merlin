"""Host-side grading of immutable candidate copies during Phase 1 authoring.

Owns snapshot, grade, evidence, promotion and redacted publication ordering.
Numerical grading remains in feedback.qa and the independent capsule evaluators.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import shutil
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from merlin.common.digest import hash_tree
from merlin.common.process.cpu_pool import shared_compute_pool
from merlin.development.backend.campaign.calibration.activation import admit_grade, grade_model_scope
from merlin.development.backend.context import InvocationContext, readback_kwargs
from merlin.development.backend.feedback.campaign import correctness as JTS
from merlin.development.backend.feedback.execution.snapshot_evaluator import measure
from merlin.development.backend.feedback.grading import completed_grade, package_rejection
from merlin.development.backend.feedback.grading import lifecycle as FL
from merlin.development.backend.feedback.grading.loop_probes import (
    _attach_codegen_scalability,
    _attach_shape_generalization,
    _shape_skip_reason,
    _skip_shape_generalization,
)
from merlin.development.backend.session import treatments as T

if TYPE_CHECKING:
    from merlin.development.backend.feedback.grading.loop_grade import GradingInputs
from merlin.development.backend.session import workspaces as OH
from merlin.development.backend.session.inputs import run_record as RI
from merlin.evaluation.capsule.evidence import certificates as capsule_availability
from merlin.target.registry.descriptor_loading import load_target_experiment

# ---- loop_grading ------------------------------------------------------------------------------


def _fast_loop_verdict_doc(
    red: dict,
    tiers_graded,
    tiers_withheld,
    *,
    structural_evidence=None,
    cohort_complete=False,
) -> dict:
    """Build the fast first verdict from the per-capsule readback. PURE, so its honesty is testable.

    `n_passed` counts capsules that passed THE TIERS THAT RAN, and the document says so. `all_pass` is a
    bool only when nothing mandatory was withheld; otherwise it is None — the JSON null that reads as
    UNKNOWN to every consumer and is falsy to every gate, so no downstream check can mistake a partial
    ladder for a converged run.
    """
    from merlin.development.backend.feedback.grading.loop_grade import _verdict_row

    rows = [_verdict_row(name, info) for name, info in sorted(red.items())]
    npass = sum(row["status"] == "pass" for row in rows)
    graded, withheld = sorted(tiers_graded), sorted(tiers_withheld)
    complete = not withheld
    # The grader joins original trace, provided-host, model and refusal scopes.
    # A trace-only shortcut would invent obligations for host/refusal rows.
    structural_complete = structural_evidence is True
    nc = len(rows)
    return {
        "qa_gate": "capsule_bench_v0_pilot",
        "stage": "first_grade_loop_tier",
        "gradeable": True,
        "tiers_graded": graded,
        "tiers_not_run": withheld,
        "mandatory_tiers_complete": complete,
        # UNKNOWN, never success: a mandatory tier that did not run may not be reported as passed.
        "all_pass": (
            bool(nc > 0 and npass == nc and structural_complete and cohort_complete is True) if complete else None
        ),
        "original_cohort_complete": cohort_complete is True,
        "structural_evidence_complete": structural_complete,
        "n_passed": npass,
        "n_capsules": nc,
        "per_capsule": rows,
        "note": (
            "FAST FIRST GRADE — the LOOP tier only, published early so you are not working blind. "
            f"Tiers graded: {graded or 'none'}. Tiers NOT run: {withheld or 'none'} — those are "
            "UNKNOWN, not passed; `all_pass` is null while any mandatory tier is outstanding. The "
            "full mandatory-ladder grade follows on the normal interval and is the one that "
            "converges the run. Contains NO reference output values."
        ),
    }


def _completed_loop_promoter(ws, frozen_source, runs, loop, cert, cover, *, inputs):
    """Promote only closed, durable selected public/dev functional passes."""
    import sys

    from merlin.development.backend.feedback.certification import (
        promotion,
    )
    from merlin.development.backend.feedback.certification import (
        tier_evidence as tier_boundary_join,
    )
    from merlin.development.backend.feedback.grading import qa
    from merlin.development.targetgen.corpus.capsule_io import discover_capsules

    te = load_target_experiment(inputs.context.descriptor)
    root = inputs.public_root() if callable(inputs.public_root) else inputs.public_root
    with shared_compute_pool():
        caps = discover_capsules((root,), labels={"public", "dev"}, contract=inputs.contract)
    excluded = set(te.effective_exclusions(cap.get("name") for cap in caps))
    allowed = {str(cap["name"]) for cap in caps if cap.get("name") and cap["name"] not in excluded}
    by_name = {str(cap["name"]): cap for cap in caps if cap.get("name")}

    def complete(result):
        name = result.get("capsule")
        from merlin.development.backend.feedback.campaign import correctness as physical
        from merlin.development.backend.feedback.campaign.curriculum import _passed

        expected_refusal = bool(physical.roster()) and _passed(result) == "expected_refusal"
        if name not in allowed or (
            (result.get("tiers", {}).get(loop) or {}).get("status") != "pass" and not expected_refusal
        ):
            return
        # A callback observes a completed result, not an in-flight file or candidate
        # supplied row. Reopen exactly its published record before attributing it.
        path = Path(runs) / "runs" / _suite_for(te.target) / name / "capsule_result.json"
        try:
            if json.loads(path.read_text()) != json.loads(json.dumps(result, allow_nan=False)):
                raise ValueError("completed capsule differs from its durable result")
            if result.get("kind") == "model" and loop == "L2" and cert == "L3" and not _independent_admission(result):
                from merlin.development.backend.feedback.certification import model_gate_vouchers

                model_gate_vouchers.issue(
                    ws,
                    frozen_source,
                    result,
                    capsule=by_name[name],
                    context=inputs.context,
                    cert_tier=cert,
                    receipt_path=path,
                    contract=inputs.contract,
                )
            promotion.promote(
                ws,
                Path(ws) / ".qa_channel",
                {
                    "per_capsule": [
                        {
                            "capsule": name,
                            "pass": True,
                            "tiers": {loop: "pass"},
                            "execution_digest": qa._execution_digest_from_result(path),
                            "tier_boundary": tier_boundary_join.optional_observation(path, tier=loop)
                            if name in physical.roster() and not expected_refusal
                            else None,
                        }
                    ]
                },
                loop,
                cert,
                cover,
                sys.stderr,
                source_ws=frozen_source,
                context=inputs.context,
                capsules_root=inputs.promotion_root,
                closed_loop_rows={name: result},
            )
        except Exception as exc:  # noqa: BLE001 -- completed feedback survives queue faults
            print(f"[first-grade] completed promotion skipped: {type(exc).__name__}: {exc}", flush=True)

    return complete


def _independent_admission(result):
    """True when the model ran under author-chosen independent admission.

    That treatment admits a model by request, with no operator-fraction gate
    and no prerequisite credit, so there is no gate admission to vouch for.
    Requiring a staged voucher there skipped every completed model promotion.
    """
    from merlin.development.backend.feedback.execution import snapshot_evaluator as snapshot_execution

    gate = result.get("model_gate_admission")
    if isinstance(gate, dict) and "policy" in gate:
        return gate.get("prerequisite_credit") is False
    return snapshot_execution.selected()


def _suite_for(target):
    from merlin.evaluation.capsule.eligibility import suite_for

    return suite_for(target)


@grade_model_scope
def fast_grade(
    ws: Path,
    run_dir: Path,
    tick: int,
    timeout: int,
    *,
    inputs: GradingInputs,
) -> dict:
    """Grade the current submission at the LOOP tier ONLY and publish it as the agent's verdict.

    Cheap by construction: `qa_loop_adapters` is the fast functional tier the corpus itself declares, so
    this costs seconds-to-minutes where the checkpoint ladder costs tens of minutes per capsule. Same
    snapshot-copy discipline as `qa_grade`, so it is safe to run while the agent is editing.
    """
    from merlin.development.backend.feedback.grading.background_admission import SubmissionNotReady
    from merlin.development.backend.feedback.grading.loop_grade import (
        GradingInputs,
        _attribute_source,
        _history_runs_roots,
        _verdict_row,
        cert_tiers_beyond_loop,
        compiler_execution,
        declared_loop_tiers,
    )

    if not (ws / "submission" / "manifest.yaml").is_file():
        raise SubmissionNotReady("no submission/manifest.yaml to grade yet")

    from merlin.development.backend.feedback.grading import capsule_feedback as _merlin_capsule_feedback_2
    from merlin.development.backend.feedback.grading import qa as _qc
    from merlin.development.backend.feedback.grading.background_admission import (
        BackgroundGuard,
        source_digest,
    )
    from merlin.development.backend.feedback.grading.grade_progress import SnapshotProgress
    from merlin.development.targetgen.grading import enforcement as _merlin_enforcement_1
    from merlin.development.targetgen.grading import model_checks as _CG
    from merlin.development.targetgen.grading import submission as _merlin_submission_1

    cand = run_dir / "_qa_work" / f"fcand_{tick}" / "submission"
    public_root = inputs.public_root() if callable(inputs.public_root) else inputs.public_root
    background_guard = BackgroundGuard(
        ws, public_root, inputs, source_digest(ws), finish_bounded_snapshot=JTS.focused()
    )
    with shared_compute_pool(timeout_s=timeout):
        if cand.parent.exists():
            shutil.rmtree(cand.parent)
        shutil.copytree(ws / "submission", cand, ignore=shutil.ignore_patterns(*RI.SNAPSHOT_IGNORED))
        RI.strip_build_state(cand)
    from merlin.development.backend.feedback.guidance.construction_plan import (
        require_ready as require_construction_plan,
    )

    require_construction_plan(cand)
    snapshot_digest = hash_tree(cand)["sha256"]
    grade_environment = admit_grade(
        candidate=cand, source_sha256=snapshot_digest, output=run_dir, kind="loop", identity={"tick": tick}
    )
    background_guard.bind_snapshot(cand)
    fruns = run_dir / "_qa_work" / f"fruns_{tick}"
    te = load_target_experiment(inputs.context.descriptor)
    from merlin.development.backend.campaign import control as joint_control

    adapters = capsule_availability.qa_loop_adapters(
        te.target,
        te.sim_via,
        declared_tiers={"L2"} if joint_control.selected() else declared_loop_tiers(inputs.policy_roots),
        **readback_kwargs(inputs.context),
    )
    _, _withheld = cert_tiers_beyond_loop(
        context=inputs.context,
        policy_roots=inputs.policy_roots,
        public_roots=lambda: inputs.public_root() if callable(inputs.public_root) else inputs.public_root,
    )
    # Resolve once before the suite: every completed functional result can feed
    # the existing queue without waiting for the last capsule in this batch.
    completed_promotion = None
    try:
        from merlin.development.backend.feedback.certification import promotion as _promotion

        loop, cert, cover = _promotion.resolve_tiers(ws, context=inputs.context, capsules_root=inputs.promotion_root)
        if loop and cert:
            completed_promotion = _completed_loop_promoter(ws, cand.parent, fruns, loop, cert, cover, inputs=inputs)
    except Exception as exc:  # noqa: BLE001 -- scheduling cannot invalidate completed loop evidence
        print(f"[first-grade] streaming promotion unavailable: {type(exc).__name__}: {exc}", flush=True)
    from merlin.development.targetgen.grading.scheduling import admission_relevance, timing_history
    from merlin.development.targetgen.grading.seeded_grade import workspace_gap

    progress = SnapshotProgress(ws, fruns, snapshot_digest, background_guard)
    from merlin.development.backend.feedback.campaign import curriculum as joint_curriculum
    from merlin.development.backend.feedback.scalability.rolling_checks import PerformanceFeedback
    from merlin.development.backend.feedback.scalability.rolling_checks import (
        Selection as ScalabilitySelection,
    )
    from merlin.development.backend.feedback.scalability.rolling_checks import lane as rolling_lane

    scalability = ScalabilitySelection(inputs.context) if JTS.focused() else None
    scalability_cases = scalability.cases if scalability is not None else {}
    # Large-case timing is its own lane: it outlives this grade and never delays its verdict.
    scale_feedback = (
        PerformanceFeedback(
            scalability,
            compiler=cand,
            runs_root=fruns,
            timeout=timeout,
            environment=grade_environment,
            lane=rolling_lane(run_dir),
        )
        if scalability_cases
        else None
    )
    curriculum = (
        joint_curriculum.native(
            run_dir,
            inputs=inputs,
            compiler_sha256=snapshot_digest,
            target=te.target,
            runs_root=fruns,
            runtime_config=readback_kwargs(inputs.context),
            additional_identity=scalability.identity,
            verify_additional=scalability.verify,
        )
        if scalability is not None
        else None
    )
    if curriculum is not None:
        curriculum.workspace = ws  # its certificate ledger says which cases await a decision (no sampling)

    def _process_completed(row):
        if scale_feedback is not None:
            # Advisory timing of one case: its failure loses that case's projection, never the grade.
            try:
                scale_feedback.completed(row)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                print(
                    f"[first-grade] rolling performance observation unavailable for {row.get('capsule')}: "
                    f"{type(error).__name__}: {str(error)[:160]}",
                    flush=True,
                )
        # Automatic checks retain the same public products as explicit selfchecks.
        # This passive path launches no timing observer and grants no fit authority.
        try:
            from merlin.development.backend.campaign.calibration import completed_calibration

            if row.get("capsule") not in scalability_cases:
                completed_calibration.register(
                    row,
                    result_path=fruns
                    / "runs"
                    / _suite_for(te.target)
                    / str(row.get("capsule", ""))
                    / "capsule_result.json",
                    compiler=cand,
                    capsules_root=public_root,
                    contract=inputs.contract,
                )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            print(f"[first-grade] calibration intake unavailable: {type(error).__name__}: {error}", flush=True)
        if curriculum is not None:
            try:
                curriculum.completed(row)
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f"[first-grade] curriculum receipt unavailable: {type(error).__name__}", flush=True)
        # An older snapshot's pass is promoted too: the ledger keys it on that snapshot's exact executable,
        # and promotion admission refuses native work whose executed bytes the current source changed.
        if row.get("capsule") not in scalability_cases and completed_promotion is not None:
            completed_promotion(row)
        publish_completed_cases()

    published_cases = set()

    def publish_completed_cases():
        # Each finished case reaches qa/latest_completed_grade.json now, not when this grade's slowest
        # case does (completed_grade.record_rows); completions that queued meanwhile share one write.
        rows = [_verdict_row(name, info) for name, info in list(progress.rows.items()) if name not in published_cases]
        if not rows:
            return
        published_cases.update(row["capsule"] for row in rows)
        try:
            completed_grade.record_rows(
                ws,
                rows,
                tick,
                graded_source_sha256=snapshot_digest,
                graded_at=_dt.datetime.now(_dt.UTC).isoformat(),
                reach_sha256=background_guard.reach_identity(),
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            print(f"[first-grade] per-case publication unavailable: {type(error).__name__}", flush=True)

    def process_completed(row):
        from merlin.development.backend.feedback.campaign.timing import environment_scope

        with environment_scope(grade_environment):
            return _process_completed(row)

    def completed(row):
        # Publish the native durable row immediately. Remaining consumers reopen
        # the same evidence on a request-owned worker, without blocking refill.
        progress.completed(row)
        completion_processing.submit(row)

    selected_roots, test_scope = JTS.select(
        public_root, submission=cand, curriculum=curriculum, additional_cases=scalability_cases
    )
    if joint_control.selected() is not None:
        remaining = joint_control.remaining()
        if remaining <= 0:
            selected_roots = []
            test_scope["execution_status"] = "original_campaign_deadline_exhausted"
        else:
            timeout = min(timeout, max(1, math.ceil(remaining)))
    history = _history_runs_roots(run_dir)
    rolling = (
        joint_curriculum.admission(
            curriculum, timeout=timeout, remaining_s=joint_control.remaining(drain=True), history=history
        )
        if curriculum is not None
        else nullcontext(timeout)
    )
    from merlin.development.backend.feedback.grading import grade_lanes
    from merlin.development.backend.feedback.grading.completions import CompletionProcessing

    grade_error = None
    try:
        with (
            rolling as case_timeout,
            compiler_execution(inputs, candidate=cand, run_dir=run_dir),
            timing_history(history),
            admission_relevance(
                source_sha256=snapshot_digest,
                reach=background_guard.case_reach,
                superseded=grade_lanes.relevance(),
                gap=workspace_gap(ws),
            ),
            CompletionProcessing(process_completed) as completion_processing,
        ):
            _fast_score = (
                {}
                if not selected_roots
                else (
                    _merlin_submission_1.grade(
                        str(cand),
                        capsules_root=selected_roots,
                        runs_root=str(fruns),
                        model_snapshot_root=fruns.resolve() / ".private_model_sources",
                        labels={"public", "dev"},
                        contract=str(
                            inputs.contract
                            if inputs.contract is not None
                            else inputs.context.repo / "src/merlin/resources/contracts"
                        ),
                        oracle_adapters=adapters,
                        timeout=case_timeout,
                        max_workers=_merlin_enforcement_1.default_grade_workers(),
                        target=te.target,
                        additional_forbidden=inputs.additional_forbidden,
                        defer_pending=background_guard.pending_reason,
                        defer_running=False,
                        on_capsule_complete=completed,
                        on_cohort_ready=progress.selected,
                    )
                    or {}
                )
            )
            if scalability is not None:
                scalability.verify()
    except Exception as e:  # noqa: BLE001 — a partial readback is still feedback, and every tier the
        # grade did not reach stays UNKNOWN in the document below rather than reading as clean.
        print(f"[first-grade] grade error: {type(e).__name__}: {str(e)[:200]}", flush=True)
        _fast_score = {}
        grade_error = f"{type(e).__name__}: {str(e)[:200]}"
    package_not_ready = (_fast_score.get("package_readiness") or {}).get("status") == "not_ready"
    red = {} if package_not_ready else _merlin_capsule_feedback_2._per_capsule_from_results(fruns)
    for row in _fast_score.get("per_capsule") or []:
        if (row.get("background_deferred") or row.get("status") == "gated") and row.get("capsule") not in red:
            red[row["capsule"]] = {
                "status": "gated",
                "tiers": {},
                **_qc.certification_gates.gated_row_fields(row, _qc._redact_detail),
            }
    from merlin.development.backend.feedback.grading.sampling import carried_rows

    red.update({name: row for name, row in carried_rows(test_scope).items() if name not in red})
    verdict = _fast_loop_verdict_doc(
        red,
        sorted(adapters),
        sorted(_withheld),
        structural_evidence=_fast_score.get("structural_evidence_all_pass"),
        cohort_complete=_qc._original_cohort_complete(_fast_score),
    )
    if scale_feedback is not None:
        # Only the observations already landed; the rest follow this grade's publication (below).
        scale_feedback.project(verdict, wait=False)
    if grade_error is not None:
        # An aborted grade reports its failure; publication keeps the last real per-test-case results
        # visible instead of replacing them with an empty grade (see visible_model_gate.publish).
        verdict["grade_error"] = {"error": grade_error, "source_sha256": snapshot_digest}
    if _fast_score.get("failure"):
        failure = _fast_score["failure"]
        verdict.update(
            all_pass=False,
            gradeable=False,
            integrity_status=_fast_score.get("integrity_status"),
            package_failure={
                "plane": failure.get("plane"),
                "category": failure.get("category"),
                "detail": _qc._redact_detail(str(failure.get("detail", ""))),
            },
            note="Compiler package rejected before capsule execution; selected obligations remain pending.",
        )
    if _fast_score.get("package_precheck_refusal"):
        verdict["package_precheck_refusal"] = {
            **_fast_score["package_precheck_refusal"],
            "source_sha256": snapshot_digest,
        }
    if package_not_ready:
        # Package bootstrap ran; no capsule or oracle did. This is readiness
        # feedback, never a reduced cohort or an empty passing grade.
        verdict.update(
            all_pass=False,
            gradeable=False,
            n_passed=0,
            n_capsules=0,
            per_capsule=[],
            mandatory_tiers_complete=False,
            tiers_graded=[],
            tiers_not_run=sorted(set(adapters) | set(_withheld)),
            package_readiness=_fast_score["package_readiness"],
            package_failure=_fast_score.get("failure"),
            note="The declared compiler tool is missing after package build; no capsules executed. "
            "Finish the declared tool; the first grade will retry shortly. The selected cohort is unchanged.",
        )
    verdict = _attribute_source(verdict, ws, snapshot_digest, stage="loop_screen")
    if background_guard.reason() is not None:
        verdict.update(
            all_pass=None,
            gradeable=False,
            mandatory_tiers_complete=False,
            background_grade={
                "status": "superseded_snapshot" if background_guard.finish_bounded_snapshot else "deferred",
                "reason": background_guard.deferred,
                "source_sha256": background_guard.source,
            },
        )
    verdict["graded_request_sha256"] = background_guard.source
    verdict["graded_reach_sha256"] = background_guard.reach_identity()  # reconciled per row (.reconcile)
    verdict = _attribute_source(verdict, ws, snapshot_digest, stage="loop_screen")
    if curriculum is not None and test_scope is not None:
        with measure("curriculum.report", verdict, events=run_dir / "_qa_work" / "host_stage_timings.jsonl"):
            test_scope["curriculum"] = curriculum.report()
        test_scope["pending"] = test_scope["curriculum"]["pending"]
    with measure("joint_test_scope.annotate", verdict, events=run_dir / "_qa_work" / "host_stage_timings.jsonl"):
        JTS.annotate(verdict, test_scope)
    with measure("progress.finish", verdict, events=run_dir / "_qa_work" / "host_stage_timings.jsonl"):
        progress.finish(verdict)
    out = run_dir / "qa_history" / f"verdict_fast_{tick}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    verdict = write_verdict(out, verdict)
    (ws / "qa").mkdir(exist_ok=True)
    archived = verdict
    with measure("completed_grade.publish", verdict, events=run_dir / "_qa_work" / "host_stage_timings.jsonl"):
        verdict = completed_grade.publish(ws, verdict, tick, write_verdict)
    if scale_feedback is not None:
        scale_feedback.follow_up(partial(completed_grade.follow_up, ws, out, archived, tick, write_verdict))
    _record_plateau(run_dir, current_source_sha256=verdict.get("current_source_sha256"))
    FL.record_channel_health(ws, run_dir)
    # The first successful loop verdict unlocks cert work immediately. The existing
    # broker queue owns native parallelism, deduplication and certification; never
    # wait for the periodic full-grade cadence or attribute these bytes to live edits.
    try:
        import sys as _sys

        from merlin.development.backend.feedback.certification.promotion import promote as _promote
        from merlin.development.backend.feedback.certification.promotion import resolve_tiers as _resolve

        _loop, _cert, _cover = _resolve(ws, context=inputs.context, capsules_root=inputs.promotion_root)
        if _loop and _cert and verdict.get("per_capsule") and background_guard.deferred is None:
            _promote(
                ws,
                ws / ".qa_channel",
                {
                    **verdict,
                    "per_capsule": [
                        row for row in verdict["per_capsule"] if row.get("capsule") not in scalability_cases
                    ],
                },
                _loop,
                _cert,
                _cover,
                _sys.stderr,
                source_ws=cand.parent,
                context=inputs.context,
                capsules_root=inputs.promotion_root,
            )
    except Exception as exc:  # noqa: BLE001 -- optional cert scheduling cannot erase loop feedback
        print(f"[first-grade] promotion skipped: {type(exc).__name__}: {exc}", flush=True)
    return verdict


def write_verdict(path: Path, verdict: dict, *, expect=None) -> dict | None:
    """Write a verdict WITH the time it was produced, and return it.

    Verdicts carried no timestamp of any kind. Under the continuous schedule the grader refreshes
    them while the agent works, so the sequence of verdicts IS the run's progress record -- and with
    no stamp inside, the only clock a reader has is the file's mtime. An mtime does not survive a
    copy, and several of these run trees have been copied between worktrees, so every progress curve
    drawn from them rests on filesystem metadata rather than on anything the run recorded.

    Stamping costs one field. `graded_at` is UTC and ISO-8601, matching the spelling every other
    record in the run tree uses. Existing verdicts are unaffected: readers fall back to the mtime and
    say so.

    ``expect(previous)`` guards a late projection onto the live verdict (:func:`.visible_model_gate.publish`).
    """
    from merlin.development.backend.campaign import control as joint_control

    verdict = dict(verdict)
    with measure("joint_control.stage_evidence", verdict):
        verdict = joint_control.stage_evidence(verdict)
    if path.name == "verdict.json" and path.parent.name == "qa":
        from merlin.development.backend.feedback.campaign.correctness import attach
        from merlin.development.backend.feedback.grading.failure_clusters import attach as attach_clusters
        from merlin.development.backend.feedback.performance.performance_headroom import (
            attach as attach_headroom,
        )

        with measure("attach_physical_correctness", verdict):
            verdict = attach(path, verdict)
        with measure("attach_performance_headroom", verdict):
            verdict = attach_headroom(path.parent.parent, verdict)
        with measure("attach_package_rejection", verdict):
            verdict = package_rejection.enrich(verdict)
        with measure("attach_failure_clusters", verdict):
            verdict = attach_clusters(path.parent.parent, verdict)
    verdict.setdefault("graded_at", _dt.datetime.now(_dt.UTC).isoformat())
    if path.name == "verdict.json" and path.parent.name == "qa":
        from merlin.development.backend.feedback.certification.visible_model_gate import publish
        from merlin.development.backend.feedback.execution.snapshots import publish_live

        return publish_live(
            path.parent.parent,
            lambda: publish(path, verdict, expect=expect),
            default=verdict if expect is None else None,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2))
    return verdict


def _record_plateau(run_dir: Path, *, current_source_sha256: str | None = None) -> None:
    """Assess this run's progress across GRADES and record it OPERATOR-SIDE. Never raises.

    OUT OF BAND, for the same reason `_write_stage_ledger` is: telling the agent it has plateaued, or
    naming the capsules that have never passed, is FEEDBACK -- and feedback defines an arm. Handing it
    over would change the treatment and make the run incomparable with every earlier one. This is for
    whoever decides whether to keep paying for the run.

    It exists because the detector that was here could not fire. `--plateau-rounds` counts consecutive
    ROUNDS with no progress, and the default schedule is `continuous` -- one long session, so exactly
    one round. Its threshold was unreachable in the mode every real run uses, and opt-in besides.
    Measured on merlincirct_g4p1_biasabi_20260906: 92/96 reached at 2.19 h, then 3.91 h more -- 64% of
    the run -- with the score never moving and nothing saying so. A grade is the unit the continuous
    schedule actually produces, so that is the unit here.
    """
    try:
        from merlin.development.targetgen.grading import plateau as _PL

        history = run_dir / "qa_history"
        # Keep pre-shape archives, but use the completed record for that exact
        # snapshot when available. A snapshot must never count as two grades.
        selected = {path.name.removeprefix("verdict_"): path for path in history.glob("verdict_*.json")}
        selected.update({path.name.removeprefix("completed_"): path for path in history.glob("completed_*.json")})
        hist = sorted(selected.values(), key=lambda q: q.stat().st_mtime)
        grades = []
        for q in hist:
            try:
                grades.append(json.loads(q.read_text()))
            except Exception:  # noqa: BLE001 -- one unreadable grade must not blind the assessment
                continue
        current_grades = [
            g
            for g in grades
            if current_source_sha256 is not None
            and g.get("graded_source_sha256") == current_source_sha256
            and g.get("snapshot_current") is True
        ]
        got = _PL.assess(current_grades)
        historical = _PL.assess(grades)
        doc = {
            "scope": "grades of the observed current source only",
            "current_source_sha256": current_source_sha256,
            "current_source_grade_available": bool(current_grades),
            "stuck": got.stuck,
            "reason": got.reason,
            "stalled_grades": got.stalled_grades,
            "n_grades": got.n_grades,
            "best_passed": got.best_passed if current_grades else None,
            "latest_passed": got.latest_passed if current_grades else None,
            "n_capsules": got.n_capsules,
            "never_passed": list(got.never_passed),
            "regressed": list(got.regressed),
            "sentence": got.sentence() if current_grades else "No grade for the observed current source is available.",
            "snapshot_history": {
                "scope": "immutable snapshots; not current-source qualification",
                "n_grades": historical.n_grades,
                "best_passed": historical.best_passed,
                "latest_passed": historical.latest_passed,
            },
            "note": (
                "operator-side only: this is never handed to the agent, because new feedback "
                "would change the arm's treatment"
            ),
        }
        (run_dir / "plateau.json").write_text(json.dumps(doc, indent=2))
        if got.stuck or got.regressed:
            print(f"[plateau] {got.sentence()}", flush=True)
    except Exception:  # noqa: BLE001 -- diagnostics may never fail a grade
        return


def _write_stage_ledger(
    run_dir,
    rnd: int,
    cand,
    runs_root,
    verdict,
    *,
    artifact_key: str | None = None,
    previous_artifact_key: str | None = None,
) -> None:
    """Record per-round artifact fingerprints beside the verdict — OUT OF BAND.

    Answers "did the agent's edit reach what was graded?" from bytes, so a plateau is one line instead of
    an investigation. Written into the run dir (a sandbox-DENIED path) and never into the agent's
    ``qa/verdict.json``: telling the agent its edit was inert is feedback, and feedback defines an arm.

    Never raises. This is diagnostics — it must not be able to fail a round that otherwise graded fine.
    """
    try:
        from merlin.development.targetgen.grading import run_ledgers as SL

        led_dir = run_dir / "rounds"
        led_dir.mkdir(parents=True, exist_ok=True)
        key = artifact_key or f"{rnd:02d}"
        if artifact_key is None:
            previous_key = f"{rnd - 1:02d}" if rnd else None
        else:
            previous_key = previous_artifact_key
        prev_p = led_dir / f"round_{previous_key}.stage_ledger.json" if previous_key is not None else None
        prev = json.loads(prev_p.read_text()) if prev_p is not None and prev_p.is_file() else None

        # Per-capsule emit dirs, found by SHAPE at any depth: a dir named for the emit output whose
        # PARENT is a graded capsule dir (it holds the capsule's own result/manifest). Depth and the
        # intervening dir names are never assumed -- no target, no suite, and no generated-root literal --
        # so a target that lays its run tree out differently is still discovered, and one that lays it out
        # unrecognizably simply yields none rather than a wrong answer.
        marks = ("capsule_result.json", "run_manifest.yaml")
        roots = {
            d.parent.name: d
            for d in Path(runs_root).rglob("generated")
            if d.is_dir() and any((d.parent / m).is_file() for m in marks)
        }
        led = SL.build(submission_dir=cand, emitted_roots=roots, previous=prev)
        led["round"] = rnd
        led["failing_and_frozen"] = SL.failing_and_frozen(led, verdict)
        (led_dir / f"round_{key}.stage_ledger.json").write_text(json.dumps(led, indent=2))
        print(f"  {SL.summarize(led)}", flush=True)
        if led["failing_and_frozen"]:
            print(
                f"  stage_ledger: FAILING AND FROZEN ({len(led['failing_and_frozen'])}): "
                f"{', '.join(led['failing_and_frozen'][:10])}"
                f"{' ...' if len(led['failing_and_frozen']) > 10 else ''}",
                flush=True,
            )
    except Exception as e:  # noqa: BLE001 - diagnostics never gate
        print(f"  stage_ledger: unavailable ({type(e).__name__}: {e})", flush=True)
