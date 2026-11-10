"""Finalize the author report, freeze/grade the result and publish completion evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from merlin.common import arrival_stamp as AS
from merlin.development.backend.audit import AnswerAudit
from merlin.development.backend.authoring_config import _PASS_LINE as _PASS_LINE
from merlin.development.backend.authoring_config import _STATUS_LINES as _STATUS_LINES
from merlin.development.backend.authoring_conformance import (
    workflow_conformance as _workflow_conformance,
)
from merlin.development.backend.authoring_report import finalization_instructions
from merlin.development.backend.authoring_session import AuthoringSession
from merlin.development.backend.feedback import brief as BRIEF
from merlin.development.backend.feedback import certification as CERT
from merlin.development.backend.feedback import lifecycle as FL
from merlin.development.backend.feedback import loop_grading as LG
from merlin.development.backend.providers import execution as EX
from merlin.development.backend.telemetry.report import (
    _emit_run_timing as _emit_run_timing,
)
from merlin.targetgen import experiment_tokens as ET


def _verified_status_line(verdict: dict) -> str:
    return BRIEF.verified_status_line(verdict, pass_line=_PASS_LINE)


def _stamp_report_status(report: Path, pass_line: str) -> bool:
    return BRIEF.stamp_report_status(report, pass_line, status_lines=_STATUS_LINES)


def finalize_report(
    ws: Path,
    run_dir: Path,
    model: str,
    effort: str,
    sandbox: str,
    bundle: dict,
    arm: str,
    verdict: dict,
    timeout: int,
    *,
    provider: EX.ProviderConfig,
    grade_callback,
    context,
    audit: AnswerAudit,
    reuse_noop_grade=None,
) -> dict:
    """After convergence, give the agent ONE bounded turn — with the PASSING verdict — to finalize
    REPORT.md/docs to the verified result, WITHOUT touching code. Re-grade; if the agent regressed the
    package, restore the pre-finalize (passing) submission. Always guarantee the report's final status
    line matches the verdict (programmatic stamp as a last resort). Returns finalize telemetry."""
    snap = run_dir / "_qa_work" / "pre_finalize" / "submission"
    if snap.exists():
        shutil.rmtree(snap.parent)
    snap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ws / "submission", snap, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))

    LG.write_verdict(ws / "qa" / "verdict.json", verdict)
    pass_line = _verified_status_line(verdict)
    model_report = "model_gate" in verdict
    (ws / "FINALIZE.md").write_text(finalization_instructions(model_report, pass_line))
    tpath = run_dir / "rounds" / "finalize.transcript.jsonl"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    epath = run_dir / "rounds" / "finalize.stderr.log"
    rc = 0
    if EX.resolve_driver(model, config=provider) == "claudecode":
        inner = (
            f"claude --print --model {model} --effort {effort} "
            f"--permission-mode bypassPermissions --add-dir {ws} "
            f"--output-format stream-json --verbose < {ws / 'FINALIZE.md'}"
        )
        cmd = (
            EX.sandbox_command(inner, ws, bundle, context=context, private_run_dir=run_dir)
            if sandbox == "bwrap"
            else inner
        )
        try:
            rc = AS.stream_stamped(
                ["bash", "-c", cmd],
                cwd=ws,
                transcript=tpath,
                stderr_path=epath,
                timeout=timeout,
                raw_path=tpath.with_name("finalize.stream.raw.jsonl"),
            )
        except subprocess.TimeoutExpired:
            rc = 124
    else:
        # Converse / OpenCode: the claude CLI can't drive these models, so skip the finalize turn; the driver
        # stamps REPORT.md's status (_stamp_report_status). An empty transcript keeps audit_transcript happy.
        tpath.write_text("")

    # re-grade: if the finalize turn broke the (passing) package, restore the snapshot
    reused = (
        EX.resolve_driver(model, config=provider) == "codex" and reuse_noop_grade is not None and reuse_noop_grade()
    )
    regrade = verdict if reused else grade_callback(ws, run_dir, 90, False, timeout)
    restored = False
    if not regrade.get("all_pass"):
        shutil.rmtree(ws / "submission")
        shutil.copytree(snap, ws / "submission")
        restored = True
    # guarantee the frozen report's status line matches the verified verdict
    stamped = _stamp_report_status(ws / "submission" / "REPORT.md", pass_line)
    result = audit.audit_transcript(tpath, arm, workspace=ws)
    return {
        "agent_rc": rc,
        **({"regrade_reused": True} if reused else {}),
        "regrade_all_pass": regrade.get("all_pass"),
        "restored_after_regression": restored,
        "status_line_stamped_by_driver": stamped,
        "answer_access_clean": result["clean"],
        "audit_hits": result["hits"],
        "transcript": str(tpath),
    }


def complete(session: AuthoringSession) -> int:
    progress = session.progress
    options = session.options
    workspace = session.workspace
    run_dir = session.run_dir
    context = session.context
    arm = session.arm
    bundle = session.bundle
    provider = session.provider
    # The authoring tree may have changed in an L3 fix round (or this process may have resumed from an
    # older checkpoint).  Refresh once more before treating it as converged.  No transcript means no
    # persisted tool evidence and therefore fails closed.
    if progress.latest_transcript is not None:
        final_conformance, progress.workflow_conformant = _workflow_conformance(
            progress.latest_transcript, workspace / "submission", arm, session.endpoint_kind, session.resolved_tools()
        )
    else:
        final_conformance = {"conformant": False, "error": "no authoring transcript"}
        progress.workflow_conformant = False
    session.checkpoint(progress.round_index)

    # On convergence, give the agent ONE bounded turn to finalize its own REPORT.md/docs to the
    # VERIFIED result (the relaunch design means the converging round's agent never saw the passing
    # verdict, so its self-reported status lags by one round). Guarantees the frozen report matches.
    finalize = None
    if (
        session.authoring_complete()
        and context.target == "gemmini"  # target-ok: Gemmini scope
        and options.experiment == "full"
    ):
        # A report-finalize turn and the driver's status-line stamp change submission bytes after the
        # last agent-source OOT commit. The model task asks for the report during authoring; preserve
        # that exact tree for the official grade and record the visible model status here.
        finalize = {
            "skipped": "gemmini model report belongs to the agent's final authoring tree",  # target-ok: Gemmini scope
            "visible_model_status_line": _verified_status_line(progress.verdict),
            "status_line_stamped_by_driver": False,
        }
    elif session.authoring_complete():
        session.verify_inputs()
        print("[finalize] converged — running bounded report-finalize turn")
        _fin_start = time.time()
        finalize = finalize_report(
            workspace,
            run_dir,
            options.model,
            options.effort,
            options.sandbox,
            bundle,
            arm,
            progress.verdict,
            min(options.round_timeout, 900),
            provider=provider,
            context=context,
            audit=session.audit,
            grade_callback=session.grade,
            reuse_noop_grade=lambda: (
                session.driver_grade_receipt[0] is not None
                and session.driver_grade_receipt[0].current(
                    submission=workspace / "submission", inputs=session.grading_inputs, run_dir=run_dir
                )
            ),
        )
        progress.active_wall_s += time.time() - _fin_start  # finalize is active work
        print(
            f"[finalize] regrade_all_pass={finalize['regrade_all_pass']} "
            f"restored={finalize['restored_after_regression']} "
            f"stamped={finalize['status_line_stamped_by_driver']}"
        )
        # Finalize is instructed to edit docs only, but the submission is not trusted until checked.  Scan
        # the final bytes against the last authoring workflow so code changes/regressions cannot inherit a
        # pre-finalize conformance True.
        final_conformance, progress.workflow_conformant = _workflow_conformance(
            progress.latest_transcript, workspace / "submission", arm, session.endpoint_kind, session.resolved_tools()
        )
        session.checkpoint(progress.round_index)
    # cumulative wall = active work (rounds + finalize, across ALL invocations) + rate-limit sleeps
    wall = round(progress.active_wall_s + progress.rate_limit_wait_s, 3)

    # combined telemetry across all rounds + finalize (total effort to produce the deliverable)
    combined = run_dir / "transcript.jsonl"
    with open(combined, "w") as out:
        for tp in sorted((run_dir / "rounds").glob("round_*.transcript.jsonl")):
            out.write(tp.read_text())
        ftp = run_dir / "rounds" / "finalize.transcript.jsonl"
        if ftp.exists():
            out.write(ftp.read_text())
    summ = ET.parse_transcript(
        combined,
        billing_mode=EX.billing_mode(options.model, config=provider),
        trust_cli_cost=EX.trust_cli_cost(options.model, config=provider),
    )
    # active-vs-waiting split (cumulative across resume-invocations): wall = active work + rate-limit
    # sleeps; active_wall_s is time actually DOING work (agent rounds + oracle grading + finalize).
    progress.active_wall_s = round(progress.active_wall_s, 3)
    timing = {
        "wall_seconds": wall,
        "active_wall_s": progress.active_wall_s,
        "rate_limit_wait_s": round(progress.rate_limit_wait_s, 3),
        "rate_limit_waits_used": progress.rate_limit_waits_used,
        "started_at": progress.started_at,
        "resumed": session.resuming,
    }
    ET.write_cost_yaml(
        summ, run_dir / "cost_time_toolcalls.yaml", wall_time_seconds=wall, model=options.model, exit_code=0
    )
    # append the active-vs-waiting split to the cost yaml (write_cost_yaml owns the rest)
    _cy = run_dir / "cost_time_toolcalls.yaml"
    _cd = yaml.safe_load(_cy.read_text()) or {}
    _cd.update(
        {
            "active_wall_s": progress.active_wall_s,
            "rate_limit_wait_s": round(progress.rate_limit_wait_s, 3),
            "rate_limit_waits_used": progress.rate_limit_waits_used,
        }
    )
    _cy.write_text(yaml.safe_dump(_cd, sort_keys=False))
    qa_summary = {
        "rounds": progress.rounds,
        "authoring_complete": session.authoring_complete(),
        "numeric_all_pass": bool(progress.verdict.get("all_pass", False)),
        "workflow_conformant": progress.workflow_conformant,
        "final_conformance": final_conformance,
        "cost_capped": progress.cost_capped,  # stopped by the batch dollar ceiling, not convergence/max_rounds
        "n_rounds": len(progress.rounds),
        "wall_seconds": wall,
        "finalize": finalize,
        "timing": timing,
        "run_config": session.environment.get("run_config", session.run_config),
        "operator_seal": progress.operator_seal,
    }

    feedback_health = FL.channel_health(workspace)
    qa_summary["feedback_health"] = feedback_health

    # Additionally sink this run's agentic telemetry into the shared aet store (opt-in,
    # MERLIN_AET_SINK=1) so it shows up in `aet spend` / `aet plot` across experiments. This is
    # purely additive — the existing experiment_tokens cost yaml above stays authoritative.
    from merlin.targetgen import aet_bridge as AB

    if AB.aet_sink_enabled(run_dir):
        AB.emit_to_aet(
            run_dir=run_dir,
            run_id=options.run_id,
            method=arm,
            model=options.model,
            target=session.target_experiment().target,
            suite="capsule-bench",
            transcript_paths=[combined],
            billing_mode=EX.billing_mode(options.model, config=provider),
        )

    # capture the final submission and run the OFFICIAL public+hidden record
    wsub = workspace / "submission"
    official_grade = {
        "complete": False,
        "grader_returncode": None,
        "manifest": str(run_dir / "run_manifest.yaml"),
        "failures": ["submission_missing"],
    }
    if wsub.exists():
        dst = run_dir / "submission"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(wsub, dst, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
    incomplete_operator_seal = bool(progress.operator_seal)
    if wsub.exists() and (progress.workflow_conformant or incomplete_operator_seal):
        # Last gate before the official process performs its public grade + freeze + hidden grade.  The
        # authoring loop can last across worktree edits and quota-window resumes, so setup-time hashing is
        # not enough: recompute every treatment byte and the private snapshot content now, fail closed on
        # drift, and hand the grader only the already-frozen hidden path.
        from merlin.development.backend import formal_invocation

        try:
            grade_rc = formal_invocation.run(
                session.prepared,
                public_capsules=CERT._public_capsules(session.grading_inputs),
                selected_rtl_facts=session.execution.selected_rtl_facts,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"official grade refused: {exc}") from exc
        _manifest_path = run_dir / "run_manifest.yaml"
        if _manifest_path.is_file():
            _manifest_doc = yaml.safe_load(_manifest_path.read_text(encoding="utf-8")) or {}
            _manifest_doc["run_config"] = session.environment.get("run_config", session.run_config)
            _manifest_doc["feedback_health"] = feedback_health
            _manifest_path.write_text(yaml.safe_dump(_manifest_doc, sort_keys=False))
        from merlin.development.backend.feedback import hidden_capsule_policy as HCP
        from merlin.development.backend.feedback import (
            private_full_model_execution as PFX,
        )
        from merlin.development.backend.feedback import private_full_models as PFM

        official_grade = CERT._selected_grade_result(
            grade_rc,
            run_dir,
            validation_scope=getattr(options, "validation_scope", "full"),
            descriptor=context.descriptor,
            require_model_check=context.target == "gemmini",  # target-ok: Gemmini scope
            required_models=PFM.requirements_for(context.descriptor),
            required_programs=PFM.program_requirements_for(context.descriptor),
            execution_gate=PFX.gate_for(
                context.descriptor, required_programs=PFM.program_requirements_for(context.descriptor)
            ),
            hidden_capsule_mode=HCP.resolve(context.descriptor, getattr(options, "hidden_capsule_mode", ""))["mode"],
        )
    elif wsub.exists():
        official_grade["failures"] = ["workflow_nonconformant"]

    # Telemetry is part of the experiment result, not a best-effort epilogue. Seal mutable rollout /
    # broker evidence and derive the unified record BEFORE deciding formal completion. Any writer or
    # integrity failure remains a named incomplete record and prevents a "complete" claim.
    _telemetry_path = run_dir / "timing_detailed.json"
    try:
        _telemetry_record = _emit_run_timing(run_dir, progress.rounds)
        _telemetry_integrity = _telemetry_record.get("telemetry_integrity") or {
            "complete": False,
            "failures": ["telemetry_integrity_missing"],
        }
    except Exception as exc:  # noqa: BLE001 — fail closed into the durable summaries below
        _telemetry_integrity = {
            "complete": False,
            "failures": [f"telemetry_writer_failed:{type(exc).__name__}:{exc}"],
        }
        _telemetry_path.write_text(json.dumps({"telemetry_integrity": _telemetry_integrity}, indent=2))
    _telemetry_bytes = _telemetry_path.read_bytes()
    telemetry = {
        "complete": _telemetry_integrity.get("complete") is True,
        "failures": list(_telemetry_integrity.get("failures") or []),
        "path": str(_telemetry_path),
        "sha256": hashlib.sha256(_telemetry_bytes).hexdigest(),
        "bytes": len(_telemetry_bytes),
    }
    qa_summary.update(
        CERT._completion_for_scope(
            validation_scope=getattr(options, "validation_scope", "full"),
            numeric_all_pass=bool(progress.verdict.get("all_pass")),
            workflow_conformant=progress.workflow_conformant,
            official_grade=official_grade,
            telemetry=telemetry,
            checkpoint=incomplete_operator_seal,
        )
    )
    (run_dir / "qa_loop_summary.yaml").write_text(yaml.safe_dump(qa_summary, sort_keys=False))
    # The official grader wrote the manifest before process telemetry existed. Bind that final process
    # record into the same manifest and ensure its top-level completion claim cannot contradict the QA
    # summary when telemetry is incomplete.
    _manifest_path = run_dir / "run_manifest.yaml"
    if _manifest_path.is_file():
        _manifest_doc = yaml.safe_load(_manifest_path.read_text(encoding="utf-8")) or {}
        _manifest_doc["telemetry"] = telemetry
        _completion = _manifest_doc.setdefault("completion", {})
        _completion["telemetry_complete"] = telemetry["complete"]
        _completion["phase2_iteration_ready"] = qa_summary.get("phase2_iteration_ready", False)
        if not telemetry["complete"]:
            _completion["formal_grade_complete"] = False
        _manifest_path.write_text(yaml.safe_dump(_manifest_doc, sort_keys=False))
    # The detailed usage record is written before the potentially long outer grade so the grader can
    # include it in its manifest.  Correct its launcher exit code once the authoritative grade exists.
    _cd = yaml.safe_load(_cy.read_text()) or {}
    _cd["exit_code"] = 0 if qa_summary["converged"] else 1
    _cy.write_text(yaml.safe_dump(_cd, sort_keys=False))
    from merlin.development.backend.capsule_timing_archive import finalize_run

    _capsule_timing = finalize_run(run_dir)
    qa_summary["capsule_timing"] = _capsule_timing
    (run_dir / "qa_loop_summary.yaml").write_text(yaml.safe_dump(qa_summary, sort_keys=False))
    if _manifest_path.is_file():
        _manifest_doc["capsule_timing"] = _capsule_timing
        _manifest_path.write_text(yaml.safe_dump(_manifest_doc, sort_keys=False))
    qa_summary = CERT._publish_capsule_handoff(
        run_dir,
        qa_summary,
        validation_scope=getattr(options, "validation_scope", "full"),
        descriptor=context.descriptor,
        transcript=progress.latest_transcript,
        endpoint_kind=session.endpoint_kind,
        tools=session.resolved_tools(),
    )
    print(
        f"\nrun complete: {run_dir}  numeric_all_pass={progress.verdict.get('all_pass')} "
        f"workflow_conformant={progress.workflow_conformant} "
        f"official_grade_complete={official_grade['complete']} formal_complete={qa_summary['formal_complete']} "
        f"rounds={len(progress.rounds)} capsule_timing_ready={_capsule_timing['timing_ready']}"
    )
    return 0 if qa_summary["converged"] else 1
