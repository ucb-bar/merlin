"""Authoring, checkpoint/resume and completion for an admitted Phase 1 run.

The native edge still owns admission. This engine has no native-controller imports;
ordinary rounds, cert repairs and completion retain one shared continuation state.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from merlin.common import arrival_stamp as AS
from merlin.common.tree_hash import hash_tree
from merlin.targetgen import experiment_tokens as ET
from merlin.targetgen.target_experiment import load_capability_manifest, load_target_experiment
from merlin_experiments.phase1 import recovery as ROQ
from merlin_experiments.phase1 import run_inputs as RI
from merlin_experiments.phase1 import treatments as T
from merlin_experiments.phase1.audit import AnswerAudit
from merlin_experiments.phase1.feedback import certification as CERT
from merlin_experiments.phase1.feedback import lifecycle as FL
from merlin_experiments.phase1.feedback import loop_grading as LG
from merlin_experiments.phase1.providers import execution as EX
from merlin_experiments.phase1.session import task_scope

if TYPE_CHECKING:
    from merlin_experiments.phase1.session import PreparedRun

READY_MARKER = "READY_FOR_BARRIER"
VERILATOR_ATTEMPTS = 3


@dataclass(frozen=True)
class AuthoringRuntime:
    """Native edge selections not already recorded by PreparedRun.

    public_root captures the legacy override at invocation time. Its descriptor
    fallback is still materialized lazily, only when grading needs public capsules.
    """

    bundle_id: str
    oracle_timing: Path
    public_root: Path | None = None


def _conformance_ever(rounds) -> dict:
    """Fold the run-level gate: true if ever satisfied, None if always inapplicable.

    Resume rebuilds this same view from persisted round records.
    """
    ever: dict = {}
    for entry in rounds or []:
        checks = ((entry or {}).get("conformance") or {}).get("checks") or {}
        for key, value in checks.items():
            if value is None:
                ever.setdefault(key, None)
            else:
                ever[key] = bool(ever.get(key)) or bool(value)
    return ever


def _workflow_conformance(
    tpath: Path, submission_dir: Path, arm: str, endpoint_kind: str, resolved_tools
) -> tuple[dict, bool]:
    """Recompute current workflow evidence from resolved tools; checker errors fail closed."""
    try:
        from merlin_experiments.phase1 import conformance as _CONF

        conf = _CONF.compute(tpath, submission_dir, arm, endpoint_kind, resolved_tools=resolved_tools)
    except Exception as exc:  # noqa: BLE001 — conformance is a formal gate; unavailable means false
        conf = {"conformant": False, "error": f"{type(exc).__name__}: {exc}"}
    return conf, conf.get("conformant") is True


def _spend_over_cap(this_round_cost) -> tuple[bool, float, float]:
    """Append subagent-inclusive cost to MERLIN_SPEND_LEDGER; return (over_cap, total, cap).

    MERLIN_MAX_SPEND_USD is a soft cap: one in-flight round per arm may overshoot.
    Missing cap/ledger disables it; unknown usage remains a visible lower bound.
    """
    import os as _os

    cap = float(_os.environ.get("MERLIN_MAX_SPEND_USD") or 0)
    ledger = _os.environ.get("MERLIN_SPEND_LEDGER")
    if cap <= 0 or not ledger:
        return False, 0.0, 0.0
    import fcntl

    # Missing usage is unknown, never zero; a timeout may prevent the terminal usage event.
    _unmeasured = this_round_cost is None
    c = None if _unmeasured else float(this_round_cost)
    p = Path(ledger)
    p.parent.mkdir(parents=True, exist_ok=True)
    total, n_unmeasured = 0.0, 0
    with open(p, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({"cost": c, "unmeasured": _unmeasured}) + "\n")
        f.flush()
        f.seek(0)
        for line in f:
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 — a malformed ledger line must not defeat the cap
                continue
            if row.get("unmeasured") or row.get("cost") is None:
                n_unmeasured += 1
                continue
            try:
                total += float(row.get("cost") or 0)
            except Exception:  # noqa: BLE001
                continue
        fcntl.flock(f, fcntl.LOCK_UN)
    if n_unmeasured:
        print(
            f"  [spend] ${total:.2f} of ${cap:.2f} measured, plus {n_unmeasured} UNMEASURED round(s) "
            f"whose usage never arrived — the true total is a LOWER BOUND",
            flush=True,
        )
    return total >= cap, total, cap


def _grading_public_root(context, public_root: Path | None = None):
    """Bind legacy public selection now; materialize lazily and cache per invocation."""
    selected = public_root
    if selected is not None:
        return selected
    descriptor = context.descriptor
    resolved = None

    def resolve():
        nonlocal resolved
        if resolved is None:
            from merlin.targetgen.target_experiment import load_target_experiment
            from merlin_experiments.corpus.admission import public_capsules_for

            resolved = public_capsules_for(load_target_experiment(descriptor))
        return resolved

    return resolve


def policy_roots(context, policy_root: Path | None = None) -> tuple[Path, ...]:
    """Resolve the native edge's descriptor fallback, preserving unresolved-tier behavior."""
    if policy_root is not None:
        return (policy_root,)
    try:
        from merlin.targetgen.target_experiment import load_target_experiment

        te = load_target_experiment(context.descriptor)
        return tuple(te.graded_roots())
    except Exception:  # noqa: BLE001 — unresolved declarations historically select the legacy ladder
        return ()


_PASS_LINE = "Baseline pilot passes all required public/dev pilot capsules and is ready for hidden pilot grading."
_STATUS_LINES = (
    _PASS_LINE,
    "Baseline pilot does not yet pass all required public/dev pilot capsules; remaining failures "
    "are listed by capsule and failure plane.",
    "Baseline pilot is not comparable because it violates the compiler/runtime/integrity boundary.",
)


def _stamp_report_status(report: Path, pass_line: str) -> bool:
    """Last-resort guarantee: ensure REPORT.md's final status line states the VERIFIED result. The
    multi-round relaunch grades AFTER the agent exits, so the converging round's agent never saw a
    passing verdict and may leave a stale 'not yet passing' line. Returns True if it had to rewrite."""
    if not report.exists():
        report.write_text(f"# REPORT\n\n## Final status line\n{pass_line}\n")
        return True
    txt = report.read_text()
    if pass_line in txt:
        return False
    new = txt
    for sl in _STATUS_LINES:
        if sl != pass_line and sl in new:
            new = new.replace(sl, pass_line)
    if new == txt:  # no known status line present -> append one
        new = txt.rstrip() + f"\n\n## Final status line (verified)\n{pass_line}\n"
    report.write_text(new)
    return True


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
    (ws / "FINALIZE.md").write_text(
        "All required public/dev pilot capsules now PASS (see qa/verdict.json: all_pass=true).\n\n"
        "Do ONLY this, then stop:\n"
        "1. Update `submission/REPORT.md` so its final status line reads EXACTLY:\n"
        f'   "{_PASS_LINE}"\n'
        "   and its body honestly reflects the verified per-capsule pass + oracle tiers.\n"
        "2. Update `submission/docs/iteration_notes.md` with the final converged state.\n"
        "DO NOT modify manifest.yaml, mlir_oot/, or any code — the package is frozen-pending and must\n"
        "keep passing. Touch only REPORT.md and docs/.\n"
    )
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
        cmd = EX.sandbox_command(inner, ws, bundle, context=context) if sandbox == "bwrap" else inner
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
        # Converse / OpenCode: the claude CLI can't drive these models, so skip the agent finalize turn —
        # the driver stamps REPORT.md's status line below (_stamp_report_status). An empty transcript keeps
        # audit_transcript happy (no tokens, no answer-access hits).
        tpath.write_text("")

    # re-grade: if the finalize turn broke the (passing) package, restore the snapshot
    regrade = grade_callback(ws, run_dir, 90, False, timeout)
    restored = False
    if not regrade.get("all_pass"):
        shutil.rmtree(ws / "submission")
        shutil.copytree(snap, ws / "submission")
        restored = True
    # guarantee the frozen report's status line matches the verified verdict
    stamped = _stamp_report_status(ws / "submission" / "REPORT.md", _PASS_LINE)
    result = audit.audit_transcript(tpath, arm, workspace=ws)
    return {
        "agent_rc": rc,
        "regrade_all_pass": regrade.get("all_pass"),
        "restored_after_regression": restored,
        "status_line_stamped_by_driver": stamped,
        "answer_access_clean": result["clean"],
        "audit_hits": result["hits"],
        "transcript": str(tpath),
    }


def execute(prepared: PreparedRun, runtime: AuthoringRuntime) -> int:
    """Continue the existing authoring/completion policy inside the caller's workspace lease."""
    from merlin.targetgen.sandbox import bwrap as _BWS

    a, treatment = prepared.request.options, prepared.request.treatment
    arm = a.arm
    bundle_id = runtime.bundle_id
    context = prepared.request.context
    _te = partial(load_target_experiment, context.descriptor)
    _manifest = partial(load_capability_manifest, context.target)
    audit = AnswerAudit.for_descriptor(_te(), prepared.request.bundle_manifest.parent)
    resolved_tools = prepared.request.resolved_tools
    ws, run_dir, bundle_dir = prepared.workspace, prepared.run_dir, prepared.bundle_dir
    bundle, _environment_record = prepared.bundle, prepared.environment
    _public_root, _policy_root, _contract_root = prepared.public_root, prepared.policy_root, prepared.contract_root
    _resuming = prepared.resuming
    _scope_roots = prepared.scope_roots
    _operator_errata_record = _environment_record.get("operator_errata")
    _run_config = prepared.request.run_config
    _verify_implementation_sources = prepared.verify_inputs
    provider = EX.ProviderConfig(a.driver, a.provider, a.subagent_model, a.background_model)
    execution = EX.ExecutionConfig(
        prepared.request.context,
        provider,
        prepared.request.resolved_tools,
        runtime.oracle_timing,
        max(0, int(a.sim_max_jobs)),
    )

    grading_inputs = LG.GradingInputs(
        context=prepared.request.context,
        arm=arm,
        public_root=_grading_public_root(context, _public_root if _public_root is not None else runtime.public_root),
        policy_roots=policy_roots(context, _policy_root),
        contract=_contract_root,
        promotion_root=_policy_root,
        additional_forbidden=("xdsl_dialects.lowering", "xdsl_dialects/lowering", "outputs_match")
        if a.experiment == "realistic" and arm == "merlin_assisted"
        else (),
    )
    _grade = T.timed(
        partial(
            LG.grade,
            inputs=grading_inputs,
            qa_runner=treatment.qa_runner,
        ),
        "qa",
        treatment.on_duration,
    )
    _launch = T.timed(
        partial(
            EX.launch, config=execution, capsules_root=_public_root, policy_root=_policy_root, contract=_contract_root
        ),
        "agent",
        treatment.on_duration,
    )
    _fast_grade = partial(LG.fast_grade, inputs=grading_inputs)
    # realistic (abc2): the self-check tool the agent runs logs each invocation here (dev-trajectory /
    # soft-failure / tool-trigger record). T0 anchors wall-offset. Inherited by every claude subprocess.
    if a.experiment == "realistic":
        os.environ["SELFCHECK_LOG"] = str(run_dir / "selfcheck_log.jsonl")
        os.environ.setdefault("SELFCHECK_T0", str(time.time()))

    # --- durable checkpoint so the experiment AND its accumulated time survive a process death ---
    # (in-process backoff handles a quota hit while alive; this handles reboot / session-end / OOM).
    # cumulative timing is split into active work vs rate-limit waiting and persisted after EACH round
    # and EACH wait, so a fresh --resume invocation continues exactly where it stopped — not from 0.
    state_p = run_dir / "qa_loop_state.yaml"
    rounds_summary: list = []
    # Mandatory startup activities may be evidenced in any round, not necessarily the last.
    _conf_ever: dict = {}

    def _conformant_over_run() -> bool:
        """Every check this arm mandates satisfied in SOME round (never all in one -- see _conf_ever).

        Defined BEFORE the round loop on purpose: the exit path that this rule exists to unblock is the
        READY-marker clear inside the loop, not only the post-loop barrier. `4e5ab2df` added this helper
        after the loop and wired the barrier; `622ac429` then rebuilt the barrier from a base without the
        fix, and the call sites reverted to the per-round flag while the helper and its test survived --
        so the original defect (a converged run whose late rounds cannot satisfy `cca_used`, looping
        until something else stops it) was live again.
        """
        applicable = [v for v in _conf_ever.values() if v is not None]
        return all(applicable) if applicable else True

    try:  # endpoint_kind — drives the per-round dev-conformance flag (asm applies only to external_backend)
        from merlin.targetgen.generate_prompt import prompt_slots as _pslots

        _endpoint_kind = _pslots(_te(), _manifest()).get("endpoint_kind", "")
    except Exception:  # noqa: BLE001
        _endpoint_kind = ""
    _best_progress = None  # plateau early-stop: best (#passed, -total_mismatch) seen so far
    _plateau_stall = 0  # consecutive rounds with no progress
    rl_waits_used = 0

    def _progress_key(v: dict) -> tuple:
        """Round progress, higher is better: (#passed, -total residual numeric mismatch). A non-passing
        capsule with no numeric mismatch (a structural fail) counts as a large residual, so a structural
        stall never reads as 'solved'. Used only to detect a plateau — never to grade."""
        tot = 0
        for pc in v.get("per_capsule") or []:
            if pc.get("status") == "pass":
                continue
            mc = pc.get("mismatch_count")
            tot += int(mc) if isinstance(mc, int) else 1_000_000
        return (v.get("n_passed") or 0, -tot)

    active_wall_s = 0.0  # cumulative time DOING work (launch+grade) across all invocations
    rate_limit_wait_s = 0.0  # cumulative time slept waiting for five-hour window resets
    started_at = datetime.now(_dt.UTC).isoformat()
    verdict = {"all_pass": False}
    workflow_conformant = False
    rnd = 0
    # Survives a --resume where the numeric/workflow checkpoint is already complete and the normal round
    # loop therefore does not launch again.  ``finalize.transcript`` is deliberately excluded: it is a
    # docs-only turn, not the authoring workflow whose mandatory tools must be evidenced.
    _saved_authoring_transcripts = sorted((run_dir / "rounds").glob("round_*.transcript.jsonl"))
    _latest_authoring_tpath = (
        None if a.seal_current else ROQ.latest_live_authoring_transcript(_saved_authoring_transcripts)
    )
    if _resuming and state_p.exists():
        st = yaml.safe_load(state_p.read_text()) or {}
        rounds_summary = st.get("rounds", []) or []
        rnd = int(st.get("next_round", 0))
        verdict = {"all_pass": bool(st.get("converged", False))}
        workflow_conformant = bool(st.get("workflow_conformant", False))
        # Rebuild the RUN-LEVEL accumulator the gates read. Without this a --resume starts it empty, so a
        # run that had already evidenced every mandated check is blocked by its own fresh accumulator --
        # the same shape of defect as gating on the current round, just moved into the resume path.
        # Prefer the persisted map; fall back to re-accumulating from the round records, so a checkpoint
        # written before this field existed still resumes correctly.
        _saved_ever = st.get("conformance_ever")
        if isinstance(_saved_ever, dict) and _saved_ever:
            _conf_ever.update(_saved_ever)
        else:
            _conf_ever.update(_conformance_ever(rounds_summary))
        cum = st.get("cumulative", {}) or {}
        active_wall_s = float(cum.get("active_wall_s", 0.0))
        rate_limit_wait_s = float(cum.get("rate_limit_wait_s", 0.0))
        rl_waits_used = int(cum.get("rl_waits_used", 0))
        started_at = cum.get("started_at", started_at)
        print(
            f"[resume] restored from checkpoint: next_round={rnd} converged={verdict['all_pass']} "
            f"workflow_conformant={workflow_conformant} over_run={_conformant_over_run()} "
            f"active={active_wall_s:.0f}s rate_limit_wait={rate_limit_wait_s:.0f}s "
            f"waits_used={rl_waits_used}"
        )

    operator_seal = None
    if a.seal_current:
        if not rounds_summary:
            raise RuntimeError("--seal-current requires at least one completed, audited round")
        if not (ws / "submission" / "manifest.yaml").is_file():
            raise RuntimeError("--seal-current requires a checkpointed submission with manifest.yaml")
        # An interrupted in-flight round may have a partial transcript on disk.  Conformance must be
        # derived from the last round the checkpoint actually records, never from unaudited tail bytes.
        completed_rounds = sorted(
            {
                int(row["round"])
                for row in rounds_summary
                if isinstance(row, Mapping) and isinstance(row.get("round"), int)
            }
        )
        if not completed_rounds:
            raise RuntimeError("--seal-current checkpoint names no completed round")
        completed_path = run_dir / "rounds" / f"round_{completed_rounds[-1]:02d}.transcript.jsonl"
        if not completed_path.is_file():
            raise RuntimeError(f"--seal-current completed-round transcript is absent: {completed_path}")
        _latest_authoring_tpath = completed_path
        operator_seal = {
            "version": 1,
            "requested": True,
            "reason": "operator accepted an incomplete functional baseline for named-waiver admission",
            "requested_at": datetime.now(_dt.UTC).isoformat(),
            "last_completed_round": completed_rounds[-1],
            "completed_rounds": len(rounds_summary),
            "checkpoint_submission_sha256": hash_tree(ws / "submission")["sha256"],
        }
        print(
            f"[operator-seal] authoring stopped at completed round {completed_rounds[-1]}; "
            "continuing through official grade + immutable freeze (NOT convergence)",
            flush=True,
        )

    def _authoring_complete() -> bool:
        """Pre-freeze gate: enough evidence to stop editing and begin the official grade."""
        return CERT._authoring_completion(bool(verdict.get("all_pass")), workflow_conformant)

    def _checkpoint(next_round: int) -> None:
        state_p.write_text(
            yaml.safe_dump(
                {
                    "run_id": a.run_id,
                    "arm": arm,
                    "model": a.model,
                    "effort": a.effort,
                    "rounds": rounds_summary,
                    "next_round": next_round,
                    "converged": _authoring_complete(),
                    "numeric_all_pass": bool(verdict.get("all_pass", False)),
                    "workflow_conformant": workflow_conformant,
                    # RUN-LEVEL view, persisted because the gates now read it: `_conf_ever` lives only in memory,
                    # so a --resume restarted it empty and a run that had ALREADY earned every mandated check
                    # would be blocked by its own fresh accumulator. The per-round flag above stays -- it is how
                    # you see WHICH round did the work.
                    "conformance_ever": dict(_conf_ever),
                    "conformant_over_run": _conformant_over_run(),
                    "cumulative": {
                        "active_wall_s": round(active_wall_s, 3),
                        "rate_limit_wait_s": round(rate_limit_wait_s, 3),
                        "rl_waits_used": rl_waits_used,
                        "started_at": started_at,
                    },
                    "last_updated": datetime.now(_dt.UTC).isoformat(),
                },
                sort_keys=False,
            )
        )

    def _sink_telemetry_now(where: str) -> None:
        """Sink this run's telemetry into the shared aet store MID-RUN, on the grader's cadence.

        The end-of-run sink further down is the authoritative one, but it is also the ONLY one, and a
        continuous run is a single round: when a run is killed, times out (rc=124) or hits its wall
        budget, that call never happens and the run records nothing at all -- no ``logs/metrics.jsonl``,
        no ``metrics/trajectory.json``, invisible to ``aet spend``/``aet plot`` -- while its transcript
        sat on disk the whole time. MEASURED on the gemmini run of 2026-09-07, which graded 93/97 across
        two rc=124 rounds and left no aet record.

        Re-emitting is safe and is why this can run on an interval: ``logs/metrics.jsonl`` is
        append-only, and every consumer reads it as "values are logged as final scalars; the LAST
        occurrence of each name wins" (``aet.trajectory.rollup._read_metrics``). So each tick supersedes
        the previous one rather than summing with it, and a killed run keeps the most recent snapshot.

        ``transcript_paths`` is deliberately left unset: the combined ``transcript.jsonl`` does not
        exist until the run finalizes, and ``aet_bridge._resolve_transcripts`` falls back to the
        per-round transcripts under ``rounds/``, which are exactly what exists while the agent works.

        Soft by construction -- telemetry must never gate a run. That softness is also why this is
        covered by a test (``merlin/tests/infra/test_aet_sink_is_periodic.py``): a sink that silently
        stops looks identical to a run that had nothing to report.
        """
        try:
            from merlin.targetgen import aet_bridge as AB

            if not AB.aet_sink_enabled(run_dir):
                return
            AB.emit_to_aet(
                run_dir=run_dir,
                run_id=a.run_id,
                method=arm,
                model=a.model,
                target=_te().target,
                suite="capsule-bench",
                billing_mode=EX.billing_mode(a.model, config=provider),
            )
        except Exception as e:  # noqa: BLE001 - telemetry may never kill a run
            print(f"[aet-sink] {where}: skipped ({type(e).__name__}: {e})", flush=True)

    if a.continuous:
        # Legacy single session: background feedback, but no formal completion claim.
        import threading

        stop = threading.Event()
        state = {"tick": 0, "verdict": verdict}

        def _grader() -> None:
            # FIRST GRADE FAST: the interval below is 900s by default and the full ladder can take far
            # longer than that on one capsule, so the agent would otherwise open with no verdict. Land
            # the cheap loop-tier one first; it names the tiers it did not run and never claims them.
            if not (ws / "qa" / "verdict.json").exists():
                try:
                    _fv = _fast_grade(ws, run_dir, FL.FIRST_BACKGROUND_TICK, min(a.qa_timeout, 900))
                    print(
                        f"[continuous] first (loop-tier) verdict: {_fv.get('n_passed')}/"
                        f"{_fv.get('n_capsules')} all_pass={_fv.get('all_pass')} "
                        f"NOT run: {_fv.get('tiers_not_run')}",
                        flush=True,
                    )
                except Exception as _fe:  # noqa: BLE001 — no submission yet: the interval grade covers it
                    print(f"[continuous] first (loop-tier) grade skipped: {type(_fe).__name__}: {_fe}", flush=True)
            while not stop.wait(max(30, int(a.grade_interval))):
                t = state["tick"] + 1
                try:
                    v = _grade(ws, run_dir, t, a.no_oracle, a.qa_timeout)
                except Exception as e:  # noqa: BLE001 — a mid-write submission must never kill the run
                    print(f"[continuous] grade {t} skipped: {type(e).__name__}: {e}", flush=True)
                    continue
                state["tick"], state["verdict"] = t, v
                print(
                    f"[continuous] grade {t}: {v.get('n_passed')}/{v.get('n_capsules')} all_pass={v.get('all_pass')}",
                    flush=True,
                )
                _checkpoint(t)
                _sink_telemetry_now(f"continuous grade {t}")
                if v.get("all_pass"):
                    stop.set()  # converged: stop grading; the session is torn down below
                    return

        _verify_implementation_sources()
        gt = threading.Thread(target=_grader, name="continuous-grader", daemon=True)
        gt.start()
        try:
            if _operator_errata_record is not None:
                RI.verify_operator_errata(_operator_errata_record, run_dir)
            from merlin_experiments.phase1.feedback import brief as _RBlaunch

            _RBlaunch.refresh_before_launch(run_dir, ws, 0)
            rc, tpath = _launch(ws, run_dir, a.model, a.effort, a.sandbox, bundle, 0, a.round_timeout, arm=arm)
        except subprocess.TimeoutExpired:
            rc, tpath = 124, run_dir / "rounds" / "round_00.transcript.jsonl"
            print("[continuous] agent session TIMEOUT (the session bound, not a round)")
        stop.set()
        # A timed join is not cancellation: if the grader is already in qa_grade it remains live and
        # races the authoritative grade below.  Preserve the same single-flight invariant as the
        # certified continuous path.
        gt.join()
        # FINAL AUTHORITATIVE GRADE: the background grades are progress reports on a moving workspace;
        # the run's verdict is a grade of the submission as the session left it.
        verdict = _grade(ws, run_dir, state["tick"] + 1, a.no_oracle, a.qa_timeout)
        conf, workflow_conformant = _workflow_conformance(
            tpath, ws / "submission", arm, _endpoint_kind, resolved_tools()
        )
        rounds_summary.append(
            {
                "round": 0,
                "mode": "continuous",
                "agent_rc": rc,
                "grades": state["tick"] + 1,
                "all_pass": verdict.get("all_pass", False),
                "workflow_conformant": workflow_conformant,
                "authoring_complete": _authoring_complete(),
                "conformance": conf,
                "n_passed": verdict.get("n_passed"),
                "n_capsules": verdict.get("n_capsules"),
            }
        )
        _checkpoint(state["tick"] + 1)
        print(
            f"\ncontinuous run complete: {run_dir}  numeric_all_pass="
            f"{verdict.get('all_pass', False)} workflow_conformant={workflow_conformant} "
            f"authoring_complete={_authoring_complete()} formal_complete=False "
            f"grades={state['tick'] + 1}"
        )
        # This legacy single-session path does not run the common L3 barrier or the post-freeze hidden
        # grader.  It can report progress, but can never report a formal success.  Use
        # ``--schedule continuous`` for the continuous, fully certified path.
        return 1

    cost_capped = False  # set if the batch dollar ceiling (MERLIN_MAX_SPEND_USD) is reached

    def _keep_going() -> bool:
        """Should the loop run another agent invocation?

        ROUNDS: the historical condition, unchanged — bounded by --max-rounds.

        CONTINUOUS: the round COUNT is not a terminator. A run stops on EVIDENCE (converged, plateaued)
        or on a declared BUDGET (wall, spend), never because an arithmetic cap ran out while the
        submission was still improving. Measured on the v12 arm-4 run: it reached its ceiling in round 0
        and then spent two more rounds and 37.9M tokens going nowhere — the round budget was both too
        loose (it kept paying after convergence) and, on other runs, too tight (a productive round cut at
        the cap). Neither failure is about the submission, which is the only thing a stop should be about.
        A safety cap remains via --max-rounds only if the caller explicitly lowers it; the default 12 is
        ignored in continuous mode so it cannot silently reimpose the very bound this removes.
        """
        if a.seal_current:
            return False
        if _authoring_complete():
            return False
        if a.schedule == "rounds":
            return rnd < a.max_rounds
        if a.max_wall_s and active_wall_s >= a.max_wall_s:
            print(
                f"[continuous] wall budget reached ({active_wall_s:.0f}s >= {a.max_wall_s}s) — "
                f"stopping honestly (not converged; reason=max_wall_s)"
            )
            return False
        return True

    while _keep_going():
        _verify_implementation_sources()
        print(f"\n===== ROUND {rnd} =====")
        _rstart = time.time()
        # A verdict must land UNDER the turn, not after it. Round 0 has no previous round to inherit
        # feedback from, and under `--schedule continuous` one turn can be the whole run — measured, an
        # agent spent 6184s with no qa/ directory at all. The fast loop-tier grade lands one in minutes;
        # in continuous mode the full mandatory-ladder grade then repeats on --grade-interval.
        _bg = FL.start_background(
            ws,
            run_dir,
            FL.GradeCadence(a.qa_timeout, a.no_oracle, a.grade_interval),
            interval_grades=(a.schedule == "continuous"),
            round_index=rnd,
            on_tick=lambda t: _sink_telemetry_now(f"round {rnd} tick {t}"),
            grade_callback=_grade,
            fast_grade_callback=_fast_grade,
        )
        try:
            if _operator_errata_record is not None:
                RI.verify_operator_errata(_operator_errata_record, run_dir)
            from merlin_experiments.phase1.feedback import brief as _RBlaunch

            _RBlaunch.refresh_before_launch(run_dir, ws, rnd)
            rc, tpath = _launch(
                ws,
                run_dir,
                a.model,
                a.effort,
                a.sandbox,
                bundle,
                rnd,
                a.round_timeout,
                arm=arm,
                continuous=(a.schedule == "continuous"),
            )
        except subprocess.TimeoutExpired:
            rc, tpath = 124, run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
            print(f"[round {rnd}] agent TIMEOUT")
        finally:
            FL.stop_background(_bg)  # the post-turn grade below is the authoritative one
        # A dead retry can overwrite this round's earlier transcript. Rediscover
        # the full round history rather than retaining a now-overwritten path.
        _latest_authoring_tpath = ROQ.latest_live_authoring_transcript(
            [*(run_dir / "rounds").glob("round_*.transcript.jsonl"), tpath]
        )

        # Daily-quota wall: a provider DAILY token limit (429 'too many tokens per day') has no short
        # window reset to sleep to, so retrying burns every remaining round producing empty results
        # (the ccb2 waste). Abort the run early + honestly instead.
        if ROQ.daily_limit_hit(tpath):
            active_wall_s += time.time() - _rstart
            print(
                f"[round {rnd}] DAILY TOKEN LIMIT hit (429 'too many tokens per day') — the provider's "
                f"daily quota is exhausted; aborting early (not converged) rather than burning the "
                f"remaining rounds. Relaunch after the daily quota resets."
            )
            _checkpoint(rnd)
            break

        # Weekly quota: checkpoint this same round and exit distinctly, retaining partial work.
        if ROQ.weekly_quota_hit(tpath):
            active_wall_s += time.time() - _rstart
            _checkpoint(rnd)  # next_round stays rnd: --resume after the weekly reset continues THIS round
            sp = ROQ.write_quota_status(run_dir, ROQ.REASON_WEEKLY, rnd, transcript=tpath)
            print(
                f"[round {rnd}] {ROQ.STATUS_WEEKLY}: weekly (seven-day) subscription budget exhausted "
                f"~mid-round; partial submission checkpointed. Relaunch with --resume (same run_id) "
                f"after the weekly reset — see {sp}. Exiting (not a converged/failed 0).",
                flush=True,
            )
            return ROQ.QUOTA_WEEKLY_EXIT_CODE

        # Rate-limit backoff: if the org five-hour budget REJECTED this round (zero work), don't
        # consume it — sleep until the window resets and retry the SAME round index.
        if ROQ.round_rejected(tpath):
            active_wall_s += time.time() - _rstart  # the rejected attempt itself was active time
            if rl_waits_used >= a.max_rate_limit_waits:
                print(
                    f"[round {rnd}] rate-limited and --max-rate-limit-waits "
                    f"({a.max_rate_limit_waits}) exhausted — stopping honestly (not converged)"
                )
                _checkpoint(rnd)
                break
            reset_epoch = a.rl_test_reset_epoch or ROQ.rate_limit_reset_epoch(tpath) or 0
            sleep_s = max(0.0, reset_epoch - time.time()) + 20  # +jitter past the boundary
            rl_waits_used += 1
            _checkpoint(rnd)  # persist BEFORE the long sleep: a death during sleep resumes at this round
            print(
                f"[round {rnd}] RATE-LIMITED (five-hour) — wait #{rl_waits_used}; sleeping "
                f"{sleep_s:.0f}s until window reset, then retrying this round",
                flush=True,
            )
            time.sleep(sleep_s)
            rate_limit_wait_s += sleep_s
            _checkpoint(rnd)
            continue  # retry same rnd, do NOT append to rounds_summary

        # A turn that never ran cannot earn a grade or consume the remaining round budget.
        _dead, _why = ROQ.agent_turn_dead(tpath)
        if _dead:
            active_wall_s += time.time() - _rstart
            _checkpoint(rnd)  # next_round stays rnd: --resume retries THIS round once the cause is fixed
            print(
                f"[round {rnd}] AGENT DID NOT RUN — {_why}. Not grading, and not reporting a "
                f"conformance verdict: nothing about the submission was exercised this round. The "
                f"submission is checkpointed; fix the driver/model/credits and relaunch with --resume "
                f"(same run_id) to retry this round.",
                flush=True,
            )
            break

        verdict = _grade(ws, run_dir, rnd, a.no_oracle, a.qa_timeout)
        # Cross-round MEMORY: write the harness-built round brief (progress log across all graded rounds +
        # the agent's own notes + a stale-notes nudge) so the NEXT fresh session carries its progress
        # instead of re-deriving it. Best-effort — never let a brief build failure end a run.
        try:
            from merlin_experiments.phase1.feedback import brief

            brief.write(run_dir, ws, rnd)
        except Exception as _e:  # noqa: BLE001
            print(f"[round {rnd}] round_brief skipped: {type(_e).__name__}: {_e}")
        # If this round was CUT SHORT mid-work but is resumable in-budget (a wall-clock timeout, rc=124),
        # its partial submission/ persists in the workspace — do NOT treat it as a converged/failed final.
        # Prepend a RESUME banner to the next round's brief so the fresh session CONTINUES the partial
        # (finish manifest.yaml + the CLI + the target artifact first) instead of restarting from scratch.
        _cut, _cut_reason = ROQ.round_was_cut_short(run_dir, rc=rc, transcript=tpath)
        if _cut and ROQ.resume_policy(_cut_reason) == ROQ.RESUME_IN_BUDGET:
            try:
                ROQ.prepend_resume_note(ws, _cut_reason)
                print(
                    f"[round {rnd}] cut short ({_cut_reason}); partial submission preserved — next round "
                    f"will RESUME it (finish manifest.yaml + CLI + target artifact first)."
                )
            except Exception as _e:  # noqa: BLE001
                print(f"[round {rnd}] resume-note skipped: {type(_e).__name__}: {_e}")
        rsum = ET.parse_transcript(
            tpath,
            billing_mode=EX.billing_mode(a.model, config=provider),
            trust_cli_cost=EX.trust_cli_cost(a.model, config=provider),
        )
        audit_result = audit.audit_transcript(tpath, arm, workspace=ws)
        # Dev-conformance GATE: numeric progress is still reported, but a nonconformant workflow cannot
        # advance to the official claim-bearing grade.
        conf, workflow_conformant = _workflow_conformance(
            tpath, ws / "submission", arm, _endpoint_kind, resolved_tools()
        )
        # ...and accumulated ACROSS the run: a check satisfied in SOME round stays satisfied. Reading it
        # per round gates on the last round alone, which blocked runs whose agent had already
        # demonstrated every mandated behaviour -- just never all of it within one round.
        for _k, _v in (conf.get("checks") or {}).items():
            if _v is None:
                _conf_ever.setdefault(_k, None)  # inapplicable to this arm; never gates
            else:
                _conf_ever[_k] = bool(_conf_ever.get(_k)) or bool(_v)
        _bad = [k for k, v in (conf.get("checks") or {}).items() if v is False]
        if _bad:
            print(
                f"[round {rnd}] NOT CONFORMANT — failing: {', '.join(_bad)} "
                "(numeric grade still reports; formal completion remains blocked)"
            )
        rounds_summary.append(
            {
                "round": rnd,
                "agent_rc": rc,
                "conformance": conf,
                "all_pass": verdict.get("all_pass"),
                "workflow_conformant": workflow_conformant,
                "authoring_complete": _authoring_complete(),
                "n_passed": verdict.get("n_passed"),
                "n_capsules": verdict.get("n_capsules"),
                "tool_calls": rsum.get("tool_calls"),
                # per-round effort split (was only recorded whole-run before) — lets us
                # plot tokens/cost/thinking PER round, not just totals.
                "tokens_total": rsum.get("tokens_total"),
                "tokens_output": rsum.get("tokens_output"),
                "tokens_cached": rsum.get("tokens_cached"),
                "tokens_input": rsum.get("tokens_input"),
                "tokens_fresh_input": rsum.get("tokens_fresh_input"),
                "tokens_cache_write": rsum.get("tokens_cache_write"),
                "usage_complete": rsum.get("usage_complete"),
                "thinking_blocks": rsum.get("thinking_blocks"),
                "tokens_reasoning": rsum.get("tokens_reasoning"),
                "estimated_cost_usd": rsum.get("estimated_cost_usd"),
                "billing_mode": rsum.get("billing_mode"),
                "subscription_notional_usd": rsum.get("subscription_notional_usd"),
                "answer_access_clean": audit_result["clean"],
                "audit_hits": audit_result["hits"],
            }
        )
        active_wall_s += time.time() - _rstart
        print(
            f"[round {rnd}] all_pass={verdict.get('all_pass')} "
            f"{verdict.get('n_passed')}/{verdict.get('n_capsules')} "
            f"answer_access_clean={audit_result['clean']}"
        )
        # Enforce the batch DOLLAR ceiling: once the shared spend ledger crosses MERLIN_MAX_SPEND_USD, stop
        # before starting the next (paid) round. Uses the authoritative subagent-inclusive per-round cost.
        _over, _spent, _cap = _spend_over_cap(rsum.get("estimated_cost_usd"))
        if _over:
            print(
                f"[round {rnd}] COST CAP: batch spend ${_spent:.2f} >= ${_cap:.2f} "
                f"(MERLIN_MAX_SPEND_USD) — stopping before the next round; zero further spend."
            )
            cost_capped = True
            rnd = rnd + 1
            _checkpoint(rnd)
            break
        rnd = rnd + 1
        _checkpoint(rnd)  # next_round advances only after a completed (non-rate-limited) round
        # Plateau early-stop: a stuck agent re-sends its (uncached) growing context every round for no
        # gain — bound that spend. Stop after N consecutive rounds with no progress (neither the pass
        # count nor the total numeric mismatch improved). Disabled with --plateau-rounds 0.
        if a.plateau_rounds and not verdict.get("all_pass"):
            _prog = _progress_key(verdict)
            if _best_progress is None or _prog > _best_progress:
                _best_progress, _plateau_stall = _prog, 0
            else:
                _plateau_stall += 1
                if _plateau_stall >= a.plateau_rounds:
                    print(
                        f"[plateau] no progress (pass count + mismatch) for {a.plateau_rounds} "
                        f"consecutive rounds; stopping early (not converged) at "
                        f"{verdict.get('n_passed')}/{verdict.get('n_capsules')} to avoid burning tokens "
                        f"on a stuck loop. Raise/relax with --plateau-rounds."
                    )
                    break
        # realistic (abc2): the agent self-paces — it self-checks via the tool and drops READY_FOR_BARRIER
        # when it believes it's done. Break to the verilator barrier on that marker OR on spike all-pass.
        ready = a.experiment == "realistic" and (ws / "submission" / READY_MARKER).exists()
        # A marker dropped at zero is a surrender, not a convergence, and it silently forfeits the rest
        # of the round budget. --min-rounds declines it: remove the marker so the next round must earn
        # it again, and hand the agent back the same failing verdict. An honest all_pass is never
        # touched -- only the self-declaration is, and only while the run is still failing.
        if ready and not verdict.get("all_pass") and rnd - 1 < a.min_rounds:
            (ws / "submission" / READY_MARKER).unlink(missing_ok=True)
            print(
                f"[round {rnd - 1}] agent dropped {READY_MARKER} at "
                f"{verdict.get('n_passed')}/{verdict.get('n_capsules')} — DECLINED "
                f"(--min-rounds {a.min_rounds}); marker cleared, continuing",
                flush=True,
            )
            ready = False
        if ready and not _conformant_over_run():
            # RUN-LEVEL, not this round: `cca_used` and friends ask about start-of-work activities, and a
            # converged agent has nothing left to enumerate, so demanding them in the same round that
            # proves completion is unsatisfiable. Measured: a run reached 27/27 in round 3 and repeated
            # it through round 7 without ever exiting, because tool calls fell 300 -> 37 -> 30 -> 20.
            (ws / "submission" / READY_MARKER).unlink(missing_ok=True)
            print(
                f"[round {rnd - 1}] agent dropped {READY_MARKER}, but mandatory tooling is not "
                "successfully evidenced in ANY round — marker cleared, continuing",
                flush=True,
            )
            ready = False
        if _authoring_complete() or ready:
            if ready:
                print(f"[round {rnd - 1}] agent dropped {READY_MARKER} — proceeding to verilator barrier")
            break

    # --- cycle-accurate cert checkpoint ------------------------------------------------------------
    # The loop gate is the fast functional tier; this is where kernels are validated on the real RTL.
    def _verilator_grade(attempt: int, eligible: list, not_promoted: list) -> dict:
        _verify_implementation_sources()
        from merlin.targetgen import capsule_grade as _CG
        from merlin.targetgen import capsule_runner as _CR
        from merlin_experiments.phase1.feedback import qa as _qc

        _cert_tier = CERT._cert_tier_name(inputs=grading_inputs)
        vcand = run_dir / "_qa_work" / f"vcand_{attempt}" / "submission"
        if vcand.parent.exists():
            shutil.rmtree(vcand.parent)
        if not (ws / "submission" / "manifest.yaml").exists():
            # Nothing to grade: every capsule is unmeasured, and unmeasured is never a pass. The
            # denominator is still the whole corpus rather than a baked constant.
            _rows, _np, _nc = CERT._l3_attempt_tally(
                {},
                list(not_promoted)
                + [{"capsule": n, "reason": "no submission/manifest.yaml to grade"} for n in eligible],
                _cert_tier,
            )
            return {
                "attempt": attempt,
                "all_pass": False,
                "n_passed": _np,
                "n_capsules": _nc,
                "n_not_promoted": _nc,
                "cert_tier": _cert_tier,
                "per_capsule": _rows,
                "rtl_checks": None,
            }
        shutil.copytree(ws / "submission", vcand, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
        RI.strip_build_state(vcand)  # clean, relocatable build for the L3 cert (abc9 L3-0/20 'build' bug)
        vruns = run_dir / "_qa_work" / f"vruns_{attempt}"
        # Cycle-accurate checkpoint = the target's FULL oracle ladder, resolved from the descriptor's
        # target+sim_via via the shared factory (gemmini/chipyard -> spike L2 + verilator L3; an arc/mlc
        # target -> its RTL-derived arc tier), so a new target's L3 cert needs no edit here.
        _te_ck = _te()
        adapters = _CR.qa_checkpoint_adapters(_te_ck.target, _te_ck.sim_via)
        # CIRCT arm only: wrap each sim adapter with the CIRCT structural screen. It is ADVISORY — it
        # records the screen verdict + wall to circt_gate_log.jsonl but NEVER skips the sim or fails the
        # tier (a structural reject can be a false-positive on a conformant-but-different kernel, and
        # skipping would both mis-fail it and bias this arm vs the ungated plain/baseline arms). Both arms
        # now run identical sims; the wrap only adds telemetry for the rtlchecks arm.
        if bundle_id.find("rtlchecks") >= 0:
            try:
                from merlin.targetgen import circt_gate as _GATE

                _glog: list = []
                adapters = {t: _GATE.gated_adapter(adp, log=_glog, target=_te_ck.target) for t, adp in adapters.items()}
                (run_dir / "circt_gate_log.jsonl").write_text("")  # reset; appended after grade
                print(f"[verilator attempt {attempt}] CIRCT screen ADVISORY (records verdict; sim always runs)")
            except Exception as e:
                print(f"[verilator attempt {attempt}] CIRCT gate unavailable ({e}); running ungated")
                _glog = None
        else:
            _glog = None
        # SUBMIT ONLY THE PROMOTED CAPSULES. `capsules_root` accepts several roots and discovers a
        # `capsule.yaml` under each, so the eligible capsule DIRECTORIES are the roots — the held-back
        # capsules are simply not among them, and no minutes of cycle-accurate sim are spent re-learning
        # a loop-tier failure the loop already reported.
        _dirs = CERT._pilot_capsule_dirs(inputs=grading_inputs)
        _roots = [str(_dirs[n]) for n in eligible if n in _dirs]
        _unresolved = [n for n in eligible if n not in _dirs]
        if _unresolved:
            # A name that resolves to no capsule directory cannot be graded, and an ungraded capsule is
            # not a pass — move it to the held-back side carrying that reason.
            not_promoted = list(not_promoted) + [
                {"capsule": n, "reason": "capsule directory not found in the pilot corpus"} for n in _unresolved
            ]
            print(
                f"[verilator attempt {attempt}] {len(_unresolved)} promoted capsule(s) have no "
                f"directory in the pilot corpus — recorded not_promoted, NOT passed"
            )
        try:
            _CG.grade(
                str(vcand),
                capsules_root=_roots,
                runs_root=str(vruns),
                labels={"public", "dev"},
                contract=str(_contract_root if _contract_root is not None else context.repo / "merlin/contract"),
                oracle_adapters=adapters,
                timeout=CERT._verilator_l3_budget(run_dir, eligible, _cert_tier, context=context),
                max_workers=_CG.default_grade_workers(),
                target=_te().target,
                additional_forbidden=grading_inputs.additional_forbidden,
            )
        except Exception as e:
            print(f"[verilator attempt {attempt}] grade error: {str(e)[:200]}")
        if _glog is not None:  # record CIRCT gate decisions (skips + per-call wall)
            skipped = sum(1 for r in _glog if r.get("sim_skipped"))
            with open(run_dir / "circt_gate_log.jsonl", "a") as gf:
                for r in _glog:
                    gf.write(json.dumps({"attempt": attempt, **r}) + "\n")
            print(f"[verilator attempt {attempt}] CIRCT gate: {skipped}/{len(_glog)} sims skipped (reject)")
        # Whole-corpus tally: graded rows plus the held-back ones. `all_pass` below is therefore
        # `n_passed == n_capsules` over the FULL corpus, so promoting a subset can never make a partial
        # run look certified.
        per, npass, nc = CERT._l3_attempt_tally(_qc._per_capsule_from_results(vruns), not_promoted, _cert_tier)
        # The Arm4 qa wrapper can derive the same answer-free RTL structural block from this attempt's
        # generated traces.  Carry it into the L3 fix verdict so that a fix round can obey (and prove) its
        # mandatory RTL-feedback readback instead of receiving a reduced verdict with that surface absent.
        _rtl_block = treatment.checkpoint_feedback
        try:
            rtl_checks = (
                _rtl_block(vruns, capsule_roots=tuple(Path(root) for root in _roots)) if callable(_rtl_block) else None
            )
        except Exception as _e:  # noqa: BLE001 — advisory data stays non-numeric, but absence is explicit
            rtl_checks = {"error": f"{type(_e).__name__}: {_e}"}
        return {
            "attempt": attempt,
            "all_pass": npass == nc and nc > 0,
            "n_passed": npass,
            "n_capsules": nc,
            "n_not_promoted": sum(1 for r in per if r.get("l3_status") == "not_promoted"),
            "cert_tier": _cert_tier,
            "per_capsule": per,
            "rtl_checks": rtl_checks,
        }

    verilator_attempts: list = []
    _ready_marker = (ws / "submission" / READY_MARKER).exists()
    # The cycle-accurate RTL barrier is a pass-gate only when the target's corpus makes its RTL-cert tier
    # MANDATORY. A prototype target graded on its functional oracle (L3 optional) skips it, so a normal run
    # is not blocked on a slow/hanging verilator; convergence then rides the functional-tier (L2) verdict.
    _run_l3, _l3_reason = CERT._cycle_accurate_checkpoint_enabled(inputs=grading_inputs)
    if not _run_l3:
        print(f"[verilator] cycle-accurate RTL (L3) barrier SKIPPED — {_l3_reason}")
    # Promote each passing capsule even while other capsules still fail the loop tier.
    _loop_tier = CERT._loop_tier_name(grading_inputs)
    _l3_eligible, _l3_held = CERT._l3_promotion(verdict, _loop_tier)
    if _run_l3 and _conformant_over_run() and not _l3_eligible:
        print(
            f"[verilator] cycle-accurate cert SKIPPED — no capsule passed the loop tier "
            f"({_loop_tier or 'status'}), so there is nothing to certify; "
            f"{len(_l3_held)} capsule(s) recorded not_promoted (UNKNOWN at the cert tier, not passed)"
        )
    if a.seal_current:
        print(
            "[operator-seal] intermediate repair barrier skipped; the official grader still runs "
            "the declared public/hidden tiers",
            flush=True,
        )
    elif CERT._l3_checkpoint_should_run(_run_l3, _conformant_over_run(), _l3_eligible):
        # Per-capsule: every capsule that cleared the loop tier is certified now, independently of the
        # others. In realistic mode this checkpoint is still the definition of done; the agent already
        # self-checked on the tool, so this confirms on the operator side. Up to VERILATOR_ATTEMPTS with
        # a fix-round between.
        print(
            f"[verilator] promoting {len(_l3_eligible)}/{len(_l3_eligible) + len(_l3_held)} capsule(s) "
            f"that passed the loop tier ({_loop_tier or 'status'}) to the cycle-accurate cert; "
            f"{len(_l3_held)} held back (recorded not_promoted — never counted as a pass)"
        )
        # NON-TERMINAL L3 cert: re-grade L3, and on failure hand the redacted cycle-accurate verdict back
        # for a fix round, REPEATING until L3 passes OR the round budget (max_rounds) is exhausted. A
        # premature READY / a single L3 timeout is therefore survivable (it just costs a round) — never a
        # run-ending event (the abc7 failure mode). The only success exit is all-L3-pass.
        _why = "READY-marker" if _ready_marker else "loop-tier promotion"
        print(
            f"[verilator] cert — NON-terminal ({_why}; iterate to a full-corpus cert pass within "
            f"{a.max_rounds - rnd} remaining rounds)"
        )
        vatt = 0
        while True:
            vatt += 1
            _vs = time.time()
            vv = _verilator_grade(vatt, _l3_eligible, _l3_held)
            active_wall_s += time.time() - _vs
            verilator_attempts.append(
                {k: vv.get(k) for k in ("attempt", "all_pass", "n_passed", "n_capsules", "n_not_promoted")}
            )
            (run_dir / "verilator_checkpoints.json").write_text(
                json.dumps(
                    {
                        "attempts": verilator_attempts,
                        "final_all_pass": vv["all_pass"],
                        "last_per_capsule": vv["per_capsule"],
                    },
                    indent=2,
                )
            )
            print(
                f"[verilator attempt {vatt}] {vv['n_passed']}/{vv['n_capsules']} pass "
                f"{vv.get('cert_tier')} (all_pass={vv['all_pass']}; "
                f"{vv.get('n_not_promoted', 0)} not_promoted)"
            )
            # In CONTINUOUS mode the round count is not a terminator anywhere, including here: passing an
            # effectively unbounded cap keeps the decision to 'done' or 'iterate', so an L3 that has not
            # passed yet keeps being worked instead of stopping because an arithmetic budget expired.
            _cap = a.max_rounds if a.schedule == "rounds" else (rnd + 1_000_000)
            _budget_reason = "max_rounds"
            # Continuous repair ignores round count but respects active-work wall budget, excluding waits.
            if a.max_wall_s and active_wall_s >= float(a.max_wall_s):
                _cap, _budget_reason = rnd, "max_wall_s"
            _decision = CERT._l3_barrier_decision(vv["all_pass"], rnd, _cap)
            if _decision == "done":
                break  # ONLY success exit
            if _decision == "budget":
                print(
                    f"[verilator] round budget exhausted at L3 ({vv['n_passed']}/{vv['n_capsules']}) — "
                    f"stopping honestly (not converged; reason={_budget_reason}, NOT a barrier timeout)"
                )
                break
            # 'iterate' — NON-terminal: feed back redacted L3 failures, clear a false READY
            # (the agent must earn it again), and launch a fix round.
            (ws / "qa").mkdir(exist_ok=True)
            (ws / "qa" / "verdict.json").write_text(json.dumps(CERT._l3_fix_verdict(vv, vatt), indent=2))
            (ws / "submission" / READY_MARKER).unlink(missing_ok=True)
            _fr = time.time()
            _fix_rc = 0
            _fix_tpath = run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
            _verify_implementation_sources()
            try:
                if _operator_errata_record is not None:
                    RI.verify_operator_errata(_operator_errata_record, run_dir)
                from merlin_experiments.phase1.feedback import brief as _RBlaunch

                _RBlaunch.refresh_before_launch(run_dir, ws, rnd)
                _fix_rc, _fix_tpath = _launch(
                    ws, run_dir, a.model, a.effort, a.sandbox, bundle, rnd, a.round_timeout, arm=arm
                )
            except subprocess.TimeoutExpired:
                _fix_rc = 124
                print("[verilator fix-round] agent TIMEOUT")
            active_wall_s += time.time() - _fr
            # Refresh promotion after each fix. Unavailable feedback retains the previous split, never a pass.
            _rs = time.time()
            try:
                _refresh = _fast_grade(ws, run_dir, FL.FIRST_BACKGROUND_TICK + 100 + vatt, min(a.qa_timeout, 900))
                _l3_eligible, _l3_held = CERT._l3_promotion(_refresh, _loop_tier)
                print(
                    f"[verilator fix-round {rnd}] loop-tier refresh: {len(_l3_eligible)} eligible, "
                    f"{len(_l3_held)} still held back"
                )
            except Exception as _re:  # noqa: BLE001 — keep the previous split; never invent a promotion
                print(
                    f"[verilator fix-round {rnd}] loop-tier refresh unavailable "
                    f"({type(_re).__name__}: {_re}); keeping the previous promotion split"
                )
            active_wall_s += time.time() - _rs  # grading is active work, like every other grade here
            # A fix round may replace the compiler after the previously-conformant round.  Recompute from
            # THIS transcript + THIS submission immediately; retaining the old True would let prior tool
            # use mask a hand-authored/non-RTL final workflow.
            _latest_authoring_tpath = ROQ.latest_live_authoring_transcript(
                [*(run_dir / "rounds").glob("round_*.transcript.jsonl"), _fix_tpath]
            )
            conf, workflow_conformant = _workflow_conformance(
                _fix_tpath, ws / "submission", arm, _endpoint_kind, resolved_tools()
            )
            _fix_bad = [k for k, v in (conf.get("checks") or {}).items() if v is False]
            if _fix_bad:
                print(
                    f"[verilator fix-round {rnd}] NOT CONFORMANT — failing: "
                    f"{', '.join(_fix_bad)} (formal completion remains blocked)"
                )
            _fix_effort = ET.parse_transcript(
                _fix_tpath,
                billing_mode=EX.billing_mode(a.model, config=provider),
                trust_cli_cost=EX.trust_cli_cost(a.model, config=provider),
            )
            rounds_summary.append(
                {
                    "round": rnd,
                    "mode": "verilator_fix",
                    "agent_rc": _fix_rc,
                    "conformance": conf,
                    "workflow_conformant": workflow_conformant,
                    "authoring_complete": _authoring_complete(),
                    "l3_attempt_repaired": vatt,
                    "tokens_total": _fix_effort.get("tokens_total"),
                    "tokens_output": _fix_effort.get("tokens_output"),
                    "tokens_cached": _fix_effort.get("tokens_cached"),
                    "tokens_input": _fix_effort.get("tokens_input"),
                    "tokens_reasoning": _fix_effort.get("tokens_reasoning"),
                    "tool_calls": _fix_effort.get("tool_calls"),
                    "estimated_cost_usd": _fix_effort.get("estimated_cost_usd"),
                }
            )
            rnd += 1
            _checkpoint(rnd)
        # realistic (abc2): the verilator barrier IS the definition of done. Make it drive `converged`
        # (so the watchdog stops correctly + finalize runs) rather than the spike gate.
        if a.experiment == "realistic":
            verdict["all_pass"] = bool(vv["all_pass"])
            verdict["n_passed"], verdict["n_capsules"] = vv["n_passed"], vv["n_capsules"]
            _checkpoint(rnd)

    # The authoring tree may have changed in an L3 fix round (or this process may have resumed from an
    # older checkpoint).  Refresh once more before treating it as converged.  No transcript means no
    # persisted tool evidence and therefore fails closed.
    if _latest_authoring_tpath is not None:
        final_conformance, workflow_conformant = _workflow_conformance(
            _latest_authoring_tpath, ws / "submission", arm, _endpoint_kind, resolved_tools()
        )
    else:
        final_conformance = {"conformant": False, "error": "no authoring transcript"}
        workflow_conformant = False
    _checkpoint(rnd)

    # On convergence, give the agent ONE bounded turn to finalize its own REPORT.md/docs to the
    # VERIFIED result (the relaunch design means the converging round's agent never saw the passing
    # verdict, so its self-reported status lags by one round). Guarantees the frozen report matches.
    finalize = None
    if _authoring_complete():
        _verify_implementation_sources()
        print("[finalize] converged — running bounded report-finalize turn")
        _fin_start = time.time()
        finalize = finalize_report(
            ws,
            run_dir,
            a.model,
            a.effort,
            a.sandbox,
            bundle,
            arm,
            verdict,
            min(a.round_timeout, 900),
            provider=provider,
            context=context,
            audit=audit,
            grade_callback=_grade,
        )
        active_wall_s += time.time() - _fin_start  # finalize is active work
        print(
            f"[finalize] regrade_all_pass={finalize['regrade_all_pass']} "
            f"restored={finalize['restored_after_regression']} "
            f"stamped={finalize['status_line_stamped_by_driver']}"
        )
        # Finalize is instructed to edit docs only, but the submission is not trusted until checked.  Scan
        # the final bytes against the last authoring workflow so code changes/regressions cannot inherit a
        # pre-finalize conformance True.
        final_conformance, workflow_conformant = _workflow_conformance(
            _latest_authoring_tpath, ws / "submission", arm, _endpoint_kind, resolved_tools()
        )
        _checkpoint(rnd)
    # cumulative wall = active work (rounds + finalize, across ALL invocations) + rate-limit sleeps
    wall = round(active_wall_s + rate_limit_wait_s, 3)

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
        billing_mode=EX.billing_mode(a.model, config=provider),
        trust_cli_cost=EX.trust_cli_cost(a.model, config=provider),
    )
    # active-vs-waiting split (cumulative across resume-invocations): wall = active work + rate-limit
    # sleeps; active_wall_s is time actually DOING work (agent rounds + oracle grading + finalize).
    active_wall_s = round(active_wall_s, 3)
    timing = {
        "wall_seconds": wall,
        "active_wall_s": active_wall_s,
        "rate_limit_wait_s": round(rate_limit_wait_s, 3),
        "rate_limit_waits_used": rl_waits_used,
        "started_at": started_at,
        "resumed": _resuming,
    }
    ET.write_cost_yaml(summ, run_dir / "cost_time_toolcalls.yaml", wall_time_seconds=wall, model=a.model, exit_code=0)
    # append the active-vs-waiting split to the cost yaml (write_cost_yaml owns the rest)
    _cy = run_dir / "cost_time_toolcalls.yaml"
    _cd = yaml.safe_load(_cy.read_text()) or {}
    _cd.update(
        {
            "active_wall_s": active_wall_s,
            "rate_limit_wait_s": round(rate_limit_wait_s, 3),
            "rate_limit_waits_used": rl_waits_used,
        }
    )
    _cy.write_text(yaml.safe_dump(_cd, sort_keys=False))
    qa_summary = {
        "rounds": rounds_summary,
        "authoring_complete": _authoring_complete(),
        "numeric_all_pass": bool(verdict.get("all_pass", False)),
        "workflow_conformant": workflow_conformant,
        "final_conformance": final_conformance,
        "cost_capped": cost_capped,  # stopped by the batch dollar ceiling, not convergence/max_rounds
        "n_rounds": len(rounds_summary),
        "wall_seconds": wall,
        "finalize": finalize,
        "timing": timing,
        "run_config": _environment_record.get("run_config", _run_config),
        "operator_seal": operator_seal,
    }

    feedback_health = FL.channel_health(ws)
    qa_summary["feedback_health"] = feedback_health

    # Additionally sink this run's agentic telemetry into the shared aet store (opt-in,
    # MERLIN_AET_SINK=1) so it shows up in `aet spend` / `aet plot` across experiments. This is
    # purely additive — the existing experiment_tokens cost yaml above stays authoritative.
    from merlin.targetgen import aet_bridge as AB

    if AB.aet_sink_enabled(run_dir):
        AB.emit_to_aet(
            run_dir=run_dir,
            run_id=a.run_id,
            method=arm,
            model=a.model,
            target=_te().target,
            suite="capsule-bench",
            transcript_paths=[combined],
            billing_mode=EX.billing_mode(a.model, config=provider),
        )

    # capture the final submission and run the OFFICIAL public+hidden record
    wsub = ws / "submission"
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
    incomplete_operator_seal = bool(operator_seal)
    if wsub.exists() and ((workflow_conformant and feedback_health["healthy"]) or incomplete_operator_seal):
        # Last gate before the official process performs its public grade + freeze + hidden grade.  The
        # authoring loop can last across worktree edits and quota-window resumes, so setup-time hashing is
        # not enough: recompute every treatment byte and the private snapshot content now, fail closed on
        # drift, and hand the grader only the already-frozen hidden path.
        _current_scope = task_scope(_te(), a.sandbox, repo=context.repo, **_scope_roots)
        _official_hidden_dir = None
        if a.sandbox == "bwrap":
            _BWS.verify_bundle_snapshot(ws, bundle, repo=context.repo)
            _verified_snapshot_root = _BWS.bundle_snapshot_root(ws).resolve(strict=True)
            _expected_hidden_dir = RI.hidden_snapshot_dir(_verified_snapshot_root, _te(), context.repo)
        else:
            _expected_hidden_dir = None
        try:
            _official_hidden_dir = RI.verify_persisted_run_inputs(
                _environment_record,
                identity={
                    "run_id": a.run_id,
                    "arm": arm,
                    "sandbox": a.sandbox,
                    "bundle_id": bundle["bundle_id"],
                    "condition": bundle.get("condition", "legacy"),
                },
                task_scope=_current_scope,
                ws=ws,
                run_dir=run_dir,
                bundle_dir=bundle_dir,
                resolved_tools=resolved_tools(),
                expected_hidden_dir=_expected_hidden_dir,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"official grade refused: {exc}") from exc
        _verify_implementation_sources()
        grade_cmd = [
            sys.executable,
            "-m",
            "merlin_experiments.phase1.feedback.formal",
            "--descriptor",
            str(context.descriptor),
            "--repo",
            str(context.repo),
            "--contract",
            str(_contract_root if _contract_root is not None else context.repo / "merlin/contract"),
            "--run-dir",
            str(run_dir),
            "--arm",
            arm,
            "--model",
            a.model,
            "--capsules",
            str(CERT._public_capsules(grading_inputs)),
        ]
        # bwrap formal runs consume the immutable operator-only copy, never the live worktree/symlink.
        # The unsandboxed mode is an explicit untrusted diagnostic and retains its historical live path.
        if _official_hidden_dir is None:
            _cc = Path(_te().capsule_corpus)
            _official_hidden_dir = (_cc if _cc.is_absolute() else (context.repo / _cc)).parent / "hidden"
        if _official_hidden_dir.is_dir():
            grade_cmd += ["--hidden-capsules", str(_official_hidden_dir)]
        if a.no_oracle:
            grade_cmd.append("--no-oracle")
        if a.skip_hidden:
            grade_cmd.append("--skip-hidden")
        from merlin_experiments.frozen_python import inherited_python_command

        grade_proc = subprocess.run(inherited_python_command(grade_cmd), cwd=str(context.repo))
        _verify_implementation_sources()
        _manifest_path = run_dir / "run_manifest.yaml"
        if _manifest_path.is_file():
            _manifest_doc = yaml.safe_load(_manifest_path.read_text(encoding="utf-8")) or {}
            _manifest_doc["run_config"] = _environment_record.get("run_config", _run_config)
            _manifest_doc["feedback_health"] = feedback_health
            _manifest_path.write_text(yaml.safe_dump(_manifest_doc, sort_keys=False))
        official_grade = CERT._official_grade_result(grade_proc.returncode, run_dir)
    elif wsub.exists() and not feedback_health["healthy"]:
        official_grade["failures"] = ["feedback_channel_unhealthy"]
    elif wsub.exists():
        official_grade["failures"] = ["workflow_nonconformant"]

    # Telemetry is part of the experiment result, not a best-effort epilogue. Seal mutable rollout /
    # broker evidence and derive the unified record BEFORE deciding formal completion. Any writer or
    # integrity failure remains a named incomplete record and prevents a "complete" claim.
    _telemetry_path = run_dir / "timing_detailed.json"
    try:
        _telemetry_record = _emit_run_timing(run_dir, rounds_summary)
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
    formal_complete = (not incomplete_operator_seal) and CERT._formal_completion(
        bool(verdict.get("all_pass")),
        workflow_conformant,
        official_grade["complete"] and feedback_health["healthy"],
        telemetry["complete"],
    )
    qa_summary.update(
        {
            "converged": formal_complete,
            "formal_complete": formal_complete,
            "official_grade": official_grade,
            "telemetry": telemetry,
        }
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
        if not telemetry["complete"]:
            _completion["formal_grade_complete"] = False
        _manifest_path.write_text(yaml.safe_dump(_manifest_doc, sort_keys=False))
    # The detailed usage record is written before the potentially long outer grade so the grader can
    # include it in its manifest.  Correct its launcher exit code once the authoritative grade exists.
    _cd = yaml.safe_load(_cy.read_text()) or {}
    _cd["exit_code"] = 0 if formal_complete else 1
    _cy.write_text(yaml.safe_dump(_cd, sort_keys=False))
    print(
        f"\nrun complete: {run_dir}  numeric_all_pass={verdict.get('all_pass')} "
        f"workflow_conformant={workflow_conformant} "
        f"official_grade_complete={official_grade['complete']} formal_complete={formal_complete} "
        f"rounds={len(rounds_summary)}"
    )
    return 0 if formal_complete else 1


def _emit_run_timing(run_dir: Path, rounds_summary: list) -> dict:
    """Read the canonical arrival-derived timing record; unknown durations remain null."""
    from merlin_experiments.phase1.telemetry import report as _TD

    out = _TD.write_run_timing(run_dir)
    return json.loads(out.read_text(encoding="utf-8"))
