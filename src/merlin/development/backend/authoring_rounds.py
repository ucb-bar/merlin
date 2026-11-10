"""Author turns, background feedback, quota recovery and declared stopping budgets."""

from __future__ import annotations

import subprocess
import time

from merlin.development.backend import recovery as ROQ
from merlin.development.backend import run_inputs as RI
from merlin.development.backend import spend as SPEND
from merlin.development.backend.authoring_config import READY_MARKER
from merlin.development.backend.authoring_conformance import (
    workflow_conformance as _workflow_conformance,
)
from merlin.development.backend.authoring_session import AuthoringSession
from merlin.development.backend.feedback import grade_reuse as GR
from merlin.development.backend.feedback import lifecycle as FL
from merlin.development.backend.providers import execution as EX
from merlin.targetgen import experiment_tokens as ET


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


def run_legacy_continuous(session: AuthoringSession) -> int:
    progress = session.progress
    options = session.options
    workspace = session.workspace
    run_dir = session.run_dir
    context = session.context
    arm = session.arm
    bundle = session.bundle
    # Legacy single session: background feedback, but no formal completion claim.
    import threading

    stop = threading.Event()
    state = {"tick": 0, "verdict": progress.verdict}

    def _grader() -> None:
        # FIRST GRADE FAST: the interval below is 900s by default and the full ladder can take far
        # longer than that on one capsule, so the agent would otherwise open with no verdict. Land
        # the cheap loop-tier one first; it names the tiers it did not run and never claims them.
        if not (workspace / "qa" / "verdict.json").exists():
            try:
                _fv = session.fast_grade(workspace, run_dir, FL.FIRST_BACKGROUND_TICK, min(options.qa_timeout, 900))
                print(
                    f"[continuous] first (loop-tier) verdict: {_fv.get('n_passed')}/"
                    f"{_fv.get('n_capsules')} all_pass={_fv.get('all_pass')} "
                    f"NOT run: {_fv.get('tiers_not_run')}",
                    flush=True,
                )
            except Exception as _fe:  # noqa: BLE001 — no submission yet: the interval grade covers it
                print(f"[continuous] first (loop-tier) grade skipped: {type(_fe).__name__}: {_fe}", flush=True)
        while not stop.wait(max(30, int(options.grade_interval))):
            t = state["tick"] + 1
            try:
                v = session.grade(workspace, run_dir, t, options.no_oracle, options.qa_timeout)
            except Exception as e:  # noqa: BLE001 — a mid-write submission must never kill the run
                print(f"[continuous] grade {t} skipped: {type(e).__name__}: {e}", flush=True)
                continue
            state["tick"], state["verdict"] = t, v
            print(
                f"[continuous] grade {t}: {v.get('n_passed')}/{v.get('n_capsules')} all_pass={v.get('all_pass')}",
                flush=True,
            )
            try:
                session.checkpoint(t)
            except Exception as e:  # noqa: BLE001 — e.g. a full disk; the next grade checkpoints again
                print(f"[continuous] checkpoint {t} skipped: {type(e).__name__}: {e}", flush=True)
            session.sink_telemetry(f"continuous grade {t}")
            if v.get("all_pass"):
                stop.set()  # converged: stop grading; the session is torn down below
                return

    session.verify_inputs()
    gt = FL.library_thread(_grader, name="continuous-grader", daemon=True)
    gt.start()
    try:
        if session.operator_errata is not None:
            RI.verify_operator_errata(session.operator_errata, run_dir)
        from merlin.development.backend.feedback import brief as _RBlaunch

        _RBlaunch.refresh_before_launch(run_dir, workspace, 0)
        rc, tpath = session.launch(
            workspace,
            run_dir,
            options.model,
            options.effort,
            options.sandbox,
            bundle,
            0,
            options.round_timeout,
            arm=arm,
        )
    except subprocess.TimeoutExpired:
        rc, tpath = 124, run_dir / "rounds" / "round_00.transcript.jsonl"
        print("[continuous] agent session TIMEOUT (the session bound, not a round)")
    stop.set()
    # A timed join is not cancellation: a grader already in qa_grade stays live and races the
    # authoritative grade below. Preserve the certified continuous path's single-flight invariant.
    FL.join_background(gt)
    # FINAL AUTHORITATIVE GRADE: the background grades are progress reports on a moving workspace;
    # the run's verdict is a grade of the submission as the session left it.
    progress.verdict = session.grade(workspace, run_dir, state["tick"] + 1, options.no_oracle, options.qa_timeout)
    if context.target == "gemmini" and options.experiment == "full":  # target-ok: Gemmini scope
        from merlin.development.backend import oot_history as _OH

        _OH.verify_turn_end(run_dir, workspace / "submission", progress.verdict.get("oot_commit"))
    conf, progress.workflow_conformant = _workflow_conformance(
        tpath, workspace / "submission", arm, session.endpoint_kind, session.resolved_tools()
    )
    progress.rounds.append(
        {
            "round": 0,
            "mode": "continuous",
            "agent_rc": rc,
            "grades": state["tick"] + 1,
            "all_pass": progress.verdict.get("all_pass", False),
            "workflow_conformant": progress.workflow_conformant,
            "authoring_complete": session.authoring_complete(),
            "conformance": conf,
            "n_passed": progress.verdict.get("n_passed"),
            "n_capsules": progress.verdict.get("n_capsules"),
            "oot_commit": progress.verdict.get("oot_commit"),
        }
    )
    session.checkpoint(state["tick"] + 1)
    print(
        f"\ncontinuous run complete: {run_dir}  numeric_all_pass="
        f"{progress.verdict.get('all_pass', False)} workflow_conformant={progress.workflow_conformant} "
        f"authoring_complete={session.authoring_complete()} formal_complete=False "
        f"grades={state['tick'] + 1}"
    )
    # This legacy single-session path does not run the common L3 barrier or the post-freeze hidden
    # grader.  It can report progress, but can never report a formal success.  Use
    # ``--schedule continuous`` for the continuous, fully certified path.
    return 1


def run_rounds(session: AuthoringSession) -> int | None:
    from merlin.development.backend import joint_control as JC

    progress = session.progress
    options = session.options
    workspace = session.workspace
    run_dir = session.run_dir
    context = session.context
    arm = session.arm
    bundle = session.bundle
    provider = session.provider
    study_snapshots = None

    def _keep_going() -> bool:
        """Should the loop run another agent invocation?

        Legacy rounds retain their caps; continuous runs stop on evidence or budget.
        Joint campaigns keep editing until the declared freeze/drain boundary.
        """
        if session.joint:
            return JC.stop_reason() is None
        if options.seal_current:
            return False
        if session.authoring_complete():
            return False
        if options.schedule == "rounds":
            return progress.round_index < options.max_rounds
        if options.max_wall_s and progress.active_wall_s >= options.max_wall_s:
            print(
                f"[continuous] wall budget reached ({progress.active_wall_s:.0f}s >= {options.max_wall_s}s) — "
                f"stopping honestly (not converged; reason=max_wall_s)"
            )
            return False
        return True

    while _keep_going():
        session.verify_inputs()
        print(f"\n===== ROUND {progress.round_index} =====")
        _rstart = time.time()
        # A verdict must land UNDER the turn, not after it. Round 0 has no previous round to inherit
        # feedback from, and under `--schedule continuous` one turn can be the whole run — measured, an
        # agent spent 6184s with no qa/ directory at all. The fast loop-tier grade lands one in minutes;
        # in continuous mode the full mandatory-ladder grade then repeats on --grade-interval.
        _bg = FL.start_background(
            workspace,
            run_dir,
            FL.GradeCadence(options.qa_timeout, options.no_oracle, options.grade_interval),
            interval_grades=(options.schedule == "continuous"),
            round_index=progress.round_index,
            on_tick=lambda t: session.sink_telemetry(f"round {progress.round_index} tick {t}"),
            grade_callback=session.background_grade,
            fast_grade_callback=session.fast_grade,
        )
        try:
            if session.operator_errata is not None:
                RI.verify_operator_errata(session.operator_errata, run_dir)
            from merlin.development.backend.feedback import brief as _RBlaunch

            _RBlaunch.refresh_before_launch(run_dir, workspace, progress.round_index)
            if session.joint:
                JC.start()
                if study_snapshots is None:
                    from merlin.development.backend.study_protocol import start_snapshots

                    study_snapshots = start_snapshots(
                        workspace,
                        run_dir,
                        context=context,
                        public_roots=session.scope_roots["public_roots"],
                        contract=session.scope_roots["contract"],
                        environment=session.environment,
                    )
            rc, tpath = session.launch(
                workspace,
                run_dir,
                options.model,
                options.effort,
                options.sandbox,
                bundle,
                progress.round_index,
                min(options.round_timeout, max(1, int(JC.remaining(drain=True))))
                if session.joint
                else options.round_timeout,
                arm=arm,
                continuous=(options.schedule == "continuous"),
            )
        except subprocess.TimeoutExpired:
            rc, tpath = 124, run_dir / "rounds" / f"round_{progress.round_index:02d}.transcript.jsonl"
            print(f"[round {progress.round_index}] agent TIMEOUT")
        finally:
            FL.stop_background(_bg)  # join the original owner before reusing or grading
        # A dead retry can overwrite this round's earlier transcript. Rediscover
        # the full round history rather than retaining a now-overwritten path.
        progress.latest_transcript = ROQ.latest_live_authoring_transcript(
            [*(run_dir / "rounds").glob("round_*.transcript.jsonl"), tpath]
        )

        # Daily-quota wall: a provider DAILY token limit (429 'too many tokens per day') has no short
        # window reset to sleep to, so retrying burns every remaining round producing empty results
        # (the ccb2 waste). Abort the run early + honestly instead.
        if ROQ.daily_limit_hit(tpath):
            progress.active_wall_s += time.time() - _rstart
            print(
                f"[round {progress.round_index}] DAILY TOKEN LIMIT hit (429 'too many tokens per day') — the provider's "
                f"daily quota is exhausted; aborting early (not converged) rather than burning the "
                f"remaining rounds. Relaunch after the daily quota resets."
            )
            session.checkpoint(progress.round_index)
            break

        # Weekly quota: checkpoint this same round and exit distinctly, retaining partial work.
        if ROQ.weekly_quota_hit(tpath):
            progress.active_wall_s += time.time() - _rstart
            session.checkpoint(
                progress.round_index
            )  # next_round stays rnd: --resume after the weekly reset continues THIS round
            sp = ROQ.write_quota_status(run_dir, ROQ.REASON_WEEKLY, progress.round_index, transcript=tpath)
            print(
                f"[round {progress.round_index}] {ROQ.STATUS_WEEKLY}: weekly (seven-day) subscription budget exhausted "
                f"~mid-round; partial submission checkpointed. Relaunch with --resume (same run_id) "
                f"after the weekly reset — see {sp}. Exiting (not a converged/failed 0).",
                flush=True,
            )
            return ROQ.QUOTA_WEEKLY_EXIT_CODE

        # Rate-limit backoff: if the org five-hour budget REJECTED this round (zero work), don't
        # consume it — sleep until the window resets and retry the SAME round index.
        if ROQ.round_rejected(tpath):
            progress.active_wall_s += time.time() - _rstart  # the rejected attempt itself was active time
            if progress.rate_limit_waits_used >= options.max_rate_limit_waits:
                print(
                    f"[round {progress.round_index}] rate-limited and --max-rate-limit-waits "
                    f"({options.max_rate_limit_waits}) exhausted — stopping honestly (not converged)"
                )
                session.checkpoint(progress.round_index)
                break
            reset_epoch = options.rl_test_reset_epoch or ROQ.rate_limit_reset_epoch(tpath) or 0
            sleep_s = max(0.0, reset_epoch - time.time()) + 20  # +jitter past the boundary
            if session.joint:
                sleep_s = min(sleep_s, JC.remaining(drain=True))
            progress.rate_limit_waits_used += 1
            session.checkpoint(
                progress.round_index
            )  # persist BEFORE the long sleep: a death during sleep resumes at this round
            print(
                f"[round {progress.round_index}] RATE-LIMITED (five-hour) — wait #{progress.rate_limit_waits_used}; sleeping "
                f"{sleep_s:.0f}s until window reset, then retrying this round",
                flush=True,
            )
            time.sleep(sleep_s)
            progress.rate_limit_wait_s += sleep_s
            session.checkpoint(progress.round_index)
            continue  # retry same rnd, do NOT append to rounds_summary

        # A turn that never ran cannot earn a grade or consume the remaining round budget.
        _dead, _why = ROQ.agent_turn_dead(tpath)
        if _dead:
            progress.active_wall_s += time.time() - _rstart
            session.checkpoint(
                progress.round_index
            )  # next_round stays rnd: --resume retries THIS round once the cause is fixed
            print(
                f"[round {progress.round_index}] AGENT DID NOT RUN — {_why}. Not grading, and not reporting a "
                f"conformance verdict: nothing about the submission was exercised this round. The "
                f"submission is checkpointed; fix the driver/model/credits and relaunch with --resume "
                f"(same run_id) to retry this round.",
                flush=True,
            )
            break

        from merlin.development.backend import joint_handoff

        if session.joint and joint_handoff.stop(run_dir)[0] is not None:  # epoch stop or the freeze: no new checks
            progress.verdict = joint_handoff.checkpoint(workspace, run_dir, progress.round_index)
        else:
            progress.verdict = GR.post_turn_grade(
                session.driver_grade_receipt[0],
                workspace,
                run_dir,
                progress.round_index,
                options.no_oracle,
                options.qa_timeout,
                inputs=session.grading_inputs,
                grade_callback=session.grade,
            )
        if context.target == "gemmini" and options.experiment == "full":  # target-ok: Gemmini scope
            from merlin.development.backend import oot_history as _OH

            _OH.verify_turn_end(run_dir, workspace / "submission", progress.verdict.get("oot_commit"))
        # Cross-round MEMORY: write the harness-built round brief (progress log across all graded rounds +
        # the agent's own notes + a stale-notes nudge) so the NEXT fresh session carries its progress
        # instead of re-deriving it. Best-effort — never let a brief build failure end a run.
        try:
            from merlin.development.backend.feedback import brief

            brief.write(run_dir, workspace, progress.round_index)
        except Exception as _e:  # noqa: BLE001
            print(f"[round {progress.round_index}] round_brief skipped: {type(_e).__name__}: {_e}")
        # If this round was CUT SHORT mid-work but is resumable in-budget (a wall-clock timeout, rc=124),
        # its partial submission/ persists in the workspace — do NOT treat it as a converged/failed final.
        # Prepend a RESUME banner to the next round's brief so the fresh session CONTINUES the partial
        # (finish manifest.yaml + the CLI + the target artifact first) instead of restarting from scratch.
        _cut, _cut_reason = ROQ.round_was_cut_short(run_dir, rc=rc, transcript=tpath)
        if _cut and ROQ.resume_policy(_cut_reason) == ROQ.RESUME_IN_BUDGET:
            try:
                ROQ.prepend_resume_note(workspace, _cut_reason)
                print(
                    f"[round {progress.round_index}] cut short ({_cut_reason}); partial submission preserved — next round "
                    f"will RESUME it (finish manifest.yaml + CLI + target artifact first)."
                )
            except Exception as _e:  # noqa: BLE001
                print(f"[round {progress.round_index}] resume-note skipped: {type(_e).__name__}: {_e}")
        rsum = ET.parse_transcript(
            tpath,
            billing_mode=EX.billing_mode(options.model, config=provider),
            trust_cli_cost=EX.trust_cli_cost(options.model, config=provider),
        )
        audit_result = session.audit.audit_transcript(tpath, arm, workspace=workspace)
        # Dev-conformance GATE: numeric progress is still reported, but a nonconformant workflow cannot
        # advance to the official claim-bearing grade.
        conf, progress.workflow_conformant = _workflow_conformance(
            tpath, workspace / "submission", arm, session.endpoint_kind, session.resolved_tools()
        )
        # ...and accumulated ACROSS the run: a check satisfied in SOME round stays satisfied. Reading it
        # per round gates on the last round alone, which blocked runs whose agent had already
        # demonstrated every mandated behaviour -- just never all of it within one round.
        for _k, _v in (conf.get("checks") or {}).items():
            if _v is None:
                progress.conformance_ever.setdefault(_k, None)  # inapplicable to this arm; never gates
            else:
                progress.conformance_ever[_k] = bool(progress.conformance_ever.get(_k)) or bool(_v)
        _bad = [k for k, v in (conf.get("checks") or {}).items() if v is False]
        if _bad:
            print(
                f"[round {progress.round_index}] NOT CONFORMANT — failing: {', '.join(_bad)} "
                "(numeric grade still reports; formal completion remains blocked)"
            )
        progress.rounds.append(
            {
                "round": progress.round_index,
                "agent_rc": rc,
                "conformance": conf,
                "all_pass": progress.verdict.get("all_pass"),
                "workflow_conformant": progress.workflow_conformant,
                "authoring_complete": session.authoring_complete(),
                "n_passed": progress.verdict.get("n_passed"),
                "n_capsules": progress.verdict.get("n_capsules"),
                "oot_commit": progress.verdict.get("oot_commit"),  # the graded package, by sha (oot_history)
                "tool_calls": rsum.get("tool_calls"),
                # per-round effort split: plot tokens/cost/thinking PER round, not just totals.
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
        progress.active_wall_s += time.time() - _rstart
        print(
            f"[round {progress.round_index}] all_pass={progress.verdict.get('all_pass')} "
            f"{progress.verdict.get('n_passed')}/{progress.verdict.get('n_capsules')} "
            f"answer_access_clean={audit_result['clean']}"
        )
        # Enforce the batch DOLLAR ceiling: once the shared spend ledger crosses MERLIN_MAX_SPEND_USD, stop
        # before starting the next (paid) round. Uses the authoritative subagent-inclusive per-round cost.
        _over, _spent, _cap = SPEND.spend_over_cap(rsum.get("estimated_cost_usd"))
        if _over:
            print(
                f"[round {progress.round_index}] COST CAP: batch spend ${_spent:.2f} >= ${_cap:.2f} "
                f"(MERLIN_MAX_SPEND_USD) — stopping before the next round; zero further spend."
            )
            progress.cost_capped = True
            progress.round_index = progress.round_index + 1
            session.checkpoint(progress.round_index)
            break
        progress.round_index = progress.round_index + 1
        session.checkpoint(progress.round_index)  # next_round advances only after a completed (non-rate-limited) round
        # Plateau early-stop: a stuck agent re-sends its (uncached) growing context every round for no
        # gain — bound that spend. Stop after N consecutive rounds with no progress (neither the pass
        # count nor the total numeric mismatch improved). Disabled with --plateau-rounds 0.
        if not session.joint and options.plateau_rounds and not progress.verdict.get("all_pass"):
            _prog = _progress_key(progress.verdict)
            if progress.best_progress is None or _prog > progress.best_progress:
                progress.best_progress, progress.plateau_stall = _prog, 0
            else:
                progress.plateau_stall += 1
                if progress.plateau_stall >= options.plateau_rounds:
                    print(
                        f"[plateau] no progress (pass count + mismatch) for {options.plateau_rounds} "
                        f"consecutive rounds; stopping early (not converged) at "
                        f"{progress.verdict.get('n_passed')}/{progress.verdict.get('n_capsules')} to avoid burning tokens "
                        f"on a stuck loop. Raise/relax with --plateau-rounds."
                    )
                    break
        # realistic (abc2): the agent self-paces — it self-checks via the tool and drops READY_FOR_BARRIER
        # when it believes it's done. Break to the verilator barrier on that marker OR on spike all-pass.
        ready = (
            not session.joint
            and options.experiment == "realistic"
            and (workspace / "submission" / READY_MARKER).exists()
        )
        # A marker dropped at zero is a surrender, not a convergence, and it silently forfeits the rest
        # of the round budget. --min-rounds declines it: remove the marker so the next round must earn
        # it again, and hand the agent back the same failing verdict. An honest all_pass is never
        # touched -- only the self-declaration is, and only while the run is still failing.
        if ready and not progress.verdict.get("all_pass") and progress.round_index - 1 < options.min_rounds:
            (workspace / "submission" / READY_MARKER).unlink(missing_ok=True)
            print(
                f"[round {progress.round_index - 1}] agent dropped {READY_MARKER} at "
                f"{progress.verdict.get('n_passed')}/{progress.verdict.get('n_capsules')} — DECLINED "
                f"(--min-rounds {options.min_rounds}); marker cleared, continuing",
                flush=True,
            )
            ready = False
        if ready and not session.conformant_over_run():
            # RUN-LEVEL, not this round: `cca_used` and friends ask about start-of-work activities, and a
            # converged agent has nothing left to enumerate, so demanding them in the same round that
            # proves completion is unsatisfiable. Measured: a run reached 27/27 in round 3 and repeated
            # it through round 7 without ever exiting, because tool calls fell 300 -> 37 -> 30 -> 20.
            (workspace / "submission" / READY_MARKER).unlink(missing_ok=True)
            print(
                f"[round {progress.round_index - 1}] agent dropped {READY_MARKER}, but mandatory tooling is not "
                "successfully evidenced in ANY round — marker cleared, continuing",
                flush=True,
            )
            ready = False
        if session.authoring_complete() or ready:
            if ready:
                print(
                    f"[round {progress.round_index - 1}] agent dropped {READY_MARKER} — proceeding to verilator barrier"
                )
            break

    if session.joint:
        if study_snapshots is not None:
            study_snapshots.close()
        from merlin.development.backend.joint_completion import complete
        from merlin.development.backend.joint_final_coverage import validate

        return complete(
            session.prepared,
            verdict=progress.verdict,
            rounds=progress.rounds,
            workflow_conformant=session.conformant_over_run(),
            final_coverage=lambda frozen, digest: validate(
                session.prepared, inputs=session.grading_inputs, frozen=frozen, compiler_sha256=digest
            ),
        )
