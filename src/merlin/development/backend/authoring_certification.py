"""Certify the promoted corpus and run bounded author repair turns."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from merlin.development.backend import recovery as ROQ
from merlin.development.backend import run_inputs as RI
from merlin.development.backend.authoring_config import READY_MARKER
from merlin.development.backend.authoring_conformance import (
    workflow_conformance as _workflow_conformance,
)
from merlin.development.backend.authoring_session import AuthoringSession
from merlin.development.backend.feedback import certification as CERT
from merlin.development.backend.feedback import lifecycle as FL
from merlin.development.backend.feedback import loop_grading as LG
from merlin.development.backend.providers import execution as EX
from merlin.development.targetgen import experiment_tokens as ET
from merlin.evaluation.capsule import availability as capsule_availability


def grade_checkpoint(
    session: AuthoringSession, attempt: int, eligible: list, not_promoted: list, *, driver_receipt=None
) -> dict:
    workspace = session.workspace
    run_dir = session.run_dir
    context = session.context
    session.verify_inputs()
    from merlin.development.backend.feedback import qa as _qc
    from merlin.development.targetgen import capsule_grade as _CG

    _cert_tier = CERT._cert_tier_name(inputs=session.grading_inputs)
    if driver_receipt is not None:
        from merlin.development.backend.feedback.grade_reuse import reuse_checkpoint

        reused = reuse_checkpoint(
            driver_receipt,
            submission=workspace / "submission",
            inputs=session.grading_inputs,
            run_dir=run_dir,
            eligible=eligible,
            held=not_promoted,
            cert_tier=_cert_tier,
            attempt=attempt,
        )
        if reused is not None:
            callback = session.treatment.checkpoint_feedback
            if callable(callback):
                try:
                    dirs = CERT._pilot_capsule_dirs(inputs=session.grading_inputs)
                    reused["rtl_checks"] = callback(
                        driver_receipt.runs, capsule_roots=tuple(Path(dirs[name]) for name in eligible)
                    )
                except Exception as exc:  # noqa: BLE001 -- structural feedback remains advisory
                    reused["rtl_checks"] = {"error": f"{type(exc).__name__}: {exc}"}
            return reused
    vcand = run_dir / "_qa_work" / f"vcand_{attempt}" / "submission"
    if vcand.parent.exists():
        shutil.rmtree(vcand.parent)
    if not (workspace / "submission" / "manifest.yaml").exists():
        # Nothing to grade: every capsule is unmeasured, and unmeasured is never a pass. The
        # denominator is still the whole corpus rather than a baked constant.
        _rows, _np, _nc = CERT._l3_attempt_tally(
            {},
            list(not_promoted) + [{"capsule": n, "reason": "no submission/manifest.yaml to grade"} for n in eligible],
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
    shutil.copytree(workspace / "submission", vcand, ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
    RI.strip_build_state(vcand)  # clean, relocatable build for the L3 cert (abc9 L3-0/20 'build' bug)
    vruns = run_dir / "_qa_work" / f"vruns_{attempt}"
    # Cycle-accurate checkpoint = the target's FULL oracle ladder, resolved from the descriptor's
    # target+sim_via via the shared factory (a chipyard target -> spike L2 + verilator L3; an arc/mlc
    # target -> its RTL-derived arc tier), so a new target's L3 cert needs no edit here.
    _te_ck = session.target_experiment()
    adapters = capsule_availability.qa_checkpoint_adapters(_te_ck.target, _te_ck.sim_via)
    # CIRCT arm only: wrap each sim adapter with the CIRCT structural screen. It is ADVISORY — it
    # records the screen verdict + wall to circt_gate_log.jsonl but NEVER skips the sim or fails the
    # tier (a structural reject can be a false-positive on a conformant-but-different kernel, and
    # skipping would both mis-fail it and bias this arm vs the ungated plain/baseline arms). Both arms
    # now run identical sims; the wrap only adds telemetry for the rtlchecks arm.
    if session.bundle_id.find("rtlchecks") >= 0:
        try:
            from merlin.development.targetgen import circt_gate as _GATE

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
    _dirs = CERT._pilot_capsule_dirs(inputs=session.grading_inputs)
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
    _checkpoint_score = None
    try:
        with LG.compiler_execution(session.grading_inputs, candidate=vcand, run_dir=run_dir):
            _checkpoint_score = _CG.grade(
                str(vcand),
                capsules_root=_roots,
                runs_root=str(vruns),
                model_snapshot_root=vruns.resolve() / ".private_model_sources",
                labels={"public", "dev"},
                contract=str(
                    session.contract_root
                    if session.contract_root is not None
                    else context.repo / "src/merlin/resources/contracts"
                ),
                oracle_adapters=adapters,
                timeout=CERT._verilator_l3_budget(
                    run_dir, eligible, _cert_tier, context=context, timing_file=session.runtime.oracle_timing
                ),
                max_workers=_CG.default_grade_workers(),
                target=session.target_experiment().target,
                additional_forbidden=session.grading_inputs.additional_forbidden,
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
    _checkpoint_rows = CERT._checkpoint_refusal_rows(_qc._per_capsule_from_results(vruns), _checkpoint_score)
    per, npass, nc = CERT._l3_attempt_tally(_checkpoint_rows, not_promoted, _cert_tier)
    # The Arm4 qa wrapper can derive the same answer-free RTL structural block from this attempt's
    # generated traces.  Carry it into the L3 fix verdict so that a fix round can obey (and prove) its
    # mandatory RTL-feedback readback instead of receiving a reduced verdict with that surface absent.
    _rtl_block = session.treatment.checkpoint_feedback
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
        "n_rtl_passed": sum(row.get("l3_status") == "pass" for row in per),
        "n_expected_refusal_passed": sum(row.get("validated_refusal") is True for row in per),
        "n_not_promoted": sum(1 for r in per if r.get("l3_status") == "not_promoted"),
        "cert_tier": _cert_tier,
        "per_capsule": per,
        "rtl_checks": rtl_checks,
    }


def run_certification(session: AuthoringSession) -> None:
    progress = session.progress
    options = session.options
    workspace = session.workspace
    run_dir = session.run_dir
    arm = session.arm
    bundle = session.bundle
    provider = session.provider
    verilator_attempts: list = []
    _ready_marker = (workspace / "submission" / READY_MARKER).exists()
    # The cycle-accurate RTL barrier is a pass-gate only when the target's corpus makes its RTL-cert tier
    # MANDATORY. A prototype target graded on its functional oracle (L3 optional) skips it, so a normal run
    # is not blocked on a slow/hanging verilator; convergence then rides the functional-tier (L2) verdict.
    _run_l3, _l3_reason = CERT._cycle_accurate_checkpoint_enabled(inputs=session.grading_inputs)
    if not _run_l3:
        print(f"[verilator] cycle-accurate RTL (L3) barrier SKIPPED — {_l3_reason}")
    # Promote each passing capsule even while other capsules still fail the loop tier.
    _loop_tier = CERT._loop_tier_name(session.grading_inputs)
    _l3_eligible, _l3_held = CERT._l3_promotion(progress.verdict, _loop_tier)
    if _run_l3 and session.conformant_over_run() and not _l3_eligible:
        print(
            f"[verilator] cycle-accurate cert SKIPPED — no capsule passed the loop tier "
            f"({_loop_tier or 'status'}), so there is nothing to certify; "
            f"{len(_l3_held)} capsule(s) recorded not_promoted (UNKNOWN at the cert tier, not passed)"
        )
    if options.seal_current:
        print(
            "[operator-seal] intermediate repair barrier skipped; the official grader still runs "
            "the declared public/hidden tiers",
            flush=True,
        )
    elif CERT._l3_checkpoint_should_run(_run_l3, session.conformant_over_run(), _l3_eligible):
        # Per-capsule: every capsule that cleared the loop tier is certified now, independently of the
        # others. In realistic mode this checkpoint is still the definition of done; the agent already
        # self-checked on the tool, so this confirms on the operator side. Repair turns retain the
        # declared round/wall budget below.
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
            f"{options.max_rounds - progress.round_index} remaining rounds)"
        )
        vatt = 0
        while True:
            vatt += 1
            _vs = time.time()
            vv = grade_checkpoint(session, vatt, _l3_eligible, _l3_held, driver_receipt=session.driver_grade_receipt[0])
            progress.active_wall_s += time.time() - _vs
            verilator_attempts.append(
                {
                    **{
                        k: vv.get(k)
                        for k in (
                            "attempt",
                            "all_pass",
                            "n_passed",
                            "n_capsules",
                            "n_not_promoted",
                            "n_rtl_passed",
                            "n_expected_refusal_passed",
                        )
                    },
                    **({"evidence_reuse": vv["evidence_reuse"]} if "evidence_reuse" in vv else {}),
                }
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
                f"[verilator attempt {vatt}] {vv['n_passed']}/{vv['n_capsules']} obligations passed; "
                f"RTL passes={vv.get('n_rtl_passed', 0)}, refusals={vv.get('n_expected_refusal_passed', 0)} "
                f"{vv.get('cert_tier')} (all_pass={vv['all_pass']}; "
                f"{vv.get('n_not_promoted', 0)} not_promoted)"
            )
            # In CONTINUOUS mode the round count is not a terminator anywhere, including here: passing an
            # effectively unbounded cap keeps the decision to 'done' or 'iterate', so an L3 that has not
            # passed yet keeps being worked instead of stopping because an arithmetic budget expired.
            _cap = options.max_rounds if options.schedule == "rounds" else (progress.round_index + 1_000_000)
            _budget_reason = "max_rounds"
            # Continuous repair ignores round count but respects active-work wall budget, excluding waits.
            if options.max_wall_s and progress.active_wall_s >= float(options.max_wall_s):
                _cap, _budget_reason = progress.round_index, "max_wall_s"
            _decision = CERT._l3_barrier_decision(vv["all_pass"], progress.round_index, _cap)
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
            (workspace / "qa").mkdir(exist_ok=True)
            (workspace / "qa" / "verdict.json").write_text(json.dumps(CERT._l3_fix_verdict(vv, vatt), indent=2))
            (workspace / "submission" / READY_MARKER).unlink(missing_ok=True)
            _fr = time.time()
            _fix_rc = 0
            _fix_tpath = run_dir / "rounds" / f"round_{progress.round_index:02d}.transcript.jsonl"
            session.verify_inputs()
            try:
                if session.operator_errata is not None:
                    RI.verify_operator_errata(session.operator_errata, run_dir)
                from merlin.development.backend.feedback import brief as _RBlaunch

                _RBlaunch.refresh_before_launch(run_dir, workspace, progress.round_index)
                _fix_rc, _fix_tpath = session.launch(
                    workspace,
                    run_dir,
                    options.model,
                    options.effort,
                    options.sandbox,
                    bundle,
                    progress.round_index,
                    options.round_timeout,
                    arm=arm,
                )
            except subprocess.TimeoutExpired:
                _fix_rc = 124
                print("[verilator fix-round] agent TIMEOUT")
            progress.active_wall_s += time.time() - _fr
            # Refresh promotion after each fix. Unavailable feedback retains the previous split, never a pass.
            _rs = time.time()
            try:
                _refresh = session.fast_grade(
                    workspace, run_dir, FL.FIRST_BACKGROUND_TICK + 100 + vatt, min(options.qa_timeout, 900)
                )
                _l3_eligible, _l3_held = CERT._l3_promotion(_refresh, _loop_tier)
                print(
                    f"[verilator fix-round {progress.round_index}] loop-tier refresh: {len(_l3_eligible)} eligible, "
                    f"{len(_l3_held)} still held back"
                )
            except Exception as _re:  # noqa: BLE001 — keep the previous split; never invent a promotion
                print(
                    f"[verilator fix-round {progress.round_index}] loop-tier refresh unavailable "
                    f"({type(_re).__name__}: {_re}); keeping the previous promotion split"
                )
            progress.active_wall_s += time.time() - _rs  # grading is active work, like every other grade here
            # A fix round may replace the compiler after the previously-conformant round.  Recompute from
            # THIS transcript + THIS submission immediately; retaining the old True would let prior tool
            # use mask a hand-authored/non-RTL final workflow.
            progress.latest_transcript = ROQ.latest_live_authoring_transcript(
                [*(run_dir / "rounds").glob("round_*.transcript.jsonl"), _fix_tpath]
            )
            conf, progress.workflow_conformant = _workflow_conformance(
                _fix_tpath, workspace / "submission", arm, session.endpoint_kind, session.resolved_tools()
            )
            _fix_bad = [k for k, v in (conf.get("checks") or {}).items() if v is False]
            if _fix_bad:
                print(
                    f"[verilator fix-round {progress.round_index}] NOT CONFORMANT — failing: "
                    f"{', '.join(_fix_bad)} (formal completion remains blocked)"
                )
            _fix_effort = ET.parse_transcript(
                _fix_tpath,
                billing_mode=EX.billing_mode(options.model, config=provider),
                trust_cli_cost=EX.trust_cli_cost(options.model, config=provider),
            )
            progress.rounds.append(
                {
                    "round": progress.round_index,
                    "mode": "verilator_fix",
                    "agent_rc": _fix_rc,
                    "conformance": conf,
                    "workflow_conformant": progress.workflow_conformant,
                    "authoring_complete": session.authoring_complete(),
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
            progress.round_index += 1
            session.checkpoint(progress.round_index)
        # realistic (abc2): the verilator barrier IS the definition of done. Make it drive `converged`
        # (so the watchdog stops correctly + finalize runs) rather than the spike gate.
        if options.experiment == "realistic":
            progress.verdict["all_pass"] = bool(vv["all_pass"])
            progress.verdict["n_passed"], progress.verdict["n_capsules"] = vv["n_passed"], vv["n_capsules"]
            session.checkpoint(progress.round_index)
