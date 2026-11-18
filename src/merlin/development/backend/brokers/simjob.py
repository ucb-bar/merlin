"""Driver-side ASYNC oracle broker — runs the agent's requested sims OUTSIDE the sandbox, redacted.

Companion to simjob_shim.py (the in-sandbox CLI). Watches <ws>/.qa_channel for `simreq_*.json`, runs each
as a SEPARATE PROCESS = the existing agent_selfcheck.py (the single grading+redaction authority — golden
expected values withheld), and writes back `simresp_<id>.json` + `simdone_<id>`. A timeout is a normal
redacted verdict, never a crash.

CONSTRAINED SIM-RUNNER (load-bearing isolation): a request may only name {an allowed simulator,
capsules that exist under the public capsule set, debug from a whitelist, workers (clamped)}. The broker
maps those to a FIXED agent_selfcheck.py argv — it NEVER execs anything the request names. So the agent
gets full sim power on its OWN submission + the harness capsules, but cannot read goldens/oracle or run
arbitrary shell outside the sandbox.

Bounded local pool (--max-jobs; unset = the shared compute pool size, else 4) + a GLOBAL cross-arm
verilator slot cap (file semaphore in a shared dir) so the 3 concurrent arms don't launch 3x the verilator
load. Exits on the existing STOP sentinel.

Usage: simjob_broker.py --ws <workspace> [--max-jobs N] [--veril-slots 2] [--poll 0.5]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from merlin.development.backend.authoring.library.compiler_library import selected_child
from merlin.development.backend.context import InvocationContext, add_context_arguments, resolve_context
from merlin.development.backend.feedback.certification import promotion as _TP
from merlin.development.backend.feedback.certification.promotion import (
    cert_already_executed as _cert_already_executed,
)
from merlin.development.backend.feedback.certification.promotion import promote as _promote  # noqa: E402
from merlin.development.backend.feedback.certification.promotion import resolve_tiers as _resolve_tiers
from merlin.development.backend.feedback.certification.promotion_admission import REQUIRED_SIM
from merlin.development.backend.feedback.certification.promotion_admission import (
    allowed_sims as _allowed_sims,
)
from merlin.development.backend.feedback.certification.promotion_admission import (
    grade_selection_for as _grade_selection_for,
)
from merlin.development.backend.feedback.execution.snapshots import recover_promotion_snapshot
from merlin.development.backend.feedback.grading import completions as _completions
from merlin.development.backend.feedback.selfcheck.diagnostic_adapters import _public_capsules, worker_command

# ---- simjob ------------------------------------------------------------------------------------


def _job_timeout_s(sim: str, ncaps: int, vpc: int) -> int:
    """Wall clock for ONE sim job: the derived per-capsule budget for every elaborated-RTL engine.

    This read ``(vpc * ncaps) if sim == "verilator" else 900``, so the engine `rtl_engine_policy`
    actually prefers was handed a flat 900 s no matter how large the request or how expensive the
    engine -- the literal "timed out after 900 seconds" that abandoned the heaviest capsules. The
    functional screen and the contract sentinel are not elaborated RTL and keep their own budget. The
    grade's own selection includes the elaborated-RTL tier, so it takes the RTL budget.
    """
    from merlin.development.backend.brokers.simjob_scheduling import _SCREEN_TIMEOUT_S, _rtl_engines

    rtl = sim in _rtl_engines() or sim == REQUIRED_SIM
    return (max(1, int(vpc)) * max(1, int(ncaps))) if rtl else _SCREEN_TIMEOUT_S


def _job_classes(env: dict, *, promoted: bool) -> dict:
    """Set one job's gSIM queue class and CPU lease class in its worker environment.

    The broker process leases at the author class, which is right for a simulation the author asked
    for and waits on. A promotion is background certification of a verdict the author already has:
    it waits behind author work for a gSIM slot, and it leases CPUs at the grade class, so an author
    self-check is served first. It is never preempted and still takes every slot no author waits for.
    """
    from merlin.common.process.cpu_pool import AUTHOR, GRADE, PRIORITY_ENV
    from merlin.development.backend.campaign import gsim_budget as gsim_queue

    env[gsim_queue.CLASS_ENV] = gsim_queue.BACKGROUND if promoted else gsim_queue.AUTHOR
    env[PRIORITY_ENV] = str(GRADE if promoted else AUTHOR)
    return env


def _default_max_jobs() -> int:
    """Every job's steps lease the shared compute pool, so its size is the only limiter (legacy: 4)."""
    from merlin.common.process.cpu_pool import compute_capacity

    return compute_capacity() or 4


@selected_child
def main(
    argv=None,
    *,
    context=None,
    capsules_root: Path | None = None,
    contract: Path | None = None,
    timing_file: Path | None = None,
    policy_capsules_root: Path | None = None,
):
    from merlin.development.backend.brokers.simjob_scheduling import (
        _NEUTRAL_SIM,
        _admitted_promotions,
        _capsule_workers,
        _completed_job_ids,
        _correctness_gsim,
        _current_source_first,
        _dedupe_sim_job,
        _freeze_direct_job_source,
        _joint_gsim_launchable,
        _observe_pending,
        _per_capsule_timeout,
        _promoted_model_continuation,
        _queue_wait_allowance,
        _release_veril_workers,
        _requeue_refused,
        _reserve_veril_workers,
        _sim_env,
        _spawn_selfcheck,
        _stop_sim_process,
        _strip_golden,
        _uses_verilator_slots,
        _valid_capsules,
    )

    ap = argparse.ArgumentParser()
    add_context_arguments(ap)
    ap.add_argument("--capsules-root", type=Path)
    ap.add_argument("--contract", type=Path)
    ap.add_argument("--timing-file", type=Path)
    ap.add_argument(
        "--policy-capsules-root", type=Path, help="frozen descriptor corpus for promotion, not the QA subset"
    )
    ap.add_argument("--ws", required=True)
    ap.add_argument("--rtl-facts", type=Path, help="Selected facts in the verified frozen input snapshot")
    ap.add_argument("--compiler-bundle-manifest", type=Path, help="host-owned archived common compiler profile")
    ap.add_argument(
        "--max-jobs", type=int, default=None, help="concurrent jobs; unset = the shared compute pool size, else 4"
    )
    ap.add_argument("--veril-slots", type=int, default=2)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument(
        "--per-capsule-timeout", type=int, default=0, help="0=derive from the cert engine's own measured cost law"
    )
    a = ap.parse_args(argv)
    if a.max_jobs is None:
        a.max_jobs = _default_max_jobs()
    from merlin.development.backend.session.inputs.run_record import select

    select(Path(a.ws), a.rtl_facts)
    context = resolve_context(a, ap, context)
    capsules_root = a.capsules_root if a.capsules_root is not None else capsules_root
    promotion_capsules_root = a.policy_capsules_root if a.policy_capsules_root is not None else policy_capsules_root
    if capsules_root is None:
        capsules_root = _public_capsules(context)
    contract = a.contract if a.contract is not None else contract
    timing_file = a.timing_file if a.timing_file is not None else timing_file
    if timing_file is None and context.harness is not None:
        timing_file = context.harness / ".oracle_timing.json"
    ws = Path(a.ws)
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)

    # Which tier is the cheap gate and which is the expensive cert, DERIVED from this target's own
    # adapter map -- never a literal, so a target with a different ladder gets the right split with no
    # edit here. loop = the fastest tier the corpus declares; cert = the deepest reachable above it.
    # Promotion is simply disabled (both None) when the endpoint exposes only one tier, rather than
    # inventing a second.
    _LOOP_TIER, _CERT_TIER, _COVER = _resolve_tiers(ws, context=context, capsules_root=promotion_capsules_root)
    print(
        f"[promote] loop={_LOOP_TIER} cert={_CERT_TIER} cover={len(_COVER) if _COVER is not None else 'all'}",
        file=sys.stderr,
        flush=True,
    )

    vpc, _budget_why = _per_capsule_timeout(a.per_capsule_timeout, context=context, timing_file=timing_file)
    print(f"[budget] per-capsule cert timeout {vpc}s: {_budget_why}", file=sys.stderr, flush=True)

    running: dict[str, dict] = {}  # jid -> {proc, slot, resp_tmp, sim}
    completed_by_key: dict[tuple[str, str, str, str], tuple[str, dict]] = {}
    # One broker is created per round, but the channel and its queue survive across rounds.  Starting
    # from an empty set replays every historical request, including paid L3 work.  A response is the
    # durable commit record; a bare simrun marker is not (the previous broker may have died mid-job), so
    # only responses are recovered here and interrupted requests remain retryable.
    claimed: set[str] = _completed_job_ids(ch)
    if _CERT_TIER is not None:
        recovered_results = {}
        _completions.recover(
            ws,
            cert_tier=_CERT_TIER,
            instrument=_TP.cert_instrument(_CERT_TIER, context=context),
            log=sys.stderr,
            recovered_results=recovered_results,
        )
        for jid, verdict in recovered_results.items():
            response = ch / f"simresp_{jid}.json"
            if not response.exists():
                response.write_text(json.dumps(verdict, indent=2))
                (ch / f"simdone_{jid}").write_text("recovered")
            claimed.add(jid)
    # STOP alone does not bound this broker's life: the sentinel is written by the driver, so if the
    # driver dies first nobody ever writes it and the broker polls forever -- while HOLDING sim slots and
    # child simulators. Three sibling brokers were found orphaned to init hours after their run ended,
    # spawned for a round that never started. Exit when the process that started us is gone.
    queued_since = {}
    orig_ppid = os.getppid()
    next_source_check = 0.0
    while True:
        if (ch / "STOP").exists() or os.getppid() != orig_ppid:
            break
        # reap finished
        for jid, j in list(running.items()):
            if j["proc"].poll() is None:
                continue
            rc = j["proc"].returncode
            timing = dict(
                j["timing"],
                finished_at_unix=time.time(),
                worker_runtime_s=max(0.0, time.monotonic() - j["started_monotonic"]),
                returncode=rc,
            )
            (ch / f"simtiming_{jid}.json").write_text(json.dumps(timing, indent=2))
            resp = ch / f"simresp_{jid}.json"
            tmp = j["resp_tmp"]
            try:
                out = (
                    _strip_golden(json.loads(Path(tmp).read_text()))
                    if Path(tmp).exists()
                    else {
                        "error": f"no verdict produced (rc={rc}); child output in simlog_{jid}.txt",
                        "all_pass": False,
                    }
                )
            except Exception as e:
                out = {"error": f"verdict parse: {e}", "all_pass": False}
            if rc == 124:  # timeout exit
                out = {
                    "sim": j["sim"],
                    "state": "timeout",
                    "all_pass": False,
                    "error": f"{j['sim']} exceeded its time budget",
                }
            if _requeue_refused(ch, jid, j, out, claimed, queued_since):
                running.pop(jid)
                continue
            if j.get("promoted") and not out.get("error"):
                try:
                    _completions.persist(
                        ws,
                        jid,
                        out,
                        identity=j.get("identity"),
                        source_digest=j.get("source_digest"),
                        source_identity_verified=bool(j.get("source_identity_verified")),
                        cert_tier=_CERT_TIER,
                        instrument=j.get("cert_instrument"),
                    )
                except Exception as exc:  # noqa: BLE001 -- ordinary recording still runs
                    print(
                        f"[promote] completion custody not saved: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            resp.write_text(json.dumps(out, indent=2))
            (ch / (f"simerr_{jid}" if out.get("error") else f"simdone_{jid}")).write_text("ok")
            if not out.get("joint_policy_refused"):
                # A refusal ran nothing; an identical later request must be admitted afresh, not answered
                # from this cache.
                completed_by_key[j["dedupe_key"]] = (jid, out)
            for alias in j.get("aliases", []):
                (ch / f"simresp_{alias}.json").write_text(json.dumps(out, indent=2))
                (ch / (f"simerr_{alias}" if out.get("error") else f"simdone_{alias}")).write_text("deduplicated")
            # PROMOTE: a capsule that just passed the loop tier earns the cert tier now, not at a round
            # boundary hours away. Skipped for a job that WAS a promotion, so a cert verdict cannot
            # re-enqueue itself.
            if not j.get("promoted") and not out.get("error"):
                try:
                    if _cert_already_executed(out, _CERT_TIER):
                        print(
                            f"[promote] {jid}: not enqueued; it already ran {_CERT_TIER}", file=sys.stderr, flush=True
                        )
                    else:
                        _promote(
                            ws,
                            ch,
                            out,
                            _LOOP_TIER,
                            _CERT_TIER,
                            _COVER,
                            sys.stderr,
                            context=context,
                            capsules_root=promotion_capsules_root,
                            source_ws=j["snapshot_root"],
                        )
                except Exception as _pe:  # noqa: BLE001 -- promotion is an optimisation, never a gate
                    print(f"[promote] skipped: {type(_pe).__name__}: {_pe}", file=sys.stderr, flush=True)
            elif j.get("promoted"):
                # A promotion's own verdict: record it, so the cert this just PAID FOR on real RTL is
                # kept instead of discarded. Without this the capsule stays `pending` forever and the
                # same bytes are re-certified on the next loop verdict.
                try:
                    # Forward the identity the promotion was ENQUEUED for, so the result lands on the
                    # exact record it belongs to even when the verdict reader produced no per-capsule
                    # artifact identity. Absent (a request written before this field existed) it is
                    # None, and the recorder keeps its previous attribution rule.
                    _TP.record_cert(
                        ws,
                        out,
                        _CERT_TIER,
                        sys.stderr,
                        identity=j.get("identity"),
                        source_identity_verified=bool(j.get("source_identity_verified")),
                        request=jid,
                    )
                except Exception as _re:  # noqa: BLE001 -- recording must never gate a run either
                    print(f"[promote] record skipped: {type(_re).__name__}: {_re}", file=sys.stderr, flush=True)
            if j.get("log"):
                try:
                    j["log"].close()
                except Exception:  # noqa: BLE001 -- closing a log must never break the reap
                    pass
            _release_veril_workers(j["slots"])
            if j.get("snapshot_root"):
                shutil.rmtree(j["snapshot_root"], ignore_errors=True)
            running.pop(jid)
        # A running job owns an immutable source snapshot. Finish it so its exact
        # executable certificate can serve a later revision that emits identical
        # bytes. The result remains attributed to its original source; queued
        # obsolete promotions are still filtered before spending a native slot.
        now = time.monotonic()
        if running and now >= next_source_check:
            next_source_check = now + 5
            try:
                current_source = _TP._submission_digest(ws)
            except (OSError, ValueError):
                current_source = None  # a mid-edit source is retried at the next observation
            if current_source is not None:
                for jid, job in list(running.items()):
                    if job["source_digest"] != current_source:
                        job["superseded_by_source"] = current_source
        # launch queued (respect local + global caps)
        queued = _observe_pending(ch.glob("simreq_*.json"), claimed, queued_since, time.monotonic())
        if len(running) < a.max_jobs:
            gsim_gate = {}
            ordered = _current_source_first(
                queued,
                ws,
                context=context,
                capsules_root=capsules_root,
                cert_tier=_CERT_TIER,
                contract=contract,
            )
            for req in _admitted_promotions(
                ordered,
                ws,
                loop_tier=_LOOP_TIER,
                cert_tier=_CERT_TIER,
                context=context,
            ):
                jid = req.stem[len("simreq_") :]
                if jid in claimed:
                    continue
                if len(running) >= a.max_jobs:
                    break
                r = json.loads(req.read_text()) if req.exists() else {}
                sim = r.get("sim")
                caps = _valid_capsules(str(r.get("capsules", "all")), capsules_root)
                if sim not in _allowed_sims(context) or caps is None:
                    # Say WHICH of the two it was, and what would be accepted. The old message named
                    # neither, and a request rejected without a remedy reads as "the oracle is broken":
                    # measured on a live run, an agent submitted twice, was rejected twice with this
                    # text, and never used the async oracle again -- while the arm that DID find the
                    # async path used it 98 times in the round its score moved 17 -> 26. An unhelpful
                    # rejection costs more than the check it protects.
                    _allowed = _allowed_sims(context)
                    if sim not in _allowed:
                        _why = (
                            f"--sim {sim!r} is not accepted for this target. Use "
                            f"{' or '.join(repr(s) for s in _allowed)}"
                            + (
                                f"; this target's tier comes from its contract, so {_NEUTRAL_SIM!r} "
                                f"means 'grade on whatever tier the contract resolves to' and the "
                                f"tier itself is chosen with --tiers"
                                if _allowed == (_NEUTRAL_SIM,)
                                else ""
                            )
                        )
                    else:
                        _named = str(r.get("capsules", "all"))
                        _why = (
                            f"--capsules {_named!r} named something this runner will not run: a "
                            f"capsule must be an existing public capsule directory name (letters, "
                            f"digits, underscore), or 'all'"
                        )
                    (ch / f"simresp_{jid}.json").write_text(
                        json.dumps(
                            {
                                "error": f"rejected: {_why}",
                                "all_pass": False,
                                "rejected_field": "sim" if sim not in _allowed else "capsules",
                            }
                        )
                    )
                    (ch / f"simerr_{jid}").write_text("rejected")
                    claimed.add(jid)
                    continue
                if not r.get("promoted"):
                    # An author job naming the pinned engine runs the grade's own selection (same
                    # tiers). A promotion keeps its single cert tier and its own ledger accounting.
                    sim = _grade_selection_for(sim)
                if not _joint_gsim_launchable(r, sim, gsim_gate, running):
                    continue  # every correctness gSIM slot is busy or wanted by an author; stays queued
                ncaps = len(_valid_capsules("all", capsules_root) or []) if caps == ["all"] else len(caps)
                workers = _capsule_workers(r.get("workers", 0), sim=sim, n_capsules=ncaps)
                veril = _uses_verilator_slots(sim)
                slots = _reserve_veril_workers(workers, a.veril_slots) if veril else ()
                if veril:
                    if not slots:
                        continue  # global verilator budget full; try later
                    workers = len(slots)
                submission = Path(ws) / "submission"
                snapshot_root = None
                source_identity_verified = False
                if bool(r.get("promoted")):
                    submission, snapshot_root, snapshot_error = recover_promotion_snapshot(
                        Path(ws),
                        r.get("submission_digest"),
                        r.get("identity"),
                        r.get("submission_snapshot"),
                        submission_digest=_TP._submission_digest,
                    )
                    if snapshot_error:
                        _release_veril_workers(slots)
                        (ch / f"simresp_{jid}.json").write_text(
                            json.dumps(
                                {"error": snapshot_error, "all_pass": False, "promotion_source_verified": False},
                                indent=2,
                            )
                        )
                        (ch / f"simerr_{jid}").write_text("source identity mismatch")
                        print(f"[promote] {jid} not launched: {snapshot_error}", file=sys.stderr, flush=True)
                        claimed.add(jid)
                        continue
                    source_identity_verified = True
                else:
                    submission, snapshot_root, snapshot_error = _freeze_direct_job_source(ws)
                    if snapshot_error:
                        _release_veril_workers(slots)
                        (ch / f"simresp_{jid}.json").write_text(
                            json.dumps({"error": snapshot_error, "all_pass": False}, indent=2)
                        )
                        (ch / f"simerr_{jid}").write_text("source identity mismatch")
                        claimed.add(jid)
                        continue
                    source_identity_verified = True
                from merlin.development.backend.feedback.guidance.construction_plan import PlanRequired
                from merlin.development.backend.feedback.guidance.construction_plan import (
                    admit as admit_construction_plan,
                )

                try:
                    construction_admission = admit_construction_plan(submission, r, request_id=jid, channel="simjob")
                except (PlanRequired, ValueError) as error:
                    _release_veril_workers(slots)
                    shutil.rmtree(snapshot_root, ignore_errors=True)
                    (ch / f"simresp_{jid}.json").write_text(
                        json.dumps(
                            {
                                "status": "not_scheduled",
                                "all_pass": False,
                                "numeric_assessment": "not_run",
                                "error": str(error),
                            }
                        )
                    )
                    (ch / f"simerr_{jid}").write_text("construction plan required")
                    claimed.add(jid)
                    continue
                if construction_admission is not None:
                    (ch / f"construction_plan_sim_{jid}.json").write_text(json.dumps(construction_admission))
                capspec = "all" if caps == ["all"] else ",".join(caps)
                launched_source_digest = _TP._submission_digest(snapshot_root)
                launched_instrument = _TP.cert_instrument(_CERT_TIER, context=context) if r.get("promoted") else None
                to = _job_timeout_s(sim, ncaps, vpc)
                from merlin.development.backend.campaign import control as joint_control

                if joint_control.selected() is not None and sim == "gsim":
                    to = min(to, joint_control.physical_policy(capspec)["per_job_s"])
                resp_tmp = ch / f"simtmp_{jid}.json"
                worker_kwargs = {"rtl_facts": a.rtl_facts} if a.rtl_facts is not None else {}
                if a.compiler_bundle_manifest is not None:
                    worker_kwargs["compiler_bundle_manifest"] = a.compiler_bundle_manifest
                argv2 = worker_command(context, capsules_root, contract, **worker_kwargs) + [
                    "--submission",
                    str(submission),
                    "--capsules",
                    capspec,
                    "--workers",
                    str(workers),
                    "--timeout",
                    str(to),
                    "--out",
                    str(resp_tmp),
                    "--progress-out",
                    str(ch / f"progress_{jid}.json"),
                ]
                if sim != _NEUTRAL_SIM:  # --sim is meaningless where the contract picks the tier
                    argv2 += ["--sim", sim]
                _tiers = str(r.get("tiers") or "").strip()
                model_vouchers = []
                try:
                    if r.get("promoted") and caps != ["all"]:
                        _tiers, model_vouchers = _promoted_model_continuation(
                            ws,
                            caps,
                            capsules_root,
                            _tiers,
                            digest=launched_source_digest,
                            context=context,
                            contract=contract,
                            cert_tier=_CERT_TIER,
                        )
                        for voucher in model_vouchers:
                            argv2 += ["--model-gate-voucher", str(voucher)]
                except (ValueError, OSError) as exc:
                    _release_veril_workers(slots)
                    (ch / f"simresp_{jid}.json").write_text(
                        json.dumps(
                            {
                                "error": f"model continuation refused: {exc}",
                                "all_pass": False,
                                "promotion_source_verified": source_identity_verified,
                            }
                        )
                    )
                    (ch / f"simerr_{jid}").write_text("model continuation not admitted")
                    claimed.add(jid)
                    continue
                if _tiers:  # validated downstream against the RESOLVED adapter map
                    argv2 += ["--tiers", _tiers]
                from merlin.development.backend.campaign.calibration.activation import (
                    request_binding,
                    request_environment,
                )

                worker_env = request_environment(
                    _sim_env(context), channel="simjob", request_id=jid, request_bytes=req.read_bytes()
                )
                from merlin.development.backend.campaign import gsim_budget as gsim_queue

                # Promotions never wait inside their child (they wait here, queued); author jobs may.
                _job_classes(worker_env, promoted=bool(r.get("promoted")))
                worker_env[gsim_queue.REQUEST_ENV] = jid
                model_binding = request_binding(worker_env)
                model_identity = (model_binding or {}).get("selection")
                dedupe_key = (launched_source_digest, capspec, sim, _tiers, json.dumps(model_identity, sort_keys=True))
                if _dedupe_sim_job(ch, jid, dedupe_key, running, completed_by_key):
                    _release_veril_workers(slots)
                    shutil.rmtree(snapshot_root, ignore_errors=True)
                    claimed.add(jid)
                    continue
                (ch / f"simrun_{jid}").write_text("running")
                # KEEP THE CHILD'S OUTPUT. This was stdout/stderr=DEVNULL, so when a job exited without
                # writing its verdict file the only trace was the broker's own "no verdict produced" --
                # a job that failed and a job that produced nothing were indistinguishable, and the
                # REASON was gone. Measured on merlincirct_arm4_func_20260901_v4: 19 promotion jobs
                # answered "no verdict produced" with no diagnostic anywhere on disk. The log is
                # per-job, beside the response, so a failure can be read after the fact.
                job_log = (ch / f"simlog_{jid}.txt").open("wb")
                try:
                    # Broker-owned frozen source/full policy root only. Requests cannot
                    # provide a gate counter, record path or narrower denominator.
                    from merlin.development.backend.feedback.certification.rolling_model_gate import (
                        broker_environment,
                    )

                    worker_env.update(
                        broker_environment(ws, promotion_capsules_root or capsules_root, snapshot_root / "submission")
                    )
                    proc = _spawn_selfcheck(
                        argv2, cwd=ws, env=worker_env, job_log=job_log, timeout_s=to + _queue_wait_allowance(r, sim)
                    )
                except Exception:
                    job_log.close()
                    _release_veril_workers(slots)
                    if snapshot_root:
                        shutil.rmtree(snapshot_root, ignore_errors=True)
                    raise
                started = time.monotonic()
                timing = {
                    "schema": "simjob_resource_timing_v1",
                    "job_id": jid,
                    "source_digest": launched_source_digest,
                    "started_at_unix": time.time(),
                    "queue_wait_s": max(0.0, started - queued_since.pop(jid, started)),
                    "reserved_verilator_slots": len(slots),
                    "worker_pid": proc.pid,
                    "configured_gsim_capacity": os.environ.get("MERLIN_GSIM_MAX_SLOTS"),
                    "configured_gsim_cpus": os.environ.get("MERLIN_GSIM_CPUS"),
                    "scope": "broker-observed queue wait and worker runtime; not native-only runtime",
                }
                (ch / f"simtiming_{jid}.json").write_text(json.dumps(timing, indent=2))
                running[jid] = {
                    "timing": timing,
                    "started_monotonic": started,
                    "proc": proc,
                    "slots": slots,
                    "resp_tmp": str(resp_tmp),
                    "sim": sim,
                    "promoted": bool(r.get("promoted")),
                    "log": job_log,
                    # which tier-state record this promotion was launched for (see the reap)
                    "identity": r.get("identity"),
                    "source_identity_verified": source_identity_verified,
                    "source_digest": launched_source_digest,
                    "cert_instrument": launched_instrument,
                    "snapshot_root": snapshot_root,
                    "dedupe_key": dedupe_key,
                    "aliases": [],
                    "correctness_gsim": _correctness_gsim(r, sim),
                }
                claimed.add(jid)
        time.sleep(a.poll)
    # drain on STOP
    for j in running.values():
        _stop_sim_process(j["proc"])
        _release_veril_workers(j["slots"])
        if j.get("snapshot_root"):
            shutil.rmtree(j["snapshot_root"], ignore_errors=True)


if __name__ == "__main__":
    from merlin.development.backend.telemetry.evidence import install

    install()  # A full log volume drops log lines, never the broker.
    from merlin.common.process.cpu_pool import AUTHOR, cpu_priority

    with cpu_priority(AUTHOR, export=True):  # Author-requested simulation jobs lease first.
        raise SystemExit(main())
