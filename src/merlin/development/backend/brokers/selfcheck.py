"""Driver-side self-check BROKER — lets a bwrap-sandboxed agent get a REDACTED self-check verdict without
the oracle ever entering its sandbox.

The redacted self-check (agent_selfcheck.py) internally needs the oracle (merlin.runtime.engine.reference) to
compute pass/fail — so it CANNOT run inside the agent's sandbox (the oracle is masked there). This broker
runs OUTSIDE the sandbox (oracle available), watches a channel dir inside the agent's (bind-mounted)
workspace, and on each request runs the real agent_selfcheck.py against the agent's submission, writing the
redacted verdict + the agent's OWN artifacts back into the workspace. The agent talks to it via the thin
shim staged at <ws>/agent_selfcheck.py.

Channel (under <ws>/.qa_channel/, RW from both sides via the bind mount):
  req_<id>.json   agent -> broker : {submission, capsules, workers, timeout, shape_coverage,
                                     offload_census, model_layers, optional sim/tiers}
  resp_<id>.json  broker -> agent : the redacted self-check JSON (golden expected values withheld)
  done_<id>       broker -> agent : marker that resp is complete
  detached_<id>   agent -> broker : the waiting client returned on purpose; keep the request until
                                    its deadline instead of judging it by the client's process
  STOP            driver -> broker: sentinel to exit (written after the agent round)

``submission`` is ``submission`` (the default) or ``variants/<name>``: see ``selfcheck_sources``.

Usage: selfcheck_broker.py --ws <workspace> [--poll 0.5]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from merlin.development.backend.authoring.library.compiler_library import selected_child
from merlin.development.backend.brokers import selfcheck_sources as sources
from merlin.development.backend.brokers import selfcheck_tuning as tuning
from merlin.development.backend.brokers.selfcheck_dispatch import (
    CAPSULES_PER_COMPUTE_CPU,
    COMMON_ACTIONS,
    _abort_reason,
    _author_cpu_backlog,
    _child_deadline_monotonic,
    _detached,
    _diagnostic_request,
    _dispatch_capacity,
    _feedback_scope_error,
    _lane_admits,
    _ordinary_capacity,
    _ordinary_capsule_limit,
    _overflow_admits,
    _pending_requests,
    _pid_namespace,
    _process_record,
    _public_corpus_size,
    _request_deadline,
    _requester_alive,
    _should_stop,
    _stop_owned_worker,
)
from merlin.development.backend.context import add_context_arguments, context_argv, resolve_context
from merlin.development.backend.feedback.certification.promotion_admission import (
    REQUIRED_SIM,
    grade_selection_for,
)
from merlin.development.backend.feedback.performance.idle_measurement import OPERATOR_ORIGINS, TUNING_ORIGIN
from merlin.development.backend.feedback.performance.idle_measurement import (
    ORIGIN as IDLE_ORIGIN,  # noqa: F401 -- stable name
)
from merlin.development.backend.feedback.performance.idle_measurement import (
    REQUEST_PREFIX as IDLE_REQUEST_PREFIX,
)
from merlin.development.backend.feedback.references.analysis import (
    broker_response as _reference_analysis_response,
)
from merlin.development.backend.feedback.selfcheck.diagnostic_adapters import _public_capsules, worker_command

# ---- selfcheck ---------------------------------------------------------------------------------


@selected_child
def main(
    argv=None,
    *,
    context=None,
    capsules_root: Path | None = None,
    contract: Path | None = None,
    policy_capsules_root: Path | None = None,
):
    from merlin.development.backend.brokers.selfcheck_dispatch import (
        FORWARDED_PROBES,
        LOOP_ERROR_BACKOFF_S,
        PROTOCOL_VERSION,
        _advisory_request,
        _atomic_write,
        _decorate_response,
        _default_sim,
        _dispatch_requests,
        _freeze_passed_source,
        _legacy_common_service,
        _log_tail,
        _report_loop_error,
        _required_rtl_engine,
        _settle_failed_request,
        _sim_policy_error,
        _submission_digest,
        _working_snapshot,
    )

    ap = argparse.ArgumentParser()
    add_context_arguments(ap)
    ap.add_argument("--capsules-root", type=Path)
    ap.add_argument("--contract", type=Path)
    ap.add_argument(
        "--policy-capsules-root", type=Path, help="frozen descriptor corpus for promotion, not the QA subset"
    )
    ap.add_argument("--ws", required=True)
    ap.add_argument("--rtl-facts", type=Path, help="Selected facts in the verified frozen input snapshot")
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--compiler-bundle-manifest", type=Path, help="host-owned archived common compiler profile")
    ap.add_argument("--one-request", help=argparse.SUPPRESS)
    ap.add_argument("--request-stop", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--request-sha256", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    from merlin.development.backend.session.inputs.run_record import select

    select(Path(a.ws), a.rtl_facts)
    context = resolve_context(a, ap, context)
    capsules_root = a.capsules_root if a.capsules_root is not None else capsules_root
    promotion_capsules_root = a.policy_capsules_root if a.policy_capsules_root is not None else policy_capsules_root
    if capsules_root is None:
        capsules_root = _public_capsules(context)
    contract = a.contract if a.contract is not None else contract
    ws = Path(a.ws)
    global _AGENT_ROOTS
    _AGENT_ROOTS = (str(ws), str(ws.resolve()))
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    if a.one_request is not None and (
        Path(a.one_request).name != a.one_request
        or not a.one_request.startswith("req_")
        or not a.one_request.endswith(".json")
        or a.one_request[len("req_") : -len(".json")] in {"", ".", ".."}
    ):
        ap.error("single request must name one channel request")
    if a.request_stop is not None and a.request_stop.parent.resolve() != ch.resolve():
        ap.error("request stop must belong to the original channel")
    if a.one_request is None and _dispatch_capacity() == 2:
        return _dispatch_requests(a, context, capsules_root, contract, promotion_capsules_root)
    common_service = _legacy_common_service(ws, a.compiler_bundle_manifest, context.repo)
    orig_ppid = os.getppid()
    seen: set[str] = set()
    health_path = (
        ch / "broker_health.json"
        if a.one_request is None
        else ch / f"health_{a.one_request[len('req_') : -len('.json')]}.json"
    )
    try:
        health = json.loads(health_path.read_text(encoding="utf-8"))
    except Exception:
        health = {}
    health = {
        "protocol": PROTOCOL_VERSION,
        "broker_starts": int(health.get("broker_starts", 0)) + 1,
        "processed": int(health.get("processed", 0)),
        "replayed": int(health.get("replayed", 0)),
        "max_queue_depth": int(health.get("max_queue_depth", 0)),
        "last_started_at_unix_ns": time.time_ns(),
    }
    _atomic_write(health_path, json.dumps(health, sort_keys=True))
    while True:
        served = None
        try:
            if a.one_request is not None and seen:
                return 0
            why = _should_stop(ch, orig_ppid)
            if a.request_stop is not None and a.request_stop.exists():
                why = "dispatcher stopped"
            if why:
                health["last_stopped_at_unix_ns"] = time.time_ns()
                health["last_stop_reason"] = why
                _atomic_write(health_path, json.dumps(health, sort_keys=True))
                print(f"[broker] exiting: {why}", file=sys.stderr)
                break
            # FIFO by ARRIVAL TIME. The previous ``sorted(glob)`` ordered by FILENAME, and a request is named
            # req_<requester-pid>_<nonce>, so service order followed pid values rather than arrival: a request
            # could be deferred behind every newly-arrived lower-sorting name and never claimed at all. That
            # was measured — an agent spent the remainder of its round waiting on a request submitted 10
            # minutes earlier while a later one was served, then ended the round with 3.5h of its budget
            # unused. Oldest-first also means the request the agent is actually blocked on is the one served.
            pending = _pending_requests(ch, seen)
            if a.one_request is not None:
                pending = [request for request in pending if request.name == a.one_request]
                if not pending:
                    return 0
            if len(pending) > int(health.get("max_queue_depth", 0)):
                health["max_queue_depth"] = len(pending)
                _atomic_write(health_path, json.dumps(health, sort_keys=True))
            if not pending:
                time.sleep(a.poll)
                continue
            # Serve ONE request per pass, then re-check STOP/parent: a self-check can run for minutes, and
            # draining a whole batch first is what let work continue after the round was over.
            req = pending[0]
            seen.add(req.name)
            served = req
            rid = req.stem[len("req_") :]
            resp = ch / f"resp_{rid}.json"
            # CLAIMED, where a waiting client can see it. One request is served at a time, and nothing
            # said which one: a client whose request sat behind a long grade saw no file of any kind and
            # could only conclude the broker was down. Measured on one run: four requests submitted
            # behind a 77-minute grade, none acknowledged, the agent reasoning "confirming a broker-wide
            # outage" and cancelling all four. The marker names the request being served and how many
            # are behind it, so "queued" and "broken" stop looking alike.
            _atomic_write(
                ch / f"claimed_{rid}",
                json.dumps({"claimed_at_unix_ns": time.time_ns(), "queued_behind_this": len(pending) - 1}),
            )
            try:
                request_bytes = req.read_bytes()
                if a.request_sha256 is not None and hashlib.sha256(request_bytes).hexdigest() != a.request_sha256:
                    raise ValueError("dispatched request bytes changed")
                r = json.loads(request_bytes)
            except Exception:
                _atomic_write(resp, json.dumps({"error": "broker: unreadable request"}))
                _atomic_write(ch / f"done_{rid}", "err")
                continue
            started_ns = time.time_ns()
            scope_error = _feedback_scope_error(r, corpus_size=_public_corpus_size(capsules_root))
            if scope_error is not None:
                doc = _decorate_response(
                    json.dumps(
                        {
                            "status": "not_scheduled",
                            "all_pass": None,
                            "error": scope_error,
                            "numeric_assessment": "not_run",
                        }
                    ),
                    r,
                    rid,
                    r.get("submission_sha256"),
                    started_ns=started_ns,
                    completed_ns=time.time_ns(),
                )
                _atomic_write(resp, json.dumps(doc))
                _atomic_write(ch / f"done_{rid}", "not_scheduled")
                continue
            try:
                # WHICH TREE: submission/ or an author-side variants/<name>/, refused before any work when
                # the request names anything else or the tree is not a real directory in the workspace.
                label = sources.request_label(r)
                graded_digest = sources.source_digest(ws, label, _submission_digest)
            except (OSError, ValueError) as error:
                from merlin.common.access import scrub_host_paths

                doc = _decorate_response(
                    json.dumps({"error": f"broker: {scrub_host_paths(str(error))}", "all_pass": False}),
                    r,
                    rid,
                    None,
                    started_ns=started_ns,
                    completed_ns=time.time_ns(),
                )
                _atomic_write(resp, json.dumps(doc))
                _atomic_write(ch / f"done_{rid}", "err")
                continue
            variant = label != sources.SUBMISSION or tuning.requested(r)
            if int(r.get("protocol", 1)) >= PROTOCOL_VERSION:
                # A REQUEST WHOSE BYTES MOVED IS SERVED, NOT REFUSED. Rejecting it was a livelock: the
                # agent asks for a check and keeps working -- which is the behaviour the tool is for -- so
                # by the time the queue drains the digest has moved and the request is stale. Every request
                # is then refused with "retry the check", the retry is stale too, and the loop goes blind.
                # Measured on merlincirct_atlas_feedback_v3_20260906: five queued requests, three distinct
                # digests, all refused; the last COMPLETED self-check was 5.5 h earlier while the agent kept
                # editing. Grading the current bytes answers the question actually being asked ("how is my
                # work doing?"), and `_decorate_response` states which bytes were graded so nothing is
                # mis-attributed. Only a genuine channel error -- a reply that would land under the wrong
                # request -- is still refused.
                identity_error = None
                if r.get("request_id") != rid:
                    identity_error = "request id does not match its channel filename"
                if identity_error:
                    doc = _decorate_response(
                        json.dumps({"error": f"broker: {identity_error}", "all_pass": False}),
                        r,
                        rid,
                        graded_digest,
                        started_ns=started_ns,
                        completed_ns=time.time_ns(),
                    )
                    _atomic_write(resp, json.dumps(doc))
                    _atomic_write(ch / f"done_{rid}", "err")
                    continue
            from merlin.development.backend.authoring.profiles import check_broker_action

            try:
                check_broker_action("candidate-diagnostics" if r.get("candidate_diagnostics") else r.get("action"))
            except FileNotFoundError as error:
                body = {"status": "unavailable", "reason": str(error), "all_pass": False}
                doc = _decorate_response(
                    json.dumps(body), r, rid, graded_digest, started_ns=started_ns, completed_ns=time.time_ns()
                )
                _atomic_write(resp, json.dumps(doc))
                _atomic_write(ch / f"done_{rid}", "ok")
                continue
            if r.get("action") == "mechanism-question":
                try:
                    from merlin.development.backend.campaign.calibration.joint_mechanism_questions import (
                        submit,
                    )

                    body = submit(r.get("mechanism_question"), request_id=rid)
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                    body = {
                        "status": "mechanism_question_denied",
                        "reason": str(error),
                        "all_pass": False,
                        "numeric_assessment": "not_run",
                        "fit_eligible": False,
                    }
                doc = _decorate_response(
                    json.dumps(body), r, rid, graded_digest, started_ns=started_ns, completed_ns=time.time_ns()
                )
                _atomic_write(resp, json.dumps(doc))
                _atomic_write(ch / f"done_{rid}", "ok")
                continue
            if r.get("action") == "reference-analysis":
                body = _reference_analysis_response(
                    r,
                    workspace=ws,
                    facts=a.rtl_facts,
                    compiler_sha256=graded_digest,
                    rid=rid,
                    started_ns=started_ns,
                    bundle_manifest=a.compiler_bundle_manifest,
                    repo=context.repo,
                    descriptor=context.descriptor,
                    target=context.target,
                )
                _atomic_write(resp, json.dumps(body))
                _atomic_write(ch / f"done_{rid}", "ok")
                continue
            if r.get("action") in COMMON_ACTIONS:
                try:
                    if common_service is None:
                        raise FileNotFoundError("versioned common transformation tooling was not admitted for this run")
                    if r["action"] == "common-check":
                        # The current common substrate admits no common check on this channel.
                        raise FileNotFoundError("common check was not admitted by the original compiler profile")
                    proof_root = a.compiler_bundle_manifest.parent / "common-transform-work"
                    proof_root.mkdir(mode=0o700, exist_ok=True)
                    remaining = int(_child_deadline_monotonic(r) - time.monotonic())
                    if remaining < 1:
                        raise TimeoutError("common request exhausted its original client deadline")
                    method = common_service.check if r["action"] == "common-check" else common_service.apply
                    body = method(r.get("transform"), work_root=proof_root, timeout_s=remaining)
                except Exception as error:
                    from merlin.common.access import scrub_host_paths

                    body = {
                        "status": "unavailable" if isinstance(error, FileNotFoundError) else "unknown",
                        "reason": scrub_host_paths(str(error)),
                        "all_pass": False,
                    }
                if r["action"] == "common-check":
                    # Error summaries remain bound to the actual request too. These
                    # hashes describe bytes; they do not turn a refusal into proof.
                    payload = r.get("transform")
                    if isinstance(payload, dict):
                        for field, key in (("before_ir", "source_sha256"), ("after_ir", "target_sha256")):
                            if isinstance(payload.get(field), str):
                                body.setdefault(key, hashlib.sha256(payload[field].encode()).hexdigest())
                doc = _decorate_response(
                    json.dumps(body),
                    r,
                    rid,
                    graded_digest,
                    started_ns=started_ns,
                    completed_ns=time.time_ns(),
                )
                _atomic_write(resp, json.dumps(doc))
                _atomic_write(ch / f"done_{rid}", "ok")
                health["processed"] = int(health.get("processed", 0)) + 1
                health["last_completed_at_unix_ns"] = time.time_ns()
                _atomic_write(health_path, json.dumps(health, sort_keys=True))
                continue
            inspection = r.get("caller_layout") or r.get("candidate_diagnostics") or r.get("compile_only")
            suite = r.get("suite", "core")
            advisory = _advisory_request(r)
            suite_error = None
            measurement = None  # an operator-authorized idle gSIM measurement (joint_gsim_measurement)
            if suite in {"scalability", "programs"}:
                if suite == "programs":
                    from merlin.development.backend.feedback.certification.model_checks import (
                        request_policy,
                        selected,
                    )
                else:
                    from merlin.development.backend.feedback.scalability.public_scalability import (
                        request_policy,
                        selected,
                    )
                from merlin.development.backend.campaign.joint_gsim_measurement import broker_admission

                probing = bool(inspection or any(r.get(k) for k in FORWARDED_PROBES))
                # Only the scalability suite has an idle gSIM measurement lane.
                admits = suite == "scalability" and not (probing or variant)
                measurement = broker_admission(rid, r) if admits else None
                try:
                    if measurement is None:
                        source_inspection = None
                        if suite == "scalability":
                            from merlin.development.backend.feedback.scalability.public_scalability_inspection import (
                                requested,
                            )

                            source_inspection = requested(r)
                        request_policy(r.get("sim"), r.get("tiers"), inspection=probing and source_inspection is None)
                    if selected(context) is None:
                        raise ValueError("no operator public suite is selected")
                except (ValueError, TypeError) as exc:
                    suite_error = f"public {suite} request refused ({type(exc).__name__})"
            elif suite != "core":
                suite_error = "unknown public suite"
            # A variant or tuning A/B without a named engine is a performance comparison: Spike cycle model.
            default_sim = "spike" if variant else _default_sim()
            sim = None if inspection else str(r.get("sim") or default_sim or "") or None
            if suite in {"scalability", "programs"}:
                sim = "gsim" if measurement is not None else "spike"
            policy_error = suite_error or (_sim_policy_error(sim) if sim and suite == "core" else None)
            if policy_error:
                doc = _decorate_response(
                    json.dumps(
                        {
                            "error": f"broker: {policy_error}",
                            "all_pass": False,
                            "sim": sim,
                            "required_rtl_engine": _required_rtl_engine(),
                        }
                    ),
                    r,
                    rid,
                    graded_digest,
                    started_ns=started_ns,
                    completed_ns=time.time_ns(),
                )
                _atomic_write(resp, json.dumps(doc))
                _atomic_write(ch / f"done_{rid}", "err")
                continue
            sim = grade_selection_for(sim) if suite == "core" else ("gsim" if measurement is not None else "spike")
            to = int(r.get("timeout", 1800))
            from merlin.development.backend.campaign import control as joint_control

            if joint_control.selected() is not None and sim == "gsim":
                to = min(
                    to,
                    measurement["timeout_s"]
                    if measurement is not None
                    else joint_control.physical_policy(str(r.get("capsules", "all")))["per_job_s"],
                )
            worker_kwargs = {"rtl_facts": a.rtl_facts} if a.rtl_facts is not None else {}
            if a.compiler_bundle_manifest is not None:
                worker_kwargs["compiler_bundle_manifest"] = a.compiler_bundle_manifest
            argv2 = worker_command(context, capsules_root, contract, **worker_kwargs) + [
                "--submission",
                str(ws / "submission"),
                "--capsules",
                str(r.get("capsules", "all")),
                "--workers",
                str(r.get("workers", 0)),
                "--timeout",
                str(to),
                # The real self-check writes --out directly.  Point it at a private name and publish
                # by rename only after it exits, so the shim can never read a half-written verdict.
                "--out",
                str(ch / f".resp_{rid}.{os.getpid()}.tmp"),
                # Progress is explicitly NOT a verdict. It is safe to publish while the final response
                # stays hidden because it contains only public capsule names and completed statuses.
                "--progress-out",
                str(ch / f"progress_{rid}.json"),
            ]
            if suite != "core":
                argv2 += ["--suite", suite]
            if sim:
                argv2 += ["--sim", sim]
            if str(r.get("tiers") or "").strip():
                argv2 += ["--tiers", str(r["tiers"]).strip()]
            # forward the shape-coverage request through the sandbox shim (the agent cannot reach the
            # oracle itself, so a flag it sets here is the only way the check runs at all)
            if r.get("shape_coverage"):
                argv2.append("--shape-coverage")
            # Every probe the real self-check offers, by the one table that names them, so a probe added
            # there and forgotten here fails a test instead of being silently unreachable.
            for key, flag in FORWARDED_PROBES.items():
                if key != "shape_coverage" and r.get(key):
                    argv2.append(flag)
            if r.get("caller_layout"):
                argv2.extend(("--caller-layout", str(r["caller_layout"])))
            if a.one_request is not None:
                argv2 += ["--artifact-root", str(ws / "selfcheck_out" / "requests" / rid)]
            snapshot_root = None
            try:
                # Run the REAL self-check OUTSIDE the sandbox (oracle available); cwd=ws so the agent's own
                # artifacts land in <ws>/selfcheck_out/ (visible to the agent through the bind mount).
                # Popen + poll rather than subprocess.run so STOP / parent-death is honored WHILE a long check
                # is in flight; run() blocked until completion and left RTL sims running past the round.
                deadline = request_deadline = _child_deadline_monotonic(r)
                gsim_budgeted = joint_control.selected() is not None and sim == "gsim"
                if gsim_budgeted:
                    deadline = min(deadline, time.monotonic() + to)
                # A long measurement outlives the broker's stop in its own handler only (never in the
                # single-process broker, whose exit the lifecycle waits for); it still ends by the
                # campaign's end.
                long_measurement = bool(measurement and measurement.get("long")) and a.one_request is not None
                if long_measurement:
                    deadline = min(deadline, time.monotonic() + max(0.0, joint_control.remaining() or 0.0))
                # Spool the child's streams to FILES, not pipes: a poll loop that never drains a PIPE
                # deadlocks as soon as the child fills the buffer, and a full-corpus self-check is chatty.
                from merlin.development.backend.feedback.selfcheck.results import operator_log_dir

                # Raw child streams can name host paths: keep them operator-side when an evidence
                # root is selected; the agent receives only the scrubbed response.
                log_dir = operator_log_dir()
                from merlin.development.backend.feedback.disclosure import restricted

                if restricted() and log_dir is None:
                    raise ValueError("restricted feedback requires operator-private evidence staging")
                log_dir = log_dir or ch
                log_out, log_err = log_dir / f".out_{rid}", log_dir / f".err_{rid}"
                staged_resp = ch / f".resp_{rid}.{os.getpid()}.tmp"
                staged_resp.unlink(missing_ok=True)
                aborted = False
                # Every check grades a private request snapshot. The author can keep
                # editing, and the response stays bound to the selected digest.
                snapshot_root = _working_snapshot(ws, graded_digest, label=label)
                r = tuning.apply_request(snapshot_root / "submission", r, digest=_submission_digest)
                from merlin.development.backend.feedback.guidance.construction_plan import (
                    admit as admit_construction_plan,
                )

                construction_admission = admit_construction_plan(
                    snapshot_root / "submission",
                    r,
                    request_id=rid,
                    channel="selfcheck",
                    inspection=any(
                        r.get(k)
                        for k in (
                            "compile_only",
                            "caller_layout",
                            "candidate_diagnostics",
                            "shape_coverage",
                            "model_layers",
                            "offload_census",
                        )
                    ),
                )
                if construction_admission is not None:
                    _atomic_write(ch / f"construction_plan_{rid}.json", json.dumps(construction_admission))
                argv2[argv2.index("--submission") + 1] = str(snapshot_root / "submission")
                from merlin.development.backend.campaign.calibration.activation import request_environment

                worker_env = request_environment(
                    os.environ, channel="selfcheck", request_id=rid, request_bytes=request_bytes
                )
                worker_env[sources.LABEL_ENV] = label
                from merlin.development.backend.campaign import gsim_budget as gsim_queue

                # An author request waits for a busy gSIM correctness slot; operator work does not.
                worker_env[gsim_queue.CLASS_ENV] = gsim_queue.AUTHOR
                gsim_queue.name_requester(worker_env, r)  # waiting is interleaved per caller thread
                if r.get("origin") in OPERATOR_ORIGINS:
                    # Operator background measurement: every author request and grade leases CPUs first.
                    from merlin.common.process.cpu_pool import AUXILIARY, PRIORITY_ENV

                    worker_env[PRIORITY_ENV] = str(AUXILIARY)
                    worker_env[gsim_queue.CLASS_ENV] = gsim_queue.BACKGROUND
                from merlin.development.backend.campaign.joint_gsim_measurement import ENV as MEASUREMENT_ENV

                worker_env.pop(MEASUREMENT_ENV, None)
                if measurement is not None:
                    worker_env[MEASUREMENT_ENV] = rid  # the child re-checks this id in the run-side ledger
                # Only submission/ may feed promotion: a variant's verdict never reaches the cert ledger.
                if (
                    suite == "core"
                    and not variant
                    and not advisory
                    and not r.get("caller_layout")
                    and not any(r.get(key) for key in FORWARDED_PROBES)
                ):
                    worker_env.update(
                        MERLIN_SELFCHECK_PROMOTION_WS=str(ws),
                        MERLIN_SELFCHECK_PROMOTION_ROOT=str(promotion_capsules_root or capsules_root),
                        MERLIN_SELFCHECK_PROMOTION_SHA256=graded_digest,
                    )
                with log_out.open("w") as fo, log_err.open("w") as fe, contextlib.ExitStack() as priority:
                    from merlin.common.process.cpu_pool import worker_pool_command
                    from merlin.experiments.frozen_python import inherited_python_command

                    proc = subprocess.Popen(
                        worker_pool_command(inherited_python_command(argv2), worker_env),
                        cwd=str(ws),
                        stdout=fo,
                        stderr=fe,
                        text=True,
                        env=worker_env,
                    )
                    from merlin.development.backend.feedback.grading.background_admission import (
                        active_full_selfcheck,
                    )

                    # The foreground marker describes submission/ work; a variant publishes none.
                    foreground = contextlib.nullcontext()
                    if suite == "core" and not variant and not advisory:
                        foreground = active_full_selfcheck(
                            ws=ws,
                            process=proc,
                            request=r,
                            source_sha256=graded_digest,
                            public=capsules_root,
                            context=context,
                            contract=contract,
                            bundle_manifest=a.compiler_bundle_manifest,
                            sim=sim,
                            dispatcher_pid=orig_ppid if a.one_request is not None else None,
                        )
                    priority.enter_context(foreground)
                    while proc.poll() is None:
                        if gsim_budgeted and gsim_queue.waiting(ch / f"progress_{rid}.json") is not None:
                            # The job budget starts when a gSIM slot is admitted, not while it waits for
                            # one; the client's own deadline and the campaign clock still bound the wait.
                            deadline = max(deadline, min(request_deadline, time.monotonic() + to))
                        why = _abort_reason(
                            ch,
                            orig_ppid,
                            req,
                            deadline,
                            request_stop=a.request_stop,
                            survive_stop=long_measurement,
                        )
                        if why:
                            if a.one_request is not None:
                                _stop_owned_worker(proc)
                            else:
                                proc.terminate()
                                try:
                                    proc.wait(timeout=30)
                                except subprocess.TimeoutExpired:
                                    proc.kill()  # a wedged RTL sim does not get to outlive us
                                    proc.wait(timeout=30)
                            staged_resp.unlink(missing_ok=True)
                            from merlin.development.backend.feedback.selfcheck.results import partial_results

                            doc = _decorate_response(
                                json.dumps(
                                    {
                                        "error": f"broker: self-check aborted ({why})",
                                        "all_pass": False,
                                        **partial_results(ch / f"progress_{rid}.json"),
                                    }
                                ),
                                r,
                                rid,
                                graded_digest,
                                started_ns=started_ns,
                                completed_ns=time.time_ns(),
                            )
                            _atomic_write(resp, json.dumps(doc))
                            _atomic_write(ch / f"done_{rid}", "err")
                            aborted = True
                            break
                        time.sleep(a.poll)
                if not aborted:
                    completed_ns = time.time_ns()
                    if staged_resp.exists():
                        raw = staged_resp.read_text(encoding="utf-8", errors="replace")
                        staged_resp.unlink(missing_ok=True)
                        doc = _decorate_response(
                            raw, r, rid, graded_digest, started_ns=started_ns, completed_ns=completed_ns
                        )
                        from merlin.development.backend.feedback.performance.idle_tuning import (
                            attach as attach_tuning,
                        )
                        from merlin.development.backend.feedback.performance.knob_lint import (
                            attach as attach_knob_candidates,
                        )

                        doc = attach_knob_candidates(attach_tuning(doc, ws), ws, submission=sources.SUBMISSION)
                        from merlin.development.backend.feedback.selfcheck.results import compact_document

                        doc = compact_document(doc, workspace=ws, namespace="selfcheck")
                        _atomic_write(resp, json.dumps(doc))
                    else:
                        out = log_out.read_text()[-20000:] if log_out.exists() else ""
                        err = _log_tail(log_err, 300)
                        from merlin.development.backend.feedback.selfcheck.results import partial_results

                        doc = _decorate_response(
                            out
                            or json.dumps(
                                {"error": err, "all_pass": False, **partial_results(ch / f"progress_{rid}.json")}
                            ),
                            r,
                            rid,
                            graded_digest,
                            started_ns=started_ns,
                            completed_ns=completed_ns,
                        )
                        _atomic_write(resp, json.dumps(doc))
            except Exception as e:
                doc = _decorate_response(
                    json.dumps({"error": f"broker: {type(e).__name__}: {str(e)[:200]}", "all_pass": False}),
                    r,
                    rid,
                    graded_digest if variant else _submission_digest(ws / "submission"),
                    started_ns=started_ns,
                    completed_ns=time.time_ns(),
                )
                _atomic_write(resp, json.dumps(doc))
            # Capture the verified source BEFORE done releases the author to edit it.
            # The optional queueing below must never hash the newly edited live workspace.
            promotion_source = None
            try:
                if resp.exists() and not variant and not advisory and suite == "core":
                    from merlin.development.backend.feedback.selfcheck.results import hydrate

                    passed = hydrate(json.loads(resp.read_text()), workspace=ws)
                    if not passed.get("error") and any(
                        row.get("pass") is True for row in passed.get("per_capsule", [])
                    ):
                        promotion_source = _freeze_passed_source(ws, graded_digest, source_ws=snapshot_root)
            except Exception as exc:
                print(f"[promote] source freeze refused: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if snapshot_root is not None:
                shutil.rmtree(snapshot_root, ignore_errors=True)
            # COMPLETE THE CALLER'S REQUEST before optional promotion.  The shim deliberately waits for both
            # response and done; publishing the verdict and then running promotion first made a slow or wedged
            # promotion indistinguishable from a lost response even though the verdict was already durable.
            if not (ch / f"done_{rid}").exists():
                _atomic_write(ch / f"done_{rid}", "ok")
            health["processed"] = int(health.get("processed", 0)) + 1
            health["last_completed_at_unix_ns"] = time.time_ns()
            _atomic_write(health_path, json.dumps(health, sort_keys=True))

            # The caller may stop the run immediately after observing ``done``. Re-check before starting
            # optional promotion: promotion is not part of the request and must not delay broker shutdown.
            if _should_stop(ch, orig_ppid):
                continue
            if inspection or variant or advisory or suite != "core":
                # Static observations cannot trigger a numerical/RTL promotion or
                # pay for a simulator after a deliberately no-execution request.
                # A variant is never submission/: its verdict promotes nothing.
                continue

            # PROMOTE off the SYNCHRONOUS path too. Promotion was first hooked into the async oracle alone,
            # and a live run then showed the agent using THIS path seven times to the async path's two -- so
            # eight verdicts completed and promotion fired zero times. Wherever a verdict is produced,
            # promotion is considered.
            try:
                import json as _j

                from merlin.development.backend.feedback.certification.promotion import cert_already_executed
                from merlin.development.backend.feedback.certification.promotion import promote as _promote
                from merlin.development.backend.feedback.certification.promotion import (
                    resolve_tiers as _resolve,
                )

                _loop, _cert, _cover = _resolve(ws, context=context, capsules_root=promotion_capsules_root)
                if _loop and _cert and resp.exists():
                    _v = _j.loads(resp.read_text())
                    if cert_already_executed(_v, _cert):
                        print(
                            f"[promote] not enqueued: this check already ran {_cert} on its own selection",
                            file=sys.stderr,
                            flush=True,
                        )
                        if isinstance(_v, dict) and not _v.get("error") and promotion_source is not None:
                            from merlin.development.backend.feedback.selfcheck.results import import_executed

                            import_executed(
                                ws,
                                _v,
                                loop_tier=_loop,
                                cert_tier=_cert,
                                source_ws=promotion_source,
                                context=context,
                                log=sys.stderr,
                            )
                    elif isinstance(_v, dict) and not _v.get("error") and promotion_source is not None:
                        _promote(
                            ws,
                            ch,
                            _v,
                            _loop,
                            _cert,
                            _cover,
                            sys.stderr,
                            context=context,
                            capsules_root=promotion_capsules_root,
                            source_ws=promotion_source,
                        )
            except Exception as _pe:  # noqa: BLE001 -- promotion is an optimisation, never a gate
                print(f"[promote] skipped: {type(_pe).__name__}: {_pe}", file=sys.stderr, flush=True)
        except OSError as error:  # A full disk must not end the broker; other errors still do.
            _report_loop_error("request loop", error)
            _settle_failed_request(ch, served, seen, error)
            time.sleep(LOOP_ERROR_BACKOFF_S)


if __name__ == "__main__":
    from merlin.development.backend.telemetry.evidence import install

    install()  # A full log volume drops log lines, never the broker.
    from merlin.common.process.cpu_pool import AUTHOR, cpu_priority

    with cpu_priority(AUTHOR, export=True):  # Author-requested self-checks lease first.
        raise SystemExit(main())
