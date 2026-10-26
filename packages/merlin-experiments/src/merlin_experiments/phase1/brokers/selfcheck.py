"""Driver-side self-check BROKER — lets a bwrap-sandboxed agent get a REDACTED self-check verdict without
the oracle ever entering its sandbox.

The redacted self-check (agent_selfcheck.py) internally needs the oracle (merlin.runtime.reference) to
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

from merlin_experiments.phase1.brokers import selfcheck_sources as sources
from merlin_experiments.phase1.brokers import selfcheck_tuning as tuning
from merlin_experiments.phase1.context import add_context_arguments, context_argv, resolve_context
from merlin_experiments.phase1.feedback.dispatch import REQUIRED_SIM, grade_selection_for
from merlin_experiments.phase1.feedback.idle_performance import OPERATOR_ORIGINS
from merlin_experiments.phase1.feedback.idle_performance import ORIGIN as IDLE_ORIGIN  # noqa: F401 -- stable name
from merlin_experiments.phase1.feedback.idle_performance import REQUEST_PREFIX as IDLE_REQUEST_PREFIX
from merlin_experiments.phase1.feedback.selfcheck import _public_capsules, worker_command

from ..compiler_library import selected_child

# Lane sizing and worker reaping live in their own module; every name stays importable from here.
from .selfcheck_lanes import (  # noqa: F401 -- re-exported for callers and tests
    CAPSULES_PER_COMPUTE_CPU,
    _diagnostic_request,
    _dispatch_capacity,
    _feedback_scope_error,
    _lane_admits,
    _ordinary_capacity,
    _ordinary_capsule_limit,
    _process_record,
    _public_corpus_size,
    _stop_owned_worker,
)


def _freeze_passed_source(ws: Path, expected_digest: str, *, source_ws: Path | None = None):
    """Bind asynchronous promotion to the source just graded, before releasing the author."""
    from merlin_experiments.phase1.feedback import snapshots

    token, error = snapshots.create_promotion_snapshot(
        ws,
        source_ws or ws,
        expected_digest,
        submission_digest=lambda root: _submission_digest(Path(root) / "submission"),
    )
    if error:
        raise RuntimeError(error)
    source = snapshots.promotion_snapshot_path(ws, token)
    if source is None or _submission_digest(source / "submission") != expected_digest:
        raise RuntimeError("passed selfcheck source snapshot differs from graded identity")
    return source


def _working_snapshot(ws: Path, expected_digest: str, *, label: str = sources.SUBMISSION) -> Path:
    """Copy one request's source under the task temp root before launching the grader.

    The selected tree (``submission/`` or ``variants/<name>/``) is read through a directory opened
    without following symlinks and always lands at ``<snapshot>/submission``, so the grader sees the
    same package layout whichever tree was asked for.
    """
    tmpdir = os.environ.get("TMPDIR")
    with sources.pinned_source(ws, label) as source:
        if any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("self-check source contains a symlink")
        if not tmpdir or not Path(tmpdir).is_absolute():
            raise ValueError("self-check requires a task-owned TMPDIR")
        root = Path(tempfile.mkdtemp(prefix="merlin_selfcheck_", dir=tmpdir))
        try:
            shutil.copytree(source, root / "submission", ignore=shutil.ignore_patterns(*_DIGEST_IGNORES))
            if _submission_digest(root / "submission") != expected_digest:
                raise ValueError(f"{label} changed while taking self-check snapshot")
            return root
        except Exception:
            shutil.rmtree(root)
            raise


#: request key -> the real self-check's flag, for every probe that replaces the capsule suite. The
#: shim writes the keys, this broker turns them back into flags; a probe missing here is one a
#: sandboxed agent cannot reach, however well it works outside the box.
FORWARDED_PROBES = {
    "shape_coverage": "--shape-coverage",
    "offload_census": "--offload-census",
    "model_layers": "--model-layers",
    "candidate_diagnostics": "--candidate-diagnostics",
    "compile_only": "--compile-only",
    "scale_probe": "--scale-probe",
    "dynamic_census": "--dynamic-census",
}
PROTOCOL_VERSION = 3
COMMON_ACTIONS = frozenset({"common-transform", "common-check"})
_DIGEST_IGNORES = frozenset({"build", "__pycache__", ".git", "selfcheck_out"})


def _submission_digest(root: Path) -> str:
    """Stable digest of authored submission bytes, excluding disposable build state."""
    from merlin.common.cpu_pool import shared_compute_pool

    with shared_compute_pool():
        root = Path(root)
        digest = hashlib.sha256()
        if not root.is_dir():
            return digest.hexdigest()
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if any(part in _DIGEST_IGNORES for part in rel.parts):
                continue
            if path.is_symlink():
                digest.update(rel.as_posix().encode() + b"\0SYMLINK\0" + os.readlink(path).encode())
            elif path.is_file():
                digest.update(rel.as_posix().encode() + b"\0")
                with path.open("rb") as f:
                    while chunk := f.read(1024 * 1024):
                        digest.update(chunk)
        return digest.hexdigest()


#: The agent's own work tree; paths under it stay readable in agent-visible error text.
_AGENT_ROOTS: tuple[str, ...] = ()


def agent_visible_text(text: str) -> str:
    """``text`` with host paths outside the agent's work tree replaced; full detail stays in the logs."""
    from merlin.common.path_scrub import scrub_host_paths

    return scrub_host_paths(text, keep=_AGENT_ROOTS)


def _log_tail(path: Path, limit: int) -> str:
    """The last ``limit`` characters of a log, scrubbed whole lines first so no path is cut mid-way."""
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    window = text[-max(limit * 8, 4000) :]
    if len(window) < len(text):
        window = window.partition("\n")[2]  # never start inside a path or other token
    return agent_visible_text(window)[-limit:]


def _decorate_response(
    raw: str, request: dict, rid: str, submission_sha256: str, *, started_ns: int, completed_ns: int
) -> dict:
    """Return a JSON verdict explicitly bound to the request and graded submission bytes."""
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise TypeError("self-check response is not an object")
    except Exception as exc:  # noqa: BLE001 -- malformed grader output becomes an explicit failure
        body = {
            "error": f"broker: unreadable self-check response ({type(exc).__name__})",
            "all_pass": False,
            "raw_tail": raw[-1000:],
        }
    for key in ("error", "raw_tail"):
        if isinstance(body.get(key), str):
            body[key] = agent_visible_text(body[key])
    requested = request.get("submission_sha256")
    requested_sim = request.get("sim")
    if isinstance(requested_sim, str) and requested_sim and requested_sim != body.get("sim", requested_sim):
        # Said, not hidden: the pinned engine's name is served by the grade's own selection.
        body["requested_sim"] = requested_sim
    body.update(
        {
            "selfcheck_protocol": PROTOCOL_VERSION,
            "selfcheck_request_id": rid,
            "submission_sha256": submission_sha256,
            "requested_submission_sha256": requested,
            "requested_at_unix_ns": request.get("requested_at_unix_ns"),
            "grade_started_at_unix_ns": started_ns,
            "graded_at_unix_ns": completed_ns,
        }
    )
    if requested and submission_sha256 and requested != submission_sha256:
        # SAID, NOT HIDDEN. The verdict describes bytes NEWER than the ones asked about, which is more
        # useful than the ones asked about -- but only while it is stated, or the agent would read a
        # verdict as being about code it has since replaced.
        body["graded_newer_bytes"] = True
        body["graded_newer_note"] = sources.newer_bytes_note(
            request.get("submission", sources.SUBMISSION) if isinstance(request, dict) else sources.SUBMISSION
        )
    # Which tree this verdict is about, on every response; a variant's also says it is not official.
    body.update(sources.response_fields(request, submission_sha256))
    body.update(tuning.response_fields(request))  # knob overrides grade a private copy, never submission/
    return body


def _rtl_engines() -> tuple[str, ...]:
    try:
        from merlin.targetgen.rtl_engine_policy import ENGINE_PRIORITY

        return tuple(ENGINE_PRIORITY)
    except Exception:  # noqa: BLE001 -- closed historical set, never widen on import failure
        return ("vcs", "gsim", "verilator")


def _required_rtl_engine() -> str | None:
    return os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip() or None


def _allowed_sims() -> tuple[str, ...]:
    """Spike plus either the one pinned RTL engine or the closed registered RTL set, and the grade's
    own selection (``REQUIRED_SIM``), which honours the pin itself."""
    required = _required_rtl_engine()
    if required is not None:
        return ("spike", required, REQUIRED_SIM) if required in _rtl_engines() else ("spike", REQUIRED_SIM)
    return ("spike",) + _rtl_engines() + (REQUIRED_SIM,)


def _default_sim() -> str:
    """An unnamed engine means every tier the official grade selects, never one engine.

    This returned the pinned RTL engine, so an unflagged request ran ``--sim gsim``: Spike plus a gSIM
    adapter built WITHOUT the grade's selection record. Its verdict could differ from the grade's and
    its exact-byte simulator certificates were keyed to a different instrument, so neither side could
    carry the other's gSIM result. The grade's own selection honours the pin, so nothing is lost.
    """
    return REQUIRED_SIM


def _sim_policy_error(sim: str) -> str | None:
    allowed = _allowed_sims()
    if sim in allowed:
        return None
    required = _required_rtl_engine()
    if required is not None:
        return f"--sim {sim!r} conflicts with MERLIN_REQUIRED_RTL_ENGINE={required!r}; allowed: {list(allowed)!r}"
    return f"--sim {sim!r} is not accepted; allowed: {list(allowed)!r}"


def _atomic_write(path: Path, text: str) -> None:
    """Publish a response/marker with no observable partial-file state."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _request_deadline(req: Path) -> float:
    """Client deadline for protocol-v2 and legacy requests.

    Legacy shims did not put their deadline in the payload, but their wait contract has always been
    ``request mtime + timeout + 240 seconds``.  Recovering it here preserves those clients while making
    their abandoned files finite work rather than a permanent restart backlog.
    """
    try:
        body = json.loads(req.read_text(encoding="utf-8"))
        explicit = body.get("deadline_unix_ns")
        if explicit is not None:
            return int(explicit) / 1_000_000_000
        timeout = max(0, int(body.get("timeout", 1800)))
    except Exception:
        timeout = 1800
    return req.stat().st_mtime + timeout + 240


def _child_deadline_monotonic(
    request: dict, *, now_wall: float | None = None, now_monotonic: float | None = None
) -> float:
    """Translate the client's durable wall-clock deadline onto this process's monotonic clock.

    The request deadline accounts for the whole suite.  Reconstructing a shorter
    ``per_capsule_timeout + 180`` deadline here used to kill full-corpus checks while their later worker
    waves were still healthy.
    """
    wall = time.time() if now_wall is None else now_wall
    monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    explicit = request.get("deadline_unix_ns")
    if explicit is not None:
        deadline = monotonic + max(0.0, int(explicit) / 1_000_000_000 - wall)
    else:
        deadline = monotonic + max(0, int(request.get("timeout", 1800))) + 240
    # A conservative suite-sized client budget cannot outlive its selected
    # campaign. This also bounds setup before the evaluator process exists.
    from .. import joint_control

    campaign = joint_control.selected()
    # Tool readiness exercises the common transformation broker before author
    # launch. Its client deadline still applies, but no campaign clock exists
    # yet. This exception cannot admit capsule evaluation before launch.
    readiness = (
        campaign is not None
        and "started_at_epoch" in campaign
        and campaign["started_at_epoch"] is None
        and request.get("action") in COMMON_ACTIONS
    )
    if campaign is not None and not readiness:
        deadline = min(deadline, monotonic + joint_control.remaining(now=wall))
    return deadline


def _pid_namespace() -> str:
    """The broker's Linux PID-namespace identity, comparable across a shared file channel."""
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return ""


def _requester_alive(request: dict) -> bool | None:
    """Whether the process waiting for this request is still the same process.

    ``None`` means a legacy request supplied no identity, so its historical deadline contract remains
    authoritative.  A protocol-v3 request with identity is a lease: once that exact PID incarnation is
    gone, replay can help nobody and may block a new client behind hours of full-suite work.
    """
    identity = request.get("requester")
    if not isinstance(identity, dict):
        # Protocol-v3 shims before the explicit start-tick lease still embedded their PID as the first
        # request-id field.  Absence is conclusive; presence is not (the PID may have been reused), so
        # retain the deadline contract in the ambiguous case.  This drains old dead-client requests on
        # the first upgraded broker restart without weakening compatibility for a possibly live client.
        if int(request.get("protocol", 1)) >= 3:
            prefix = str(request.get("request_id") or "").partition("_")[0]
            if prefix.isdecimal() and not Path(f"/proc/{prefix}").exists():
                return False
        return None
    requester_namespace = identity.get("pid_namespace")
    # A numeric PID and /proc start tick are only comparable within one PID namespace.  Returning False
    # for a sandbox-local PID looked up in the broker's host /proc discarded every live self-check.  A
    # foreign namespace (and protocol-v3 identities written before this field existed) retains the
    # request's durable deadline lease; only a same-namespace identity may be declared dead here.
    if not requester_namespace or requester_namespace != _pid_namespace():
        return None
    try:
        pid = int(identity["pid"])
        expected_start = int(identity["start_ticks"])
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        state, actual_start = tail[0], int(tail[19])
    except FileNotFoundError:
        return False
    except (OSError, KeyError, TypeError, ValueError, IndexError):
        return False
    return state != "Z" and actual_start == expected_start


def _detached(request: Path) -> bool:
    """Whether the request's client returned on purpose and left it to its deadline lease.

    A client that submits with a short wait budget and collects later with ``--attach`` exits while
    its request is still queued or running. Judged by that process alone, a same-namespace request
    would be dropped or aborted as abandoned the moment the client returned. The marker is written
    before the client exits, so its absence still means a client that died mid-wait.
    """
    rid = request.name[len("req_") : -len(".json")]
    return bool(rid) and (request.parent / f"detached_{rid}").exists()


def _pending_requests(ch: Path, seen: set[str], *, now: float | None = None) -> list[Path]:
    """Outstanding, live requests in arrival order.

    ``seen`` is process-local, while completion markers and deadlines survive a broker restart.  All
    three are needed: relying only on ``seen`` caused a restarted broker to rerun every historical
    full-corpus grade before it reached the request whose client was actually waiting.
    """
    now = time.time() if now is None else now
    pending: list[tuple[float, Path]] = []
    for req in ch.glob("req_*.json"):
        if req.name in seen:
            continue
        rid = req.stem[len("req_") :]
        if (ch / f"done_{rid}").exists() and (ch / f"resp_{rid}.json").exists():
            continue
        from ..joint_gsim_measurement import REQUEST_PREFIX as MEASUREMENT_PREFIX
        from ..joint_gsim_measurement import long_request

        if rid.startswith(MEASUREMENT_PREFIX) and long_request(rid, running=True):
            continue  # a long measurement outlived an earlier broker and is still being served
        try:
            body = json.loads(req.read_text(encoding="utf-8"))
            if _requester_alive(body) is False and not _detached(req):
                continue
            if now > _request_deadline(req):
                continue
            arrival = req.stat().st_mtime
        except FileNotFoundError:
            continue
        pending.append((arrival, req))
    return [request for _, request in sorted(pending, key=lambda row: row[0])]


def _abort_reason(
    ch: Path,
    orig_ppid: int,
    request: Path,
    deadline: float,
    *,
    request_stop: Path | None = None,
    survive_stop: bool = False,
) -> str | None:
    """Why the self-check in flight must stop now, or ``None`` to let it run.

    A CANCEL CANCELS. A client cancels by renaming its request away; the broker had already claimed
    it and ran to completion regardless, holding the one slot. Measured on one run: 4,609 seconds of
    oracle time spent on a request the agent stopped reading 37 minutes earlier, while four later
    requests waited behind it and were never served.

    ``survive_stop`` is set only for a long gSIM measurement the idle lane authorized
    (:func:`..joint_gsim_measurement.long_request`): the broker's STOP, its parent's exit and the
    dispatcher's stop then do not end it -- it runs to its own deadline, which is bounded by the
    campaign's end -- unless a process is waiting for a shared compute slot while none is free (final
    grading must never wait for a measurement's slot). Cancellation, an unreadable request and the
    deadline still stop it.
    """
    stop = _should_stop(ch, orig_ppid)
    if not stop and request_stop is not None and request_stop.exists():
        stop = "dispatcher stopped"
    if stop:
        if not survive_stop:
            return stop
        from ..joint_gsim_measurement import shared_pool_contended

        if shared_pool_contended():
            return f"{stop}; a shared compute slot is wanted while none is free"
    if not request.exists():
        return "the requester cancelled it"
    try:
        if _requester_alive(json.loads(request.read_bytes())) is False and not _detached(request):
            return "requester exited"
    except (OSError, ValueError, AttributeError):
        return "request changed or became unreadable"
    if time.monotonic() > deadline:
        return "timed out"
    return None


def _should_stop(ch: Path, orig_ppid: int) -> str | None:
    """Why the broker must exit now, or None to keep serving.

    Checked BEFORE each request and while one is in flight. Two conditions, both observed to matter:
    the driver's STOP sentinel (previously only tested between batches, so a ~10-minute self-check kept
    running after the round ended), and the death of the QA loop that started us — an orphaned broker
    was seen reparented to init, still spawning self-checks and RTL sims to answer requests whose
    channel no one was reading any more.
    """
    if (ch / "STOP").exists():
        return "STOP"
    if os.getppid() != orig_ppid:
        return "parent exited"
    return None


#: Pause after a loop pass fails outside any request's own handling (e.g. a full disk), so the
#: broker keeps serving once the condition clears instead of exiting with its queue unanswered.
LOOP_ERROR_BACKOFF_S = 15


def _report_loop_error(where: str, error: BaseException) -> None:
    try:
        print(
            f"[broker] {where} pass failed, continuing: {type(error).__name__}: {str(error)[:200]}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:  # noqa: BLE001 -- the log may be on the same full disk
        pass


def _settle_failed_request(ch: Path, request: Path | None, seen: set[str], error: BaseException) -> None:
    """Answer the request a failed pass was serving, or leave it queued when no answer can be written.

    A claimed request whose response never lands would otherwise wait for its deadline. If even the
    error response cannot be written -- the usual case on a full disk -- the request goes back to the
    queue, and the next pass after the backoff serves it again.
    """
    if request is None:
        return
    rid = request.stem[len("req_") :]
    if (ch / f"done_{rid}").exists() and (ch / f"resp_{rid}.json").exists():
        return
    try:
        _atomic_write(
            ch / f"resp_{rid}.json",
            json.dumps(
                {
                    "error": f"broker: {type(error).__name__}: {str(error)[:200]}",
                    "all_pass": False,
                    "selfcheck_protocol": PROTOCOL_VERSION,
                    "selfcheck_request_id": rid,
                }
            ),
        )
        _atomic_write(ch / f"done_{rid}", "err")
    except OSError:
        seen.discard(request.name)


def _handler_command(a, context, capsules_root, contract, promotion_root, request, stop, *, request_bytes=None):
    """Original command and input paths; request data cannot choose a handler."""
    command = [sys.executable, "-m", "merlin_experiments.phase1.brokers.selfcheck"]
    command += context_argv(context) + [
        "--ws",
        str(Path(a.ws).resolve()),
        "--poll",
        str(a.poll),
        "--capsules-root",
        str(capsules_root),
        "--one-request",
        request.name,
        "--request-stop",
        str(stop),
        "--request-sha256",
        hashlib.sha256(request.read_bytes() if request_bytes is None else request_bytes).hexdigest(),
    ]
    for flag, value in (
        ("--contract", contract),
        ("--policy-capsules-root", promotion_root),
        ("--rtl-facts", a.rtl_facts),
        ("--compiler-bundle-manifest", a.compiler_bundle_manifest),
    ):
        if value is not None:
            command += [flag, str(value)]
    from ..compiler_library import child_arguments

    return [*command, *child_arguments()]


def _dispatch_requests(a, context, capsules_root, contract, promotion_root):
    """Two independent processes: one numerical lane and one diagnostic lane.

    Each handler reconstructs its own original service and retains the existing
    source, deadline, cancellation and publication checks. The parent never holds
    a compute lease across a handler wait or shares execution scopes with it.
    """
    from merlin.common.cpu_pool import worker_pool_command
    from merlin_experiments.frozen_python import inherited_python_command

    from ..joint_gsim_measurement import long_request

    ws = Path(a.ws).resolve()
    ch, orig_ppid = ws / ".qa_channel", os.getppid()
    seen, active = set(), {}
    health_path = ch / "broker_health.json"
    try:
        health = json.loads(health_path.read_bytes())
    except (OSError, ValueError):
        health = {}
    ordinary_capacity = _ordinary_capacity()
    # Variants share the ordinary lane; their own bound keeps the rest of it for submission/ checks.
    variant_capacity = sources.variant_capacity(ordinary_capacity)
    health.update(
        protocol=PROTOCOL_VERSION,
        dispatch_capacity=ordinary_capacity + 1,
        ordinary_capacity=ordinary_capacity,
        variant_capacity=variant_capacity,
        broker_starts=int(health.get("broker_starts", 0)) + 1,
        processed=int(health.get("processed", 0)),
        max_queue_depth=int(health.get("max_queue_depth", 0)),
        last_started_at_unix_ns=time.time_ns(),
    )
    try:
        while not _should_stop(ch, orig_ppid):
            try:
                for name, job in tuple(active.items()):
                    reason = None
                    if job["process"].poll() is None:
                        reason = _abort_reason(
                            ch,
                            orig_ppid,
                            job["request"],
                            job["deadline"],
                            request_stop=job["stop"],
                            survive_stop=job["long"],
                        )
                        if reason is None:
                            continue
                        # The handler may still be reconstructing the selected
                        # library/source, before it reaches its own child poll loop.
                        # Reap the original tree before publishing/refilling its lane.
                        _stop_owned_worker(job["process"])
                    request, rid = job["request"], name[len("req_") : -len(".json")]
                    if not ((ch / f"done_{rid}").exists() and (ch / f"resp_{rid}.json").exists()):
                        _atomic_write(
                            ch / f"resp_{rid}.json",
                            json.dumps(
                                {
                                    "all_pass": False,
                                    "error": (
                                        f"broker: self-check aborted ({reason}) during handler preparation/evaluation"
                                        if reason
                                        else "broker: request handler ended without durable completion"
                                    ),
                                    "request_lifecycle": {
                                        "status": "aborted" if reason else "worker_exited",
                                        "scope": "handler preparation and evaluation; no numerical verdict",
                                        "started_at_unix_ns": job["started_ns"],
                                        "completed_at_unix_ns": time.time_ns(),
                                        "deadline_monotonic": job["deadline"],
                                    },
                                    "selfcheck_protocol": PROTOCOL_VERSION,
                                    "selfcheck_request_id": rid,
                                    "submission_sha256": None,
                                    **job["source_fields"],
                                }
                            ),
                        )
                        _atomic_write(ch / f"done_{rid}", "err")
                    job["stop"].unlink(missing_ok=True)
                    del active[name]
                    health["processed"] += 1
                pending = _pending_requests(ch, seen)
                # Stable FIFO within each lane; operator background requests after every author request.
                pending.sort(
                    key=lambda request: (
                        _diagnostic_request(request),
                        request.name.startswith("req_" + IDLE_REQUEST_PREFIX),
                    )
                )
                health["max_queue_depth"] = max(health["max_queue_depth"], len(pending))
                for request in pending:
                    try:
                        request_bytes = request.read_bytes()
                    except FileNotFoundError:
                        continue
                    diagnostic = _diagnostic_request(request, request_bytes=request_bytes)
                    variant = sources.is_variant_request(request_bytes)
                    if not _lane_admits(
                        _lane_holders(ch, active),
                        diagnostic=diagnostic,
                        variant=variant,
                        ordinary_capacity=ordinary_capacity,
                        variant_capacity=variant_capacity,
                    ):
                        continue
                    stop = ch / f".handler-stop_{secrets.token_hex(12)}"
                    try:
                        command = _handler_command(
                            a,
                            context,
                            capsules_root,
                            contract,
                            promotion_root,
                            request,
                            stop,
                            request_bytes=request_bytes,
                        )
                        from ..joint_model_activation import request_environment
                        handler_env = request_environment(os.environ, channel="selfcheck",
                            request_id=request.stem.removeprefix("req_"), request_bytes=request_bytes)
                    except FileNotFoundError:
                        continue  # Ordinary cancellation must not stop another lane.
                    except (ValueError, KeyError, TypeError) as error:
                        rid = request.stem.removeprefix("req_")
                        _atomic_write(ch / f"resp_{rid}.json", json.dumps({
                            "error": "request model admission refused: " + type(error).__name__,
                            "all_pass": False, "numeric_assessment": "not_run"}))
                        _atomic_write(ch / f"done_{rid}", "err")
                        seen.add(request.name)
                        continue
                    process = subprocess.Popen(
                        worker_pool_command(inherited_python_command(command), handler_env), cwd=ws, env=handler_env
                    )
                    stat = _process_record(process.pid)
                    seen.add(request.name)
                    try:
                        body = json.loads(request_bytes)
                    except ValueError:
                        body = {}
                    # Include queue time even for historical requests without an
                    # explicit deadline; never renew their budget on admission.
                    deadline_body = dict(body) if isinstance(body, dict) else {}
                    deadline_body.setdefault("deadline_unix_ns", int(_request_deadline(request) * 1e9))
                    active[request.name] = {
                        "deadline": _child_deadline_monotonic(deadline_body),
                        "started_ns": time.time_ns(),
                        "process": process,
                        "request": request,
                        "stop": stop,
                        "diagnostic": diagnostic,
                        "variant": variant,
                        "source_fields": sources.response_fields(body if isinstance(body, dict) else {}, None),
                        "start_ticks": stat[1] if stat else None,
                        # Only an authorized long gSIM measurement outlives the broker's stop.
                        "long": long_request(request.name[len("req_") : -len(".json")]),
                    }
                health["active_handlers"] = len(active)
                health["handlers"] = [
                    {
                        "request_id": name[len("req_") : -len(".json")],
                        "pid": job["process"].pid,
                        "start_ticks": job["start_ticks"],
                        "diagnostic": job["diagnostic"],
                        "elapsed_s": max(0, (time.time_ns() - job["started_ns"]) / 1e9),
                        "deadline_remaining_s": max(0, job["deadline"] - time.monotonic()),
                        "submission": job["source_fields"]["selfcheck_submission"],
                    }
                    for name, job in active.items()
                ]
                health["active_variants"] = sum(job["variant"] for job in active.values())
                health["queued_requests"] = len(_pending_requests(ch, seen))
                _atomic_write(health_path, json.dumps(health, sort_keys=True))
                time.sleep(a.poll)
            except OSError as error:  # A full disk must not end the dispatcher; other errors still do.
                _report_loop_error("dispatcher", error)
                time.sleep(LOOP_ERROR_BACKOFF_S)
    finally:
        _drain_handlers(active)
    return 0


def _lane_holders(ch: Path, active: dict) -> list[dict]:
    """Handlers occupying a lane: a child waiting for a gSIM correctness slot runs nothing yet.

    Counting it would let a queue of waiting gSIM requests hold every lane a Spike check could use.
    """
    from .. import joint_gsim_queue as gsim_queue

    return [job for name, job in active.items()
            if gsim_queue.waiting(ch / f"progress_{name[len('req_'):-len('.json')]}.json") is None]


def _drain_handlers(active: dict) -> list[str]:
    """Stop every handler at the dispatcher's exit, except a running long gSIM measurement.

    Each handler is asked to drain its existing cancellation path before it is reaped (STOP and
    parent loss are independently checked by handlers). A long measurement the lane authorized is
    left running: its handler ignores the stop, ends at its own deadline (bounded by the campaign's
    end) and yields if final grading waits for a shared slot (:func:`_abort_reason`). Returns the
    request names left running.
    """
    left = [name for name, job in active.items() if job.get("long") and job["process"].poll() is None]
    stopping = {name: job for name, job in active.items() if name not in left}
    for name in left:
        print(f"[broker] long gSIM measurement {name} continues past the broker stop", file=sys.stderr, flush=True)
    for job in stopping.values():
        try:
            _atomic_write(job["stop"], "stop")
        except OSError:
            _stop_owned_worker(job["process"])  # No stop file: reap the original tree directly.
    for job in stopping.values():
        try:
            job["process"].wait(timeout=35)
        except subprocess.TimeoutExpired:
            _stop_owned_worker(job["process"])
        job["stop"].unlink(missing_ok=True)
    return left


def _legacy_common_service(workspace, manifest, repo):
    """External evaluation has no authority to issue legacy common-pass proofs."""
    from ..feedback.snapshot_execution import selected

    if manifest is None or selected():
        return None
    from .common_transforms import CommonTransformService

    return CommonTransformService(workspace=workspace, bundle_manifest=manifest, repo=repo)


@selected_child
def main(
    argv=None,
    *,
    context=None,
    capsules_root: Path | None = None,
    contract: Path | None = None,
    policy_capsules_root: Path | None = None,
):
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
    from ..frozen_facts import select

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
                from merlin.common.path_scrub import scrub_host_paths

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
            if r.get("action") == "mechanism-question":
                try:
                    from ..joint_mechanism_questions import submit
                    body = submit(r.get("mechanism_question"), request_id=rid)
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                    body = {"status": "mechanism_question_denied", "reason": str(error),
                            "all_pass": False, "numeric_assessment": "not_run", "fit_eligible": False}
                doc = _decorate_response(json.dumps(body), r, rid, graded_digest,
                    started_ns=started_ns, completed_ns=time.time_ns())
                _atomic_write(resp, json.dumps(doc))
                _atomic_write(ch / f"done_{rid}", "ok")
                continue
            if r.get("action") in COMMON_ACTIONS:
                try:
                    if common_service is None:
                        raise FileNotFoundError("versioned common transformation tooling was not admitted for this run")
                    if r["action"] == "common-check" and common_service.profile.get("common_substrate") not in (
                        "merlin.common_compiler_substrate.v3",
                        "merlin.common_compiler_substrate.v4",
                        "merlin.common_compiler_substrate.v5",
                        "merlin.common_compiler_substrate.v6",
                        "merlin.common_compiler_substrate.v7",
                        "merlin.common_compiler_substrate.v8",
                        "merlin.common_compiler_substrate.v9",
                        "merlin.common_compiler_substrate.v10",
                        "merlin.common_compiler_substrate.v11",
                        "merlin.common_compiler_substrate.v12",
                        "merlin.common_compiler_substrate.v13",
                        "merlin.common_compiler_substrate.v14",
                        "merlin.common_compiler_substrate.v15",
                    ):
                        raise FileNotFoundError("common check was not admitted by the original compiler profile")
                    proof_root = a.compiler_bundle_manifest.parent / "common-transform-work"
                    proof_root.mkdir(mode=0o700, exist_ok=True)
                    remaining = int(_child_deadline_monotonic(r) - time.monotonic())
                    if remaining < 1:
                        raise TimeoutError("common request exhausted its original client deadline")
                    method = common_service.check if r["action"] == "common-check" else common_service.apply
                    body = method(r.get("transform"), work_root=proof_root, timeout_s=remaining)
                except Exception as error:
                    from merlin.common.path_scrub import scrub_host_paths

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
            suite_error = None
            measurement = None  # an operator-authorized idle gSIM measurement (joint_gsim_measurement)
            if suite in {"scalability", "programs"}:
                if suite == "programs":
                    from ..feedback.public_programs import request_policy, selected
                else:
                    from ..feedback.public_scalability import request_policy, selected
                from ..joint_gsim_measurement import broker_admission

                probing = bool(inspection or any(r.get(k) for k in FORWARDED_PROBES))
                # Only the scalability suite has an idle gSIM measurement lane.
                admits = suite == "scalability" and not (probing or variant)
                measurement = broker_admission(rid, r) if admits else None
                try:
                    if measurement is None:
                        request_policy(r.get("sim"), r.get("tiers"), inspection=probing)
                    if selected(context) is None:
                        raise ValueError("no operator public suite is selected")
                except (ValueError, TypeError) as exc:
                    suite_error = f"public {suite} request refused ({type(exc).__name__})"
            elif suite != "core":
                suite_error = "unknown public suite"
            sim = None if inspection else str(r.get("sim") or _default_sim() or "") or None
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
            from .. import joint_control

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
                from ..feedback.raw_selfcheck_evidence import operator_log_dir

                # Raw child streams can name host paths: keep them operator-side when an evidence
                # root is selected; the agent receives only the scrubbed response.
                log_dir = operator_log_dir() or ch
                log_out, log_err = log_dir / f".out_{rid}", log_dir / f".err_{rid}"
                staged_resp = ch / f".resp_{rid}.{os.getpid()}.tmp"
                staged_resp.unlink(missing_ok=True)
                aborted = False
                # Every check grades a private request snapshot. The author can keep
                # editing, and the response stays bound to the selected digest.
                snapshot_root = _working_snapshot(ws, graded_digest, label=label)
                r = tuning.apply_request(snapshot_root / "submission", r, digest=_submission_digest)
                from ..feedback.construction_plan import admit as admit_construction_plan

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
                from ..joint_model_activation import request_environment
                worker_env = request_environment(os.environ, channel="selfcheck", request_id=rid,
                                                 request_bytes=request_bytes)
                worker_env[sources.LABEL_ENV] = label
                from .. import joint_gsim_queue as gsim_queue

                # An author request waits for a busy gSIM correctness slot; operator work does not.
                worker_env[gsim_queue.CLASS_ENV] = gsim_queue.AUTHOR
                if r.get("origin") in OPERATOR_ORIGINS:
                    # Operator background measurement: every author request and grade leases CPUs first.
                    from merlin.common.cpu_pool import AUXILIARY, PRIORITY_ENV

                    worker_env[PRIORITY_ENV] = str(AUXILIARY)
                    worker_env[gsim_queue.CLASS_ENV] = gsim_queue.BACKGROUND
                from ..joint_gsim_measurement import ENV as MEASUREMENT_ENV

                worker_env.pop(MEASUREMENT_ENV, None)
                if measurement is not None:
                    worker_env[MEASUREMENT_ENV] = rid  # the child re-checks this id in the run-side ledger
                # Only submission/ may feed promotion: a variant's verdict never reaches the cert ledger.
                if (
                    suite == "core"
                    and not variant
                    and not r.get("caller_layout")
                    and not any(r.get(key) for key in FORWARDED_PROBES)
                ):
                    worker_env.update(
                        MERLIN_SELFCHECK_PROMOTION_WS=str(ws),
                        MERLIN_SELFCHECK_PROMOTION_ROOT=str(promotion_capsules_root or capsules_root),
                        MERLIN_SELFCHECK_PROMOTION_SHA256=graded_digest,
                    )
                with log_out.open("w") as fo, log_err.open("w") as fe, contextlib.ExitStack() as priority:
                    from merlin.common.cpu_pool import worker_pool_command
                    from merlin_experiments.frozen_python import inherited_python_command

                    proc = subprocess.Popen(
                        worker_pool_command(inherited_python_command(argv2), worker_env),
                        cwd=str(ws),
                        stdout=fo,
                        stderr=fe,
                        text=True,
                        env=worker_env,
                    )
                    from ..feedback.background_admission import active_full_selfcheck

                    # The foreground marker describes submission/ work; a variant publishes none.
                    foreground = contextlib.nullcontext()
                    if suite == "core" and not variant:
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
                            doc = _decorate_response(
                                json.dumps({"error": f"broker: self-check aborted ({why})", "all_pass": False}),
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
                        from ..feedback.idle_tuning import attach as attach_tuning
                        from ..feedback.knob_lint import attach as attach_knob_candidates

                        doc = attach_knob_candidates(attach_tuning(doc, ws), ws, submission=sources.SUBMISSION)
                        _atomic_write(resp, json.dumps(doc))
                    else:
                        out = log_out.read_text()[-20000:] if log_out.exists() else ""
                        err = _log_tail(log_err, 300)
                        doc = _decorate_response(
                            out or json.dumps({"error": err, "all_pass": False}),
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
                if resp.exists() and not variant and suite == "core":  # only core results can promote
                    passed = json.loads(resp.read_text())
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
            if inspection or variant or suite != "core":
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

                from merlin_experiments.phase1.feedback.promotion import cert_already_executed
                from merlin_experiments.phase1.feedback.promotion import promote as _promote
                from merlin_experiments.phase1.feedback.promotion import resolve_tiers as _resolve

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
                            from merlin_experiments.phase1.feedback.selfcheck_certificates import import_executed

                            import_executed(ws, _v, loop_tier=_loop, cert_tier=_cert, source_ws=promotion_source,
                                            context=context, log=sys.stderr)
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
    from merlin_experiments.phase1.telemetry.resilient_output import install

    install()  # A full log volume drops log lines, never the broker.
    from merlin.common.cpu_pool import AUTHOR, cpu_priority

    with cpu_priority(AUTHOR, export=True):  # Author-requested self-checks lease first.
        raise SystemExit(main())
