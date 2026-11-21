"""Translate durable request deadlines and enforce the selected campaign boundary."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from merlin.development.backend.authoring.library.compiler_library import selected_child
from merlin.development.backend.brokers import selfcheck_sources as sources
from merlin.development.backend.brokers import selfcheck_tuning as tuning
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

# ---- selfcheck_deadlines -----------------------------------------------------------------------

COMMON_ACTIONS = frozenset({"common-transform", "common-check"})


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
    from merlin.development.backend.campaign import control as joint_control

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


# ---- selfcheck_lanes ---------------------------------------------------------------------------
# The self-check broker's lane sizing and worker reaping (:mod:`.selfcheck`).
#
# How many request handlers the dispatcher runs at once and of which kind, the operator's per-request
# capsule bound, and how one handler's original process tree is stopped. The broker re-exports every
# name, so ``selfcheck._lane_admits`` and the rest stay where callers and tests look them up.


def _forwarded_probes():

    return FORWARDED_PROBES


def _dispatch_capacity() -> int:
    from merlin.common.process.cpu_pool import compute_capacity

    # A shared compute pool arbitrates the handlers' CPU, so it selects the dispatcher by default.
    value = os.environ.get("MERLIN_SELFCHECK_BROKER_JOBS", "2" if compute_capacity() else "1")
    if value not in {"1", "2"}:
        raise ValueError("MERLIN_SELFCHECK_BROKER_JOBS must be 1 or 2")
    return int(value)


CAPSULES_PER_COMPUTE_CPU = 8


def _public_corpus_size(capsules_root) -> int | None:
    """Capsules in the public corpus this broker grades, or ``None`` when it cannot be counted."""
    if capsules_root is None:
        return None
    roots = [capsules_root] if isinstance(capsules_root, (str, Path)) else list(capsules_root)
    try:
        return len({path.parent for root in roots for path in Path(root).rglob("capsule.yaml")}) or None
    except OSError:
        return None


def _ordinary_capsule_limit(corpus_size: int | None = None) -> int | None:
    """The operator's per-request capsule bound, or ``None`` when none is selected.

    ``auto`` sizes it from the shared compute pool the request's tiers will lease from
    (``MERLIN_COMPUTE_CPUS``), falling back to this process's CPU affinity when no pool is selected.
    An incomplete pool selection refuses, as it does for every other shared-compute consumer.
    ``auto`` is never below the public corpus (``corpus_size``): sized from the pool alone it was
    8 x 18 = 144 against a 146-capsule suite, which refused the full check the author is told to run.
    An explicit number is the operator's own bound and is used exactly.
    """
    raw = os.environ.get("MERLIN_SELFCHECK_MAX_CAPSULES", "").strip()
    if not raw:
        return None
    if raw == "auto":
        from merlin.common.process.cpu_pool import grading_cpus

        cpus = len(grading_cpus()) or len(os.sched_getaffinity(0))
        return max(CAPSULES_PER_COMPUTE_CPU * max(1, cpus), corpus_size or 0)
    if not raw.isdecimal() or int(raw) < 1:
        raise ValueError("MERLIN_SELFCHECK_MAX_CAPSULES must be a positive integer or 'auto'")
    return int(raw)


def _feedback_scope_error(request, *, corpus_size: int | None = None):
    """Operator-selected iteration bound; formal qualification uses its own path.

    Under ``auto`` a full-suite request is admitted once the corpus is counted, because the bound
    covers it; an explicit number keeps full-suite feedback disabled, as that operator chose.
    """
    limit = _ordinary_capsule_limit(corpus_size)
    if limit is None:
        return None
    if request.get("action") or request.get("caller_layout") or any(request.get(key) for key in _forwarded_probes()):
        return None
    selected = str(request.get("capsules", "all"))
    names = {name.strip() for name in selected.split(",") if name.strip()}
    full_suite_fits = bool(corpus_size) and os.environ.get("MERLIN_SELFCHECK_MAX_CAPSULES", "").strip() == "auto"
    if (selected.strip() == "all" and not full_suite_fits) or len(names) > limit:
        return (
            f"iteration feedback accepts at most {limit} named capsules per request; select affected capsules. "
            "Full-suite feedback during authoring is disabled by the operator. "
            "The separate final frozen qualification still covers the complete corpus."
        )
    return None


def _ordinary_capacity():
    from merlin.common.process.cpu_pool import compute_capacity

    # Concurrent numerical handlers; each one's steps lease the shared pool, which is the only limiter.
    pool = compute_capacity()
    raw = os.environ.get("MERLIN_SELFCHECK_ORDINARY_JOBS", str(pool or 1))
    if not raw.isascii() or not raw.isdecimal() or not 1 <= int(raw) <= max(2, pool):
        raise ValueError("MERLIN_SELFCHECK_ORDINARY_JOBS must be 1 or 2, or at most the shared compute pool size")
    return int(raw)


def _process_record(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(stat[1]), int(stat[19]), stat[0]
    except (OSError, ValueError, IndexError):
        return None


def _stop_owned_worker(process):
    """Stop only this original process tree, checking birth before each signal."""
    owned = {}
    root = _process_record(process.pid)
    if root is None:
        process.wait(timeout=5)
        return
    owned[process.pid] = root[1]
    records = {}
    for directory in Path("/proc").iterdir():
        if directory.name.isdecimal():
            row = _process_record(int(directory.name))
            if row is not None:
                records[int(directory.name)] = row
    changed = True
    while changed:
        children = {pid: row[1] for pid, row in records.items() if row[0] in owned and pid not in owned}
        owned.update(children)
        changed = bool(children)
    for kind in (signal.SIGTERM, signal.SIGKILL):
        for pid, birth in reversed(tuple(owned.items())):
            current = _process_record(pid)
            if current is not None and current[1] == birth and current[2] != "Z":
                try:
                    os.kill(pid, kind)
                except ProcessLookupError:
                    pass
        end = time.monotonic() + 3
        while time.monotonic() < end:
            if not any(
                (row := _process_record(pid)) is not None and row[1] == birth and row[2] != "Z"
                for pid, birth in owned.items()
            ):
                break
            time.sleep(0.02)
    process.wait(timeout=5)


def _diagnostic_request(request: Path, *, request_bytes=None) -> bool:
    try:
        body = json.loads(request.read_bytes() if request_bytes is None else request_bytes)
        return bool(
            body.get("action") in ("mechanism-question", "reference-analysis")
            or body.get("caller_layout")
            or any(body.get(key) for key in _forwarded_probes())
        )
    except (OSError, ValueError, AttributeError):
        return False  # The unchanged handler reports the malformed request.


def _lane_admits(
    active, *, diagnostic: bool, variant: bool, ordinary_capacity: int, variant_capacity: int, overflow: bool = False
) -> bool:
    """Whether one more handler of this kind may start beside the ``active`` ones.

    Up to ``ordinary_capacity`` numerical requests run at once plus one diagnostic probe. Variant
    requests share the numerical lane, at most ``variant_capacity`` of them, so a queue of variants
    never holds every slot a ``submission/`` check could use. A request skipped here stays queued in
    arrival order without blocking a later request of another kind. ``overflow`` (see
    :func:`_overflow_admits`) lets one numerical request start past the ordinary lanes.
    """
    active = tuple(active)
    if diagnostic:
        return sum(job["diagnostic"] for job in active) < 1
    variants = sum(job["variant"] and not job["diagnostic"] for job in active)
    if variant and variants >= variant_capacity:
        return False
    return overflow or sum(not job["diagnostic"] for job in active) < ordinary_capacity


def _author_cpu_backlog() -> int | None:
    """Author-class leases queued in the shared compute pool now; ``None`` without a pool."""
    from merlin.common.process.cpu_pool import AUTHOR, queued_waiters

    return queued_waiters(AUTHOR)


def _overflow_admits(active, *, ordinary_capacity: int, author_backlog: int | None, pool: int) -> bool:
    """Whether one author request may start past the ordinary lanes now.

    A handler leases a CPU only for its build and simulator steps and spends the rest of its life
    copying, starting interpreters and waiting on children, so a lane count equal to the CPU count
    keeps only part of the pool busy with author work. Measured on one run: with every lane held and
    96 requests queued at the dispatcher, self-checks used about 8 of 20 reserved CPUs while
    background grading and promotions used 7 and 4 sat idle. A request queued here is invisible to
    the pool's priority order, so those CPUs went to lower classes.

    While no author lease is queued in the pool (``author_backlog == 0``), every admitted author step
    is being served and one more author request would take a free CPU, or the next one a lower class
    releases. Once author leases queue, admission stops and the pool orders the backlog. Work that
    runs without a lease cannot appear as a backlog, so the extra handlers are bounded by the pool's
    own size (``pool``): at most one more handler per reserved CPU.
    """
    if author_backlog != 0 or pool < 1:
        return False
    return sum(not job["diagnostic"] for job in tuple(active)) < ordinary_capacity + pool


# ---- selfcheck_requests ------------------------------------------------------------------------
# Durable self-check request leases, cancellation and broker stop conditions.
#
# The dispatcher and its workers use the same protocol decisions across restarts.
# Deadline translation stays in selfcheck_deadlines; this module only decides
# whether an existing request is pending or must stop, without grading it.


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
        from merlin.development.backend.campaign.joint_gsim_measurement import (
            REQUEST_PREFIX as MEASUREMENT_PREFIX,
        )
        from merlin.development.backend.campaign.joint_gsim_measurement import long_request

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
        from merlin.development.backend.campaign.joint_gsim_measurement import shared_pool_contended

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


# ---- selfcheck ---------------------------------------------------------------------------------
# Driver-side self-check BROKER — lets a bwrap-sandboxed agent get a REDACTED self-check verdict without
# the oracle ever entering its sandbox.
#
# The redacted self-check (agent_selfcheck.py) internally needs the oracle (merlin.runtime.engine.reference) to
# compute pass/fail — so it CANNOT run inside the agent's sandbox (the oracle is masked there). This broker
# runs OUTSIDE the sandbox (oracle available), watches a channel dir inside the agent's (bind-mounted)
# workspace, and on each request runs the real agent_selfcheck.py against the agent's submission, writing the
# redacted verdict + the agent's OWN artifacts back into the workspace. The agent talks to it via the thin
# shim staged at <ws>/agent_selfcheck.py.
#
# Channel (under <ws>/.qa_channel/, RW from both sides via the bind mount):
#   req_<id>.json   agent -> broker : {submission, capsules, workers, timeout, shape_coverage,
#                                      offload_census, model_layers, optional sim/tiers}
#   resp_<id>.json  broker -> agent : the redacted self-check JSON (golden expected values withheld)
#   done_<id>       broker -> agent : marker that resp is complete
#   detached_<id>   agent -> broker : the waiting client returned on purpose; keep the request until
#                                     its deadline instead of judging it by the client's process
#   STOP            driver -> broker: sentinel to exit (written after the agent round)
#
# ``submission`` is ``submission`` (the default) or ``variants/<name>``: see ``selfcheck_sources``.
#
# Usage: selfcheck_broker.py --ws <workspace> [--poll 0.5]


def _freeze_passed_source(ws: Path, expected_digest: str, *, source_ws: Path | None = None):
    """Bind asynchronous promotion to the source just graded, before releasing the author."""
    from merlin.development.backend.feedback.execution import snapshots

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

_DIGEST_IGNORES = frozenset({"build", "__pycache__", ".git", "selfcheck_out"})


def _submission_digest(root: Path) -> str:
    """Stable digest of authored submission bytes, excluding disposable build state."""
    from merlin.common.process.cpu_pool import shared_compute_pool

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


_AGENT_ROOTS: tuple[str, ...] = ()


def agent_visible_text(text: str) -> str:
    """``text`` with host paths outside the agent's work tree replaced; full detail stays in the logs."""
    from merlin.common.access import scrub_host_paths

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


def _advisory_request(request: dict) -> bool:
    """Background timings and sweep baselines are observations, never promotion inputs."""
    experiment = request.get("tuning_experiment")
    return request.get("origin") in (IDLE_ORIGIN, TUNING_ORIGIN) or (
        isinstance(experiment, dict) and experiment.get("schema") == "merlin.idle_tuning_experiment.v1"
    )


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
    if _advisory_request(request):
        body.update(official=False, advisory_only=True)
    from merlin.development.backend.feedback.references import (
        experiment_guidance as _merlin_experiment_guidance_1,
    )
    from merlin.development.backend.feedback.references import experiments as reference_experiments

    body.pop("reference_experiment_result", None)  # only the owned request/result join can issue this observation
    _merlin_experiment_guidance_1.attach(body, current_compiler_sha256=submission_sha256)
    _merlin_experiment_guidance_1.observe_result(request, body, base_compiler_sha256=submission_sha256)
    return body


def _rtl_engines() -> tuple[str, ...]:
    try:
        from merlin.target.simulators.engine_policy import ENGINE_PRIORITY

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
    if path.name.startswith(("resp_", "progress_")) and path.suffix == ".json":
        from merlin.development.backend.feedback.disclosure import json_text

        text = json_text(text, workspace=path.parent.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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
    command = [sys.executable, "-m", "merlin.development.backend.brokers.selfcheck"]
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
    from merlin.development.backend.authoring.library.compiler_library import child_arguments

    return [*command, *child_arguments()]


def _dispatch_requests(a, context, capsules_root, contract, promotion_root):
    """Two independent processes: one numerical lane and one diagnostic lane.

    Each handler reconstructs its own original service and retains the existing
    source, deadline, cancellation and publication checks. The parent never holds
    a compute lease across a handler wait or shares execution scopes with it.
    """
    from merlin.common.process.cpu_pool import worker_pool_command
    from merlin.development.backend.campaign.joint_gsim_measurement import long_request
    from merlin.experiments.frozen_python import inherited_python_command

    ws = Path(a.ws).resolve()
    ch, orig_ppid = ws / ".qa_channel", os.getppid()
    seen, active = set(), {}
    health_path = ch / "broker_health.json"
    try:
        health = json.loads(health_path.read_bytes())
    except (OSError, ValueError):
        health = {}
    from merlin.common.process.cpu_pool import compute_capacity

    ordinary_capacity = _ordinary_capacity()
    # Variants share the ordinary lane; their own bound keeps the rest of it for submission/ checks.
    variant_capacity = sources.variant_capacity(ordinary_capacity)
    overflow_capacity = compute_capacity()
    health.update(
        protocol=PROTOCOL_VERSION,
        dispatch_capacity=ordinary_capacity + 1,
        ordinary_capacity=ordinary_capacity,
        overflow_capacity=overflow_capacity,
        overflow_admitted=int(health.get("overflow_admitted", 0)),
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
                # Read the pool's author backlog at most once per pass, and admit at most one request
                # past the ordinary lanes per pass: a new handler takes seconds to reach its first lease.
                backlog = unread = object()
                for request in pending:
                    try:
                        request_bytes = request.read_bytes()
                    except FileNotFoundError:
                        continue
                    diagnostic = _diagnostic_request(request, request_bytes=request_bytes)
                    variant = sources.is_variant_request(request_bytes)
                    holders = _lane_holders(ch, active)
                    lanes = dict(
                        diagnostic=diagnostic,
                        variant=variant,
                        ordinary_capacity=ordinary_capacity,
                        variant_capacity=variant_capacity,
                    )
                    overflow = False
                    if not _lane_admits(holders, **lanes):
                        if diagnostic or request.name.startswith("req_" + IDLE_REQUEST_PREFIX):
                            continue  # operator background work never starts past the lanes
                        if backlog is unread:
                            backlog = _author_cpu_backlog()
                        overflow = _overflow_admits(
                            holders, ordinary_capacity=ordinary_capacity, author_backlog=backlog, pool=overflow_capacity
                        )
                        if not _lane_admits(holders, **lanes, overflow=overflow):
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
                        from merlin.development.backend.campaign.calibration.activation import (
                            request_environment,
                        )

                        handler_env = request_environment(
                            os.environ,
                            channel="selfcheck",
                            request_id=request.stem.removeprefix("req_"),
                            request_bytes=request_bytes,
                        )
                    except FileNotFoundError:
                        continue  # Ordinary cancellation must not stop another lane.
                    except (ValueError, KeyError, TypeError) as error:
                        rid = request.stem.removeprefix("req_")
                        _atomic_write(
                            ch / f"resp_{rid}.json",
                            json.dumps(
                                {
                                    "error": "request model admission refused: " + type(error).__name__,
                                    "all_pass": False,
                                    "numeric_assessment": "not_run",
                                }
                            ),
                        )
                        _atomic_write(ch / f"done_{rid}", "err")
                        seen.add(request.name)
                        continue
                    process = subprocess.Popen(
                        worker_pool_command(inherited_python_command(command), handler_env), cwd=ws, env=handler_env
                    )
                    stat = _process_record(process.pid)
                    seen.add(request.name)
                    if overflow:
                        backlog = None  # one request past the lanes per pass
                        health["overflow_admitted"] += 1
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
    from merlin.development.backend.campaign import gsim_budget as gsim_queue

    return [
        job
        for name, job in active.items()
        if gsim_queue.waiting(ch / f"progress_{name[len('req_') : -len('.json')]}.json") is None
    ]


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
    from merlin.development.backend.feedback.execution.snapshot_evaluator import selected

    if manifest is None or selected():
        return None
    from merlin.development.targetgen.execution.host.common_transforms import CommonTransformService

    return CommonTransformService(workspace=workspace, bundle_manifest=manifest, repo=repo)
