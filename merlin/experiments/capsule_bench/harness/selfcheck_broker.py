"""Driver-side self-check BROKER — lets a bwrap-sandboxed agent get a REDACTED self-check verdict without
the oracle ever entering its sandbox.

The redacted self-check (agent_selfcheck.py) internally needs the oracle (merlin.runtime.reference) to
compute pass/fail — so it CANNOT run inside the agent's sandbox (the oracle is masked there). This broker
runs OUTSIDE the sandbox (oracle available), watches a channel dir inside the agent's (bind-mounted)
workspace, and on each request runs the real agent_selfcheck.py against the agent's submission, writing the
redacted verdict + the agent's OWN artifacts back into the workspace. The agent talks to it via the thin
shim staged at <ws>/agent_selfcheck.py.

Channel (under <ws>/.qa_channel/, RW from both sides via the bind mount):
  req_<id>.json   agent -> broker : {sim, capsules, workers, timeout, shape_coverage}
  resp_<id>.json  broker -> agent : the redacted self-check JSON (golden expected values withheld)
  done_<id>       broker -> agent : marker that resp is complete
  STOP            driver -> broker: sentinel to exit (written after the agent round)

Usage: selfcheck_broker.py --ws <workspace> [--poll 0.5]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

SELFCHECK = Path(__file__).resolve().parent / "agent_selfcheck.py"
PROTOCOL_VERSION = 3
_DIGEST_IGNORES = frozenset({"build", "__pycache__", ".git", "selfcheck_out"})


def _submission_digest(root: Path) -> str:
    """Stable digest of authored submission bytes, excluding disposable build state."""
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


def _decorate_response(raw: str, request: dict, rid: str, submission_sha256: str, *,
                       started_ns: int, completed_ns: int) -> dict:
    """Return a JSON verdict explicitly bound to the request and graded submission bytes."""
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise TypeError("self-check response is not an object")
    except Exception as exc:  # noqa: BLE001 -- malformed grader output becomes an explicit failure
        body = {"error": f"broker: unreadable self-check response ({type(exc).__name__})",
                "all_pass": False, "raw_tail": raw[-1000:]}
    requested = request.get("submission_sha256")
    body.update({
        "selfcheck_protocol": PROTOCOL_VERSION,
        "selfcheck_request_id": rid,
        "submission_sha256": submission_sha256,
        "requested_submission_sha256": requested,
        "requested_at_unix_ns": request.get("requested_at_unix_ns"),
        "grade_started_at_unix_ns": started_ns,
        "graded_at_unix_ns": completed_ns,
    })
    if requested and requested != submission_sha256:
        # SAID, NOT HIDDEN. The verdict describes bytes NEWER than the ones asked about, which is more
        # useful than the ones asked about -- but only while it is stated, or the agent would read a
        # verdict as being about code it has since replaced.
        body["graded_newer_bytes"] = True
        body["graded_newer_note"] = (
            "you edited the submission while this check was queued, so it graded your CURRENT bytes "
            "rather than the ones present when you asked. The verdict is about the newer code.")
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
    """Spike plus either the one pinned RTL engine or the closed registered RTL set."""
    required = _required_rtl_engine()
    if required is not None:
        return ("spike", required) if required in _rtl_engines() else ("spike",)
    return ("spike",) + _rtl_engines()


def _default_sim() -> str:
    required = _required_rtl_engine()
    return required if required in _rtl_engines() else "verilator"


def _sim_policy_error(sim: str) -> str | None:
    allowed = _allowed_sims()
    if sim in allowed:
        return None
    required = _required_rtl_engine()
    if required is not None:
        return (f"--sim {sim!r} conflicts with MERLIN_REQUIRED_RTL_ENGINE={required!r}; "
                f"allowed: {list(allowed)!r}")
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


def _child_deadline_monotonic(request: dict, *, now_wall: float | None = None,
                              now_monotonic: float | None = None) -> float:
    """Translate the client's durable wall-clock deadline onto this process's monotonic clock.

    The request deadline accounts for the whole suite.  Reconstructing a shorter
    ``per_capsule_timeout + 180`` deadline here used to kill full-corpus checks while their later worker
    waves were still healthy.
    """
    wall = time.time() if now_wall is None else now_wall
    monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    explicit = request.get("deadline_unix_ns")
    if explicit is not None:
        return monotonic + max(0.0, int(explicit) / 1_000_000_000 - wall)
    return monotonic + max(0, int(request.get("timeout", 1800))) + 240


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


def _pending_requests(ch: Path, seen: set[str], *, now: float | None = None) -> list[Path]:
    """Outstanding, live requests in arrival order.

    ``seen`` is process-local, while completion markers and deadlines survive a broker restart.  All
    three are needed: relying only on ``seen`` caused a restarted broker to rerun every historical
    full-corpus grade before it reached the request whose client was actually waiting.
    """
    now = time.time() if now is None else now
    pending: list[Path] = []
    for req in ch.glob("req_*.json"):
        if req.name in seen:
            continue
        rid = req.stem[len("req_"):]
        if (ch / f"done_{rid}").exists() and (ch / f"resp_{rid}.json").exists():
            continue
        try:
            body = json.loads(req.read_text(encoding="utf-8"))
            if _requester_alive(body) is False:
                continue
            if now > _request_deadline(req):
                continue
            req.stat()
        except FileNotFoundError:
            continue
        pending.append(req)
    return sorted(pending, key=lambda p: p.stat().st_mtime)


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--poll", type=float, default=0.5)
    a = ap.parse_args(argv)
    ws = Path(a.ws)
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    orig_ppid = os.getppid()
    seen: set[str] = set()
    health_path = ch / "broker_health.json"
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
        why = _should_stop(ch, orig_ppid)
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
        rid = req.stem[len("req_"):]
        resp = ch / f"resp_{rid}.json"
        try:
            r = json.loads(req.read_text())
        except Exception:
            _atomic_write(resp, json.dumps({"error": "broker: unreadable request"}))
            _atomic_write(ch / f"done_{rid}", "err")
            continue
        started_ns = time.time_ns()
        graded_digest = _submission_digest(ws / "submission")
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
                    r, rid, graded_digest, started_ns=started_ns, completed_ns=time.time_ns())
                _atomic_write(resp, json.dumps(doc))
                _atomic_write(ch / f"done_{rid}", "err")
                continue
        sim = str(r.get("sim", _default_sim()))
        policy_error = _sim_policy_error(sim)
        if policy_error:
            doc = _decorate_response(json.dumps({
                "error": f"broker: {policy_error}", "all_pass": False, "sim": sim,
                "required_rtl_engine": _required_rtl_engine(),
            }), r, rid, graded_digest, started_ns=started_ns, completed_ns=time.time_ns())
            _atomic_write(resp, json.dumps(doc))
            _atomic_write(ch / f"done_{rid}", "err")
            continue
        to = int(r.get("timeout", 1800))
        argv2 = [sys.executable, str(SELFCHECK),
                 "--submission", str(ws / "submission"),
                 "--sim", sim,
                 "--capsules", str(r.get("capsules", "all")),
                 "--workers", str(r.get("workers", 8)),
                 "--timeout", str(to),
                 # The real self-check writes --out directly.  Point it at a private name and publish
                 # by rename only after it exits, so the shim can never read a half-written verdict.
                 "--out", str(ch / f".resp_{rid}.{os.getpid()}.tmp"),
                 # Progress is explicitly NOT a verdict. It is safe to publish while the final response
                 # stays hidden because it contains only public capsule names and completed statuses.
                 "--progress-out", str(ch / f"progress_{rid}.json")]
        # forward the shape-coverage request through the sandbox shim (the agent cannot reach the
        # oracle itself, so a flag it sets here is the only way the check runs at all)
        if r.get("shape_coverage"):
            argv2.append("--shape-coverage")
        try:
            # Run the REAL self-check OUTSIDE the sandbox (oracle available); cwd=ws so the agent's own
            # artifacts land in <ws>/selfcheck_out/ (visible to the agent through the bind mount).
            # Popen + poll rather than subprocess.run so STOP / parent-death is honored WHILE a long check
            # is in flight; run() blocked until completion and left RTL sims running past the round.
            deadline = _child_deadline_monotonic(r)
            # Spool the child's streams to FILES, not pipes: a poll loop that never drains a PIPE
            # deadlocks as soon as the child fills the buffer, and a full-corpus self-check is chatty.
            log_out, log_err = ch / f".out_{rid}", ch / f".err_{rid}"
            staged_resp = ch / f".resp_{rid}.{os.getpid()}.tmp"
            staged_resp.unlink(missing_ok=True)
            aborted = False
            with log_out.open("w") as fo, log_err.open("w") as fe:
                proc = subprocess.Popen(argv2, cwd=str(ws), stdout=fo, stderr=fe, text=True)
                while proc.poll() is None:
                    why = _should_stop(ch, orig_ppid)
                    if why or time.monotonic() > deadline:
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            proc.kill()                 # a wedged RTL sim does not get to outlive us
                            proc.wait(timeout=30)
                        staged_resp.unlink(missing_ok=True)
                        doc = _decorate_response(
                            json.dumps({"error": f"broker: self-check aborted ({why or 'timed out'})",
                                        "all_pass": False}),
                            r, rid, graded_digest, started_ns=started_ns,
                            completed_ns=time.time_ns())
                        _atomic_write(resp, json.dumps(doc))
                        _atomic_write(ch / f"done_{rid}", "err")
                        aborted = True
                        break
                    time.sleep(a.poll)
            if not aborted:
                completed_ns = time.time_ns()
                completed_digest = _submission_digest(ws / "submission")
                if completed_digest != graded_digest:
                    staged_resp.unlink(missing_ok=True)
                    doc = _decorate_response(
                        json.dumps({"error": "broker: submission changed during grading; retry the check",
                                    "all_pass": False}),
                        r, rid, completed_digest, started_ns=started_ns, completed_ns=completed_ns)
                    _atomic_write(resp, json.dumps(doc))
                elif staged_resp.exists():
                    raw = staged_resp.read_text(encoding="utf-8", errors="replace")
                    staged_resp.unlink(missing_ok=True)
                    doc = _decorate_response(raw, r, rid, completed_digest,
                                             started_ns=started_ns, completed_ns=completed_ns)
                    _atomic_write(resp, json.dumps(doc))
                else:
                    out = log_out.read_text()[-20000:] if log_out.exists() else ""
                    err = log_err.read_text()[-300:] if log_err.exists() else ""
                    doc = _decorate_response(
                        out or json.dumps({"error": err, "all_pass": False}),
                        r, rid, completed_digest, started_ns=started_ns, completed_ns=completed_ns)
                    _atomic_write(resp, json.dumps(doc))
        except Exception as e:
            doc = _decorate_response(
                json.dumps({"error": f"broker: {type(e).__name__}: {str(e)[:200]}",
                            "all_pass": False}),
                r, rid, _submission_digest(ws / "submission"), started_ns=started_ns,
                completed_ns=time.time_ns())
            _atomic_write(resp, json.dumps(doc))
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

        # PROMOTE off the SYNCHRONOUS path too. Promotion was first hooked into the async oracle alone,
        # and a live run then showed the agent using THIS path seven times to the async path's two -- so
        # eight verdicts completed and promotion fired zero times. Wherever a verdict is produced,
        # promotion is considered.
        try:
            import json as _j
            from tier_promote import promote as _promote, resolve_tiers as _resolve
            _loop, _cert, _cover = _resolve(ws)
            if _loop and _cert and resp.exists():
                _v = _j.loads(resp.read_text())
                if isinstance(_v, dict) and not _v.get("error"):
                    _promote(ws, ch, _v, _loop, _cert, _cover, sys.stderr)
        except Exception as _pe:  # noqa: BLE001 -- promotion is an optimisation, never a gate
            print(f"[promote] skipped: {type(_pe).__name__}: {_pe}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
