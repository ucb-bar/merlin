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
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

SELFCHECK = Path(__file__).resolve().parent / "agent_selfcheck.py"


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
    while True:
        why = _should_stop(ch, orig_ppid)
        if why:
            print(f"[broker] exiting: {why}", file=sys.stderr)
            break
        # FIFO by ARRIVAL TIME. The previous ``sorted(glob)`` ordered by FILENAME, and a request is named
        # req_<requester-pid>_<nonce>, so service order followed pid values rather than arrival: a request
        # could be deferred behind every newly-arrived lower-sorting name and never claimed at all. That
        # was measured — an agent spent the remainder of its round waiting on a request submitted 10
        # minutes earlier while a later one was served, then ended the round with 3.5h of its budget
        # unused. Oldest-first also means the request the agent is actually blocked on is the one served.
        pending = _pending_requests(ch, seen)
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
        sim = str(r.get("sim", _default_sim()))
        policy_error = _sim_policy_error(sim)
        if policy_error:
            _atomic_write(resp, json.dumps({
                "error": f"broker: {policy_error}", "all_pass": False, "sim": sim,
                "required_rtl_engine": _required_rtl_engine(),
            }))
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
                 "--out", str(ch / f".resp_{rid}.{os.getpid()}.tmp")]
        # forward the shape-coverage request through the sandbox shim (the agent cannot reach the
        # oracle itself, so a flag it sets here is the only way the check runs at all)
        if r.get("shape_coverage"):
            argv2.append("--shape-coverage")
        try:
            # Run the REAL self-check OUTSIDE the sandbox (oracle available); cwd=ws so the agent's own
            # artifacts land in <ws>/selfcheck_out/ (visible to the agent through the bind mount).
            # Popen + poll rather than subprocess.run so STOP / parent-death is honored WHILE a long check
            # is in flight; run() blocked until completion and left RTL sims running past the round.
            deadline = time.monotonic() + to + 180
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
                        _atomic_write(resp, json.dumps(
                            {"error": f"broker: self-check aborted ({why or 'timed out'})"}))
                        _atomic_write(ch / f"done_{rid}", "err")
                        aborted = True
                        break
                    time.sleep(a.poll)
            if not aborted:
                if staged_resp.exists():
                    os.replace(staged_resp, resp)
                else:
                    out = log_out.read_text()[-20000:] if log_out.exists() else ""
                    err = log_err.read_text()[-300:] if log_err.exists() else ""
                    _atomic_write(resp, out or json.dumps({"error": err}))
        except Exception as e:
            _atomic_write(resp, json.dumps({"error": f"broker: {type(e).__name__}: {str(e)[:200]}"}))
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
        if not (ch / f"done_{rid}").exists():
            _atomic_write(ch / f"done_{rid}", "ok")


if __name__ == "__main__":
    raise SystemExit(main())
