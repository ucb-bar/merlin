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
import subprocess
import sys
import time
from pathlib import Path

SELFCHECK = Path(__file__).resolve().parent / "agent_selfcheck.py"


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
    ap.add_argument("--plateau-checks", type=int, default=0,
                    help="stop serving self-checks after N consecutive full-corpus checks that made no "
                         "progress (0 = never). The round-level --plateau-rounds terminator cannot fire "
                         "inside a single round; this is the same test at self-check granularity.")
    a = ap.parse_args(argv)
    ws = Path(a.ws)
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    orig_ppid = os.getppid()
    seen: set[str] = set()
    import plateau as _PL
    stall = _PL.Detector(a.plateau_checks)
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
        pending = sorted((p for p in ch.glob("req_*.json") if p.name not in seen),
                         key=lambda p: p.stat().st_mtime)
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
            resp.write_text(json.dumps({"error": "broker: unreadable request"}))
            (ch / f"done_{rid}").write_text("err")
            continue
        # A STALLED round stops BUYING self-checks. Each full-corpus check costs the oracle 80-120s and
        # the agent blocks on it, so a loop that has stopped improving spends most of its remaining wall
        # here. Answer immediately and truthfully instead: the request is not dropped (a silent drop
        # would hang the agent's poll), it is served a terminal verdict naming why.
        if stall.stalled():
            resp.write_text(json.dumps({
                "error": "self-check closed for this round: " + stall.why(),
                "plateau": True, "n_checks_stalled": stall.stall,
                "note": "Further self-checks will not be served this round. Your last verdict stands. "
                        "If you have an untried idea, make the edit and let the ROUND grade judge it."}))
            (ch / f"done_{rid}").write_text("plateau")
            continue
        to = int(r.get("timeout", 1800))
        argv2 = [sys.executable, str(SELFCHECK),
                 "--submission", str(ws / "submission"),
                 "--sim", str(r.get("sim", "spike")),
                 "--capsules", str(r.get("capsules", "all")),
                 "--workers", str(r.get("workers", 8)),
                 "--timeout", str(to),
                 "--out", str(resp)]
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
                        resp.write_text(json.dumps(
                            {"error": f"broker: self-check aborted ({why or 'timed out'})"}))
                        (ch / f"done_{rid}").write_text("err")
                        aborted = True
                        break
                    time.sleep(a.poll)
            if not aborted and not resp.exists():
                out = log_out.read_text()[-20000:] if log_out.exists() else ""
                err = log_err.read_text()[-300:] if log_err.exists() else ""
                resp.write_text(out or json.dumps({"error": err}))
        except Exception as e:
            resp.write_text(json.dumps({"error": f"broker: {type(e).__name__}: {str(e)[:200]}"}))
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
        # Fold this verdict into the in-round plateau test. AFTER the response is written, so a stall is
        # never the reason a completed check goes unanswered -- the agent always gets the verdict it
        # waited for, and only the NEXT request is refused.
        try:
            if resp.exists():
                _sv = json.loads(resp.read_text())
                if isinstance(_sv, dict) and stall.observe(_sv, str(r.get("capsules", "all"))):
                    print(f"[plateau] {stall.why()} — closing the self-check loop for this round",
                          file=sys.stderr, flush=True)
        except Exception as _se:  # noqa: BLE001 -- detection must never break serving
            print(f"[plateau] skipped: {type(_se).__name__}: {_se}", file=sys.stderr, flush=True)
        if not (ch / f"done_{rid}").exists():
            (ch / f"done_{rid}").write_text("ok")


if __name__ == "__main__":
    raise SystemExit(main())
