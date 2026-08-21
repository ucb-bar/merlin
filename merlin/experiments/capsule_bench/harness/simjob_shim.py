"""In-sandbox ASYNC oracle CLI — staged as <ws>/simjob.py under --sandbox bwrap.

Lets the agent run the simulators (spike / verilator / vcs) on its OWN submission, per-capsule, WITHOUT
blocking its turn: `submit` returns a job id immediately; the heavy sim runs OUTSIDE the sandbox (the
driver-side simjob_broker), and the agent `poll`s for the redacted verdict. This is how the agent gets
cycle-accurate verilator feedback even though one capsule takes minutes.

Imports NOTHING from merlin (the oracle is masked in the sandbox); it only talks to the broker over the
shared <ws>/.qa_channel directory. The broker runs the SAME redacted grader (agent_selfcheck.py) the
self-check uses, so goldens never enter the box.

Subcommands (same channel, simjob_* prefixes):
  submit  --sim {spike|verilator|vcs} --capsules <csv|all> [--debug NAME...] [--workers N]
            -> prints {"job_id","state":"queued","n_capsules"} and returns immediately
  poll    --job-id ID        -> {"job_id","state":queued|running|done|error|canceled, "result": <redacted|null>}
  wait    --job-id ID [--timeout S]   -> poll-loop (bounded; default short so a turn can't hang), prints result
  cancel  --job-id ID        -> kill a queued/running job (frees its simulator slot for your next submit)
  list                        -> this workspace's jobs + states

Verilator is slow (minutes/capsule): prefer `submit` a few per-capsule jobs, then `poll` — do NOT
`wait` on a big verilator batch. A stale full-suite job holds a simulator slot for hours: `cancel` it
as soon as its result no longer matters (e.g. you changed the submission).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parent
CH = WS / ".qa_channel"


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _state(jid: str) -> tuple[str, dict | None]:
    if (CH / f"simcanceled_{jid}").exists():
        return "canceled", _read(CH / f"simresp_{jid}.json")
    if (CH / f"simerr_{jid}").exists():
        return "error", _read(CH / f"simresp_{jid}.json")
    if (CH / f"simdone_{jid}").exists():
        return "done", _read(CH / f"simresp_{jid}.json")
    if (CH / f"simcancel_{jid}").exists():
        return "cancel_requested", None
    if (CH / f"simrun_{jid}").exists():
        return "running", None
    if (CH / f"simreq_{jid}.json").exists():
        return "queued", None
    return "unknown", None


def _submit(a) -> int:
    CH.mkdir(parents=True, exist_ok=True)
    jid = f"{os.getpid()}_{int(time.time() * 1000) % 1000000}"
    caps = [c.strip() for c in a.capsules.split(",") if c.strip()] if a.capsules != "all" else "all"
    n = len(caps) if isinstance(caps, list) else "all"   # the broker knows the real 'all' count, not us
    (CH / f"simreq_{jid}.json").write_text(json.dumps({
        "sim": a.sim, "capsules": a.capsules, "debug": a.debug or [],
        "workers": a.workers, "submitted_at": int(time.time())}))
    print(json.dumps({"job_id": jid, "state": "queued", "sim": a.sim, "n_capsules": n,
                      "note": "poll with: simjob.py poll --job-id %s" % jid}))
    return 0


def _poll(a) -> int:
    st, res = _state(a.job_id)
    print(json.dumps({"job_id": a.job_id, "state": st, "result": res}))
    return 0


def _wait(a) -> int:
    deadline = time.time() + a.timeout
    while time.time() < deadline:
        st, res = _state(a.job_id)
        if st in ("done", "error"):
            print(json.dumps({"job_id": a.job_id, "state": st, "result": res}))
            return 0 if (res or {}).get("all_pass") else 1
        time.sleep(1.0)
    print(json.dumps({"job_id": a.job_id, "state": "running",
                      "note": "still running after --timeout; keep polling (verilator is slow)"}))
    return 0


def _cancel(a) -> int:
    """Request cancellation: drop a sentinel the broker honors on its next poll tick. Kills a RUNNING
    job (freeing its simulator slot) or voids a QUEUED one; a job already done/error is left as-is."""
    st, _ = _state(a.job_id)
    if st in ("done", "error", "canceled", "unknown"):
        print(json.dumps({"job_id": a.job_id, "state": st,
                          "note": "nothing to cancel (job already finished or unknown)"}))
        return 0
    (CH / f"simcancel_{a.job_id}").write_text("cancel")
    print(json.dumps({"job_id": a.job_id, "state": "cancel_requested",
                      "note": "the broker honors this on its next tick; poll to confirm 'canceled'"}))
    return 0


def _list(a) -> int:
    jobs = {}
    for req in sorted(CH.glob("simreq_*.json")) if CH.is_dir() else []:
        jid = req.stem[len("simreq_"):]
        jobs[jid] = _state(jid)[0]
    print(json.dumps({"jobs": jobs}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Async oracle runner (spike/verilator/vcs) for your submission.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("--sim", choices=["spike", "verilator", "vcs"], required=True)
    s.add_argument("--capsules", default="all"); s.add_argument("--debug", nargs="*", default=[])
    s.add_argument("--workers", type=int, default=1); s.set_defaults(fn=_submit)
    p = sub.add_parser("poll"); p.add_argument("--job-id", required=True, dest="job_id"); p.set_defaults(fn=_poll)
    w = sub.add_parser("wait"); w.add_argument("--job-id", required=True, dest="job_id")
    w.add_argument("--timeout", type=int, default=120); w.set_defaults(fn=_wait)
    c = sub.add_parser("cancel"); c.add_argument("--job-id", required=True, dest="job_id"); c.set_defaults(fn=_cancel)
    ls = sub.add_parser("list"); ls.set_defaults(fn=_list)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
