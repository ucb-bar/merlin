"""In-sandbox ASYNC oracle CLI — staged as <ws>/simjob.py under --sandbox bwrap.

Lets the agent run the simulators (spike / gsim / verilator / vcs) on its OWN submission, per-capsule, WITHOUT
blocking its turn: `submit` returns a job id immediately; the heavy sim runs OUTSIDE the sandbox (the
driver-side simjob_broker), and the agent `poll`s for the redacted verdict. This is how the agent gets
cycle-accurate verilator feedback even though one capsule takes minutes.

Imports NOTHING from merlin (the oracle is masked in the sandbox); it only talks to the broker over the
shared <ws>/.qa_channel directory. The broker runs the SAME redacted grader (agent_selfcheck.py) the
self-check uses, so goldens never enter the box.

Subcommands (same channel, simjob_* prefixes):
  submit  --sim <accepted-engine> --capsules <csv|all> [--debug NAME...] [--workers N]
            -> prints {"job_id","state":"queued","n_capsules"} and returns immediately
  poll    --job-id ID        -> {"job_id","state":queued|running|pending|done|error,
                                "result": <redacted|null>}
  wait    --job-id ID [--timeout S]   -> poll-loop (bounded; default short so a turn can't hang), prints result
  list                        -> this workspace's jobs + states

The broker sizes parallel capsule workers by engine when --workers is omitted; its global Verilator
slots bound the total across jobs. Poll long batches instead of blocking an agent turn.
"""

from __future__ import annotations

import argparse
import json
import os
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
    if (CH / f"simerr_{jid}").exists():
        return "error", _read(CH / f"simresp_{jid}.json")
    if (CH / f"simdone_{jid}").exists():
        return "done", _read(CH / f"simresp_{jid}.json")
    if (CH / f"simrun_{jid}").exists():
        progress = _read(CH / f"progress_{jid}.json")
        if isinstance(progress, dict) and progress.get("status") in {"running", "grading_complete"}:
            return "pending", progress
        return "running", None
    if (CH / f"simreq_{jid}.json").exists():
        return "queued", None
    return "unknown", None


def _submit(a) -> int:
    CH.mkdir(parents=True, exist_ok=True)
    jid = f"{os.getpid()}_{int(time.time() * 1000) % 1000000}"
    caps = [c.strip() for c in a.capsules.split(",") if c.strip()] if a.capsules != "all" else "all"
    n = len(caps) if isinstance(caps, list) else 20
    (CH / f"simreq_{jid}.json").write_text(
        json.dumps(
            {
                "sim": a.sim,
                "capsules": a.capsules,
                "debug": a.debug or [],
                "workers": a.workers,
                "submitted_at": int(time.time()),
            }
        )
    )
    print(
        json.dumps(
            {
                "job_id": jid,
                "state": "queued",
                "sim": a.sim,
                "n_capsules": n,
                "note": f"poll with: simjob.py poll --job-id {jid}",
            }
        )
    )
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
    st, progress = _state(a.job_id)
    if st in ("done", "error"):
        print(json.dumps({"job_id": a.job_id, "state": st, "result": progress}))
        return 0 if (progress or {}).get("all_pass") else 1
    print(json.dumps({"job_id": a.job_id, "state": st, "result": progress,
                      "note": "still pending after --timeout; keep polling"}))
    return 0


def _list(a) -> int:
    jobs = {}
    for req in sorted(CH.glob("simreq_*.json")) if CH.is_dir() else []:
        jid = req.stem[len("simreq_") :]
        jobs[jid] = _state(jid)[0]
    print(json.dumps({"jobs": jobs}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Async oracle runner (spike/verilator/vcs) for your submission.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    # The accepted set is the BROKER's derived allowlist, not a baked ladder: a target that declares no
    # bespoke sim grades on its contract-resolved tier and accepts only the neutral "contract" token.
    # Baking the chipyard ladder here left such a target's agent unable to submit ANY oracle job.
    # NO client-side allowlist. This shim runs INSIDE the sandbox, where the broker module is not
    # granted, so any mirror of its allowlist silently falls back to one target's ladder and rejects a
    # legitimate request before it ever reaches the channel. Validation is the BROKER's job -- it runs
    # OUTSIDE the sandbox, is the trust boundary, and already fails closed on an unknown sim with a
    # visible "rejected" response. Keeping a second, drifting copy here bought nothing and cost a
    # whole target its in-sandbox oracle.
    s = sub.add_parser("submit")
    s.add_argument("--sim", required=True)
    s.add_argument("--capsules", default="all")
    s.add_argument("--debug", nargs="*", default=[])
    s.add_argument("--workers", type=int, default=0, help="parallel capsule workers; 0 lets the broker size the batch")
    s.set_defaults(fn=_submit)
    p = sub.add_parser("poll")
    p.add_argument("--job-id", required=True, dest="job_id")
    p.set_defaults(fn=_poll)
    w = sub.add_parser("wait")
    w.add_argument("--job-id", required=True, dest="job_id")
    w.add_argument("--timeout", type=int, default=120)
    w.set_defaults(fn=_wait)
    ls = sub.add_parser("list")
    ls.set_defaults(fn=_list)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
