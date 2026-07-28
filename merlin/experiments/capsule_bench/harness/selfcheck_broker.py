"""Driver-side self-check BROKER — lets a bwrap-sandboxed agent get a REDACTED self-check verdict without
the oracle ever entering its sandbox.

The redacted self-check (agent_selfcheck.py) internally needs the oracle (merlin.runtime.reference) to
compute pass/fail — so it CANNOT run inside the agent's sandbox (the oracle is masked there). This broker
runs OUTSIDE the sandbox (oracle available), watches a channel dir inside the agent's (bind-mounted)
workspace, and on each request runs the real agent_selfcheck.py against the agent's submission, writing the
redacted verdict + the agent's OWN artifacts back into the workspace. The agent talks to it via the thin
shim staged at <ws>/agent_selfcheck.py.

Channel (under <ws>/.qa_channel/, RW from both sides via the bind mount):
  req_<id>.json   agent -> broker : {sim, capsules, workers, timeout}
  resp_<id>.json  broker -> agent : the redacted self-check JSON (golden expected values withheld)
  done_<id>       broker -> agent : marker that resp is complete
  STOP            driver -> broker: sentinel to exit (written after the agent round)

Usage: selfcheck_broker.py --ws <workspace> [--poll 0.5]
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SELFCHECK = Path(__file__).resolve().parent / "agent_selfcheck.py"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--poll", type=float, default=0.5)
    a = ap.parse_args(argv)
    ws = Path(a.ws)
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    while True:
        if (ch / "STOP").exists():
            break
        for req in sorted(ch.glob("req_*.json")):
            if req.name in seen:
                continue
            seen.add(req.name)
            rid = req.stem[len("req_"):]
            resp = ch / f"resp_{rid}.json"
            try:
                r = json.loads(req.read_text())
            except Exception:
                resp.write_text(json.dumps({"error": "broker: unreadable request"}))
                (ch / f"done_{rid}").write_text("err")
                continue
            to = int(r.get("timeout", 1800))
            argv2 = [sys.executable, str(SELFCHECK),
                     "--submission", str(ws / "submission"),
                     "--sim", str(r.get("sim", "spike")),
                     "--capsules", str(r.get("capsules", "all")),
                     "--workers", str(r.get("workers", 8)),
                     "--timeout", str(to),
                     "--out", str(resp)]
            try:
                # run the REAL self-check OUTSIDE the sandbox (oracle available); cwd=ws so the agent's
                # own artifacts land in <ws>/selfcheck_out/ (visible to the agent through the bind mount).
                p = subprocess.run(argv2, cwd=str(ws), capture_output=True, text=True, timeout=to + 180)
                if not resp.exists():
                    resp.write_text(p.stdout or json.dumps({"error": (p.stderr or "")[-300:]}))
            except subprocess.TimeoutExpired:
                resp.write_text(json.dumps({"error": "broker: self-check timed out"}))
            except Exception as e:
                resp.write_text(json.dumps({"error": f"broker: {type(e).__name__}: {str(e)[:200]}"}))
            (ch / f"done_{rid}").write_text("ok")
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
