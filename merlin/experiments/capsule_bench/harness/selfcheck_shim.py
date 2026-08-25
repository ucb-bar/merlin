"""In-sandbox self-check SHIM — staged as <ws>/agent_selfcheck.py under --sandbox bwrap.

Same CLI as the real agent_selfcheck.py, but imports NOTHING from merlin (the oracle is masked in the
sandbox). It forwards the request to the driver-side broker via <ws>/.qa_channel and prints the redacted
verdict the broker returns. From the agent's view this is identical to running the self-check directly —
it gets pass/fail + its own artifacts (in ./selfcheck_out/) — but the oracle/goldens never enter the box.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path


def _verdict(txt: str):
    """The verdict object out of the broker's reply, or ``None`` if there isn't one.

    The reply is the real self-check's whole stdout, and the grader prints human diagnostics on that
    same stream, so the JSON is not always the first thing in it. Scan for the document rather than
    assuming the text IS one -- structurally, via ``raw_decode`` from each ``{`` (no regex).
    """
    try:
        whole = json.loads(txt)
        if isinstance(whole, dict):
            return whole
    except Exception:
        pass
    dec, i = json.JSONDecoder(), txt.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(txt, i)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        i = txt.find("{", i + 1)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Agent self-check (redacted; routed to the driver-side broker).")
    ap.add_argument("--submission", default="submission")
    ap.add_argument("--sim", choices=["spike", "verilator", "vcs"], default="spike")
    ap.add_argument("--capsules", default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="")
    # Forwarded like every other flag. Without this the sandboxed agent -- which is EVERY real run --
    # gets "unrecognized arguments" and the shape-coverage report is unreachable exactly where it
    # matters, while working fine when tested outside the box.
    ap.add_argument("--shape-coverage", action="store_true")
    a = ap.parse_args(argv)

    ws = Path(__file__).resolve().parent          # the shim lives at <ws>/agent_selfcheck.py
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    rid = f"{os.getpid()}_{int(time.time() * 1000) % 1000000}"
    (ch / f"req_{rid}.json").write_text(json.dumps(
        {"sim": a.sim, "capsules": a.capsules, "workers": a.workers, "timeout": a.timeout,
         "shape_coverage": bool(a.shape_coverage)}))
    resp, done = ch / f"resp_{rid}.json", ch / f"done_{rid}"
    deadline = time.time() + a.timeout + 240
    while time.time() < deadline:
        if done.exists() and resp.exists():
            txt = resp.read_text()
            print(txt)
            if a.out:
                Path(a.out).write_text(txt)
            _v = _verdict(txt)
            if _v is None:
                # FAIL CLOSED. This used to `return 0` -- so a reply the shim could not read was
                # indistinguishable from a clean run, and every exit-code check downstream (the agent's
                # own, the conformance probe, the shape-coverage gate below) silently read as a pass.
                print(json.dumps({"error": "self-check reply was not parseable as JSON — "
                                           "treating as FAILED, not as clean"}))
                return 2
            # the shape-coverage report has no `all_pass`; its verdict is `all_covered`
            if a.shape_coverage:
                return 0 if _v.get("all_covered") else 1
            return 0 if _v.get("all_pass") else 1
        time.sleep(0.4)
    print(json.dumps({"error": "self-check broker did not respond (timeout) — tell the operator"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
