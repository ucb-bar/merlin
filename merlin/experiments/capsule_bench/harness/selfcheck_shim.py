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
import secrets
import sys
import time
from pathlib import Path


def _request_id() -> str:
    """A diagnostic request id which stays unique across PID-namespace and clock reuse.

    Namespace PIDs are short-lived and the previous millisecond clock component wrapped every
    1,000,000 ms.  Both values have collided in long rounds, at which point a new shim can observe an
    old request's response and completion marker.  Keep them for operator diagnostics, but add a
    cryptographic nonce and use the full nanosecond clock rather than a modulo clock.
    """
    return f"{os.getpid()}_{time.time_ns()}_{secrets.token_hex(16)}"


def _atomic_write(path: Path, text: str) -> None:
    """Publish one channel file only after all of its bytes are durable."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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
    # DEFAULTS TO THE CERTIFYING SIM, not the screen. The capsules declare a cycle-accurate cert
    # tier as mandatory, and this ladder runs cheapest-measured-first with fail-fast, so the
    # screen still refutes a broken submission at screen cost -- what changes is that a capsule
    # which PASSES the screen goes on to certify instead of stopping there. Choosing "spike"
    # explicitly is a legitimate fast screen, but it CANNOT certify: the mandatory cert tier
    # reports unavailable and the capsule is not a pass.
    ap.add_argument("--sim", choices=["spike", "verilator", "vcs"], default="verilator")
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
    rid = _request_id()
    deadline = time.time() + a.timeout + 240
    _atomic_write(ch / f"req_{rid}.json", json.dumps(
        {"protocol": 2, "request_id": rid, "deadline_unix_ns": int(deadline * 1_000_000_000),
         "sim": a.sim, "capsules": a.capsules, "workers": a.workers, "timeout": a.timeout,
         "shape_coverage": bool(a.shape_coverage)}))
    resp, done = ch / f"resp_{rid}.json", ch / f"done_{rid}"
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
