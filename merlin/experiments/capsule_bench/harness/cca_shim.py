"""In-sandbox CCA-CONTRACT shim — staged as BOTH <ws>/cca_contract.py and <ws>/action_catalog.py for
the assisted arms (arm-3 / arm-4), so the mandated calls resolve as plain imports:

    from cca_contract import check_bijection
    check_bijection("atlas")                      # the CCA<->CompilerAction bijection report

    from action_catalog import escalation_ladder
    escalation_ladder("spatial.dataflow", "atlas")  # the FLAG->..->CODEGEN route ladder for one axis

Imports NOTHING from merlin (full merlin does not import in the sandbox: merlin/kernels/types.py shadows
stdlib ``types`` on a flat sys.path, and regions.py needs the unstaged merlin.common.paths). It forwards
each request to the driver-side :mod:`cca_broker` via <ws>/.cca_channel and returns the parsed JSON.
Both entry points are ORACLE-FREE (public CCA structure — no golden is ever involved).

The module is staged under two names, so it exposes BOTH functions regardless of which name imports it.

CLI (convenience):
    python cca_contract.py check-bijection atlas
    python action_catalog.py escalation-ladder spatial.dataflow atlas
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

_TIMEOUT = 120


def _forward(req: dict, timeout: int = _TIMEOUT) -> dict:
    """Send one request to the driver-side cca_broker via <ws>/.cca_channel and return the parsed JSON."""
    ws = Path(__file__).resolve().parent               # the shim lives at <ws>/{cca_contract,action_catalog}.py
    ch = ws / ".cca_channel"
    ch.mkdir(parents=True, exist_ok=True)
    rid = f"{os.getpid()}_{int(time.time() * 1000) % 1000000}"
    (ch / f"req_{rid}.json").write_text(json.dumps(req))
    resp, done = ch / f"resp_{rid}.json", ch / f"done_{rid}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if done.exists() and resp.exists():
            try:
                return json.loads(resp.read_text())
            except Exception:  # noqa: BLE001
                return {"error": "cca broker returned unparseable JSON"}
        time.sleep(0.3)
    return {"error": "cca broker did not respond (timeout) — tell the operator"}


def check_bijection(target: str) -> dict:
    """The CCA<->CompilerAction bijection report for ``target`` (orphan fields/routes, unclassified,
    ladder errors, plus the KNOWN_OPEN-subtracted ``unexpected`` view and the ``clean`` verdict)."""
    return _forward({"cmd": "check_bijection", "target": target})


def escalation_ladder(axis: str, target: str) -> dict:
    """The full weakest->strongest route ladder for one CCA ``axis`` on ``target`` — each rung's action
    class, concrete seam file, and whether it is forkable today. Returns {axis, target, ladder, n}."""
    return _forward({"cmd": "escalation_ladder", "axis": axis, "target": target})


def main(argv=None):
    ap = argparse.ArgumentParser(description="CCA contract tools (bijection report + escalation ladder).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("check-bijection", help="diff the CCA schema against the routed axes for a target")
    b.add_argument("target", help="the backend/target to check (e.g. atlas)")
    e = sub.add_parser("escalation-ladder", help="the route ladder (FLAG->..->CODEGEN) for one axis")
    e.add_argument("axis", help="the CCA axis (e.g. spatial.dataflow)")
    e.add_argument("target", help="the backend/target (e.g. atlas)")
    a = ap.parse_args(argv)

    if a.cmd == "check-bijection":
        out = check_bijection(a.target)
    else:
        out = escalation_ladder(a.axis, a.target)
    print(json.dumps(out, indent=2))
    return 1 if isinstance(out, dict) and out.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
