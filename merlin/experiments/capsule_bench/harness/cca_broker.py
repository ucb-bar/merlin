"""Driver-side CCA-CONTRACT broker — gives the assisted arms (arm-3 / arm-4) the two mandated
Common-Compute-Abstraction introspection tools, WITHOUT full merlin having to import inside the sandbox.

The arm-4 prompt MANDATES ``cca_contract.check_bijection(target)`` and
``action_catalog.escalation_ladder(axis, target)``, but neither can run in the bwrap sandbox:
``merlin/kernels/types.py`` shadows Python's stdlib ``types`` when the kernels dir is flat on
``sys.path`` (a circular-import crash), and ``regions.py`` needs the unstaged ``merlin.common.paths``.
A driver-side broker running OUTSIDE the sandbox — with the whole of merlin legitimately on the path —
sidesteps both, exactly like :mod:`isa_tools_broker`.

It is ORACLE-FREE and reads NO golden: ``check_bijection`` diffs the PUBLIC CCA schema against the
PUBLIC action-catalog routes (the CCA<->CompilerAction bijection, hand-authored public structure) and
``escalation_ladder`` lists the public FLAG->KNOB->HEURISTIC->PASS->CODEGEN route ladder for an axis.
No capsule input, no expected output, no answer surface is ever touched — nothing here is a cheat.

Mirrors :mod:`isa_tools_broker`: watch ``<ws>/.cca_channel``, answer each request with a JSON result.

Channel (under ``<ws>/.cca_channel/``):
  req_<id>.json   agent -> broker : {cmd: check_bijection|escalation_ladder, target|axis, ...}
  resp_<id>.json  broker -> agent : the tool result JSON
  done_<id>       broker -> agent : completion marker
  STOP            driver -> broker: sentinel to exit
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_HERE))                                   # _common (parity with isa_tools_broker)
sys.path.insert(0, str(_REPO / "merlin" / "python"))            # merlin (out-of-box, oracle-free use)


def _bijection_to_dict(report) -> dict:
    """Serialize a :class:`cca_contract.BijectionReport` dataclass faithfully. asdict captures the five
    fields (backend / orphan_fields / orphan_routes / unclassified / ladder_errors); the derived ``clean``
    property and the KNOWN_OPEN-subtracted ``unexpected`` report are the two things the enforcing ratchet
    actually reads, so surface both explicitly (asdict omits property + method results)."""
    out = dataclasses.asdict(report)
    out["clean"] = bool(report.clean)
    unexpected = report.unexpected()
    out["unexpected"] = {
        "orphan_fields": unexpected.orphan_fields,
        "orphan_routes": unexpected.orphan_routes,
        "unclassified": unexpected.unclassified,
        "ladder_errors": unexpected.ladder_errors,
        "clean": bool(unexpected.clean),
    }
    return out


def _handle(req: dict) -> dict:
    cmd = req.get("cmd")

    if cmd == "check_bijection":
        target = (req.get("target") or "").strip()
        if not target:
            return {"error": "check_bijection needs a target (the backend to diff)"}
        from merlin.kernels import cca_contract
        return _bijection_to_dict(cca_contract.check_bijection(target))

    if cmd == "escalation_ladder":
        target = (req.get("target") or "").strip()
        axis = (req.get("axis") or "").strip()
        if not target or not axis:
            return {"error": "escalation_ladder needs both axis and target"}
        from merlin.kernels import action_catalog
        # escalation_ladder already returns a list of plain dicts (action_class/target_seam/forkable_now/
        # seam_file/seam_kind/needs_new_code) — JSON-clean as-is. Wrap so an empty ladder is still an
        # honest, non-error answer (an axis with no route is a real, informative result).
        ladder = action_catalog.escalation_ladder(axis, target)
        return {"axis": axis, "target": target, "ladder": ladder, "n": len(ladder)}

    return {"error": f"unknown cmd {cmd!r} (use check_bijection|escalation_ladder)"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True)
    ap.add_argument("--poll", type=float, default=0.4)
    a = ap.parse_args(argv)
    ch = Path(a.ws) / ".cca_channel"
    ch.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    while True:
        if (ch / "STOP").exists():
            break
        for req_f in sorted(ch.glob("req_*.json")):
            if req_f.name in seen:
                continue
            seen.add(req_f.name)
            rid = req_f.stem[len("req_"):]
            resp = ch / f"resp_{rid}.json"
            try:
                out = _handle(json.loads(req_f.read_text()))
            except Exception as e:  # noqa: BLE001 — never crash the broker on one bad request
                out = {"error": f"cca broker: {type(e).__name__}: {str(e)[:200]}"}
            resp.write_text(json.dumps(out, indent=2))
            (ch / f"done_{rid}").write_text("ok")
        time.sleep(a.poll)


if __name__ == "__main__":
    raise SystemExit(main())
