"""Invocation-time Phase 1 spend-ledger accounting.

The ledger is a soft stop between authoring rounds, not a grading verdict.
Importing this module does not read credentials, environment, or run state.
"""

from __future__ import annotations

import json
from pathlib import Path


def spend_over_cap(this_round_cost) -> tuple[bool, float, float]:
    """Append subagent-inclusive cost to MERLIN_SPEND_LEDGER; return (over_cap, total, cap).

    MERLIN_MAX_SPEND_USD is a soft cap: one in-flight round per arm may overshoot.
    Missing cap/ledger disables it; unknown usage remains a visible lower bound.
    """
    import os as _os

    cap = float(_os.environ.get("MERLIN_MAX_SPEND_USD") or 0)
    ledger = _os.environ.get("MERLIN_SPEND_LEDGER")
    if cap <= 0 or not ledger:
        return False, 0.0, 0.0
    import fcntl

    # Missing usage is unknown, never zero; a timeout may prevent the terminal usage event.
    _unmeasured = this_round_cost is None
    c = None if _unmeasured else float(this_round_cost)
    p = Path(ledger)
    p.parent.mkdir(parents=True, exist_ok=True)
    total, n_unmeasured = 0.0, 0
    with open(p, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({"cost": c, "unmeasured": _unmeasured}) + "\n")
        f.flush()
        f.seek(0)
        for line in f:
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 — a malformed ledger line must not defeat the cap
                continue
            if row.get("unmeasured") or row.get("cost") is None:
                n_unmeasured += 1
                continue
            try:
                total += float(row.get("cost") or 0)
            except Exception:  # noqa: BLE001
                continue
        fcntl.flock(f, fcntl.LOCK_UN)
    if n_unmeasured:
        print(
            f"  [spend] ${total:.2f} of ${cap:.2f} measured, plus {n_unmeasured} UNMEASURED round(s) "
            f"whose usage never arrived — the true total is a LOWER BOUND",
            flush=True,
        )
    return total >= cap, total, cap
