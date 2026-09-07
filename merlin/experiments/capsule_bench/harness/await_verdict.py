#!/usr/bin/env python3
"""In-sandbox BLOCKING WAIT for the next harness grade — staged as ``<ws>/await_verdict.py``.

Under the continuous schedule the harness re-grades a snapshot of the workspace every
``--grade-interval`` seconds and refreshes ``qa/verdict.json`` underneath the agent. Every other wait in
the toolbox blocks: ``agent_selfcheck.py`` returns when the broker answers, ``simjob.py wait`` returns
when a job lands. This one did not exist, so the only way to notice a new grade was to look again --
and an agent that must look again writes a polling loop:

    timeout 25s tail --pid=$PID -f /dev/null; stat -c %y qa/verdict.json; ls -l .qa_channel/resp_*

MEASURED on one 6.1 h run: 89 such commands aimed at ``qa/verdict.json`` and 77 at the self-check
channel, ~15.5 min of deliberate sleeping. The sleeping is not the expensive part -- each poll is also
a model round trip, and at that run's ~10.8 s of model time per turn the looking cost roughly 27 min.
Twelve percent of the run went into asking whether something had finished.

Imports NOTHING from merlin and reads only ``qa/verdict.json``, which the agent may already read: this
is a wait, not a new channel, and it can therefore be staged into the sandbox as-is.

    python await_verdict.py                     # block until the NEXT grade lands
    python await_verdict.py --timeout 900       # ... or give up after 15 min and say so
    python await_verdict.py --since-ns <ns>     # wait for a grade newer than a stamp you already hold

It always terminates and always prints one JSON object. ``waited`` says what happened -- ``graded`` (a
new grade landed), ``timeout`` (none did, in the time allowed), or ``absent`` (no verdict file exists
yet and none appeared). A timeout is not an error: the grade interval may simply be longer than the
wait, and the exit status says which so a shell can branch on it without parsing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

#: How often to look. The grade interval is minutes; this only bounds how late the return is.
_POLL_S = 2.0

#: Default ceiling. Deliberately longer than the default grade interval (900 s) so an ordinary wait
#: succeeds rather than reporting a timeout the agent would then have to interpret.
_DEFAULT_TIMEOUT_S = 1200


def _stamp(path: Path) -> "int | None":
    """The verdict's modification time in NANOSECONDS, or ``None`` if it is not there yet.

    Nanoseconds because a re-grade can land inside the same second, and a whole-second stamp would
    then report "nothing new" for a grade that had in fact just happened. What is being waited for is
    a GRADE, not a change of score -- two consecutive grades may agree exactly, and the agent still
    needs to know the second one ran.
    """
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _summary(path: Path) -> dict:
    """The parts of the verdict worth echoing, or an explanation. Never raises, never invents."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"readable": False, "why": f"{type(exc).__name__}: {exc}"}
    if not isinstance(doc, dict):
        return {"readable": False, "why": "the verdict is not a JSON object"}
    out = {"readable": True}
    for key in ("n_passed", "n_capsules", "all_pass", "integrity_status", "highest_tier"):
        if key in doc:
            out[key] = doc[key]
    failing = [name for name, status in (doc.get("per_capsule") or {}).items()
               if isinstance(status, str) and status != "pass"] \
        if isinstance(doc.get("per_capsule"), dict) else \
        [row.get("capsule") for row in (doc.get("per_capsule") or [])
         if isinstance(row, dict) and row.get("status") != "pass"]
    out["failing"] = sorted(n for n in failing if n)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdict", default="qa/verdict.json",
                    help="the harness verdict to wait on (default: qa/verdict.json)")
    ap.add_argument("--timeout", type=int, default=_DEFAULT_TIMEOUT_S,
                    help=f"seconds to wait before giving up and saying so (default {_DEFAULT_TIMEOUT_S})")
    ap.add_argument("--since-ns", type=int, default=0,
                    help="wait for a grade strictly newer than this st_mtime_ns; 0 (default) means "
                         "'newer than whatever is there right now'")
    a = ap.parse_args(argv)

    path = Path(a.verdict)
    # BASELINE FIRST, THEN WAIT. Read before sleeping, or a grade that lands between the two is missed
    # and the agent waits a full interval for the one after it.
    baseline = a.since_ns or _stamp(path) or 0
    deadline = time.monotonic() + max(1, a.timeout)
    while True:
        now = _stamp(path)
        if now is not None and now > baseline:
            out = {"waited": "graded", "verdict": str(path), "mtime_ns": now,
                   "waited_s": round(a.timeout - max(0.0, deadline - time.monotonic()), 1)}
            out.update(_summary(path))
            print(json.dumps(out, indent=2))
            return 0
        if time.monotonic() >= deadline:
            kind = "timeout" if now is not None else "absent"
            print(json.dumps({
                "waited": kind, "verdict": str(path), "mtime_ns": now,
                "waited_s": a.timeout,
                "note": ("no new grade landed within the wait; the grade interval may be longer than "
                         "--timeout, so this is not necessarily a fault. Keep working and wait again."
                         if kind == "timeout" else
                         "no verdict file exists yet -- the first grade has not completed."),
            }, indent=2))
            return 2
        time.sleep(min(_POLL_S, max(0.0, deadline - time.monotonic())))


if __name__ == "__main__":
    raise SystemExit(main())
