"""T4 — static guard that the realistic task prompt carries the four reliability nudges, so a future
edit can't silently drop them (each was a real abc7 failure mode)."""
from __future__ import annotations
import re
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "task" / "TASK_realistic.md"
CHECKS = {
    "READY precondition (spike clean first)": re.compile(r"FULLY clean.*READY_FOR_BARRIER|do \*\*NOT\*\* write `READY_FOR_BARRIER`|premature READY", re.I | re.S),
    "async verilator via simjob": re.compile(r"simjob\.py submit --sim verilator", re.I),
    "verilator failure is non-terminal": re.compile(r"NOT the end|fix and retry", re.I),
    "VCS unavailable": re.compile(r"VCS .*UNAVAILABLE|VCS \(L4\) is UNAVAILABLE", re.I),
}


def main():
    txt = TASK.read_text()
    ok = True
    for name, rx in CHECKS.items():
        hit = bool(rx.search(txt))
        ok = ok and hit
        print(f"  [{'PASS' if hit else 'FAIL'}] {name}")
    print(f"\nT4 prompt nudges: {'all present' if ok else 'MISSING some'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
