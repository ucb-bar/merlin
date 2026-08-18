#!/usr/bin/env python3
"""Gate: every target coupling in the core tree is DECLARED, and the count only falls.

``merlin/contract/overfit_register.yaml`` says where the target-agnostic core is still welded to a
specific target, what each weld costs, and what would remove it. A hand-maintained debt list rots, so
this gate does not trust it: it re-measures the live tree with the same scan the name gate uses and
compares.

Two directions of drift, both caught:

* a coupled module that is **not** declared -> FAIL. New debt has to be written down, with an owner and
  a removal condition, rather than absorbed silently. This is the whole point: absorbing was the old
  behaviour and it is how 54 dependencies accumulated where no allowlist knew about them.
* a declared module that **no longer couples** -> reported as ready to delete. Stale entries make the
  count meaningless, and a register that overstates the debt gets ignored just as fast as one that
  understates it.

Schema is enforced per status, because a required-fields rule with no escape hatch just pressures people
into inventing rationales. ``status: triaged`` means somebody read the code and can say why the fact is
not derivable (``why``), what it costs (``blocks``) and what would remove it (``removal_condition``).
``status: untriaged`` means it was surfaced by a tool and nobody has looked yet, and must say by what and
when. A fabricated removal condition reads as rigour and is worse than an honest gap.

    python build_tools/scripts/check_overfit_register.py            # full check (exit 1 on failure)
    python build_tools/scripts/check_overfit_register.py --summary  # the debt table, exit 0
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTER = ROOT / "merlin" / "contract" / "overfit_register.yaml"
NAME_GATE = ROOT / "build_tools" / "scripts" / "check_no_target_name.py"

REQUIRED = ("id", "kind", "status", "items", "owner")
REQUIRED_TRIAGED = ("why", "blocks", "removal_condition")
REQUIRED_UNTRIAGED = ("surfaced_by", "surfaced_on")
KINDS = ("target_name", "target_coupling", "runtime_concession", "dead_seam")
STATUSES = ("triaged", "untriaged")


def _load_name_gate():
    """Import the name gate by path, so the two gates cannot disagree about what coupling means."""
    spec = importlib.util.spec_from_file_location("_name_gate", NAME_GATE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {NAME_GATE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _coupled_files() -> set[str]:
    """The live set of in-scope modules that depend on a specific target."""
    gate = _load_name_gate()
    return {line.split(":", 1)[0] for line in gate.coupling_inventory()}


def load_register(path: Path | None = None) -> list[dict]:
    import yaml

    p = path or REGISTER
    if not p.is_file():
        raise SystemExit(f"no overfit register at {p} — the gate cannot verify what is not declared")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    entries = raw.get("entries") or []
    if not isinstance(entries, list):
        raise SystemExit(f"{p}: `entries` must be a list")
    return entries


def schema_problems(entries: list[dict]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        where = entry.get("id") or f"entries[{i}]"
        for field in REQUIRED:
            if not entry.get(field):
                problems.append(f"{where}: missing required field {field!r}")
        if entry.get("id") in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(entry.get("id"))
        kind, status = entry.get("kind"), entry.get("status")
        if kind and kind not in KINDS:
            problems.append(f"{where}: kind {kind!r} not in {list(KINDS)}")
        if status and status not in STATUSES:
            problems.append(f"{where}: status {status!r} not in {list(STATUSES)}")
        needed = REQUIRED_TRIAGED if status == "triaged" else REQUIRED_UNTRIAGED
        for field in needed:
            if not entry.get(field):
                problems.append(f"{where}: status {status!r} requires {field!r}")
        if not isinstance(entry.get("items"), list):
            problems.append(f"{where}: `items` must be a list of repo-relative paths")
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    entries = load_register()
    declared: dict[str, str] = {}
    for entry in entries:
        for item in entry.get("items") or []:
            declared[str(item)] = entry.get("id", "?")

    problems = schema_problems(entries)
    live = _coupled_files()
    undeclared = sorted(live - set(declared))
    stale = sorted(set(declared) - live)
    # A declared path that is not a coupled module is only "stale" if it is the kind of entry the scan can
    # see. runtime_concession / dead_seam entries name a file for provenance, not because the scan found
    # it, so they are exempt from the stale check.
    scanned_kinds = {"target_name", "target_coupling"}
    scan_visible = {item for entry in entries if entry.get("kind") in scanned_kinds
                    for item in (entry.get("items") or [])}
    stale = [s for s in stale if s in scan_visible]

    if "--summary" in argv:
        print(f"overfit register: {len(entries)} entr(y/ies) covering {len(declared)} item(s); "
              f"{len(live)} coupled module(s) measured live")
        for entry in entries:
            n = len(entry.get("items") or [])
            print(f"  [{entry.get('status','?'):9}] {entry.get('kind','?'):18} {entry.get('id','?')} "
                  f"({n} item(s), owner {entry.get('owner','?')})")
        return 0

    failed = False
    if problems:
        failed = True
        print(f"[FAIL] overfit-register: {len(problems)} schema problem(s):")
        for p in problems:
            print(f"  - {p}")
    if undeclared:
        failed = True
        # Display the register path relative to the repo when it is inside it, else verbatim — a test or a
        # caller may point this at a register elsewhere, and crashing on the error path would hide the error.
        try:
            shown = REGISTER.relative_to(ROOT)
        except ValueError:
            shown = REGISTER
        print(f"[FAIL] overfit-register: {len(undeclared)} module(s) depend on a specific target and are "
              f"NOT declared in {shown}:")
        for u in undeclared:
            print(f"  - {u}")
        print("  Add them to an entry with an owner and a removal condition. New coupling is a decision, "
              "not an accident: declare it or do not introduce it.")
    if stale:
        print(f"[note] overfit-register: {len(stale)} declared item(s) no longer couple to a target — "
              "delete them so the count keeps meaning something:")
        for s in stale:
            print(f"  - {s}")

    if failed:
        return 1
    n_untriaged = sum(len(e.get("items") or []) for e in entries if e.get("status") == "untriaged")
    print(f"[  ok] overfit-register: all {len(live)} coupled module(s) declared "
          f"({len(entries)} entries, {n_untriaged} item(s) still untriaged). The count may only fall.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
