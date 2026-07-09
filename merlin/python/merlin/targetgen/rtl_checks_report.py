"""Advisory surfacing for the RTL-derived checks layer (Phase 1 — ADVISORY, opt-in, un-gated).

Wires :mod:`merlin.targetgen.rtl_checks` *around* the frozen runner without touching it: given a
capsule-bench run directory, it reads the already-decoded ``generated/instruction_trace.json``,
resolves the declared capsule + the target's RTL facts, calls :func:`rtl_checks.screen`, and writes a
sibling ``rtl_checks.json`` next to ``capsule_result.json``. Pure advisory — it imports nothing from
``capsule_runner`` / ``capsule_grade`` / the merlin/contract schemas, mutates no existing artifact, and
changes no pass/fail verdict. Callers that want a cost pre-screen can use :func:`prescreen` and choose
to skip the expensive oracle on ``verdict == "reject"`` (the saving is realised caller-side; the frozen
runner is never consulted about it).

This is design Phase "advisory surfacing" + the opt-in pre-screen helper from
``artifacts/perf-bench/gemmini/rtl_checks_layer_design.md``. The deeper phases (feed into the
agent loop, label runs checks-active, T1 spike-byproduct checks) remain follow-ups.

CLI::

    # screen one run dir or a whole runs/ tree, writing rtl_checks.json per capsule run:
    python -m merlin.targetgen.rtl_checks_report <run_dir_or_runs_root> [--write] [--quiet]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from . import rtl_checks
from .rtl.facts import target_contract_path

_REPO = Path(__file__).resolve().parents[4]  # .../oscar-merlin
_GEMMINI_CONTRACT = target_contract_path("gemmini")
# capsule.yaml definitions come from the corpus locator (canonical contract corpus + the perf corpus);
# every capsule dir is named after the capsule, matching the run dir name.
from merlin.common.corpora import capsule_corpus_roots
_CAPSULE_ROOTS = capsule_corpus_roots()


# --------------------------------------------------------------------------- RTL facts (real, cheap)
def _find_key(obj: Any, key: str) -> Any:
    """Depth-first search for the first value under ``key`` anywhere in a nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def load_rtl_facts(contract_path: Path | None = None,
                   introspect_json: Path | None = None) -> dict:
    """RTL facts from the authoritative target_contract.yaml (mesh + scratchpad), with an optional
    ``rtl/introspect.dump_facts`` JSON override. Falls back to rtl_checks.DEFAULT_RTL_FACTS if the
    contract is missing — the checks are still RTL-grounded, just from the curated default."""
    facts = dict(rtl_checks.DEFAULT_RTL_FACTS)
    cp = contract_path or _GEMMINI_CONTRACT
    try:
        doc = yaml.safe_load(cp.read_text(encoding="utf-8"))
        mesh = _find_key(doc, "mesh")
        if isinstance(mesh, dict) and "rows" in mesh and "cols" in mesh:
            facts["mesh"] = [int(mesh["rows"]), int(mesh["cols"])]
        elif isinstance(mesh, list) and len(mesh) == 2:
            facts["mesh"] = [int(mesh[0]), int(mesh[1])]
        spad = _find_key(doc, "resident_storage_bytes")
        if isinstance(spad, int):
            facts["scratchpad_bytes"] = int(spad)
        facts["from"] = f"target_contract.yaml ({cp.relative_to(_REPO)})"
    except FileNotFoundError:
        pass
    if introspect_json and Path(introspect_json).is_file():
        ov = json.loads(Path(introspect_json).read_text(encoding="utf-8"))
        m = _find_key(ov, "mesh")
        if isinstance(m, dict) and "rows" in m:
            facts["mesh"] = [int(m["rows"]), int(m["cols"])]
        spad = _find_key(ov, "resident_storage_bytes")
        if isinstance(spad, int):
            facts["scratchpad_bytes"] = spad
        facts["from"] = f"introspect override ({introspect_json})"
    return facts


# ------------------------------------------------------------------------------- capsule resolution
def build_capsule_index() -> dict[str, Path]:
    """Map capsule name -> capsule.yaml across the known corpora (dir name == capsule name)."""
    idx: dict[str, Path] = {}
    for root in _CAPSULE_ROOTS:
        if not root.is_dir():
            continue
        for cy in root.rglob("capsule.yaml"):
            name = cy.parent.name
            idx.setdefault(name, cy)
    return idx


def _load_capsule(name: str, index: dict[str, Path]) -> dict | None:
    p = index.get(name)
    if p is None:
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------------- screen one / many
def _capsule_name_for(run_capsule_dir: Path) -> str:
    """Capsule name for a run dir: prefer capsule_result.json's field, else the dir name."""
    res = run_capsule_dir / "capsule_result.json"
    if res.is_file():
        try:
            return json.loads(res.read_text(encoding="utf-8")).get("capsule") or run_capsule_dir.name
        except Exception:
            pass
    return run_capsule_dir.name


def screen_run(run_capsule_dir: Path, rtl_facts: dict, index: dict[str, Path],
               write: bool = True) -> rtl_checks.CheckReport | None:
    """Screen a single capsule run dir (the one holding capsule_result.json + generated/).

    Returns the CheckReport (and writes rtl_checks.json beside capsule_result.json when ``write``),
    or None if there is no decoded trace to screen."""
    trace_path = run_capsule_dir / "generated" / "instruction_trace.json"
    if not trace_path.is_file():
        return None
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    capsule = _load_capsule(_capsule_name_for(run_capsule_dir), index)
    rep = rtl_checks.screen(trace, capsule, rtl_facts)
    if write:
        (run_capsule_dir / "rtl_checks.json").write_text(
            json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
    return rep


def iter_run_capsule_dirs(root: Path):
    """Yield every capsule run dir under ``root`` (a single run dir, a run's bench dir, or a
    runs/ tree). A capsule run dir is identified by containing a ``generated/`` subdir."""
    if (root / "generated" / "instruction_trace.json").is_file():
        yield root
        return
    for trace in root.rglob("generated/instruction_trace.json"):
        yield trace.parent.parent


def screen_tree(root: Path, write: bool = True) -> list[tuple[str, rtl_checks.CheckReport]]:
    facts = load_rtl_facts()
    index = build_capsule_index()
    out: list[tuple[str, rtl_checks.CheckReport]] = []
    for d in sorted(iter_run_capsule_dirs(root)):
        rep = screen_run(d, facts, index, write=write)
        if rep is not None:
            out.append((d.name, rep))
    return out


# ----------------------------------------------------------------------------- opt-in cost prescreen
def prescreen(run_capsule_dir: Path) -> rtl_checks.CheckReport | None:
    """Opt-in helper for a mining/codegen driver deciding whether to pay for an oracle run.

    Returns a CheckReport; the caller MAY skip L2..L5 when ``rep.verdict == 'reject'``. Does not
    write anything and never consults the frozen runner — the cost saving is purely the caller's
    choice not to invoke the oracle."""
    return screen_run(run_capsule_dir, load_rtl_facts(), build_capsule_index(), write=False)


# ------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="a capsule run dir, a run's bench dir, or a runs/ tree")
    ap.add_argument("--write", action="store_true",
                    help="write rtl_checks.json beside each capsule_result.json (default: dry-run)")
    ap.add_argument("--quiet", action="store_true", help="summary line only")
    a = ap.parse_args(argv)
    reps = screen_tree(Path(a.root), write=a.write)
    if not reps:
        print("no decoded traces found under", a.root)
        return 1
    n_rej = sum(1 for _, r in reps if r.verdict == "reject")
    n_warn = sum(1 for _, r in reps if r.verdict == "warn")
    if not a.quiet:
        for name, r in reps:
            mark = {"reject": "REJECT", "warn": "warn  ", "ok": "ok    "}[r.verdict]
            fails = ", ".join(c.id for c in r.checks if c.status == "fail") or "-"
            print(f"  {mark} {name:34s} err={r.n_error} warn={r.n_warn}  fails: {fails}")
    verb = "wrote" if a.write else "screened (dry-run)"
    print(f"{verb} {len(reps)} rtl_checks reports — {n_rej} reject, {n_warn} warn, "
          f"{len(reps) - n_rej - n_warn} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
