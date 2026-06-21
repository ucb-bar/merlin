"""Resumable cartesian conformance dispatcher (abc-testing-style).

Sweeps (rung x oracle), records each run via the aet substrate, and appends to a jsonl
ledger keyed by run_id. Reruns skip rows already marked correct (resumable across the
hours-long Verilator/FireSim sweeps), so a sweep can be killed and resumed. ``result_fn`` is
an injection seam (tests pass a fake to exercise the cartesian/ledger/resume logic without a
simulator).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .gemmini_suite import record_gemmini_run


def _load_ledger(ledger_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[rec["run_id"]] = rec
    return done


def run_sweep(rungs: list[str], simulators: list[str], *, runs_root: str | Path,
              ledger_path: str | Path, backends: list[str] | None = None, force: bool = False,
              result_fn: Callable[..., dict] | None = None,
              timeout: int = 600) -> list[dict[str, Any]]:
    """Run (or resume) the (rung x simulator x codegen_backend) sweep; one row per cell.

    ``backends`` is the codegen selector (NOT the oracle): subset of
    {"mlir_inline_asm", "legacy_c"}. The run_id encodes all three, so cells are isolated and
    the resumable ledger skips already-passing cells, reruns failed/missing cells only."""
    runs_root = Path(runs_root)
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    backends = backends or ["mlir_inline_asm_rocc"]
    done = _load_ledger(ledger_path)
    rows: list[dict] = []
    for rung in rungs:
        for sim in simulators:
            for backend in backends:
                run_id = f"{rung}_{sim}_{backend}_seed000"
                prior = done.get(run_id)
                if prior and prior.get("correct") and not force:
                    rows.append({**prior, "skipped": True})
                    continue
                result = result_fn(rung, sim, backend) if result_fn is not None else None
                summ = record_gemmini_run(rung, sim, runs_root=runs_root, codegen_backend=backend,
                                          result=result, timeout=timeout)
                rec = {"run_id": run_id, "rung": rung, "simulator": sim, "codegen_backend": backend,
                       "correct": bool(summ["correct"]), "oracle": summ["oracle"],
                       "cycle_accurate": summ.get("cycle_accurate"),
                       "cycles": summ["metrics"].get("cycles"), "run_path": summ["run_path"]}
                with ledger_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                done[run_id] = rec
                rows.append({**rec, "skipped": False})
    return rows


def summarize(rows: list[dict]) -> str:
    """Compact table from in-memory sweep rows (convenience; the canonical, provenance-bearing
    table is ``summarize_from_manifests``, derived from recorded manifests)."""
    cols = ("rung", "backend", "oracle", "rtl?", "correct", "cycles", "skipped")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        o = r.get("oracle", {})
        lines.append(f"| {r['rung']} | {r.get('codegen_backend','?')} | {o.get('kind','?')} | "
                     f"{o.get('derived_from_rtl','?')} | {r['correct']} | "
                     f"{r.get('cycles','?')} | {r.get('skipped', False)} |")
    return "\n".join(lines)


def summarize_from_manifests(runs_root: str | Path) -> str:
    """Render the headline table STRICTLY from recorded run manifests + artifact manifests.

    Acceptance test: the table is reproducible from the ledger alone, NO hand-authored YAML.
    Reads each runs/.../run_manifest.yaml + artifact_manifest.json."""
    import yaml
    runs_root = Path(runs_root)
    cols = ("rung", "backend", "oracle.kind", "derived_from_rtl", "cycle_accurate",
            "correct", "cycles", "run_id", "artifacts")
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    rows = []
    for man in sorted(runs_root.rglob("run_manifest.yaml")):
        m = yaml.safe_load(man.read_text())
        meta = m.get("metadata", {})
        oracle = meta.get("oracle", {})
        art_path = man.parent / "artifact_manifest.json"
        arts = []
        if art_path.exists():
            for a in json.loads(art_path.read_text()).get("artifacts", []):
                arts.append(f"{a.get('kind')}:{a.get('origin')}")
        rows.append((m.get("benchmark"), m.get("codegen_backend"), oracle.get("kind"),
                     oracle.get("derived_from_rtl"), meta.get("cycle_accurate"),
                     m.get("status") == "pass", meta.get("cycles"), m.get("run_id"),
                     ", ".join(arts)))
    for r in sorted(rows, key=lambda x: (str(x[1]), str(x[2]), str(x[0]))):
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Gemmini conformance sweep (resumable).")
    ap.add_argument("--rungs", default="C0,Q0")
    ap.add_argument("--simulators", default="spike")
    ap.add_argument("--backends", default="mlir_inline_asm_rocc")
    ap.add_argument("--runs-root", default="runs/gemmini_cert")
    ap.add_argument("--ledger", default="runs/gemmini_cert/ledger.jsonl")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary-only", action="store_true",
                    help="render the table from existing manifests; run nothing")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    if args.summary_only:
        print(summarize_from_manifests(args.runs_root))
        return 0
    rows = run_sweep(args.rungs.split(","), args.simulators.split(","),
                     runs_root=args.runs_root, ledger_path=args.ledger,
                     backends=args.backends.split(","), force=args.force, timeout=args.timeout)
    print(summarize_from_manifests(args.runs_root))
    return 0 if all(r["correct"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
