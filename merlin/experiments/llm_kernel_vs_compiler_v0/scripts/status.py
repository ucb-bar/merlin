#!/usr/bin/env python3
"""Derive the study's current state from what is on disk, so the register cannot quietly go stale.

WHY THIS EXISTS. `TASKS.md` is the plan of record and is hand-maintained, which means it drifts: this
week three tasks it called open had already landed, and two it called done were defects. So the plan
lives in the register, but every NUMBER in this report is read back from the run dirs, the matrix
files and the test tree at the moment it is asked. A row that disagrees with the register is a signal
that one of them is wrong, and it is printed rather than reconciled.

Nothing here re-runs an agent or an oracle; it is a read-only view.

    scripts/status.py                     # the whole board
    scripts/status.py --json out.json     # machine-readable, for the dashboard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = Path("/scratch/agustin/tmp/kvc-runs")

# The register's own vocabulary. Anything else in a state cell is reported as unrecognised rather
# than silently bucketed as open -- a typo must not read as progress.
STATES = ("DONE", "PARTIAL", "RUNNING", "OPEN")

#: Runs invalidated by a harness defect, keyed (matrix tag, method) -> why. They are NOT deleted --
#: a voided run still consumed tokens and money, and hiding it would understate what the study cost.
#: They are excluded from solved-rate and cycle claims, and reported on their own line.
VOID = {
    ("s3", "codex_kernel"):
        "codex ran its own bwrap inside ours; its shell died and it wrote nothing (fixed in 6cd2ee0d)",
    ("s3", "bedrock_kernel"):
        "same round as the voided codex cells; superseded by s4 under the fixed sandbox",
    ("smoke", "codex_kernel"): "pre-sandbox-fix smoke",
    ("smoke", "bedrock_kernel"): "pre-sandbox-fix smoke",
    ("smoke", "gemini_kernel"): "single-seed smoke, not a claim-bearing run",
}


# -- the register ---------------------------------------------------------------------------------

def read_register(tasks_md: Path) -> list[dict]:
    """Parse the task tables out of TASKS.md structurally (split on the table delimiter, no regex)."""
    tasks: list[dict] = []
    phase = ""
    for line in tasks_md.read_text().splitlines():
        s = line.strip()
        if s.startswith("## "):
            phase = s[3:].strip()
            continue
        if not s.startswith("|") or not s.endswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) != 3:
            continue
        tid, task, state_cell = cells
        if tid in ("id", "---") or set(tid) <= set("- :"):
            continue
        state = "UNRECOGNISED"
        for candidate in STATES:
            if candidate in state_cell:
                state = candidate
                break
        note = state_cell.replace("**", "")
        if "—" in note:
            note = note.split("—", 1)[1].strip()
        elif note.strip() == state:
            note = ""
        tasks.append({"id": tid, "phase": phase, "task": task, "state": state, "note": note})
    return tasks


# -- the measured side ----------------------------------------------------------------------------

def read_matrices(runs_root: Path) -> dict:
    """Roll up every matrix file. Failures stay in the distribution; they cost tokens too."""
    by_method: dict[str, dict] = {}
    voided: dict[str, dict] = {}
    matrices: list[dict] = []
    for path in sorted(runs_root.glob("matrix_*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            matrices.append({"tag": path.stem, "error": str(exc)})
            continue
        results = doc.get("results") or []
        matrices.append({
            "tag": path.stem.split("matrix_", 1)[-1],
            "path": str(path),
            "jobs": doc.get("jobs"),
            "wall_seconds": doc.get("wall_seconds"),
            "solved": sum(1 for r in results if r.get("solved")),
            "runs": len(results),
        })
        tag = path.stem.split("matrix_", 1)[-1]
        for r in results:
            cost = r.get("cost") or {}
            reason = VOID.get((tag, r.get("method", "?")))
            sink = voided if reason else by_method
            acc = sink.setdefault(r.get("method", "?"), {
                "runs": 0, "solved": 0, "tokens": 0, "billed_usd": 0.0, "notional_usd": 0.0,
                "agent_seconds": 0.0, "eval_seconds": 0.0, "unpriced_rounds": 0,
                "models": set(), "cycles": {}, "void_reasons": set(),
            })
            if reason:
                acc["void_reasons"].add(f"{tag}: {reason}")
            acc["runs"] += 1
            acc["solved"] += 1 if r.get("solved") else 0
            acc["tokens"] += cost.get("tokens_total") or 0
            acc["billed_usd"] += cost.get("billed_usd") or 0.0
            acc["notional_usd"] += cost.get("notional_usd") or 0.0
            acc["agent_seconds"] += cost.get("agent_seconds") or 0.0
            acc["eval_seconds"] += cost.get("eval_seconds") or 0.0
            acc["unpriced_rounds"] += cost.get("rounds_unpriced") or 0
            if r.get("model"):
                acc["models"].add(r["model"])
            if r.get("best_cycles") is not None and r.get("solved"):
                acc["cycles"].setdefault(r.get("task", "?"), []).append(r["best_cycles"])
    for acc in list(by_method.values()) + list(voided.values()):
        acc["models"] = sorted(acc["models"])
        acc["void_reasons"] = sorted(acc["void_reasons"])
    return {"matrices": matrices, "by_method": by_method, "voided": voided}


def read_baselines(runs_root: Path) -> dict:
    """The reference numbers every arm is relative to, or an explicit absence."""
    path = runs_root / "baselines.json"
    if not path.is_file():
        return {"available": False, "reason": f"no {path.name}; run scripts/baseline.py"}
    doc = json.loads(path.read_text())
    doc["available"] = True
    return doc


def performance(measured: dict, baselines: dict) -> dict:
    """Reference-relative speedup per task and arm, plus the geometric mean over tasks.

    Two rules the numbers depend on, both of which have been got wrong on this project before:
    an INCORRECT kernel earns no performance credit and is absent here rather than scoring 1.0; and a
    speedup may only compare counts from the SAME fidelity, because the functional model's estimate
    and the cycle-accurate measurement are different quantities wearing the same unit.
    """
    if not baselines.get("available"):
        return {"available": False, "reason": baselines.get("reason")}
    base = {t: v.get("cycles") for t, v in baselines.get("tasks", {}).items()
            if v.get("cycles") is not None}
    per_arm: dict[str, dict] = {}
    for arm, runs in measured.items():
        rows = {}
        for task, cycles in sorted(runs.items()):
            if task not in base or not cycles:
                continue
            best, med = min(cycles), sorted(cycles)[len(cycles) // 2]
            rows[task] = {
                "baseline_cycles": base[task], "n": len(cycles),
                "best_cycles": best, "median_cycles": med,
                "speedup_best": base[task] / best, "speedup_median": base[task] / med,
                "seed_spread": max(cycles) / min(cycles),
            }
        if rows:
            product = 1.0
            for r in rows.values():
                product *= r["speedup_median"]
            per_arm[arm] = {"tasks": rows, "geomean_speedup": product ** (1.0 / len(rows)),
                            "tasks_scored": len(rows)}
    return {"available": True, "fidelity": baselines.get("fidelity"),
            "cycle_accurate": baselines.get("fidelity") == "cert",
            "baseline_label": baselines.get("label"), "by_arm": per_arm}


def read_tests(repo: Path) -> dict:
    """Count the study's own tests without importing them (a broken import must not read as zero)."""
    files = {
        "test_evaluation_schema.py", "test_kvc_kernel_misroute.py", "test_kvc_inventory.py",
        "test_kvc_method_routing.py", "test_agent_transcript_tokens.py",
        "test_task_basis.py", "test_kernel_library.py",
    }
    out = {"files": 0, "cases": 0, "missing": []}
    for name in sorted(files):
        path = repo / "merlin" / "tests" / "infra" / name
        if not path.is_file():
            out["missing"].append(name)
            continue
        out["files"] += 1
        out["cases"] += sum(1 for ln in path.read_text().splitlines()
                            if ln.startswith("def test_"))
    return out


def critical_path(tasks: list[dict]) -> list[dict]:
    """The ordered spine from TASKS.md's own critical-path section, resolved to live states."""
    spine = ["1.3", "3.3", "3.5", "1.2", "4.9", "4.10", "4.13", "5.4", "5.6", "5.7"]
    by_id = {t["id"]: t for t in tasks}
    return [by_id[i] for i in spine if i in by_id]


def build(runs_root: Path, repo: Path) -> dict:
    tasks = read_register(EXP / "TASKS.md")
    baselines = read_baselines(runs_root)
    counts = {s: sum(1 for t in tasks if t["state"] == s) for s in STATES}
    counts["UNRECOGNISED"] = sum(1 for t in tasks if t["state"] == "UNRECOGNISED")
    measured = read_matrices(runs_root)
    return {
        "experiment": EXP.name,
        "runs_root": str(runs_root),
        "tasks": tasks,
        "counts": counts,
        "critical_path": critical_path(tasks),
        "measured": measured,
        "baselines": baselines,
        "performance": performance({k: v["cycles"] for k, v in measured["by_method"].items()},
                                   baselines),
        "tests": read_tests(repo),
    }


def render(state: dict) -> str:
    lines: list[str] = []
    c = state["counts"]
    total = sum(c.values())
    lines.append(f"REGISTER  {total} tasks  "
                 + "  ".join(f"{k} {c[k]}" for k in ("DONE", "PARTIAL", "RUNNING", "OPEN")
                             if c.get(k)))
    if c.get("UNRECOGNISED"):
        lines.append(f"  !! {c['UNRECOGNISED']} rows have no recognised state")
    lines.append("")
    lines.append("CRITICAL PATH (blocking order)")
    for t in state["critical_path"]:
        lines.append(f"  {t['state']:8s} {t['id']:5s} {t['task']}")
    lines.append("")
    lines.append("MEASURED (read back from the run dirs, not from the register)")
    for name, m in sorted(state["measured"]["by_method"].items()):
        lines.append(f"  {name:16s} {m['solved']:2d}/{m['runs']:<2d} solved   "
                     f"{m['tokens']:>12,} tok   billed ${m['billed_usd']:.2f}   "
                     f"notional ${m['notional_usd']:.2f}   agent {m['agent_seconds']/3600:.2f} h")
        if m["unpriced_rounds"]:
            lines.append(f"  {'':16s} !! {m['unpriced_rounds']} unpriced rounds "
                         f"(recorded UNPRICED, never $0)")
    lines.append("")
    voided = state["measured"]["voided"]
    if voided:
        lines.append("")
        lines.append("VOIDED by a harness defect -- excluded from every rate, but still counted as spend")
        for name, m in sorted(voided.items()):
            lines.append(f"  {name:16s} {m['runs']:2d} runs   {m['tokens']:>12,} tok   "
                         f"billed ${m['billed_usd']:.2f}   notional ${m['notional_usd']:.2f}")
            for why in m["void_reasons"]:
                lines.append(f"  {'':16s} - {why}")
    lines.append("")
    for mx in state["measured"]["matrices"]:
        if "error" in mx:
            lines.append(f"  matrix {mx['tag']}: UNREADABLE ({mx['error']})")
        else:
            lines.append(f"  matrix {mx['tag']:6s} {mx['solved']:2d}/{mx['runs']:<2d} solved   "
                         f"{(mx['wall_seconds'] or 0)/60:.0f} min wall")
    perf = state["performance"]
    lines.append("")
    if not perf.get("available"):
        lines.append(f"PERFORMANCE  unavailable -- {perf.get('reason')}")
    else:
        est = "" if perf["cycle_accurate"] else "  (ESTIMATE: functional model, not cycle-accurate)"
        lines.append(f"PERFORMANCE vs {perf['baseline_label']} at fidelity "
                     f"{perf['fidelity']}{est}")
        for arm, a in sorted(perf["by_arm"].items()):
            lines.append(f"  {arm:16s} geomean {a['geomean_speedup']:.3f}x "
                         f"over {a['tasks_scored']} task(s)")
            for task, r in sorted(a["tasks"].items()):
                lines.append(f"  {'':16s}  {task:22s} base {r['baseline_cycles']:>8,}  "
                             f"med {r['median_cycles']:>8,} ({r['speedup_median']:.3f}x)  "
                             f"best {r['best_cycles']:>8,} ({r['speedup_best']:.3f}x)  "
                             f"seed spread {r['seed_spread']:.3f}x")
    t = state["tests"]
    lines.append("")
    lines.append(f"TESTS     {t['cases']} cases across {t['files']} study files"
                 + (f"   !! missing {', '.join(t['missing'])}" if t["missing"] else ""))
    return "\n".join(lines)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--repo", type=Path, default=EXP.parents[2])
    ap.add_argument("--json", type=Path, help="also write the machine-readable state here")
    args = ap.parse_args(argv)

    state = build(args.runs_root, args.repo)
    print(render(state))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(state, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
