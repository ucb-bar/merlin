"""Derive a comparison study's state from what is on disk, and score its arms against a reference.

WHY THIS EXISTS. A study keeps a hand-maintained task register (markdown tables of
``| id | task | state |``) as its plan of record, and a hand-maintained register drifts: in the
kernel-vs-compiler study three rows called work open that had already landed, and two it called done
were defects. So the plan stays in the register, but every NUMBER this module reports is read back
from the study's run matrices and its baseline record at the moment it is asked. A row that
disagrees with the measured side is printed rather than reconciled.

Two scoring rules the numbers depend on, both of which have been got wrong before:

* An arm's cycles mean nothing without a reference. Absolute counts cannot show that a metric has
  stopped discriminating: three kernels landing within 0.1% of each other reads as agreement rather
  than as a resolution limit. Every arm is therefore reported as a speedup over a baseline record,
  produced by the same grader, oracle and fidelity as the arms.
* A speedup compares counts from the SAME fidelity, and an incorrect result earns no performance
  credit. A functional model's estimate and a cycle-accurate measurement are different quantities
  wearing the same unit, so a result whose fidelity differs from the baseline's is set aside with its
  reason instead of being divided.

Runs invalidated by a harness defect are named in a reviewed void list. They stay out of every rate
but are still reported as spend: a voided run consumed tokens, and hiding it understates the cost.

Nothing here runs an agent, a grader or a simulator; it is a read-only view. It knows no target:
the baseline record names its own target, package, fidelity and certifying tiers.

    merlin experiment study board --register TASKS.md --runs-root <runs> [--void void.yaml] [--json out.json]
    merlin experiment study baseline --label ref --package <pkg> --target <t> --fidelity fast \\
        --certifying-tier L2 --out <runs>/baselines.json TASK=<capsule_result.json> ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

#: The register's own state vocabulary. Anything else in a state cell is UNRECOGNISED rather than
#: silently bucketed as open: a typo must not read as progress.
STATES = ("DONE", "PARTIAL", "RUNNING", "OPEN")
UNRECOGNISED = "UNRECOGNISED"

#: The heading that opens a register's ordered critical path; the task ids it names, in order, are
#: the spine the board resolves to live states.
CRITICAL_PATH_HEADING = "## critical path"


# -- the register ---------------------------------------------------------------------------------


def read_register(text: str) -> list[dict[str, str]]:
    """Every task row of a register, parsed structurally (table cells, no pattern matching)."""
    tasks: list[dict[str, str]] = []
    phase = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            phase = stripped[3:].strip()
            continue
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 3:
            continue
        task_id, task, state_cell = cells
        if task_id == "id" or set(task_id) <= set("-: "):
            continue
        state = next((name for name in STATES if name in state_cell), UNRECOGNISED)
        note = state_cell.replace("**", "")
        head, dash, tail = note.partition("—")
        note = tail.strip() if dash else ("" if head.strip() == state else head.strip())
        tasks.append({"id": task_id, "phase": phase, "task": task, "state": state, "note": note})
    return tasks


def _task_ids(text: str) -> list[str]:
    """Tokens of the form ``<int>.<int>``, in order of appearance (an explicit tokenizer)."""
    found: list[str] = []
    token = ""
    for char in text + " ":
        if char.isdigit() or char == ".":
            token += char
            continue
        head, dot, tail = token.strip(".").partition(".")
        if dot and head.isdigit() and tail.isdigit() and token.strip(".") not in found:
            found.append(token.strip("."))
        token = ""
    return found


def critical_path(text: str, tasks: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """The register's own ordered critical path, each id resolved to its live row.

    Read from the register's critical-path section rather than written down here, so the spine moves
    when the plan does. An id the section names but no table row defines is reported as such.
    """
    section: list[str] = []
    inside = False
    for line in text.splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("## "):
            inside = lowered.startswith(CRITICAL_PATH_HEADING)
            continue
        if inside:
            section.append(line)
    by_id = {row["id"]: row for row in tasks}
    return [
        dict(by_id[task_id])
        if task_id in by_id
        else {"id": task_id, "phase": "", "task": "(no row defines it)", "state": UNRECOGNISED, "note": ""}
        for task_id in _task_ids("\n".join(section))
    ]


# -- the measured side ----------------------------------------------------------------------------


def read_void(path: Path | None) -> dict[tuple[str, str], str]:
    """The reviewed void list: ``{matrix tag: {method: reason}}`` -> ``{(tag, method): reason}``."""
    if path is None:
        return {}
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("voided"), dict):
        raise ValueError(f"{path}: a void list maps `voided` to {{matrix tag: {{method: reason}}}}")
    out: dict[tuple[str, str], str] = {}
    for tag, methods in document["voided"].items():
        if not isinstance(methods, dict):
            raise ValueError(f"{path}: voided.{tag} must map methods to reasons")
        for method, reason in methods.items():
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{path}: voided.{tag}.{method} needs a stated reason")
            out[(str(tag), str(method))] = reason.strip()
    return out


def _cost(row: Mapping[str, Any], key: str) -> float:
    value = (row.get("cost") or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def rollup_matrices(
    paths: Iterable[Path], void: Mapping[tuple[str, str], str], *, declared_fidelity: str | None = None
) -> dict[str, Any]:
    """Roll up every matrix file. Failures stay in the distribution, because they cost tokens too.

    A row's fidelity is the one it records. ``declared_fidelity`` is the operator's statement of the
    grader fidelity for rows that record none; it is kept apart (``declared:<value>``) so a reader can
    tell a recorded fidelity from a declared one, and a row without either stays ``UNKNOWN``.
    """
    by_method: dict[str, dict[str, Any]] = {}
    voided: dict[str, dict[str, Any]] = {}
    matrices: list[dict[str, Any]] = []
    for path in sorted(paths):
        tag = path.stem.partition("matrix_")[2] or path.stem
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            matrices.append({"tag": tag, "path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        results = document.get("results")
        if not isinstance(results, list):
            matrices.append({"tag": tag, "path": str(path), "error": "no `results` list"})
            continue
        matrices.append(
            {
                "tag": tag,
                "path": str(path),
                "jobs": document.get("jobs"),
                "wall_seconds": document.get("wall_seconds"),
                "solved": sum(1 for row in results if row.get("solved")),
                "runs": len(results),
            }
        )
        for row in results:
            method = str(row.get("method") or "UNKNOWN")
            reason = void.get((tag, method))
            sink = voided if reason else by_method
            acc = sink.setdefault(
                method,
                {
                    "runs": 0,
                    "solved": 0,
                    "tokens": 0.0,
                    "billed_usd": 0.0,
                    "notional_usd": 0.0,
                    "agent_seconds": 0.0,
                    "unpriced_rounds": 0.0,
                    "models": set(),
                    "cycles": {},
                    "fidelity": {},
                    "void_reasons": set(),
                },
            )
            if reason:
                acc["void_reasons"].add(f"{tag}: {reason}")
            acc["runs"] += 1
            acc["solved"] += 1 if row.get("solved") else 0
            for key, name in (
                ("tokens", "tokens_total"),
                ("billed_usd", "billed_usd"),
                ("notional_usd", "notional_usd"),
                ("agent_seconds", "agent_seconds"),
                ("unpriced_rounds", "rounds_unpriced"),
            ):
                acc[key] += _cost(row, name)
            if row.get("model"):
                acc["models"].add(str(row["model"]))
            # Only a SOLVED run's cycles can be credited; an incorrect kernel's speed is not a result.
            if row.get("solved") and isinstance(row.get("best_cycles"), int):
                task = str(row.get("task") or "UNKNOWN")
                acc["cycles"].setdefault(task, []).append(row["best_cycles"])
                recorded = row.get("fidelity")
                fidelity = (
                    str(recorded) if recorded else f"declared:{declared_fidelity}" if declared_fidelity else "UNKNOWN"
                )
                acc["fidelity"].setdefault(task, set()).add(fidelity)
    for acc in [*by_method.values(), *voided.values()]:
        acc["models"] = sorted(acc["models"])
        acc["void_reasons"] = sorted(acc["void_reasons"])
        acc["fidelity"] = {task: sorted(values) for task, values in acc["fidelity"].items()}
    return {"matrices": matrices, "by_method": by_method, "voided": voided}


# -- the reference --------------------------------------------------------------------------------


def certifying_cycles(result: Mapping[str, Any], certifying_tiers: Iterable[str]) -> tuple[int | None, str | None]:
    """The cycle count of a graded capsule result, taken from a certifying tier ALONE.

    A tier that only proves the kernel ran cannot supply a latency the study may quote. When several
    certifying tiers report cycles, the highest-ranked (last in sorted order) is the one returned.
    """
    allowed = set(certifying_tiers)
    tiers = result.get("tiers") or {}
    chosen: tuple[int | None, str | None] = (None, None)
    for name in sorted(tiers):
        cycles = (tiers[name] or {}).get("cycles")
        if name in allowed and isinstance(cycles, int):
            chosen = (cycles, name)
    return chosen


def baseline_record(
    results: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
    package: str,
    target: str,
    fidelity: str,
    certifying_tiers: Sequence[str],
) -> dict[str, Any]:
    """The reference record every arm is scored against, built from already-graded capsule results.

    The reference package is graded by the same grader, oracle and fidelity as every arm; this only
    records what that grade said. A task whose result does not PASS, or that no certifying tier
    timed, carries ``cycles: None`` and is never divided by.
    """
    if not certifying_tiers:
        raise ValueError("a baseline needs at least one certifying tier")
    tasks: dict[str, dict[str, Any]] = {}
    for task, result in sorted(results.items()):
        cycles, tier = certifying_cycles(result, certifying_tiers)
        passed = result.get("status") == "pass"
        tasks[task] = {
            "status": result.get("status"),
            "cycles": cycles if passed else None,
            "cycles_tier": tier if passed else None,
            "tier_status": {name: (row or {}).get("status") for name, row in (result.get("tiers") or {}).items()},
            "numeric_status": (result.get("numeric") or {}).get("status"),
        }
    return {
        "label": label,
        "package": package,
        "target": target,
        "fidelity": fidelity,
        "certifying_tiers": sorted(certifying_tiers),
        "measured_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "tasks": tasks,
    }


def read_baseline(path: Path) -> dict[str, Any]:
    """The baseline record, or an explicit absence that says how to produce one."""
    if not path.is_file():
        return {"available": False, "reason": f"no {path.name}; record one with `merlin experiment study baseline`"}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("tasks"), dict):
        return {"available": False, "reason": f"{path} is not a baseline record"}
    return {**document, "available": True}


def reference_relative(by_method: Mapping[str, Mapping[str, Any]], baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Reference-relative speedup per task and arm, and each arm's geometric mean over tasks."""
    if not baseline.get("available"):
        return {"available": False, "reason": baseline.get("reason")}
    fidelity = baseline.get("fidelity")
    base = {
        task: row["cycles"]
        for task, row in (baseline.get("tasks") or {}).items()
        if isinstance(row, Mapping) and isinstance(row.get("cycles"), int) and row["cycles"] > 0
    }
    per_arm: dict[str, Any] = {}
    for arm, acc in sorted(by_method.items()):
        rows: dict[str, Any] = {}
        set_aside: dict[str, str] = {}
        for task, cycles in sorted((acc.get("cycles") or {}).items()):
            if task not in base:
                set_aside[task] = "the baseline has no passing, certifying-tier count for this task"
                continue
            seen = (acc.get("fidelity") or {}).get(task, ["UNKNOWN"])
            if [value.removeprefix("declared:") for value in seen] != [fidelity]:
                set_aside[task] = f"arm fidelity {seen} is not the baseline's {fidelity!r}"
                continue
            ordered = sorted(cycles)
            best, median = ordered[0], ordered[len(ordered) // 2]
            rows[task] = {
                "baseline_cycles": base[task],
                "n": len(ordered),
                "best_cycles": best,
                "median_cycles": median,
                "speedup_best": base[task] / best,
                "speedup_median": base[task] / median,
                "seed_spread": ordered[-1] / best,
                "fidelity_source": "declared" if seen[0].startswith("declared:") else "recorded",
            }
        entry: dict[str, Any] = {"tasks": rows, "set_aside": set_aside, "tasks_scored": len(rows)}
        if rows:
            entry["geomean_speedup"] = math.exp(sum(math.log(r["speedup_median"]) for r in rows.values()) / len(rows))
        per_arm[arm] = entry
    return {
        "available": True,
        "baseline_label": baseline.get("label"),
        "target": baseline.get("target"),
        "fidelity": fidelity,
        "certifying_tiers": baseline.get("certifying_tiers"),
        "by_arm": per_arm,
    }


# -- the board ------------------------------------------------------------------------------------


def build(
    register: Path,
    runs_root: Path,
    *,
    void: Path | None = None,
    baseline: Path | None = None,
    declared_fidelity: str | None = None,
) -> dict[str, Any]:
    text = register.read_text(encoding="utf-8")
    tasks = read_register(text)
    counts = {state: sum(1 for row in tasks if row["state"] == state) for state in (*STATES, UNRECOGNISED)}
    measured = rollup_matrices(runs_root.glob("matrix_*.json"), read_void(void), declared_fidelity=declared_fidelity)
    reference = read_baseline(baseline if baseline is not None else runs_root / "baselines.json")
    return {
        "register": str(register),
        "runs_root": str(runs_root),
        "tasks": tasks,
        "counts": counts,
        "critical_path": critical_path(text, tasks),
        "measured": measured,
        "baseline": reference,
        "performance": reference_relative(measured["by_method"], reference),
    }


def render(state: Mapping[str, Any]) -> str:
    lines: list[str] = []
    counts = state["counts"]
    lines.append(
        f"REGISTER  {sum(counts.values())} tasks  "
        + "  ".join(f"{name} {counts[name]}" for name in STATES if counts.get(name))
    )
    if counts.get(UNRECOGNISED):
        lines.append(f"  !! {counts[UNRECOGNISED]} row(s) have no recognised state")
    lines += ["", "CRITICAL PATH (the register's own order)"]
    lines += [f"  {row['state']:12s} {row['id']:6s} {row['task']}" for row in state["critical_path"]]
    lines += ["", "MEASURED (read back from the run matrices, not from the register)"]
    for name, acc in sorted(state["measured"]["by_method"].items()):
        lines.append(
            f"  {name:18s} {acc['solved']:3d}/{acc['runs']:<3d} solved  {int(acc['tokens']):>12,} tok  "
            f"billed ${acc['billed_usd']:.2f}  notional ${acc['notional_usd']:.2f}  "
            f"agent {acc['agent_seconds'] / 3600:.2f} h"
        )
        if acc["unpriced_rounds"]:
            lines.append(f"  {'':18s} !! {int(acc['unpriced_rounds'])} unpriced round(s), recorded UNPRICED, never $0")
    if state["measured"]["voided"]:
        lines += ["", "VOIDED by a harness defect: out of every rate, still counted as spend"]
        for name, acc in sorted(state["measured"]["voided"].items()):
            lines.append(
                f"  {name:18s} {acc['runs']:3d} run(s)  {int(acc['tokens']):>12,} tok  "
                f"billed ${acc['billed_usd']:.2f}  notional ${acc['notional_usd']:.2f}"
            )
            lines += [f"  {'':18s} - {why}" for why in acc["void_reasons"]]
    for matrix in state["measured"]["matrices"]:
        if "error" in matrix:
            lines.append(f"  matrix {matrix['tag']}: UNREADABLE ({matrix['error']})")
        else:
            lines.append(f"  matrix {matrix['tag']:8s} {matrix['solved']:3d}/{matrix['runs']:<3d} solved")
    performance = state["performance"]
    lines.append("")
    if not performance.get("available"):
        lines.append(f"PERFORMANCE  unavailable: {performance.get('reason')}")
    else:
        lines.append(
            f"PERFORMANCE vs {performance['baseline_label']} ({performance['target']}) at fidelity "
            f"{performance['fidelity']}, certifying tiers {performance['certifying_tiers']}"
        )
        for arm, entry in sorted(performance["by_arm"].items()):
            if "geomean_speedup" in entry:
                lines.append(
                    f"  {arm:18s} geomean {entry['geomean_speedup']:.3f}x over {entry['tasks_scored']} task(s)"
                )
            else:
                lines.append(f"  {arm:18s} no task scored")
            for task, row in sorted(entry["tasks"].items()):
                lines.append(
                    f"  {'':18s}  {task:24s} base {row['baseline_cycles']:>9,}  med {row['median_cycles']:>9,} "
                    f"({row['speedup_median']:.3f}x)  best {row['best_cycles']:>9,} ({row['speedup_best']:.3f}x)  "
                    f"seed spread {row['seed_spread']:.3f}x"
                )
            for task, why in sorted(entry["set_aside"].items()):
                lines.append(f"  {'':18s}  {task:24s} SET ASIDE: {why}")
    return "\n".join(lines)


def _board(args: argparse.Namespace) -> int:
    state = build(
        args.register,
        args.runs_root,
        void=args.void,
        baseline=args.baseline,
        declared_fidelity=args.matrix_fidelity,
    )
    print(render(state))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def _baseline(args: argparse.Namespace) -> int:
    results: dict[str, Mapping[str, Any]] = {}
    for spec in args.results:
        task, sep, path = spec.partition("=")
        if not sep or not task or not path:
            print(f"expected TASK=PATH, got {spec!r}", file=sys.stderr)
            return 2
        results[task] = json.loads(Path(path).read_text(encoding="utf-8"))
    record = baseline_record(
        results,
        label=args.label,
        package=args.package,
        target=args.target,
        fidelity=args.fidelity,
        certifying_tiers=args.certifying_tier,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    for task, row in record["tasks"].items():
        print(f"{task:24s} {str(row['status']):8s} cycles={row['cycles']} tier={row['cycles_tier']}")
    print(f"wrote {args.out}")
    return 0 if all(row["cycles"] is not None for row in record["tasks"].values()) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="merlin experiment study", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    board = commands.add_parser("board", help="the register beside what the run matrices measured")
    board.add_argument("--register", type=Path, required=True, help="the study's task register (markdown)")
    board.add_argument("--runs-root", type=Path, required=True, help="directory holding matrix_*.json")
    board.add_argument("--void", type=Path, help="reviewed void list (YAML: voided: {tag: {method: reason}})")
    board.add_argument("--baseline", type=Path, help="baseline record (default: <runs-root>/baselines.json)")
    board.add_argument(
        "--matrix-fidelity",
        help="the grader fidelity of matrix rows that record none (kept as declared, never as recorded)",
    )
    board.add_argument("--json", type=Path, help="also write the machine-readable state here")
    board.set_defaults(handler=_board)
    base = commands.add_parser("baseline", help="record the reference every arm is scored against")
    base.add_argument("--label", required=True, help="what this baseline IS, recorded beside the numbers")
    base.add_argument("--package", required=True, help="the reference package that was graded")
    base.add_argument("--target", required=True)
    base.add_argument("--fidelity", required=True, help="the grader fidelity the results came from")
    base.add_argument("--certifying-tier", action="append", required=True, help="a tier that compared outputs")
    base.add_argument("--out", type=Path, required=True)
    base.add_argument("results", nargs="+", help="TASK=PATH to that task's graded capsule result JSON")
    base.set_defaults(handler=_baseline)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
