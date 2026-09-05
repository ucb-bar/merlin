#!/usr/bin/env python3
"""Measure whether any cheap signal orders schedules the way the timing oracle does.

WHY THIS SCRIPT EXISTS. Phase 2 wants to eliminate a bad candidate without paying for a cycle-accurate
measurement. Every cheap signal proposed for that job is a HYPOTHESIS about ordering, and this repo has
now caught three that looked reasonable and were not: the correctness simulator (46.1%), a per-command
cost model accurate to 8.1% on magnitude (39.3%), and a tile-pressure heuristic that scored 0.804
overall while every decided pair came from one family and a workload INSIDE that family scored 0.486.
A signal below chance is not weak -- it points the wrong way, and an agent told to use it will follow
it. So a signal earns exposure by measurement here, or it stays in the receipt as recorded evidence.

WHAT IT MEASURES, AND ON WHAT. The comparison the search actually makes: two candidate programs for
the SAME workload, which is faster. Labels come from runs already on disk -- every capsule-bench run
directory that holds both a decoded command trace and a console carrying the oracle's cycle count is
one labelled program, at no simulator cost. Programs are deduplicated by the CONTENT of their command
stream, not by the file: a trace records the path it was decoded from, so hashing the file counts one
program measured twenty times as twenty programs and inflates every count by that factor.

WHAT IT DOES NOT DO. It does not fit a signal and report the fitted number. Any parameter a signal
needs is fitted on one half of the workloads and every rate reported is measured on the other half,
because a signal fitted and scored on the same programs is scored on what it memorised.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.common import paths                                        # noqa: E402
from merlin.perf import barrier_arms, depgraph, rank_validation, schedule_pressure  # noqa: E402
from merlin.perf.deps import rocc                                      # noqa: E402

#: The console line the oracle writes its cycle count on: a metric keyword, a metric name, a value.
METRIC_KEYWORD = "METRIC"
CYCLE_METRIC = "cycles"
TRACE_NAME = "instruction_trace.json"
CONSOLE_NAME = "rtl_gsim_console.log"

#: Acceptance, applied to every signal alike. Chance itself lives in :mod:`merlin.perf.rank_validation`.
MINIMUM_RATE = 0.60
MINIMUM_DECIDED = 40
MINIMUM_SLICE_DECIDED = 20
MINIMUM_SLICES = 2


def _oracle_cycles(console: Path) -> int | None:
    """The oracle's cycle count for one run, or None when its console does not carry one."""
    for line in console.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == METRIC_KEYWORD and parts[1] == CYCLE_METRIC:
            try:
                return int(parts[2])
            except ValueError:
                return None
    return None


def _program_digest(trace: Mapping[str, Any]) -> str:
    """Identity of the COMMAND STREAM, independent of where it was decoded from.

    A trace carries the path it came from, so a file digest makes one program measured many times
    look like many programs and multiplies every count below by the number of times it was re-run.
    """
    body = [[row.get("class"), row.get("funct"), row.get("decoded"), row.get("rs1"), row.get("rs2")]
            for row in (trace.get("instructions") or [])]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _workload_of(run_dir: Path, prefix: str) -> str:
    """The capsule a run directory belongs to, from the run name the harness built it with."""
    name = run_dir.name
    if name.startswith(prefix):
        name = name[len(prefix):]
    for suffix in ("_baseline", "_candidate"):
        cut = name.find(suffix)
        if cut >= 0:
            return name[:cut]
    return name


def harvest(roots: "Sequence[Path]", *, prefix: str) -> dict[str, dict[str, Any]]:
    """``{program digest: record}`` for every run under ``roots`` that carries a trace AND a cycle count.

    A program measured more than once must measure the SAME; a disagreement is recorded on the
    record rather than averaged away, because two different times for one program means the label is
    not a property of the program and every rate computed from it is unsound.
    """
    found: dict[str, dict[str, Any]] = {}
    for root in roots:
        for console in root.rglob(CONSOLE_NAME):
            run_dir = console.parent.parent
            trace_path = run_dir / "generated" / TRACE_NAME
            if not trace_path.exists():
                continue
            cycles = _oracle_cycles(console)
            if cycles is None:
                continue
            trace = json.loads(trace_path.read_text())
            digest = _program_digest(trace)
            record = found.setdefault(digest, {
                "digest": digest, "workload": _workload_of(run_dir, prefix),
                "trace": str(trace_path), "measured": set(), "runs": 0})
            record["measured"].add(cycles)
            record["runs"] += 1
    return found


def _family_of(workload: str) -> str:
    """The coarser slice a workload belongs to: its corpus family, which the capsule id leads with."""
    head = workload.split("_", 1)[0]
    return "".join(c for c in head if not c.isdigit()) or head


# ---------------------------------------------------------------------------------------------------
# the signals under test -- every one scored "smaller is better"
# ---------------------------------------------------------------------------------------------------
def _graph(trace: Mapping[str, Any], *, masks, roles, issue_cycles: float, separation: float):
    program = rocc.program_from_trace(trace, flag_masks=masks, roles=roles)
    issue = depgraph.IssueModel(issue_cycles=issue_cycles, stall_unit=issue_cycles,
                                tier="fitted", provenance="fitted on the training half")
    dag = depgraph.build_dag(program.instructions, program.effects, issue=issue, stall_mnemonic="",
                             roles=program.roles,
                             resolved_separations={"separation.accelerator": separation})
    return program, dag


def score_signals(records: "Sequence[Mapping[str, Any]]", *, target: str,
                  issue_cycles: float, separation: float) -> dict[str, dict[str, float]]:
    """``{signal: {program digest: score}}`` for every signal, over every record it can read."""
    masks, roles = rocc.flag_masks_for(target), rocc.roles_for(target)
    out: dict[str, dict[str, float]] = {name: {} for name in (
        "command_count", "depgraph_makespan", "depgraph_critical_path", "tile_pressure",
        "barrier_count")}
    for record in records:
        trace = json.loads(Path(record["trace"]).read_text())
        digest = record["digest"]
        rows = trace.get("instructions") or []
        out["command_count"][digest] = float(len(rows))
        program, dag = _graph(trace, masks=masks, roles=roles, issue_cycles=issue_cycles,
                              separation=separation)
        order = list(range(len(program.instructions)))
        if order:
            out["depgraph_makespan"][digest] = depgraph.makespan(dag, order)
            out["depgraph_critical_path"][digest] = depgraph.critical_path(dag).cycles
        pressure = schedule_pressure.peak_live_tiles(rows)
        if pressure.get("status") == "counted":
            out["tile_pressure"][digest] = float(pressure["peak_live_tiles"])
        barriers = barrier_arms.count_barriers({"commands": [
            {"opcode": str(r.get("class") or "")} for r in rows if isinstance(r, Mapping)]})
        if isinstance(barriers.get("barriers"), int):
            out["barrier_count"][digest] = float(barriers["barriers"])
    return out


def fit_graph_parameters(train: "Sequence[Mapping[str, Any]]", *, target: str,
                         issue_grid: "Sequence[float]", separation_grid: "Sequence[float]"
                         ) -> dict[str, Any]:
    """Pick the graph's two free parameters on the TRAINING workloads only.

    The sequencer's per-command cost and the one unpriced separation class are the graph's only free
    quantities. Neither is derivable from anything on disk here -- pricing the separation properly
    needs a per-cycle program counter, which nothing in this repo currently emits -- so they are
    FITTED, and fitting them is exactly why every rate this script reports comes from the other half.
    """
    programs = [rank_validation.Program(workload=r["workload"], program=r["digest"],
                                        measured=float(sorted(r["measured"])[0]),
                                        group=_family_of(r["workload"])) for r in train]
    pairs = rank_validation.ordered_pairs(programs)
    best: dict[str, Any] | None = None
    for issue_cycles in issue_grid:
        for separation in separation_grid:
            scored = score_signals(train, target=target, issue_cycles=issue_cycles,
                                   separation=separation)["depgraph_makespan"]
            got = rank_validation.agreement(pairs, scored)
            rate = got.rate
            if rate is None:
                continue
            if best is None or rate > best["train_rate"]:
                best = {"issue_cycles": issue_cycles, "separation": separation,
                        "train_rate": rate, "train_decided": got.decided}
    return best or {"issue_cycles": issue_grid[0], "separation": separation_grid[0],
                    "train_rate": None, "train_decided": 0}


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, help="the target whose ISA facts the traces decode against")
    ap.add_argument("--root", action="append", required=True, type=Path,
                    help="a directory to harvest labelled runs from; repeatable")
    ap.add_argument("--run-prefix", default="", help="the prefix the harness names run directories with")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="separation a signal must show before it is taken to have an opinion")
    ap.add_argument("--margin-sweep", type=float, nargs="*", default=(0.0, 1.0, 2.0, 3.0),
                    help="margins to report alongside the headline, so a rate that rose only "
                         "because a contradicting slice went silent is visible")
    args = ap.parse_args(argv)

    found = harvest(args.root, prefix=args.run_prefix)
    inconsistent = {d: sorted(r["measured"]) for d, r in found.items() if len(r["measured"]) > 1}
    records = [dict(r, measured=sorted(r["measured"])) for r in found.values()]
    if not records:
        print("no labelled program found: a run needs both a decoded trace and a console cycle count")
        return 1

    workloads = sorted({r["workload"] for r in records})
    train_names = {w for i, w in enumerate(workloads) if i % 2 == 0}
    train = [r for r in records if r["workload"] in train_names]
    test = [r for r in records if r["workload"] not in train_names]

    fitted = fit_graph_parameters(train, target=args.target, issue_grid=(1.0, 2.0, 4.0, 8.0),
                                  separation_grid=(0.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0))
    scored = score_signals(test, target=args.target, issue_cycles=fitted["issue_cycles"],
                           separation=fitted["separation"])
    programs = [rank_validation.Program(workload=r["workload"], program=r["digest"],
                                        measured=float(r["measured"][0]),
                                        group=_family_of(r["workload"])) for r in test]
    pairs = rank_validation.ordered_pairs(programs)

    signals: dict[str, Any] = {}
    for name, score in sorted(scored.items()):
        overall = rank_validation.agreement(pairs, score, margin=args.margin)
        by_workload = rank_validation.held_out(programs, score, by="workload", margin=args.margin)
        by_family = rank_validation.held_out(programs, score, by="group", margin=args.margin)
        signals[name] = {
            "verdict": rank_validation.verdict(
                overall, by_workload, minimum_rate=MINIMUM_RATE, minimum_decided=MINIMUM_DECIDED,
                minimum_slice_decided=MINIMUM_SLICE_DECIDED, minimum_slices=MINIMUM_SLICES),
            "by_family": {n: a.to_dict() for n, a in by_family.items()},
            "scored_programs": len(score),
        }

    # A HIGHER MARGIN CAN RAISE A RATE BY SILENCING ITS COUNTER-EVIDENCE, and the headline alone
    # cannot show it. Measured here: tile pressure reads 0.728 at margin 0 with two qualifying
    # slices, one of them at 0.273 -- pointing BACKWARDS -- and 0.901 at margin 3 with that slice
    # gone entirely (33 decided pairs -> 0). The signal did not improve; the workload that
    # contradicted it stopped being counted. So every margin is reported, with the decided count and
    # the qualifying slices at each, and a rate is read together with what it stopped deciding.
    sweep: dict[str, Any] = {}
    for margin in sorted(set(float(m) for m in args.margin_sweep)):
        row: dict[str, Any] = {}
        for name, score in sorted(scored.items()):
            got = rank_validation.agreement(pairs, score, margin=margin)
            per_slice = rank_validation.held_out(programs, score, by="workload", margin=margin)
            row[name] = {
                "decided": got.decided, "rate": got.rate,
                "qualifying_slices": sorted(n for n, a in per_slice.items()
                                            if a.decided >= MINIMUM_SLICE_DECIDED),
                "slices_below_chance": sorted(
                    n for n, a in per_slice.items()
                    if a.rate is not None and a.rate < rank_validation.CHANCE and a.decided),
            }
        sweep[f"{margin:g}"] = row

    report = {
        "schema_version": 1,
        "target": args.target,
        "roots": [str(p) for p in args.root],
        "labelled": {
            "programs": len(records), "workloads": len(workloads),
            "families": sorted({_family_of(w) for w in workloads}),
            "runs_harvested": sum(r["runs"] for r in found.values()),
            "programs_measured_more_than_once": sum(1 for r in found.values() if r["runs"] > 1),
            "programs_with_disagreeing_labels": inconsistent,
        },
        "split": {"train_workloads": sorted(train_names),
                  "test_workloads": sorted(w for w in workloads if w not in train_names),
                  "train_programs": len(train), "test_programs": len(test),
                  "held_out_pairs": len(pairs)},
        "fitted_on_train_only": fitted,
        "thresholds": {"minimum_rate": MINIMUM_RATE, "minimum_decided": MINIMUM_DECIDED,
                       "minimum_slice_decided": MINIMUM_SLICE_DECIDED,
                       "minimum_slices": MINIMUM_SLICES, "chance": rank_validation.CHANCE,
                       "margin": args.margin},
        "signals": signals,
        "margin_sweep": sweep,
        "exposable": sorted(n for n, s in signals.items() if s["verdict"]["exposable"]),
    }
    out = args.out or (paths.artifacts_dir() / "perf-bench" / args.target /
                       "ordering_signal_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"labelled {len(records)} distinct program(s) over {len(workloads)} workload(s) "
          f"from {report['labelled']['runs_harvested']} run(s)")
    print(f"held out {len(pairs)} within-workload ordered pair(s) on "
          f"{len(report['split']['test_workloads'])} workload(s)")
    for name, entry in sorted(signals.items()):
        overall = entry["verdict"]["overall"]
        rate = overall.get("rate")
        print(f"  {name:24s} {overall['agreed']:5d}/{overall['decided']:5d} decided = "
              f"{(f'{rate:.3f}' if rate is not None else '   n/a')}  "
              f"{'EXPOSABLE' if entry['verdict']['exposable'] else 'refused'}")
        if not entry["verdict"]["exposable"]:
            for reason in entry["verdict"]["reasons"]:
                print(f"      - {reason}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
