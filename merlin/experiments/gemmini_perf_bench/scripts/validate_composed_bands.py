#!/usr/bin/env python3
"""Measure whether a composed cycle BAND actually contains the cycles the oracle measured.

This is the acceptance gate on :mod:`merlin.perf.compose_estimate`, and nothing that module produces
may be shown to an authoring agent until this has been run and reported. The reasoning is the same one
that condemned every cheap ORDERING signal in this tree: a signal is exposed on measured agreement, not
on the plausibility of its derivation.

What is being tested is narrower than "is the estimate good". A band makes exactly one claim -- that
the true cycle count lies between a structural floor and a serial ceiling -- and that claim is false
the moment a measurement falls outside. So the statistic is the CONTAINMENT RATE, with n stated, plus
the two failure directions reported separately: a measurement BELOW the floor means the floor is not a
floor (a mis-derived peak, or work the counter did not see), and one ABOVE the ceiling means the
serial sum is not a ceiling (an event outside the calibrated vocabulary, or overlap the model prices
as free). Those are different defects and averaging them into one rate would hide both.

Labels cost nothing: every capsule-bench run directory that holds both an emitted command buffer and a
console carrying ``METRIC cycles`` is one labelled program. Programs are keyed by the CONTENT of the
command buffer, never by its path -- a run directory differs per run, so hashing the file would count
one program measured twenty times as twenty programs and inflate every count here.

A program measured more than once must measure the SAME. A disagreement is recorded rather than
averaged, because two cycle counts for one program means the label is not a property of the program
and every rate computed from it is unsound.

Usage::

    validate_composed_bands.py --target gemmini --root out/runs/<target>/perf-bench/agent_stages
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))

from merlin.common.paths import artifacts_dir  # noqa: E402
from merlin.perf import compose_estimate as CE  # noqa: E402

METRIC_KEYWORD = "METRIC"
CYCLE_METRIC = "cycles"
BUFFER_NAME = "command_buffer.json"
CONSOLE_NAME = "rtl_gsim_console.log"

#: Written under the perf-bench concern for the target, alongside the ordering-signal verdict.
ARTIFACT_NAME = "composed_band_validation.json"


def _oracle_cycles(console: Path) -> int | None:
    for line in console.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == METRIC_KEYWORD and parts[1] == CYCLE_METRIC:
            try:
                return int(parts[2])
            except ValueError:
                return None
    return None


def _program_digest(buffer: Mapping[str, object]) -> str:
    """Identity of the emitted PROGRAM, independent of where it was written."""
    body = [[row.get("opcode"), row.get("operands"), row.get("attributes")]
            for row in (buffer.get("commands") or []) if isinstance(row, Mapping)]
    payload = {"commands": body, "tensors": buffer.get("tensors")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _workload_of(run_dir: Path, prefix: str) -> str:
    name = run_dir.name
    if name.startswith(prefix):
        name = name[len(prefix):]
    for suffix in ("_baseline", "_candidate"):
        cut = name.find(suffix)
        if cut >= 0:
            return name[:cut]
    return name


def harvest(roots: Sequence[Path], *, prefix: str) -> dict[str, dict]:
    """``{program digest: record}`` for every run carrying both a command buffer and a cycle count."""
    found: dict[str, dict] = {}
    for root in roots:
        for console in root.rglob(CONSOLE_NAME):
            run_dir = console.parent.parent
            buffer_path = run_dir / "generated" / BUFFER_NAME
            if not buffer_path.exists():
                continue
            cycles = _oracle_cycles(console)
            if cycles is None:
                continue
            try:
                buffer = json.loads(buffer_path.read_text())
            except (OSError, ValueError):
                continue
            digest = _program_digest(buffer)
            record = found.setdefault(digest, {
                "digest": digest, "workload": _workload_of(run_dir, prefix),
                "buffer": buffer, "buffer_path": str(buffer_path), "measured": set(), "runs": 0})
            record["measured"].add(cycles)
            record["runs"] += 1
    return found


def slowest_rate(records: Sequence[Mapping], *, peak_macs_per_cycle: float) -> tuple[dict, dict]:
    """The slowest measured MACs-per-cycle PER COMPUTE CLASS, with what each was taken from.

    This is the parameter the empirical ceiling divides by, so it MUST be derived from programs the
    containment rate is not then measured on. Fitting and testing on one set would report how well a
    bound covers the very data that set it, which is not a rate anybody can act on.

    Split by compute class because one global rate, though sound, produced bands 95.7x wide -- the
    slowest and fastest classes on this machine differ by 35x, and a bound covering both is that much
    looser than either needs. The class is read off the emitted program, not fitted.
    """
    per_class: dict[str, list[tuple[float, str]]] = {}
    for record in records:
        if len(record["measured"]) != 1:
            continue
        measured = float(next(iter(record["measured"])))
        floor = CE._structural_floor(record["buffer"], peak_macs_per_cycle)
        if floor["status"] != CE.DERIVED or not floor.get("counts_every_command") or measured <= 0:
            continue
        klass = CE.compute_class(record["buffer"])
        if klass is None:
            continue
        per_class.setdefault(klass, []).append((floor["macs"] / measured, record["workload"]))
    if not per_class:
        return {}, {"reason": "no training program prices its whole command stream"}
    rates, provenance = {}, {}
    for klass, observed in per_class.items():
        rate, workload = min(observed)
        rates[klass] = rate
        provenance[klass] = {"slowest_macs_per_cycle": rate, "slowest_from": workload,
                             "n_training_programs": len(observed),
                             "fastest_observed": max(observed)[0]}
    return rates, provenance


def score(records: Sequence[Mapping], *, target: str, peak_macs_per_cycle: float,
          rates_by_class: Mapping[str, float]) -> dict:
    """Containment rate over every program whose band derives, with both miss directions separated."""
    rows, undecided = [], []
    below = above = inside = 0
    for record in records:
        if len(record["measured"]) != 1:
            undecided.append({"workload": record["workload"], "digest": record["digest"][:12],
                              "reason": f"one program measured {sorted(record['measured'])}"})
            continue
        measured = float(next(iter(record["measured"])))
        klass = CE.compute_class(record["buffer"])
        derived = CE.band(record["buffer"], target=target,
                          peak_macs_per_cycle=peak_macs_per_cycle,
                          slowest_macs_per_cycle=rates_by_class.get(klass or ""))
        if derived.get("status") != CE.DERIVED:
            undecided.append({"workload": record["workload"], "digest": record["digest"][:12],
                              "reason": str(derived.get("reason") or "band not derived")})
            continue
        lower, upper = float(derived["lower"]), float(derived["upper"])
        if measured < lower:
            verdict, direction = "MISS", "below_floor"
            below += 1
        elif measured > upper:
            verdict, direction = "MISS", "above_ceiling"
            above += 1
        else:
            verdict, direction = "CONTAINED", "inside"
            inside += 1
        rows.append({"workload": record["workload"], "digest": record["digest"][:12],
                     "measured_cycles": measured, "lower": lower, "upper": upper,
                     "width": upper - lower, "verdict": verdict, "direction": direction,
                     "compute_class": klass,
                     "position": None if upper == lower else (measured - lower) / (upper - lower)})
    decided = len(rows)
    workloads = sorted({row["workload"] for row in rows})
    # PER-WORKLOAD, NOT JUST OVERALL. An aggregate rate can be carried by one workload with many
    # programs while another fails entirely -- the exact shape that made a margin-swept ordering rate
    # look exposable when the slice contradicting it had simply stopped being counted.
    per_workload = {}
    for name in workloads:
        slice_rows = [row for row in rows if row["workload"] == name]
        held = sum(1 for row in slice_rows if row["verdict"] == "CONTAINED")
        per_workload[name] = {"decided": len(slice_rows), "contained": held,
                              "rate": held / len(slice_rows) if slice_rows else None}
    return {
        "target": target, "peak_macs_per_cycle": peak_macs_per_cycle,
        "programs_decided": decided, "programs_undecided": len(undecided),
        "workloads": len(workloads),
        "containment": {
            "contained": inside, "below_floor": below, "above_ceiling": above,
            "rate": (inside / decided) if decided else None,
        },
        "per_workload": per_workload,
        "worst_workload": (min(per_workload.items(), key=lambda kv: kv[1]["rate"])[0]
                           if per_workload else None),
        "rows": sorted(rows, key=lambda row: (row["workload"], row["digest"])),
        "undecided": undecided[:40],
        "licence": ("a band may ELIMINATE a candidate and may never certify one; this rate says only "
                    "how often the interval contained the measurement, never where inside it"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--root", action="append", default=[],
                        help="a run tree to harvest (repeatable)")
    parser.add_argument("--run-prefix", default="",
                        help="run-directory name prefix stripped when naming the workload")
    parser.add_argument("--peak-macs-per-cycle", type=float, default=None,
                        help="override the derived structural peak (derived from RTL facts by default)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    peak = args.peak_macs_per_cycle
    if peak is None:
        # Derived exactly the way the measurement path derives it, so the floor here and the
        # utilization denominator there cannot disagree about which machine they describe. Refuses on
        # zero peaks (no evidenced array) and on more than one (picking would choose the machine).
        from merlin.perf.contract import derive_contract  # noqa: PLC0415
        contract = derive_contract(args.target)
        peaks = []
        for resource in getattr(contract, "resources", ()):
            term = (getattr(resource, "terms", None) or {}).get("peak_macs_per_cycle")
            value = getattr(term, "value", None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                peaks.append((resource.name, value))
        if len(peaks) != 1:
            names = ", ".join(sorted(name for name, _ in peaks)) or "none"
            print(f"refusing: {len(peaks)} compute peak(s) derived for {args.target!r} ({names}); "
                  f"pass --peak-macs-per-cycle to name the one this corpus runs on", file=sys.stderr)
            return 2
        peak = float(peaks[0][1])

    roots = [Path(r) for r in args.root] or [Path("out/runs") / args.target]
    records = list(harvest(roots, prefix=args.run_prefix).values())
    if not records:
        print(f"refusing: no run under {[str(r) for r in roots]} carries both a {BUFFER_NAME} and a "
              f"console with a {CYCLE_METRIC} metric", file=sys.stderr)
        return 2

    # SPLIT BY WORKLOAD, NOT BY PROGRAM. Two programs of the same workload are two candidates for one
    # shape and share its rate almost exactly, so splitting by program would leak the answer across
    # the boundary and report a containment rate that no unseen workload will reproduce.
    workloads = sorted({record["workload"] for record in records})
    held_out = {name for index, name in enumerate(workloads) if index % 2 == 1}
    train = [record for record in records if record["workload"] not in held_out]
    test = [record for record in records if record["workload"] in held_out]
    rates, provenance = slowest_rate(train, peak_macs_per_cycle=peak)
    if not rates:
        print(f"refusing: {provenance.get('reason')}", file=sys.stderr)
        return 2

    report = score(test, target=args.target, peak_macs_per_cycle=peak, rates_by_class=rates)
    report["rates_fitted_on_training_workloads_only"] = {
        "training_workloads": len(workloads) - len(held_out),
        "held_out_workloads": len(held_out), "by_compute_class": provenance}
    report["in_sample"] = score(train, target=args.target, peak_macs_per_cycle=peak,
                                rates_by_class=rates)["containment"]

    out = Path(args.out) if args.out else (
        artifacts_dir() / "perf-bench" / args.target / ARTIFACT_NAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    containment = report["containment"]
    rate = containment["rate"]
    print(f"{report['programs_decided']} program(s) over {report['workloads']} workload(s); "
          f"{report['programs_undecided']} undecided")
    print(f"contained {containment['contained']}  below-floor {containment['below_floor']}  "
          f"above-ceiling {containment['above_ceiling']}"
          + (f"  rate {rate:.3f}" if rate is not None else ""))
    if report["worst_workload"]:
        worst = report["per_workload"][report["worst_workload"]]
        print(f"worst workload {report['worst_workload']}: {worst['contained']}/{worst['decided']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
