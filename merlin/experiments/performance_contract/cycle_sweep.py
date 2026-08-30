#!/usr/bin/env python3
"""Cycle-accurate sweep of the whole capsule corpus, on BOTH elaborated-RTL engines.

Two engines run the *identical* program spec, so anything they disagree about is a property of an
engine and not of the program:

``vsim``
    The Verilator harness. Reads the design's ``io.dbg`` occupancy tap -- every unit the RTL exports
    as a top-level busy port, including each DMA channel -- and reports ``idle``, ``overlap_any``
    and ``overlap_kind`` as JOINT counts. It is the citable instrument.
``gsim``
    The GSIM engine on the same ``.fir``, ~50x faster, and able to read signals that are *not*
    top-level ports: the program counter and the internal FSM states. It cannot see the per-channel
    DMA ports the Verilator tap reads. The two are complementary, not redundant, and neither alone
    gives an honest idle figure -- each one's blind spot is charged to idle.

WHY BOTH, AND WHY THE OVERLAP NUMBER IS THE POINT
-------------------------------------------------
Every overlap figure this program had until now came from a source whose buckets *partition* the
timeline -- and a partition reports zero overlap whether or not the hardware overlaps. That is an
artifact of the instrument, not a measurement of the machine (see
:func:`merlin.perf.headroom.composition_operator`, which refuses such a source by design). A joint
occupancy vector is the only thing that can settle it, and both engines sample one.

The rules for turning per-cycle traces into a joint vector live in :mod:`merlin.perf.occupancy` --
subsumption, the FSM idle calibration, and the cross-engine merge -- because each of them is a
property of the measurement rather than of this target. This module is the driver: it runs the
engines, hands their traces to those rules, and writes the product.

KINDS ARE DECLARED BY THE PRODUCER, NEVER INFERRED FROM A SPELLING
------------------------------------------------------------------
Whether a unit is compute or movement is a fact about the target, so this module never decides it.
The Verilator harness emits a ``kind`` alongside every unit it reports, and :func:`kind_map` reads
the column kinds out of *that* declaration. A column with no declared kind is excluded from the
across-kind overlap rather than guessed into one side of it, so that count is a LOWER BOUND and
says so.

Usage::

    cycle_sweep.py run     --target T [--engine both|vsim|gsim] [--capsule C ...]
    cycle_sweep.py report  --target T
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from merlin.common import artifacts as A                                          # noqa: E402
from merlin.perf.observations import (                                            # noqa: E402
    ALIAS_COLLISIONS_KEY,
    BUSY_PREFIX,
    IDLE_QUANTITY,
    IN_PROGRAM_SUFFIX,
    OVERLAP_ACROSS_KINDS,
    OVERLAP_OBSERVED,
    PARTITIONED_KEY,
    SAMPLED_QUANTITY,
    TIMING_OBSERVATIONS_KEY,
    UNMEASURED_UNITS_KEY,
)
from merlin.perf.occupancy import (                                               # noqa: E402
    calibrate_state_idle,
    joint_counts,
    merge_engines,
)

#: Trace columns that are bookkeeping rather than occupancy.
_NON_UNIT_COLUMNS = ("cycle", "halted", "pc", "fetch_pc", "n_busy")
#: A column whose value is a state ENCODING rather than a busy bit.
_STATE_SUFFIX = "_state"
#: What a column's name has to end with for the producer's kind declaration to be findable.
_BUSY_SUFFIX = "Busy"


def gsim_dir() -> Path:
    """Where the built GSIM engine and the program specs live."""
    d = os.environ.get("MERLIN_GSIM_ATLAS_DIR")
    if not d:
        raise SystemExit("set MERLIN_GSIM_ATLAS_DIR to the built GSIM engine directory")
    return Path(d)


def vsim_dir() -> Path:
    d = os.environ.get("MERLIN_EXT_ATLAS_VSIM")
    if not d:
        raise SystemExit("set MERLIN_EXT_ATLAS_VSIM to the built Verilator oracle directory")
    return Path(d)


def _one_json_line(stdout: str) -> dict:
    line = next((ln for ln in reversed(stdout.splitlines()) if ln.strip().startswith("{")), None)
    if line is None:
        raise RuntimeError(f"engine printed no JSON: {stdout[-400:]}")
    return json.loads(line)


def _run(binp: Path, args: list[str], timeout: int) -> tuple[dict, float]:
    t0 = time.perf_counter()
    p = subprocess.run([str(binp), *args], capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{binp.name} rc={p.returncode}: {p.stderr[-400:]}")
    return _one_json_line(p.stdout), wall


def run_vsim(spec: Path, trace: Path | None = None, *, timeout: int = 900) -> tuple[dict, float]:
    """Run one spec on the Verilator oracle. Returns (result, wall_s), measured SERIALLY.

    ``trace`` asks the harness for its own per-cycle dump via the spec key it already supports. The
    dump is ~34 B/cycle and regenerable by replay, so it belongs in the purgeable cache.
    """
    if trace is not None:
        body = json.loads(spec.read_text())
        body["per_cycle_csv"] = str(trace)
        spec = Path(str(spec) + ".traced")
        spec.write_text(json.dumps(body))
    return _run(vsim_dir() / "vobj_dir" / "VAtlasCore", [str(spec)], timeout)


def run_gsim(spec: Path, trace: Path | None, *, timeout: int = 900) -> tuple[dict, float]:
    """Run one spec on GSIM, optionally dumping a per-cycle trace. Returns (result, wall_s)."""
    return _run(gsim_dir() / "atlas_gsim_sim",
                [str(spec)] + ([str(trace)] if trace else []), timeout)


def read_trace(path: Path) -> dict[str, list[str]]:
    """A per-cycle CSV as ``{column: [value per cycle]}``."""
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"empty trace: {path}")
    return {c: [r[c] for r in rows] for c in rows[0]}


def unit_columns(trace: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """``(port_columns, state_columns)`` -- occupancy bits and state encodings."""
    cols = [c for c in trace if c not in _NON_UNIT_COLUMNS]
    return ([c for c in cols if not c.endswith(_STATE_SUFFIX)],
            [c for c in cols if c.endswith(_STATE_SUFFIX)])


def occupancy_of(trace: dict[str, list[str]], calibration: dict | None) -> tuple[dict, list[str]]:
    """``(occupancy, unmeasured)`` -- busy-per-cycle per column, plus what could not be read.

    A state column joins only when the corpus calibrated its idle encoding AND it varies in this
    program. Everything else is reported unmeasured; it is never defaulted to idle, which is the
    reading that flatters the result.
    """
    ports, states = unit_columns(trace)
    hot = {c: [v not in ("0", "") for v in trace[c]] for c in ports}
    idle_value = (calibration or {}).get("idle_value")
    unmeasured: list[str] = []
    for s in states:
        if idle_value is None or len(set(trace[s])) < 2:
            unmeasured.append(s)
            continue
        hot[s] = [v != idle_value for v in trace[s]]
    return hot, unmeasured


def kind_map(vsim_result: dict) -> dict[str, str]:
    """The producer's OWN declaration of what each unit is, read off its timing block."""
    out: dict[str, str] = {}
    for e in vsim_result.get(TIMING_OBSERVATIONS_KEY) or []:
        q, kind = e.get("quantity", ""), e.get("kind")
        if kind and q.startswith(BUSY_PREFIX) and q.endswith(IN_PROGRAM_SUFFIX):
            out[q[len(BUSY_PREFIX):-len(IN_PROGRAM_SUFFIX)]] = kind
    return out


def column_kinds(columns: list[str], declared: dict[str, str]) -> dict[str, str]:
    """Kinds for the columns the producer declared. A column it did not declare is simply absent."""
    out = {}
    for c in columns:
        if c in declared:
            out[c] = declared[c]
        elif c.endswith(_BUSY_SUFFIX) and c[: -len(_BUSY_SUFFIX)] in declared:
            out[c] = declared[c[: -len(_BUSY_SUFFIX)]]
    return out


def _summarise_vsim(result: dict) -> dict:
    """The joint counts the Verilator block carries, refusing to default any of them."""
    out: dict[str, object] = {"idle_cycles": None, "overlap_any": None,
                              "overlap_across_kinds": None, "sampled_cycles": None,
                              "partitioned": None, "busy": {}}
    for e in result.get(TIMING_OBSERVATIONS_KEY) or []:
        q, v = e.get("quantity", ""), e.get("value")
        if v is None:
            continue
        if q == IDLE_QUANTITY:
            out["idle_cycles"] = int(v)
        elif q == OVERLAP_OBSERVED:
            out["overlap_any"] = int(v)
        elif q == OVERLAP_ACROSS_KINDS:
            out["overlap_across_kinds"] = int(v)
        elif q == SAMPLED_QUANTITY:
            out["sampled_cycles"] = int(v)
        elif q.startswith(BUSY_PREFIX) and q.endswith(IN_PROGRAM_SUFFIX):
            out["busy"][q[len(BUSY_PREFIX):-len(IN_PROGRAM_SUFFIX)]] = int(v)
        if PARTITIONED_KEY in e:
            out["partitioned"] = e[PARTITIONED_KEY]
    return out


def sweep(target: str, engines: tuple[str, ...], capsules: list[str] | None,
          trace_dir: Path) -> tuple[list[dict], dict | None]:
    """Run every spec on every engine SERIALLY, then merge their traces into one joint vector."""
    specs = sorted(p for p in gsim_dir().glob("spec_*.json")
                   if "_pad" not in p.name and "_nohalt" not in p.name)
    if capsules:
        want = set(capsules)
        specs = [p for p in specs if p.stem[len("spec_"):] in want]
    trace_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for sp in specs:
        name = sp.stem[len("spec_"):]
        row: dict = {"capsule": name, "target": target, "concurrency": 1,
                     "words": len(json.loads(sp.read_text())["words"])}
        for engine, runner in (("vsim", run_vsim), ("gsim", run_gsim)):
            if engine not in engines:
                continue
            tr = trace_dir / (f"{name}.vsim.csv" if engine == "vsim" else f"{name}.csv")
            try:
                res, wall = runner(sp, tr)
                row[engine] = {"cycles": res.get("cycles"), "halted": res.get("halted"),
                               "reads": res.get("reads"), "writes": res.get("writes"),
                               "wall_s": round(wall, 4), "trace": str(tr),
                               "alias_collisions": res.get(ALIAS_COLLISIONS_KEY)}
                if engine == "vsim":
                    row[engine]["summary"] = _summarise_vsim(res)
                    row[engine]["declared_kinds"] = kind_map(res)
                    row[engine]["unmeasured_units"] = res.get(UNMEASURED_UNITS_KEY)
            except Exception as exc:                       # an engine failure is data, not a crash
                row[engine] = {"error": str(exc)[:300]}
        a, b = row.get("vsim", {}).get("cycles"), row.get("gsim", {}).get("cycles")
        row["cycles_match"] = (a == b) if (a is not None and b is not None) else None
        rows.append(row)
        print(json.dumps({"capsule": name, "words": row["words"],
                          "vsim_cycles": a, "gsim_cycles": b,
                          "cycles_match": row["cycles_match"]}), flush=True)

    # The FSM idle encoding is a property of the DESIGN, so it is calibrated over the whole corpus
    # and only then applied. Per-trace calibration let a program that never exercised the paired
    # unit withdraw a calibration the rest of the corpus had established.
    traces = {r["capsule"]: read_trace(Path(r["gsim"]["trace"]))
              for r in rows if "trace" in r.get("gsim", {})}
    calib = None
    if traces:
        any_tr = next(iter(traces.values()))
        ports, states = unit_columns(any_tr)
        calib = calibrate_state_idle(list(traces.values()), states, ports)
        print(f"\nFSM idle calibration: value={calib['idle_value']!r} "
              f"paired_with={calib['paired_with']} over {calib['checked_traces']} traces")

    for r in rows:
        gtr = traces.get(r["capsule"])
        if gtr is None:
            continue
        ghot, gunmeasured = occupancy_of(gtr, calib)
        declared = r.get("vsim", {}).get("declared_kinds", {})
        r["gsim"]["analysis"] = joint_counts(ghot, column_kinds(list(ghot), declared)) | {
            "unmeasured_columns": gunmeasured}
        vpath = r.get("vsim", {}).get("trace")
        if not vpath:
            continue
        vhot, vunmeasured = occupancy_of(read_trace(Path(vpath)), calib)
        merged, prov = merge_engines(vhot, ghot)
        r["union"] = joint_counts(merged, column_kinds(list(merged), declared)) | {
            "merge": prov, "unmeasured_columns": sorted(set(vunmeasured) & set(gunmeasured))}
    return rows, calib


def latest_sweep(target: str) -> dict:
    base = A.artifacts_dir() / "perf-ledger" / target / "v2"
    files = sorted(base.glob("*/cycle_sweep.json"))
    if not files:
        raise SystemExit(f"no sweep product under {base}; run `cycle_sweep.py run` first")
    return json.loads(files[-1].read_text())


def report(target: str) -> int:
    """Derive the composition operator from the JOINT occupancy, and say what it rests on."""
    from merlin.perf.decompose import ActivitySource, Resource, ResourceKind
    from merlin.perf.headroom import composition_operator

    d = latest_sweep(target)
    rows = [r for r in d["rows"] if "union" in r]
    sources, observed = [], {}
    for r in rows:
        u = r["union"]
        kinds = r["vsim"]["declared_kinds"]
        res = tuple(Resource(name=c, kind=ResourceKind(kinds[c]), busy_cycles=int(u["busy"][c]))
                    for c in u["joint_columns"] if c in kinds)
        if not res:
            continue
        sources.append(ActivitySource(
            workload=r["capsule"], total_cycles=int(r["vsim"]["cycles"]), resources=res,
            partitioned=False,
            provenance="joint per-cycle occupancy, union of two elaborated-RTL engines"))
        observed[r["capsule"]] = int(u["overlap_any"])

    total = sum(s.total_cycles for s in sources)
    print(f"sources: {len(sources)} workloads, joint (partitioned=False)")
    print(f"observed overlap: {sum(observed.values())} cycles across {total}\n")
    out = composition_operator(sources, observed_overlap_cycles=observed)
    if isinstance(out, tuple):
        op, eta = out
        print(f"COMPOSITION OPERATOR = {op.name}   (eta = {eta:.4f})")
        print("  eta is the realised fraction of the OVERLAPPABLE time: 0 -> SUM, 1 -> MAX.")
    else:
        print(f"REFUSED: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--target", required=True)
    r = sub.add_parser("run")
    r.add_argument("--target", required=True)
    r.add_argument("--engine", default="both", choices=("both", "vsim", "gsim"))
    r.add_argument("--capsule", action="append")
    r.add_argument("--trace-dir", default=None)
    args = ap.parse_args()

    if args.cmd == "report":
        return report(args.target)

    engines = ("vsim", "gsim") if args.engine == "both" else (args.engine,)
    trace_dir = Path(args.trace_dir) if args.trace_dir else A.cache_dir(f"occupancy/{args.target}")
    rows, calib = sweep(args.target, engines, args.capsule, trace_dir)

    pd = A.new_product("perf-ledger", version=2, target=args.target,
                       notes="cycle-accurate corpus sweep on both elaborated-RTL engines")
    (pd.path / "cycle_sweep.json").write_text(json.dumps(
        {"target": args.target, "engines": list(engines), "concurrency": 1,
         "trace_dir": str(trace_dir), "state_idle_calibration": calib, "rows": rows}, indent=1))
    print(f"\nwrote {pd.path / 'cycle_sweep.json'}  ({len(rows)} capsules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
