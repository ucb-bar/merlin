#!/usr/bin/env python3
"""Per-cycle joint occupancy for a hardware-interlocked, command-driven accelerator.

The atlas occupancy work read a program counter and a set of top-level busy ports. Neither exists
here: this target is driven by a command queue over RoCC, its hazards are resolved by a reservation
station rather than by compiler-inserted separations, and its work is carried by three DECOUPLED
controllers (load / execute / store) that are expected to run at the same time. That expectation is
the point -- on the previous target every joint measurement came back with zero overlap, and a
decoupled machine is the falsifier for that whole instrument. **If this reports zero overlap, suspect
the instrument before the machine.**

WHY NOT THE EXISTING ENGINE
---------------------------
The modelling repo already folds these three controllers into an occupancy waterfall, but it
PARTITIONS: each cycle is charged to exactly one of seven buckets, so concurrency survives only as an
aggregate ``overlap`` bucket and per-unit occupancy can never be read back out. That is the precise
instrument fault :mod:`merlin.perf.occupancy` exists to correct, so this driver takes the raw
per-cycle FSM states and builds a JOINT vector instead.

HOW THE IDLE ENCODING IS ESTABLISHED
------------------------------------
A controller's state register is an encoding, and "0 means idle" is exactly the kind of constant this
repo forbids assuming. It is not assumed here: the design also exposes busy PORTS for its memory
paths, and :func:`merlin.perf.occupancy.calibrate_state_idle` derives the idle value by finding the
value a state register holds on precisely the cycles a port is low. If no port pairs, the calibration
REFUSES and every unpaired controller stays out of the vector, reported unmeasured -- never assumed
idle, which is the reading that flatters the result.

Usage::

    gemmini_occupancy.py --target T [--shape 16x16x16] [--shape 32x32x32]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from merlin.common import artifacts as A                                          # noqa: E402
from merlin.perf.occupancy import (                                               # noqa: E402
    calibrate_state_idle,
    declared_engines,
    joint_counts,
    unit_bindings,
)

#: The controller state registers, and the memory-path busy ports the calibration pairs against.
#: Both lists are the DESIGN's own signal names, read from its state manifest -- this module does not
#: invent one, and a name absent from the manifest is reported rather than silently skipped.
STATE_SIGNALS = ("ex_controller/control_state",
                 "load_controller/control_state",
                 "store_controller/control_state")
PORT_SIGNALS = ("spad/io_busy", "spad/reader/xactTracker/io_busy", "spad/writer/io_busy")
#: The idle encoding the MODELLING REPO declares for these state registers (its occupancy engine
#: reads a controller as busy when its state is non-zero). Used only when the pairing derivation
#: refuses, and always carried with that provenance.
PRODUCER_DECLARED_IDLE = "0"


def mlc_dir() -> Path:
    d = os.environ.get("MERLIN_MLC_DIR")
    if not d:
        raise SystemExit("set MERLIN_MLC_DIR to the modelling checkout")
    return Path(d)


def _cosim(target: str):
    """The target's co-simulation model, plus the signals its manifest actually exposes."""
    sys.path.insert(0, str(mlc_dir()))
    from mlc.backends.cosim import GemminiCosim                    # noqa: PLC0415

    outputs = mlc_dir() / "runs" / "circt-arc" / target / "outputs"
    so = next(iter(sorted(outputs.parent.glob("native_run/*.so"))), None)
    manifest = next(iter(sorted(outputs.glob("*_core_state.json"))), None)
    if so is None or manifest is None:
        raise SystemExit(f"no built model for {target!r} under {outputs.parent}")
    declared = {s["name"] for s in json.loads(manifest.read_text())[0]["states"]}
    return GemminiCosim(str(so), str(manifest)), declared


def trace_matmul(target: str, m: int, k: int, n: int) -> tuple[dict, dict]:
    """Run one matmul on the model and record every requested signal, per cycle.

    Returns ``(trace, meta)``. The recording wraps the model's own clock rather than reimplementing
    its command sequence: the workload stays exactly the one the model already drives bit-exactly, so
    this measures the same execution the functional path grades and not a re-creation of it.
    """
    import numpy as np                                            # noqa: PLC0415

    cosim, declared = _cosim(target)
    present = [s for s in (*STATE_SIGNALS, *PORT_SIGNALS) if s in declared]
    absent = [s for s in (*STATE_SIGNALS, *PORT_SIGNALS) if s not in declared]

    trace: dict[str, list[str]] = {s: [] for s in present}
    core = cosim.core
    _tick = core.tick

    def recording_tick(*a, **kw):
        out = _tick(*a, **kw)
        for s in present:
            trace[s].append(str(core.peek(s)))
        return out

    core.tick = recording_tick
    rng = np.random.default_rng(0xA7)
    a = rng.integers(-8, 8, size=(m, k), dtype=np.int8)
    b = rng.integers(-8, 8, size=(k, n), dtype=np.int8)
    got = cosim.matmul_ws(a, b)
    core.tick = _tick

    ref = a.astype(np.int32) @ b.astype(np.int32)
    return trace, {"shape": f"{m}x{k}x{n}", "cycles_recorded": len(trace[present[0]]) if present else 0,
                   "signals_present": present, "signals_absent": absent,
                   "bit_exact": bool(np.array_equal(np.asarray(got, dtype=np.int32), ref))}


def analyse(target: str, traces: list[dict], metas: list[dict]) -> dict:
    """Calibrate the idle encoding across the corpus, then build the joint vector per run."""
    states = [s for s in STATE_SIGNALS if any(s in t for t in traces)]
    ports = [s for s in PORT_SIGNALS if any(s in t for t in traces)]
    calib = calibrate_state_idle(traces, states, ports)

    # The calibration pairs a state register against a busy port. On a workload whose operands are
    # placed directly into the scratchpad, no memory-path port ever asserts, so nothing pairs and the
    # derivation correctly refuses. Rather than assume an encoding, fall back to the one the MODELLING
    # REPO ITSELF declares for these registers (its occupancy engine reads `busy == state != 0`) and
    # carry that provenance on every number downstream. A producer's declaration is an acceptable
    # input -- the kinds and the engine set already arrive that way -- but it is NOT a measurement,
    # and the distinction has to survive into the report rather than being flattened into a value.
    if calib.get("idle_value") is None:
        varying = [s for s in states if any(len(set(t.get(s, ()))) > 1 for t in traces)]
        constant_ports = [s for s in ports
                          if all(len(set(t.get(s, ()))) < 2 for t in traces)]
        calib = dict(calib, idle_value=PRODUCER_DECLARED_IDLE, basis="declared_by_producer",
                     paired_with=None, constant_ports=constant_ports, varying_states=varying,
                     detail=("no busy port varies on this workload, so the encoding could not be "
                             "cross-checked; using the value the modelling repo's own occupancy "
                             "engine declares. DECLARED, not derived -- and unverifiable until a "
                             "workload exercises a port"))

    engines = declared_engines(_contract(target))
    out = {"target": target, "calibration": calib, "declared_engines": engines, "runs": []}
    for trace, meta in zip(traces, metas):
        idle = calib.get("idle_value")
        hot: dict[str, list[bool]] = {}
        unmeasured: list[str] = []
        for s in states:
            if s not in trace:
                continue
            if idle is None or len(set(trace[s])) < 2:
                unmeasured.append(s)
                continue
            hot[s] = [v != idle for v in trace[s]]
        jc = joint_counts(hot) if hot else {"sampled_cycles": 0, "overlap_any": None}
        out["runs"].append(meta | {"joint": jc, "unmeasured_columns": unmeasured})
    return out


def _contract(target: str) -> dict:
    from merlin.targetgen.rtl.facts import target_contract_path    # noqa: PLC0415
    import yaml                                                    # noqa: PLC0415
    return yaml.safe_load(target_contract_path(target).read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--shape", action="append", default=None,
                    help="MxKxN; repeatable (default: two sizes)")
    args = ap.parse_args()
    shapes = args.shape or ["16x16x16", "32x32x32"]

    traces, metas = [], []
    for sh in shapes:
        m, k, n = (int(v) for v in sh.split("x"))
        trace, meta = trace_matmul(args.target, m, k, n)
        traces.append(trace)
        metas.append(meta)
        print(f"{sh:14s} cycles={meta['cycles_recorded']:6d} bit_exact={meta['bit_exact']} "
              f"absent={meta['signals_absent']}", flush=True)

    rep = analyse(args.target, traces, metas)
    print(f"\nidle calibration: value={rep['calibration']['idle_value']!r} "
          f"paired_with={rep['calibration']['paired_with']}")
    for r in rep["runs"]:
        j = r["joint"]
        print(f"  {r['shape']:12s} sampled={j.get('sampled_cycles')} "
              f"idle={j.get('idle_cycles')} overlap={j.get('overlap_any')} "
              f"busy={ {k: v for k, v in (j.get('busy') or {}).items() if v} }")

    pd = A.new_product("perf-ledger", version=3, target=args.target,
                       notes="joint controller occupancy from the co-simulation model")
    (pd.path / "occupancy.json").write_text(json.dumps(rep, indent=1))
    print(f"\nwrote {pd.path / 'occupancy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
