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


class _in_mlc_checkout:
    """Run a block with the modelling checkout as the working directory, then put it back.

    Not cosmetic. The modelling repo resolves its own artifact handles -- the compiled model, the state
    manifest, the discovered-interface cache -- as paths RELATIVE to the process working directory, so
    importing it from elsewhere makes it report "no discovered-interface cache for <target>": a
    perfectly built model reported as an absent one. That reading is what a calibration record then
    repeats as "no cycle-accurate trace obtainable in this environment", which is a statement about a
    working directory wearing the clothes of a statement about the host.

    Nothing is written into that checkout -- other sessions work in those trees -- and the previous
    directory is restored even when the model raises.
    """

    def __init__(self, where: Path):
        self._where, self._prev = Path(where), None

    def __enter__(self):
        self._prev = Path.cwd()
        os.chdir(self._where)
        return self._where

    def __exit__(self, *exc):
        if self._prev is not None:
            os.chdir(self._prev)
        return False


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

    with _in_mlc_checkout(mlc_dir()):
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


def instance_modules(target: str) -> dict[str, str]:
    """``{instance name: module name}`` from the target's OWN elaborated HW dialect.

    The co-simulation state manifest addresses a signal by its INSTANCE path
    (``ex_controller/control_state``), while a capability contract and a synthesis FSM inventory both
    name a unit by its MODULE (``ExecuteController``). Nothing joins the two but the design itself, and
    the join must come from the design: turning ``ex_controller`` into ``ExecuteController`` by pattern
    is precisely the guess the repo's cardinal rule forbids, and it is wrong the moment a design
    instantiates a module under a name that does not resemble it.

    Parsed structurally -- ``partition`` and ``split`` over ``hw.instance "<name>" @<Module>``, no
    pattern matching -- and an unparseable or absent dialect returns an empty map, which leaves every
    column UNBOUND and reported rather than bound to a guess.
    """
    outputs = mlc_dir() / "runs" / "circt-arc" / target / "outputs"
    hw = next(iter(sorted(outputs.glob("*_core_hw.mlir"))), None)
    if hw is None:
        return {}
    out: dict[str, str] = {}
    for line in hw.read_text(encoding="utf-8", errors="replace").splitlines():
        _, sep, rest = line.partition("hw.instance ")
        if not sep or not rest.startswith('"'):
            continue
        name, quoted, tail = rest[1:].partition('"')
        at = tail.find("@")
        if not quoted or at == -1:
            continue
        module = tail[at + 1:].split("(")[0].split("<")[0].strip()
        if name and module:
            out.setdefault(name, module)
    return out


def engine_by_module(target: str) -> dict[str, str]:
    """``{module name: the engine that module IS}``, from the target's own declarations.

    Two sources, both the target's own: a contract compute unit that names its ``rtl_module``, and the
    synthesis FSM inventory, whose engines are named by module already. A module in neither is absent,
    not defaulted -- a column in it stays unbound and is reported as such.
    """
    from merlin.targetgen.rtl.fsm import fsm_inventory                # noqa: PLC0415

    out: dict[str, str] = {}
    for unit in (_contract(target) or {}).get("compute_units") or ():
        module, name = unit.get("rtl_module"), unit.get("name")
        if module and name:
            out.setdefault(str(module), str(name))
    try:
        for reg in fsm_inventory(target):
            module = str(getattr(reg, "module", "") or "")
            if module:
                out.setdefault(module, module)
    except OSError:
        pass                          # no extraction on disk: the contract's own units still stand
    return out


def mechanism_traces(target: str, traces: list[dict], metas: list[dict]) -> list[dict]:
    """The per-cycle recordings, in the shape ``perf_calibrate.py --traces`` consumes.

    The summary this driver already writes is the AGGREGATE (joint busy, overlap, idle). The columns it
    was computed from lived only in this process, so the calibration seam -- which needs the per-cycle
    values, not the totals -- had nothing to read and every eta in a calibration record stayed UNKNOWN
    beside a real measurement. This writes the columns out.

    What the file DECLARES is the part that is not derivable from the samples: which columns are busy
    ports and which are state registers whose encoding must be calibrated, and which engine each column
    belongs to. The binding names the DESIGN's own controller modules, because that is what the
    instrument read; whether the target's capability contract declares those engines is the contract's
    business, and where it does not, the calibration refuses the reading and says so. Renaming a column
    onto whatever engine the contract happens to declare, so the reading resolves, is the failure this
    seam is built to prevent -- three decoupled controllers reported as one systolic array would put
    their concurrency inside a single engine, where it cannot be seen.

    ``bit_exact`` travels with each trace as the work fingerprint: an occupancy vector from a run that
    computed the wrong thing is not this machine's behaviour on that workload.
    """
    instances = instance_modules(target)
    engines = engine_by_module(target)
    out = []
    for trace, meta in zip(traces, metas):
        present = list(meta.get("signals_present") or trace.keys())
        states = [c for c in present if c in STATE_SIGNALS]
        ports = [c for c in present if c in PORT_SIGNALS]
        binding, unbound = {}, []
        for c in states:
            engine = engines.get(instances.get(c.split("/")[0], ""), "")
            if engine:
                binding[c] = engine
            else:
                unbound.append(c)
        out.append({
            "capsule": f"cosim_matmul_{meta.get('shape')}",
            "columns": {c: list(trace[c]) for c in present if c in trace},
            # Derived: instance -> module from the design's HW dialect, module -> engine from the
            # target's own declarations. A column whose instance resolves to no declared engine is left
            # OUT of the binding and reported unbound, never attached to whichever engine is nearest.
            "binding": binding,
            "port_columns": ports,
            "state_columns": states,
            "unmeasured_units": list(meta.get("signals_absent") or ()),
            "work": (f"{meta.get('shape')} i8 matmul, bit_exact={meta.get('bit_exact')}"
                     if meta.get("bit_exact") is not None else None),
            # The recording wraps the model's clock and samples state; it does not observe when a
            # controller's work COMPLETED. Stated, because defaulting it True satisfies a gate nothing
            # measured.
            "completion_observable": False,
            "provenance": (f"co-simulation model of {target} under MERLIN_MLC_DIR, recorded per cycle "
                           f"around its own matmul driver; column->engine derived via the design's "
                           f"own hw.instance map"
                           + (f"; UNBOUND (no declared engine for their module): {sorted(unbound)}"
                              if unbound else "")),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--shape", action="append", default=None,
                    help="MxKxN; repeatable (default: two sizes)")
    ap.add_argument("--write-trace", action="store_true",
                    help="also write mechanism_trace.json -- the PER-CYCLE columns, in the shape "
                         "perf_calibrate.py --traces consumes. Without it only the aggregate is kept "
                         "and the calibration seam has nothing to read")
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
    if args.write_trace:
        mt = pd.path / "mechanism_trace.json"
        mt.write_text(json.dumps(mechanism_traces(args.target, traces, metas), indent=1) + "\n")
        print(f"wrote {mt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
