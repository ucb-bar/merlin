#!/usr/bin/env python3
"""Joint occupancy from a target's OWN hardware combination counters, on real elaborated RTL.

THE INSTRUMENT THIS EXISTS TO REPLACE
-------------------------------------
The per-cycle occupancy drivers beside this one need a co-simulation model: a built ``.so`` plus a
state manifest, from a modelling checkout that is not this repo. Where that model is absent every eta
in a mechanism calibration stays UNKNOWN, and the calibration record honestly says so -- but the
target's RTL is right there and perfectly able to answer. And where the model IS present, the workload
it drives places operands straight into the scratchpad, so the movement controllers never run and the
joint vector has one live column: a vector that reports zero overlap by construction.

A target whose RTL carries COMBINATION performance counters answers without either. The hardware
already counts the cycles in which each SUBSET of its engines was busy, so realised overlap is a
counter value rather than an inference, and it is measured on the same elaborated RTL, through the
same compile-and-run path, that certifies a capsule.

WHAT IS AND IS NOT DERIVED
--------------------------
Nothing about the target is written down here. The counter block -- which engines exist, which
combinations are counted, what event code each one has -- is factored out of the target's OWN shipped
header by :mod:`merlin.perf.hw_counters`, the bracket that configures and reads them is emitted by the
same module, and the workload is an ABI command buffer that names no target. The target is a
parameter; the backend comes from the registry.

Two things are refused rather than guessed:

* **A run whose result is not bit-exact contributes no occupancy.** A wrong kernel's controller
  occupancy is not the machine's behaviour on that workload. Every run here is gated on the backend's
  own reference comparison, and a failing run is recorded with its counters DROPPED and the reason
  kept, never quietly averaged in.
* **A counter that did not print is absent, not zero.** :func:`~merlin.perf.hw_counters.parse_counter_output`
  attributes by name, and a value that did not come back stays out of the mapping -- which makes the
  whole reading UNKNOWN downstream rather than making the overlap look smaller than it was.

The output is the ``--counters`` input to ``perf_calibrate.py``: a list of readings, each carrying its
counter values, the cycle window, and the provenance of the run that produced them.

Usage::

    counter_occupancy.py --target T [--shape 16x16x16] [--shape 32x32x32] [--simulator verilator]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from merlin.common import artifacts as A                                          # noqa: E402
from merlin.common import provenance as P                                         # noqa: E402
from merlin.perf import hw_counters as HC                                         # noqa: E402

#: The environment switch the target harnesses read to place the counter bracket. OPT-IN by design:
#: the graded harness must stay byte-identical unless a caller asks for the instrumentation, so this
#: driver sets it for its OWN runs rather than expecting it to be set globally.
COUNTER_OPT_IN = "MERLIN_HW_COUNTERS"

#: The product topic and version. Its own concern: this is a raw instrument reading, not a
#: calibration, and the calibration record cites it rather than containing it.
TOPIC = "perf-counters"
PRODUCT_VERSION = 1
RECORD_NAME = "counter_occupancy.json"


def matmul_cb(target: str, m: int, k: int, n: int) -> dict:
    """An ABI command buffer for one ``m x k x n`` resident-weight matmul.

    The generic command vocabulary, not a target's own: the point of measuring here is that the same
    workload description can be run on any backend that implements the ABI.
    """
    tag = f"{m}_{k}_{n}"
    return {
        "abi_version": "0.1",
        "target": target,
        "tensors": {
            f"w_{tag}": {"shape": [k, n], "dtype": "i8", "role": "weight"},
            f"a_{tag}": {"shape": [m, k], "dtype": "i8", "role": "input"},
            f"y_{tag}": {"shape": [m, n], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": f"w_{tag}", "dst": f"w_{tag}_res"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT",
             "operands": {"lhs": f"a_{tag}", "rhs": f"w_{tag}_res", "dst": f"acc_{tag}"}},
            {"opcode": "COMMIT", "operands": {"src": f"acc_{tag}", "dst": f"y_{tag}"},
             "attributes": {"epilogue": [], "output_dtype": "i32"}},
            {"opcode": "EVICT", "operands": {"handle": f"w_{tag}_res"}},
        ],
    }


def counter_block(target: str) -> dict:
    """The target's derived combination-counter block, or a refusal that says which of the two it is.

    ``absent`` (the headers were read and expose no such block) and ``unavailable`` (no header could
    be read) are kept apart by :func:`~merlin.perf.hw_counters.counters_for_target` and kept apart
    here: the first is a fact about the machine, the second is a fact about this host.
    """
    got = HC.counters_for_target(target)
    if got.get("status") != "derived":
        raise SystemExit(
            f"{target}: no combination-counter block ({got.get('status')}): {got.get('why')}")
    return got


def measure(target: str, cb: dict, *, workload: str, simulator: str, workdir: Path,
            timeout: int) -> dict:
    """Compile + run ONE workload with the counter bracket in place; return the reading or the refusal.

    The bracket is placed by the target's own harness emitter under :data:`COUNTER_OPT_IN`; this driver
    does not edit a harness. If the emitter declined (an unreadable header, more counters than slots)
    the console simply carries no counter lines, and that comes back as an empty ``values`` mapping
    with the console kept -- an instrument that did not fire, not a machine with no overlap.
    """
    import os

    from merlin.runtime.backends import base as BK

    backend = BK.get_backend(target)
    if not backend.available(simulator):
        raise SystemExit(f"{target}: the {simulator!r} oracle is not available on this host")

    prev = os.environ.get(COUNTER_OPT_IN)
    os.environ[COUNTER_OPT_IN] = "1"
    started = time.time()
    try:
        res = backend.run_command_buffer(cb, workdir=str(workdir), simulator=simulator,
                                         timeout=timeout)
    finally:
        if prev is None:
            os.environ.pop(COUNTER_OPT_IN, None)
        else:
            os.environ[COUNTER_OPT_IN] = prev
    elapsed = round(time.time() - started, 1)

    console = str(res.get("console") or "")
    (workdir / "console.txt").write_text(console, encoding="utf-8")
    values = HC.parse_counter_output(console)
    metrics = res.get("metrics") or {}
    correct = res.get("correct")
    oracle = dict(res.get("oracle") or {})

    out = {
        "workload": workload,
        "simulator": simulator,
        "oracle": oracle,
        "bit_exact": correct,
        "elapsed_s": elapsed,
        "total_cycles": int(metrics.get("cycles") or 0) or None,
        "cycle_source": metrics.get("cycle_source"),
        "cycle_window": metrics.get("cycle_window"),
        # The harness brackets the kernel with a cycle read; it does not observe WHEN an engine's work
        # completed, only that the engine was busy. Stated rather than defaulted -- concurrency_traits
        # refuses without it, and a default True is how an unmeasured trait becomes a satisfied gate.
        "completion_observable": False,
        "values": {k: int(v) for k, v in sorted(values.items())},
        "provenance": (f"{simulator} elaborated-RTL run of {workload} with the counter bracket "
                       f"emitted from this target's own shipped counter header"),
    }
    if correct is not True:
        out["values"] = {}
        out["dropped"] = (
            "the run was not bit-exact against the backend's reference, so its counters describe a "
            "kernel that computed the wrong thing rather than this machine's behaviour on this "
            "workload. The values are DROPPED, not averaged in")
    elif not values:
        out["dropped"] = (
            "the run was bit-exact but its console carried no counter line, so the bracket did not "
            "fire. That is an instrument that did not read, not a machine with no overlap")
    return out


def provenance(target: str, pins: list[str], header: str) -> dict:
    """Which hardware revision these counter values are ABOUT, verified rather than asserted.

    A counter reading is a hardware verdict: it says how much this device overlapped. A verdict
    attributed to the wrong revision is worse than no verdict, because it gets cited -- a microkernel
    was once certified 31/31 against the only revision containing the unit under test while the
    revision named for the tapeout does not contain that unit at all, and nothing in the artifact said
    which one the numbers belonged to.

    The pins are NAMED BY THE CALLER, not by this module: which revisions a target's result depends on
    is a property of that target, and writing a pin name here would make this one target's driver. Each
    is verified by CONTENT through :mod:`merlin.common.provenance`, and a pin that fails verification is
    RECORDED as failing rather than dropped -- a result that quietly stopped citing its pin is exactly
    the silence this convention exists to break.
    """
    verified = {}
    for name in pins:
        try:
            verified[name] = P.verify(name)
        except Exception as exc:                    # noqa: BLE001 -- an unknown pin is not a clean run
            return {"unavailable": f"pin {name!r}: {type(exc).__name__}: {exc}"}
    return P.record(pins=verified, sources=[header],
                    extra={"counter_header": header,
                           "pins_declared": sorted(pins),
                           "pins_note": ("no pin was declared, so which hardware revision these "
                                         "counters came from is UNRECORDED -- pass --pin"
                                         if not pins else "")})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read a target's own combination performance counters "
                                             "around a workload running on its elaborated RTL.")
    ap.add_argument("--target", required=True)
    ap.add_argument("--shape", action="append", default=None,
                    help="MxKxN; repeatable (default: two sizes, since one point cannot separate a "
                         "rate from a fixed intercept)")
    ap.add_argument("--simulator", default="verilator",
                    help="the elaborated-RTL engine to run on (the counters are RTL state; a "
                         "functional model does not have them)")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--workdir", type=Path, default=None,
                    help="where to build the ELFs (default: a temporary directory)")
    ap.add_argument("--pin", action="append", default=None, metavar="NAME",
                    help="a hardware pin from merlin/contract/hardware_pins.yaml that this reading is "
                         "ABOUT; repeatable. Verified by content and recorded. Without one the record "
                         "says the revision is UNRECORDED rather than implying it was checked")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = ap.parse_args(argv)

    shapes = args.shape or ["16x16x16", "32x32x32"]
    block = counter_block(args.target)
    print(f"counters derived from {block['header']}")
    print(f"  engines={block['counters']['engines']} "
          f"complete={block['counters']['complete']}")

    import tempfile
    holder = None
    if args.workdir is None:
        holder = tempfile.TemporaryDirectory(prefix="merlin_counter_occupancy_")
        root = Path(holder.name)
    else:
        root = args.workdir
        root.mkdir(parents=True, exist_ok=True)

    readings = []
    try:
        for sh in shapes:
            m, k, n = (int(v) for v in sh.split("x"))
            work = root / sh
            work.mkdir(parents=True, exist_ok=True)
            got = measure(args.target, matmul_cb(args.target, m, k, n), workload=f"matmul_{sh}",
                          simulator=args.simulator, workdir=work, timeout=args.timeout)
            readings.append(got)
            print(f"{sh:14s} bit_exact={got['bit_exact']} cycles={got['total_cycles']} "
                  f"elapsed={got['elapsed_s']}s counters={len(got['values'])}")
            if got.get("dropped"):
                print(f"               DROPPED: {got['dropped'][:160]}")
    finally:
        if holder is not None:
            holder.cleanup()

    record = {"schema_version": 1, "kind": "counter_occupancy", "target": args.target,
              "provenance": provenance(args.target, list(args.pin or ()), block["header"]),
              "counter_block": block, "readings": readings,
              "n_usable": sum(1 for r in readings if r["values"])}
    if args.dry_run:
        print(json.dumps(record, indent=1))
        print("\n--dry-run: nothing written")
        return 0

    pd = A.new_product(TOPIC, version=PRODUCT_VERSION, target=args.target,
                       notes="joint occupancy from the target's own combination counters")
    out = pd.add_artifact(RECORD_NAME)
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    pd.write_manifest()
    print(f"\nwrote {out}")
    return 0 if record["n_usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
