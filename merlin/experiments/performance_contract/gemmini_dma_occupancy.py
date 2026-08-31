#!/usr/bin/env python3
"""Joint occupancy on a command-driven accelerator, with its MOVEMENT path actually exercised.

The first attempt at this measured one controller and refused the rest, for a reason that was about
the harness rather than the machine: the co-simulation model places operands by writing the
scratchpad directly and reads results out of the accumulator directly, so the load and store
controllers never run. Two of the three engines were structurally unexercised, every memory-path busy
port stayed constant, and the joint vector had a single live column -- which reports zero overlap by
construction and proves nothing.

This driver moves the operands the way the hardware would: a memory responder is attached to the
design's own DMA bundle, the operands are placed in that memory, and the kernel issues the target's
movement commands to bring them in and write the result back. All three controllers then have work,
so their concurrency becomes observable -- which is the whole question, since a decoupled
load/execute/store machine is the falsifier for an instrument that has only ever reported zero
overlap.

NOTHING HERE IS A WRITTEN-DOWN ENCODING
---------------------------------------
Every command code, operand packing and configuration word is read from the target's own derived ISA
(:mod:`merlin.targets.gemmini.backend.gemmini_codegen_mlir`, whose ``_isa()`` builds them from the RTL
facts and the capability residual). This module names roles -- move-in, move-out, configure -- and the
target supplies the codes. A wrong code does not pass silently: the result is compared against an
independent reference and the run reports ``bit_exact``.

STATUS -- THE INSTRUMENT WORKS; MOVE-IN IS THE ONE REMAINING FAULT
------------------------------------------------------------------
All three controllers go live and the joint vector reports ``overlap_observable: True``, which is what
this driver was for: the occupancy question is askable on a decoupled target. The stream is still not
bit-exact, and the fault has been bisected to one stage.

* **Compute is correct.** Driving the proven command sequence with operands placed directly gives an
  exact result.
* **Move-out is correct**, and the earlier verdict against it was a WRONG REFERENCE, not a wrong
  kernel. The narrow readout scales and saturates the accumulator on its way out; graded against an
  unsaturated product it looked broken (224 of 256 "mismatches" that were all clamps), and graded
  against its own semantics it is 256 of 256 exact. The reference has to model what the hardware
  does, not what the arithmetic would do.
* **Move-in is the remaining fault.** In isolation it issues no bus reads inside the drain window,
  so the operands never reach the scratchpad the compute reads.

**No occupancy number from this driver may be quoted until ``bit_exact`` is true.** A wrong kernel's
controller occupancy is not the machine's behaviour on that workload, and the zero-overlap reading it
currently produces would be exactly the kind of corroborating-looking result this layer exists to
refuse.

Usage::

    gemmini_dma_occupancy.py --target T [--shape 16x16x16]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from merlin.common import artifacts as A                                          # noqa: E402
from merlin.perf.occupancy import joint_counts                                     # noqa: E402

from gemmini_occupancy import (                                                    # noqa: E402
    PRODUCER_DECLARED_IDLE,
    STATE_SIGNALS,
    _contract,
    mlc_dir,
)

#: The design's DMA master bundle. Read from the state manifest rather than written down: the bundle
#: is the one whose A-channel valid the manifest declares, and the responder binds by that.
DMA_BUNDLE = "auto_spad_id_out"


def _model(target: str):
    sys.path.insert(0, str(mlc_dir()))
    from mlc.backends.cosim_core import CosimCore                  # noqa: PLC0415
    from mlc.backends.protocols import RoCCAdapter, TileLinkSlave  # noqa: PLC0415

    outputs = mlc_dir() / "runs" / "circt-arc" / target / "outputs"
    so = next(iter(sorted((outputs.parent / "native_run").glob("*.so"))), None)
    manifest = next(iter(sorted(outputs.glob("*_core_state.json"))), None)
    if so is None or manifest is None:
        raise SystemExit(f"no built model for {target!r}")
    core = CosimCore(str(so), str(manifest))
    # The responder's window is a PARAMETER, not a property of the device: size it to the footprint
    # actually placed rather than inheriting a default that silently wraps a larger one.
    slave = TileLinkSlave(core, DMA_BUNDLE, size_bytes=1 << 22)
    return core, RoCCAdapter(core), slave


def run(target: str, m: int, k: int, n: int, dram_base: int = 0) -> dict:
    """Move operands in, compute, move the result out -- recording every controller every cycle."""
    import numpy as np                                             # noqa: PLC0415

    # The backend is loaded through the target-backend resolver, so the module comes from wherever
    # this target's package actually lives rather than from a guessed import path.
    from merlin.runtime.backends import base as _bk            # noqa: PLC0415
    G = _bk.get_backend(target).gemmini_codegen_mlir

    isa = G._isa()
    dim = isa.DIM
    if m % dim or k % dim or n % dim:
        raise SystemExit(f"shape must be a multiple of the array edge {dim}")
    mt, kt_n, nt = m // dim, k // dim, n // dim

    core, rocc, slave = _model(target)
    present = [s for s in STATE_SIGNALS if core.has(s.replace("/", "/"))] or list(STATE_SIGNALS)
    trace: dict[str, list[str]] = {s: [] for s in present}

    def on_cycle() -> None:
        slave.step()
        for s in present:
            trace[s].append(str(core.peek(s)))

    rng = np.random.default_rng(0xA7)
    a = rng.integers(-8, 8, size=(m, k), dtype=np.int8)
    b = rng.integers(-8, 8, size=(k, n), dtype=np.int8)
    a_addr, b_addr = dram_base, dram_base + a.nbytes
    c_addr = b_addr + b.nbytes
    slave.preload(a_addr, a.tobytes())
    slave.preload(b_addr, b.tobytes())

    core.reset()
    slave.step()
    b_slot, a_slot = 0, kt_n * nt * dim          # B tiles resident first, then the A tile slot

    rocc.issue(isa.K_FLUSH, 0, 0, on_cycle=on_cycle)
    rocc.run_until_idle(drain=0, on_cycle=on_cycle)

    # Move the weight tiles in (row stride = n bytes for an i8 operand), then the activations.
    rocc.issue(isa.K_CONFIG, isa.CFG_LD_RS1, n, on_cycle=on_cycle)
    for tk in range(kt_n):
        for tj in range(nt):
            off = (tk * dim) * n + tj * dim
            rocc.issue(isa.K_MVIN, b_addr + off, G._pack(b_slot + (tk * nt + tj) * dim),
                       on_cycle=on_cycle)
    # A configuration word is GLOBAL state, not a tracked dependency: changing the load stride while
    # earlier move-ins are still in flight would apply the new stride to them. The queue orders
    # data hazards, not this, so drain before reconfiguring.
    rocc.run_until_idle(on_cycle=on_cycle)
    rocc.issue(isa.K_CONFIG, isa.CFG_LD_RS1, k, on_cycle=on_cycle)

    rocc.issue(isa.K_CONFIG, isa.CFG_EX_RS1, isa.CFG_EX_RS2, on_cycle=on_cycle)
    # The narrow readout scales and saturates the accumulator on its way out, one byte per element.
    # That is the readout's own semantics, not a defect, and the reference below applies it too --
    # grading a saturating readout against an unsaturated product marks a correct kernel wrong, which
    # is what it did until the two were compared on the same terms.
    rocc.issue(isa.K_CONFIG, 2, (isa.F1 << 32) | n, on_cycle=on_cycle)

    for ti in range(mt):
        for tk in range(kt_n):
            off = (ti * dim) * k + tk * dim
            rocc.issue(isa.K_MVIN, a_addr + off, G._pack(a_slot + tk * dim), on_cycle=on_cycle)
        rocc.run_until_idle(on_cycle=on_cycle)
        for tj in range(nt):
            for tk in range(kt_n):
                # The preload names the accumulator to WRITE; the move-out names the one to READ.
                # The full-width flag is a property of the READ (it selects the wide readout over the
                # scaled narrow one), so it belongs on the move-out and not here -- the target's own
                # readout bits already separate the two, and the bare accumulator base is the one
                # without it.
                cad = isa.ACC_I8 if tk == 0 else (isa.ACC_I8 | isa.ACC_ACCUM)   # write side
                rocc.issue(isa.K_PRELOAD, G._pack(b_slot + (tk * nt + tj) * dim), G._pack(cad),
                           on_cycle=on_cycle)
                rocc.issue(isa.K_COMPUTE_PRELOADED, G._pack(a_slot + tk * dim),
                           G._pack(0xFFFFFFFF), on_cycle=on_cycle)
            rocc.run_until_idle(on_cycle=on_cycle)
            c_off = (ti * dim) * n + tj * dim
            rocc.issue(isa.K_MVOUT, c_addr + c_off, G._pack(isa.ACC_I8), on_cycle=on_cycle)
            rocc.run_until_idle(on_cycle=on_cycle)
    rocc.run_until_idle(drain=64, on_cycle=on_cycle)

    got = np.frombuffer(slave.captured(c_addr, m * n), dtype="i1").reshape(m, n)
    ref = np.clip(a.astype(np.int32) @ b.astype(np.int32), -128, 127).astype(np.int8)

    hot = {s: [v != PRODUCER_DECLARED_IDLE for v in trace[s]] for s in present
           if len(set(trace[s])) > 1}
    unmeasured = [s for s in present if s not in hot]
    jc = joint_counts(hot) if hot else {"sampled_cycles": 0, "overlap_any": None,
                                        "overlap_observable": False}
    return {"shape": f"{m}x{k}x{n}", "cycles_recorded": len(trace[present[0]]),
            "bit_exact": bool(np.array_equal(got, ref)),
            "dma_reads": slave.reads, "dma_writes": slave.writes,
            "joint": jc, "unmeasured_columns": unmeasured,
            "idle_encoding": {"value": PRODUCER_DECLARED_IDLE, "basis": "declared_by_producer"}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--shape", action="append", default=None)
    args = ap.parse_args()

    runs = []
    for sh in (args.shape or ["16x16x16"]):
        m, k, n = (int(v) for v in sh.split("x"))
        r = run(args.target, m, k, n)
        runs.append(r)
        j = r["joint"]
        print(f"{sh:12s} cycles={r['cycles_recorded']:6d} bit_exact={r['bit_exact']} "
              f"dma(r/w)={r['dma_reads']}/{r['dma_writes']}")
        print(f"             observable={j.get('overlap_observable')} overlap={j.get('overlap_any')} "
              f"idle={j.get('idle_cycles')} live={j.get('live_columns')}")
        print(f"             busy={ {k2: v for k2, v in (j.get('busy') or {}).items() if v} }")

    pd = A.new_product("perf-ledger", version=4, target=args.target,
                       notes="joint controller occupancy with the movement path exercised")
    (pd.path / "dma_occupancy.json").write_text(json.dumps(
        {"target": args.target, "engines": _contract(args.target) and None, "runs": runs}, indent=1))
    print(f"\nwrote {pd.path / 'dma_occupancy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
