#!/usr/bin/env python
"""Build one Zephyr FireSim leg -- a DEVICE image (contractions routed to a matrix unit) or its
CONTROL twin (identical lowering, nothing routed) -- and audit that the image is what it claims to be.

WHY THIS FILE EXISTS AT ALL. Its predecessor, ``fs_legs.py``, is named in the provenance ``sources`` of
the FireSim whole-model matrix-unit result and NO COPY SURVIVES: it was a session scratchpad that was
later purged, so the cited number cannot be reproduced. Its successor then lived as a loose
``zeph_leg.py`` under ``/scratch/agustin/tmp`` and was the ONLY driver for building these legs. Twice
now the script behind a cited number has been a temp file. Committing it is the fix.

WHY ZEPHYR RATHER THAN BARE METAL, measured rather than preferred: it already carries the ALIVE
heartbeat (``zephyr_model`` DEBUG_HEARTBEAT_S) so a hung run is distinguishable from a slow one, it has
a debug twin, it reaches the second vector hart through the OpenMP shim (1.765x on TinyLlama), and it is
the runtime both the TinyLlama reference and the hardware rig use -- so a bare-metal-vs-Zephyr
comparison across two models was never valid in the first place.

TARGET-AGNOSTIC. The matrix unit and its configuration are REQUIRED ARGUMENTS, not literals: the
original script hardcoded one unit name and one config, which is exactly the overfit this repo's
cardinal rule forbids (and this directory is a ``check_no_target_name.py`` scan root). Everything about
the unit -- its instruction encodings, and therefore the audit -- is derived from its own contract via
``opu_shim.derive_encodings(opu_shim.load_contract(unit))``.

THE AUDIT IS THE POINT. A device leg that routed nothing, and a control leg that accidentally routed
something, both produce plausible numbers and a successful build. So the image is disassembled and its
unit-instruction counts are checked against what the leg CLAIMS: device must carry them, control must
carry none. Either way it fails closed rather than handing back a leg that measures the wrong thing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin" / "python"))


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(prog="zephyr_firesim_leg", description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle", help="capture bundle name under out/artifacts/recaptures/")
    ap.add_argument("leg", choices=("device", "control"), help="route to the matrix unit, or not")
    ap.add_argument("outdir")
    ap.add_argument("--unit", required=True,
                    help="matrix unit name, as its contract names it (derives the encodings + audit)")
    ap.add_argument("--config", required=True,
                    help="the unit's hardware configuration (the generator config the bitstream built)")
    ap.add_argument("--board", required=True,
                    help="board descriptor; it declares the DRAM the region check prices against. "
                         "ram_bytes_override alone does NOT help -- _region_overlay checks the request "
                         "against the BOARD's declared DRAM, so the descriptor is the fix (measured: a "
                         "3584 MB region was correctly refused against a 256 MB declaration)")
    ap.add_argument("--package", required=True,
                    help="RVV package directory under out/artifacts/targets/ (schedule + cflags)")
    ap.add_argument("--harts", type=int, default=2)
    ap.add_argument("--vlen", type=int, default=512)
    ap.add_argument("--arena-mb", type=int, default=0,
                    help="0 = derive from the model's activation peak (see below)")
    a = ap.parse_args(argv)

    from merlin.common.ir_lock import IR_LOCK
    from merlin.common.mlir_query import activation_peak_bytes
    from merlin.common.paths import repo_root
    from merlin.kernels.decode import opu as opu_audit
    from merlin.llvmlower import opu_shim
    from merlin.llvmlower.impr_features import PEROP_BLOCK_NAME, OPU_MATMUL_NAME
    from merlin.mining.registry import load_rvv_package
    from merlin.runtime.backends import zephyr_model as zm

    bundle = repo_root() / "out/artifacts/recaptures" / a.bundle
    if not (bundle / "model.mlir").is_file():
        raise SystemExit(f"[leg] {bundle} is not a capture bundle (no model.mlir)")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pkg = load_rvv_package(repo_root() / "out/artifacts/targets" / a.package)

    with IR_LOCK:
        peak = activation_peak_bytes(bundle / "model.mlir")
    # `free()` is a no-op in this runtime, so the arena must cover the SUM of allocations, not the
    # liveness peak. Provision generously; the post-build region check fails closed if still short.
    arena_mb = a.arena_mb or max(
        512, (int(peak or 0) * 3 // 2 + 512 * 1024 * 1024 + 2 ** 20 - 1) // 2 ** 20)

    routed = a.leg == "device"
    feats = {PEROP_BLOCK_NAME} | ({OPU_MATMUL_NAME} if routed else set())
    kw = dict(matrix=zm.MatrixRouting(unit=a.unit, config=a.config)) if routed else {}

    t0 = time.time()
    b = zm.build_app(bundle, str(out), board=a.board, backend="rvv", rvv_hart=0,
                     cpus=max(2, a.harts), n_harts=a.harts, int8_compute=True,
                     features=frozenset(feats), rvv_schedule=pkg.schedule_text, arena_mb=arena_mb,
                     cflags_override=pkg.cflags + zm._CFLAGS_COMMON, vlen=a.vlen,
                     inputs_npz=bundle / "inputs.npz", debug=True, **kw)
    print(f"[leg] built ZEPHYR {a.leg} {a.bundle} harts={a.harts} vlen={a.vlen} arena={arena_mb}MB "
          f"in {time.time() - t0:.0f}s hash={b.get('build_hash')} ram={b.get('ram_bytes')} "
          f"elf={b.get('elf')}", flush=True)

    # THE AUDIT. Encodings derived from the unit's own contract, never a literal opcode.
    enc = opu_shim.derive_encodings(opu_shim.load_contract(a.unit)).encodings
    counts = {k: int(v) for k, v in sorted((opu_audit.audit_object(b["elf"], enc).counts or {}).items())
              if v}
    print(f"[leg] unit instruction counts: {counts or 'NONE'}", flush=True)
    if routed and not counts:
        raise SystemExit("[leg] FAIL: device image carries NONE of the unit's instructions — it would "
                         "have measured the unrouted lowering under the device leg's name")
    if not routed and counts:
        raise SystemExit(f"[leg] FAIL: control image CARRIES the unit's instructions ({counts}) — it is "
                         "not a control")
    print("[leg] AUDIT OK")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
