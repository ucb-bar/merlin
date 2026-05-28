#!/usr/bin/env python3
"""Diff dronet.q.int8.with_intermediate outputs between gemmini and a
reference path, to localize which dispatch first diverges on Gemmini.

Strategy
--------
The `dronet.q.int8.with_intermediate.onnx` model exposes 4 outputs:
  steer, collision, steer_QuantizeLinear_Input, linear_1
which give us 4 checkpoints inside the model. We compare each checkpoint
between:

  - ONNX runtime (golden — pure CPU fp32 emulation of the quantized model)
  - The gemmini-compiled VMFB run via host IREE Python bindings

If the host IREE run produces matching intermediates for both, the bug is
HW-only (Gemmini RTL diverges from libgemmini). If the IREE run diverges
at intermediate i, we know which dispatch range (i-1 → i) contains the
buggy code, and we can dig into just that subgraph.

Input is hardcoded to all-uint8(127) (matches the dronet quantized
input domain). Outputs are hashed by xxhash64.

Usage:
  conda run -n merlin-dev uv run python3 tools/diff_dronet_dispatches.py
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path("/scratch2/agustin/merlin")
ONNX_PATH = ROOT / "models/dronet/dronet.q.int8.with_intermediate.onnx"
GEMMINI_VMFB = (
    ROOT
    / "build/compiled_models/dronet/firesim_shuttle_gemmini_Gemmini_dronet.q.int8.with_intermediate"
    / "dronet.q.int8.with_intermediate.vmfb"
)


def hash_tensor(arr: np.ndarray) -> str:
    h = hashlib.sha256(arr.tobytes()).hexdigest()
    return f"0x{h[:16]}"


def main() -> int:
    if not ONNX_PATH.exists():
        print(f"missing {ONNX_PATH}", file=sys.stderr)
        return 2
    if not GEMMINI_VMFB.exists():
        print(f"missing {GEMMINI_VMFB}", file=sys.stderr)
        return 2

    # Deterministic input: all-ones in fp32 (dronet input is fp32 -> internal quantize)
    inp = np.ones((1, 3, 112, 112), dtype=np.float32)

    # ------- Golden: ONNX runtime -------
    print(f"=== ONNX runtime golden ({ONNX_PATH.name}) ===")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    print(f"  inputs : {in_name} {sess.get_inputs()[0].shape} {sess.get_inputs()[0].type}")
    print(f"  outputs: {out_names}")
    golden = sess.run(None, {in_name: inp})
    print()
    print(f"{'output':40s} {'shape':20s} {'dtype':10s} {'hash':18s}")
    print("-" * 95)
    for name, arr in zip(out_names, golden):
        print(f"{name:40s} {str(arr.shape):20s} {str(arr.dtype):10s} {hash_tensor(arr)}")

    # ------- IREE host runtime: try to load gemmini VMFB -------
    print()
    print("=== attempt to run Gemmini-compiled VMFB on host IREE runtime ===")
    print(f"  vmfb: {GEMMINI_VMFB}")
    try:
        import iree.runtime as irt  # type: ignore

        # The vmfb targets riscv64-unknown-elf, not the host. The host runtime
        # cannot load these dispatch ELFs — but we can still parse the bytecode
        # module to see its exported function signatures and any embedded data.
        try:
            with open(GEMMINI_VMFB, "rb") as f:
                vmfb_bytes = f.read()
            config = irt.Config("local-sync")
            ctx = irt.SystemContext(config=config)
            ctx.add_vm_module(irt.VmModule.from_flatbuffer_blob(config.vm_instance, vmfb_bytes))
            module = ctx.modules.module
            for fn_name in dir(module):
                if fn_name.startswith("_"):
                    continue
                print(f"  exported: {fn_name}")
            # Attempting an actual invoke will fail because dispatch ELFs are
            # riscv64. We just report what's reachable.
        except Exception as e:
            print(f"  host runtime cannot execute riscv64 dispatch ELFs (expected): {type(e).__name__}: {e}")
    except ImportError:
        print("  iree.runtime not installed; skipping host-IREE step")

    # ------- Plan B: extract per-output reference for FireSim comparison -------
    print()
    print("=== golden output values for FireSim comparison ===")
    print("Run dronet × Gemmini × FireSim with the with_intermediate VMFB (after")
    print("updating MERLIN_OUTPUT_COUNT to 4 in merlin_hetero_runner). Compare each")
    print("output[i]'s sha256-prefix against the values above. The first row that")
    print("mismatches identifies which dispatch range produced wrong output:")
    print()
    print("  outputs[0] = steer     (final FC head 0)")
    print("  outputs[1] = collision (final FC head 1)")
    print("  outputs[2] = steer_QuantizeLinear_Input (last activation before FC)")
    print("  outputs[3] = linear_1  (penultimate dense)")
    print()
    print("If outputs[3] (linear_1) already differs → the bug is in the conv stack.")
    print("If outputs[3] matches but outputs[2] doesn't → bug is in last conv/quantize.")
    print("If outputs[2] matches but outputs[0/1] don't → bug is in the 1×1×2048 FC heads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
