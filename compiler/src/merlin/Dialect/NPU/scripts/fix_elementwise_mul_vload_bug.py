"""Fix the elementwise_mul kernel's VMEM register mismatch.

The DMA load writes B to VMEM register x2 (0x900), but the subsequent
vloads read from VMEM register x5 (0x800) — the DRAM register, not the
VMEM register. Swap rs1=5 → rs1=2 on the B-half vloads.
"""

import json
from pathlib import Path

MANIFEST = Path("/scratch2/agustin/merlin/benchmarks/SaturnNPU/kernel_library/manifest.json")
manifest = json.loads(MANIFEST.read_text())
kernel = manifest["kernels"]["elementwise_mul"]
insts = kernel["instructions"]
patched = 0
for i, ins in enumerate(insts):
    if ins["mnemonic"] != "vload":
        continue
    args = ins["args"]
    # The two problematic vloads are the ones reading from x5 (DRAM B reg)
    # immediately after the B DMA — they should read from x2 (VMEM B reg).
    if args.get("rs1") == 5:
        args["rs1"] = 2
        patched += 1
if patched:
    kernel["patch_points"] = []  # will be re-annotated
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"patched {patched} vloads in elementwise_mul")
else:
    print("no patch applied")
