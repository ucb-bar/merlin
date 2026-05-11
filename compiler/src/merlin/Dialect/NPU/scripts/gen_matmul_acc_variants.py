"""Synthesize matmul_acc_first/mid/last manifest entries from the existing matmul.

The matmul kernel performs C = A @ B in 26 instructions. The K-tiled variants
share the same scalar+DMA setup and weight push, but differ in:
  - which vmatmul flavor is issued (overwrite vs accumulate),
  - whether the MXU accumulator is popped + stored (only the LAST iter does).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

MANIFEST = Path("/scratch2/agustin/merlin/benchmarks/SaturnNPU/kernel_library/manifest.json")


def _drop_keys(obj: dict, keys: tuple[str, ...]) -> dict:
    return {k: v for k, v in obj.items() if k not in keys}


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    if "matmul" not in manifest["kernels"]:
        print("ERROR: base 'matmul' kernel missing")
        return 1

    base = manifest["kernels"]["matmul"]
    base_insts = base["instructions"]

    # Sanity-check the instruction layout we depend on.
    expected_vmatmul_idx = 17
    if (
        base_insts[expected_vmatmul_idx]["mnemonic"] != "vmatmul.mxu0"
        or base_insts[19]["mnemonic"] != "vmatpop.bf16.acc.mxu0"
    ):
        print("ERROR: matmul kernel layout has changed; refusing to synthesize")
        return 1

    # Common prefix: scalar setup + DMA loads + waits + vload + vmatpush.weight
    # spans indices [0..16].
    PREFIX = copy.deepcopy(base_insts[:expected_vmatmul_idx])
    # Output suffix used only by `_last`: delay + vmatpop + vstores + dma.store +
    # dma.wait spans [18..24].
    OUTPUT_SUFFIX = copy.deepcopy(base_insts[expected_vmatmul_idx + 1 : -1])
    ECALL = copy.deepcopy(base_insts[-1])
    assert ECALL["mnemonic"] == "ecall"

    def _matmul_op(mnemonic: str) -> dict:
        return copy.deepcopy(
            {
                "mnemonic": mnemonic,
                "args": base_insts[expected_vmatmul_idx]["args"],
            }
        )

    # _first: overwrite-mode multiply, no pop/store. Accumulator now holds
    # the partial product for the first K-tile.
    first = PREFIX + [_matmul_op("vmatmul.mxu0"), ECALL]
    # _mid: accumulate-add into the running MXU sum, no pop/store.
    mid = PREFIX + [_matmul_op("vmatmul.acc.mxu0"), ECALL]
    # _last: accumulate-add, then drain accumulator to MRF, vstore, DMA store.
    last = PREFIX + [_matmul_op("vmatmul.acc.mxu0")] + OUTPUT_SUFFIX + [ECALL]

    # Per-variant manifest entries. tile_shape and layouts mirror the base
    # matmul; the patch_points block is left empty here and populated by
    # annotate_kernel_patch_points.py downstream.
    base_meta = _drop_keys(
        base, ("instructions", "patch_points", "num_instructions", "cycles", "symbol_prefix", "source_file")
    )

    def _variant_entry(suffix: str, instructions: list[dict]) -> dict:
        entry = copy.deepcopy(base_meta)
        entry["symbol_prefix"] = f"npu_uk_matmul_acc_{suffix}"
        entry["source_file"] = "kernel_library/matmul_acc.py"
        entry["num_instructions"] = len(instructions)
        # Cycle estimate: drop the ~30-cycle delay + drain/store/DMA tail
        # (~7 instructions / ~7-10 cycles) for first/mid; keep the base count
        # for last.
        entry["cycles"] = base["cycles"] if suffix == "last" else int(base["cycles"]) - 30
        entry["patch_points"] = []
        entry["instructions"] = instructions
        return entry

    manifest["kernels"]["matmul_acc_first"] = _variant_entry("first", first)
    manifest["kernels"]["matmul_acc_mid"] = _variant_entry("mid", mid)
    manifest["kernels"]["matmul_acc_last"] = _variant_entry("last", last)

    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote 3 matmul_acc_* entries to {MANIFEST}")
    print(f"  first: {len(first)} insts, mid: {len(mid)} insts, last: {len(last)} insts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
