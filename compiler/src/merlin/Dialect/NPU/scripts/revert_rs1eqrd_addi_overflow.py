"""Revert my `rs1==rd` addi-overflow transforms in the manifest.

The fix_manifest_addi_overflow.py script transformed
    addi rd=X, rs1=X, imm=K     (K > 2047)
into
    lui rd=31, imm=hi
    [addi rd=31, rs1=31, imm=lo]    (omitted when lo == 0)
    add rd=X, rs1=X, rs2=31

This matches the LITERAL interpretation (x_new = x_old + K). But the autocomp
tool that generated the manifest appears to have relied on RISC-V's sign-
extended interpretation (x_new = x_old + (K - 4096)) for the rs1==rd cases —
e.g. a 2048-byte transfer size encoded as `addi rd=7, rs1=7, imm=2048` sign-
extends to -2048 and gives x_new = 4096 + (-2048) = 2048.

So for rs1==rd cases, the original single-instruction encoding was correct on
the simulator. This script reverses the triplet back to one addi. Idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path("/scratch2/agustin/merlin/benchmarks/SaturnNPU/kernel_library/manifest.json")
SCRATCH_REG = 31


def _try_match_triplet(insts: list[dict], i: int) -> tuple[int, dict] | None:
    """At position i, look for a lui[, addi], add triplet that my fix injected.

    Returns (length, replacement_addi) on match, None otherwise.
    """
    if i >= len(insts):
        return None
    a = insts[i]
    if a["mnemonic"] != "lui":
        return None
    args = a.get("args", {})
    if args.get("rd") != SCRATCH_REG or args.get("rs1", 0) != 0:
        return None
    hi = args.get("imm", 0)

    j = i + 1
    lo = 0
    if (
        j < len(insts)
        and insts[j]["mnemonic"] == "addi"
        and insts[j].get("args", {}).get("rd") == SCRATCH_REG
        and insts[j].get("args", {}).get("rs1") == SCRATCH_REG
    ):
        lo = insts[j]["args"].get("imm", 0)
        j += 1

    if j >= len(insts) or insts[j]["mnemonic"] != "add":
        return None
    add_args = insts[j].get("args", {})
    if add_args.get("rs2") != SCRATCH_REG:
        return None
    rd = add_args.get("rd")
    rs1 = add_args.get("rs1")
    if rd is None or rd != rs1:
        return None

    original_imm = (hi << 12) + lo
    # Only fire when the original imm would have been out of 12-bit range,
    # otherwise the triplet may have been hand-written and is not ours.
    if -2048 <= original_imm <= 2047:
        return None

    replacement = {
        "mnemonic": "addi",
        "args": {
            **{k: v for k, v in add_args.items() if k != "rs2"},
            "rd": rd,
            "rs1": rd,
            "imm": original_imm,
        },
    }
    return (j + 1 - i, replacement)


def _revert(insts: list[dict]) -> tuple[list[dict], int]:
    out: list[dict] = []
    reverts = 0
    i = 0
    while i < len(insts):
        match = _try_match_triplet(insts, i)
        if match is not None:
            length, new_inst = match
            out.append(new_inst)
            reverts += 1
            i += length
            continue
        out.append(insts[i])
        i += 1
    return out, reverts


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    total_reverts = 0
    for name, kernel in manifest["kernels"].items():
        new_insts, reverts = _revert(list(kernel["instructions"]))
        if reverts:
            kernel["instructions"] = new_insts
            kernel["num_instructions"] = len(new_insts)
            kernel["patch_points"] = []
            print(f"  {name}: reverted {reverts} lui/add triplet(s); " f"{len(kernel['instructions'])} insts")
            total_reverts += reverts

    if total_reverts:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nreverted {total_reverts} triplets")
    else:
        print("no triplets to revert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
