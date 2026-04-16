"""Repair manifest kernels whose ``addi imm`` exceeds the signed 12-bit range.

Rewrites two patterns in place:

  ``addi rd, x0, K`` with K outside [-2048, 2047]
      → ``lui rd, hi(K); addi rd, rd, lo(K)``

  ``addi rd, rd, K`` with K outside [-2048, 2047]
      → ``lui SCRATCH, hi(K); addi SCRATCH, SCRATCH, lo(K); add rd, rd, SCRATCH``

SCRATCH = x31 (no kernel uses regs above x21 today).

Idempotent. Updates ``num_instructions`` and clears ``patch_points`` so the
annotator picks up the new indices on the next run.
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST = Path("/scratch2/agustin/merlin/benchmarks/SaturnNPU/kernel_library/manifest.json")
SCRATCH_REG = 31


def _split(value: int) -> tuple[int, int]:
    hi = (value + 0x800) >> 12
    lo = value - (hi << 12)
    assert -2048 <= lo <= 2047, (value, hi, lo)
    return hi, lo


def _wrap(mnemonic: str, base: dict, **fields) -> dict:
    args = {k: v for k, v in base.items() if k not in fields and k != "imm"}
    args.update(fields)
    return {"mnemonic": mnemonic, "args": args}


def _expand_inst(inst: dict) -> list[dict]:
    if inst["mnemonic"] != "addi":
        return [inst]
    args = inst.get("args", {})
    imm = args.get("imm", 0)
    if -2048 <= imm <= 2047:
        return [inst]
    rd = args.get("rd", 0)
    rs1 = args.get("rs1", 0)
    hi, lo = _split(imm)
    if rs1 == 0:
        out = [_wrap("lui", args, rd=rd, rs1=0, imm=hi)]
        if lo != 0:
            out.append(_wrap("addi", args, rd=rd, rs1=rd, imm=lo))
        return out
    # rs1 == rd cumulative-bump cases are LEFT UNCHANGED. The autocomp
    # generator relied on RISC-V's 12-bit sign-extension semantics for those
    # (e.g. imm=2048 sign-extends to -2048, which gives the intended 2048-byte
    # transfer size when added to a 4096-base register). Rewriting them to
    # literal add would break that arithmetic.
    if rs1 == rd:
        return [inst]
    raise ValueError(f"unhandled overflowing addi: rd={rd}, rs1={rs1}, imm={imm}")


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    rewritten_total = 0
    for name, kernel in manifest["kernels"].items():
        old = kernel["instructions"]
        new: list[dict] = []
        rewritten = 0
        for inst in old:
            expanded = _expand_inst(inst)
            if len(expanded) != 1:
                rewritten += 1
            new.extend(expanded)
        if rewritten:
            kernel["instructions"] = new
            kernel["num_instructions"] = len(new)
            kernel["patch_points"] = []
            print(f"  {name}: rewrote {rewritten} addi op(s); " f"{len(old)} → {len(new)} insts")
            rewritten_total += rewritten

    if rewritten_total:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nrewrote {rewritten_total} addi op(s) across the manifest")
    else:
        print("no overflowing addi found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
