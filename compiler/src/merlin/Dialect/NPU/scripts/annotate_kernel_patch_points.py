#!/usr/bin/env python3
"""Annotate patch_points in benchmarks/SaturnNPU/kernel_library/manifest.json.

For each kernel, walks its instruction list, identifies the contiguous scalar
prefix that sets up DRAM addresses + transfer sizes, and records per-DMA-role
metadata the compiler can use to re-emit scalar setup per invocation.

Output format (per kernel, added to manifest.json):
    "scalar_prefix_length": <int>,   # leading addi/lui ops the compiler may
                                      # replace with its own setup
    "patch_points": [
        {
          "role": "dram_in_0",       # dram_in_N, dram_out_N, dram_const_N,
                                      # or transfer_size
          "register": 1,              # scalar reg number the role writes to
          "original_value": 0,        # the hardcoded value in the unpatched
                                      # kernel (for sanity check)
        },
        ...
    ]

Role assignment rules:
    - Each scalar register used as rs1 of a `dma.load.ch<N>` becomes
      "dram_in_K" (K = 0, 1, ... in order of first DMA load appearance).
    - Each scalar register used as rd of a `dma.store.ch<N>` becomes
      "dram_out_K".
    - Each scalar register used as rs2 of any DMA op (the transfer size) is
      annotated as "transfer_size" but NOT patched — it stays kernel-controlled.
    - Scalar registers that the kernel computes but never feeds into a DMA
      (e.g. VMEM ptrs) stay unpatched and are preserved by the compiler.

Usage:
    uv run python compiler/src/merlin/Dialect/NPU/scripts/annotate_kernel_patch_points.py

    # Dry run — show what would change without writing:
    uv run python compiler/.../annotate_kernel_patch_points.py --dry-run

Idempotent: running twice produces the same manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).resolve().parents[6] / "benchmarks" / "SaturnNPU" / "kernel_library" / "manifest.json"

# Scalar ops that write an architectural register (`rd`).
SCALAR_WRITING_MNEMONICS = {
    "addi",
    "lui",
    "slli",
    "srli",
    "srai",
    "andi",
    "ori",
    "add",
    "sub",
}


def simulate_scalar_registers(instructions: list[dict[str, Any]]) -> dict[int, int]:
    """Simulate the scalar register file through the instruction stream.

    Returns a map from register number → its final computed value at the point
    just before any DMA.load/store/wait. Only tracks the subset of scalar ops
    we see in kernel prefixes (addi, lui, chained combinations).
    """
    xrf: dict[int, int] = {0: 0}  # x0 is hardwired to 0 in RISC-V.
    for inst in instructions:
        mnemonic = inst["mnemonic"]
        args = inst.get("args", {})
        if mnemonic == "addi":
            rd = args.get("rd", 0)
            rs1 = args.get("rs1", 0)
            imm = args.get("imm", 0)
            # Sign-extend 12-bit immediate to match RISC-V / npu_model semantics.
            imm12 = imm & 0xFFF
            if imm12 & 0x800:
                imm12 -= 0x1000
            if rd != 0:
                xrf[rd] = xrf.get(rs1, 0) + imm12
        elif mnemonic == "lui":
            rd = args.get("rd", 0)
            imm = args.get("imm", 0)
            if rd != 0:
                xrf[rd] = imm << 12
        elif mnemonic == "add":
            rd = args.get("rd", 0)
            rs1 = args.get("rs1", 0)
            rs2 = args.get("rs2", 0)
            if rd != 0:
                xrf[rd] = xrf.get(rs1, 0) + xrf.get(rs2, 0)
        elif mnemonic == "sub":
            rd = args.get("rd", 0)
            rs1 = args.get("rs1", 0)
            rs2 = args.get("rs2", 0)
            if rd != 0:
                xrf[rd] = xrf.get(rs1, 0) - xrf.get(rs2, 0)
        # Stop simulating once the scalar prefix ends (i.e. we hit the first
        # non-scalar op).
    return xrf


def trace_register_chains(
    instructions: list[dict[str, Any]],
) -> dict[int, list[dict[int, dict[str, Any]]]]:
    """Per register, record the list of instruction chains that define it.

    Each chain is a list of {index, mnemonic, imm, rs1} entries in program
    order, ending at the last write before the register is consumed (by a DMA
    or otherwise). A register can have multiple chains if it gets reassigned
    between consumers.

    Returns: {reg_number: [chain, chain, ...]}
    """
    # For simplicity, we return a per-register history of all writing
    # instructions in order. The compiler will look up the chain ending at
    # the position of the DMA that consumes the register.
    history: dict[int, list[dict[str, Any]]] = {}
    for idx, inst in enumerate(instructions):
        mnemonic = inst["mnemonic"]
        if mnemonic not in SCALAR_WRITING_MNEMONICS:
            continue
        args = inst.get("args", {})
        rd = args.get("rd", 0)
        if rd == 0:
            continue
        entry = {
            "index": idx,
            "mnemonic": mnemonic,
            "imm": args.get("imm", 0),
            "rs1": args.get("rs1", 0),
        }
        history.setdefault(rd, []).append(entry)
    return history


def collect_patch_points(
    instructions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Walk the DMA ops and map scalar registers to DMA roles.

    Roles:
        dram_in_K  — scalar reg used as rs1 of the K-th unique dma.load reg
        dram_out_K — scalar reg used as rd  of the K-th unique dma.store reg
        transfer_size — scalar reg used as rs2 of any DMA op (informational
                         only — not patched; kernel controls transfer size)

    Each entry includes the list of instruction indices that compute the
    register's value up to its first DMA use, so the compiler can rewrite them
    in place.
    """
    reg_history = trace_register_chains(instructions)

    def value_at(idx: int, reg: int) -> int:
        """Value of `reg` after executing instructions[0..idx)."""
        return simulate_scalar_registers(instructions[:idx]).get(reg, 0)

    def chain_until(reg: int, before_idx: int) -> list[dict[str, Any]]:
        """All scalar writes to `reg` with index < before_idx."""
        return [e for e in reg_history.get(reg, []) if e["index"] < before_idx]

    dram_in_regs: list[tuple[int, int]] = []  # (reg, dma_idx) in first-use order
    dram_out_regs: list[tuple[int, int]] = []
    transfer_size_regs: dict[int, int] = {}  # reg -> first dma_idx
    seen_in: set[int] = set()
    seen_out: set[int] = set()

    for idx, inst in enumerate(instructions):
        mnemonic = inst["mnemonic"]
        args = inst.get("args", {})
        if mnemonic == "dma.load.ch<N>":
            src_reg = args.get("rs1", 0)
            size_reg = args.get("rs2", 0)
            if src_reg and src_reg not in seen_in:
                dram_in_regs.append((src_reg, idx))
                seen_in.add(src_reg)
            if size_reg and size_reg not in transfer_size_regs:
                transfer_size_regs[size_reg] = idx
        elif mnemonic == "dma.store.ch<N>":
            dst_reg = args.get("rd", 0)
            size_reg = args.get("rs2", 0)
            if dst_reg and dst_reg not in seen_out:
                dram_out_regs.append((dst_reg, idx))
                seen_out.add(dst_reg)
            if size_reg and size_reg not in transfer_size_regs:
                transfer_size_regs[size_reg] = idx

    points: list[dict[str, Any]] = []
    for k, (reg, dma_idx) in enumerate(dram_in_regs):
        chain = chain_until(reg, dma_idx)
        points.append(
            {
                "role": f"dram_in_{k}",
                "register": reg,
                "original_value": value_at(dma_idx, reg),
                "instructions": [e["index"] for e in chain],
            }
        )
    for k, (reg, dma_idx) in enumerate(dram_out_regs):
        chain = chain_until(reg, dma_idx)
        points.append(
            {
                "role": f"dram_out_{k}",
                "register": reg,
                "original_value": value_at(dma_idx, reg),
                "instructions": [e["index"] for e in chain],
            }
        )
    for reg, dma_idx in sorted(transfer_size_regs.items()):
        chain = chain_until(reg, dma_idx)
        points.append(
            {
                "role": "transfer_size",
                "register": reg,
                "original_value": value_at(dma_idx, reg),
                "instructions": [e["index"] for e in chain],
            }
        )
    return points


def annotate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a new manifest with patch_points populated for every kernel."""
    new = json.loads(json.dumps(manifest))  # deep copy
    for name, kernel in new["kernels"].items():
        instructions = kernel.get("instructions", [])
        kernel["patch_points"] = collect_patch_points(instructions)
    return new


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help=f"Path to manifest.json (default: {MANIFEST_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the annotation summary without writing the file.",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Error: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    manifest = json.loads(args.manifest.read_text())
    annotated = annotate_manifest(manifest)

    # Print summary.
    print(f"{'kernel':<22} #dram_in  #dram_out  original_values")
    print(f"{'-' * 22} {'-' * 8} {'-' * 9} {'-' * 40}")
    for name, kernel in annotated["kernels"].items():
        pts = kernel.get("patch_points", [])
        n_in = sum(1 for p in pts if p["role"].startswith("dram_in"))
        n_out = sum(1 for p in pts if p["role"].startswith("dram_out"))
        vals = [f"{p['role']}={p['original_value']:#x}" for p in pts if not p["role"] == "transfer_size"]
        print(f"{name:<22} {n_in:>8} {n_out:>9}  {', '.join(vals)}")

    if args.dry_run:
        print("\nDry-run; no file written.")
        return 0

    args.manifest.write_text(json.dumps(annotated, indent=2) + "\n")
    print(f"\nWrote annotated manifest to: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
