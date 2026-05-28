#!/usr/bin/env python3
"""Per-dispatch accelerator-coverage analysis of a merlin-compiled VMFB.

Disassembles the embedded RISC-V ELF inside a .vmfb, classifies every
dispatch entry-point by what backend opcodes it actually emits, and
reports a per-(dispatch, backend) coverage table.

This tells you whether every matmul in your model is actually running
on the Gemmini / OPU accelerator OR has fallen back to scalar CPU
code (which would silently work but at 100-1000x slowdown).

Detectors used:
  - Gemmini (RoCC custom-3, opcode 0x7B with funct3=3)
    funct7=0: CONFIG_EX/ST/LD
    funct7=2: MVIN          funct7=3: MVOUT
    funct7=4: COMPUTE_PRELOADED   funct7=5: COMPUTE_ACCUMULATED
    funct7=6: PRELOAD       funct7=7: FLUSH
    funct7=8..13: LOOP_WS_*  (hardware loop)
  - OPU (RISC-V V extension scalable .insn with funct3=0, opcode 0x57)
  - RVV vector ops (vsetvli, vle/vse, vfmadd, ...): RISC-V V extension
  - Scalar CPU fallback (regular RV64GC ops only — no .insn / no V)

Usage:
    ./merlin coverage-check <vmfb>
    ./merlin coverage-check <vmfb> --csv out.csv
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def _find_objdump() -> Path:
    """Locate a riscv64 objdump. Override via $MERLIN_RISCV_OBJDUMP."""
    # Import here to avoid a circular `coverage` import at module load time
    # (tools/coverage_cmd.py is the registered shim that triggers this).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import utils

    try:
        return utils.find_toolchain_binary(
            "riscv64-unknown-elf-objdump",
            env_var="MERLIN_RISCV_OBJDUMP",
            aliases=("riscv64-zephyr-elf-objdump",),
        )
    except FileNotFoundError as e:
        raise SystemExit(f"no riscv64 objdump found: {e}") from None


def _extract_elf(vmfb: Path, dst: Path) -> None:
    data = vmfb.read_bytes()
    matches = [m.start() for m in re.finditer(b"\x7fELF", data)]
    if not matches:
        raise SystemExit(f"no embedded ELF in {vmfb}")
    # IREE embeds one ELF per executable-target. We just take the first
    # (the main RISC-V dispatch ELF). MX builds may have multiple but
    # for the firesim targets we care about, there's exactly one.
    dst.write_bytes(data[matches[0] :])


def _disassemble(objdump: Path, elf: Path) -> str:
    out = subprocess.run(
        [str(objdump), "-d", "-M", "no-aliases", str(elf)],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout


# Pre-compiled regexes for fast scanning
_FUNC_RE = re.compile(r"^([0-9a-f]+)\s+<([^>]+)>:")
_INSN_RE = re.compile(r"^\s+([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)\s*(.*)$")


def _classify_insn(word_hex: str, mnemonic: str, args: str) -> str:
    """Map a single instruction to a class label."""
    word = int(word_hex, 16)
    op = word & 0x7F

    # Custom-3 RoCC (Gemmini)
    if op == 0x7B:
        funct7 = (word >> 25) & 0x7F
        funct3 = (word >> 12) & 0x7
        if funct3 != 3:
            return f"custom3-other(f7={funct7})"
        # Gemmini opcode classification
        mapping = {
            0: "gemmini.CONFIG",
            1: "gemmini.MVIN2",
            2: "gemmini.MVIN",
            3: "gemmini.MVOUT",
            4: "gemmini.COMPUTE_PRELOADED",
            5: "gemmini.COMPUTE_ACCUMULATED",
            6: "gemmini.PRELOAD",
            7: "gemmini.FLUSH",
            8: "gemmini.LOOP_WS_CONFIG_BOUNDS",
            9: "gemmini.LOOP_WS_CONFIG_ADDRS_AB",
            10: "gemmini.LOOP_WS_CONFIG_ADDRS_DC",
            11: "gemmini.LOOP_WS_CONFIG_STRIDES_AB",
            12: "gemmini.LOOP_WS_CONFIG_STRIDES_DC",
            13: "gemmini.LOOP_WS",
            14: "gemmini.MVIN3",
        }
        return mapping.get(funct7, f"gemmini.unknown(f7={funct7})")

    # OP-V (RISC-V V extension): opcode 0x57
    if op == 0x57:
        funct3 = (word >> 12) & 0x7
        # funct3=0 is OPIVV (vector-vector), 1=OPFVV, etc. For OPU,
        # we look for specific opcodes that hit OPU's funct6 fields.
        # Most generic detection: any opcode 0x57 with funct3==0 in
        # the "unknown" range indicates an OPU instruction (because
        # the disassembler doesn't recognize them as standard V ops).
        if "unknown" in mnemonic.lower() or mnemonic == ".insn":
            return "opu"
        return "rvv"

    # Standard V vsetvli/vsetivli/vsetvl
    if mnemonic.startswith("vset"):
        return "rvv"
    # V load/store
    if mnemonic.startswith(("vle", "vse", "vlse", "vsse", "vluxei", "vsuxei")):
        return "rvv"
    # V arithmetic
    if mnemonic.startswith(("vfm", "vfadd", "vmul", "vmacc", "vadd", "vmv.", "vredsum", "vredmax")):
        return "rvv"

    return "scalar"


def _walk_functions(disasm: str):
    """Yield (func_addr, func_name, instructions) tuples."""
    current = None
    insns: list[tuple[str, str, str]] = []
    for line in disasm.splitlines():
        m = _FUNC_RE.match(line)
        if m:
            if current is not None:
                yield current[0], current[1], insns
            current = (m.group(1), m.group(2))
            insns = []
            continue
        m = _INSN_RE.match(line)
        if m:
            insns.append((m.group(2), m.group(3), m.group(4)))
    if current is not None:
        yield current[0], current[1], insns


def _classify_function(insns) -> dict[str, int]:
    counts: dict[str, int] = {}
    for word_hex, mnemonic, args in insns:
        cls = _classify_insn(word_hex, mnemonic, args)
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def _summarize_function(name: str, counts: dict[str, int]) -> str:
    g = sum(v for k, v in counts.items() if k.startswith("gemmini."))
    o = counts.get("opu", 0)
    r = counts.get("rvv", 0)
    s = counts.get("scalar", 0)
    total = g + o + r + s + sum(v for k, v in counts.items() if k.startswith("custom3-other"))
    label = "scalar"
    if g > 0 and o == 0:
        label = "gemmini"
    elif o > 0 and g == 0:
        label = "opu"
    elif r > total * 0.05 and g == 0 and o == 0:
        label = "rvv"
    return f"{label} (g={g} o={o} r={r} s={s})"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("vmfb", type=Path, help=".vmfb file to inspect")
    p.add_argument("--csv", type=Path, default=None, help="Write per-function CSV")
    args = p.parse_args()

    if not args.vmfb.exists():
        raise SystemExit(f"not found: {args.vmfb}")

    objdump = _find_objdump()
    with tempfile.NamedTemporaryFile(suffix=".elf", delete=False) as tmp:
        elf = Path(tmp.name)
    try:
        _extract_elf(args.vmfb, elf)
        disasm = _disassemble(objdump, elf)
    finally:
        elf.unlink(missing_ok=True)

    rows = []
    totals: dict[str, int] = {}
    for addr, name, insns in _walk_functions(disasm):
        counts = _classify_function(insns)
        rows.append((addr, name, counts))
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v

    # Filter to dispatch entries (skip helpers like library_query stubs)
    interesting = [
        (a, n, c)
        for (a, n, c) in rows
        if "dispatch" in n.lower()
        or "encoding" in n.lower()
        or "matmul" in n.lower()
        or "conv" in n.lower()
        or "initializer" in n.lower()
        or n.startswith("main_graph")
        or "$async_dispatch" in n
    ]
    if not interesting:
        # Fall back to ALL functions
        interesting = rows

    print(f"# VMFB: {args.vmfb}")
    print(f"# Functions in ELF: {len(rows)} total, {len(interesting)} dispatch-ish")
    print()
    print(f"{'addr':<10} {'classification':<35}  function")
    print("-" * 120)
    by_label: dict[str, int] = {}
    for addr, name, counts in interesting:
        summary = _summarize_function(name, counts)
        label = summary.split(" ", 1)[0]
        by_label[label] = by_label.get(label, 0) + 1
        print(f"{addr:<10} {summary:<35}  {name}")

    print()
    print("# Per-class function count (only the dispatch-ish set)")
    for k in sorted(by_label, key=lambda k: -by_label[k]):
        print(f"  {k:<10} : {by_label[k]} functions")

    print()
    print("# Per-class TOTAL instruction count (entire ELF)")
    for k in sorted(totals, key=lambda k: -totals[k]):
        print(f"  {k:<40} {totals[k]:>8}")

    if args.csv:
        import csv

        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            classes = sorted(set().union(*[set(c.keys()) for _, _, c in rows]))
            w.writerow(["addr", "function", "label"] + classes)
            for addr, name, counts in rows:
                label = _summarize_function(name, counts).split(" ", 1)[0]
                row = [addr, name, label] + [counts.get(c, 0) for c in classes]
                w.writerow(row)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
