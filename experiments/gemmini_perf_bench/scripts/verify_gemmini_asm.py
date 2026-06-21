"""Verify an IREE (or any) ELF actually offloads matmul to Gemmini, by DISASSEMBLING the embedded
dispatch and counting Gemmini RoCC custom-3 (opcode 0x7b) instructions.

WHY: an IREE rebuild (e.g. switching the host VM module bytecode->EmitC) could silently change the
dispatch lowering and fall back to a scalar/RVV CPU kernel — which would still pass a numeric self-check
but would NOT be using the systolic array, so any cycle number would be meaningless. The only sure check
is the assembly: a correct Gemmini matmul dispatch contains custom-3 (0x7b) RoCC ops (mvin/mvout/preload/
compute/config). Reference: the bytecode-built 16x16x16 dispatch carries 14 such ops.

The IREE ELF embeds its dispatch as a static-pie ELF64 shared object inside its data (embedded-elf
loader). This finds every embedded ELF, disassembles it, and reports the custom-opcode histogram.

Usage: verify_gemmini_asm.py <elf> [<elf> ...]   ->  PASS if any embedded dispatch has >0 custom-3 ops.
"""
from __future__ import annotations
import re, subprocess, sys
from pathlib import Path

OBJDUMP = "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-objdump"
CUSTOM = {0x0b: "custom-0", 0x2b: "custom-1", 0x5b: "custom-2", 0x7b: "custom-3 (gemmini RoCC)"}


def _embedded_elfs(data: bytes) -> list[tuple[int, bytes]]:
    """Return (offset, bytes) for each candidate embedded ELF64 RISC-V object (skip the host ELF at 0)."""
    out = []
    for m in re.finditer(b"\x7fELF", data):
        off = m.start()
        if off == 0:
            continue
        # EI_CLASS==2 (ELF64), e_machine==243 (RISC-V) at offset+18 (LE u16)
        if data[off + 4] != 2:
            continue
        mach = int.from_bytes(data[off + 18:off + 20], "little")
        if mach != 243:
            continue
        out.append((off, data[off:]))
    return out


def _custom_hist(elf_bytes: bytes) -> tuple[int, dict]:
    p = Path("/tmp/_vg_dispatch.elf"); p.write_bytes(elf_bytes)
    try:
        dis = subprocess.run([OBJDUMP, "-d", str(p)], capture_output=True, text=True).stdout
    except FileNotFoundError:
        print(f"  (objdump not found at {OBJDUMP})"); return 0, {}
    total = 0; hist = {}
    for ln in dis.splitlines():
        m = re.match(r"\s*[0-9a-f]+:\s+([0-9a-f]{8})\s", ln)
        if m:
            total += 1
            op = int(m.group(1), 16) & 0x7f
            if op in CUSTOM:
                hist[op] = hist.get(op, 0) + 1
    return total, hist


def verify(elf: Path) -> bool:
    data = elf.read_bytes()
    embedded = _embedded_elfs(data)
    print(f"== {elf.name} ({len(data)} B) — {len(embedded)} embedded RISC-V ELF dispatch(es) ==")
    gemmini_total = 0
    for off, blob in embedded:
        total, hist = _custom_hist(blob)
        g = hist.get(0x7b, 0)
        gemmini_total += g
        pretty = ", ".join(f"{CUSTOM[op]}={n}" for op, n in sorted(hist.items())) or "none"
        print(f"  dispatch@{off}: {total} insns decoded; custom: {pretty}")
    ok = gemmini_total > 0
    print(f"  -> {'PASS' if ok else 'FAIL'}: {gemmini_total} gemmini custom-3 (0x7b) ops "
          f"({'offloaded to systolic array' if ok else 'NO gemmini ops — fell back to CPU?'})")
    return ok


def main(argv=None):
    args = argv or sys.argv[1:]
    if not args:
        print(__doc__); return 2
    results = {a: verify(Path(a)) for a in args}
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
