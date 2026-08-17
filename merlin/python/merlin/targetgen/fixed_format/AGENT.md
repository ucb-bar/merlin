# AGENT.md — merlin/python/merlin/targetgen/fixed_format

## Purpose

Building device images for targets whose ISA is a fixed-format re-encoding of a stock one.

## Modules

- `boot.py` — Fork-free build of the boot/BSP object (the crt0-like startup) for a fixed-format target.
- `link.py` — Fork-free link + relocation patch for a target whose ISA is a fixed-format re-encoding.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

**Nothing here may know which device it is building for.** Every ISA fact — field positions, opcode
values, instruction width, stride ratio — comes from the target's derived `IsaModel`, and `target` is a
parameter on the public entrypoints. The only literals are RISC-V psABI relocation numbers and stock
rv32 opcode values, which are the source substrate's contract rather than any accelerator's.

**A fact the model cannot supply is a parameter, not a default.** These modules were previously named
after the first target that exercised them, and the name hid one such fact: `build_bsp` emitted a
literal `__mu_num_warps` for the launch-width shim. That symbol belongs to one BSP, so a second target
would have linked against its own name while this builder silently produced an unreferenced data word.
It is now `occupancy=(symbol, value)`, supplied by the backend that owns the target, and
`occupancy_shim` raises rather than guessing. Expect more of these: when you find one, add a
parameter and a test that asserts the vendor spelling is *absent* from the output.

**Fail closed.** A relocation type that is not modelled, an operand the derived encoder cannot
represent, or an input placement that does not reproduce the stock layout byte-for-byte all raise. A
single wrong relocation panics the device at `pc:0`, where a silent mis-encode is far more expensive to
find than a build failure.
