#!/usr/bin/env python3
"""Prune a chipyard TestHarness FIRRTL down to a DUT sub-module (e.g. ChipTop).

Rationale (GSIM_ENGINE.md sec.5): the full SoC `TestHarness.fir` aborts in GSIM at
`splitArray.cpp:115` on chipyard test-harness plumbing (the TSI<->TileLink serial
bridge `ram$tsi2tl$...`, SerialRAM/SimTSI, and `_success_T` assertion nodes). Those
nodes only exist because `TestHarness` instantiates them. Re-rooting the circuit at
the synthesizable DUT one level below (ChipTop / DigitalTop) drops that whole subtree.

This does a pure structural rewrite of the `.fir` text:
  1. strip the inline annotation blob  `circuit X :%[[ ... ]]`  (annotations only
     drive dedup/transforms; removing them is simulation-semantics-preserving).
  2. re-root: `circuit <DUT> :`
  3. keep ONLY the module/extmodule/intmodule definitions reachable (by `inst ... of`)
     from <DUT>. Everything harness-only (TestHarness, SerialRAM, SimTSI, TSIToTileLink,
     SimDRAM, UARTAdapter, the *_SerialRAM serdes, ...) is dropped entirely.

No regex; only line-prefix / token splitting.
"""
import sys


DECL_KEYWORDS = ("module ", "extmodule ", "intmodule ")


def is_module_header(line: str) -> bool:
    # Top-level module decls are indented exactly two spaces in firtool output.
    if not line.startswith("  ") or line.startswith("   "):
        return False
    body = line[2:]
    for kw in DECL_KEYWORDS:
        if body.startswith(kw):
            return True
    return False


def module_name_of(header: str) -> str:
    body = header.strip()
    for kw in DECL_KEYWORDS:
        if body.startswith(kw):
            rest = body[len(kw):]
            # name terminates at whitespace or ':'
            name = rest.split(":", 1)[0].strip().split()[0]
            return name
    raise ValueError(f"not a module header: {header!r}")


def instances_in(block_lines):
    """Yield the target module name of every `inst <name> of <Module>` line."""
    for ln in block_lines:
        s = ln.strip()
        if s.startswith("inst "):
            # inst <name> of <Module> @[...]
            parts = s.split()
            # parts = ['inst', name, 'of', Module, ...]
            if len(parts) >= 4 and parts[2] == "of":
                yield parts[3]


def main():
    if len(sys.argv) != 4:
        print("usage: gsim_prune_to_dut.py <in.fir> <DUT_module> <out.fir>", file=sys.stderr)
        sys.exit(2)
    in_path, dut, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(in_path, "r") as f:
        lines = f.readlines()

    # --- locate FIRRTL version + circuit line, skip annotation blob ---
    version_line = "FIRRTL version 3.3.0\n"
    circuit_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("FIRRTL version"):
            version_line = ln if ln.endswith("\n") else ln + "\n"
        if ln.startswith("circuit "):
            circuit_idx = i
            break
    if circuit_idx is None:
        print("ERROR: no 'circuit' line found", file=sys.stderr)
        sys.exit(1)

    circuit_line = lines[circuit_idx]
    body_start = circuit_idx + 1
    if "%[[" in circuit_line:
        # annotation blob: find closing ']]' line
        for j in range(circuit_idx + 1, len(lines)):
            if lines[j].rstrip("\n").strip() == "]]":
                body_start = j + 1
                break

    # --- split remaining lines into module blocks ---
    blocks = {}          # name -> list[str]
    order = []           # preserve original emission order
    cur_name = None
    cur = []
    for ln in lines[body_start:]:
        if is_module_header(ln):
            if cur_name is not None:
                blocks[cur_name] = cur
            cur_name = module_name_of(ln)
            order.append(cur_name)
            cur = [ln]
        else:
            if cur_name is not None:
                cur.append(ln)
            # lines before the first module header (blank tail of annotations) are dropped
    if cur_name is not None:
        blocks[cur_name] = cur

    if dut not in blocks:
        print(f"ERROR: DUT module {dut!r} not found. Have {len(blocks)} modules.", file=sys.stderr)
        sys.exit(1)

    # --- BFS reachability from DUT ---
    reachable = set()
    stack = [dut]
    missing = set()
    while stack:
        m = stack.pop()
        if m in reachable:
            continue
        reachable.add(m)
        blk = blocks.get(m)
        if blk is None:
            missing.add(m)
            continue
        for tgt in instances_in(blk):
            if tgt not in reachable:
                stack.append(tgt)

    kept = [m for m in order if m in reachable]

    # --- emit ---
    with open(out_path, "w") as out:
        out.write(version_line)
        out.write(f"circuit {dut} :\n")
        for m in kept:
            out.writelines(blocks[m])

    dropped = [m for m in order if m not in reachable]
    print(f"total modules parsed : {len(blocks)}")
    print(f"reachable from {dut}  : {len(kept)}")
    print(f"dropped (harness/etc): {len(dropped)}")
    # show a sample of dropped harness modules for the record
    interesting = [m for m in dropped if any(k in m for k in
                   ("TestHarness", "SerialRAM", "TSI", "SimDRAM", "UARTAdapter",
                    "SimUART", "SimJTAG", "Serdes", "SerialTL", "tsi", "Sim"))]
    if interesting:
        print("dropped harness-relevant modules:")
        for m in interesting[:40]:
            print(f"  - {m}")
    if missing:
        print(f"NOTE: {len(missing)} referenced modules had no definition (blackbox extmodules expected):")
        for m in sorted(missing)[:20]:
            print(f"  ? {m}")


if __name__ == "__main__":
    main()
