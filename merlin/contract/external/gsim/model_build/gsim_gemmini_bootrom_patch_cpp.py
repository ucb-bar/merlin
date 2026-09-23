#!/usr/bin/env python3
"""Bake the gemmini matmul kernel into GSIM's generated ChipTop0.cpp by rewriting
the BootROM constant assignments `system$bootrom_domain$bootrom$rom[i] = 0x..;`.

GSIM lowers ChipTop's combinational `wire rom` to constant stores inside an
event-gated block in subStep0(); that block re-runs and clobbers any harness-side
overwrite before the Rocket fetches, so the kernel must live in the .cpp source.
This rewrites rom[0..N-1] to the kernel words (gemmini_bootrom.h from
gsim_gemmini_bootrom_bake.py) and leaves rom[N..511] (the device-tree blob) intact.

Idempotent: re-running restores the same values.  No regex (repo rule): pure
string search for the exact `rom[i] = ` prefix + line-end replace.

Usage:  gsim_gemmini_bootrom_patch_cpp.py <chiptop0_cpp> <gemmini_bootrom.h>
"""
import sys


def parse_words(header_path):
    """Read GEMMINI_BOOTROM_WORDS[...] = { 0x..ull, ... } from the baked header."""
    words = []
    with open(header_path) as f:
        collecting = False
        for line in f:
            s = line.strip()
            if "GEMMINI_BOOTROM_WORDS" in s and "{" in s:
                collecting = True
                continue
            if collecting:
                if s.startswith("};"):
                    break
                # line form: 0x0505151e07f0051bull, // rom[0] @ 0x10000
                tok = s.split(",", 1)[0].strip()
                if tok.endswith("ull"):
                    tok = tok[:-3]
                if tok.startswith("0x"):
                    words.append(int(tok, 16))
    return words


def patch(cpp_path, header_path):
    words = parse_words(header_path)
    if not words:
        raise SystemExit("no rom words parsed from " + header_path)
    prefix = "system$bootrom_domain$bootrom$rom["
    with open(cpp_path) as f:
        lines = f.readlines()
    patched = 0
    for k, line in enumerate(lines):
        pos = line.find(prefix)
        if pos < 0:
            continue
        rest = line[pos + len(prefix):]
        rb = rest.find("]")
        if rb < 0:
            continue
        try:
            idx = int(rest[:rb])
        except ValueError:
            continue
        if idx < 0 or idx >= len(words):
            continue
        # keep leading indentation, rewrite the whole assignment for this index
        indent = line[:pos]
        lines[k] = "%s%s%d] = 0x%016xull;\n" % (indent, prefix, idx, words[idx])
        patched += 1
    if patched != len(words):
        raise SystemExit("expected to patch %d rom words, patched %d" % (len(words), patched))
    with open(cpp_path, "w") as f:
        f.writelines(lines)
    print("patched %d bootrom words in %s" % (patched, cpp_path))
    for i, w in enumerate(words):
        print("  rom[%2d] = 0x%016x" % (i, w))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    patch(sys.argv[1], sys.argv[2])
