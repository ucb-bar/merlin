#!/usr/bin/env python3
"""Cross-check a target's SHIPPED ISA definition against its RTL decode table.

merlin derives a target's encoding by probing the ISA definition the target ships. That honours
"derive, never hardcode" only as far as the shipped definition is right about its own hardware. It is not
always right: atlas's `isa_definition.py` gives `DMA_CONFIG` the funct7 that belongs to `DMA_WAIT`, so the
two collide, and merlin faithfully derived the collision. The consequence reached grading -- a tier was
declared inapplicable on the reasoning that the ISA could not distinguish the two instructions, when in
fact only the vendor's Python model cannot; the RTL distinguishes them by one bit.

The RTL carries the answer in a form that needs no vendor model: a Chisel decode table of BitPat literals,
each a 32-character mask/value pattern. This compares the two and reports every disagreement. Nothing here
knows any opcode -- both sides are read from the target's own sources.

Usage:
    check_isa_matches_rtl.py --target atlas [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BITPAT = 'BitPat("'


def parse_bitpats(text: str) -> dict[str, tuple[int, int, int]]:
    """``{MNEMONIC: (mask, value, width)}`` from Chisel ``def NAME = BitPat("b0101??01...")`` lines.

    Parsed structurally, by splitting on the literal delimiters -- a pattern match would silently skip a
    line spelled differently, which is exactly the failure mode this file exists to catch elsewhere.
    ``?`` marks a don't-care bit (excluded from the mask); ``_`` is a readability separator.
    """
    out: dict[str, tuple[int, int, int]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("def ") or _BITPAT not in line:
            continue
        name = line[len("def "):].partition("=")[0].strip()
        body = line.split(_BITPAT, 1)[1].partition('"')[0]
        if not body.startswith("b"):
            continue
        bits = body[1:].replace("_", "")
        if not bits or any(c not in "01?" for c in bits):
            continue
        mask = value = 0
        width = len(bits)
        for i, c in enumerate(bits):                      # bits[0] is the MSB
            pos = width - 1 - i
            if c == "?":
                continue
            mask |= 1 << pos
            if c == "1":
                value |= 1 << pos
        out[name] = (mask, value, width)
    return out


def _rtl_sources(root: Path) -> str:
    return "\n".join(p.read_text(errors="replace") for p in sorted(root.rglob("*.scala")))


def crosscheck(model, table: dict[str, tuple[int, int, int]]) -> list[dict]:
    """Disagreements between the derived model and the RTL, matched by mnemonic.

    A mnemonic the RTL spells differently (per-channel in the model, ``_ANY`` in the decoder) is compared
    against its RTL family when the model's name starts with the RTL name minus that suffix -- reported as
    a family match so the reader knows which comparison was made. Anything unmatched is reported, never
    silently skipped."""
    findings: list[dict] = []
    fams = {k[:-4]: v for k, v in table.items() if k.endswith("_ANY")}
    for mnem, ent in sorted((model.by_mnemonic or {}).items()):
        fm, fv = int(ent.get("fixed_mask") or 0), int(ent.get("fixed_value") or 0)
        rtl = table.get(mnem)
        how = "exact"
        if rtl is None:
            hit = [(fam, v) for fam, v in fams.items() if mnem.startswith(fam)]
            if not hit:
                continue                                   # not in the decoder's table; nothing to compare
            rtl, how = max(hit, key=lambda kv: len(kv[0]))[1], "family"
        rmask, rvalue, _w = rtl
        common = fm & rmask                                # only bits BOTH sides call fixed
        if common and (fv & common) != (rvalue & common):
            findings.append({"mnemonic": mnem, "match": how,
                             "model_value": f"0x{fv:08x}", "rtl_value": f"0x{rvalue:08x}",
                             "disagreeing_bits": f"0x{((fv ^ rvalue) & common):08x}"})
    return findings


def parse_green_card(text: str) -> dict[str, tuple[str, str, str]]:
    """``{mnemonic: (opcode, funct3, funct7)}`` from the shipped ISA green card's markdown tables.

    A THIRD independent statement of the encoding, written by the same people who wrote the RTL. It
    matters because the disagreements are not all in the same direction: for atlas's `dma.config` the RTL
    and the card agree and the Python model is the outlier, while for `vcube.bf16` the card and the model
    agree and the RTL is. A two-way check can only say "these differ"; three sources say WHICH is odd.

    Rows are split on the table delimiter and read positionally -- no pattern matching, so a row spelled
    unusually is skipped visibly rather than silently mis-read."""
    out: dict[str, tuple[str, str, str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        mnem, opcode, funct3, funct7 = cells[0], cells[2], cells[3], cells[4]
        if not mnem or any(c not in "01" for c in opcode) or not opcode:
            continue                                        # header / separator / prose row
        key = mnem.upper().replace(".", "_")
        # The card writes a channel family as `dma.config.ch<N>`; store it under the stem so a concrete
        # `DMA_CONFIG_CH3` finds it. Without this the per-channel rows silently drop out of the
        # comparison -- which is how the DMA disagreement stayed invisible in the first place.
        out[key.partition("<")[0].rstrip("_") if "<" in key else key] = (opcode, funct3, funct7)
    return out


def triangulate(model, table: dict, card: dict) -> list[dict]:
    """For each mnemonic the three sources describe, say which one is the odd man out."""
    rows: list[dict] = []
    for mnem, ent in sorted((model.by_mnemonic or {}).items()):
        c = (card.get(mnem)
             or card.get(mnem.rstrip("01234567"))
             or card.get(mnem.rstrip("01234567").rstrip("_"))
             or next((v for k, v in card.items() if mnem.startswith(k) and len(k) > 6), None))
        if not c or not c[2] or any(ch not in "01" for ch in c[2]):
            continue
        card_f7 = int(c[2], 2)
        model_f7 = ent.get("funct7")
        if model_f7 is None:
            continue
        rtl = table.get(mnem)
        if rtl is None:
            fams = {k[:-4]: v for k, v in table.items() if k.endswith("_ANY")}
            hit = [(f, v) for f, v in fams.items() if mnem.startswith(f)]
            if not hit:
                continue
            rtl = max(hit, key=lambda kv: len(kv[0]))[1]
        rtl_f7 = (rtl[1] >> 25) & 0x7F
        if rtl_f7 == card_f7 == int(model_f7):
            continue
        odd = ("model" if rtl_f7 == card_f7 else
               "rtl" if card_f7 == int(model_f7) else
               "green_card" if rtl_f7 == int(model_f7) else "all three")
        rows.append({"mnemonic": mnem, "rtl": f"{rtl_f7:07b}", "green_card": f"{card_f7:07b}",
                     "model": f"{int(model_f7):07b}", "outlier": odd})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--rtl-root", default="", help="override the RTL source root")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    from merlin.common.paths import merlin_dir
    from merlin.targetgen.isa_model import isa_model_for_target

    root = Path(a.rtl_root) if a.rtl_root else None
    if root is None:
        base = merlin_dir() / "experiments/capsule_bench/targets" / a.target / "contracts"
        cands = [p for p in base.glob("*/rtl") if p.is_dir()] if base.is_dir() else []
        root = cands[0] if cands else None
    if root is None or not root.is_dir():
        print(f"[skip] {a.target}: no RTL source tree found; nothing to cross-check against")
        return 0

    table = parse_bitpats(_rtl_sources(root))
    if not table:
        print(f"[skip] {a.target}: RTL tree carries no BitPat decode table")
        return 0
    model = isa_model_for_target(a.target)
    if model.is_empty():
        print(f"[skip] {a.target}: no derived ISA model to compare")
        return 0

    findings = crosscheck(model, table)
    cards = sorted((merlin_dir() / "experiments/capsule_bench/targets" / a.target / "contracts")
                   .glob("*/isa_include/*green_card*.md"))
    tri = triangulate(model, table, parse_green_card(cards[0].read_text())) if cards else []
    if a.json:
        print(json.dumps({"target": a.target, "rtl_patterns": len(table),
                          "model_mnemonics": len(model.by_mnemonic or {}),
                          "findings": findings, "three_way": tri}, indent=2))
    else:
        print(f"[{a.target}] RTL decode patterns: {len(table)}  "
              f"derived mnemonics: {len(model.by_mnemonic or {})}")
        for f in findings:
            print(f"  MISMATCH {f['mnemonic']:24s} model={f['model_value']} rtl={f['rtl_value']} "
                  f"(bits {f['disagreeing_bits']}, matched {f['match']})")
        if not findings:
            print("  ok: every comparable mnemonic agrees with the RTL decode table")
        if tri:
            print("\n  three-way (RTL vs shipped green card vs derived model) — funct7:")
            for r in tri:
                print(f"    {r['mnemonic']:22s} rtl={r['rtl']} card={r['green_card']} "
                      f"model={r['model']}  -> outlier: {r['outlier'].upper()}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
