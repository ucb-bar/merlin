"""How much of a target's DERIVED instruction set its capsule corpus actually exercises.

A capsule count is not a coverage number: it has no denominator. "27 capsules, 14 pass" says nothing
about how much of the machine was driven, and a corpus can grow indefinitely while exercising the same
narrow slice. The denominator that exists is the target's own derived ISA -- every op its RTL defines --
and the numerator is what the EMITTED kernels were observed to contain.

Measured on the first target this was run against: the corpus exercised 25 of 72 derived tensor/DMA ops
(35%). The uncovered set was not scattered noise but four structured holes -- every op belonging to the
second matrix unit, 30 of 32 DMA ops (including every ``wait``, on a design whose own grader tells agents
to await DMA before readback), the fp8 accumulator read-out on an fp8 datapath, and every non-row
reduction. One capsule NAMED for the second matrix unit passed while emitting no instruction of that
unit at all, because its declared expectation was written at class granularity and the class covers both
units (see :func:`merlin.targetgen.isa_disasm.coverage`).

Target-agnostic by construction: the vocabulary, the roles and the decode all come from the target's own
:class:`~merlin.targetgen.isa_model.IsaModel`. Nothing here names a target, an op or a unit.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, Mapping

from . import isa_disasm
from .isa_model import IsaModel

# NO ROLE IS EXCLUDED, and that is a correction rather than a preference. Filtering the universe down to
# the "interesting" roles looks tidy and silently deletes real capability: on the first target this ran
# against, every DMA op carries the SCALAR role in the derived taxonomy (they share one register-operand
# class), so excluding scalar dropped 32 instructions -- among them the single largest hole in the corpus
# and the one its own grader points agents at. A denominator you curate is a denominator you can flatter.
#
# The holes are surfaced by REPORTING structure instead: per derived class and per derived role. A class
# whose mnemonics are 2-of-32 covered is visible as such without anyone having to nominate it in advance,
# and both groupings come from the model, so nothing here invents a category.


def words_of(kernel: str | Path, *, inst_width: int = 32) -> list[int]:
    """The instruction words of a raw kernel image. Little-endian, trailing partial word dropped."""
    raw = Path(kernel).read_bytes()
    n = inst_width // 8
    usable = len(raw) // n * n
    fmt = {2: "H", 4: "I", 8: "Q"}.get(n)
    if fmt is None:
        raise ValueError(f"unsupported instruction width {inst_width} bits")
    return list(struct.unpack(f"<{usable // n}{fmt}", raw[:usable]))


def universe(model: IsaModel, *, roles: Iterable[str] | None = None) -> set[str]:
    """Every derived mnemonic -- the DENOMINATOR. ``roles`` narrows it, but the default is everything the
    target's ISA defines, so a hole cannot be filtered out of the number by choosing a scope."""
    keep = set(roles) if roles is not None else None
    return {mnem for mnem, ent in model.by_mnemonic.items()
            if keep is None or ent.get("role") in keep}


def exercised(model: IsaModel, kernels: Mapping[str, str | Path]) -> dict[str, set[str]]:
    """``{capsule: {exact mnemonics its emitted kernel contains}}``, restricted to the in-scope universe."""
    uni = universe(model)
    out: dict[str, set[str]] = {}
    for name, path in kernels.items():
        recs = isa_disasm.disassemble(model, words_of(path, inst_width=model.inst_width))
        out[name] = {m for m in isa_disasm.present_mnemonics(recs) if m in uni}
    return out


def corpus_coverage(model: IsaModel, kernels: Mapping[str, str | Path]) -> dict:
    """The corpus's ISA coverage: what fraction of the derived in-scope ops any capsule exercises.

    ``uncovered_by_role`` groups the holes the way a corpus author has to act on them -- a role with
    nothing covered is a whole capability no capsule drives. ``ambiguous`` records words that matched more
    than one signature, because a coverage claim built on an ambiguous decode is not evidence; it is
    reported rather than silently resolved to the first hit.
    """
    uni = universe(model)
    per = exercised(model, kernels)
    seen: set[str] = set()
    for s in per.values():
        seen |= s
    uncovered = uni - seen
    by_role: dict[str, list[str]] = {}
    for m in sorted(uncovered):
        by_role.setdefault(str(model.by_mnemonic[m].get("role")), []).append(m)
    # Per DERIVED CLASS, so a capability that is technically covered but barely -- one channel of eight,
    # one matrix unit of two -- reads as the hole it is instead of a tick in a role that looks covered.
    by_class: dict[str, dict] = {}
    for mnem in sorted(uni):
        cls = str(model.by_mnemonic[mnem].get("class") or mnem)
        d = by_class.setdefault(cls, {"role": model.by_mnemonic[mnem].get("role"),
                                      "covered": [], "uncovered": []})
        (d["covered"] if mnem in seen else d["uncovered"]).append(mnem)
    for d in by_class.values():
        n = len(d["covered"]) + len(d["uncovered"])
        d["n_covered"], d["n_total"] = len(d["covered"]), n
        d["ratio"] = len(d["covered"]) / n if n else None
    ambiguous: dict[str, list[str]] = {}
    for name, path in kernels.items():
        recs = isa_disasm.disassemble(model, words_of(path, inst_width=model.inst_width))
        amb = sorted({tuple(r["ambiguous_mnemonics"]) for r in recs if r.get("ambiguous_mnemonics")})
        if amb:
            ambiguous[name] = [list(a) for a in amb]
    return {
        "target": model.target,
        "n_universe": len(uni),
        "n_covered": len(seen),
        "ratio": (len(seen) / len(uni)) if uni else None,
        "covered": sorted(seen),
        "uncovered": sorted(uncovered),
        "uncovered_by_role": by_role,
        "by_class": by_class,
        "per_capsule": {k: sorted(v) for k, v in sorted(per.items())},
        "ambiguous_decodes": ambiguous,
    }


def render_markdown(cov: dict) -> str:
    """The report as a table with EXPLICIT uncovered rows -- nothing is implied covered."""
    ratio = cov.get("ratio")
    pct = "n/a" if ratio is None else f"{100 * ratio:.0f}%"
    lines = [f"# ISA corpus coverage — {cov['target']}", "",
             f"**{cov['n_covered']} / {cov['n_universe']} derived instructions exercised ({pct}).**", ""]
    if cov["uncovered_by_role"]:
        lines += ["## Never exercised by any capsule", "", "| role | instructions |", "|---|---|"]
        for role, mnems in sorted(cov["uncovered_by_role"].items()):
            lines.append(f"| `{role}` | {', '.join('`' + m + '`' for m in mnems)} |")
        lines.append("")
    if cov.get("by_class"):
        lines += ["## Per derived instruction class", "", "| class | role | covered |", "|---|---|---|"]
        for cls, d in sorted(cov["by_class"].items(), key=lambda kv: (kv[1]["ratio"] or 0, kv[0])):
            lines.append(f"| `{cls}` | `{d['role']}` | {d['n_covered']}/{d['n_total']} |")
        lines.append("")
    if cov["ambiguous_decodes"]:
        groups = sorted({tuple(g) for gs in cov["ambiguous_decodes"].values() for g in gs})
        lines += ["## Ambiguous decodes (coverage here is not evidence)", "",
                  f"{len(cov['ambiguous_decodes'])} capsule(s) contain words matching more than one "
                  f"signature: " + "; ".join("/".join(g) for g in groups), ""]
    return "\n".join(lines)
