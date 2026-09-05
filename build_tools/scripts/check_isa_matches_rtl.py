#!/usr/bin/env python3
"""Gate: a target's DECLARED instruction encoding must agree with its own hardware's decode evidence.

merlin derives a target's encoding by probing whatever ISA definition that target ships. That honours
"derive, never hardcode" only as far as the shipped definition is right about its own hardware, and a
shipped definition is a document: it can be wrong. When it is, every backend that faithfully derives from
it emits a word that assembles cleanly and decodes to something else — no illegal instruction, no error,
just wrong numbers days later.

All the reasoning lives in :mod:`merlin.targetgen.isa_rtl_crosscheck`; this is the command-line gate over
it. Nothing here knows an opcode or a target: both sides of every comparison are read from the target's
own sources.

EXIT CODES — the distinction is the point.
    0  every comparable instruction agrees, or the disagreement is recorded in the errata registry
    1  the hardware contradicts the declared encoding and nobody has written it down
    2  UNKNOWN: no usable hardware evidence covered a single instruction. NOT a pass. A check that could
       not run must never report success — this repo has lost days to exactly that shape.

Usage:
    check_isa_matches_rtl.py --target <t> [--json] [--show-covered]
    check_isa_matches_rtl.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin" / "python"))

from merlin.targetgen import isa_rtl_crosscheck as X  # noqa: E402


def _print_report(rep, *, show_covered: bool) -> None:
    print(f"\n[{rep.target}] status={rep.status.upper()}  "
          f"model={rep.model_mnemonics} mnemonics (provenance: {rep.model_provenance})")

    print("  evidence sources:")
    for s in rep.sources:
        state = "USABLE " if s.usable else ("CIRCULAR" if s.circular else
                                            ("present" if s.present else "absent  "))
        print(f"    [{state:8s}] {s.kind:24s} entries={s.entries:<5d} {s.note}")
        if s.provenance and s.usable:
            print(f"{'':16s}from: {s.provenance}")

    # COVERAGE FIRST, and never optional. 111 model classes that no decode pattern mentions reading as
    # "clean" is the exact failure this gate exists to make impossible, so the number nobody wants to
    # look at is printed before the number everybody does.
    cov, unc = rep.covered_mnemonics, rep.uncovered_mnemonics
    print(f"  coverage: {len(cov)} instruction(s) actually compared against hardware evidence, "
          f"{len(unc)} NOT COVERED by any usable source")
    # Per source as well as in union: a union that covers everything can still hide a source that saw
    # almost nothing, and "which source actually looked at this instruction" is what a reader needs to
    # judge how much the AGREE is worth.
    for s in rep.sources:
        if not s.usable:
            continue
        n = {v: len([f for f in rep.findings if f.source == s.kind and f.verdict == v])
             for v in (X.AGREE, X.DISAGREE, X.NOT_COVERED)}
        print(f"    via {s.kind:24s} agree={n[X.AGREE]:<4d} disagree={n[X.DISAGREE]:<4d} "
              f"not_covered={n[X.NOT_COVERED]}")
    if unc:
        show = sorted(unc)
        print(f"    not covered: {', '.join(show[:12])}" + (f" … (+{len(show)-12} more)" if len(show) > 12 else ""))
        print("    ^ these were NOT checked. Absence of evidence is not evidence of agreement.")

    for n in rep.notes:
        print(f"  note: {n}")

    if show_covered:
        for f in rep.by_verdict(X.AGREE):
            print(f"    ok   {f.mnemonic:24s} {f.source} {f.declared}=={f.evidence}")

    undeclared = X.undeclared_disagreements(rep)
    outliers = rep.outliers()
    if outliers:
        print(f"  DISAGREEMENTS ({len(outliers)}; {len(undeclared)} not recorded in the errata registry):")
        for mnem, row in sorted(outliers.items()):
            mark = "UNDECLARED" if mnem in undeclared else "declared  "
            print(f"    [{mark}] {mnem:24s} spec={row['declared']}")
            for src, ev in sorted(row["evidence"].items()):
                print(f"{'':18s}contradicted by {src}: {ev}")
            if row["hardware_for"]:
                print(f"{'':18s}but AGREED by hardware source(s): {', '.join(row['hardware_for'])}")
            if row["authored_for"]:
                print(f"{'':18s}backed by authored source(s): {', '.join(row['authored_for'])}"
                      "  (a document does not outrank an extraction)")
        if undeclared:
            print("\n  To resolve: decide which side is right and record it in "
                  f"{X.errata_path()}. Ready-to-paste:\n")
            print(_errata_yaml(rep, undeclared))


def _errata_yaml(rep, undeclared: dict) -> str:
    """The registry stanza for the undeclared findings — printed so a human edits prose, not structure.
    Deliberately leaves `authoritative` and `rationale` for a person: the checker knows which sources
    disagree, it does not know which one is right, and a tool that filled those in would be recording its
    own guess as a review."""
    out = [f"  {rep.target}:"]
    for mnem, row in sorted(undeclared.items()):
        hw = sorted(row["evidence"].values())
        out += [f"    {mnem}:",
                f"      declared: {row['declared']!r}",
                f"      hardware: {hw[0]!r}" if hw else "      hardware: null",
                f"      sources_against_spec: [{', '.join(sorted(row['hardware_against']))}]",
                f"      sources_for_spec: [{', '.join(sorted(row['authored_for'] + row['hardware_for']))}]",
                "      authoritative: FILL_ME   # rtl | spec | unresolved",
                "      rationale: FILL_ME",
                "      upstream: null"]
    return "\n".join(out)


def _targets() -> list[str]:
    from merlin.common.paths import merlin_dir
    base = merlin_dir() / "experiments" / "capsule_bench" / "targets"
    return sorted(p.parent.name for p in base.glob("*/target_experiment.yaml"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--target")
    g.add_argument("--all", action="store_true", help="every target that ships a descriptor")
    ap.add_argument("--rtl-root", default="", help="override the RTL source root")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show-covered", action="store_true", help="also list the instructions that agree")
    a = ap.parse_args(argv)

    targets = _targets() if a.all else [a.target]
    reports = [X.crosscheck(t, rtl_root=a.rtl_root or None) for t in targets]

    if a.json:
        print(json.dumps([X.erratum(r) | {"undeclared": sorted(X.undeclared_disagreements(r))}
                          for r in reports], indent=2, default=str))
    else:
        for r in reports:
            _print_report(r, show_covered=a.show_covered)

    bad = [r for r in reports if X.undeclared_disagreements(r)]
    unknown = [r for r in reports if r.status == X.UNKNOWN]
    if not a.json:
        print(f"\n{'='*70}")
        for r in reports:
            print(f"  {r.target:12s} {r.status.upper():12s} "
                  f"compared={len(r.covered_mnemonics):<4d} not_covered={len(r.uncovered_mnemonics):<4d} "
                  f"undeclared_disagreements={len(X.undeclared_disagreements(r))}")
        if bad:
            print("\nFAIL: a target's own hardware contradicts the encoding merlin derives for it, and the "
                  "disagreement is not recorded. A backend deriving from that encoding emits a word the "
                  "hardware executes as a DIFFERENT instruction, silently.")
        elif unknown:
            print("\nUNKNOWN: no usable hardware evidence covered a single instruction for at least one "
                  "target. This is not a pass — nothing was verified.")
        else:
            print("\nOK: every comparable instruction agrees with its hardware, or is recorded as an "
                  "erratum.")
    return 1 if bad else (2 if unknown else 0)


if __name__ == "__main__":
    sys.exit(main())
