#!/usr/bin/env python3
"""Gate: a capsule that declares a parametric readout stage must state the parameter.

An ``acc_scale`` stage with no ``acc_scale`` multiplier is not refused by the engines a contraction's
commit goes through: the golden, the reference and the simulator each read the absent multiplier as
1.0. They then agree with each other on a bare saturating cast, and a backend that ignores the scale
altogether passes. Measured: one shipped, graded capsule written to test exactly that stage
declares no multiplier. A ``requant`` stage has the same shape with its shift, and that one the
engines already refuse.

This gate requires the parameter of every public capsule that declares such a stage, on the
operation or on any of its per-command entries. Known debt lives in
``stage_parameter_ratchet.txt``, one capsule name per line, and may only shrink: closing an entry
means regenerating that capsule's golden, which changes what a running campaign is scored on, so it
is done at a campaign boundary with the sign-coverage conversion.

    python build_tools/scripts/check_stage_parameters.py            # exit 1 on a new offender
    python build_tools/scripts/check_stage_parameters.py --write    # regenerate the ledger
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "merlin" / "contract" / "capsules"
LEDGER = ROOT / "build_tools" / "scripts" / "stage_parameter_ratchet.txt"
#: stage -> the attribute that carries its parameter.
PARAMETER_OF = {"acc_scale": "acc_scale", "requant": "requant_shift"}


def offenders(corpus: Path = CORPUS) -> list[str]:
    found: list[str] = []
    for path in sorted(corpus.rglob("capsule.yaml")):
        try:
            capsule = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if capsule.get("label", "public") != "public":
            continue
        attrs = (capsule.get("operation") or {}).get("attributes")
        if not isinstance(attrs, dict):
            continue
        commands = [attrs, *(c for c in attrs.get("matmuls") or () if isinstance(c, dict))]
        for command in commands:
            for stage in command.get("epilogue") or ():
                key = PARAMETER_OF.get(str(stage))
                if key and command.get(key) is None and attrs.get(key) is None:
                    found.append(str(capsule.get("name") or path.parent.name))
    return sorted(set(found))


def main(argv: list[str]) -> int:
    found = offenders()
    if "--write" in argv:
        LEDGER.write_text(
            "# Public capsules that declare a parametric readout stage without its parameter.\n"
            "# May only shrink; see check_stage_parameters.py.\n" + "".join(f"{name}\n" for name in found),
            encoding="utf-8",
        )
        print(f"wrote {LEDGER.relative_to(ROOT)} ({len(found)} entries)")
        return 0
    known = set()
    if LEDGER.is_file():
        known = {
            line.strip()
            for line in LEDGER.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
    new, stale = sorted(set(found) - known), sorted(known - set(found))
    for name in new:
        print(f"[FAIL] {name}: declares a parametric stage and no parameter; every engine would assume one")
    for name in stale:
        print(f"[FAIL] {name}: is in the ledger and no longer offends (or is gone); remove the entry")
    if not new and not stale:
        print(f"[  ok] stage-parameters: {len(found)} ledgered capsule(s); no new one.")
    return 1 if new or stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
