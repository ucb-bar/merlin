#!/usr/bin/env python3
"""Gate: a capsule that tests a sign-sensitive stage must be able to produce a negative value.

Every epilogue stage behaves differently on a negative accumulator than on a positive one: an
activation clamps it, a requantization rounds and saturates it in the other direction, a pool's
padding identity is wrong for it, a bias may push a value across zero. The default capsule
stimulus is non-negative (``runtime.commandbuffer.DEFAULT_STIMULUS_RANGE``), so with it the
accumulator is never negative and none of those behaviours can be observed. Measured: a capsule
declaring a relu epilogue returned 126 of 256 outputs negative on hardware -- the raw
accumulator, activation never applied -- while its sibling with the same declaration passed,
because on non-negative inputs ``max(0, x)`` is the identity. A check that cannot fail is not a
check.

A capsule declares its stimulus as ``stimulus_range: [lo, hi]``. This gate requires ``lo < 0`` of
every public capsule whose operation carries an epilogue stage (or is one standing alone).

Known debt lives in ``sign_coverage_ratchet.txt``, one capsule name per line, and may only shrink.
An entry for a capsule that has since declared a signed range, or is gone, fails the gate until it
is removed.

    python build_tools/scripts/check_sign_coverage.py            # exit 1 on a new unsigned capsule
    python build_tools/scripts/check_sign_coverage.py --write    # regenerate the ledger
    python build_tools/scripts/check_sign_coverage.py --list
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "merlin" / "python",):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

CORPUS = ROOT / "merlin" / "contract" / "capsules"
LEDGER = ROOT / "build_tools" / "scripts" / "sign_coverage_ratchet.txt"


def unsigned(corpus: Path = CORPUS) -> list[str]:
    """Public capsules that test a sign-sensitive stage on a stimulus that cannot go negative."""
    from merlin.runtime.commandbuffer import STIMULUS_RANGE_KEY, stimulus_range
    from merlin.targetgen.sign_sensitivity import is_sign_sensitive

    found: list[str] = []
    for path in sorted(corpus.rglob("capsule.yaml")):
        try:
            capsule = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if capsule.get("label", "public") != "public":
            continue
        operation = capsule.get("operation") or {}
        attributes = operation.get("attributes") if isinstance(operation.get("attributes"), dict) else {}
        if not is_sign_sensitive({"op": operation.get("op"), **(attributes or {})}):
            continue
        low, _high = stimulus_range({"params": {STIMULUS_RANGE_KEY: capsule.get(STIMULUS_RANGE_KEY)}})
        if low >= 0:
            found.append(str(capsule.get("name") or path.parent.name))
    return sorted(set(found))


def _ledger() -> list[str]:
    if not LEDGER.is_file():
        return []
    return [
        line.split("#", 1)[0].strip()
        for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    found = unsigned()
    if "--list" in arguments:
        print("\n".join(found))
        return 0
    if "--write" in arguments:
        header = (
            "# Public capsules that test a sign-sensitive epilogue stage on a non-negative stimulus.\n"
            "# May only shrink: declare `stimulus_range: [lo, hi]` with lo < 0 on the capsule and\n"
            "# regenerate its golden.\n"
            "# growth-accepted: the gate is new; this is debt it discovered, not debt added.\n"
        )
        LEDGER.write_text(header + "".join(f"{name}\n" for name in found), encoding="utf-8")
        print(f"wrote {LEDGER.relative_to(ROOT)} ({len(found)} entries)")
        return 0
    known = set(_ledger())
    new = [name for name in found if name not in known]
    stale = sorted(known - set(found))
    for name in new:
        print(
            f"[FAIL] unsigned stimulus: {name} tests a sign-sensitive stage and its stimulus cannot "
            f"go negative, so the stage's behaviour on a negative value is never observed. Declare "
            f"`stimulus_range` with a negative lower bound."
        )
    for name in stale:
        print(
            f"[FAIL] stale ledger entry: {name} is signed or gone -- remove it from "
            f"{LEDGER.relative_to(ROOT)} so the ledger keeps shrinking."
        )
    if new or stale:
        return 1
    print(f"[  ok] sign coverage: {len(found)} known unsigned capsule(s) in the ledger; no new one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
