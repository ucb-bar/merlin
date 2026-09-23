#!/usr/bin/env python3
"""Gate: no capsule may be certifiable only at a tier that cannot see what it exercises.

A target's contract declares what each oracle tier is blind to (``oracle_blind_spots``), as classes
of program in the vocabulary the corpus states about itself. A capsule in such a class whose
deepest reachable tier is one of the blind ones can be screened and never certified: its pass says
nothing, and the corpus is counting it as coverage.

The measured case behind this: one executable passed the functional tier with no wrong output and
failed on the FPGA with every output wrong, and nothing in the result said the functional tier
could not have seen it.

Targets are DISCOVERED (every capsule-bench descriptor); a target with no declared blind spot
contributes nothing. A malformed registry fails the gate: an entry that silently matched nothing
would read as "no blind spot".

    python build_tools/scripts/check_oracle_blind_spots.py           # exit 1 on a finding
    python build_tools/scripts/check_oracle_blind_spots.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "merlin" / "python",):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

DESCRIPTORS = ROOT / "merlin" / "experiments" / "capsule_bench" / "targets"


def _capsules(roots) -> list[dict]:
    found: list[dict] = []
    for root in roots:
        for path in sorted(Path(root).rglob("capsule.yaml")):
            try:
                capsule = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            if isinstance(capsule, dict):
                found.append(capsule)
    return found


def _tracked_contract(target: str) -> dict:
    """The target's TRACKED contract. Read from the file, not through the target registry: this gate
    is about declarations under review, and resolving a name there searches generated trees too."""
    from merlin.common.paths import targets_dir

    path = targets_dir() / target / "contracts" / "target_contract.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def reports() -> list[dict]:
    from merlin.targetgen import oracle_blind_spots as blind
    from merlin.targetgen.target_experiment import load_target_experiment

    out: list[dict] = []
    for descriptor in sorted(DESCRIPTORS.glob("*/target_experiment.yaml")):
        experiment = load_target_experiment(descriptor)
        target = str(getattr(experiment, "target", "") or descriptor.parent.name)
        try:
            spots = blind.for_target(target, contract=_tracked_contract(target))
        except blind.BlindSpotError as error:
            out.append({"target": target, "descriptor": descriptor.parent.name, "malformed": str(error)})
            continue
        if not spots:
            continue
        report = blind.audit(target, _capsules(experiment.graded_roots()), spots=spots).to_dict()
        out.append({**report, "descriptor": descriptor.parent.name})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    found = reports()
    if args.json:
        print(json.dumps(found, indent=2, sort_keys=True))
    failed = False
    for report in found:
        if report.get("malformed"):
            failed = True
            print(f"[FAIL] {report['target']}: malformed blind-spot registry: {report['malformed']}")
            continue
        for finding in report["findings"]:
            failed = True
            print(
                f"[FAIL] {report['descriptor']}: {finding['capsule']} reaches {finding['reaches']} and is "
                f"{finding['finding']} (needs {finding['needs']}; blind: "
                f"{sorted({row['id'] for rows in finding['blind'].values() for row in rows})})"
            )
        if not args.json:
            print(
                f"[  ok] {report['descriptor']}: {report['capsules_in_a_blind_class']} of {report['capsules']} "
                f"capsule(s) sit in a declared blind class; {len(report['findings'])} certifiable only "
                f"inside it. Declared and exercised by no capsule: "
                f"{report['blind_spots_no_capsule_exercises'] or 'none'}"
            )
    if not found and not args.json:
        print("[  ok] oracle-blind-spots: no target declares a blind spot")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
