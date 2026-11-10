#!/usr/bin/env python3
"""Reject public capsules that accept the existing zero/mean/midrange answers.

The numerical test delegates to the existing optional oracle and falsifiability
algorithm. Missing or malformed inputs are not clean verdicts. Fresh CI may
explicitly report unavailable independent goldens as UNMEASURED; standalone
release audits refuse them. This gate never changes a tolerance or writes goldens.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from merlin.common.paths import repo_root  # noqa: E402
from merlin.targetgen import capsule_common, capsule_golden, numeric_falsifiability  # noqa: E402


def capsule_root() -> Path:
    return repo_root() / "experiments/corpora"


def _capsules(root: Path, *, tracked_only: bool) -> list[Path]:
    if tracked_only:
        relative = root.resolve().relative_to(ROOT)
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--", str(relative)], cwd=ROOT, capture_output=True, check=True
        )
        return sorted(
            ROOT / member
            for member in listed.stdout.decode().split("\0")
            if member and Path(member).name == "capsule.yaml" and "hidden" not in Path(member).parts
        )
    found = []
    for directory, children, members in os.walk(root, followlinks=False):
        # Do not enter private holdouts or indirect roots, even for an explicit corpus.
        children[:] = [name for name in children if name != "hidden" and not (Path(directory) / name).is_symlink()]
        if "capsule.yaml" in members:
            found.append(Path(directory) / "capsule.yaml")
    return sorted(found)


def offenders(root: Path, *, tracked_only: bool = False, summary: dict | None = None) -> list[dict]:
    """Find accepted candidates and explicit load/evaluation refusals."""
    found = []
    paths = _capsules(root, tracked_only=tracked_only)
    counts = {
        "public_declarations": len(paths),
        "applicable_policies": 0,
        "assessed_policies": 0,
        "assessed_outputs": 0,
    }
    if not paths:
        if summary is not None:
            summary.update(counts)
        return [{"status": "unloadable", "capsule": "<corpus>", "why": "no public capsule declarations found"}]
    for path in paths:
        directory = path.parent
        try:
            relative = path.relative_to(root)
            if any((root / Path(*relative.parts[:index])).is_symlink() for index in range(1, len(relative.parts) + 1)):
                raise ValueError("capsule declaration is indirect")
            capsule = capsule_common.load_capsule(directory)
        except Exception as exc:  # noqa: BLE001 - loading failure remains a failed gate
            found.append({"status": "unloadable", "capsule": str(directory), "why": str(exc)})
            continue
        policy = capsule.get("numeric_policy") or {}
        if policy.get("compare") != "tolerance_float":
            continue
        counts["applicable_policies"] += 1
        try:
            expected = capsule_golden.golden(capsule, directory)
        except capsule_golden.UnavailableGolden as exc:
            found.append({"status": "unmeasured", "capsule": str(directory), "why": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 - malformed oracle/evaluator failures cannot be allowed through
            found.append({"status": "unloadable", "capsule": str(directory), "why": f"oracle evaluation failed: {exc}"})
            continue
        # One unavailable output cannot discard a measured failure in another.
        outputs = (
            [{name: value} for name, value in expected.items()]
            if isinstance(expected, dict) and expected
            else [expected]
        )
        invalid = False
        for output in outputs:
            try:
                reports = numeric_falsifiability.audit_outputs(policy, output, require_measurable=True)
            except Exception as exc:  # noqa: BLE001 - invalid supplied outputs cannot be allowed through
                found.append(
                    {"status": "unloadable", "capsule": str(directory), "why": f"invalid oracle outputs: {exc}"}
                )
                invalid = True
                continue
            counts["assessed_outputs"] += 1
            found.extend(
                {"status": "accepted", "capsule": directory.name, "path": str(directory), **row} for row in reports
            )
        counts["assessed_policies"] += not invalid
    if summary is not None:
        summary.update(counts)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="explicit public corpus; default is tracked repository declarations")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--allow-unmeasured",
        action="store_true",
        help="report unavailable oracle outputs without granting a complete audit",
    )
    options = parser.parse_args(argv)
    root = options.root if options.root is not None else capsule_root()
    if not root.is_dir() or root.is_symlink():
        print(f"no ordinary public capsule corpus at {root}", file=sys.stderr)
        return 2
    try:
        coverage: dict = {}
        found = offenders(root, tracked_only=options.root is None, summary=coverage)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"numeric falsifiability: corpus discovery failed: {exc}", file=sys.stderr)
        return 2
    accepted = [row for row in found if row["status"] == "accepted"]
    unloadable = [row for row in found if row["status"] == "unloadable"]
    unmeasured = [row for row in found if row["status"] == "unmeasured"]
    report = {
        "root": str(root),
        "accepted": accepted,
        "unloadable": unloadable,
        "unmeasured": unmeasured,
        "complete": not (unloadable or unmeasured),
        "allow_unmeasured": options.allow_unmeasured,
        "coverage": coverage,
    }
    if options.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"numeric falsifiability: {len(accepted)} accepted constant answer(s), "
            f"{len(unloadable)} unloadable declaration(s), {len(unmeasured)} UNMEASURED oracle(s)"
        )
        print(f"  assessed {coverage['assessed_policies']}/{coverage['applicable_policies']} applicable policies")
        if unloadable or unmeasured:
            print("  PARTIAL audit: unavailable inputs have not received a clean numerical verdict")
        for row in accepted:
            print(f"  FAIL {row['capsule']}/{row['output']} accepts {row['answer']!r}")
        for row in unloadable + unmeasured:
            print(f"  {row['status'].upper()} {row['capsule']}: {row['why']}")
    if accepted:
        return 1
    if unloadable or (unmeasured and not options.allow_unmeasured):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
