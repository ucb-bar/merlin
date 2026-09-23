#!/usr/bin/env python3
"""Gate: every RTL-facts artifact in the tree declares a family and matches that family's shape.

`merlin/contract/schemas/rtl_facts.schema.json` describes what a facts artifact is. A schema nothing
runs describes nothing: it validates in a test someone wrote once, drifts from the artifacts it claims
to cover, and reads as coverage. This is the caller.

FINDING THE ARTIFACTS IS STRUCTURAL, NOT BY FILENAME. A glob for `*facts*.json` catches per-capsule
receipts, coverage reports and DSE contracts, none of which are facts artifacts and all of which would
be reported as undecidable. An artifact is a JSON object with a `facts` MAPPING and at least one of
`schema_version` / `generator` / `inputs`. Receipts have none of that shape.

WHAT IT REFUSES TO CONCLUDE. A checkout with no facts artifact is not a clean checkout, it is one that
examined nothing, and it says which. RTL facts are generated and gitignored, so that is the NORMAL state
in CI -- which is exactly why the count is printed rather than implied. The tracked fixtures are real
artifacts and are covered, so CI is not vacuous.

Exit codes: 0 clean, 1 an artifact does not match its family, 2 CANNOT DECIDE (the schema is missing or
unreadable, or the work list could not be built).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "merlin" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_GATE = "rtl-facts-schema"

#: Where facts artifacts live. The cache is purgeable and usually absent; the fixtures are tracked and
#: are what makes this gate non-vacuous where there is no RTL.
SEARCH = (
    "merlin/targets/*/contracts/rtl_facts/facts.json",
    "merlin/tests/fixtures/**/contracts/rtl_facts/facts.json",
    "merlin/tests/data/**/*facts*.json",
    "out/artifacts/cache/rtl_introspect/*/facts.json",
    "out/artifacts/targets/*/contracts/rtl_facts/facts.json",
)


def is_facts_artifact(doc: object) -> bool:
    """Whether ``doc`` is a facts artifact, decided by SHAPE.

    An empty ``facts`` body is still an artifact -- it is the distinct state "the extractor ran and
    grounded nothing", which a consumer must be able to tell from a rich body, so it is validated too.
    """
    if not isinstance(doc, dict) or not isinstance(doc.get("facts"), dict):
        return False
    return any(k in doc for k in ("schema_version", "generator", "inputs"))


def artifacts(paths: list[Path] | None = None) -> list[tuple[Path, dict]]:
    found: list[tuple[Path, dict]] = []
    seen: set[Path] = set()
    candidates = paths if paths is not None else [p for pat in SEARCH for p in ROOT.glob(pat)]
    for path in sorted(candidates):
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # not JSON at all: not this gate's business
        if is_facts_artifact(doc):
            found.append((path, doc))
    return found


def verdict(found: list[tuple[Path, dict]]) -> tuple[list[str], int]:
    """``(problems, rc)`` -- pure, so a test can assert it without a tree."""
    from merlin.targetgen.rtl.facts import validate_facts

    problems: list[str] = []
    for path, doc in found:
        rel = path.relative_to(ROOT) if path.is_absolute() and ROOT in path.parents else path
        for problem in validate_facts(doc):
            problems.append(f"{rel}: {problem}")
    return problems, (1 if problems else 0)


def _unexaminable(reason: str, *, stop_hook: bool) -> int:
    text = f"{_GATE}: {reason}; NOTHING was examined, which is not the same as clean."
    if stop_hook:
        print(json.dumps({"decision": "block", "reason": text}))
        return 0
    print(f"[FAIL] {text}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--staged", action="store_true", help="only examine artifacts this commit touches")
    parser.add_argument("--stop-hook", action="store_true", help="report a refusal as a Stop-hook JSON decision")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    schema = ROOT / "merlin" / "contract" / "schemas" / "rtl_facts.schema.json"
    if not schema.is_file():
        return _unexaminable(f"no schema at {schema}", stop_hook=args.stop_hook)

    paths = None
    if args.staged:
        try:
            changed = subprocess.run(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()
        except (OSError, subprocess.CalledProcessError) as exc:
            return _unexaminable(f"could not list the staged files ({exc})", stop_hook=args.stop_hook)
        paths = [ROOT / c for c in changed if c.endswith(".json")]
        if not paths:
            return 0

    try:
        found = artifacts(paths)
        problems, rc = verdict(found)
    except Exception as exc:  # noqa: BLE001
        return _unexaminable(f"the artifacts could not be examined ({exc})", stop_hook=args.stop_hook)

    if args.json:
        print(json.dumps({"examined": [str(p) for p, _ in found], "problems": problems}, indent=2))
        return rc
    for problem in problems:
        print(f"[FAIL] {_GATE}: {problem}")
    tag = "FAIL" if problems else "  ok"
    print(
        f"[{tag}] {_GATE}: {len(found)} facts artifact(s) examined, {len(problems)} not matching their "
        "declared family. RTL facts are generated and gitignored, so a low count is normal where there "
        "is no RTL -- it is printed rather than implied so it cannot read as coverage."
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
