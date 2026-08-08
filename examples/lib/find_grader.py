"""Print the path of a `grade.py` that can actually score a log for MODEL.

Not just any grade.py. The script is the same in every package (one vendored template), but the
REFERENCES it scores against are the ones sitting next to it — so handing it a model that package does not
carry produces a confident-looking wrong story, or, for a log that never completed, a right answer for the
wrong reason. Picking by "first directory on the glob" was doing exactly that.

Preference order: an explicitly named package, then any package whose manifest lists the model, then
nothing — because "no grader for this model" is a better outcome than a grader that cannot see its
reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from merlin.common.artifacts import artifacts_dir


def _models(pkg: Path) -> set[str]:
    try:
        man = json.loads((pkg / "manifest.json").read_text())
    except Exception:                                                 # noqa: BLE001
        return set()
    return {b.get("model") for b in man.get("binaries", []) if b.get("model")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefer", default=None, help="a package directory to try first")
    a = ap.parse_args(argv)

    candidates = []
    if a.prefer:
        candidates.append(Path(a.prefer))
    root = artifacts_dir() / "delivery"
    if root.is_dir():
        candidates += sorted(p for p in root.iterdir() if p.is_dir() and (p / "grade.py").is_file())

    seen = set()
    for pkg in candidates:
        if pkg in seen or not (pkg / "grade.py").is_file():
            continue
        seen.add(pkg)
        if a.model in _models(pkg):
            print(pkg / "grade.py")
            return 0

    print(f"no packaged grader carries references for '{a.model}'. Build one:\n"
          f"  ./run.sh package --full        # or --models {a.model}\n"
          f"or unpack a delivered zip next to it:\n"
          f"  python -m zipfile -e <package>.zip {root}/", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
