"""Resolve the pinned reference selection against explicit local roots; never execute references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "upstream-root", "bundle-root", "contract", "facts", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(Path(__file__).with_name("inputs.template.json").read_bytes())
    roots = dict(
        input=args.input_root,
        upstream=args.upstream_root,
        bundle=args.bundle_root,
        contract=args.contract,
        facts=args.facts,
    )
    for item in [*spec["archives"], *spec["files"], spec["contract"], spec["facts"]]:
        role = item.pop("input_root")
        selected = roots[role] / item["path"] if "path" in item else roots[role]
        selected = selected.absolute()
        with selected.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError("selected input differs from the pinned reference: " + str(selected))
        item["path"] = str(selected)
    if args.out.exists():
        raise ValueError("choose a new input selection output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
