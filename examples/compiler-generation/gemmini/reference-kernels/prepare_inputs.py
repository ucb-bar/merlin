"""Resolve the pinned reference selection against explicit local roots; never execute references.

The optional ahead-of-time compile inputs (pinned toolchain plus extra headers) are kept only
when both roots are given; otherwise the selection omits them and the builder stays parse-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def resolve(item, roots):
    role = item.pop("input_root")
    selected = roots[role] / item["path"] if "path" in item else roots[role]
    selected = selected.absolute()
    with selected.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
            raise ValueError("selected input differs from the pinned reference: " + str(selected))
    item["path"] = str(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "upstream-root", "bundle-root", "contract", "facts", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--toolchain-root", type=Path, help="pinned cross-toolchain install prefix (optional)")
    parser.add_argument("--aot-header-root", type=Path, help="pinned extra-header checkout (optional)")
    args = parser.parse_args()
    spec = json.loads(Path(__file__).with_name("inputs.template.json").read_bytes())
    roots = dict(
        input=args.input_root,
        upstream=args.upstream_root,
        bundle=args.bundle_root,
        contract=args.contract,
        facts=args.facts,
        toolchain=args.toolchain_root,
        aot_headers=args.aot_header_root,
    )
    for item in [*spec["archives"], *spec["files"], spec["contract"], spec["facts"]]:
        resolve(item, roots)
    if (args.toolchain_root is None) != (args.aot_header_root is None):
        raise ValueError("AOT compilation needs both --toolchain-root and --aot-header-root")
    if args.toolchain_root is None:
        spec.pop("aot")
    else:
        toolchain = spec["aot"]["toolchain"]
        pinned = [toolchain["driver"], *(c for c in toolchain["components"].values() if "input_root" in c)]
        for tree in spec["aot"]["include_trees"]:
            pinned.extend(f for f in tree["files"].values() if "input_root" in f)
        for item in pinned:
            resolve(item, roots)
    if args.out.exists():
        raise ValueError("choose a new input selection output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
