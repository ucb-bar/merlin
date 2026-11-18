#!/usr/bin/env python3
"""Materialize an explicitly selected compiler library, without dependency discovery.

Run in the trusted host setup, never as a compiler candidate. The selection is
an authored source roster; hash/import-closure checks do not themselves review
or qualify it. Namespace-only initializers intentionally expose no parent API.
"""

import argparse
import json
import sys
from pathlib import Path

import _source_layout

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_source_layout.core_package(ROOT).parent))
from merlin.targetgen.compiler_library_export import export  # noqa: E402 - use the selected checkout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    contract = export(json.loads(args.selection.read_text()), args.source_root, args.destination)
    print(
        json.dumps(
            dict(
                contract_sha256=contract.sha256, members=len(contract.members), public_modules=contract.public_modules
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
