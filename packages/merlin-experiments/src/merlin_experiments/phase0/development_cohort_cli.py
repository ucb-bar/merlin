"""Explicit bounded public source preparation; never run/admit a Phase 2 search."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from . import capacity_families as C
from . import development_cohort as F
from . import development_envelopes as E


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-architectures", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--capacity-policy", type=Path, required=True)
    parser.add_argument("--capacity-sweep", required=True)
    parser.add_argument("--budgets", type=Path, required=True)
    parser.add_argument("--max-spectrum", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    facts = json.loads(args.facts.read_bytes())
    policy = yaml.safe_load(args.capacity_policy.read_bytes())
    selected = [row for row in policy["sweeps"] if row["id"] == args.capacity_sweep]
    if len(selected) != 1:
        parser.error("select exactly one original capacity-family sweep")
    basis = C.derive_basis(args.target, facts)
    architecture = E.derive(
        args.public_architectures,
        axis_granules={"M": basis.tile_rows, "N": basis.tile_cols, "K": basis.tile_rows},
        resource_sha256=basis.digest(),
        max_spectrum=args.max_spectrum,
    )
    capacity, skipped = F.capacity_requests(selected[0], target=args.target, facts=facts)
    provenance = {
        "architecture_derivation": architecture,
        "capacity_skipped": skipped,
        "facts": F._identity(args.facts),
        "capacity_policy": F._identity(args.capacity_policy),
        "capacity_sweep": args.capacity_sweep,
        "producer_sources": {
            path.name: F._identity(path) for path in sorted(Path(__file__).parent.glob("development_*.py"))
        },
    }
    manifest = F.freeze(
        architecture["requests"] + capacity,
        args.output,
        provenance=provenance,
        budgets=json.loads(args.budgets.read_bytes()),
        resource_basis=basis.to_dict(),
        progress=lambda n, total: (
            print(f"source/reference preparation {n}/{total}", flush=True) if n % 10 == 0 else None
        ),
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output / "manifest.json"),
                "manifest_sha256": hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest(),
                "source_cohort_sha256": manifest["source_cohort_sha256"],
                "statuses": dict(Counter(row["status"] for row in manifest["members"])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
