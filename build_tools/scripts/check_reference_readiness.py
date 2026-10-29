"""Inspect selected references through the existing analyzer and retain readiness.

No compiler, reference program, simulator or author is executed. Complete reports
are retained per region so bounded broker pages cannot hide unresolved coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    from merlin_experiments.phase1.feedback.reference_analysis import _analyze
    from merlin_experiments.phase1.feedback.reference_readiness import assess

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--reference",
        action="append",
        required=True,
        help="visible workspace-relative ELF, listing or disassembly; repeat for each artifact",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, path in enumerate(args.reference):
        kwargs = dict(
            workspace=workspace,
            reference_root=None,
            facts=args.facts.absolute(),
            contract=args.contract.absolute(),
            target=args.target,
        )
        inventory, _ = _analyze(dict(operation="inventory", path=path, limit=16), **kwargs)
        all_regions = list(inventory.get("regions", []))
        while inventory.get("next_offset") is not None:
            inventory, _ = _analyze(
                dict(operation="inventory", path=path, limit=16, offset=inventory["next_offset"]), **kwargs
            )
            all_regions.extend(inventory.get("regions", []))
        lifted = []
        for region in all_regions:
            selected = region["stable_id"]
            report, _ = _analyze(dict(path=path, region=selected, limit=1), **kwargs)
            lifted.extend(report["regions"])
            (output / f"{index}-{selected}.json").write_text(json.dumps(report, indent=2) + "\n")
            projection = report["regions"][0].get("mechanism_projection") or {}
            print(
                path,
                selected,
                projection.get("decoded_sites"),
                projection.get("counted_control", {}).get("status"),
                flush=True,
            )
        reports.append(dict(inventory, regions=lifted))
    result = assess(reports)
    (output / "readiness.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "region_count", "ready_regions", "missing")}), flush=True)
    return 0 if result["launch_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
