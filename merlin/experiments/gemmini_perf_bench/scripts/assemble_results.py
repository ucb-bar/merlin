#!/usr/bin/env python3
"""Verify that per-replicate Arm4 files exactly compose the sealed result sequence.

``perf_results.json`` is sealed by the runner and is the sole reporting input.  This command never
rewrites or reorders it; it only proves that each independently written pair file agrees with it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _pbcommon as PB
import perf_reporting as PR


def _run_dir(run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise PR.ReportingGateError("performance run ID must be an explicit simple directory name")
    return PB.RUNS / run_id


def _pair_file(run: Path, identity: tuple[str, str, str]) -> Path:
    family, capsule, replicate = identity
    return run / f"{family}__{capsule}__{replicate}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run = _run_dir(args.run_id)
    campaign, sealed_rows, counts = PR.load_reportable_run(run)
    expected = PR.expected_identities(campaign)

    assembled: list[dict] = []
    used_paths: set[Path] = set()
    offset = 0
    while offset < len(expected):
        family, capsule, _simulator, replicate = expected[offset]
        base = (family, capsule, replicate)
        pair_expected = expected[offset:offset + 2]
        if (len(pair_expected) != 2
                or any((row[0], row[1], row[3]) != base for row in pair_expected)):
            raise PR.ReportingGateError(
                f"expected identities do not form one simulator pair: {base}")
        path = _pair_file(run, base)
        if path in used_paths:
            raise PR.ReportingGateError(f"multiple identities resolve to per-replicate file {path}")
        used_paths.add(path)
        if path.is_symlink() or not path.is_file():
            raise PR.ReportingGateError(f"per-replicate result is absent or linked: {path}")
        try:
            pair = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise PR.ReportingGateError(
                f"per-replicate result is unreadable at {path}: {exc}") from exc
        if not isinstance(pair, list) or len(pair) != 2:
            raise PR.ReportingGateError(
                f"per-replicate result must contain exactly two cells: {path}")
        observed = tuple(
            PR.identity_tuple(row.get("identity"), owner="per-replicate result")
            if isinstance(row, dict) else PR.identity_tuple(None, owner="per-replicate result")
            for row in pair
        )
        if observed != pair_expected:
            raise PR.ReportingGateError(
                f"per-replicate result {path.name!r} disagrees with expected identities")
        assembled.extend(pair)
        offset += 2

    if assembled != sealed_rows:
        raise PR.ReportingGateError(
            "per-replicate result files do not compose the exact sealed perf_results sequence")
    PR.validate_rows(assembled, expected, counts)
    print(f"verified {len(assembled)} sealed Arm4 cells from {len(used_paths)} "
          f"per-replicate files; left {run / 'perf_results.json'} unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
