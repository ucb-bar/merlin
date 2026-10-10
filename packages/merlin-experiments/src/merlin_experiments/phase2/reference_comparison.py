"""OPERATOR-SIDE comparison of a measured-claims campaign with a reference implementation's cycles.

    rows = reference_rows(paired_rows, load_reference(path))      # one row per measured cell
    document = comparison_document(rows, reference, campaign=...)  # written beside the campaign

A reference document (``merlin_perf_reference_v1``) carries per-capsule cycle counts of some other
compiler on the same timing engine, plus the identity of that engine. This module joins it to the
campaign's ``paired_cycles.json`` rows and reports ``candidate / reference`` and ``baseline /
reference`` for every capsule both sides measured. It is a REPORT column for the operator: it is
written only into the campaign's own host-side measurement directory, never into a stage mount,
broker response or agent-visible feedback document, and it never gates the campaign.

A reference cycle count measured on a different engine binary than the campaign's is not a divisor:
the join refuses it (``engine_mismatch``) rather than computing a ratio across machines. A capsule the
reference did not measure is stated as such, never omitted.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "merlin_perf_reference_v1"
COMPARISON_SCHEMA = "merlin_perf_reference_comparison_v1"
FILENAME = "reference_comparison.json"


class ReferenceError(ValueError):
    pass


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def load_reference(path: Path) -> dict[str, Any]:
    """Read and shape-check a ``merlin_perf_reference_v1`` document."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA:
        raise ReferenceError(f"{path} is not a {SCHEMA} document")
    capsules = document.get("capsules")
    if not isinstance(capsules, Mapping):
        raise ReferenceError(f"{path} carries no per-capsule table")
    for name, row in capsules.items():
        if not isinstance(row, Mapping):
            raise ReferenceError(f"{path}: capsule {name!r} is not a mapping")
    return dict(document)


def reference_rows(
    paired_rows: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    *,
    campaign_engine_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """One row per paired cell, with the reference's cycles and both ratios where they are defined."""
    capsules = reference.get("capsules") if isinstance(reference.get("capsules"), Mapping) else {}
    reference_engine = (
        (reference.get("engine") or {}).get("binary_sha256") if isinstance(reference.get("engine"), Mapping) else None
    )
    engine_mismatch = bool(campaign_engine_sha256 and reference_engine and campaign_engine_sha256 != reference_engine)
    rows = []
    for row in paired_rows:
        capsule = str(row.get("capsule"))
        entry = capsules.get(capsule)
        cycles = _positive_int(entry.get("gsim_cycles")) if isinstance(entry, Mapping) else None
        out = {
            "family": row.get("family"),
            "capsule": capsule,
            "replicate": row.get("replicate"),
            "baseline_cycles": row.get("baseline_cycles"),
            "candidate_cycles": row.get("candidate_cycles"),
            "reference_cycles": cycles,
            "candidate_over_reference": None,
            "baseline_over_reference": None,
            "state": "compared",
        }
        if engine_mismatch:
            out["state"] = "engine_mismatch"
        elif cycles is None:
            out["state"] = "reference_unmeasured"
        else:
            for arm in ("candidate", "baseline"):
                measured = _positive_int(row.get(f"{arm}_cycles"))
                if measured is not None and row.get("comparable") is True:
                    out[f"{arm}_over_reference"] = round(measured / cycles, 4)
            if out["candidate_over_reference"] is None:
                out["state"] = "campaign_not_comparable"
        rows.append(out)
    return rows


def _geomean(values: Sequence[float]) -> float | None:
    import math  # noqa: PLC0415

    values = [v for v in values if v and v > 0]
    if not values:
        return None
    return round(math.exp(sum(math.log(v) for v in values) / len(values)), 4)


def comparison_document(
    rows: Sequence[Mapping[str, Any]], reference: Mapping[str, Any], *, campaign: str
) -> dict[str, Any]:
    compared = [r for r in rows if r.get("state") == "compared"]
    total_ref = sum(int(r["reference_cycles"]) for r in compared)
    total_cand = sum(int(r["candidate_cycles"]) for r in compared)
    return {
        "schema": COMPARISON_SCHEMA,
        "visibility": "operator_only",
        "campaign": campaign,
        "reference": {
            "label": reference.get("label"),
            "engine": reference.get("engine"),
            "source": reference.get("source"),
        },
        "rows": list(rows),
        "summary": {
            "cells": len(rows),
            "compared": len(compared),
            "states": {s: sum(1 for r in rows if r.get("state") == s) for s in sorted({r["state"] for r in rows})},
            "geomean_candidate_over_reference": _geomean([r["candidate_over_reference"] for r in compared]),
            "geomean_baseline_over_reference": _geomean(
                [r["baseline_over_reference"] for r in compared if r.get("baseline_over_reference")]
            ),
            "total_candidate_over_total_reference": round(total_cand / total_ref, 4) if total_ref else None,
        },
    }


def write_comparison(campaign_dir: Path, reference_path: Path, *, campaign_engine_sha256: str | None = None) -> Path:
    """Join ``<campaign_dir>/paired_cycles.json`` with the reference; write the sidecar beside it."""
    campaign_dir = Path(campaign_dir)
    paired = json.loads((campaign_dir / "paired_cycles.json").read_text(encoding="utf-8"))
    reference = load_reference(reference_path)
    rows = reference_rows(paired, reference, campaign_engine_sha256=campaign_engine_sha256)
    document = comparison_document(rows, reference, campaign=campaign_dir.name)
    document["reference"]["path"] = str(Path(reference_path).resolve())
    target = campaign_dir / FILENAME
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("campaign_dir", type=Path, help="a measured-claims campaign directory (paired_cycles.json)")
    parser.add_argument("--reference", type=Path, required=True, help="a merlin_perf_reference_v1 JSON")
    parser.add_argument("--engine-sha256", default=None, help="the campaign's timing-engine binary SHA-256")
    args = parser.parse_args(argv)
    path = write_comparison(args.campaign_dir, args.reference, campaign_engine_sha256=args.engine_sha256)
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    print(
        f"{path}: {summary['compared']}/{summary['cells']} compared; "
        f"geomean candidate/reference {summary['geomean_candidate_over_reference']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
