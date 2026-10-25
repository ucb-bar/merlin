#!/usr/bin/env python3
"""Retain complete Overrank boundaries, native comparisons, proof audit and replay evidence.

Each --stage LABEL PATH names a captured prepare-collapse-overrank-matmul directory.
Positive cases are constructed and transformed by the production pass, including nested regions.
Shared Main/CLI, reviewed specifications and other workstreams are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root
from merlin.verify.lean.check import _file_ref
from merlin.verify.lean.overrank import audit, check_boundary, replay, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", nargs=2, action="append", default=[], metavar=("LABEL", "PATH"))
    parser.add_argument("--compiler-python", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=artifacts_dir() / "verification/merlin-passes/frontend/overrank-matmul"
    )
    args = parser.parse_args()
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    history = base / "runs" / run_id
    history.mkdir(parents=True)
    sys.path.insert(0, str(repo_root() / "merlin/tests/ir"))
    from lean_overrank_cases import actual, fixture
    from lean_overrank_native import CASES, compare

    summary = {
        "schema": "merlin.lean.overrank-review.v1",
        "run_id": run_id,
        "started": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "boundaries": [],
        "native": [],
        "intent_approved": False,
        "literal_source_equivalence_claimed": False,
    }

    def retain(label, before, after, metadata=None):
        if Path(label).name != label or label in {".", ".."}:
            raise ValueError("invalid boundary label")
        destination = base / "boundaries" / label
        attempt = destination / "attempts" / run_id
        print("Check", label, flush=True)
        receipt = check_boundary(before, after, attempt, compiler_python=args.compiler_python)
        if not receipt["checked"]:
            raise RuntimeError(receipt)
        shutil.copyfile(attempt / "receipt.json", destination / "receipt.json")
        result = replay(destination / "receipt.json", attempt / "replay", compiler_python=args.compiler_python)
        if not result["fresh"] or not result["matches_recorded"] or not result["checked"]:
            raise RuntimeError(result)
        summary["boundaries"].append(
            {
                "label": label,
                "receipt": _file_ref(destination / "receipt.json"),
                "sites": receipt["covered_sites"],
                "replay": _file_ref(attempt / "replay/replay.json"),
                "captured_stage_metadata": _file_ref(metadata) if metadata else None,
                "retained_model_exercises_pass": bool(receipt["covered_sites"]) if metadata else None,
            }
        )

    for label, path in args.stage:
        stage = Path(path).resolve()
        meta = json.loads((stage / "meta.json").read_text())
        if meta["boundary"] != "prepare-collapse-overrank-matmul":
            raise ValueError("wrong captured boundary")
        for side in ("before", "after"):
            digest = hashlib.sha256((stage / f"{side}.mlir").read_bytes()).hexdigest()
            if meta.get(f"{side}_sha256", digest) != digest:
                raise ValueError("captured boundary digest changed")
        retain(label, stage / "before.mlir", stage / "after.mlir", stage / "meta.json")
    for kind in ("f32", "f64"):
        for index, (batch, k, n) in enumerate(CASES):
            label = f"{kind}-shape-{index}"
            case = history / "fixtures" / label
            case.mkdir(parents=True)
            before, after, count = actual(fixture(batch, k, n, kind))
            assert count == 1
            (case / "before.mlir").write_text(before)
            (case / "after.mlir").write_text(after)
            retain(label, case / "before.mlir", case / "after.mlir")
            print("Native", label, flush=True)
            native = base / "native" / run_id / label
            report = compare(batch, k, n, kind, args.compiler_python, native)
            summary["native"].append(
                {
                    "label": label,
                    "report": _file_ref(native / "comparison.json"),
                    "samples": report["samples"],
                    "nan_payload_bit_mismatch_samples": len(report["bit_mismatches"]),
                    "non_nan_mismatches": len(report["non_nan_mismatches"]),
                }
            )
    case = history / "fixtures/nested"
    case.mkdir(parents=True)
    before, after, count = actual(fixture(nested=True))
    assert count == 2
    (case / "before.mlir").write_text(before)
    (case / "after.mlir").write_text(after)
    retain("nested", case / "before.mlir", case / "after.mlir")
    write(history / "proof-audit.json", audit())
    summary.update(
        completed=datetime.now(UTC).isoformat(),
        proof_audit=_file_ref(history / "proof-audit.json"),
        status="proved_relative_to_provisional_intent_no_human_approval",
        native_contract="exact bits for non-NaN results; NaN classification, payload differences retained",
    )
    checkpoint = base / "checkpoint/preserved.json"
    if checkpoint.is_file():
        preserved = json.loads(checkpoint.read_text())
        changed = [
            r["path"] for r in preserved if hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() != r["sha256"]
        ]
        if changed:
            raise RuntimeError({"preserved_checkpoint_changed": changed})
        summary["preserved_checkpoint"] = {
            "files_checked": len(preserved),
            "changed": [],
            "manifest": _file_ref(checkpoint),
        }
    write(history / "summary.json", summary)
    write(base / "summary.json", summary)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
