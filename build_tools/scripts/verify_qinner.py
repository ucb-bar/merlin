#!/usr/bin/env python3
"""Reproduce isolated Qinner proofs, actual-pass boundaries, byte certificates and native evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root
from merlin.verify.lean.check import _file_ref
from merlin.verify.lean.qinner import actual, audit, check_boundary, replay, write


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--compiler-python", type=Path, required=True)
    parser.add_argument("--stage", nargs=3, action="append", default=[], metavar=("LABEL", "PROPAGATION", "LIFT"))
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    sys.path.insert(0, str(repo_root() / "merlin/tests/ir"))
    import numpy as np
    from lean_qinner_cases import arrays, fixture, native_compare

    base = (options.output or artifacts_dir() / "verification/merlin-passes/frontend/qinner").resolve()
    run_id = uuid.uuid4().hex
    run = base / "runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    write(run / "proof-audit.json", audit())
    retained = []
    for label, propagation, lift in options.stage:
        source = Path(propagation) / "before.mlir"
        target = Path(lift) / "after.mlir"
        retained.append((label, source, target, None))
    for label, kind, shape, nested in (
        ("f32-direct", "f32", (4,), False),
        ("f32-nested", "f32", (4,), True),
        ("f64-rank3", "f64", (1, 2, 1), False),
        ("i8-nested", "i8", (4,), True),
        ("f32-empty", "f32", (0, 3), False),
        ("i1-direct", "i1", (4,), False),
    ):
        work = run / label
        work.mkdir()
        source = work / "before.mlir"
        source.write_text(fixture(kind, shape, nested=nested))
        target = work / "after.mlir"
        target.write_text(actual(source.read_text())["lifted"])
        extra = work / "extra.npz"
        np.savez(extra, **arrays(kind, shape))
        retained.append((label, source, target, extra))
    boundaries = []
    for label, source, target, extra in retained:
        print(f"checking {label}", flush=True)
        folder = base / "boundaries" / label
        attempt = folder / "attempts" / run_id
        receipt = check_boundary(source, target, attempt, compiler_python=options.compiler_python, extra_path=extra)
        assert receipt["checked"], (label, receipt)
        write(folder / "receipt.json", receipt)
        result = replay(folder / "receipt.json", attempt / "replay", compiler_python=options.compiler_python)
        assert result["fresh"] and result["checked"] and result["matches_recorded"], (label, result)
        boundaries.append(
            {
                "label": label,
                "kind": receipt["boundary_kind"],
                "sites": receipt["covered_sites"],
                "receipt": str(folder / "receipt.json"),
                "original_inputs": [_file_ref(source), _file_ref(target)],
                "replay": result,
            }
        )
    native = []
    for kind, shape in (("f32", (4,)), ("f64", (1, 2, 1)), ("i8", (4,)), ("f32", (0, 3))):
        for nested in (False, True):
            label = f"{kind}-{'x'.join(map(str, shape))}-{'nested' if nested else 'direct'}"
            print(f"native {label}", flush=True)
            result = native_compare(
                options.compiler_python, base / "native" / run_id / label, kind=kind, shape=shape, nested=nested
            )
            native.append(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"inputs", "extended_inputs", "results", "reference_results"}
                }
            )
    summary = {
        "run_id": run_id,
        "approved": False,
        "verdict": "abstracted",
        "boundaries": boundaries,
        "native": native,
        "runtime_archive_authenticity_proved": False,
        "target_descriptor_delivery_proved": False,
        "numerical_laws_used": [],
        "shared_driver_changed": False,
    }
    write(run / "summary.json", summary)
    write(base / "summary.json", summary)
    manifest = []
    owned = [
        repo_root() / "src/merlin/verify/lean/qinner.py",
        repo_root() / "build_tools/scripts/verify_qinner.py",
        *list((repo_root() / "merlin/lean/MerlinVerify").rglob("Qinner*.lean")),
        repo_root() / "merlin/tests/ir/lean_qinner_cases.py",
        repo_root() / "merlin/tests/ir/test_lean_qinner.py",
    ]
    for path in sorted(
        [*owned, *(path for path in base.rglob("*") if path.is_file() and path.name != "manifest.json")]
    ):
        manifest.append({"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write(base / "manifest.json", manifest)
    print(json.dumps({"run_id": run_id, "boundaries": len(boundaries), "native_cases": len(native), "approved": False}))


if __name__ == "__main__":
    main()
