#!/usr/bin/env python3
"""Retain scalar-square checker, native observations, replay and an explicit review packet.

Pass --stage LABEL PATH for each captured prepare-scalar-squares directory. This runner does
not change the production pass, shared Lean executable, theorem pins or specification lock.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root
from merlin.verify.lean.check import _file_ref
from merlin.verify.lean.scalar_square import check_stage, replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", nargs=2, action="append", required=True, metavar=("LABEL", "PATH"))
    parser.add_argument("--compiler-python", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=artifacts_dir() / "verification/merlin-passes/frontend/scalar-square"
    )
    args = parser.parse_args()
    base = args.output.resolve()
    base.mkdir(parents=True, exist_ok=True)
    validation = base / "validation"
    validation.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex
    native_root = base / "native" / run_id
    sys.path.insert(0, str(repo_root() / "merlin/tests/ir"))
    from lean_scalar_square_cases import actual, captured_probe, scalar
    from lean_scalar_square_native import differential

    summary = {"started": datetime.now(UTC).isoformat(), "command": sys.argv, "captures": [], "native": []}
    for label, path in args.stage:
        if Path(label).name != label or label in {".", ".."}:
            raise ValueError("labels must be single directory names")
        stage = Path(path).resolve()
        print(f"Checking {label}", flush=True)
        receipt = check_stage(stage, base / label, compiler_python=args.compiler_python)
        if not receipt.get("conditional_match"):
            raise RuntimeError(f"boundary refused: {receipt}")
        print(f"Replaying {label}", flush=True)
        replayed = replay(
            base / label / "receipt.json",
            Path(receipt["report"]["path"]).parent / "replay",
            compiler_python=args.compiler_python,
        )
        if not replayed["fresh"] or not replayed["matches_recorded"]:
            raise RuntimeError(replayed)
        report = json.loads(Path(receipt["report"]["path"]).read_text())
        summary["captures"].append(
            {
                "label": label,
                "stage_meta": _file_ref(stage / "meta.json"),
                "receipt": _file_ref(base / label / "receipt.json"),
                "sites": report["covered_sites"],
                "replay": replayed,
                "audit_theorems": report["proof_audit"]["theorems_audited"],
            }
        )
        print(f"Native extracted operation: {label}", flush=True)
        before, after, count = captured_probe(stage)
        assert count == 1
        native = differential(before, after, "f32", 1, args.compiler_python, native_root / label)
        summary["native"].append(
            {
                "label": label,
                "report": _file_ref(native_root / label / "differential.json"),
                "samples": native["samples"],
                "jit_bit_mismatches": len(native["bit_mismatches"]),
                "forced_libm_status": native["forced_libm_status"],
            }
        )
    for kind in ("f32", "f64"):
        print(f"Native complete exceptional-value sample: {kind}", flush=True)
        before, after, _ = actual(scalar(kind))
        native = differential(before, after, kind, 2, args.compiler_python, native_root / kind)
        summary["native"].append(
            {
                "label": kind,
                "report": _file_ref(native_root / kind / "differential.json"),
                "samples": native["samples"],
                "jit_bit_mismatches": len(native["bit_mismatches"]),
                "forced_libm_status": native["forced_libm_status"],
            }
        )
    env = dict(os.environ, MERLIN_M2M_PYTHON=str(args.compiler_python), MERLIN_LEAN_REQUIRED="1")
    for mark, files in (
        (
            "not slow",
            ["test_lean_scalar_square_boundary.py", "test_lean_scalar_square_native.py", "test_lean_scalar_square.py"],
        ),
        ("slow", ["test_lean_scalar_square_boundary.py"]),
    ):
        label = "fast" if mark == "not slow" else "slow"
        print(f"Qualification tests: {label}", flush=True)
        with (validation / f"{label}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-m",
                    mark,
                    *["merlin/tests/ir/" + file for file in files],
                    "-q",
                    "--junitxml=" + str(validation / f"{label}.xml"),
                ],
                cwd=repo_root(),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
    summary.update(
        completed=datetime.now(UTC).isoformat(),
        status="provisional_numerical_law_not_discharged",
        numerical_law_discharged=False,
        whole_capture_source_equivalence_proved=False,
    )
    (base / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    history = validation / run_id
    history.mkdir()
    (history / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
