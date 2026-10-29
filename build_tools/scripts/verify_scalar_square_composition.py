#!/usr/bin/env python3
"""Retain conditional scalar-square program claims without changing prior evidence or approvals."""

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
from merlin.verify.lean.scalar_square_composition import check, replay_program


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-python", type=Path, required=True)
    args = parser.parse_args()
    base = artifacts_dir() / "verification/merlin-passes/frontend/scalar-square"
    output = base / "composition"
    output.mkdir(parents=True, exist_ok=True)
    prior = json.loads((base / "summary.json").read_text())
    preserved = [_file_ref(base / "summary.json"), _file_ref(base / "review/manifest.json")]
    previous_manifest = json.loads((base / "review/manifest.json").read_text())
    for item in previous_manifest["native"]:
        preserved.extend(item["retained_files"])
    changed = [ref["path"] for ref in preserved if _file_ref(Path(ref["path"])) != ref]
    if changed:
        raise ValueError(f"prior native evidence was already changed: {changed}")
    write(output / "counterexamples-preserved.json", {"files": preserved})
    summary = {
        "schema": "merlin.lean.scalar-square.program-summary.v1",
        "started": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "captures": [],
        "counterexamples": _file_ref(output / "counterexamples-preserved.json"),
        "strict_source_claim_instantiated": False,
        "policy_claim_instantiated": False,
        "numerical_law_discharged": False,
        "policy_approved": False,
    }
    for captured in prior["captures"]:
        label = captured["label"]
        if Path(label).name != label or label in {".", ".."}:
            raise ValueError(label)
        print(f"Checking program composition: {label}", flush=True)
        original = base / label / "receipt.json"
        receipt = check(original, output / label, compiler_python=args.compiler_python)
        work = Path(receipt["report"]["path"]).parent
        print(f"Replaying program composition: {label}", flush=True)
        replayed = replay_program(
            output / label / "receipt.json", work / "replay", compiler_python=args.compiler_python
        )
        if not replayed["fresh"] or not replayed["matches_recorded"]:
            raise ValueError(replayed)
        report = json.loads(Path(receipt["report"]["path"]).read_text())
        original_receipt = json.loads(original.read_text())
        template = {
            "schema": "merlin.lean.scalar-square.policy-review.pending.v1",
            "approved": False,
            "reviewer": None,
            "source": original_receipt["inputs"]["before"]["original"],
            "source_export": original_receipt["exports"]["before"],
            "multiplication_policy_export": original_receipt["exports"]["after"],
            "program_correspondence": receipt["report"],
            "policy": "MerlinVerify.Rules.ScalarSquareClaims.policyMeaning",
            "entries": report["functions"],
            "index_bits": report["index_bits"],
            "float_model": None,
            "context_domain": None,
            "domain": "MerlinVerify.Rules.ScalarSquareClaims.Inputs(source, index_bits, entry)",
            "entry_domain_inhabited": "not_discharged",
            "policy_definedness": "not_discharged",
            "original_source_equivalence_claimed": False,
        }
        template_path = work / "policy-review.pending.json"
        write(template_path, template)
        summary["captures"].append(
            {
                "label": label,
                "receipt": _file_ref(output / label / "receipt.json"),
                "sites": report["covered_sites"],
                "unsupported_runtime_types": report["unsupported_runtime_types"],
                "audit_theorems": json.loads(Path(receipt["proof_audit"]["path"]).read_text())["theorems_audited"],
                "replay": replayed,
                "pending_policy_review": _file_ref(template_path),
            }
        )
    env = dict(os.environ, MERLIN_M2M_PYTHON=str(args.compiler_python), MERLIN_LEAN_REQUIRED="1")
    for name, mark, files in (
        ("fast", "not slow", ["test_lean_scalar_square_composition.py"]),
        ("slow", "slow", ["test_lean_scalar_square_composition.py", "test_lean_scalar_square_boundary.py"]),
    ):
        print(f"Qualification: {name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-m",
                    mark,
                    *["merlin/tests/ir/" + file for file in files],
                    "--junitxml=" + str(output / f"{name}.xml"),
                ],
                cwd=repo_root(),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
    changed = [ref["path"] for ref in preserved if _file_ref(Path(ref["path"])) != ref]
    if changed:
        raise ValueError(f"prior native evidence changed: {changed}")
    summary.update(completed=datetime.now(UTC).isoformat(), counterexamples_preserved=True)
    write(output / "summary.json", summary)
    history = output / "history" / uuid.uuid4().hex
    history.mkdir(parents=True)
    write(history / "summary.json", summary)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
