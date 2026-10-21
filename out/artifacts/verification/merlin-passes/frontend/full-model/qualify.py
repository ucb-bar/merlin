#!/usr/bin/env python3
"""Reproduce actual full-boundary candidate checking and checkpoint-native qualification.

The returned receipts are inactive and do not assert full-model definedness. Native samples
qualify interpretation fidelity; they never supply facts to a Lean soundness proof.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

from merlin.common.paths import artifacts_dir, merlin_dir, repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import fix_bool_fptosi
from merlin.verify.lean import differential as diff
from merlin.verify.lean import frontend_runtime as runtime
from merlin.xdsl_dialects._common import text as to_text


def captured_boundary(destination: Path) -> tuple[Path, Path]:
    """Regenerate ignored inputs without rewriting any historical qualification record."""
    destination.mkdir(parents=True, exist_ok=True)
    capture = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    module = parse_mlir_text(capture.read_text())
    before, after = (destination / f"{side}.mlir" for side in ("before", "after"))
    before.write_text(to_text(module, generic=True))
    if fix_bool_fptosi(module) != 1:
        raise RuntimeError("the captured bool-repair population changed")
    after.write_text(to_text(module, generic=True))
    return before, after


def main() -> None:
    root = repo_root()
    frontend = artifacts_dir() / "verification/merlin-passes/frontend"
    output = frontend / "full-model"
    output.mkdir(parents=True, exist_ok=True)
    compiler = Path(os.environ["MERLIN_COMPILER_PYTHON"])
    before, after = captured_boundary(frontend / "actual")
    # No extraction assertion supplies a proof fact: Lean rechecks the actual exported ASTs.
    receipt = runtime.verify_candidate(
        before,
        after,
        output / "boundary",
        function="forward",
        start=91,
        compiler_python=compiler,
        suffix=(after,),
    )
    if receipt["status"] != "verified_candidate":
        raise RuntimeError(receipt.get("reason", "candidate unavailable"))
    runtime.validate_receipt(receipt)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    test_file = merlin_dir() / "tests/ir/test_lean_frontend_mask_runtime.py"
    spec = importlib.util.spec_from_file_location("mask_checkpoint_qualification", test_file)
    assert spec and spec.loader
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    tools = diff.Tools(
        compiler,
        root / "out/build/lean/build/bin/merlin-lean",
        Path.home() / ".elan/bin/lake",
        Path(os.environ["MERLIN_MLIR_INSTALL"]) / "lib/libmlir_c_runner_utils.so",
    )
    native = []
    for name, source, n in [(f"independent-{n}", tests.independent_selection(n), n) for n in (0, 1, 8, 1024)] + [
        ("actual-selection", tests.actual_selection(), 1024)
    ]:
        work = output / "work/native" / name
        tests.compare_selection(source, n, work, tools)
        result_path = work / "native/native-results.json"
        result = json.loads(result_path.read_text())
        native.append(
            {
                "case": name,
                "samples": len(result["results"]),
                "source": runtime._ref(work / "subject.mlir"),
                "export": runtime._ref(work / "subject.json"),
                "lean_results": runtime._ref(work / "candidate-results.json"),
                "native_results": runtime._ref(result_path),
                "mlir_revision": result["mlir_revision"],
                "runner_utils_sha256": result["runner_utils_sha256"],
                "pipeline": result["pipeline"],
            }
        )
    tests.test_pinned_torch_selection_oracle(output / "work")
    pinned = output / "work/pinned-torch-results.json"
    pinned_result = json.loads(pinned.read_text())
    result = {
        "schema": "merlin.frontend.runtime-checkpoint.qualification.v1",
        "status": "inactive candidate; runtime admission pending",
        "receipt": runtime._ref(output / "receipt.json"),
        "interpretation_sha256": receipt["interpretation"]["sha256"],
        "initialized_program_scope": "actual first 116 top-level operations, including all nested operations",
        "repair_policy_scope": "complete actual before/after module; all eligible f32 sites and unrelated operations",
        "suffix_scope": "one actual alpha-identical suffix stage; transformed suffixes explicitly unsupported",
        "whole_model_defined": False,
        "native_samples": sum(r["samples"] for r in native),
        "native": native,
        "pinned_torch": {
            "torch": pinned_result["torch"],
            "numpy": pinned_result["numpy"],
            "samples": len(pinned_result["results"]),
        },
        "pinned_torch_artifact": runtime._ref(pinned),
        "native_is_universal_proof": False,
    }
    (output / "qualification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "native_samples": result["native_samples"]}))


if __name__ == "__main__":
    main()
