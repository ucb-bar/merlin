#!/usr/bin/env python3
"""Fresh actual-boundary/slice receipt and native attention-index qualification."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from merlin.common.paths import artifacts_dir, merlin_dir, repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import fix_bool_fptosi
from merlin.verify.lean import differential as diff
from merlin.verify.lean import frontend_runtime as runtime
from merlin.xdsl_dialects._common import text as to_text


def main() -> None:
    sys.path.insert(0, str(merlin_dir() / "tests/ir"))
    import test_lean_frontend_ranges as cases

    output = artifacts_dir() / "verification/merlin-passes/frontend/index-bounds"
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    compiler = Path(os.environ["MERLIN_COMPILER_PYTHON"])
    source = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    module = parse_mlir_text(source.read_text())
    before, after = work / "before.mlir", work / "after.mlir"
    before.write_text(to_text(module, generic=True))
    if fix_bool_fptosi(module) != 1:
        raise RuntimeError("captured repair population changed")
    after.write_text(to_text(module, generic=True))
    receipt = runtime.verify_candidate(
        before,
        after,
        work / "boundary",
        function="forward",
        start=91,
        compiler_python=compiler,
        suffix=(after,),
        bounds_slices=((122, 35),),
    )
    if receipt["status"] != "verified_candidate":
        raise RuntimeError(receipt.get("reason", "checker unavailable"))
    runtime.validate_receipt(receipt)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tools = diff.Tools(
        compiler,
        repo_root() / "out/build/lean/build/bin/merlin-lean",
        Path.home() / ".elan/bin/lake",
        Path(os.environ["MERLIN_MLIR_INSTALL"]) / "lib/libmlir_c_runner_utils.so",
    )
    native = []
    for name, text, n in [(f"independent-{n}", cases.independent_gather(n), n) for n in (0, 1, 8)] + [
        ("actual-attention", cases.actual_attention(), 1024),
        ("independent-position-grid", cases.independent_positions(), 16),
    ]:
        samples = [[[str(i % 2) for i in range(n)]], [["1"] * n], [["0"] * n]]
        dest = work / name
        answer = cases.checked(text, dest / "proof", tools)
        if not answer["accepted"]:
            raise RuntimeError(f"Lean refused {name}")
        lean, document, mlir = cases.candidate(text, samples, dest / "execute", tools)
        inputs, outputs = diff.signature(document, "subject")
        actual = diff.execute_mlir(mlir, samples, dest / "native", tools, inputs=inputs, outputs=outputs)
        diff.assert_equal(lean, actual["results"], outputs, samples)
        native.append(
            {
                "case": name,
                "samples": len(samples),
                "mlir_revision": actual["mlir_revision"],
                "pipeline": actual["pipeline"],
                "runner_utils_sha256": actual["runner_utils_sha256"],
                "mlir": runtime._ref(mlir),
                "native": runtime._ref(dest / "native/native-results.json"),
                "lean": runtime._ref(dest / "execute/candidate-results.json"),
            }
        )
    qualification = {
        "schema": "merlin.frontend.index-bounds.qualification.v1",
        "active": False,
        "receipt": runtime._ref(output / "receipt.json"),
        "interpretation_sha256": receipt["interpretation"]["sha256"],
        "initial_segment_top_level_operations": 116,
        "independent_actual_slice": {"start": 122, "count": 35, "input_domain": "typed initialized cut inputs"},
        "whole_function_defined": False,
        "ordinary_source_equivalence": False,
        "native_samples": sum(r["samples"] for r in native),
        "native": native,
        "native_is_universal_proof": False,
        "skipped": [],
    }
    (output / "qualification.json").write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"accepted": True, "native_samples": qualification["native_samples"], "whole_function_defined": False}
        )
    )


if __name__ == "__main__":
    main()
