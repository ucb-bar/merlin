#!/usr/bin/env python3
"""Qualify the actual composed checkpoint and exact same-interpretation suffixes."""

from __future__ import annotations

import json
import os
import subprocess
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
    import test_lean_frontend_embedding as embedding
    import test_lean_frontend_positions as positions
    import test_lean_frontend_runtime_cse as sharing

    output = artifacts_dir() / "verification/merlin-passes/frontend/position-bounds"
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
        continuation_count=123,
    )
    if receipt["status"] != "verified_candidate":
        raise RuntimeError(receipt.get("reason", "checker unavailable"))
    runtime.validate_receipt(receipt)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # A refused continuation is recorded separately; it cannot be relabelled as
    # successful full-model execution or hidden by the successful repair claim.
    request = json.loads(Path(receipt["request"]["path"]).read_text())
    request.update(mode="position_scope", count=125)
    path = work / "first-nonboolean-cast-request.json"
    path.write_text(json.dumps(request))
    proc = subprocess.run(
        [str(repo_root() / "out/build/lean/build/bin/merlin-lean-frontend-runtime"), str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    refusal = json.loads(proc.stdout)
    if refusal["accepted"]:
        raise RuntimeError("nonboolean float conversion unexpectedly accepted")
    (work / "first-nonboolean-cast-result.json").write_text(json.dumps(refusal, indent=2) + "\n")
    tools = diff.Tools(
        compiler,
        repo_root() / "out/build/lean/build/bin/merlin-lean",
        Path.home() / ".elan/bin/lake",
        Path(os.environ["MERLIN_MLIR_INSTALL"]) / "lib/libmlir_c_runner_utils.so",
    )
    native = []

    def record(name: str, native_path: Path, lean_path: Path | None = None) -> None:
        result = json.loads(native_path.read_text())
        native.append(
            {
                "case": name,
                "samples": len(result["results"]),
                "native": runtime._ref(native_path),
                "lean": runtime._ref(lean_path) if lean_path else None,
                "mlir_revision": result["mlir_revision"],
                "pipeline": result["pipeline"],
                "runner_utils_sha256": result["runner_utils_sha256"],
            }
        )

    for rows, red in ((0, 31), (2, 0), (1, 1), (2, 31), (32, 31), (32, 31)):
        slot = len(native)
        name = f"count-{rows}-{red}-{slot}"
        text = (
            positions.actual_slice(71 if slot == 4 else 75, 4) if slot >= 4 else positions.independent_count(rows, red)
        )
        dest = work / name
        if not positions.checked(text, dest / "proof", tools, upper=red + 1)["accepted"]:
            raise RuntimeError(f"count postcondition refused: {name}")
        positions.compare_count(text, rows, red, dest, tools)
        record(name, dest / "native/native-results.json", dest / "execute/lean-results.json")
    dest = work / "embedding"
    dest.mkdir(exist_ok=True)
    embedding.test_actual_embedding_bounds_native_and_pinned_oracle(dest, tools)
    record("actual-embedding-observer", dest / "native/native-results.json")
    for nested, callee in ((False, False), (True, False), (True, True)):
        name = f"closed-sharing-{nested}-{callee}"
        dest = work / name
        dest.mkdir(exist_ok=True)
        sharing.test_closed_cse_preserves_complete_program(dest, tools, nested, callee)
        for label in ("before", "after"):
            record(name + "-" + label, dest / label / "native/native-results.json", dest / label / "lean-results.json")
    dest = work / "pinned-counts"
    dest.mkdir(exist_ok=True)
    positions.test_pinned_torch_count_oracle(dest)
    pinned = [
        runtime._ref(dest / "pinned-torch-counts.json"),
        runtime._ref(work / "embedding/pinned-torch-embedding.json"),
    ]
    qualification = {
        "schema": "merlin.frontend.position-bounds.qualification.v1",
        "active": False,
        "receipt": runtime._ref(output / "receipt.json"),
        "interpretation_sha256": receipt["interpretation"]["sha256"],
        "initial_segment_top_level_operations": 239,
        "selection_start": 91,
        "continuation_count": 123,
        "whole_function_defined": False,
        "ordinary_source_equivalence": False,
        "native_samples": sum(row["samples"] for row in native),
        "native": native,
        "pinned_oracles": pinned,
        "embedding_test": "native versus pinned Torch; universal execution/bounds checked separately in Lean",
        "nonalpha_suffix_receipt": (
            "qualified separately on actual 157-op checkpoint plus two synthetic closed constants"
        ),
        "native_is_universal_proof": False,
        "skipped": [],
        "first_nonboolean_conversion_refusal": runtime._ref(work / "first-nonboolean-cast-result.json"),
    }
    (output / "qualification.json").write_text(json.dumps(qualification, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "accepted": True,
                "initial_operations": 239,
                "native_samples": qualification["native_samples"],
                "whole_function_defined": False,
            }
        )
    )


if __name__ == "__main__":
    main()
