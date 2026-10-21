#!/usr/bin/env python3
"""Retain inactive-candidate qualification bound to actual exports and model source bytes.

Run after the Lean build from this checkout with PYTHONPATH=src and the documented oracle
variables. This orchestrates real Lean checks and upstream execution; it supplies no proof facts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import fix_bool_fptosi
from merlin.verify.lean import differential as diff
from merlin.verify.lean.export import export_file
from merlin.xdsl_dialects._common import text as to_text


def snapshot(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def call(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=600).stdout


def main() -> None:
    root = repo_root()
    output = Path(__file__).resolve().parent
    work = output / "work"
    work.mkdir(parents=True, exist_ok=True)
    fixture = root / "merlin/tests/fixtures/lean_frontend_mask_runtime.lean"
    tests = root / "merlin/tests/ir/test_lean_frontend_mask_runtime.py"
    spec = importlib.util.spec_from_file_location("frontend_mask_qualification", tests)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compiler = Path(os.environ["MERLIN_COMPILER_PYTHON"])
    pinned = Path(os.environ["MERLIN_TORCH_PYTHON"])
    lake = Path.home() / ".elan/bin/lake"
    checker = root / "out/build/lean/build/bin/merlin-lean"
    runner = Path(os.environ["MERLIN_MLIR_INSTALL"]) / "lib/libmlir_c_runner_utils.so"
    tools = diff.Tools(compiler, checker, lake, runner)
    bindings: dict = {}
    frontend = output.parent
    actual_dir = frontend / "actual"
    actual_dir.mkdir(parents=True, exist_ok=True)
    capture = root / "merlin/contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    captured = parse_mlir_text(capture.read_text())
    before = to_text(captured, generic=True)
    assert fix_bool_fptosi(captured) == 1
    after = to_text(captured, generic=True)
    previous = json.loads((frontend / "qualification-20261021.json").read_text())
    for side, source in (("before", before), ("after", after)):
        mlir, exported = actual_dir / f"{side}.mlir", actual_dir / f"{side}.json"
        mlir.write_text(source)
        export_file(mlir, exported, backend="mlir", m2m_python=str(compiler))
        assert snapshot(mlir)["sha256"] == previous[side]["sha256"]
        assert snapshot(exported)["sha256"] == previous["exports"][side]["sha256"]
    for side in ("before", "after"):
        exported = frontend / "actual" / f"{side}.json"
        request = work / f"{side}-inspect-request.json"
        request.write_text(json.dumps({"export": exported.read_text(), "inspect": True}))
        result = json.loads(
            call([str(lake), "env", "lean", "--run", str(fixture), str(request)], cwd=root / "merlin/lean")
        )
        assert result["projections"][0]["accepted"], result
        assert result["count_providers"] == [True, False], result
        assert result["mask_alias"], result
        bindings[side] = {
            "mlir": snapshot(frontend / "actual" / f"{side}.mlir"),
            "export": snapshot(exported),
            "lean": result,
        }
    repair = json.loads(
        call(
            [
                str(checker),
                "check-frontend-bool",
                str(frontend / "actual/before.json"),
                str(frontend / "actual/after.json"),
            ]
        )
    )
    assert repair["verdict"] == "verified", repair
    repair_request = work / "candidate-repair-request.json"
    repair_request.write_text(
        json.dumps(
            {
                "export": (actual_dir / "after.json").read_text(),
                "repair_export": (actual_dir / "before.json").read_text(),
            }
        )
    )
    candidate_repair = json.loads(
        call([str(lake), "env", "lean", "--run", str(fixture), str(repair_request)], cwd=root / "merlin/lean")
    )
    assert candidate_repair["accepted"], candidate_repair
    actual = module.actual_selection()
    (output / "smolvla-cut.mlir").write_text(actual)
    cut = module.inspect_candidate(actual, work / "actual-cut-check", tools)
    assert cut["whole_cut"]
    native = []
    for name, source, n in [(f"independent-{n}", module.independent_selection(n), n) for n in (0, 1, 8, 1024)] + [
        ("actual-smolvla", actual, 1024)
    ]:
        case = work / name
        module.compare_selection(source, n, case, tools)
        result = json.loads((case / "native/native-results.json").read_text())
        native.append(
            {
                "case": name,
                "elements": n,
                "samples": len(result["results"]),
                "mlir_revision": result["mlir_revision"],
                "pipeline": result["pipeline"],
                "map_contract": result["map_contract"],
                "source": snapshot(case / "subject.mlir"),
                "export": snapshot(case / "subject.json"),
                "lean_results": snapshot(case / "candidate-results.json"),
                "native_results": snapshot(case / "native/native-results.json"),
                "runner_utils_sha256": result["runner_utils_sha256"],
            }
        )
    module.test_pinned_torch_selection_oracle(work)
    oracle = json.loads((work / "pinned-torch-results.json").read_text())
    sources = [
        root / "merlin/lean/MerlinVerify/Rules" / f"{name}.lean"
        for name in (
            "FrontendMaskRuntime",
            "FrontendMaskBounds",
            "FrontendMaskProgram",
            "FrontendMaskExecution",
            "FrontendMaskCount",
            "FrontendMaskCountProgram",
            "FrontendMaskAlias",
            "FrontendMaskTemplate",
            "FrontendMaskWhole",
            "FrontendMaskProjection",
            "FrontendMaskRefusals",
            "FrontendBoolSimulationWith",
            "FrontendBoolCandidateChain",
        )
    ]
    sources += [root / "merlin/lean/MerlinVerify/Rules/FrontendF32Comparison.lean", fixture, tests]
    result = {
        "schema": "merlin.frontend.runtime-candidate.qualification.v1",
        "status": "proved under inactive candidate; runtime interpretation and admission await review",
        "scope": "46-operation SmolVLA cut plus its function wrapper; not full-model progress",
        "iteration_limit": 1024,
        "index_bits": 64,
        "input_contract": [
            "initialized tensor<1x32x32xi1>",
            "initialized tensor<1x1024xi64>",
            "initialized tensor<1x1024xi64>",
        ],
        "formal_theorems": [
            "MerlinVerify.Rules.FrontendMaskProjection.check_sound",
            "MerlinVerify.Rules.FrontendMaskWhole.check_sound",
            "MerlinVerify.Rules.FrontendBoolSimulationWith.check_sound_candidate",
            "MerlinVerify.Rules.FrontendBoolCandidateChain.compose",
        ],
        "bindings": bindings,
        "cut": {"source": snapshot(output / "smolvla-cut.mlir"), "lean": cut},
        "repair_acceptance": {
            "result": repair,
            "candidate_result": candidate_repair,
            "checker": snapshot(checker),
            "note": (
                "Same actual-AST acceptance predicate; its candidate soundness theorem names the explicit new OpSem. "
                "This does not alter an active pipeline receipt."
            ),
        },
        "sources": [snapshot(path) for path in sources],
        "specification": snapshot(root / "merlin/lean/MerlinVerify/Spec/FrontendIntent.lean"),
        "spec_lock": snapshot(root / "merlin/lean/SPEC_LOCK"),
        "f32_approval": snapshot(frontend / "f32-policy-approval.json"),
        "native": native,
        "native_samples": sum(x["samples"] for x in native),
        "pinned_torch": {
            "torch": oracle["torch"],
            "python": call([str(pinned), "--version"]).strip(),
            "numpy": oracle["numpy"],
            "samples": len(oracle["results"]),
            "evidence": snapshot(work / "pinned-torch-results.json"),
        },
        "lean_version": call([str(lake.parent / "lean"), "--version"]).strip(),
        "compiler_versions": json.loads(
            call(
                [
                    str(compiler),
                    "-c",
                    "import sys,json,importlib.metadata as m; "
                    "print(json.dumps({'python':sys.version,'torch_mlir':m.version('torch-mlir'),"
                    "'xdsl':m.version('xdsl')}))",
                ]
            )
        ),
        "skips": [],
        "remaining": [
            "runtime semantic review and lock/claim admission",
            "full-model establishment of the cut contract and other, non-exhaustively inventoried obligations",
            "downstream suffix proofs under the same interpretation before candidate chain activation",
        ],
    }
    (output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
