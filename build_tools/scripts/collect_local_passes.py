#!/usr/bin/env python3
"""Collect existing pass-local qualification artifacts without promoting their proof scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from merlin.common.paths import repo_root


def ref(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def tests(path: Path) -> dict:
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    return {
        "artifact": ref(path),
        **{k: sum(int(s.get(k, "0")) for s in suites) for k in ("tests", "failures", "errors", "skipped")},
    }


def collect(destination: Path) -> dict:
    root = repo_root()
    retained = destination / "retained-final-tests"
    qualification = tests(destination / "retained-final-tests.xml")
    regression = tests(destination / "regression-tests.xml")
    if any(t[k] for t in (qualification, regression) for k in ("failures", "errors", "skipped")):
        raise ValueError("qualification requires passing tests without skips")
    native, oracles, checkers, exceptions = [], [], [], []
    for path in sorted(retained.rglob("native-results.json")):
        result = json.loads(path.read_text())
        request = path.with_name("native-request.json")
        source = Path(json.loads(request.read_text())["source"])
        native.append(
            {
                "case": str(path.relative_to(retained)),
                "result": ref(path),
                "request": ref(request),
                "source": ref(source),
                "mlir_revision": result["mlir_revision"],
                "samples": len(result["results"]),
                "observed_words": sum(map(len, result["results"])),
            }
        )
    for path in sorted(retained.rglob("*.json")):
        if path.name.startswith("torch") or path.name.startswith("nonfinite-reference"):
            result = json.loads(path.read_text())
            if "torch" in result:
                oracles.append(
                    {
                        "artifact": ref(path),
                        "torch": result["torch"],
                        "torchao": result.get("torchao"),
                        "package_sources": result.get("sources"),
                        "python": result.get("python"),
                        "source_input": ref(path.with_name("oracle-request.json"))
                        if path.with_name("oracle-request.json").is_file()
                        else None,
                    }
                )
        if path.name == "checker.json":
            result = json.loads(path.read_text())
            record = {"artifact": ref(path), "accepted": result["accepted"], "status": result["status"]}
            sources = [path.parent / side / "subject.mlir" for side in ("before", "after")]
            record["sources"] = [ref(p) for p in sources if p.is_file()]
            checkers.append(record)
    for path in sorted(retained.rglob("empty-reference-*.stderr")):
        exceptions.append(
            {"artifact": ref(path), "kind": "pinned TorchAO empty-zero-point TypeError; not numerical agreement"}
        )
    protected = [
        "src/merlin/llvmlower/passes_xdsl.py",
        "src/merlin/llvmlower/torchao_affine.py",
        "merlin/lean/Main.lean",
        "merlin/lean/lakefile.toml",
        "merlin/lean/SPEC_LOCK",
        "merlin/lean/MerlinVerify/Spec/FrontendIntent.lean",
        "merlin/lean/MerlinVerify/Sem/Float.lean",
    ]
    protected_records = []
    for name in protected:
        original = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=root, check=True, capture_output=True).stdout
        current = ref(root / name)
        expected = hashlib.sha256(original).hexdigest()
        if current["sha256"] != expected:
            raise ValueError(f"protected file changed: {name}")
        protected_records.append({"relative_path": name, "sha256": expected, "unchanged_from_checkout_baseline": True})
    changed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "merlin/lean/MerlinVerify/Sem",
            "merlin/lean/MerlinVerify/Spec",
            "src/merlin/llvmlower",
            "merlin/lean/Main.lean",
            "merlin/lean/lakefile.toml",
            "merlin/lean/SPEC_LOCK",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if changed.stdout:
        raise ValueError("protected source changes: " + changed.stdout)
    audit = json.loads((destination / "audit-all.json").read_text())
    if audit["violations"]:
        raise ValueError("Lean audit violations")
    census = json.loads((destination / "capsule-census.json").read_text())
    return {
        "schema": "merlin.local-passes.qualification.v1",
        "approved_full_verification": False,
        "bool_to_float": "proved complete-program f32 frontend-policy correspondence; admission pending",
        "torchao_affine": "materialized-program/scalar component proved; captured-call reference bridge NOT proved",
        "tests": qualification,
        "regressions": regression,
        "lean_audit": audit,
        "source_capsules_in_denominator": census["declared_capsule_files"],
        "relevant_capsules": len(census["relevant_capsules"]),
        "quant_ext_sites": sum(s["count"] for s in census["quant_ext_signatures"]),
        "native": native,
        "oracles": oracles,
        "checker_cases": checkers,
        "reference_refusals": exceptions,
        "native_evidence_is_not_a_formal_proof": True,
        "pytorch_cast_domain": ref(destination / "pytorch-cast-domain.json")
        if (destination / "pytorch-cast-domain.json").is_file()
        else None,
        "protected": protected_records,
        "references": ref(destination / "reference-intake.json"),
        "receipts": ref(destination / "receipts.json"),
        "known_guard_failures": {
            "lean": ref(destination / "guard.log"),
            "structure": ref(destination / "structure.log"),
        },
        "test_sources": [
            ref(root / "merlin/tests/ir" / name)
            for name in ("test_lean_bool_to_float.py", "test_lean_torchao_affine.py")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=repo_root() / "out/artifacts/verification/merlin-passes/local-passes"
    )
    args = parser.parse_args()
    result = collect(args.out)
    (args.out / "qualification.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "tests": result["tests"]["tests"],
                "regressions": result["regressions"]["tests"],
                "audit": result["lean_audit"],
                "approved_full_verification": False,
            }
        )
    )


if __name__ == "__main__":
    main()
