#!/usr/bin/env python3
"""Record versioned reference intake and the fixed local-capsule denominator.

Run from the separate frontend checkout. This records provenance and inventories; it does
not approve a contract, infer a missing capture version or replace any Lean checker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from merlin.common.paths import merlin_dir, repo_root

REFERENCE_COMMIT = "0cc289de52a10bed256f32c90cfbb54a21b3c81f"
REFERENCE_FILES = (
    "examples/gemmini/numerics/README.md",
    "examples/gemmini/numerics/QUANTIZATION.md",
    "examples/gemmini/protocol/README.md",
    "examples/gemmini/protocol/experiment.yaml",
    "examples/gemmini/protocol/run-manifest.template.yaml",
    "examples/gemmini/protocol/recipes/resnet50.json",
    "examples/gemmini/protocol/recipes/tiny_llama.json",
    "examples/gemmini/protocol/recipes/smolvla.json",
    "examples/gemmini/artifacts/README.md",
)


def ref(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=repo_root(), check=True, capture_output=True).stdout


def intake(destination: Path) -> dict:
    records = []
    recipes = {}
    for name in REFERENCE_FILES:
        snapshot = destination / "references" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(git("show", f"{REFERENCE_COMMIT}:{name}"))
        records.append(
            {
                "git_path": name,
                "git_blob": git("rev-parse", f"{REFERENCE_COMMIT}:{name}").decode().strip(),
                **ref(snapshot),
            }
        )
        if "/recipes/" in name:
            recipe = json.loads(snapshot.read_text())
            payload = {k: v for k, v in recipe.items() if k not in ("recipe_sha256", "why")}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            recipes[snapshot.stem] = {
                "file_sha256": ref(snapshot)["sha256"],
                "recipe_sha256": digest,
                "activation": recipe["activation"],
                "weight": recipe["weight"],
            }
    result = {
        "schema": "merlin.local-passes.reference-intake.v1",
        "git_commit": REFERENCE_COMMIT,
        "branch": "feat/el4-experiment",
        "files": records,
        "selected_recipes": recipes,
        "historical_guide_versions": {
            "torch": "2.10.0",
            "torchao": {"smolvla": "0.15.0", "resnet50": "0.16.0", "tiny_llama": "0.16.0"},
            "model2mlir": "905d2c918f83",
            "recipe_sha256": "6497b3dde13e91c1787ec6bce07674866861da6389bccc7f3388cda7e386df51",
        },
        "synthetic_dynamic_recipe": {
            "mapping": "SYMMETRIC",
            "granularity": "per_token",
            "quant_min": -127,
            "quant_max": 127,
            "eps": "f32(1e-5)",
            "zero_point": 0,
            "input": "f32",
            "output": "i8",
        },
        "selected_capture_receipt": None,
        "actual_capture_version_recipe_admitted": False,
        "reference_scope": (
            "Selected static grouped-16 recipes and historical captures are distinct "
            "from the synthetic dynamic affine recipe."
        ),
        "contract_review": "pending; implementation-aware proposal requires independent review",
    }
    (destination / "reference-intake.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def census(destination: Path) -> dict:
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import fix_bool_sitofp

    paths = sorted((merlin_dir() / "contract/capsules").rglob("capsule.interface.mlir"))
    records = []
    signatures = Counter()
    for path in paths:
        source = path.read_text()
        if not any(name in source for name in ("sitofp", "quant_ext.", "torchao_")):
            continue
        module = parse_mlir_text(source)
        calls = Counter()
        sites = []
        for op in module.walk():
            name = getattr(getattr(op, "op_name", None), "data", op.name)
            if op.name == "func.call":
                callee = str(op.properties.get("callee", op.attributes.get("callee", "")))
                if "torchao_" in callee:
                    calls[callee] += 1
            if name.startswith("quant_ext."):
                attrs = {
                    k: str(v)
                    for k, v in {**op.attributes, **op.properties}.items()
                    if not k.startswith("prov.") and not k.startswith("op_name")
                }
                signature = {
                    "name": name,
                    "inputs": [str(v.type) for v in op.operands],
                    "results": [str(v.type) for v in op.results],
                    "attrs": attrs,
                }
                sites.append(signature)
                signatures[json.dumps(signature, sort_keys=True)] += 1
        count = fix_bool_sitofp(module)
        records.append(
            {
                "artifact": ref(path),
                "bool_to_float_sites": count,
                "opaque_torchao_calls": dict(calls),
                "quant_ext_sites": sites,
            }
        )
    result = {
        "schema": "merlin.local-passes.capsule-census.v1",
        "declared_capsule_files": len(paths),
        "relevant_capsules": records,
        "quant_ext_signatures": [{**json.loads(s), "count": n} for s, n in sorted(signatures.items())],
        "actual_capture_versions": "unbound; capsule names and recipe labels do not identify TorchAO versions",
        "qualification_scope": "Inventory only. Neither source typing nor a Python count is a Lean proof fact.",
    }
    (destination / "capsule-census.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=repo_root() / "out/artifacts/verification/merlin-passes/local-passes"
    )
    parser.add_argument(
        "--receipts", action="store_true", help="build and check actual M4 and synthetic affine boundaries"
    )
    parser.add_argument("--compiler-python", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    intake(args.out)
    report = census(args.out)
    if args.receipts:
        if args.compiler_python is None:
            parser.error("--receipts requires --compiler-python")
        make_receipts(args.out, args.compiler_python)
    print(
        json.dumps(
            {
                "reference_commit": REFERENCE_COMMIT,
                "capsule_files": report["declared_capsule_files"],
                "signature_groups": len(report["quant_ext_signatures"]),
                "approved_full_verification": False,
            }
        )
    )


def make_receipts(destination: Path, compiler_python: Path) -> None:
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import fix_bool_sitofp
    from merlin.llvmlower.torchao_affine import lower_torchao_affine_quant
    from merlin.verify.lean import local_passes
    from merlin.xdsl_dialects._common import text as to_text

    capture = merlin_dir() / "contract/capsules/model/M4_smolvla_denoise_gemmini/capsule.interface.mlir"
    synthetic = """module attributes {prov.quantization = "int8_dyn_act_int8_weight"} {
      func.func private @torchao_choose_qparams_affine_default(tensor<2x4xf32>) -> tensor<2xf32>
      func.func private @torchao_quantize_affine_default(tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xi8>
      func.func @subject(%x: tensor<2x4xf32>) -> (tensor<2xf32>, tensor<2x4xi8>) {
        %s = func.call @torchao_choose_qparams_affine_default(%x) : (tensor<2x4xf32>) -> tensor<2xf32>
        %q = func.call @torchao_quantize_affine_default(%x, %s) : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2x4xi8>
        return %s, %q : tensor<2xf32>, tensor<2x4xi8>
      }}"""
    results = []
    for name, source, mode, transform, expected in (
        ("smolvla-denoise-bool", capture.read_text(), "bool-to-float", fix_bool_sitofp, 2),
        ("synthetic-affine", synthetic, "torchao-affine", lower_torchao_affine_quant, 2),
    ):
        work = destination / name
        work.mkdir(parents=True, exist_ok=True)
        module = parse_mlir_text(source)
        before, after = (work / f"{side}.mlir" for side in ("before", "after"))
        before.write_text(to_text(module, generic=True))
        count = transform(module)
        if count != expected:
            raise RuntimeError(f"changed production denominator for {name}: {count}")
        module.verify()
        after.write_text(to_text(module, generic=True))
        receipt = local_passes.verify(
            mode,
            before,
            after,
            work,
            compiler_python=compiler_python,
            suffix=(after,),
            references=(destination / "reference-intake.json",),
        )
        if receipt["status"] != local_passes.STATUSES[mode]:
            raise RuntimeError(f"{name}: {receipt.get('reason', receipt.get('answer'))}")
        local_passes.replay(receipt, compiler_python=compiler_python)
        results.append(
            {
                "name": name,
                "kind": "public capsule" if mode == "bool-to-float" else "synthetic; not an FX capture",
                "sites": count,
                "status": receipt["status"],
                "receipt": ref(Path(receipt["work"]) / "receipt.json"),
            }
        )
    (destination / "receipts.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
