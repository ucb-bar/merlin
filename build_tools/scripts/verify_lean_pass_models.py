#!/usr/bin/env python3
"""Compile captured models with the actual Lean gates and retain their bound exports.

Run from an installed Merlin environment or with PYTHONPATH=src. Each capture is an MLIR
file or a directory containing model.mlir. All output is new: existing evidence is never overwritten. A comparison build
checks that enabling validation preserves the returned LLVM IR bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from merlin.common.artifacts import new_product
from merlin.llvmlower.lower import lower_model
from merlin.verify.pass_validation import CheckerTools, PassValidation
from merlin.verify.speculative_gate import SpeculativeGate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--sdk-bin", type=Path, required=True)
    parser.add_argument("--compiler-python", type=Path, required=True)
    parser.add_argument("--routes", nargs="+", choices=["xdsl", "textual"], default=["xdsl", "textual"])
    args = parser.parse_args()
    # Resolving a virtualenv's interpreter symlink selects the system environment.
    tools = CheckerTools(args.sdk_bin.resolve(), args.compiler_python.absolute())
    os.environ["MERLIN_COMPILER_PYTHON"] = str(tools.compiler_python)
    # Select validation explicitly on the checked build only.
    os.environ.pop("MERLIN_VALIDATE_PASSES", None)
    captures = [capture.resolve() for capture in args.capture]
    for capture in captures:
        source = capture if capture.is_file() else capture / "model.mlir"
        if not source.is_file():
            parser.error(f"capture is neither an MLIR file nor a directory with model.mlir: {capture}")
    product = new_product("verification", version=1, notes="CSE and quantization compiler gate model exports")
    output = product.path.resolve()
    rows = []
    for index, capture in enumerate(captures):
        source = capture if capture.is_file() else capture / "model.mlir"
        label = capture.parent.name if capture.is_file() else capture.name
        source_quantizers = source.read_text().count('"quant_ext.quantize_per_tensor"')
        for route in dict.fromkeys(args.routes):
            work = output / f"{index:02d}-{label}" / route
            row = {
                "capture": str(capture),
                "source": str(source),
                "source_sha256": sha256(source),
                "source_quantizers": source_quantizers,
                "route": route,
            }
            started = time.monotonic()
            try:
                before = lower_model(source.read_text(), work / "ungated", targets=(), textual=route == "textual")
                gate = SpeculativeGate(work / "pass-validation", interrupt=True, workers=1)
                try:
                    checked = lower_model(
                        source.read_text(),
                        work / "gated",
                        targets=(),
                        textual=route == "textual",
                        pass_validation=PassValidation(gate, tools),
                    )
                finally:
                    gate.close()
                row.update(
                    verdicts=gate.verdicts,
                    llvm_sha256=sha256(checked.ll_path),
                    comparison_sha256=sha256(before.ll_path),
                    llvm_identical=checked.ll_path.read_bytes() == before.ll_path.read_bytes(),
                )
                row["cse_checked"] = any(v["label"] == "cse" for v in gate.verdicts)
                row["source_quantizer_checked"] = any(
                    v["label"] in {"lower_quant_ext", "preprocess_text_textual"}
                    and '"quant_ext.quantize_per_tensor"'
                    in (work / "pass-validation" / v["job"] / "before.mlir").read_text()
                    for v in gate.verdicts
                )
                row["ok"] = (
                    row["llvm_identical"]
                    and row["cse_checked"]
                    and (not source_quantizers or row["source_quantizer_checked"])
                    and all(v["verdict"] == "accepted" for v in gate.verdicts)
                )
            except Exception as error:
                row.update(ok=False, error=f"{type(error).__name__}: {error}")
            row["seconds"] = round(time.monotonic() - started, 3)
            rows.append(row)
            (output / "models.json").write_text(
                json.dumps(
                    {
                        "schema": "merlin.lean_pass_models.v1",
                        "rows": rows,
                        "scope": "recorded tensor pass boundaries; downstream LLVM lowering remains trusted",
                    },
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps({k: v for k, v in row.items() if k != "verdicts"}), flush=True)
    print(f"Evidence and exact before/after exports: {output}", flush=True)
    return 0 if all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
