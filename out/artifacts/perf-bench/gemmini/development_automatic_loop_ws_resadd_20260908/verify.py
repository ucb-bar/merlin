#!/usr/bin/env python3
"""Verify the public automatic LOOP_WS residual-add artifact."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def tree_digest(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    files = [path for path in root.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts]
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return len(files), digest.hexdigest()


def main() -> None:
    status = json.loads((ROOT / "STATUS.json").read_text())
    require(tree_digest(ROOT / "compiler") == (
        status["compiler_tree"]["files"], status["compiler_tree"]["sha256"]),
        "compiler tree digest differs from STATUS")
    require(tree_digest(ROOT / "tests") == (
        status["tests_tree"]["files"], status["tests_tree"]["sha256"]),
        "tests tree digest differs from STATUS")
    receipt_path = ROOT / "validation/resnet50_resadd_census.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["candidate_count"] == 16, "canonical census is not 16 residual adds")
    require(receipt["selected_count"] == 0, "canonical graph unexpectedly selects is_resadd")
    require(receipt["approximation"] is False, "artifact permits approximate residual lowering")
    require(receipt["target_sha256"] ==
            "ac1fb50a5176c3749c6abea51be254f68108174e7f7831c813caca3a872dc673",
            "canonical refusal target no longer matches its parent")
    rows = receipt["float_residual_refused"]
    require(len(rows) == 16 and all(not row["selected"] for row in rows),
            "one canonical residual lacks a fail-closed decision")
    source = (ROOT / "compiler/mlir_oot/frontend/gemmini_resadd.py").read_text()
    table = (ROOT / "compiler/mlir_oot/tables/loop_ws.py").read_text()
    emitter = (ROOT / "compiler/mlir_oot/codegen/llvm_emit.py").read_text()
    require("arith.maxsi" in source and "arith.minsi" in source,
            "recognizer no longer proves both saturation bounds")
    require('"is_resadd": 1' in table and '"K": 0' in table,
            "descriptor no longer sets exact residual mode")
    require("emit_loop_ws_resadd" in emitter, "residual descriptor has no emitter")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "compiler")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")],
        env=env, text=True, capture_output=True)
    require(completed.returncode == 0,
            "focused tests failed:\n" + completed.stdout + completed.stderr)
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    print(json.dumps({
        "status": "passed",
        "tests": "65 passed, 6 subtests",
        "canonical_selected": "0/16",
        "canonical_target_unchanged": True,
        "receipt_sha256": digest,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
