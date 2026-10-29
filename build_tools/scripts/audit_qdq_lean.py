"""Audit the pinned supported Q/DQ claims without rebuilding all Mathlib modules."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root

ROOT = repo_root()
LEAN = ROOT / "merlin/lean"
GUARD = ROOT / "build_tools/scripts/check_lean_guard.py"
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=artifacts_dir() / "verification/qdq-claims.json")
    args = parser.parse_args()
    sdk = Path(os.environ["MERLIN_LEAN_SDK_BIN"]).absolute()
    if not (sdk / "lake").is_file():
        raise RuntimeError("selected Lean SDK is missing")
    pinned = runpy.run_path(str(GUARD))["PINNED"]
    selected = {
        decl: module
        for decl, module in pinned.items()
        if "W8A8" not in decl
        and any(marker in decl for marker in ("Quantize", "Dequant", "chain_typed", "TypedModule", "supportCheck"))
    }
    env = {**os.environ, "PATH": f"{sdk}{os.pathsep}{os.environ.get('PATH', '')}"}
    results = {}
    for decl, module in selected.items():
        proc = subprocess.run(
            [str(sdk / "lake"), "exe", "merlin-lean-audit", "--axioms", decl, module],
            cwd=LEAN,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode:
            raise RuntimeError(f"audit failed for {decl}: {proc.stdout[-300:]} {proc.stderr[-300:]}")
        report = json.loads(proc.stdout.strip().splitlines()[-1])
        if report["decl"] != decl or not set(report["axioms"]) <= ALLOWED:
            raise AssertionError(f"unapproved axiom or changed declaration: {report}")
        results[decl] = {"module": module, "axioms": report["axioms"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"{len(results)} pinned Q/DQ and typed-chain claims audited; only approved axioms")


if __name__ == "__main__":
    main()
