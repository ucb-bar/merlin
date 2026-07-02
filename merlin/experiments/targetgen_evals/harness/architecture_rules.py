"""Architecture rules checker.

Each rule returns a dict:
  {"rule_id": str, "name": str, "passed": bool, "severity": str, "message": str}
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _rule(rule_id: str, name: str, passed: bool, severity: str, message: str) -> dict:
    return {"rule_id": rule_id, "name": name, "passed": passed,
            "severity": severity, "message": message}


def check_r1(run_dir: Path, manifest: dict) -> dict:
    target = manifest["target"]
    expected = run_dir / "generated" / f"{target}-mlir"
    if expected.exists():
        return _rule("R1", "generated-repo-naming", True, "info",
                     f"generated/{target}-mlir/ exists")
    return _rule("R1", "generated-repo-naming", False, "error",
                 f"generated/{target}-mlir/ does not exist; "
                 f"all generated output must live under this directory")


def check_r2(run_dir: Path, manifest: dict) -> dict:
    target = manifest["target"]
    xdsl_dir = run_dir / "generated" / f"{target}-mlir" / "xdsl"
    if xdsl_dir.exists() and any(xdsl_dir.iterdir()):
        return _rule("R2", "xdsl-before-tablegen", True, "info",
                     f"xdsl/ directory exists and is non-empty")
    return _rule("R2", "xdsl-before-tablegen", False, "warning",
                 "xdsl/ directory is absent or empty; "
                 "xDSL artifacts must exist before TableGen/C++ promotion")


def check_r3(run_dir: Path, manifest: dict) -> dict:
    if manifest.get("promotion_flag", False):
        return _rule("R3", "no-premature-tablegen", True, "info",
                     "promotion_flag is set; TableGen/C++ generation is permitted")
    target = manifest["target"]
    gen_dir = run_dir / "generated" / f"{target}-mlir"
    violations = []
    if gen_dir.exists():
        for ext in (".td", ".cpp"):
            violations.extend(gen_dir.rglob(f"*{ext}"))
    if violations:
        files = ", ".join(str(v.relative_to(run_dir)) for v in violations[:5])
        return _rule("R3", "no-premature-tablegen", False, "error",
                     f"TableGen/C++ files found before promotion: {files}")
    return _rule("R3", "no-premature-tablegen", True, "info",
                 "No TableGen/C++ files found (correct for xDSL-first workflow)")


def check_r4(run_dir: Path, manifest: dict, repo_root: Path) -> dict:
    git_hash = manifest.get("git_hash_at_init", "")
    if not git_hash or git_hash == "unknown":
        return _rule("R4", "merlin-core-immutable", False, "warning",
                     "git_hash_at_init is unknown; cannot verify Merlin core was not modified")
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", git_hash, "--", "merlin/"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        changed = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if changed:
            return _rule("R4", "merlin-core-immutable", False, "error",
                         f"Merlin core files modified since init: {', '.join(changed[:5])}")
        return _rule("R4", "merlin-core-immutable", True, "info",
                     "No Merlin core files modified since run init")
    except subprocess.CalledProcessError as e:
        return _rule("R4", "merlin-core-immutable", False, "warning",
                     f"Could not run git diff: {e.stderr.strip()}")


def _check_dialect_plan_rules(run_dir: Path, manifest: dict) -> list[dict]:
    plan_path = run_dir / "contracts" / "dialect_plan.yaml"
    if not plan_path.exists():
        info = "dialect_plan.yaml not present; skipping op-level checks (OK for empty run)"
        return [
            _rule("R5", "op-evidence", True, "info", info),
            _rule("R6", "op-verifier-coverage", True, "info", info),
            _rule("R7", "op-lowering-exit", True, "info", info),
            _rule("R8", "no-scheduling-in-semantics", True, "info", info),
            _rule("R9", "no-runtime-in-types", True, "info", info),
            _rule("R10", "unsupported-fails-early", True, "info", info),
        ]

    import yaml
    try:
        with open(plan_path) as f:
            plan = yaml.safe_load(f) or {}
    except Exception as e:
        msg = f"Could not parse dialect_plan.yaml: {e}"
        return [_rule(f"R{i}", name, False, "error", msg)
                for i, name in enumerate(
                    ["op-evidence", "op-verifier-coverage", "op-lowering-exit",
                     "no-scheduling-in-semantics", "no-runtime-in-types",
                     "unsupported-fails-early"], start=5)]

    ops = plan.get("ops", [])
    results = []

    # R5: every op has evidence
    missing_evidence = [op.get("name", "?") for op in ops if not op.get("evidence")]
    results.append(_rule("R5", "op-evidence", not missing_evidence,
                         "error" if missing_evidence else "info",
                         f"Ops missing evidence: {missing_evidence}" if missing_evidence
                         else "All ops have evidence"))

    # R6: every op has verifier coverage
    missing_verifier = [op.get("name", "?") for op in ops if not op.get("verifier")]
    results.append(_rule("R6", "op-verifier-coverage", not missing_verifier,
                         "error" if missing_verifier else "info",
                         f"Ops missing verifier: {missing_verifier}" if missing_verifier
                         else "All ops have verifier coverage"))

    # R7: every op has at least one lowering exit
    missing_lowering = [op.get("name", "?") for op in ops if not op.get("lowering_exits")]
    results.append(_rule("R7", "op-lowering-exit", not missing_lowering,
                         "error" if missing_lowering else "info",
                         f"Ops missing lowering exit: {missing_lowering}" if missing_lowering
                         else "All ops have a lowering exit"))

    # R8: scheduling policy not in op semantics (check for 'schedule' key in op defs)
    scheduling_violations = [op.get("name", "?") for op in ops if op.get("scheduling_policy")]
    results.append(_rule("R8", "no-scheduling-in-semantics", not scheduling_violations,
                         "error" if scheduling_violations else "info",
                         f"Ops with scheduling_policy in semantics: {scheduling_violations}"
                         if scheduling_violations
                         else "No scheduling policy embedded in op semantics"))

    # R9: runtime launch details not in types
    type_violations = [op.get("name", "?") for op in ops if op.get("runtime_launch_in_type")]
    results.append(_rule("R9", "no-runtime-in-types", not type_violations,
                         "error" if type_violations else "info",
                         f"Ops with runtime launch encoded in types: {type_violations}"
                         if type_violations
                         else "No runtime launch details in pure types"))

    # R10: unsupported cases have explicit handling
    missing_unsupported = [op.get("name", "?") for op in ops
                           if op.get("has_unsupported_cases") and not op.get("unsupported_handling")]
    results.append(_rule("R10", "unsupported-fails-early", not missing_unsupported,
                         "error" if missing_unsupported else "info",
                         f"Ops with unsupported cases but no explicit handling: {missing_unsupported}"
                         if missing_unsupported
                         else "All unsupported cases have explicit handling"))
    return results


def check_all(run_dir: Path, manifest: dict, repo_root: Path) -> list[dict]:
    results = [
        check_r1(run_dir, manifest),
        check_r2(run_dir, manifest),
        check_r3(run_dir, manifest),
        check_r4(run_dir, manifest, repo_root),
    ]
    results.extend(_check_dialect_plan_rules(run_dir, manifest))
    return results
