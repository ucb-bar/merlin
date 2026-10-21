#!/usr/bin/env python3
"""Generate and mechanically check an unapplied frontend lock-admission proposal.

The real SPEC_LOCK and guard stay untouched. A passing candidate check means only
that the proposed hashes/statements are internally consistent; it is not review.
Run from this checkout after the Lean build, with PYTHONPATH=src.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root

PINS = {
    "MerlinVerify.Rules.FrontendBoolSimulation.check_sound": "MerlinVerify.Rules.FrontendBoolSimulation",
    "MerlinVerify.Rules.FrontendBoolSimulation.check_sound_ieee": "MerlinVerify.Rules.FrontendBoolSimulation",
    "MerlinVerify.Rules.FrontendBoolQualified.support_sound": "MerlinVerify.Rules.FrontendBoolQualified",
    "MerlinVerify.Rules.FrontendBoolQualified.defined_sound": "MerlinVerify.Rules.FrontendBoolQualified",
    "MerlinVerify.Rules.FrontendBoolChain.chain_after_source_policy": "MerlinVerify.Rules.FrontendBoolChain",
    "MerlinVerify.Rules.FrontendBoolChain.chain_after_source_policy_strict": "MerlinVerify.Rules.FrontendBoolChain",
    "MerlinVerify.Rules.FrontendF32Comparison.model_correct": "MerlinVerify.Rules.FrontendF32Comparison",
    "MerlinVerify.Rules.FrontendBoolDefined.compareRule_sound": "MerlinVerify.Rules.FrontendBoolDefined",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch(old: str, new: str, path: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}", n=1))


def main() -> None:
    root = repo_root()
    output = artifacts_dir() / "verification/merlin-passes/frontend/admission-candidate"
    output.mkdir(parents=True, exist_ok=True)
    guard_path = root / "build_tools/scripts/check_lean_guard.py"
    spec = importlib.util.spec_from_file_location("frontend_admission_guard", guard_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the authoritative guard")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    original_lock = guard.LOCK.read_bytes()
    original_guard = guard_path.read_bytes()
    original_pins = dict(guard.PINNED)
    guard.PINNED |= PINS
    statements = guard.pinned_statements()
    if statements is None:
        raise RuntimeError("statement extraction failed; build the named modules first")
    statement_run = subprocess.run(
        ["lake", "exe", "merlin-lean-audit", "--statements", ",".join(PINS), *sorted(set(PINS.values()))],
        cwd=guard.LEAN,
        env=guard._lake_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    statement_text = json.loads(statement_run.stdout.strip().splitlines()[-1])["statements"]
    (output / "statements.json").write_text(json.dumps(statement_text, indent=2) + "\n")
    candidate = guard.locked_files() | statements
    old = json.loads(original_lock)
    differences = {
        k: {"old": old.get(k), "candidate": candidate.get(k)}
        for k in sorted(old.keys() | candidate.keys())
        if old.get(k) != candidate.get(k)
    }
    expected = {
        "MerlinVerify/Spec/FrontendIntent.lean",
        "schemas/lean_receipt.schema.json",
        "schemas/lean_chain.schema.json",
        *(f"statement:{name}" for name in PINS if name not in original_pins),
    }
    if differences.keys() != expected:
        raise RuntimeError(f"unexpected lock drift: {sorted(differences.keys() ^ expected)}")
    candidate_text = json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    lock_copy = output / "SPEC_LOCK.candidate"
    lock_copy.write_text(candidate_text)
    guard_text = original_guard.decode()
    insert = "".join(f'    "{name}": "{module}",\n' for name, module in PINS.items() if name not in original_pins)
    marker = "}\n\n\ndef locked_files()"
    if guard_text.count(marker) != 1:
        raise RuntimeError("guard layout changed; review the insertion point")
    proposed_guard = guard_text.replace(marker, insert + marker)
    (output / "admission.patch").write_text(
        patch(guard_text, proposed_guard, "build_tools/scripts/check_lean_guard.py")
        + patch(original_lock.decode(), candidate_text, "merlin/lean/SPEC_LOCK")
    )
    guard.LOCK = lock_copy
    consistent = guard.check_lock(False, statements)
    if not consistent:
        raise RuntimeError("candidate lock failed the authoritative comparison")
    if (root / "merlin/lean/SPEC_LOCK").read_bytes() != original_lock or guard_path.read_bytes() != original_guard:
        raise RuntimeError("real lock or guard changed during proposal generation")
    record = {
        "status": "proposal; not applied or admitted",
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "statements_sha256": sha((output / "statements.json").read_bytes()),
        "base_lock_sha256": sha(original_lock),
        "base_guard_sha256": sha(original_guard),
        "candidate_lock_sha256": sha(candidate_text.encode()),
        "patch_sha256": sha((output / "admission.patch").read_bytes()),
        "candidate_comparison": "passed against authoritative check_lock",
        "human_review": "pending integration-owner admission; scoped f32 policy approval is already recorded",
        "new_pins": PINS,
        "differences": differences,
        "unchanged_real_inputs": True,
    }
    (output / "review.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"output": str(output), "candidate_comparison": consistent, "differences": len(differences)}))


if __name__ == "__main__":
    main()
