#!/usr/bin/env python3
"""Guardrails for the Lean checkers (docs/design/merlin_lean_verification.md §1.2, §8.6).

1. Audit: build ``merlin/lean`` and run ``merlin-lean-audit`` over every module. It fails on any
   theorem depending on an axiom beyond {propext, Classical.choice, Quot.sound} (so ``sorry``,
   native_decide/bv_decide axioms), any ``axiom`` outside ``MerlinVerify.Assumptions``, and any
   opaque/partial/unsafe/implemented_by/extern constant (the compiled checker must be the proved one).
2. Lock: ``merlin/lean/SPEC_LOCK`` pins the SHA-256 of every frozen interface file and every file
   under ``Spec/``, ``Claims/`` and ``Assumptions/``. Any change fails until the lock is regenerated
   with ``--update-lock``, which is the reviewed action (it prints what changed).

Usage: check_lean_guard.py [--lock-only] [--update-lock]
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEAN = ROOT / "merlin" / "lean"
LOCK = LEAN / "SPEC_LOCK"
FROZEN = [
    "MerlinVerify/Core/AST.lean",
    "MerlinVerify/Sem/Outcome.lean",
    "MerlinVerify/Sem/Float.lean",
    "MerlinVerify/Sem/Value.lean",
    "MerlinVerify/Sem/Refine.lean",
    "MerlinVerify/Sem/Program.lean",
    "MerlinVerify/Sem/Ops.lean",
    "MerlinVerify/Check/Types.lean",
    "lakefile.toml",
    "lean-toolchain",
    "lake-manifest.json",
]
SCHEMAS = ["lean_ir.schema.json", "lean_cert.schema.json", "lean_receipt.schema.json", "lean_chain.schema.json"]
LOCKED_DIRS = ["MerlinVerify/Spec", "MerlinVerify/Claims", "MerlinVerify/Assumptions"]


def locked_files() -> dict[str, str]:
    files: dict[str, Path] = {f: LEAN / f for f in FROZEN}
    for s in SCHEMAS:
        files[f"schemas/{s}"] = ROOT / "merlin" / "contract" / "schemas" / s
    for d in LOCKED_DIRS:
        for p in sorted((LEAN / d).rglob("*.lean")) if (LEAN / d).exists() else []:
            files[str(p.relative_to(LEAN))] = p
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sorted(files.items()) if p.exists()}


def check_lock(update: bool) -> bool:
    current = locked_files()
    if update:
        LOCK.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"wrote {LOCK.relative_to(ROOT)} ({len(current)} files)")
        return True
    if not LOCK.exists():
        print("[FAIL] merlin/lean/SPEC_LOCK missing; run with --update-lock after review")
        return False
    pinned = json.loads(LOCK.read_text())
    changed = sorted(k for k in set(current) | set(pinned) if current.get(k) != pinned.get(k))
    for k in changed:
        state = "added" if k not in pinned else "removed" if k not in current else "changed"
        print(f"[FAIL] locked file {state}: {k} (review, then --update-lock)")
    return not changed


def run_audit() -> bool:
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.elan' / 'bin'}:{env.get('PATH', '')}"
    build = subprocess.run(["lake", "build"], cwd=LEAN, env=env, capture_output=True, text=True)
    if build.returncode != 0:
        print("[FAIL] lake build failed:\n" + build.stdout[-2000:] + build.stderr[-2000:])
        return False
    mods = sorted(str(p.relative_to(LEAN).with_suffix("")).replace("/", ".") for p in (LEAN / "MerlinVerify").rglob("*.lean"))
    audit = subprocess.run(["lake", "exe", "merlin-lean-audit", *mods], cwd=LEAN, env=env, capture_output=True, text=True)
    if audit.returncode not in (0, 1):
        print("[FAIL] merlin-lean-audit error:\n" + audit.stderr[-2000:])
        return False
    report = json.loads(audit.stdout.strip().splitlines()[-1])
    for v in report["violations"]:
        print(f"[FAIL] {v}")
    print(f"audit: {report['theorems_audited']} theorems, {len(report['violations'])} violations")
    return audit.returncode == 0 and not report["violations"]


def main(argv: list[str]) -> int:
    update = "--update-lock" in argv
    ok = check_lock(update)
    if "--lock-only" not in argv and not update:
        ok = run_audit() and ok
    print("lean guard: OK" if ok else "lean guard: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
