#!/usr/bin/env python3
"""Guardrails for the Lean checkers (docs/design/merlin_lean_verification.md §1.2, §8.6).

1. Audit: build ``merlin/lean`` and run ``merlin-lean-audit`` over every module. It fails on any
   theorem depending on an axiom beyond {propext, Classical.choice, Quot.sound} (so ``sorry``,
   native_decide/bv_decide axioms), any ``axiom`` outside ``MerlinVerify.Assumptions``, and any
   opaque/partial/unsafe/implemented_by/extern constant (the compiled checker must be the proved one).
2. Lock: ``merlin/lean/SPEC_LOCK`` pins the SHA-256 of every frozen interface file, every semantics
   file under ``Sem/`` (the meaning of every claim), the typed-input domain (``Check/Typing.lean``)
   and every file under ``Spec/``, ``Claims/`` and ``Assumptions/``; and the elaborated statement of
   every claim theorem (``PINNED``), so a theorem that still compiles but says something else fails.
   Any change fails until the lock is regenerated with ``--update-lock``, which prints what changed
   for review. ``--lock-only`` checks the files only (statements need the built checker).

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
    "MerlinVerify/Check/Typing.lean",
    "lakefile.toml",
    "lean-toolchain",
    "lake-manifest.json",
]
SCHEMAS = ["lean_ir.schema.json", "lean_cert.schema.json", "lean_receipt.schema.json", "lean_chain.schema.json"]
LOCKED_DIRS = ["MerlinVerify/Sem", "MerlinVerify/Spec", "MerlinVerify/Claims", "MerlinVerify/Assumptions"]
# Proof-only files under a locked directory: their changes cannot alter what any claim means.
PROOF_ONLY = {"MerlinVerify/Sem/Facts.lean", "MerlinVerify/Sem/OutcomeLaws.lean"}
# Claim theorems whose elaborated statements are pinned, with a module that declares each.
PINNED = {
    "MerlinVerify.Check.c1Check_sound": "MerlinVerify.Check.C1",
    "MerlinVerify.Check.moduleCheck_sound": "MerlinVerify.Check.Module",
    "MerlinVerify.Check.supportCheck_sound": "MerlinVerify.Check.Rulebook",
    "MerlinVerify.Check.supportCheck_safe": "MerlinVerify.Check.Rulebook",
    "MerlinVerify.Check.funcRefines_of_support": "MerlinVerify.Check.Rulebook",
    "MerlinVerify.Check.chain_refinesC": "MerlinVerify.Check.Chain",
    "MerlinVerify.Check.chain_refines": "MerlinVerify.Check.Chain",
    "MerlinVerify.Check.chain_defined_refines": "MerlinVerify.Check.Chain",
    "MerlinVerify.Check.Definedness.definedCheck_sound": "MerlinVerify.Check.DefinedLinalg",
    "MerlinVerify.Check.chain_extended_defined_refines": "MerlinVerify.Check.ChainExtended",
    "MerlinVerify.Check.chain_typed_extended_defined_refines": "MerlinVerify.Check.ChainExtended",
    "MerlinVerify.Check.DefinedExtended.check_sound": "MerlinVerify.Check.DefinedExtended",
    "MerlinVerify.Check.Definedness.Refined.checkOps_sound": "MerlinVerify.Check.DefinedRefinements",
    "MerlinVerify.Check.Definedness.Refined.checkRegion_sound": "MerlinVerify.Check.DefinedRefinements",
    "MerlinVerify.Check.Definedness.Refined.check_sound": "MerlinVerify.Check.DefinedRefinements",
    "MerlinVerify.Check.Definedness.Bounds.check_sound": "MerlinVerify.Check.DefinedBounds",
    "MerlinVerify.Check.Definedness.Bounds.extractRule_sound": "MerlinVerify.Check.DefinedBounds",
    "MerlinVerify.Check.Definedness.Bounds.initializedRule_sound": "MerlinVerify.Check.DefinedBounds",
    "MerlinVerify.Check.Definedness.Bounds.overwriteRule_sound": "MerlinVerify.Check.DefinedBounds",
    "MerlinVerify.Check.Definedness.IndexRanges.constantRule_sound": "MerlinVerify.Check.IndexRanges",
    "MerlinVerify.Check.TypedC1.check_sound": "MerlinVerify.Check.TypedC1",
    "MerlinVerify.Check.TypedModule.check_sound": "MerlinVerify.Check.TypedModule",
    "MerlinVerify.Check.chain_typed_refines": "MerlinVerify.Check.TypedChain",
    "MerlinVerify.Check.chain_typed_defined_refines": "MerlinVerify.Check.TypedChain",
    "MerlinVerify.Rules.DequantScalar.perTensor_body_sound": "MerlinVerify.Rules.DequantScalar",
    "MerlinVerify.Rules.DequantScalar.perChannel_body_sound": "MerlinVerify.Rules.DequantScalar",
    "MerlinVerify.Rules.DequantScalar.checkBody_sound": "MerlinVerify.Rules.DequantScalar",
    "MerlinVerify.Rules.DequantScalar.body_poison_sound": "MerlinVerify.Rules.DequantScalar",
    "MerlinVerify.Rules.DequantScalar.checkBody_poison_sound": "MerlinVerify.Rules.DequantScalar",
    "MerlinVerify.Rules.DequantTensor.region_sound": "MerlinVerify.Rules.DequantTensor",
    "MerlinVerify.Rules.DequantTensor.checkLayout_sound": "MerlinVerify.Rules.DequantTensor",
    "MerlinVerify.Rules.DequantTensor.checked_generic_sound": "MerlinVerify.Rules.DequantTensor",
    "MerlinVerify.Rules.DequantTensor.tensor_contract_sound": "MerlinVerify.Rules.DequantTensor",
    "MerlinVerify.Rules.DequantSource.generic_result_sound": "MerlinVerify.Rules.DequantSource",
    "MerlinVerify.Rules.DequantSite.check_sound": "MerlinVerify.Rules.DequantSite",
    "MerlinVerify.Check.dequantRule_sound": "MerlinVerify.Check.RulesDequant",
    "MerlinVerify.Check.Definedness.dequant_result_defined": "MerlinVerify.Check.DefinedDequant",
    "MerlinVerify.Check.Definedness.dequantDefinedRule_sound": "MerlinVerify.Check.DefinedDequant",
    "MerlinVerify.Rules.FillGeneralize.expand_sound": "MerlinVerify.Rules.FillGeneralize",
    "MerlinVerify.Rules.CopyGeneralize.expand_sound": "MerlinVerify.Rules.CopyGeneralize",
    "MerlinVerify.Rules.CopyGeneralize.meaning_success": "MerlinVerify.Rules.CopyGeneralize",
    "MerlinVerify.Check.Definedness.copyDefinedRule_sound": "MerlinVerify.Check.DefinedLinalg",
    "MerlinVerify.Rules.LayoutGeneralize.expand_sound": "MerlinVerify.Rules.LayoutGeneralize",
    "MerlinVerify.Rules.LayoutGeneralize.meaning_success": "MerlinVerify.Rules.LayoutGeneralize",
    "MerlinVerify.Check.reduceRule_sound": "MerlinVerify.Check.RulesLinalg",
    "MerlinVerify.Check.Definedness.layoutDefinedRule_sound": "MerlinVerify.Check.DefinedLinalg",
    "MerlinVerify.Check.Definedness.floatBinaryRule_sound": "MerlinVerify.Check.DefinedRules",
    "MerlinVerify.Check.Definedness.scalarExtractRule_sound": "MerlinVerify.Check.DefinedRules",
    "MerlinVerify.Check.Definedness.compareRule_sound": "MerlinVerify.Check.DefinedControl",
    "MerlinVerify.Check.Definedness.scalarSelectRule_sound": "MerlinVerify.Check.DefinedControl",
    "MerlinVerify.Check.Definedness.tensorSelectRule_sound": "MerlinVerify.Check.DefinedControl",
    "MerlinVerify.Check.Definedness.floatingUnaryRule_sound": "MerlinVerify.Check.DefinedFloating",
    "MerlinVerify.Check.Definedness.floatingFmaRule_sound": "MerlinVerify.Check.DefinedFloating",
    "MerlinVerify.Rules.ScalarCarrier.from_elements_sem": "MerlinVerify.Rules.ScalarCarrierKernel",
    "MerlinVerify.Rules.ScalarCarrier.extract_sem": "MerlinVerify.Rules.ScalarCarrierKernel",
    "MerlinVerify.Check.Definedness.Carrier.rule_sound": "MerlinVerify.Check.DefinedCarrier",
    "MerlinVerify.Check.Definedness.genericDefinedRuleWith_sound": "MerlinVerify.Check.DefinedLinalg",
    "MerlinVerify.Check.Definedness.overwriteDefinedRuleWith_sound": "MerlinVerify.Check.DefinedLinalg",
    "MerlinVerify.Rules.IntegerFold.context_independent": "MerlinVerify.Rules.IntegerFold",
    "MerlinVerify.Rules.IntegerFold.run_sound": "MerlinVerify.Rules.IntegerFold",
    "MerlinVerify.Rules.IntegerFold.check_sound": "MerlinVerify.Rules.IntegerFold",
    "MerlinVerify.Rules.IntegerIdentity.meaning": "MerlinVerify.Rules.IntegerIdentity",
    "MerlinVerify.Rules.IntegerIdentity.site_sound": "MerlinVerify.Rules.IntegerIdentity",
    "MerlinVerify.Rules.IntegerIdentity.check_sound": "MerlinVerify.Rules.IntegerIdentity",
    "MerlinVerify.Rules.IntegerIdempotent.meaning": "MerlinVerify.Rules.IntegerIdempotent",
    "MerlinVerify.Rules.IntegerIdempotent.eliminate_sound": "MerlinVerify.Rules.IntegerIdempotent",
    "MerlinVerify.Rules.IntegerIdempotent.normalize_sound": "MerlinVerify.Rules.IntegerIdempotent",
    "MerlinVerify.Rules.IntegerCast.meaning": "MerlinVerify.Rules.IntegerCast",
    "MerlinVerify.Rules.IntegerCast.learn_sound": "MerlinVerify.Rules.IntegerCast",
    "MerlinVerify.Rules.IntegerCast.advance_sound": "MerlinVerify.Rules.IntegerCast",
    "MerlinVerify.Rules.IntegerCast.replacement_sound": "MerlinVerify.Rules.IntegerCast",
    "MerlinVerify.Rules.IntegerCast.normalize_sound": "MerlinVerify.Rules.IntegerCast",
    "MerlinVerify.Rules.IntegerCastDead.total_exec": "MerlinVerify.Rules.IntegerCastDead",
    "MerlinVerify.Rules.IntegerCastDead.drop_sound": "MerlinVerify.Rules.IntegerCastDead",
    "MerlinVerify.Rules.IntegerCastDead.normalize_sound": "MerlinVerify.Rules.IntegerCastDead",
    "MerlinVerify.Check.TypedNormalizeC1.check_sound": "MerlinVerify.Check.TypedNormalizeC1",
    "MerlinVerify.Check.SharedNormalize.check_sound": "MerlinVerify.Check.SharedNormalize",
    "MerlinVerify.Check.ScalarCarrier.normalize_sound": "MerlinVerify.Check.ScalarCarrier",
    "MerlinVerify.Check.ScalarCarrier.check_sound": "MerlinVerify.Check.ScalarCarrier",
    "MerlinVerify.Check.Definedness.IndexArithmetic.rule_sound": "MerlinVerify.Check.IndexArithmetic",
    "MerlinVerify.Check.Definedness.IndexArithmetic.evidence_sound": "MerlinVerify.Check.IndexArithmetic",
    "MerlinVerify.Check.DefinedConversions.floatFormatRule_sound": "MerlinVerify.Check.DefinedConversions",
    "MerlinVerify.Check.DefinedConversions.integerRule_sound": "MerlinVerify.Check.DefinedConversions",
    "MerlinVerify.Check.DefinedConversions.rules_sound": "MerlinVerify.Check.DefinedConversions",
    "MerlinVerify.Check.TypedDead.drop_sound": "MerlinVerify.Check.TypedDead",
    "MerlinVerify.Check.TypedAlias.continuation": "MerlinVerify.Check.TypedAlias",
    "MerlinVerify.Check.TypedAlias.rename_sound": "MerlinVerify.Check.TypedAlias",
    "MerlinVerify.Check.ReshapeFacts.learn_sound": "MerlinVerify.Check.ReshapeFacts",
    "MerlinVerify.Check.ReshapeFacts.advance_sound": "MerlinVerify.Check.ReshapeFacts",
    "MerlinVerify.Check.ReshapeFacts.compose_sound": "MerlinVerify.Check.ReshapeFacts",
    "MerlinVerify.Check.ReshapeFacts.enrich_sound": "MerlinVerify.Check.ReshapeFacts",
    "MerlinVerify.Check.ReshapeDead.total_exec": "MerlinVerify.Check.ReshapeDead",
    "MerlinVerify.Check.ReshapeDead.drop_sound": "MerlinVerify.Check.ReshapeDead",
    "MerlinVerify.Check.ReshapeNested.check_sound": "MerlinVerify.Check.ReshapeNested",
    "MerlinVerify.Check.ReshapeNormalize.meaning": "MerlinVerify.Check.ReshapeNormalize",
    "MerlinVerify.Check.ReshapeNormalize.normalize_sound": "MerlinVerify.Check.ReshapeNormalize",
    "MerlinVerify.Rules.IntegerOrder.swap_step": "MerlinVerify.Rules.IntegerOrder",
    "MerlinVerify.Rules.IntegerOrder.operations_equal": "MerlinVerify.Rules.IntegerOrder",
    "MerlinVerify.Check.IntegerBody.check_sound": "MerlinVerify.Check.IntegerBody",
    "MerlinVerify.Rules.ReshapeCompose.check_sound": "MerlinVerify.Rules.ReshapeCompose",
    "MerlinVerify.Rules.ReshapeGroups.valid_merge": "MerlinVerify.Rules.ReshapeGroups",
    "MerlinVerify.Rules.ReshapeIndex.compose": "MerlinVerify.Rules.ReshapeIndex",
    "MerlinVerify.Rules.ReshapeMixed.recover_input_shape": "MerlinVerify.Rules.ReshapeMixed",
    "MerlinVerify.Rules.ReshapeMixed.check_sound": "MerlinVerify.Rules.ReshapeMixed",
    "MerlinVerify.Rules.SplatScalar.checkLayout_sound": "MerlinVerify.Rules.SplatScalar",
    "MerlinVerify.Check.SplatNormalize.normalize_sound": "MerlinVerify.Check.SplatNormalize",
    "MerlinVerify.Check.SplatC1.check_sound": "MerlinVerify.Check.SplatC1",
    "MerlinVerify.Rules.StructuredBody.genericSem_equal": "MerlinVerify.Rules.StructuredBody",
    "MerlinVerify.Check.SplatNormalizer.check_sound": "MerlinVerify.Check.SplatNormalizer",
    "MerlinVerify.Check.SplatNested.region_sound": "MerlinVerify.Check.SplatNested",
    "MerlinVerify.Check.SplatNested.liftRun_sound": "MerlinVerify.Check.SplatNested",
    "MerlinVerify.Check.SplatNestedC1.check_sound": "MerlinVerify.Check.SplatNestedC1",
    "MerlinVerify.Rules.StructuredInputs.genericSem_congr": "MerlinVerify.Rules.StructuredInputs",
    "MerlinVerify.Rules.SplatInputs.checkLayout_sound": "MerlinVerify.Rules.SplatInputs",
    "MerlinVerify.Check.BindingFacts.forget_step": "MerlinVerify.Check.BindingFacts",
    "MerlinVerify.Check.UniformFacts.splat_exec": "MerlinVerify.Check.UniformFacts",
    "MerlinVerify.Check.UniformFacts.coupled": "MerlinVerify.Check.UniformFacts",
    "MerlinVerify.Check.UniformFacts.scan_sound": "MerlinVerify.Check.UniformFacts",
}


def locked_files() -> dict[str, str]:
    files: dict[str, Path] = {f: LEAN / f for f in FROZEN}
    for s in SCHEMAS:
        files[f"schemas/{s}"] = ROOT / "merlin" / "contract" / "schemas" / s
    for d in LOCKED_DIRS:
        for p in sorted((LEAN / d).rglob("*.lean")) if (LEAN / d).exists() else []:
            if str(p.relative_to(LEAN)) not in PROOF_ONLY:
                files[str(p.relative_to(LEAN))] = p
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sorted(files.items()) if p.exists()}


def _lake_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.elan' / 'bin'}:{env.get('PATH', '')}"
    return env


def pinned_statements() -> dict[str, str] | None:
    """SHA-256 of each pinned theorem's elaborated statement (needs a built checker)."""
    mods = sorted(set(PINNED.values()))
    proc = subprocess.run(
        ["lake", "exe", "merlin-lean-audit", "--statements", ",".join(PINNED), *mods],
        cwd=LEAN,
        env=_lake_env(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("[FAIL] could not read pinned statements:\n" + proc.stderr[-2000:])
        return None
    stmts = json.loads(proc.stdout.strip().splitlines()[-1])["statements"]
    return {f"statement:{d}": hashlib.sha256(stmts[d].encode()).hexdigest() for d in PINNED}


def check_lock(update: bool, statements: dict[str, str] | None) -> bool:
    """Compare files (and, when given, pinned statements) with the lock; ``update`` rewrites it."""
    current = locked_files() | (statements or {})
    if update:
        LOCK.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"wrote {LOCK.relative_to(ROOT)} ({len(current)} files)")
        return True
    if not LOCK.exists():
        print("[FAIL] merlin/lean/SPEC_LOCK missing; run with --update-lock after review")
        return False
    pinned = json.loads(LOCK.read_text())
    if statements is None:  # files only
        pinned = {k: v for k, v in pinned.items() if not k.startswith("statement:")}
    changed = sorted(k for k in set(current) | set(pinned) if current.get(k) != pinned.get(k))
    for k in changed:
        state = "added" if k not in pinned else "removed" if k not in current else "changed"
        what = "pinned statement" if k.startswith("statement:") else "locked file"
        print(f"[FAIL] {what} {state}: {k} (review, then --update-lock)")
    return not changed


def build() -> bool:
    # The library alone is the default target; the checker and audit executables must be rebuilt
    # too, or receipts would be written by a stale checker.
    proc = subprocess.run(
        ["lake", "build", "MerlinVerify", "merlin-lean", "merlin-lean-audit"],
        cwd=LEAN,
        env=_lake_env(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print("[FAIL] lake build failed:\n" + proc.stdout[-2000:] + proc.stderr[-2000:])
    return proc.returncode == 0


def run_audit() -> bool:
    env = _lake_env()
    mods = sorted(
        str(p.relative_to(LEAN).with_suffix("")).replace("/", ".") for p in (LEAN / "MerlinVerify").rglob("*.lean")
    )
    audit = subprocess.run(
        ["lake", "exe", "merlin-lean-audit", *mods], cwd=LEAN, env=env, capture_output=True, text=True
    )
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
    if "--lock-only" in argv and not update:
        ok = check_lock(False, None)
        print("pinned statements not checked (--lock-only)")
    elif not build():
        ok = False
    else:
        statements = pinned_statements()
        ok = statements is not None and check_lock(update, statements)
        if not update:
            ok = run_audit() and ok
    print("lean guard: OK" if ok else "lean guard: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
