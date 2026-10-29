#!/usr/bin/env python3
"""Prepare the bf16 pass claim and unapplied specification/pin patch for human review.

This is an evidence publisher, not an optimization checker. It consumes byte-bound reports
from the Lean checker and requires fresh bindings, complete site qualification and replay.
It never approves a policy or updates SPEC_LOCK. After human approval, the same command can
publish the scoped claim as verified only when the approved specification and every theorem
statement are present in the reviewed lock.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root
from merlin.verify.bf16_policy_gate import ADMISSION, REVIEW, require_admission
from merlin.verify.bf16_policy_gate import ROOTS as GATE_ROOTS
from merlin.verify.gate_errors import GateError
from merlin.verify.lean.bf16_f32acc import BOUNDARY, CONTRACT_THEOREMS, SEGMENT, replay
from merlin.verify.lean.check import _file_ref, _stale

SPEC = "MerlinVerify/Spec/Bf16F32AccContract.lean"
POLICY_FILES = [
    "MerlinVerify/Rules/Bf16F32AccContract.lean",
    "MerlinVerify/Rules/Bf16F32AccBody.lean",
    "MerlinVerify/Rules/Bf16F32AccReference.lean",
    "MerlinVerify/Rules/W8A8TensorTrace.lean",
    "MerlinVerify/Rules/W8A8TensorOperands.lean",
    "MerlinVerify/Rules/W8A8Fill.lean",
]
CLAIM_FILES = [
    "MerlinVerify/Check/Bf16F32Acc.lean",
    "MerlinVerify/Check/Bf16F32AccChain.lean",
    "MerlinVerify/Check/Bf16F32AccSites.lean",
    "MerlinVerify/Check/ParametricAlpha.lean",
    "MerlinVerify/Check/Alpha.lean",
]
REVIEW_FILES = [*POLICY_FILES, *CLAIM_FILES]
THEOREMS = {
    **CONTRACT_THEOREMS,
    **{f"{module}.{name}": module for _, module, check, qualify in (BOUNDARY, SEGMENT) for name in (check, qualify)},
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reviewed(lean: Path, lock: dict[str, str], statements: dict[str, str]) -> bool:
    """Require complete pins AND installed independently attributable policy admission.

    This publisher creates neither admission nor human review. Their authentic
    installation is a trusted-host governance premise, not established by hashes.
    """
    pins_match = (
        set(statements) == set(THEOREMS)
        and all(
            (lean / name).is_file() and lock.get(name) == digest((lean / name).read_bytes())
            for name in [SPEC, *REVIEW_FILES]
        )
        and all(lock.get(f"statement:{name}") == digest(statement.encode()) for name, statement in statements.items())
    )
    if not pins_match:
        return False
    try:
        admitted = require_admission(lean.parent.parent)
    except GateError:
        return False
    return set(admitted["statements"]) == set(GATE_ROOTS) and all(
        statements.get(name) == statement for name, statement in admitted["statements"].items()
    )


def theorem_statements(root: Path) -> tuple[dict[str, str], dict]:
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.elan/bin'}:{env.get('PATH', '')}"
    subprocess.run(
        [sys.executable, str(root / "build_tools/scripts/check_lean_guard.py"), "--lock-only"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    modules = sorted({"Main", *THEOREMS.values()})
    subprocess.run(
        ["lake", "build", "merlin-lean", "merlin-lean-audit", *modules],
        cwd=root / "merlin/lean",
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    audited = subprocess.run(
        ["lake", "exe", "merlin-lean-audit", *modules],
        cwd=root / "merlin/lean",
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(audited.stdout.strip().splitlines()[-1])
    if audit["violations"]:
        raise ValueError(f"checker dependency audit refused: {audit}")
    process = subprocess.run(
        ["lake", "exe", "merlin-lean-audit", "--statements", ",".join(THEOREMS), *sorted(set(THEOREMS.values()))],
        cwd=root / "merlin/lean",
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    statements = json.loads(process.stdout.strip().splitlines()[-1])["statements"]
    if set(statements) != set(THEOREMS):
        raise ValueError("the auditor did not return every claim statement")
    return statements, audit


def pinning_patch(root: Path) -> str:
    guard = "build_tools/scripts/check_lean_guard.py"
    old = (root / guard).read_text()
    frozen = "".join(f'    "{name}",\n' for name in REVIEW_FILES if f'"{name}",' not in old)
    pins = "".join(f'    "{name}": "{module}",\n' for name, module in THEOREMS.items() if f'"{name}":' not in old)
    updated = old.replace("FROZEN = [\n", "FROZEN = [\n" + frozen, 1)
    updated = updated.replace("PINNED = {\n", "PINNED = {\n" + pins, 1)
    patch = list(difflib.unified_diff(old.splitlines(True), updated.splitlines(True), f"a/{guard}", f"b/{guard}"))
    contract = "merlin/lean/" + POLICY_FILES[0]
    original = (root / contract).read_text()
    shim = "-- Approved specification is locked under Spec; preserve existing imports.\n"
    shim += "import MerlinVerify.Spec.Bf16F32AccContract\n"
    approved = original.replace(
        "REVIEW CANDIDATE: written by\n"
        "the author of its checker after reading the implementing pass, so it is not barrier-authored\n"
        "(design §1.2 item 6) and it is not a reviewed `Spec/` entry. Its six clauses are:",
        "The original candidate was written after reading the pass; independent human review is\n"
        "required before applying the patch that installs this Spec entry. Its six clauses are:",
    )
    patch += difflib.unified_diff(original.splitlines(True), shim.splitlines(True), f"a/{contract}", f"b/{contract}")
    patch += difflib.unified_diff([], approved.splitlines(True), "/dev/null", "b/merlin/lean/" + SPEC)
    return "".join(patch)


def checked_suite(path: Path) -> dict[str, int]:
    suites = ET.parse(path).getroot().findall("testsuite")
    totals = {
        key: sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "failures", "errors", "skipped")
    }
    if not totals["tests"] or any(totals[key] for key in ("failures", "errors", "skipped")):
        raise ValueError(f"qualification suite is not complete and passing: {path}: {totals}")
    return totals


def evidence(path: Path, compiler: Path, work: Path) -> dict:
    receipt = json.loads(path.read_text())
    if stale := _stale(receipt, str(path)):
        raise ValueError(f"stale receipt: {stale}")
    for reference in [receipt["cert"], receipt["before"], receipt["after"], *receipt["exports"].values()]:
        if _file_ref(Path(reference["path"])) != reference:
            raise ValueError(f"changed bound bytes: {reference['path']}")
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    for reference in report["program_exports"]:
        if _file_ref(Path(reference["path"])) != reference:
            raise ValueError(f"changed intermediate export: {reference['path']}")
    boundary = report.get("policy_boundary", report)
    local = boundary["pass_boundary"]
    count = boundary["candidates"]
    if not (local["checked"] and local["semantic_match"] and local["site_qualification_passed"]):
        raise ValueError(f"pass proof refused: {path}")
    if local["covered_sites"] != count or len(local["site_obligations"]) != count:
        raise ValueError(f"incomplete candidate coverage: {path}")
    if not all(site["qualification_passed"] for site in local["site_obligations"]):
        raise ValueError(f"unqualified local source: {path}")
    if "segment_match" in report and not report["segment_match"]:
        raise ValueError(f"recorded refinement suffix failed: {path}")
    # Reports and hashes are untrusted hints. Re-execute the verified checker before publishing
    # a claim; even a forged all-green report with freshly recomputed hashes must be rejected.
    rechecked = replay(path, m2m_python=compiler, work=work)
    if not rechecked["fresh"] or not rechecked["matches_recorded"]:
        raise ValueError(f"replay refused the claimed evidence: {path}: {rechecked}")
    return {
        "receipt": _file_ref(path),
        "report": receipt["cert"],
        "checker": receipt["lean"],
        "exporter": receipt["exporter"],
        "boundary": receipt["boundary"],
        "candidates": count,
        "pass_boundary": local,
        "whole_capture_qualification": boundary["qualification_passed"],
        "suffix": report.get("suffix_boundaries", []),
        "replay": rechecked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipts", type=Path, default=artifacts_dir() / "verification/merlin-passes/frontend/bf16-f32acc"
    )
    parser.add_argument("--compiler-python", type=Path, required=True)
    args = parser.parse_args()
    root, base = repo_root(), args.receipts.resolve()
    lock_ref = _file_ref(root / "merlin/lean/SPEC_LOCK")
    installed_spec = [SPEC] if (root / "merlin/lean" / SPEC).is_file() else []
    policy_refs = [_file_ref(root / "merlin/lean" / name) for name in [*REVIEW_FILES, *installed_spec]]
    statements, audit = theorem_statements(root)
    lock = json.loads((root / "merlin/lean/SPEC_LOCK").read_text())
    work = base / "validation" / "claim-replay" / uuid.uuid4().hex
    work.mkdir(parents=True)
    paths = sorted(base.glob("*/receipt.json"))
    with ThreadPoolExecutor(max_workers=3) as executor:
        records = list(executor.map(lambda path: evidence(path, args.compiler_python, work / path.parent.name), paths))
    shutil.rmtree(work)
    if not records or not any(record["candidates"] > 0 for record in records):
        raise ValueError("no positive numerical boundary evidence")
    fast = checked_suite(base / "validation/review-fast.xml")
    replay = checked_suite(base / "validation/review-replay.xml")
    if replay["tests"] != len(records):
        raise ValueError("replay roster does not cover the retained receipts")
    for reference in [lock_ref, *policy_refs, *(record["receipt"] for record in records)]:
        if _file_ref(Path(reference["path"])) != reference:
            raise ValueError(f"claim inputs changed during publication: {reference['path']}")
    for record in records:
        path = Path(record["receipt"]["path"])
        if stale := _stale(json.loads(path.read_text()), str(path)):
            raise ValueError(f"receipt became stale during publication: {stale}")
    approved = reviewed(root / "merlin/lean", lock, statements)
    approval_refs = None
    if approved:
        admission = require_admission(root)
        if any(
            record["checker"] != admission["lean"] or record["exporter"] != admission["exporter"] for record in records
        ):
            raise ValueError("publication checker/exporter differs from installed admission")
        approval_refs = {"admission": _file_ref(root / ADMISSION), "review": _file_ref(root / REVIEW)}
    package = {
        "schema": "merlin.lean.bf16-pass-claim.v1",
        "claim_type": "policy",
        "status": "verified" if approved else "human_review_pending",
        "human_reviewed": approved,
        "approval_records": approval_refs,
        "scope": "Merlin bf16 preparation pass between model2MLIR output and the OOT compiler handoff",
        "claim": "Complete output equality to the source-derived contract under SSA-invariant operation semantics; "
        "strict local policy refinement for every recognized contraction on all typed nonpoison tensor operands "
        "at the receipt's index width.",
        "whole_shared_interval_verified": False,
        "whole_capture_definedness_claimed": False,
        "remaining_approval": []
        if approved
        else [
            "independent policy and claim-scope review; apply the prepared pin patch "
            "and regenerate SPEC_LOCK and receipts after approval"
        ],
        "trusted_boundary": (
            "input exporter/round trip and the reviewed interpretation of primitives; "
            "OOT and upstream lowering excluded"
        ),
        "fusion_scope": (
            "uniform fusion theorem only; actual lowering's contraction behavior remains a stated model/TCB assumption"
        ),
        "spec_lock": lock_ref,
        "policy_files": policy_refs,
        "dependency_audit": audit,
        "statements": {
            name: {"module": THEOREMS[name], "sha256": digest(statement.encode()), "statement": statement}
            for name, statement in statements.items()
        },
        "receipts": records,
        "fast_suite": fast,
        "replay_suite": replay,
    }
    destination = base / "review"
    destination.mkdir(exist_ok=True)
    if approved:
        if require_admission(root) != admission or any(
            _file_ref(Path(ref["path"])) != ref for ref in approval_refs.values()
        ):
            raise ValueError("installed policy approval changed during publication")
    (destination / ("claim.json" if approved else "candidate.json")).write_text(json.dumps(package, indent=2) + "\n")
    if not approved:
        (destination / "after-approval.patch").write_text(pinning_patch(root))
    print(
        json.dumps(
            {key: package[key] for key in ("status", "human_reviewed", "whole_shared_interval_verified")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
