"""Replayable receipts for ordered floating contractions under the proved W8A8 model.

Source and policy-reference bounds are separate requests. Their prefixes retain actual
SSA facts; unsupported transcendental numerics, native fidelity and policy review stay open.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from merlin.verify.lean import (
    bounded_numerics,
    ieee_runtime,
    numerical_policy,
    ordered_evidence,
    policy,
    rotary_evidence,
    rotary_policy,
)
from merlin.verify.lean.check import _file_ref, _function_attempt, _lean_block, _snapshot, _work, _write, lean_root
from merlin.verify.lean.export import MLIR_EXPORT

_REPORT_FIELDS = {
    "evidence",
    "requested_prefix_checks_passed",
    "proof_axioms",
    "runtime",
    "runtime_sha256",
    "candidate_sources",
    "input_contracts",
    "exponent_contracts",
    "stage",
    "compiler_python",
    "structural_preflight",
    "structural_receipt",
    "program_exports",
    "checker_timeout_seconds",
}


def _sources() -> list[dict[str, Any]]:
    paths = {Path(record["path"]) for record in rotary_policy._sources()}
    paths.update(
        Path(__file__).resolve().parent / name
        for name in ("ordered_policy.py", "ordered_evidence.py", "ieee_runtime.py")
    )
    return [_file_ref(path) for path in sorted(paths)]


def check_w8a8_ordered_draft(
    stage: Path,
    destination: Path,
    *,
    input_contracts: Path,
    exponent_contracts: Path,
    m2m_python: Path,
    scope: str = "captured",
    index_bits: int = 64,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Check actual source/reference prefix bounds without admitting whole-pass correctness."""
    receipt = (destination / "receipt.json").resolve()
    if receipt.exists() or receipt in {input_contracts.resolve(), exponent_contracts.resolve()}:
        raise ValueError("ordered qualification requires a new destination distinct from its contracts")
    return _function_attempt(
        destination,
        index_bits,
        lambda: _check(
            stage, destination, input_contracts, exponent_contracts, m2m_python, scope, index_bits, timeout_seconds
        ),
    )


def _check(stage, destination, input_contracts, exponent_contracts, m2m_python, scope, index_bits, timeout_seconds):
    if destination.resolve() == stage.resolve():
        raise ValueError("ordered destination must differ from its source stage")
    sources = _sources()
    work = _work(destination)
    originals = {
        "input_contracts": _snapshot(input_contracts, work / "input-contracts.json"),
        "exponent_contracts": _snapshot(exponent_contracts, work / "exponent-contracts.json"),
    }
    snapshots = {
        "input_contracts": _file_ref(work / "input-contracts.json"),
        "exponent_contracts": _file_ref(work / "exponent-contracts.json"),
    }
    preflight = policy.check_w8a8_chain_draft(
        [stage],
        work / "structural",
        m2m_python=m2m_python,
        scope=scope,
        index_bits=index_bits,
        timeout_seconds=timeout_seconds,
    )
    if "cert" not in preflight:
        raise ValueError(f"ordered preflight failed: {preflight.get('reason')}")
    programs = rotary_policy._decode(Path(preflight["cert"]["path"]))["program_exports"]
    before = rotary_policy._decode(Path(programs[0]["path"]))
    roster = rotary_policy._roster(before)
    bounds = bounded_numerics.decode(Path(snapshots["input_contracts"]["path"]).read_text())
    if set(bounds) != set(roster):
        raise ValueError("ordered integer contracts must name every source function exactly once")
    entries = ordered_evidence.contracts(rotary_policy._decode(Path(snapshots["exponent_contracts"]["path"])), roster)
    runtime = ieee_runtime.capture(
        lean_root() / "W8A8OrderedMain.lean",
        required_module="MerlinVerify.Driver.W8A8Ordered",
        diagnostics=work / "runtime-integrity",
    )
    proofs = {root: runtime.axioms(root, module) for root, module in ordered_evidence.ROOTS.items()}
    if any(not set(axioms) <= {"propext", "Classical.choice", "Quot.sound"} for axioms in proofs.values()):
        raise ValueError("ordered contraction proof depends on unapproved axioms")
    arguments = [
        programs[0]["path"],
        programs[1]["path"],
        snapshots["input_contracts"]["path"],
        snapshots["exponent_contracts"]["path"],
        str(index_bits),
        scope,
    ]
    process = runtime.run(arguments, timeout_seconds=timeout_seconds)
    (work / "driver.stdout").write_text(process.stdout)
    (work / "driver.stderr").write_text(process.stderr)
    (work / "driver-process.json").write_text(
        json.dumps({"arguments": arguments, "returncode": process.returncode}) + "\n"
    )
    if process.returncode != 1:
        raise ValueError(f"ordered driver failed (exit {process.returncode}): {process.stderr[-300:]}")
    evidence = rotary_policy._decode(work / "driver.stdout")
    ordered_evidence.validate(evidence, entries=entries, before=before, index_bits=index_bits, scope=scope)
    passed = ordered_evidence.requested_passed(evidence)
    report = work / "ordered-report.json"
    report.write_text(
        json.dumps(
            {
                "evidence": evidence,
                "requested_prefix_checks_passed": passed,
                "proof_axioms": proofs,
                "runtime": runtime.evidence,
                "runtime_sha256": runtime.digest,
                "candidate_sources": sources,
                **{key: {"original": originals[key], "snapshot": snapshots[key]} for key in originals},
                "stage": str(stage.resolve()),
                "compiler_python": _file_ref(m2m_python),
                "structural_preflight": preflight["cert"],
                "structural_receipt": _file_ref(work / "structural/receipt.json"),
                "program_exports": programs,
                "checker_timeout_seconds": timeout_seconds,
            },
            indent=2,
        )
        + "\n"
    )
    receipt = {
        **preflight,
        "boundary": "w8a8-ordered-contraction-prefix-candidate",
        "checker": "MerlinVerify.Check.PolicyW8A8OrderedPrefix.entry",
        "theorem": ordered_evidence.PREFIX,
        "domain": (
            f"explicit W8A8 IEEE arithmetic; input contracts {snapshots['input_contracts']['sha256']}; "
            f"scoped exponent contracts {snapshots['exponent_contracts']['sha256']}"
        ),
        "axioms": proofs[ordered_evidence.PREFIX],
        "functions": list(roster),
        "lean": {**preflight["lean"], "checker_sha256": runtime.digest},
        "cert": _file_ref(report),
        "verdict": "abstracted",
        "reason": (
            "requested scoped prefix bounds checked; whole-pass, policy/native admission remain open"
            if passed
            else "requested scoped prefix bounds or structural match refused"
        ),
    }
    numerical_policy._fresh(
        [
            *sources,
            *programs,
            *originals.values(),
            *snapshots.values(),
            receipt["before"],
            receipt["after"],
            receipt["cert"],
            preflight["cert"],
            _file_ref(work / "structural/receipt.json"),
            _file_ref(m2m_python),
        ]
    )
    if not ordered_evidence.same(sources, _sources()):
        raise ValueError("ordered source inventory changed")
    if not ordered_evidence.same(_lean_block(), preflight["lean"]):
        raise ValueError("ordered structural checker changed during verification")
    if _file_ref(MLIR_EXPORT)["sha256"] != preflight["exporter"]["executable_sha256"]:
        raise ValueError("ordered authoritative exporter changed during verification")
    runtime.assert_current()
    return _write(destination, receipt)


def replay(receipt_path: Path, destination: Path, *, m2m_python: Path) -> dict[str, Any]:
    """Regenerate authoritative exports, scoped checks and audits in a separate invocation."""
    if (destination / "receipt.json").exists():
        raise ValueError("ordered replay requires a new destination")
    original = _file_ref(receipt_path)
    receipt = rotary_policy._decode(receipt_path)
    if (
        receipt.get("boundary") != "w8a8-ordered-contraction-prefix-candidate"
        or receipt.get("theorem") != ordered_evidence.PREFIX
        or receipt.get("verdict") != "abstracted"
    ):
        raise ValueError("receipt is not an unapproved ordered-contraction candidate")
    numerical_policy._fresh([receipt["cert"], receipt["before"], receipt["after"]])
    report = rotary_policy._decode(Path(receipt["cert"]["path"]))
    rotary_evidence._object(report, _REPORT_FIELDS, "ordered report")
    numerical_policy._fresh(
        [
            *report["candidate_sources"],
            *report["runtime"]["files"],
            *report["program_exports"],
            report["compiler_python"],
            report["structural_preflight"],
            report["structural_receipt"],
            *[
                report[key][part]
                for key in ("input_contracts", "exponent_contracts")
                for part in ("original", "snapshot")
            ],
        ]
    )
    if not ordered_evidence.same(_sources(), report["candidate_sources"]) or not ordered_evidence.same(
        _file_ref(m2m_python), report["compiler_python"]
    ):
        raise ValueError("ordered replay selected different sources or compiler")
    fresh_receipt = check_w8a8_ordered_draft(
        Path(report["stage"]),
        destination,
        input_contracts=Path(report["input_contracts"]["snapshot"]["path"]),
        exponent_contracts=Path(report["exponent_contracts"]["snapshot"]["path"]),
        m2m_python=m2m_python,
        scope=report["evidence"]["policy_scope"],
        index_bits=report["evidence"]["index_bits"],
        timeout_seconds=report["checker_timeout_seconds"],
    )
    if "cert" not in fresh_receipt:
        raise ValueError(f"ordered replay failed: {fresh_receipt.get('reason')}")
    fresh = rotary_policy._decode(Path(fresh_receipt["cert"]["path"]))
    for key in (
        "evidence",
        "requested_prefix_checks_passed",
        "runtime",
        "runtime_sha256",
        "proof_axioms",
        "candidate_sources",
    ):
        if not ordered_evidence.same(fresh[key], report[key]):
            raise ValueError(f"ordered replay differs in {key}")
    for key in ("before", "after", "checker", "theorem", "domain", "axioms", "functions", "lean", "verdict", "reason"):
        if not ordered_evidence.same(fresh_receipt[key], receipt[key]):
            raise ValueError(f"ordered replay receipt differs in {key}")
    if not ordered_evidence.same(
        [(item["sha256"], item["bytes"]) for item in fresh["program_exports"]],
        [(item["sha256"], item["bytes"]) for item in report["program_exports"]],
    ):
        raise ValueError("ordered replay exports changed")
    numerical_policy._fresh([original, receipt["cert"], receipt["before"], receipt["after"]])
    return fresh_receipt
