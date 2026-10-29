"""Fresh receipts for the proved W8A8 arithmetic family and explicit input domains.

The generic routes remain unchanged. This wrapper invokes the named specialization,
audits its construction and application roots, and binds actual exports/contracts.
Native primitive fidelity, independent policy admission and full-model execution are separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from merlin.verify.lean import (
    bounded_numerics,
    ieee_evidence,
    ieee_runtime,
    numerical_policy,
    rotary_evidence,
    rotary_policy,
)
from merlin.verify.lean.check import _file_ref, _function_attempt, _work, _write, lean_root


def _sources() -> list[dict[str, Any]]:
    paths = {Path(record["path"]) for record in rotary_policy._sources()}
    paths.update(
        Path(__file__).resolve().parent / name for name in ("ieee_policy.py", "ieee_evidence.py", "ieee_runtime.py")
    )
    return [_file_ref(path) for path in sorted(paths)]


def check_w8a8_ieee_draft(
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
    """Specialize actual generic checker evidence to the proved arithmetic family."""
    receipt = (destination / "receipt.json").resolve()
    if receipt.exists() or receipt in {input_contracts.resolve(), exponent_contracts.resolve()}:
        raise ValueError("explicit-model qualification requires a new destination distinct from its contracts")
    return _function_attempt(
        destination,
        index_bits,
        lambda: _check(
            stage, destination, input_contracts, exponent_contracts, m2m_python, scope, index_bits, timeout_seconds
        ),
    )


def _check(stage, destination, input_contracts, exponent_contracts, m2m_python, scope, index_bits, timeout_seconds):
    if destination.resolve() == stage.resolve():
        raise ValueError("explicit-model destination must differ from its source stage")
    sources = _sources()
    work = _work(destination)
    preliminary = rotary_policy.check_w8a8_rotary_draft(
        stage,
        work / "generic",
        input_contracts=input_contracts,
        exponent_contracts=exponent_contracts,
        m2m_python=m2m_python,
        scope=scope,
        index_bits=index_bits,
        timeout_seconds=timeout_seconds,
    )
    if "cert" not in preliminary:
        raise ValueError(f"explicit-model preflight failed: {preliminary.get('reason')}")
    generic = rotary_policy._decode(Path(preliminary["cert"]["path"]))
    programs = generic["program_exports"]
    inputs, caps = generic["input_contracts"]["snapshot"], generic["exponent_contracts"]["snapshot"]
    before = rotary_policy._decode(Path(programs[0]["path"]))
    bounds = bounded_numerics.decode(Path(inputs["path"]).read_text())
    entries = rotary_evidence.contracts(rotary_policy._decode(Path(caps["path"])), rotary_policy._roster(before))
    runtime = ieee_runtime.capture(
        lean_root() / "W8A8IEEEMain.lean",
        required_module="MerlinVerify.Driver.W8A8IEEE",
        diagnostics=work / "runtime-integrity",
    )
    proofs = {root: runtime.axioms(root, module) for root, module in ieee_evidence.ROOTS.items()}
    if any(not set(axioms) <= {"propext", "Classical.choice", "Quot.sound"} for axioms in proofs.values()):
        raise ValueError("explicit-model proof has unapproved axioms")
    arguments = [programs[0]["path"], programs[1]["path"], inputs["path"], caps["path"], str(index_bits), scope]
    process = runtime.run(arguments, timeout_seconds=timeout_seconds)
    (work / "driver.stdout").write_text(process.stdout)
    (work / "driver.stderr").write_text(process.stderr)
    (work / "driver-process.json").write_text(
        json.dumps({"arguments": arguments, "returncode": process.returncode}) + "\n"
    )
    if process.returncode != 1:
        raise ValueError(f"explicit-model driver failed (exit {process.returncode}): {process.stderr[-300:]}")
    value = rotary_policy._decode(work / "driver.stdout")
    ieee_evidence.validate(
        value,
        entries=entries,
        bounds=bounds,
        before=before,
        generic=generic["evidence"],
        index_bits=index_bits,
        scope=scope,
    )
    prefix_passed = rotary_evidence.requested_passed(value["generic_rotary"])
    path = work / "ieee-report.json"
    path.write_text(
        json.dumps(
            {
                "evidence": value,
                "requested_prefix_checks_passed": prefix_passed,
                "runtime": runtime.evidence,
                "runtime_sha256": runtime.digest,
                "proof_axioms": proofs,
                "candidate_sources": sources,
                "generic_report": preliminary["cert"],
                "generic_receipt": _file_ref(work / "generic/receipt.json"),
                "stage": str(stage.resolve()),
                "compiler_python": _file_ref(m2m_python),
                "program_exports": programs,
                "input_contracts": generic["input_contracts"],
                "exponent_contracts": generic["exponent_contracts"],
                "checker_timeout_seconds": timeout_seconds,
            },
            indent=2,
        )
        + "\n"
    )
    receipt = {
        **preliminary,
        "boundary": "w8a8-explicit-arithmetic-candidate",
        "checker": "MerlinVerify.Check.PolicyW8A8RotaryPrefix.entry",
        "theorem": ieee_evidence.PREFIX,
        "domain": f"explicit IEEE arithmetic family; {preliminary['domain']}; native/policy admission separate",
        "axioms": proofs[ieee_evidence.PREFIX],
        "lean": {**preliminary["lean"], "checker_sha256": runtime.digest},
        "cert": _file_ref(path),
        "verdict": "abstracted",
        "reason": (
            "explicit logical model and requested prefix bounds checked; policy/native admission remain open"
            if prefix_passed
            else "explicit model exists; requested prefix bounds or structural match refused"
        ),
    }
    numerical_policy._fresh(
        [
            *sources,
            *programs,
            receipt["before"],
            receipt["after"],
            receipt["cert"],
            preliminary["cert"],
            _file_ref(work / "generic/receipt.json"),
            _file_ref(m2m_python),
            generic["input_contracts"]["original"],
            inputs,
            generic["exponent_contracts"]["original"],
            caps,
        ]
    )
    if sources != _sources():
        raise ValueError("explicit-model source inventory changed")
    runtime.assert_current()
    return _write(destination, receipt)


def replay(receipt_path: Path, destination: Path, *, m2m_python: Path) -> dict[str, Any]:
    """Regenerate both generic checks and the actual model specialization before comparing."""
    if (destination / "receipt.json").exists():
        raise ValueError("explicit-model replay requires a new destination")
    original = _file_ref(receipt_path)
    receipt = rotary_policy._decode(receipt_path)
    if (
        receipt.get("boundary") != "w8a8-explicit-arithmetic-candidate"
        or receipt.get("theorem") != ieee_evidence.PREFIX
        or receipt.get("verdict") != "abstracted"
    ):
        raise ValueError("receipt is not a draft explicit-model candidate")
    numerical_policy._fresh([receipt["cert"], receipt["before"], receipt["after"]])
    report = rotary_policy._decode(Path(receipt["cert"]["path"]))
    numerical_policy._fresh(
        [
            *report["candidate_sources"],
            *report["runtime"]["files"],
            *report["program_exports"],
            report["compiler_python"],
            report["generic_report"],
            report["generic_receipt"],
            *[report[k][part] for k in ("input_contracts", "exponent_contracts") for part in ("original", "snapshot")],
        ]
    )
    if (
        not rotary_policy._same_json(_sources(), report["candidate_sources"])
        or _file_ref(m2m_python) != report["compiler_python"]
    ):
        raise ValueError("explicit-model replay selected different sources or compiler")
    fresh_receipt = check_w8a8_ieee_draft(
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
        raise ValueError(f"explicit-model replay failed: {fresh_receipt.get('reason')}")
    fresh = rotary_policy._decode(Path(fresh_receipt["cert"]["path"]))
    for key in (
        "evidence",
        "requested_prefix_checks_passed",
        "runtime",
        "runtime_sha256",
        "proof_axioms",
        "candidate_sources",
    ):
        if not rotary_policy._same_json(fresh[key], report[key]):
            raise ValueError(f"explicit-model replay differs in {key}")
    for key in ("before", "after", "checker", "theorem", "domain", "axioms", "functions", "lean", "verdict", "reason"):
        if not rotary_policy._same_json(fresh_receipt[key], receipt[key]):
            raise ValueError(f"explicit-model replay receipt differs in {key}")
    if not rotary_policy._same_json(
        [(item["sha256"], item["bytes"]) for item in fresh["program_exports"]],
        [(item["sha256"], item["bytes"]) for item in report["program_exports"]],
    ):
        raise ValueError("explicit-model replay exports changed")
    numerical_policy._fresh([original])
    return fresh_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--input-contracts", type=Path, required=True)
    parser.add_argument("--exponent-contracts", type=Path, required=True)
    parser.add_argument("--m2m-python", type=Path, required=True)
    parser.add_argument("--scope", choices=("captured", "all"), default="captured")
    parser.add_argument("--index-bits", type=int, default=64)
    args = parser.parse_args()
    result = check_w8a8_ieee_draft(
        args.stage,
        args.destination,
        input_contracts=args.input_contracts,
        exponent_contracts=args.exponent_contracts,
        m2m_python=args.m2m_python,
        scope=args.scope,
        index_bits=args.index_bits,
    )
    print(json.dumps(result, indent=2))
    return 1 if result["verdict"] == "abstracted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
