"""Replayable W8A8 prefix-bound qualification on explicit per-argument domains.

This route certifies requested facts about actual source/reference prefixes. It
does not certify whole-function initialized execution or admit numerical laws.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from merlin.verify.lean import bounded_numerics, interpreted, numerical_policy, policy, rotary_evidence
from merlin.verify.lean.check import _file_ref, _function_attempt, _lean_block, _snapshot, _work, _write, lean_root
from merlin.verify.lean.export import MLIR_EXPORT

_THEOREM = "MerlinVerify.Check.PolicyW8A8RotaryPrefix.entry_bound"
_ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def _same_json(left: Any, right: Any) -> bool:
    """Preserve JSON number/boolean distinctions that Python container equality erases."""
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _sources() -> list[dict[str, Any]]:
    paths = {Path(record["path"]) for record in numerical_policy._sources()}
    paths.update(Path(__file__).resolve().parent / name for name in ("rotary_policy.py", "rotary_evidence.py"))
    return [_file_ref(path) for path in sorted(paths)]


def _decode(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key in rotary inputs or evidence")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique)


def _roster(program: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        op["attrs"]["sym_name"]["value"]: op["regions"][0]["blocks"][0]["args"]
        for op in program["module"]["ops"]
        if op["name"] == "func.func" and op["regions"] and op["regions"][0]["blocks"]
    }


def check_w8a8_rotary_draft(
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
    """Check one actual pass boundary plus requested initialized prefix bounds."""
    receipt = (destination / "receipt.json").resolve()
    if receipt.exists() or receipt in {input_contracts.resolve(), exponent_contracts.resolve()}:
        raise ValueError("qualification requires a new destination, distinct from its input contracts")
    return _function_attempt(
        destination,
        index_bits,
        lambda: _check(
            stage, destination, input_contracts, exponent_contracts, m2m_python, scope, index_bits, timeout_seconds
        ),
    )


def _check(
    stage: Path,
    destination: Path,
    input_contracts: Path,
    exponent_contracts: Path,
    m2m_python: Path,
    scope: str,
    index_bits: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    if destination.resolve() == stage.resolve():
        raise ValueError("receipt destination must differ from its source stage")
    sources = _sources()
    work = _work(destination)
    original_inputs = _snapshot(input_contracts, work / "input-contracts.json")
    original_caps = _snapshot(exponent_contracts, work / "exponent-contracts.json")
    inputs, caps = _file_ref(work / "input-contracts.json"), _file_ref(work / "exponent-contracts.json")
    bounds = bounded_numerics.decode(Path(inputs["path"]).read_text())
    preflight = policy.check_w8a8_chain_draft(
        [stage],
        work / "structural",
        m2m_python=m2m_python,
        scope=scope,
        index_bits=index_bits,
        timeout_seconds=timeout_seconds,
    )
    if "cert" not in preflight:
        raise ValueError(f"rotary preflight failed: {preflight.get('reason', 'missing evidence')}")
    preliminary = _decode(Path(preflight["cert"]["path"]))
    programs = preliminary["program_exports"]
    before = _decode(Path(programs[0]["path"]))
    roster = _roster(before)
    if set(bounds) != set(roster):
        raise ValueError("integer contracts must name every source function exactly once")
    entries = rotary_evidence.contracts(_decode(Path(caps["path"])), roster)
    runtime = interpreted.capture(lean_root() / "W8A8RotaryMain.lean", required_module="MerlinVerify.Driver.W8A8Rotary")
    proofs = {root: runtime.axioms(root, module) for root, module in rotary_evidence.ROOTS.items()}
    if any(not set(axioms) <= _ALLOWED_AXIOMS for axioms in proofs.values()):
        raise ValueError("rotary proof depends on unapproved axioms")
    arguments = [programs[0]["path"], programs[1]["path"], inputs["path"], caps["path"], str(index_bits), scope]
    process = runtime.run(arguments, timeout_seconds=timeout_seconds)
    (work / "driver.stdout").write_text(process.stdout)
    (work / "driver.stderr").write_text(process.stderr)
    (work / "driver-process.json").write_text(
        json.dumps({"arguments": arguments, "returncode": process.returncode}) + "\n"
    )
    if process.returncode != 1:
        raise ValueError(f"rotary candidate driver failed (exit {process.returncode}): {process.stderr[-300:]}")
    evidence = json.loads(process.stdout)
    rotary_evidence.validate(evidence, entries=entries, before=before, index_bits=index_bits, scope=scope)
    passed = rotary_evidence.requested_passed(evidence)
    report = work / "rotary-report.json"
    report.write_text(
        json.dumps(
            {
                "evidence": evidence,
                "requested_prefix_checks_passed": passed,
                "proof_axioms": proofs,
                "runtime": runtime.evidence,
                "runtime_sha256": runtime.digest,
                "candidate_sources": sources,
                "input_contracts": {"original": original_inputs, "snapshot": inputs},
                "exponent_contracts": {"original": original_caps, "snapshot": caps},
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
        "boundary": "w8a8-rotary-prefix-candidate",
        "checker": "MerlinVerify.Check.PolicyW8A8RotaryPrefix.entry",
        "theorem": _THEOREM,
        "domain": (
            f"finite/nonzero f32 candidate; integer contracts {inputs['sha256']}; exponent contracts {caps['sha256']}"
        ),
        "axioms": proofs[_THEOREM],
        "functions": list(roster),
        "lean": {**preflight["lean"], "checker_sha256": runtime.digest},
        "cert": _file_ref(report),
        "verdict": "abstracted",
        "reason": (
            "requested source/reference prefix bounds checked; full numerical execution and admission remain open"
            if passed
            else "requested prefix bounds or structural match refused"
        ),
    }
    live = [
        original_inputs,
        original_caps,
        inputs,
        caps,
        receipt["cert"],
        receipt["before"],
        receipt["after"],
        preflight["cert"],
        _file_ref(work / "structural/receipt.json"),
        _file_ref(m2m_python),
        *programs,
    ]
    numerical_policy._fresh([*sources, *live])
    if sources != _sources() or _lean_block() != preflight["lean"]:
        raise ValueError("rotary source inventory or structural checker changed")
    if _file_ref(MLIR_EXPORT)["sha256"] != preflight["exporter"]["executable_sha256"]:
        raise ValueError("authoritative exporter changed during verification")
    runtime.assert_current()
    return _write(destination, receipt)


def replay(receipt_path: Path, destination: Path, *, m2m_python: Path) -> dict[str, Any]:
    """Regenerate exports, live numerical checks and fresh audits before comparing evidence."""
    original = _file_ref(receipt_path)
    receipt = _decode(receipt_path)
    if (
        receipt.get("boundary") != "w8a8-rotary-prefix-candidate"
        or receipt.get("theorem") != _THEOREM
        or receipt.get("verdict") != "abstracted"
    ):
        raise ValueError("receipt is not an unapproved rotary-prefix candidate")
    numerical_policy._fresh([receipt["cert"], receipt["before"], receipt["after"]])
    report = _decode(Path(receipt["cert"]["path"]))
    numerical_policy._fresh(
        [
            *report["candidate_sources"],
            *report["runtime"]["files"],
            *report["program_exports"],
            report["input_contracts"]["original"],
            report["input_contracts"]["snapshot"],
            report["exponent_contracts"]["original"],
            report["exponent_contracts"]["snapshot"],
            report["compiler_python"],
            report["structural_preflight"],
            report["structural_receipt"],
        ]
    )
    if not _same_json(_sources(), report["candidate_sources"]) or not _same_json(
        _file_ref(m2m_python), report["compiler_python"]
    ):
        raise ValueError("replay source inventory or compiler differs from its receipt")
    result = check_w8a8_rotary_draft(
        Path(report["stage"]),
        destination,
        input_contracts=Path(report["input_contracts"]["snapshot"]["path"]),
        exponent_contracts=Path(report["exponent_contracts"]["snapshot"]["path"]),
        m2m_python=m2m_python,
        scope=report["evidence"]["policy_scope"],
        index_bits=report["evidence"]["index_bits"],
        timeout_seconds=report["checker_timeout_seconds"],
    )
    if "cert" not in result:
        raise ValueError("rotary replay did not publish fresh checked evidence")
    fresh = _decode(Path(result["cert"]["path"]))
    for key in (
        "evidence",
        "requested_prefix_checks_passed",
        "proof_axioms",
        "runtime",
        "runtime_sha256",
        "candidate_sources",
    ):
        if not _same_json(fresh[key], report[key]):
            raise ValueError(f"rotary replay differs in {key}")
    for key in ("before", "after", "checker", "theorem", "domain", "axioms", "functions", "lean", "verdict", "reason"):
        if not _same_json(result[key], receipt[key]):
            raise ValueError(f"rotary replay receipt differs in {key}")
    if not _same_json(
        [(item["sha256"], item["bytes"]) for item in fresh["program_exports"]],
        [(item["sha256"], item["bytes"]) for item in report["program_exports"]],
    ):
        raise ValueError("rotary replay exports changed")
    numerical_policy._fresh([original])
    return result


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
    result = check_w8a8_rotary_draft(
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
