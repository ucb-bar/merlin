"""Replayable complete-program correspondence to the draft W8A8 policy reference.

This standalone route establishes matching outcomes under the named operational
semantics. Successful reference execution, policy admission, native fidelity and
original-source equivalence remain separate obligations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from merlin.verify.lean import correspondence_evidence as evidence
from merlin.verify.lean import interpreted, numerical_policy, policy
from merlin.verify.lean.check import _file_ref, _function_attempt, _lean_block, _work, _write, lean_root
from merlin.verify.lean.export import MLIR_EXPORT


def _sources() -> list[dict[str, Any]]:
    paths = {Path(record["path"]) for record in numerical_policy._sources()}
    paths.update(
        Path(__file__).resolve().parent / name for name in ("correspondence_policy.py", "correspondence_evidence.py")
    )
    return [_file_ref(path) for path in sorted(paths)]


def check_w8a8_correspondence_draft(
    stages: list[Path],
    destination: Path,
    *,
    m2m_python: Path,
    scope: str = "captured",
    index_bits: int = 64,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Check an actual changed policy boundary and every supplied joined suffix."""
    if (destination / "receipt.json").exists():
        raise ValueError("correspondence qualification requires a new destination")
    return _function_attempt(
        destination,
        index_bits,
        lambda: _check(stages, destination, m2m_python, scope, index_bits, timeout_seconds),
    )


def _check(stages, destination, m2m_python, scope, index_bits, timeout_seconds):
    if not stages or destination.resolve() in {stage.resolve() for stage in stages}:
        raise ValueError("correspondence destination must differ from its nonempty stage list")
    sources = _sources()
    work = _work(destination)
    preflight = policy.check_w8a8_chain_draft(
        stages,
        work / "structural",
        m2m_python=m2m_python,
        scope=scope,
        index_bits=index_bits,
        timeout_seconds=timeout_seconds,
    )
    if "cert" not in preflight:
        raise ValueError(f"correspondence export/preflight failed: {preflight.get('reason')}")
    preliminary = evidence.read(Path(preflight["cert"]["path"]))
    programs = preliminary["program_exports"]
    if len(programs) != len(stages) + 1:
        raise ValueError("correspondence exports omitted an actual joined stage")
    before, after = (evidence.read(Path(programs[index]["path"])) for index in (0, 1))
    runtime = interpreted.capture(
        lean_root() / "W8A8CorrespondenceMain.lean", required_module="MerlinVerify.Driver.W8A8Correspondence"
    )
    proofs = {root: runtime.axioms(root, module) for root, module in evidence.ROOTS.items()}
    if any(not set(axioms) <= {"propext", "Classical.choice", "Quot.sound"} for axioms in proofs.values()):
        raise ValueError("correspondence theorem has unapproved axioms")
    arguments = [programs[0]["path"], programs[1]["path"], str(index_bits), scope]
    arguments.extend(program["path"] for program in programs[2:])
    process = runtime.run(arguments, timeout_seconds=timeout_seconds)
    (work / "driver.stdout").write_text(process.stdout)
    (work / "driver.stderr").write_text(process.stderr)
    (work / "driver-process.json").write_text(
        json.dumps({"arguments": arguments, "returncode": process.returncode}) + "\n"
    )
    if process.returncode != 1:
        raise ValueError(f"correspondence driver failed (exit {process.returncode}): {process.stderr[-300:]}")
    actual = json.loads(process.stdout)
    evidence.validate(
        actual, before=before, after=after, suffix_count=len(stages) - 1, index_bits=index_bits, scope=scope
    )
    report_path = work / "correspondence-report.json"
    report_path.write_text(
        json.dumps(
            {
                "evidence": actual,
                "runtime": runtime.evidence,
                "runtime_sha256": runtime.digest,
                "proof_axioms": proofs,
                "candidate_sources": sources,
                "stages": [str(stage.resolve()) for stage in stages],
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
        "boundary": "w8a8-complete-correspondence-candidate",
        "checker": f"{evidence.MODULE}.check",
        "theorem": evidence.THEOREM,
        "domain": "all contexts, functions and inputs under W8A8SourceSemantics; success and policy admission separate",
        "axioms": proofs[evidence.THEOREM],
        "functions": actual["source_functions"],
        "lean": {**preflight["lean"], "checker_sha256": runtime.digest},
        "cert": _file_ref(report_path),
        "verdict": "abstracted",
        "reason": (
            "complete target/reference outcomes checked; draft policy and successful execution remain separate"
            if actual["complete_correspondence_passed"]
            else "complete target/reference policy correspondence refused"
        ),
    }
    numerical_policy._fresh(
        [
            *sources,
            *programs,
            receipt["before"],
            receipt["after"],
            receipt["cert"],
            preflight["cert"],
            _file_ref(work / "structural/receipt.json"),
            _file_ref(m2m_python),
        ]
    )
    if sources != _sources() or _lean_block() != preflight["lean"]:
        raise ValueError("correspondence source inventory or structural checker changed")
    if _file_ref(MLIR_EXPORT)["sha256"] != preflight["exporter"]["executable_sha256"]:
        raise ValueError("authoritative exporter changed during correspondence checking")
    runtime.assert_current()
    return _write(destination, receipt)


def replay(receipt_path: Path, destination: Path, *, m2m_python: Path) -> dict[str, Any]:
    """Regenerate actual exports and checks; a rehashed report cannot certify itself."""
    if (destination / "receipt.json").exists():
        raise ValueError("correspondence replay requires a new destination")
    original = _file_ref(receipt_path)
    receipt = evidence.read(receipt_path)
    if (
        receipt.get("boundary") != "w8a8-complete-correspondence-candidate"
        or receipt.get("theorem") != evidence.THEOREM
        or receipt.get("verdict") != "abstracted"
    ):
        raise ValueError("receipt is not a draft complete-correspondence candidate")
    numerical_policy._fresh([receipt["cert"], receipt["before"], receipt["after"]])
    report = evidence.read(Path(receipt["cert"]["path"]))
    numerical_policy._fresh(
        [
            *report["candidate_sources"],
            *report["runtime"]["files"],
            *report["program_exports"],
            report["compiler_python"],
            report["structural_preflight"],
            report["structural_receipt"],
        ]
    )
    if not evidence.same(_sources(), report["candidate_sources"]) or not evidence.same(
        _file_ref(m2m_python), report["compiler_python"]
    ):
        raise ValueError("correspondence replay source inventory or compiler differs")
    fresh_receipt = check_w8a8_correspondence_draft(
        [Path(stage) for stage in report["stages"]],
        destination,
        m2m_python=m2m_python,
        scope=report["evidence"]["policy_scope"],
        index_bits=report["evidence"]["index_bits"],
        timeout_seconds=report["checker_timeout_seconds"],
    )
    if "cert" not in fresh_receipt:
        raise ValueError(f"correspondence replay failed: {fresh_receipt.get('reason')}")
    fresh = evidence.read(Path(fresh_receipt["cert"]["path"]))
    for key in ("evidence", "runtime", "runtime_sha256", "proof_axioms", "candidate_sources"):
        if not evidence.same(report[key], fresh[key]):
            raise ValueError(f"correspondence replay differs in {key}")
    for key in ("before", "after", "checker", "theorem", "domain", "axioms", "functions", "lean", "verdict", "reason"):
        if not evidence.same(receipt[key], fresh_receipt[key]):
            raise ValueError(f"correspondence replay receipt differs in {key}")
    if not evidence.same(
        [(item["sha256"], item["bytes"]) for item in report["program_exports"]],
        [(item["sha256"], item["bytes"]) for item in fresh["program_exports"]],
    ):
        raise ValueError("correspondence replay changed its actual program exports")
    numerical_policy._fresh([original])
    return fresh_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", type=Path, nargs="+")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--m2m-python", type=Path, required=True)
    parser.add_argument("--scope", choices=("captured", "all"), default="captured")
    parser.add_argument("--index-bits", type=int, default=64)
    args = parser.parse_args()
    result = check_w8a8_correspondence_draft(
        args.stages, args.destination, m2m_python=args.m2m_python, scope=args.scope, index_bits=args.index_bits
    )
    print(json.dumps(result, indent=2))
    return 1 if result["verdict"] == "abstracted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
