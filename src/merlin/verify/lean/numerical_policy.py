"""Replayable W8A8 numerical-candidate checks and separately certified source prefixes.

The finite/nonzero-f32 domain is a stronger, provisional domain. Arithmetic agreement and the
existing ten numerical laws remain premises; every receipt remains abstracted, including when
the complete conditional checker accepts. Historical bounded receipts are left unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from merlin.verify.lean import bounded_numerics, interpreted, numerical_evidence, policy
from merlin.verify.lean.check import _file_ref, _function_attempt, _lean_block, _snapshot, _work, _write, lean_root
from merlin.verify.lean.export import MLIR_EXPORT

_ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def _sources() -> list[dict[str, Any]]:
    """Bind candidate sources as well as the actual interpreted proof-artifact capsule."""
    root = lean_root()
    python = Path(__file__).resolve().parent
    files = {
        *root.glob("*.lean"),
        *(root / "MerlinVerify").rglob("*.lean"),
        *(
            python / name
            for name in (
                "numerical_policy.py",
                "numerical_evidence.py",
                "bounded_numerics.py",
                "interpreted.py",
                "policy.py",
                "policy_numerics.py",
                "check.py",
                "export.py",
            )
        ),
    }
    return [_file_ref(path) for path in sorted(files)]


def _fresh(records: list[dict[str, Any]]) -> None:
    for record in records:
        if _file_ref(Path(record["path"])) != record:
            raise ValueError("a numerical candidate source, input, export or report changed")


def check_w8a8_numerical_chain_draft(
    stages: list[Path],
    destination: Path,
    *,
    input_contracts: Path,
    m2m_python: Path,
    scope: str = "captured",
    index_bits: int = 64,
    handoff: Path | None = None,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Check actual complete boundaries and record the longest independently rechecked prefixes."""
    if input_contracts.resolve() == (destination / "receipt.json").resolve():
        raise ValueError("input contracts cannot be the destination receipt file")
    return _function_attempt(
        destination,
        index_bits,
        lambda: _check(
            stages,
            destination,
            input_contracts=input_contracts,
            m2m_python=m2m_python,
            scope=scope,
            index_bits=index_bits,
            handoff=handoff,
            timeout_seconds=timeout_seconds,
        ),
    )


def _check(
    stages: list[Path],
    destination: Path,
    *,
    input_contracts: Path,
    m2m_python: Path,
    scope: str,
    index_bits: int,
    handoff: Path | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    if not stages or destination.resolve() in {stage.resolve() for stage in stages}:
        raise ValueError("numerical candidate destination must differ from the nonempty stage list")
    work = _work(destination)
    original_contract = _snapshot(input_contracts, work / "input-contracts.json")
    snapshot = _file_ref(work / "input-contracts.json")
    bounds = bounded_numerics.decode(Path(snapshot["path"]).read_text())
    sources = _sources()
    compiler = _file_ref(m2m_python)
    preflight = policy.check_w8a8_chain_draft(
        stages,
        work / "structural",
        m2m_python=m2m_python,
        scope=scope,
        index_bits=index_bits,
        handoff=handoff,
        timeout_seconds=timeout_seconds,
    )
    if "cert" not in preflight:
        raise ValueError(f"numerical candidate preflight failed: {preflight.get('reason', 'missing evidence')}")
    preflight_ref = preflight["cert"]
    preflight_receipt = _file_ref(work / "structural/receipt.json")
    preliminary = json.loads(Path(preflight_ref["path"]).read_text())
    programs = preliminary["program_exports"]
    before = json.loads(Path(programs[0]["path"]).read_text())
    roster = bounded_numerics.functions(before)
    if set(bounds) != set(roster):
        raise ValueError("input contracts must name every source function exactly once")
    runtime = interpreted.capture(
        lean_root() / "W8A8NumericalMain.lean", required_module="MerlinVerify.Driver.W8A8Numerical"
    )
    proofs = {theorem: runtime.axioms(theorem, module) for theorem, module in numerical_evidence.ROOTS.items()}
    if any(not set(axioms) <= _ALLOWED_AXIOMS for axioms in proofs.values()):
        raise ValueError("numerical candidate proof has unapproved axioms")
    arguments = [
        programs[0]["path"],
        programs[1]["path"],
        snapshot["path"],
        str(index_bits),
        scope,
        *[program["path"] for program in programs[2:]],
    ]
    proc = runtime.run(arguments, timeout_seconds=timeout_seconds)
    # Preserve raw process evidence before decoding: an execution failure cannot become acceptance.
    (work / "driver.stdout").write_text(proc.stdout)
    (work / "driver.stderr").write_text(proc.stderr)
    (work / "driver-process.json").write_text(json.dumps({"arguments": arguments, "returncode": proc.returncode}))
    if proc.returncode != 1:
        raise ValueError(f"numerical candidate driver failed (exit {proc.returncode}): {proc.stderr[-300:]}")
    numerical = json.loads(proc.stdout)
    numerical_evidence.validate(
        numerical,
        bounds=bounds,
        before=before,
        index_bits=index_bits,
        scope=scope,
        suffix_count=len(stages) - 1,
    )
    report = work / "numerical-policy-report.json"
    report.write_text(
        json.dumps(
            {
                "numerical": numerical,
                "proof_axioms": proofs,
                "runtime": runtime.evidence,
                "runtime_sha256": runtime.digest,
                "candidate_sources": sources,
                "input_contracts": {"original": original_contract, "snapshot": snapshot, "entries": bounds},
                "stages": [str(stage.resolve()) for stage in stages],
                "compiler_python": compiler,
                "structural_preflight": preflight_ref,
                "structural_preflight_receipt": preflight_receipt,
                "boundary_links": preliminary["boundary_links"],
                "program_exports": programs,
                "handoff": preliminary["handoff"],
                "checker_timeout_seconds": timeout_seconds,
            },
            indent=2,
        )
        + "\n"
    )
    receipt = {
        **preflight,
        "boundary": "w8a8-numerical-candidate-suffix",
        "checker": f"{numerical_evidence.MODULE}.check",
        "theorem": f"{numerical_evidence.MODULE}.check_sound",
        "domain": f"{numerical_evidence.INPUT_DOMAIN}; contracts {snapshot['sha256']}; explicit arithmetic agreement",
        "axioms": proofs[f"{numerical_evidence.MODULE}.check_sound"],
        "functions": roster,
        "lean": {**preflight["lean"], "checker_sha256": runtime.digest},
        "cert": _file_ref(report),
        "verdict": "abstracted",
        "reason": (
            "conditional candidate checks passed; policy/domain review and concrete numerical admission remain required"
            if numerical["conditional_initialized_policy_refinement"]
            else "complete candidate qualification refused; checked prefixes provide partial evidence"
        ),
    }
    live = [
        original_contract,
        snapshot,
        preflight_ref,
        preflight_receipt,
        receipt["cert"],
        receipt["before"],
        receipt["after"],
        *programs,
        compiler,
    ]
    for link in preliminary["boundary_links"]:
        live.extend([link["before"], link["after"]])
    if preliminary["handoff"] is not None:
        live.extend(preliminary["handoff"].values())
    _fresh([*sources, *live])
    if sources != _sources():
        raise ValueError("numerical candidate source inventory changed")
    if _file_ref(MLIR_EXPORT)["sha256"] != preflight["exporter"]["executable_sha256"]:
        raise ValueError("exporter changed during numerical verification")
    if _lean_block() != preflight["lean"]:
        raise ValueError("preflight checker or configuration changed during numerical verification")
    runtime.assert_current()
    return _write(destination, receipt)


def replay(receipt_path: Path, destination: Path, *, m2m_python: Path) -> dict[str, Any]:
    """Regenerate exports, audits and checks; a forged green report cannot certify its own replay."""
    if receipt_path.resolve() == (destination / "receipt.json").resolve():
        raise ValueError("replay requires a new destination receipt file")
    original = _file_ref(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("checker") != f"{numerical_evidence.MODULE}.check"
        or receipt.get("theorem") != f"{numerical_evidence.MODULE}.check_sound"
        or receipt.get("verdict") != "abstracted"
    ):
        raise ValueError("not a provisional numerical-candidate receipt")
    _fresh([original, receipt["cert"], receipt["before"], receipt["after"]])
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    expected_exports = {"before": report["program_exports"][0], "after": report["program_exports"][-1]}
    if (
        receipt["before"] != report["boundary_links"][0]["before"]
        or receipt["after"] != report["boundary_links"][-1]["after"]
        or receipt.get("exports") != expected_exports
    ):
        raise ValueError("receipt changed its actual program binding")
    runtime_digest = hashlib.sha256(
        json.dumps(report["runtime"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if runtime_digest != report["runtime_sha256"]:
        raise ValueError("receipt changed its numerical runtime inventory")
    references = [
        *report["candidate_sources"],
        *report["runtime"]["files"],
        *report["program_exports"],
        report["input_contracts"]["original"],
        report["input_contracts"]["snapshot"],
        report["structural_preflight"],
        report["structural_preflight_receipt"],
    ]
    for link in report["boundary_links"]:
        references.extend([link["before"], link["after"]])
    if report["handoff"] is not None:
        references.extend(report["handoff"].values())
    _fresh(references)
    for key in ("original", "snapshot"):
        decoded = bounded_numerics.decode(Path(report["input_contracts"][key]["path"]).read_text())
        if decoded != report["input_contracts"]["entries"]:
            raise ValueError("receipt changed its bound input contract")
    if report["compiler_python"] != _file_ref(m2m_python):
        raise ValueError("replay selected a different compiler interpreter")
    numerical = report["numerical"]
    expected_domain = (
        f"{numerical_evidence.INPUT_DOMAIN}; contracts {report['input_contracts']['snapshot']['sha256']}; "
        "explicit arithmetic agreement"
    )
    if (
        receipt.get("domain") != expected_domain
        or receipt["lean"]["checker_sha256"] != report["runtime_sha256"]
        or receipt.get("axioms") != report["proof_axioms"].get(receipt["theorem"])
    ):
        raise ValueError("receipt changed its numerical domain, runtime or theorem audit")
    before = json.loads(Path(report["program_exports"][0]["path"]).read_text())
    numerical_evidence.validate(
        numerical,
        bounds=report["input_contracts"]["entries"],
        before=before,
        index_bits=receipt["index_bits"],
        scope=numerical["policy_scope"],
        suffix_count=len(report["stages"]) - 1,
    )
    fresh = check_w8a8_numerical_chain_draft(
        [Path(stage) for stage in report["stages"]],
        destination,
        input_contracts=Path(report["input_contracts"]["original"]["path"]),
        m2m_python=m2m_python,
        scope=numerical["policy_scope"],
        index_bits=receipt["index_bits"],
        handoff=Path(report["handoff"]["record"]["path"]) if report["handoff"] else None,
        timeout_seconds=report["checker_timeout_seconds"],
    )
    if "cert" not in fresh:
        raise ValueError(f"numerical replay failed: {fresh.get('reason')}")
    actual = json.loads(Path(fresh["cert"]["path"]).read_text())
    if any(
        actual[key] != report[key]
        for key in ("numerical", "runtime", "runtime_sha256", "proof_axioms", "candidate_sources")
    ):
        raise ValueError("receipt does not match recomputed numerical evidence")
    identity_keys = (
        "schema",
        "boundary",
        "claim_type",
        "policy",
        "checker",
        "theorem",
        "index_bits",
        "domain",
        "before",
        "after",
        "exporter",
        "lean",
        "axioms",
        "verdict",
        "assumptions",
        "functions",
        "reason",
    )
    if any(fresh.get(key) != receipt.get(key) for key in identity_keys):
        raise ValueError("receipt identity does not match the recomputed numerical claim")
    _fresh([original, receipt["cert"], *references])
    return fresh


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", type=Path, nargs="+")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--input-contracts", type=Path, required=True)
    parser.add_argument("--m2m-python", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "captured"), default="captured")
    parser.add_argument("--index-bits", type=int, default=64)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    receipt = check_w8a8_numerical_chain_draft(
        args.stages,
        args.destination,
        input_contracts=args.input_contracts,
        m2m_python=args.m2m_python,
        scope=args.scope,
        index_bits=args.index_bits,
        handoff=args.handoff,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(receipt))
    return 1 if receipt["verdict"] == "abstracted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
