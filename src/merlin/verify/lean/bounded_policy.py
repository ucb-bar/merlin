"""Byte-bound bounded W8A8 draft receipts through an independently pinned Lean entry point.

Bounds are input-domain restrictions, never facts supplied by a callee name or a receipt's
verdict. This path always remains abstracted until an independently reviewed policy and
concrete numerical-law admission have their own locked verification route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from merlin.verify.lean import bounded_numerics, interpreted, policy
from merlin.verify.lean.check import _file_ref, _function_attempt, _lean_block, _snapshot, _work, _write, lean_root
from merlin.verify.lean.export import MLIR_EXPORT


def check_w8a8_bounded_chain_draft(
    stages: list[Path],
    destination: Path,
    *,
    input_contracts: Path,
    m2m_python: Path,
    scope: str = "captured",
    index_bits: int = 64,
    handoff: Path | None = None,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Recheck exact programs and declared bounds; never accept an earlier verdict as a premise."""
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
        raise ValueError("bounded policy destination must differ from the nonempty input stage list")
    work = _work(destination)
    snapshot = work / "input-contracts.json"
    original_contract = _snapshot(input_contracts, snapshot)
    contract_snapshot = _file_ref(snapshot)
    bounds = bounded_numerics.decode(snapshot.read_text())

    # Reuse authoritative round trips, actual byte-identical boundary joins and handoff
    # checks. Its unrestricted-domain diagnostics remain separate from the bounded claim.
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
        raise ValueError(f"bounded W8A8 export preflight failed: {preflight.get('reason', 'missing evidence')}")
    preflight_ref = preflight["cert"]
    preflight_receipt = _file_ref(work / "structural/receipt.json")
    exporter = _file_ref(MLIR_EXPORT)
    if exporter["sha256"] != preflight["exporter"]["executable_sha256"]:
        raise ValueError("exporter changed after the bounded W8A8 round trip")
    preflight_report = json.loads(Path(preflight_ref["path"]).read_text())
    programs = preflight_report["program_exports"]
    roster = bounded_numerics.functions(json.loads(Path(programs[0]["path"]).read_text()))
    if set(bounds) != set(roster):
        raise ValueError("input contracts must name every actual source function exactly once")
    theorem = f"{bounded_numerics.MODULE}.check_sound"
    runtime = interpreted.capture(
        lean_root() / "W8A8BoundedMain.lean", required_module="MerlinVerify.Driver.W8A8Bounded"
    )
    if _lean_block() != preflight["lean"]:
        raise ValueError("preflight checker or configuration changed during verification")
    axioms = runtime.axioms(theorem, bounded_numerics.MODULE)
    if not set(axioms) <= {"propext", "Classical.choice", "Quot.sound"}:
        raise ValueError("bounded W8A8 theorem has unapproved axioms")
    proc = runtime.run(
        [
            programs[0]["path"],
            programs[1]["path"],
            str(snapshot),
            str(index_bits),
            scope,
            *[p["path"] for p in programs[2:]],
        ],
        timeout_seconds=timeout_seconds,
    )
    if proc.returncode != 1:
        raise ValueError(
            f"bounded W8A8 driver must refuse draft admission (exit {proc.returncode}): {proc.stderr[-300:]}"
        )
    numerical = json.loads(proc.stdout)
    bounded_numerics.validate(
        numerical, bounds=bounds, roster=roster, index_bits=index_bits, scope=scope, suffix_count=len(stages) - 1
    )
    numerical["axioms"] = axioms
    report = work / "bounded-policy-report.json"
    report.write_text(
        json.dumps(
            {
                "conditional_numerical": numerical,
                "input_contracts": {"original": original_contract, "snapshot": contract_snapshot, "entries": bounds},
                "runtime": runtime.evidence,
                "runtime_sha256": runtime.digest,
                "structural_preflight": preflight_ref,
                "structural_preflight_receipt": preflight_receipt,
                "boundary_links": preflight_report["boundary_links"],
                "program_exports": programs,
                "handoff": preflight_report["handoff"],
                "checker_timeout_seconds": timeout_seconds,
            },
            indent=2,
        )
        + "\n"
    )
    passed = numerical["conditional_initialized_policy_refinement"]
    receipt = {
        **preflight,
        "boundary": "w8a8-bounded-policy-suffix",
        "checker": f"{bounded_numerics.MODULE}.check",
        "theorem": theorem,
        "domain": f"{bounded_numerics.INPUT_DOMAIN}; exact declared bounds snapshot {contract_snapshot['sha256']}",
        "axioms": axioms,
        "functions": roster,
        "lean": {**preflight["lean"], "checker_sha256": runtime.digest},
        "cert": _file_ref(report),
        "verdict": "abstracted",
        "reason": (
            "conditional bounded W8A8 checks passed; "
            "independent policy/domain review and numerical-law admission remain required"
            if passed
            else "bounded W8A8 qualification or an actual suffix boundary was refused"
        ),
    }
    live = [
        original_contract,
        contract_snapshot,
        preflight_ref,
        preflight_receipt,
        exporter,
        receipt["cert"],
        *programs,
    ]
    for link in preflight_report["boundary_links"]:
        live.extend([link["before"], link["after"]])
    if preflight_report["handoff"] is not None:
        live.extend(preflight_report["handoff"].values())
    for reference in live:
        if _file_ref(Path(reference["path"])) != reference:
            raise ValueError("a bounded policy input, contract, handoff or report changed during verification")
    runtime.assert_current()
    if _file_ref(MLIR_EXPORT)["sha256"] != preflight["exporter"]["executable_sha256"]:
        raise ValueError("exporter changed during verification")
    if _lean_block() != preflight["lean"]:
        raise ValueError("preflight checker or configuration changed during verification")
    return _write(destination, receipt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", type=Path, nargs="+")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--input-contracts", type=Path, required=True)
    parser.add_argument("--m2m-python", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "captured"), default="captured")
    parser.add_argument("--index-bits", type=int, default=64)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    receipt = check_w8a8_bounded_chain_draft(
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
