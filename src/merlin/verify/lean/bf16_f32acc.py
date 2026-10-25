"""Byte-bound receipts and replay for the f32-accumulation contract of ``lower_bf16_matmul_f32acc``.

The Lean checker validates the whole actual output against a reference derived from the source
alone (``merlin-lean check-bf16-f32acc``). Given further recorded stages it also checks the
downstream refinement suffix, so the last program refines the contract reference
(``check-bf16-f32acc-chain``). A segment ends before the next numerics-changing boundary, whose own
policy check composes at the byte-identical join. The contract is a review candidate written
after reading the pass, so every receipt verdict stays ``abstracted`` until independent review.
The ``cert`` report carries what the receipt schema has no field for: per-site licenses, the
candidate denominator, fusion premises, strict-domain qualification, each intermediate export and
the axioms of every contract theorem.

Replay re-snapshots the bound bytes, round-trips the exact recorded exports and reruns the
current checker. It reports stale bindings separately from the rechecked outcome, so a rebuilt
checker is visible rather than silently trusted.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from merlin.verify.lean.check import (
    LEDGER,
    _attempt,
    _check_exit,
    _exported_from,
    _exporter_block,
    _file_ref,
    _finish,
    _lean_block,
    _snapshot,
    _stale,
    _work,
    _write,
    lean_exe,
    theorem_axioms,
)
from merlin.verify.lean.export import round_trip_exported

POLICY = "bf16-f32acc-contract:review-candidate"
BOUNDARY = ("check-bf16-f32acc", "MerlinVerify.Check.Bf16F32Acc", "check_sound", "qualify_sound")
SEGMENT = ("check-bf16-f32acc-chain", "MerlinVerify.Check.Bf16F32AccChain", "check_sound", "qualify_sound")
CHECKER = f"{BOUNDARY[1]}.check"
THEOREM = f"{BOUNDARY[1]}.{BOUNDARY[2]}"
CONTRACT_THEOREMS = {
    "MerlinVerify.Check.Bf16F32AccExact.check_sound": "MerlinVerify.Check.Bf16F32AccExact",
    "MerlinVerify.Check.Bf16F32AccExact.defined_execution": "MerlinVerify.Check.Bf16F32AccExact",
    "MerlinVerify.Check.Bf16F32AccSites.check_sound": "MerlinVerify.Check.Bf16F32AccSites",
    "MerlinVerify.Check.Bf16F32AccSites.collectOps_coverage": "MerlinVerify.Check.Bf16F32AccSites",
    "MerlinVerify.Rules.Bf16F32AccNonvacuity.operand_domain_nonempty": "MerlinVerify.Rules.Bf16F32AccNonvacuity",
    "MerlinVerify.Rules.Bf16F32AccTensor.accumulate_sound": "MerlinVerify.Rules.Bf16F32AccTensor",
    "MerlinVerify.Rules.Bf16F32AccTensor.truncate_sound": "MerlinVerify.Rules.Bf16F32AccTensor",
    "MerlinVerify.Rules.Bf16F32AccTensor.zero_fill_sound": "MerlinVerify.Rules.Bf16F32AccTensor",
    "MerlinVerify.Rules.Bf16F32AccContract.accumulate_cell": "MerlinVerify.Rules.Bf16F32AccContract",
    "MerlinVerify.Rules.Bf16F32AccContract.contract_distinct": "MerlinVerify.Rules.Bf16F32AccContract",
    "MerlinVerify.Rules.Bf16F32AccSite.accumulating_op_sound": "MerlinVerify.Rules.Bf16F32AccSite",
    "MerlinVerify.Rules.Bf16F32AccSite.final_op_sound": "MerlinVerify.Rules.Bf16F32AccSite",
    "MerlinVerify.Rules.Bf16F32AccFusion.fused_step": "MerlinVerify.Rules.Bf16F32AccFusion",
}
ASSUMPTIONS = [
    *LEDGER,
    "f32-accumulation contract is a review candidate written after reading the pass (not barrier-authored)",
    "contract-flagged multiply-add fusion: covered by Bf16F32AccFusion.fused_step only when the target fuses "
    "every contract step or none, the operand extension is injective, and the source has no licensed "
    "operations (cert fusion_premises)",
]
BOUNDARY_DOMAIN = "every input and context, wherever the contract reference is modelled (conditional policy refinement)"
SEGMENT_DOMAIN = (
    "nonpoison inputs of the declared argument types; contexts with {}-bit index (conditional policy refinement)"
)


def _identity(chain: bool) -> tuple[str, str, str, str]:
    command, module, theorem, qualify = SEGMENT if chain else BOUNDARY
    return command, module, f"{module}.{theorem}", f"{module}.{qualify}"


def run_checker(programs: list[Path], *, index_bits: int, timeout_seconds: int) -> dict[str, Any]:
    """Run the Lean driver on a policy boundary, plus a suffix when more than two programs are given."""
    chain = len(programs) > 2
    command, module, theorem, _ = _identity(chain)
    paths = [str(p) for p in programs]
    args = [*paths[:2], str(index_bits), *paths[2:]]
    try:
        proc = subprocess.run(
            [str(lean_exe()), command, *args], capture_output=True, text=True, check=False, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"f32-accumulation checker timed out after {timeout_seconds} s; no claim") from exc
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"checker error: {proc.stderr.strip()[-500:]}")
    result = json.loads(proc.stdout)
    _check_exit(proc, result)
    identity = (result.get("checker"), result.get("theorem"), result.get("claim_type"), result.get("index_bits"))
    if identity != (f"{module}.check", theorem, "policy", index_bits):
        raise ValueError("checker returned a different obligation or input domain")
    if result.get("policy_reviewed") is not False or result.get("verdict") != "abstracted":
        raise ValueError("an unreviewed contract cannot produce a reviewed or verified result")
    if chain and len(result["suffix_boundaries"]) != len(programs) - 2:
        raise ValueError("checker did not cover every downstream boundary")
    return result


def boundary_of(result: dict[str, Any]) -> dict[str, Any]:
    """The policy-boundary part of a report, for either command."""
    return result.get("policy_boundary", result)


def matched(result: dict[str, Any]) -> bool:
    return result["segment_match"] if "segment_match" in result else result["conditional_match"]


def check_stage(
    stage: Path, destination: Path, *, m2m_python: Path, index_bits: int = 64, timeout_seconds: int = 1800
) -> dict[str, Any]:
    """Check one recorded boundary directory and publish ``destination/receipt.json``."""
    return check_stages(
        [stage], destination, m2m_python=m2m_python, index_bits=index_bits, timeout_seconds=timeout_seconds
    )


def check_stages(
    stages: list[Path],
    destination: Path,
    *,
    m2m_python: Path,
    index_bits: int = 64,
    handoff: Path | None = None,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Check the bf16 boundary followed by its recorded downstream stages, joined byte for byte."""
    return _attempt(
        destination,
        lambda: _check_stages(
            stages,
            destination,
            m2m_python=m2m_python,
            index_bits=index_bits,
            handoff=handoff,
            timeout_seconds=timeout_seconds,
        ),
    )


def _check_stages(
    stages: list[Path],
    destination: Path,
    *,
    m2m_python: Path,
    index_bits: int,
    handoff: Path | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    from merlin.verify.lean.check import _export_checked

    if not stages:
        raise ValueError("a receipt needs the bf16 policy boundary")
    if destination.resolve() in {stage.resolve() for stage in stages}:
        raise ValueError("receipt destination must differ from the recorded stages")
    metas = [json.loads((stage / "meta.json").read_text()) for stage in stages]
    if metas[0]["pass"] != "merlin.llvmlower.passes_xdsl.lower_bf16_matmul_f32acc":
        raise ValueError("the first stage is not the lower_bf16_matmul_f32acc boundary")
    chain = len(stages) > 1 or handoff is not None
    _, module, theorem, qualify = _identity(chain)
    work = _work(destination)
    links, programs = [], []
    for index, (stage, meta) in enumerate(zip(stages, metas, strict=True)):
        refs = {}
        for key in ("before", "after"):
            snapshot = work / f"{index}_{key}.mlir"
            refs[key] = _snapshot(stage / f"{key}.mlir", snapshot)
            if meta[f"{key}_sha256"] != refs[key]["sha256"]:
                raise ValueError(f"stage {meta['boundary']} {key} bytes do not match their recorded digest")
            if key == "after" or index == 0:
                programs.append(snapshot)
        links.append({"boundary": meta["boundary"], "pass": meta["pass"], **refs})
    for previous, following in zip(links, links[1:]):
        if any(previous["after"][k] != following["before"][k] for k in ("sha256", "bytes")):
            raise ValueError("the recorded stages do not form a byte-identical chain")
    handoff_binding = None
    if handoff is not None:
        record = _snapshot(handoff, work / "handoff.json")
        bound = json.loads((work / "handoff.json").read_text())
        subject = {key: bound[key] for key in ("path", "sha256", "bytes")}
        if bound.get("schema") != "merlin.lean.handoff.v1" or _file_ref(Path(subject["path"])) != subject:
            raise ValueError("handoff record is unsupported or its subject changed")
        if any(subject[k] != links[-1]["after"][k] for k in ("sha256", "bytes")):
            raise ValueError("the segment does not end at the recorded compiler handoff")
        handoff_binding = {"record": record, "subject": subject}
    receipt: dict[str, Any] = {
        "schema": "merlin.lean.receipt.v1",
        "boundary": metas[0]["boundary"] if not chain else f"{metas[0]['boundary']}..{metas[-1]['boundary']}",
        "claim_type": "policy",
        "policy": POLICY,
        "checker": f"{module}.check",
        "theorem": theorem,
        "before": links[0]["before"],
        "after": links[-1]["after"],
        "index_bits": index_bits,
        "domain": SEGMENT_DOMAIN.format(index_bits) if chain else BOUNDARY_DOMAIN,
        "exporter": _exporter_block(m2m_python),
        "lean": _lean_block(),
        "axioms": theorem_axioms(theorem, module),
        "verdict": "unmeasured",
        "assumptions": list(ASSUMPTIONS),
    }
    exports = [_export_checked(path, work / f"rt_{path.stem}", m2m_python=m2m_python) for path in programs]
    refusals = [why for _, why in exports if why is not None]
    receipt["exporter"]["round_trip"] = "different" if refusals else "identical"
    if refusals:
        receipt.update(verdict="abstracted", reason=refusals[0])
        return _write(destination, receipt)
    exported = [path for path, _ in exports]
    program_exports = [_file_ref(path) for path in exported]
    receipt["exports"] = {"before": program_exports[0], "after": program_exports[-1]}
    result = run_checker(exported, index_bits=index_bits, timeout_seconds=timeout_seconds)
    report = work / "contract-report.json"
    report.write_text(
        json.dumps(
            {
                **result,
                "boundary_links": links,
                "program_exports": program_exports,
                "handoff": handoff_binding,
                "qualification_axioms": theorem_axioms(qualify, module),
                "contract_theorem_axioms": {
                    name: theorem_axioms(name, owner) for name, owner in CONTRACT_THEOREMS.items()
                },
                "checker_timeout_seconds": timeout_seconds,
            },
            indent=2,
        )
        + "\n"
    )
    obligations = boundary_of(result)["function_obligations"]
    receipt.update(
        functions=[entry["function"] for entry in obligations],
        cert=_file_ref(report),
        verdict="abstracted",
        reason=result["reason"] if matched(result) else f"refused: {result['reason']}",
    )
    live = [*(link[key] for link in links for key in ("before", "after")), *program_exports, receipt["cert"]]
    if handoff_binding is not None:
        live += list(handoff_binding.values())
    for ref in live:
        if _file_ref(Path(ref["path"])) != ref:
            raise ValueError("a bound input or report changed during verification")
    return _finish(destination, receipt)


SUMMARY = (
    "conditional_match",
    "qualification_passed",
    "candidates",
    "sites",
    "function_obligations",
    "reason",
    "pass_boundary",
)


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = {key: boundary_of(result)[key] for key in SUMMARY}
    for key in ("segment_match", "segment_qualified", "suffix_boundaries"):
        if key in result:
            summary[key] = result[key]
    return summary


def replay(receipt_path: Path, *, m2m_python: Path, work: Path, timeout_seconds: int = 1800) -> dict[str, Any]:
    """Recheck a receipt's exact exports with the current checker; never trust its verdict string."""
    import jsonschema

    from merlin.common.paths import repo_root

    receipt = json.loads(receipt_path.read_text())
    schema = json.loads((repo_root() / "merlin/contract/schemas/lean_receipt.schema.json").read_text())
    jsonschema.validate(receipt, schema)
    recorded = json.loads(Path(receipt["cert"]["path"]).read_text())
    chain = "segment_match" in recorded
    _, module, theorem, _ = _identity(chain)
    identity = (receipt["checker"], receipt["theorem"], receipt["claim_type"], receipt["policy"])
    if identity != (f"{module}.check", theorem, "policy", POLICY):
        raise ValueError("receipt names another obligation")
    stale = _stale(receipt, "receipt")
    if receipt["exporter"] != {**_exporter_block(m2m_python), "round_trip": "identical"}:
        stale.append("receipt: exporter identity or round trip differs from the current exporter")
    if receipt["axioms"] != theorem_axioms(theorem, module):
        stale.append("receipt: axioms differ from the current audited theorem")
    links = recorded["boundary_links"]
    sources = [links[0]["before"], *(link["after"] for link in links)]
    exports = recorded["program_exports"]
    if exports[0] != receipt["exports"]["before"] or exports[-1] != receipt["exports"]["after"]:
        raise ValueError("receipt exports differ from the report's program exports")
    work.mkdir(parents=True, exist_ok=False)
    programs = []
    for index, (source_ref, export_ref) in enumerate(zip(sources, exports, strict=True)):
        source, exported = work / f"{index}.mlir", work / f"{index}.json"
        for ref, snapshot in ((source_ref, source), (export_ref, exported)):
            if _snapshot(Path(ref["path"]), snapshot) != ref:
                raise ValueError(f"receipt program {index} bytes changed or are missing")
        if not _exported_from(exported, source):
            raise ValueError(f"program {index} export does not name its bound source")
        trip = round_trip_exported(
            source, exported, work / f"rt_{index}", lean_exe=lean_exe(), backend="mlir", m2m_python=m2m_python
        )
        if trip.status != "identical":
            raise ValueError(f"program {index} round trip: {trip.status}: {trip.detail[:300]}")
        programs.append(exported)
    rechecked = run_checker(programs, index_bits=receipt["index_bits"], timeout_seconds=timeout_seconds)
    before, after = _summary(recorded), _summary(rechecked)
    differences = [key for key in before if before[key] != after.get(key)]
    return {
        "receipt": str(receipt_path.resolve()),
        "fresh": not stale,
        "stale": stale,
        "rechecked": after,
        "matches_recorded": not differences,
        "differences": differences,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    from merlin.llvmlower.toolchain import compiler_python

    parser = argparse.ArgumentParser(prog="python -m merlin.verify.lean.bf16_f32acc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check the bf16 boundary (and downstream stages) and write receipt.json")
    check.add_argument("destination", type=Path)
    check.add_argument("stages", type=Path, nargs="+", help="the bf16 stage directory, then downstream stages")
    check.add_argument("--handoff", type=Path, default=None, help="require the segment to end at this handoff")
    again = sub.add_parser("replay", help="recheck a receipt's bound exports with the current checker")
    again.add_argument("receipt", type=Path)
    again.add_argument("work", type=Path)
    for each in (check, again):
        each.add_argument("--m2m-python", type=Path, default=None)
        each.add_argument("--timeout", type=int, default=1800)
    check.add_argument("--index-bits", type=int, default=64)
    args = parser.parse_args(argv)
    m2m = args.m2m_python or compiler_python()
    if args.command == "check":
        receipt = check_stages(
            args.stages,
            args.destination,
            m2m_python=m2m,
            index_bits=args.index_bits,
            handoff=args.handoff,
            timeout_seconds=args.timeout,
        )
        print(json.dumps({k: receipt.get(k) for k in ("boundary", "verdict", "reason")}, indent=2))
        return 0 if receipt.get("exports") else 1
    outcome = replay(args.receipt, m2m_python=m2m, work=args.work, timeout_seconds=args.timeout)
    print(json.dumps(outcome, indent=2))
    return 0 if outcome["matches_recorded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
