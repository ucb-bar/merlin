"""Strict conditional chain receipts: exact originals, snapshots and live replay.

This separate mode combines direct composite checking with the supported common
primitive transport proof. It never issues the shared broker's live authority,
approves effect semantics or upgrades a model execution to source definedness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.common.paths import repo_root

from . import lean_graph_state as stable
from . import lean_graph_state_chain as direct
from . import lean_graph_state_receipt_proofs as proofs
from . import lean_graph_state_receipt_schema as schema
from .lean_graph_state_chain_kernel import prepare
from .lean_graph_state_chain_transport import check_composition
from .lean_graph_state_receipt_binding import STAGE_NAMES, check_binding


def _originals(boundaries):
    if (
        type(boundaries) not in (tuple, list)
        or len(boundaries) != 2
        or any(type(edge) not in (tuple, list) or len(edge) != 2 for edge in boundaries)
    ):
        raise ValueError("strict mode requires exactly two actual before/after boundaries")
    return [Path(path).resolve() for edge in boundaries for path in edge]


def _profiles():
    return [stable.prepare()["manifest"], prepare()["manifest"]]


def _sources():
    directory = repo_root() / "src/merlin/verify"
    new = [
        *directory.glob("lean_graph_state_chain*.py"),
        *directory.glob("lean_graph_state_receipt*.py"),
        *(directory / "lean/Merlin").glob("GraphStateChain*.lean"),
        *(directory / "lean/Merlin").glob("GraphStateReceipt*.lean"),
    ]
    base = [
        directory / "lean_graph_state.py",
        directory / "lean_graph_state_capture.py",
        directory / "lean/Merlin/GraphStateAudit.lean",
    ]
    base += [directory / f"lean/Merlin/{name}.lean" for name in stable.SOURCES]
    base += [repo_root() / "src/merlin/common/paths.py", repo_root() / "src/merlin/common/owned_process.py"]
    return [stable.file_ref(path) for path in sorted(set(new + base))]


def _copy(record, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(Path(record["path"]).read_bytes())
    identity = stable.file_ref(path)
    if any(identity[key] != record[key] for key in ("sha256", "bytes")):
        raise ValueError("actual original changed during exact snapshot retention")
    return identity


def _same_buffers(original, snapshot):
    if Path(original["path"]).read_bytes() != Path(snapshot["path"]).read_bytes():
        raise ValueError("actual original and unchanged checked snapshot byte buffers differ")


def issue(boundaries, destination, *, reference_root, contract, original_contract_identity):
    """Run the complete live check before recording its conditional evidence."""
    schema.digest(original_contract_identity)
    paths = _originals(boundaries)
    originals = [stable.file_ref(path) for path in paths]
    for key in ("sha256", "bytes"):
        if originals[1][key] != originals[2][key]:
            raise ValueError("strict actual middle-stage join differs")
    sources = _sources()
    selected_tool = stable.file_ref(stable.LEAN)
    if selected_tool["sha256"] != stable.LEAN_SHA256:
        raise ValueError("selected actual Lean executable differs from intake")
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    snapshots = [_copy(record, root / "snapshots" / name) for record, name in zip(originals, STAGE_NAMES, strict=True)]
    original_contract = stable.file_ref(contract)
    contract_snapshot = _copy(original_contract, root / "snapshots/contract.json")
    pairs = []
    for index, original in enumerate(sources):
        snapshot = _copy(original, root / "source-closure" / f"{index:03d}-{Path(original['path']).name}")
        pairs.append({"original": original, "snapshot": snapshot})
    snapshot_edges = [(snapshots[0]["path"], snapshots[1]["path"]), (snapshots[2]["path"], snapshots[3]["path"])]
    direct_report = direct.check_chain(
        snapshot_edges,
        root / "checks/direct",
        reference_root=reference_root,
        contract=contract_snapshot["path"],
        original_contract_sha256=original_contract_identity,
    )
    if direct_report["status"] != "graph_model_checked":
        raise ValueError("actual complete chain was refused or unavailable")
    for index in range(2):
        stage = json.loads((root / f"checks/direct/stage-{index:03d}/report.json").read_text())
        if type(stage["actual_exchanges"]) is not int or stage["actual_exchanges"] <= 0:
            raise ValueError("strict composed-positive mode requires both actual stages to change order")
    independent = check_composition(
        snapshot_edges, root / "checks/independent", reference_root=reference_root, contract=contract_snapshot["path"]
    )
    if independent["status"] != "kernel_model_checked":
        raise ValueError("supported common-primitive semantic composition was refused or unavailable")
    queries = [folder / f"{module}.lean" for folder, module in proofs.query_roster(root, include_binding=False)]
    binding = check_binding(root, paths, contract, pairs, queries)
    if binding["status"] != "kernel_model_checked":
        raise ValueError("exact original/snapshot kernel binding was unavailable")
    receipt = {
        "schema": schema.SCHEMA,
        "version": 1,
        "mode": schema.MODE,
        "producer_revision": stable.REFERENCE_REVISION,
        "originals": originals,
        "snapshots": snapshots,
        "contract": original_contract,
        "contract_snapshot": contract_snapshot,
        "original_contract_identity": original_contract_identity,
        "selected_tool": selected_tool,
        "selected_profiles": _profiles(),
        "sources": pairs,
        "proof_inventory": proofs.inventory(root),
        "dependency_inventory": proofs.dependencies(root),
        "direct_report": stable.file_ref(root / "checks/direct/chain.json"),
        "independent_report": stable.file_ref(root / "checks/independent/report.json"),
        "binding_report": stable.file_ref(root / "checks/binding/report.json"),
        "claims": {
            "status": "conditional_model_checked",
            "public_acceptance_issued": False,
            "source_contract_approved": False,
            "native_fidelity_proved": False,
            "numerical_gates_discharged": False,
            "full_model_claim": False,
        },
    }
    schema.validate(receipt)
    if not stable.fresh(
        [
            *originals,
            *snapshots,
            original_contract,
            contract_snapshot,
            selected_tool,
            *sources,
            *(row["snapshot"] for row in pairs),
            *receipt["selected_profiles"],
            *receipt["proof_inventory"],
            *receipt["dependency_inventory"],
        ]
    ):
        raise ValueError("actual receipt input/code/proof/runtime identity changed during verification")
    for original, snapshot in zip(originals, snapshots, strict=True):
        _same_buffers(original, snapshot)
    _same_buffers(original_contract, contract_snapshot)
    stable._write(root / "receipt.json", receipt)
    return receipt


def _fixed_snapshot(record, expected):
    expected = Path(expected).absolute()
    if expected.resolve() != expected or Path(record["path"]) != expected or stable.file_ref(expected) != record:
        raise ValueError("snapshot path/redirection or actual saved bytes differ from fixed receipt binding")


def validate_live(receipt_path, boundaries, *, contract):
    """Validate actual bytes and closed inventories; never trust saved status alone."""
    root = Path(receipt_path).resolve().parent
    receipt, parsed_identity = schema.load_buffer(receipt_path)
    paths = _originals(boundaries)
    actuals = [stable.file_ref(path) for path in paths]
    if receipt["originals"] != actuals or receipt["contract"] != stable.file_ref(contract):
        raise ValueError("receipt originals do not identify the explicitly selected live actual inputs")
    for actual, snapshot, name in zip(actuals, receipt["snapshots"], STAGE_NAMES, strict=True):
        _fixed_snapshot(snapshot, root / "snapshots" / name)
        _same_buffers(actual, snapshot)
    _fixed_snapshot(receipt["contract_snapshot"], root / "snapshots/contract.json")
    _same_buffers(receipt["contract"], receipt["contract_snapshot"])
    if actuals[1]["sha256"] != actuals[2]["sha256"] or actuals[1]["bytes"] != actuals[2]["bytes"]:
        raise ValueError("actual middle join changed")
    if (
        receipt["selected_tool"] != stable.file_ref(stable.LEAN)
        or receipt["selected_tool"]["sha256"] != stable.LEAN_SHA256
    ):
        raise ValueError("receipt tool does not match selected actual executable bytes")
    if receipt["selected_profiles"] != _profiles():
        raise ValueError("receipt proof/tool source profile differs from current selected exact closure")
    current = _sources()
    if [row["original"] for row in receipt["sources"]] != current:
        raise ValueError("receipt source identity/closure differs from the current selected source")
    for index, pair in enumerate(receipt["sources"]):
        _fixed_snapshot(
            pair["snapshot"], root / "source-closure" / f"{index:03d}-{Path(pair['original']['path']).name}"
        )
        _same_buffers(pair["original"], pair["snapshot"])
    if receipt["proof_inventory"] != proofs.inventory(root):
        raise ValueError("proof inventory is incomplete or does not match real retained query/object/audit buffers")
    if receipt["dependency_inventory"] != proofs.dependencies(root):
        raise ValueError("actual tool/import/runtime/semantic dependency closure differs")
    expected_reports = [
        root / "checks/direct/chain.json",
        root / "checks/independent/report.json",
        root / "checks/binding/report.json",
    ]
    for key, expected in zip(("direct_report", "independent_report", "binding_report"), expected_reports, strict=True):
        if receipt[key] != stable.file_ref(expected):
            raise ValueError("actual checked relation report binding differs")
    selected_capture = json.loads((root / "checks/direct/complete/projection/capture.json").read_text())
    if receipt["original_contract_identity"] != selected_capture["contract_sha256"]:
        raise ValueError("recorded original semantic identity differs from its actual selected contract")
    if stable.file_ref(receipt_path) != parsed_identity:
        raise ValueError("receipt changed after parsing/hashing the selected exact input buffer")
    return receipt


def replay(receipt_path, boundaries, destination, *, reference_root, contract):
    """Fresh proof-object audit/recompilation AND full actual-model verification."""
    selected_receipt = stable.file_ref(receipt_path)
    old = validate_live(receipt_path, boundaries, contract=contract)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    reproduced = proofs.reproduce(Path(receipt_path).resolve().parent, root / "retained-proof-reproduction")
    fresh = issue(
        boundaries,
        root / "fresh",
        reference_root=reference_root,
        contract=contract,
        original_contract_identity=old["original_contract_identity"],
    )
    old_queries = proofs.query_roster(Path(receipt_path).resolve().parent)
    new_queries = proofs.query_roster(root / "fresh")
    for (before, module), (after, other) in zip(old_queries, new_queries, strict=True):
        if module != other or (before / f"{module}.lean").read_bytes() != (after / f"{module}.lean").read_bytes():
            raise ValueError("fresh actual query differs from retained receipt claim")
        if (before / f"{module}.olean").read_bytes() != (after / f"{module}.olean").read_bytes():
            raise ValueError("fresh kernel proof object differs from retained exact claim")
    if not stable.fresh(
        [
            selected_receipt,
            *old["originals"],
            *old["snapshots"],
            *old["proof_inventory"],
            *old["dependency_inventory"],
            old["contract"],
            old["contract_snapshot"],
        ]
    ):
        raise ValueError("selected original/receipt/snapshot/proof/import identity changed across fresh replay")
    result = {
        "schema": "merlin.graph-state-chain.strict-replay.v1",
        "original": selected_receipt,
        "fresh": stable.file_ref(root / "fresh/receipt.json"),
        "same_claim": True,
        "reproduced_proofs": len(reproduced),
        "claims": fresh["claims"],
    }
    stable._write(root / "replay.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("issue", "replay"))
    parser.add_argument("--stages", nargs=4, required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--original-contract-identity")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    boundaries = [(args.stages[0], args.stages[1]), (args.stages[2], args.stages[3])]
    if args.mode == "issue":
        result = issue(
            boundaries,
            args.destination,
            reference_root=args.reference,
            contract=args.contract,
            original_contract_identity=args.original_contract_identity,
        )
        print(json.dumps(result["claims"]))
    else:
        result = replay(
            args.receipt, boundaries, args.destination, reference_root=args.reference, contract=args.contract
        )
        print(json.dumps({"same_claim": result["same_claim"], "claims": result["claims"]}))


if __name__ == "__main__":
    main()
