"""Command-line sealing for the narrow-epilogue work-order policy."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from . import narrow_epilogue_work_order as _core


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Seal one capability-proven narrow-epilogue Phase-2 catalog/work order")
    parser.add_argument("iteration", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("capability_evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iteration-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--capability-sha256", required=True)
    args = parser.parse_args(argv)
    for label, path in (("iteration", args.iteration), ("capability evidence", args.capability_evidence)):
        if (
            not path.is_absolute()
            or path.resolve() != path
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_mode & 0o222
        ):
            parser.error(f"{label} must be an absolute, read-only, non-symlink regular file")
    iteration_raw = args.iteration.read_bytes()
    capability_raw = args.capability_evidence.read_bytes()
    if not _core._pin(args.iteration_sha256) or _core._raw_sha256(iteration_raw) != args.iteration_sha256:
        parser.error("iteration raw SHA-256 pin does not match")
    if not _core._pin(args.capability_sha256) or _core._raw_sha256(capability_raw) != args.capability_sha256:
        parser.error("capability evidence raw SHA-256 pin does not match")
    try:
        record, capability = json.loads(iteration_raw), json.loads(capability_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        parser.error(f"iteration or capability evidence is not valid JSON: {exc}")
    try:
        documents = _core.build_narrow_epilogue_mechanism_documents(
            record,
            candidate=args.candidate,
            expected_candidate_sha256=args.candidate_sha256,
            iteration_record_sha256=args.iteration_sha256,
            capability_evidence=capability,
            capability_evidence_file_sha256=args.capability_sha256,
        )
        if _core.hash_tree(args.candidate)["sha256"] != args.candidate_sha256:
            raise ValueError("candidate changed while deriving the sealed work order")
        args.output.mkdir(mode=0o755, parents=False, exist_ok=False)
        names = {
            "compiler_edit_contract.json": documents["edit_contract"],
            "mechanism_catalog.json": documents["catalog"],
            "mechanism_work_order.json": documents["work_order"],
            "source_site_inventory.json": documents["inventory"],
        }
        artifacts = {
            name: {"sha256": _core._write_once(args.output / name, document)} for name, document in names.items()
        }
        receipt = {
            "schema": "sealed_narrow_epilogue_work_order_receipt_v1",
            "mechanism_id": _core.MECHANISM_ID,
            "candidate_sha256": args.candidate_sha256,
            "iteration_record_sha256": args.iteration_sha256,
            "capability_evidence_file_sha256": args.capability_sha256,
            "capability_evidence_sha256": capability.get("sha256"),
            "inventory_sha256": documents["inventory"]["sha256"],
            "catalog_sha256": documents["catalog"]["sha256"],
            "work_order_sha256": documents["work_order"]["sha256"],
            "artifacts": dict(artifacts),
        }
        receipt["sha256"] = _core._digest(receipt)
        artifacts["receipt.json"] = {"sha256": _core._write_once(args.output / "receipt.json", receipt)}
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    report = {
        "status": "ready_for_authoring",
        "mechanism_id": _core.MECHANISM_ID,
        "output": str(args.output),
        "eligible_counts": [len(member["eligible"]) for member in documents["inventory"]["members"]],
        "already_narrow_counts": [len(member["already_narrow"]) for member in documents["inventory"]["members"]],
        "missing_representation_counts": [
            len(member["missing_representation"]) for member in documents["inventory"]["members"]
        ],
        "missing_capability_counts": [
            len(member["missing_capability"]) for member in documents["inventory"]["members"]
        ],
        "residual_second_operand_counts": [
            len(member["residual_second_operand"]) for member in documents["inventory"]["members"]
        ],
        "float_or_unsupported_stage_counts": [
            len(member["float_or_unsupported_stage"]) for member in documents["inventory"]["members"]
        ],
        "uncaptured_counts": [len(member["uncaptured"]) for member in documents["inventory"]["members"]],
        "artifacts": artifacts,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
