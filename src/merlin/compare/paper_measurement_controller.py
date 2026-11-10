"""Controller-owned paper measurement production, normalization and CLI."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .paper_measurement_build import _preflight_build_before_private_io, _retain_contract
from .paper_measurement_contract import _contract
from .paper_measurement_execution import _run_contract
from .paper_measurement_protocol import (
    _ISSUANCE_KIND,
    _ISSUED_RECEIPTS,
    CONTROLLER_ID,
    _bound_file,
    _canonical_sha,
    _ephemeral_issuance_key,
    _fsync,
    _issue_receipt,
    _sha_file,
    _write,
    issuance_fingerprint,
    retain_controller_sources,
)
from .paper_measurement_verification import verify_receipt


def produce_receipt(contract_path: str | Path, output_dir: str | Path) -> Path:
    """Rebuild, execute, observe, and retain one paper cell."""
    contract_path = Path(contract_path).resolve()
    contract, cell = _contract(contract_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _preflight_build_before_private_io(contract_path)
    retained_contract = _retain_contract(contract_path, output)
    raw = _run_contract(retained_contract)
    stdout, stderr = raw.pop("_captured_stdout"), raw.pop("_captured_stderr")
    capture = output / "capture"
    capture.mkdir()
    stdout_path, stderr_path = capture / "stdout", capture / "stderr"
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    _fsync(stdout_path)
    _fsync(stderr_path)
    raw["process_capture"] = {
        "stdout": {"path": stdout_path.relative_to(output).as_posix(), "sha256": _sha_file(stdout_path)},
        "stderr": {"path": stderr_path.relative_to(output).as_posix(), "sha256": _sha_file(stderr_path)},
    }
    raw_path = output / "raw_measurement.json"
    _write(raw_path, raw)
    raw_sha = _sha_file(raw_path)
    from aet.tracking.run_logger import EvalRunLogger

    aet_dir = output / "aet"
    logger = EvalRunLogger.start(
        project="merlin",
        suite="paper-cell",
        target=contract["target"],
        method="benchmark",
        seed=0,
        run_id=contract["run_id"],
        run_path=aet_dir,
        tracking_mode="local",
    )
    logger.log_event(
        "benchmark.completed",
        {"status": "complete", "command_sha256": raw["driver"]["command_sha256"], "raw_measurement_sha256": raw_sha},
    )
    run_path = logger.write_run_record(
        extra={
            "agentic": False,
            "token_accounting": "not_applicable",
            "study_sha256": contract["study_sha256"],
            "cell_sha256": _canonical_sha(cell),
            "raw_measurement_sha256": raw_sha,
            "benchmark_command_sha256": raw["driver"]["command_sha256"],
        }
    )
    logger.finish("ok" if raw["lifecycle"]["status"] == "pass" else "fail")
    logger.close()
    events_path = aet_dir / "logs" / "events.jsonl"
    for path in (run_path, events_path):
        _fsync(path)
    controller_source = output / "paper_measurement_controller.py"
    shutil.copy2(Path(__file__), controller_source)
    _fsync(controller_source)
    source_closure = retain_controller_sources(output)
    issuance_id = str(uuid.uuid4())
    receipt_path = output / "receipt.yaml"
    with _ephemeral_issuance_key(receipt_path, issuance_id) as (private_key, public_key):
        receipt = {
            "schema_version": 7,
            "kind": "paper_controller_measurement_receipt_v7",
            "status": "finalized",
            "controller_id": CONTROLLER_ID,
            "study_sha256": contract["study_sha256"],
            "run_id": contract["run_id"],
            "cell": cell,
            "command_sha256": raw["driver"]["command_sha256"],
            "controller_source": {"path": controller_source.name, "sha256": _sha_file(controller_source)},
            "controller_sources": source_closure,
            "contract": {"path": retained_contract.name, "sha256": _sha_file(retained_contract)},
            "raw_measurement": {"path": raw_path.name, "sha256": raw_sha},
            "issuance": {"id": issuance_id, "kind": _ISSUANCE_KIND, "public_key_sha256": _sha_file(public_key)},
            "aet_lifecycle": {
                "run_record": {"path": str(run_path.relative_to(output)), "sha256": _sha_file(run_path)},
                "events": {"path": str(events_path.relative_to(output)), "sha256": _sha_file(events_path)},
            },
        }
        receipt_path.write_text(yaml.safe_dump(receipt, sort_keys=True), encoding="utf-8")
        _fsync(receipt_path)
        _issue_receipt(
            receipt_path, receipt, raw_sha, _sha_file(retained_contract), private_key=private_key, public_key=public_key
        )
    _ISSUED_RECEIPTS[receipt_path.resolve()] = (
        _sha_file(receipt_path),
        _sha_file(raw_path),
        _sha_file(retained_contract),
    )
    return receipt_path


def normalize_receipt(receipt_path: str | Path) -> dict[str, Any]:
    receipt_path = Path(receipt_path).resolve()
    receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        raise ValueError("controller receipt must be a mapping")
    contract_path = _bound_file(receipt_path.parent, receipt.get("contract"), "measurement contract")
    raw_path = _bound_file(receipt_path.parent, receipt.get("raw_measurement"), "raw measurement")
    contract, _ = _contract(contract_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": 2,
        "run_id": contract["run_id"],
        **contract["result_identity"],
        "artifact_sha256": contract["artifact_sha256"],
        **{
            field: raw[field]
            for field in (
                "session",
                "lifecycle",
                "correctness",
                "quality",
                "timing",
                "memory",
                "execution",
                "provenance",
            )
        },
        "measurement_receipt": {
            "path": str(receipt_path),
            "sha256": _sha_file(receipt_path),
            "aet_run_id": contract["run_id"],
            "command_sha256": receipt["command_sha256"],
        },
    }
    from .paper import validate_paper_result

    validate_paper_result(result)
    verify_receipt(receipt_path, expected_result=result, expected_study_sha256=contract["study_sha256"])
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m merlin.compare.paper_measurement_controller")
    actions = parser.add_subparsers(dest="action", required=True)
    produce = actions.add_parser("produce")
    produce.add_argument("contract")
    produce.add_argument("output_dir")
    produce_result = actions.add_parser("produce-result")
    produce_result.add_argument("contract")
    produce_result.add_argument("output_dir")
    produce_result.add_argument("result")
    fingerprint = actions.add_parser("issuance-fingerprint")
    fingerprint.add_argument("receipt")
    arguments = parser.parse_args(argv)
    if arguments.action == "issuance-fingerprint":
        print(issuance_fingerprint(arguments.receipt))
        return 0
    if arguments.action == "produce":
        print(produce_receipt(arguments.contract, arguments.output_dir))
        return 0
    receipt = produce_receipt(arguments.contract, arguments.output_dir)
    result = normalize_receipt(receipt)
    Path(arguments.result).write_text(yaml.safe_dump(result, sort_keys=True), encoding="utf-8")
    print(arguments.result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONTROLLER_ID",
    "issuance_fingerprint",
    "main",
    "normalize_receipt",
    "produce_receipt",
    "verify_receipt",
]
