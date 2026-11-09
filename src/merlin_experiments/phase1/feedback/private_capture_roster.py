"""Exact complete-program roster checks for operator-private captures."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from merlin.runtime.modelinputs import file_sha256


def captured_input_provenance(
    programs: Sequence[tuple[str, Path]], expected: Mapping[str, bool | None]
) -> dict[str, dict]:
    """Report the attested loader's input claim, without its private file path."""
    result = {}
    for name, stage in programs:
        meta_path = stage / "meta.json"
        if meta_path.is_symlink() or not meta_path.is_file():
            raise ValueError(f"{name} has no ordinary capture input-provenance record")
        meta = json.loads(meta_path.read_bytes())
        if not isinstance(meta, Mapping):
            raise ValueError(f"{name} has malformed capture input provenance")
        declared = meta.get("loader_provenance")
        if meta.get("loader_provenance_status") != "declared" or not isinstance(declared, Mapping):
            raise ValueError(f"{name} has no declared loader input provenance")
        paper_ready = meta.get("loader_paper_ready")
        synthetic_fields = [declared[key] for key in ("synthetic_inputs", "synthetic_tokens") if key in declared]
        if len(synthetic_fields) > 1 and synthetic_fields[0] is not synthetic_fields[1]:
            raise ValueError(f"{name} has conflicting synthetic-input declarations")
        synthetic = synthetic_fields[0] if synthetic_fields else None
        if paper_ready is not None and type(paper_ready) is not bool:
            raise ValueError(f"{name} has a malformed paper-readiness declaration")
        if synthetic is not None and type(synthetic) is not bool:
            raise ValueError(f"{name} has a malformed synthetic-input declaration")
        observed = {"paper_ready": paper_ready, "synthetic_inputs": synthetic}
        if any(observed.get(key) is not value for key, value in expected.items()):
            raise ValueError(f"{name} input provenance differs from the selected complete-model scope")
        input_source = declared.get("input_source", declared.get("token_source"))
        input_sha256 = declared.get("input_sha256", declared.get("token_sha256"))
        result[name] = {
            **observed,
            "input_source": input_source if isinstance(input_source, str) else None,
            "input_sha256": input_sha256 if isinstance(input_sha256, str) else None,
            "meta_sha256": file_sha256(meta_path),
            "scope": "input provenance only; no paper accuracy or full-model numerical result",
        }
    return result


def captured_programs(capture: Path, expected: Sequence[str]) -> list[tuple[str, Path]]:
    """Use every program in the attested root session, never an operator-picked slice."""
    # A complete single-network capture may also carry a version-1 execution
    # session contract (for example an image stream). Only a root session
    # receipt identifies the version-2 multi-program capture protocol.
    if tuple(expected) == ("model",) and not (capture / "session-receipt.json").exists():
        if (capture / "stages").exists():
            raise ValueError("single-network declaration received an unbound stage directory")
        if any(
            path.is_symlink() or not path.is_file()
            for path in (capture / "model.mlir", capture / "capture_receipt.json")
        ):
            raise ValueError("single-network declaration has no ordinary source model and receipt")
        contract_path = capture / "session_contract.yaml"
        if contract_path.exists():
            if contract_path.is_symlink() or not contract_path.is_file():
                raise ValueError("single-network execution contract is indirect or malformed")
            contract = yaml.safe_load(contract_path.read_bytes())
            if not isinstance(contract, Mapping) or contract.get("version") != 1:
                raise ValueError("single-network execution contract is not version 1")
        return [("model", capture)]
    contract_path = capture / "session_contract.yaml"
    receipt_path = capture / "session-receipt.json"
    if any(path.is_symlink() or not path.is_file() for path in (contract_path, receipt_path)):
        raise ValueError("complete multi-program capture has no ordinary root session contract/receipt")
    contract = yaml.safe_load(contract_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    if not isinstance(contract, Mapping) or not isinstance(receipt, Mapping):
        raise ValueError("complete session contract/receipt is malformed")
    names = list(expected)
    programs = contract.get("programs")
    observed = receipt.get("programs")
    if (
        contract.get("version") != 2
        or contract.get("stages") != names
        or not isinstance(programs, list)
        or not isinstance(observed, list)
        or receipt.get("schema") != "merlin.model_session_capture.v1"
        or receipt.get("session_contract_sha256") != file_sha256(contract_path)
        or len(programs) != len(names)
        or len(observed) != len(names)
        or (len(names) > 1 and not contract.get("bindings"))
    ):
        raise ValueError("root session does not bind the declared complete program roster")
    stage_root = capture / "stages"
    if (
        stage_root.is_symlink()
        or not stage_root.is_dir()
        or sorted(p.name for p in stage_root.iterdir()) != sorted(names)
    ):
        raise ValueError("capture contains missing or extra session stage directories")
    result = []
    for name, entry, attested in zip(names, programs, observed, strict=True):
        stage = stage_root / name
        if (
            not isinstance(entry, Mapping)
            or entry.get("name") != name
            or entry.get("bundle") != f"stages/{name}"
            or not isinstance(attested, Mapping)
            or attested.get("name") != name
            or attested.get("ok") is not True
            or attested.get("opaque") != 0
            or attested.get("receipt_sha256") != file_sha256(stage / "capture_receipt.json")
            or stage.is_symlink()
            or not stage.is_dir()
        ):
            raise ValueError(f"session stage {name} is unverified, opaque, or not contract-bound")
        result.append((name, stage))
    return result
