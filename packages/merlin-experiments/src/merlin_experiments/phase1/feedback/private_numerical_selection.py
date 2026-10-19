"""Explicit private full-chain criterion; legacy CPU-host execution stays separate."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import yaml

from merlin.compile.model_execution_inputs import file_sha256

SCHEMA = "merlin.phase1.private_numerical_selection.v1"
RESULT = "merlin.phase1.private_quantized_chain_execution.v1"
MODE = "selected_quantized_full_chain"
POLICY = {"atol": 2e-6, "rtol": 1e-5, "equal_nan": False, "finite_required": True}


def _ordinary(value, *, exists=True):
    if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("private numerical selection requires absolute ordinary paths")
    path = Path(value)
    if any(p.is_symlink() for p in (path, *path.parents)) or (exists and not path.is_file()):
        raise ValueError("private numerical input is missing or indirect")
    return path


def read(spec):
    """Absence preserves the original descriptor's CPU-host criterion."""
    document = yaml.safe_load(Path(spec).read_bytes())
    if not isinstance(document, dict):
        raise ValueError("private numerical selection requires a private specification mapping")
    selected = document.get("numerical_qualification")
    if selected is None:
        return None
    fields = {
        "schema",
        "mode",
        "config",
        "static_root",
        "models",
        "reference_manifest",
        "reference_sums",
        "comparison",
        "output_scope",
        "float_policy",
        "kernel_exact",
        "deadline_unix",
    }
    if (
        not isinstance(selected, dict)
        or set(selected) != fields
        or selected["schema"] != SCHEMA
        or selected["mode"] != MODE
        or selected["comparison"] != "quantized-reference"
        or selected["output_scope"] != "all_declared"
        or selected["float_policy"] != POLICY
        or selected["float_policy"].get("equal_nan") is not False
        or selected["float_policy"].get("finite_required") is not True
        or selected["kernel_exact"] is not True
        or type(selected["deadline_unix"]) not in (int, float)
        or not math.isfinite(selected["deadline_unix"])
        or selected["deadline_unix"] <= 0
    ):
        raise ValueError("private full-chain numerical criterion is malformed or incomplete")
    _ordinary(selected["config"])
    _ordinary(selected["static_root"], exists=False)
    for key in ("models", "reference_manifest", "reference_sums"):
        pin = selected[key]
        if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
            raise ValueError("private numerical input requires an exact file pin")
        if file_sha256(_ordinary(pin["path"])) != pin["sha256"]:
            raise ValueError("selected numerical reference bytes changed")
    return selected


def _wait(path, deadline):
    while not path.exists():
        if time.time() >= deadline:
            raise ValueError("selected private qualification deadline expired before evidence completed")
        time.sleep(min(1, max(0, deadline - time.time())))
    return _ordinary(str(path))


def _static(selected, *, run_dir, descriptor, candidate_sha256, wait=False):
    from . import private_model_chain

    path = Path(selected["static_root"]) / "static-index.json"
    path = _wait(path, selected["deadline_unix"]) if wait else _ordinary(str(path))
    raw = path.read_bytes()
    index = json.loads(raw)
    if (
        not isinstance(index, dict)
        or set(index) != {"schema", "run_dir", "records"}
        or index["schema"] != "merlin.phase1.native_static_chain_index.v1"
        or index["run_dir"] != str(Path(run_dir).absolute())
        or not isinstance(index["records"], dict)
    ):
        raise ValueError("private static index is not its selected original native owner")
    result = private_model_chain.join(
        index["records"], run_dir=run_dir, descriptor=descriptor, candidate_sha256=candidate_sha256
    )
    if path.read_bytes() != raw:
        raise ValueError("private static index changed during canonical join")
    return result


def consume(*, run_dir, descriptor, submission, private_spec_path):
    """Reopen actual canonical producers, then the existing full-chain verifier."""
    from merlin.compile.model_execution_inputs import strict_tree_sha256

    from . import private_chain_receipt

    selected = read(private_spec_path)
    if selected is None:
        raise ValueError("no selected private full-chain criterion")
    root, deadline = Path(selected["static_root"]), selected["deadline_unix"]
    candidate = strict_tree_sha256(submission)["sha256"]
    descriptor = _ordinary(str(Path(descriptor).absolute()))
    static = _static(selected, run_dir=run_dir, descriptor=descriptor, candidate_sha256=candidate, wait=True)
    receipt_path = _wait(root / "full-chain-qualification.json", deadline)
    proof = private_chain_receipt.verify(
        receipt_path, run_dir=run_dir, private_spec_path=private_spec_path, candidate_sha256=candidate
    )
    if read(private_spec_path) != selected:
        raise ValueError("selected private qualification inputs changed during intake")
    numerical = {
        "schema": RESULT,
        "mode": MODE,
        "passed": True,
        "target": static["target"],
        "candidate_tree_sha256": candidate,
        "private_spec_path": str(Path(private_spec_path).absolute()),
        "private_spec_sha256": file_sha256(private_spec_path),
        "run_dir": str(Path(run_dir).absolute()),
        "receipt_path": str(receipt_path),
        "descriptor": {"path": str(descriptor), "sha256": file_sha256(descriptor)},
        "numerical_proof": proof,
        "host_execution": {"selected": False, "status": "not_selected"},
        "scope": "canonical native static proof AND selected sealed quantized full chain; no legacy CPU-twin claim",
    }
    if (
        proof.get("schema") != "merlin.phase1.selected_quantized_full_chain.v1"
        or proof.get("passed") is not True
        or proof.get("candidate_tree_sha256") != candidate
        or proof.get("private_spec_sha256") != numerical["private_spec_sha256"]
        or set(proof.get("models", {})) != set(static["required_programs"])
    ):
        raise ValueError("private full-chain numerical proof did not revalidate")
    return static, numerical


def complete(record, *, static, gate, candidate_sha256):
    """No caller-supplied pass flag substitutes for the canonical receipt verifier."""
    from . import private_chain_receipt

    try:
        if not isinstance(static, dict) or not isinstance(record, dict):
            return False
        spec = _ordinary(record["private_spec_path"])
        selected = read(spec)
        if (
            selected is None
            or record["schema"] != RESULT
            or record["mode"] != MODE
            or record["passed"] is not True
            or record["candidate_tree_sha256"] != candidate_sha256
            or record["private_spec_sha256"] != file_sha256(spec)
            or static.get("private_spec_sha256") != file_sha256(spec)
            or static.get("candidate_tree_sha256") != candidate_sha256
            or static.get("passed") is not True
            or record.get("target") != static.get("target")
            or static.get("required_programs") != {name: list(programs) for name, programs in gate["programs"].items()}
            or record["host_execution"] != {"selected": False, "status": "not_selected"}
            or record["receipt_path"] != str(Path(selected["static_root"]) / "full-chain-qualification.json")
        ):
            return False
        proof = private_chain_receipt.verify(
            _ordinary(record["receipt_path"]),
            run_dir=Path(record["run_dir"]),
            private_spec_path=spec,
            candidate_sha256=candidate_sha256,
        )
        descriptor = record["descriptor"]
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            return False
        path = _ordinary(descriptor["path"])
        if (
            file_sha256(path) != descriptor["sha256"]
            or _static(selected, run_dir=Path(record["run_dir"]), descriptor=path, candidate_sha256=candidate_sha256)
            != static
        ):
            return False
        return (
            proof == record["numerical_proof"]
            and proof.get("passed") is True
            and proof.get("schema") == "merlin.phase1.selected_quantized_full_chain.v1"
            and proof.get("candidate_tree_sha256") == candidate_sha256
            and proof.get("private_spec_sha256") == file_sha256(spec)
            and set(proof.get("models", {})) == set(gate["programs"])
        )
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
        return False
