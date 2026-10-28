"""Derive's evidence for a preselected capture taken from its issue-time attestation.

``capture issue --attestation-output`` already replays a capture in the sandbox and writes the
sealed attestation. Derive may use that file instead of replaying the capture a second time. This
never trusts the file's contents: every byte it binds is read again now and compared. The
selection must still hash to the independently supplied digest, the attestation file must hash to
the digest the operator recorded when it was written, the run must be owner-only, the pending
receipt must still bind the selected plan, policy, tools and the current issuer source, the
source, guest-root and output snapshots must still equal the receipt, and the attestation is
passed through the same admission gate (:func:`require_verified_execution`) that re-reads the
model, the materialized capture receipt and the pending receipt. What is not repeated is the
sandbox re-execution and the full inventory of the shared read-only runtime store; those were
verified when the attestation was issued.
"""

from __future__ import annotations

from pathlib import Path

from merlin.common import strict_json
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_static import _canonical_path, _digest, _file_digest

from .capture_execution_attestation import (
    PRESELECTED_REPLAY_SCHEMA,
    SCHEMA,
    AttestationNotVerified,
    require_verified_execution,
)
from .capture_selection import SCHEMA as SELECTION_SCHEMA
from .capture_selection import bound_selection

EVIDENCE_MODE = "issue_time_attestation"
EVIDENCE_STATUS = "verified_issue_time_attestation"


def attested_one(
    selection_path: Path, selected_sha256: str, model_path: Path, attestation_path: Path, attestation_sha256: str
) -> tuple[dict, dict, dict]:
    """Same triple as a replay: (evidence record, sealed attestation, loaded selection)."""
    selected, run, model, full_capture, pending, receipt = bound_selection(
        selection_path, expected_sha256=selected_sha256, model_path=model_path
    )
    if selected["schema"] != SELECTION_SCHEMA or full_capture:
        raise ValueError("attested capture evidence needs a v1 selection of a single model")
    if selected["issuer_source_sha256"] != _file_digest(Path(sealed_m2m.__file__)):
        raise ValueError("selected sealed M2M issuer source differs from the current issuer")
    attestation = _canonical_path(Path(attestation_path), exists=True)
    if attestation.is_symlink() or not attestation.is_file():
        raise AttestationNotVerified("capture attestation must be an ordinary file")
    raw = attestation.read_bytes()
    if _digest(raw) != attestation_sha256:
        raise AttestationNotVerified("capture attestation bytes differ from the recorded issue-time identity")
    try:
        document = strict_json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise AttestationNotVerified("capture attestation is unreadable") from exc
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise AttestationNotVerified("capture attestation has an unsupported schema")
    named, capture = document.get("selection") or {}, document.get("capture") or {}
    if (
        named.get("sha256") != selected_sha256
        or not isinstance(named.get("path"), str)
        or _canonical_path(Path(named["path"]), exists=True) != _canonical_path(Path(selection_path), exists=True)
        or capture.get("run_dir") != str(run)
        or capture.get("model_path") != str(model)
    ):
        raise AttestationNotVerified("capture attestation names a different selection, run or model")
    require_verified_execution(document)  # re-reads the model, both receipts and the pending receipt now
    source, runtime, output = run / "snapshots/source", run / "snapshots/guest-root", run / "capture"
    plan = receipt["plan"]
    sealed_m2m._validate_snapshots(
        source,
        runtime,
        output,
        schema=receipt["schema"],
        **({"runtime_reuse": plan["runtime_reuse"]} if "runtime_reuse" in plan else {}),
    )
    if (
        sealed_m2m._snapshot_tree(source),
        sealed_m2m._snapshot_tree(runtime, allow_dangling_symlinks=True),
        sealed_m2m._snapshot_tree(output),
    ) != (receipt.get("source"), receipt.get("guest_root"), receipt.get("output")):
        raise ValueError("sealed capture snapshot or output bytes differ from the issued receipt")
    if sealed_m2m._materialized_v2(output, source, output, plan) != receipt.get("materialized"):
        raise ValueError("sealed capture materialized receipt differs from the issued receipt")
    if _file_digest(source / "workload/loader.py") != plan["loader_sha256"]:
        raise ValueError("sealed capture loader differs from the selected loader")
    if _file_digest(pending) != capture["sealed_receipt_sha256"]:
        raise AttestationNotVerified("sealed receipt changed after the attestation")
    if _digest(Path(selection_path).read_bytes()) != selected_sha256:
        raise ValueError("capture selection changed during attestation check")
    evidence = {
        "schema": PRESELECTED_REPLAY_SCHEMA,
        "status": EVIDENCE_STATUS,
        "selection_sha256": selected_sha256,
        "sealed_receipt_sha256": capture["sealed_receipt_sha256"],
        "source_closure_verified": False,
        "phase0_admission": "not_granted",
        "model_sha256": capture["model_sha256"],
        "capture_receipt_sha256": capture["receipt_sha256"],
        "evidence_mode": EVIDENCE_MODE,
        "attestation_sha256": attestation_sha256,
    }
    return evidence, document, selected
