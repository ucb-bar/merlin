"""Diagnostic evidence for the separate Phase 0 capture-execution boundary.

The model2MLIR receipt verifies materialized capture members. It does not prove
which source, checkpoint, Python dependencies, or ambient files the process read.
A diagnostic made after a run cannot upgrade that run to a sealed execution, even
if its bytes still match.

Exactly one issuer is admitted: Merlin's own sealed Model2MLIR CPU runner
(``capture_execution.sealed_m2m``, receipt schema ``merlin.sealed_m2m_cpu.v2``),
and only for a run that was PRESELECTED before it existed
(``phase0.capture_selection``) and then replayed in a fresh sandbox. Admitting
that runner is an operator policy decision: its receipt is unsigned and its
Python runtime closure is the copied venv rather than an independently pinned
dependency set; the decision accepts those residuals and records them in every
attestation it issues. Every admission re-reads the selection, the sealed
receipt, the model and the materialized receipt from disk and compares their
digests, so an edited attestation document cannot pass on its flags alone.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from merlin.common import strict_json
from merlin.targetgen.application_inventory import verify_capture_receipt

SCHEMA = "merlin.capture_execution_attestation.v1"
SCHEMA_V2 = "merlin.capture_execution_attestation.v2"
#: The one admitted issuer: the sealed Model2MLIR CPU runner's receipt schema.
SEALED_M2M_ISSUER = "merlin.sealed_m2m_cpu.v2"
SEALED_M2M_ISSUER_V3 = "merlin.sealed_m2m_cpu.v3"
_VERIFIED_ISSUERS: frozenset[str] = frozenset({SEALED_M2M_ISSUER, SEALED_M2M_ISSUER_V3})
PRESELECTED_REPLAY_SCHEMA = "merlin.phase0.preselected_capture_replay.v1"
SEALED_M2M_POLICY = {
    "decision": "operator policy decision 2026-10-01: admit the sealed Model2MLIR CPU runner as a verified issuer",
    "scope": "preselected, fresh sealed_m2m_cpu.v2 runs replayed in a fresh sandbox; no other issuer or schema",
    "accepted_residuals": [
        "the sealed M2M receipt is unsigned",
        "the Python runtime closure is the copied selected venv, not an independently pinned dependency set",
    ],
}
SEALED_M2M_POLICY_V3 = {
    "decision": "operator policy decision: admit only replayed, preselected sealed M2M v3 source and checkpoint bytes",
    "scope": "one complete saved CPU program or session with exact checkpoint, declared environment and program roster",
    "accepted_residuals": [
        "the sealed M2M receipt is unsigned",
        "the Python runtime closure is a copied selected venv rather than an independently pinned dependency set",
    ],
}
SEALED_M2M_ASSESSMENT_SCHEMA = "merlin.phase0.sealed_m2m_assessment.v2"
_REQUIRED_CONTROLS = (
    "fresh_private_source_snapshot",
    "complete_runtime_and_checkpoint_snapshot",
    "network_namespace_disabled",
    "ambient_checkout_home_and_cache_inaccessible",
    "source_and_runtime_mounted_read_only",
    "new_capture_output_directory",
    "post_execution_source_and_output_byte_verification",
)


class AttestationNotVerified(ValueError):
    """No supported fresh, sealed execution has earned source-closure admission."""


def _sha256(path: Path) -> tuple[int, str]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"source member is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)

    def identity(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    if identity(before) != identity(after):
        raise ValueError(f"source member changed while being inspected: {path}")
    return before.st_size, digest.hexdigest()


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError("selected source path must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"unsafe selected source path: {value!r}")
    return path


def _inventory(root: Path, selections: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Inventory exact selected members, including directory membership and empty dirs.

    This is only a diagnostic byte inventory. It does not discover Python imports
    or the files an earlier capture process actually opened.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"source root is absent or indirect: {root}")
    selected = [_relative(value) for value in selections]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("source selection must be nonempty and unique")
    if any(a != b and (a.is_relative_to(b) or b.is_relative_to(a)) for a in selected for b in selected):
        raise ValueError("selected source paths overlap")
    inventory: dict[str, dict[str, Any]] = {}

    def visit(path: Path, relative: str) -> None:
        if path.is_symlink():
            raise ValueError(f"selected source contains a symlink: {relative}")
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            children = sorted(path.iterdir(), key=lambda member: member.name)
            inventory[relative] = {"kind": "directory", "members": [member.name for member in children]}
            for child in children:
                visit(child, f"{relative}/{child.name}")
        elif stat.S_ISREG(mode):
            size, digest = _sha256(path)
            inventory[relative] = {"kind": "file", "bytes": size, "sha256": digest}
        else:
            raise ValueError(f"selected source contains a nonregular member: {relative}")

    for relative in selected:
        path = root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f"selected source contains a symlink: {relative}")
        if not path.exists():
            raise ValueError(f"selected source is absent: {relative}")
        visit(path, relative.as_posix())
    return dict(sorted(inventory.items()))


def diagnose_capture(
    capture_dir: Path,
    source_root: Path,
    selected_sources: Sequence[str],
) -> dict[str, Any]:
    """Describe current bytes without asserting an earlier execution was sealed."""
    capture_dir, source_root = Path(capture_dir).absolute(), Path(source_root).absolute()
    source_members = _inventory(source_root, selected_sources)
    source_payload = json.dumps(source_members, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    materialized = verify_capture_receipt(capture_dir / "model.mlir")
    return {
        "schema": SCHEMA,
        "status": "diagnostic_only",
        "source_closure_verified": False,
        "fresh_execution": False,
        "issuer": None,
        "capture": {"path": str(capture_dir), "materialized_receipt": materialized},
        "selected_source": {
            "root": str(source_root),
            "members": source_members,
            "inventory_sha256": hashlib.sha256(source_payload).hexdigest(),
        },
        "execution": {"observed": False, "controls_required": list(_REQUIRED_CONTROLS), "controls_verified": []},
        "blockers": [
            "selected current source bytes do not identify the source bytes read by the capture process",
            "no fresh private source/runtime/checkpoint snapshot was executed",
            "network and ambient filesystem isolation were not observed",
            "materialized capture verification does not establish source closure",
        ],
    }


def write_diagnostic(path: Path, document: Mapping[str, Any]) -> None:
    """Write once to a separate evidence path; never edit a capture or its receipt."""
    if document.get("schema") != SCHEMA or document.get("status") != "diagnostic_only":
        raise ValueError("only diagnostic attestations may be written by this module")
    if document.get("source_closure_verified") is not False or document.get("fresh_execution") is not False:
        raise ValueError("diagnostic attestation cannot assert verified execution")
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("attestation destination is indirect")
    capture = Path(str((document.get("capture") or {}).get("path", ""))).absolute()
    source = Path(str((document.get("selected_source") or {}).get("root", ""))).absolute()
    destination = path.absolute()
    if destination.is_relative_to(capture) or destination.is_relative_to(source):
        raise ValueError("diagnostic attestation must live outside the capture and source trees")
    raw = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def require_verified_execution(document: Mapping[str, Any]) -> None:
    """Admission gate: only the preselected sealed M2M runner, re-verified from disk.

    A model2MLIR receipt, an old capture, a diagnostic receipt, or edited JSON with
    ``source_closure_verified: true`` cannot pass this gate. The sealed M2M issuer
    passes only while the selection, sealed receipt, model and materialized receipt
    it names still carry the attested digests and bind the preselected plan.
    """
    if document.get("schema") not in {SCHEMA, SCHEMA_V2}:
        raise AttestationNotVerified("unsupported capture execution attestation schema")
    if (
        document.get("status") != "verified_sealed_execution"
        or document.get("source_closure_verified") is not True
        or document.get("fresh_execution") is not True
    ):
        raise AttestationNotVerified("capture has no verified fresh sealed execution")
    if document.get("issuer") not in _VERIFIED_ISSUERS:
        raise AttestationNotVerified("no supported Merlin sealed execution issuer has verified this capture")
    if document.get("issuer") == SEALED_M2M_ISSUER and document.get("schema") == SCHEMA:
        _require_sealed_m2m_bytes(document)
    elif document.get("issuer") == SEALED_M2M_ISSUER_V3 and document.get("schema") == SCHEMA_V2:
        _require_sealed_m2m_v3_bytes(document)
    else:
        raise AttestationNotVerified("sealed execution issuer and attestation schema do not match")


def sealed_m2m_tree_snapshot(root: Path) -> dict[str, Any]:
    """Use the sealed issuer's exact member/size/mode digest, not a compiler tree hash."""
    from merlin_experiments.capture_execution import sealed_m2m

    return sealed_m2m._snapshot_tree(Path(root))


def _is_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sealed_m2m_bindings(selection_path: Path, selection_sha256: str, model_path: Path) -> dict[str, Any]:
    """Re-read every byte the sealed M2M attestation names; raise on any difference."""
    from merlin_experiments.capture_execution import sealed_m2m
    from merlin_experiments.capture_execution.sealed_static import _canonical_path

    from .capture_selection import load

    try:
        selected = load(Path(selection_path), expected_sha256=selection_sha256)
        run = _canonical_path(Path(selected["run_dir"]), exists=True)
        model = _canonical_path(Path(model_path), exists=True)
    except (OSError, ValueError, KeyError) as exc:
        raise AttestationNotVerified(f"sealed M2M selection or capture is unreadable: {exc}") from exc
    if model != run / "capture/model.mlir" or not model.is_file():
        raise AttestationNotVerified("attested model is not the preselected sealed run's capture/model.mlir")
    pending, materialized = run / "sealed_m2m_pending.json", model.parent / "capture_receipt.json"
    if any(path.is_symlink() or not path.is_file() for path in (pending, materialized)):
        raise AttestationNotVerified("sealed M2M receipt or materialized capture receipt is absent or indirect")
    try:
        receipt = strict_json.loads(pending.read_bytes())
        digests = {
            name: _sha256(path)[1]
            for name, path in (
                ("sealed_receipt_sha256", pending),
                ("model_sha256", model),
                ("receipt_sha256", materialized),
            )
        }
    except (OSError, ValueError) as exc:
        raise AttestationNotVerified(f"sealed M2M evidence bytes cannot be read consistently: {exc}") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != sealed_m2m.SCHEMA
        or receipt.get("status") != "pending_replay"
        or receipt.get("capture_selection_sha256") != selection_sha256
        or receipt.get("plan") != selected.get("plan")
        or receipt.get("policy_sha256") != selected.get("sandbox_policy_sha256")
        or receipt.get("bwrap_sha256") != (selected.get("bwrap") or {}).get("sha256")
        or receipt.get("issuer_sha256") != selected.get("issuer_source_sha256")
    ):
        raise AttestationNotVerified("sealed M2M receipt does not bind the preselected plan, policy and tools")
    if verify_capture_receipt(model).get("status") != "verified_materialized":
        raise AttestationNotVerified("sealed M2M materialized capture receipt is unverified")
    return {"run_dir": str(run), "model_path": str(model), **digests}


def _require_sealed_m2m_bytes(document: Mapping[str, Any]) -> None:
    capture = document.get("capture") or {}
    selection = document.get("selection") or {}
    if (
        document.get("policy") != SEALED_M2M_POLICY
        or (document.get("replay") or {}).get("schema") != PRESELECTED_REPLAY_SCHEMA
        or (document.get("replay") or {}).get("status") != "verified_preselected_replay"
        or not _is_sha(selection.get("sha256"))
        or not isinstance(selection.get("path"), str)
        or not isinstance(capture.get("model_path"), str)
        or not all(_is_sha(capture.get(key)) for key in ("model_sha256", "receipt_sha256", "sealed_receipt_sha256"))
    ):
        raise AttestationNotVerified("sealed M2M attestation lacks its selection, replay or byte bindings")
    observed = _sealed_m2m_bindings(Path(selection["path"]), selection["sha256"], Path(capture["model_path"]))
    for key in ("model_sha256", "receipt_sha256", "sealed_receipt_sha256", "run_dir", "model_path"):
        if observed[key] != capture.get(key):
            raise AttestationNotVerified(f"sealed M2M attested {key} differs from the bytes on disk")


def attest_sealed_m2m(replay: Mapping[str, Any], *, selection_path: Path, model_path: Path) -> dict[str, Any]:
    """Issue the admitted attestation from one ``capture_selection.verify`` replay record.

    The replay must be the preselected-replay record for exactly these selection
    bytes, and its digests must equal the bytes on disk now. The returned document
    passes ``require_verified_execution`` only while those bytes stay unchanged.
    """
    if (
        not isinstance(replay, Mapping)
        or replay.get("schema") != PRESELECTED_REPLAY_SCHEMA
        or replay.get("status") != "verified_preselected_replay"
        or not all(
            _is_sha(replay.get(key))
            for key in ("selection_sha256", "model_sha256", "capture_receipt_sha256", "sealed_receipt_sha256")
        )
    ):
        raise AttestationNotVerified("only a verified preselected sealed M2M replay can be attested")
    observed = _sealed_m2m_bindings(Path(selection_path), replay["selection_sha256"], Path(model_path))
    if (observed["model_sha256"], observed["receipt_sha256"], observed["sealed_receipt_sha256"]) != (
        replay["model_sha256"],
        replay["capture_receipt_sha256"],
        replay["sealed_receipt_sha256"],
    ):
        raise AttestationNotVerified("replayed capture bytes differ from the bytes on disk")
    document = {
        "schema": SCHEMA,
        "status": "verified_sealed_execution",
        "source_closure_verified": True,
        "fresh_execution": True,
        "issuer": SEALED_M2M_ISSUER,
        "policy": copy.deepcopy(SEALED_M2M_POLICY),
        "selection": {"path": str(Path(selection_path).absolute()), "sha256": replay["selection_sha256"]},
        "replay": {"schema": replay["schema"], "status": replay["status"]},
        "capture": observed,
    }
    require_verified_execution(document)
    return document


def _sealed_m2m_v3_bindings(selection_path: Path, selection_sha256: str, capture_path: Path) -> dict[str, Any]:
    """Bind a complete v3 single program or every program of a saved session."""
    from merlin_experiments.capture_execution import sealed_m2m
    from merlin_experiments.capture_execution.sealed_static import _canonical_path, _file_digest

    from .capture_selection import SCHEMA_V2 as SELECTION_V2
    from .capture_selection import load

    try:
        selected = load(selection_path, expected_sha256=selection_sha256)
        if selected.get("schema") != SELECTION_V2:
            raise AttestationNotVerified("session capture lacks a v2 preselection")
        run = _canonical_path(Path(selected["run_dir"]), exists=True)
        capture = _canonical_path(Path(capture_path), exists=True)
        if capture != run / "capture" or not capture.is_dir():
            raise AttestationNotVerified("session capture is not the selected run's complete output root")
        pending = run / "sealed_m2m_pending.json"
        if pending.is_symlink() or not pending.is_file():
            raise AttestationNotVerified("session sealed receipt is absent or indirect")
        receipt = strict_json.loads(pending.read_bytes())
        plan = selected["plan"]
        if (
            receipt.get("schema") != sealed_m2m.SCHEMA_V3
            or receipt.get("status") != "pending_replay"
            or receipt.get("capture_selection_sha256") != selection_sha256
            or receipt.get("plan") != plan
            or receipt.get("policy_sha256") != selected.get("sandbox_policy_sha256")
            or receipt.get("issuer_sha256") != selected.get("issuer_source_sha256")
            or receipt.get("bwrap_sha256") != (selected.get("bwrap") or {}).get("sha256")
            or _file_digest(Path(selected["bwrap"]["path"])) != receipt["bwrap_sha256"]
        ):
            raise AttestationNotVerified("session receipt does not bind the selected issuer, policy and tools")
        source, runtime = run / "snapshots/source", run / "snapshots/guest-root"
        if sealed_m2m._snapshot_tree(source) != receipt.get("source"):
            raise AttestationNotVerified("session source bytes differ from the sealed receipt")
        if sealed_m2m._snapshot_tree(runtime, allow_dangling_symlinks=True) != receipt.get("guest_root"):
            raise AttestationNotVerified("session runtime bytes differ from the sealed receipt")
        if sealed_m2m_tree_snapshot(capture) != receipt.get("output"):
            raise AttestationNotVerified("session output bytes differ from the sealed receipt")
        reuse = plan.get("runtime_reuse")
        if reuse is not None:
            from merlin_experiments.capture_execution import readonly_runtime

            readonly_runtime.verify(plan, reuse, runtime)
            sealed_m2m._verify_staged_selection(
                plan, source, Path(reuse["entry"]), runtime_rows=readonly_runtime.effective_rows(reuse)
            )
        else:
            sealed_m2m._verify_staged_selection(plan, source, runtime)
        materialized = sealed_m2m._materialized_v3(capture, source, capture, plan)
        if materialized.get("kind") not in {"single", "session"} or materialized != receipt.get("materialized"):
            raise AttestationNotVerified("capture does not retain its verified materialized program set")
        bound = {
            "run_dir": str(run),
            "capture_path": str(capture),
            "kind": materialized["kind"],
            "capture_tree_sha256": receipt["output"]["sha256"],
            "sealed_receipt_sha256": _file_digest(pending),
            "integer_contractions": materialized["integer_contractions"],
        }
        if materialized["kind"] == "session":
            bound.update(
                session_contract_sha256=materialized["session_contract_sha256"],
                session_receipt_sha256=materialized["session_receipt_sha256"],
                programs=materialized["programs"],
            )
        else:
            bound.update(
                model_sha256=_file_digest(capture / "model.mlir"),
                receipt_sha256=_file_digest(capture / "capture_receipt.json"),
            )
        return bound
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        if isinstance(exc, AttestationNotVerified):
            raise
        raise AttestationNotVerified(f"sealed session evidence is unreadable or changed: {exc}") from exc


def _require_sealed_m2m_v3_bytes(document: Mapping[str, Any]) -> None:
    capture = document.get("capture") or {}
    selection = document.get("selection") or {}
    if (
        document.get("policy") != SEALED_M2M_POLICY_V3
        or (document.get("replay") or {}).get("schema") != PRESELECTED_REPLAY_SCHEMA
        or (document.get("replay") or {}).get("status") != "verified_preselected_replay"
        or not _is_sha(selection.get("sha256"))
        or not isinstance(selection.get("path"), str)
        or not isinstance(capture.get("capture_path"), str)
    ):
        raise AttestationNotVerified("session attestation lacks its v3 selection or replay bindings")
    observed = _sealed_m2m_v3_bindings(Path(selection["path"]), selection["sha256"], Path(capture["capture_path"]))
    if observed != capture:
        raise AttestationNotVerified("session attestation differs from selected program or output bytes")


def attest_sealed_m2m_v3(replay: Mapping[str, Any], *, selection_path: Path, capture_path: Path) -> dict[str, Any]:
    """Issue a v3 attestation for a replayed complete program or saved session."""
    if (
        not isinstance(replay, Mapping)
        or replay.get("schema") != PRESELECTED_REPLAY_SCHEMA
        or replay.get("status") != "verified_preselected_replay"
        or replay.get("capture_kind") not in {"single", "session"}
        or not all(
            _is_sha(replay.get(key)) for key in ("selection_sha256", "sealed_receipt_sha256", "capture_tree_sha256")
        )
    ):
        raise AttestationNotVerified("only a fully replayed v3 capture can be attested")
    observed = _sealed_m2m_v3_bindings(Path(selection_path), replay["selection_sha256"], Path(capture_path))
    if observed["kind"] != replay["capture_kind"]:
        raise AttestationNotVerified("replayed capture kind differs from the selected output")
    fields = ["sealed_receipt_sha256", "capture_tree_sha256", "integer_contractions"]
    if observed["kind"] == "session":
        fields += ["session_contract_sha256", "session_receipt_sha256", "programs"]
    else:
        fields += ["model_sha256", "capture_receipt_sha256"]
    for key in fields:
        expected = observed["receipt_sha256"] if key == "capture_receipt_sha256" else observed[key]
        if replay.get(key) != expected:
            raise AttestationNotVerified("replayed capture or program bytes differ from the selected output")
    document = {
        "schema": SCHEMA_V2,
        "status": "verified_sealed_execution",
        "source_closure_verified": True,
        "fresh_execution": True,
        "issuer": SEALED_M2M_ISSUER_V3,
        "policy": copy.deepcopy(SEALED_M2M_POLICY_V3),
        "selection": {"path": str(Path(selection_path).absolute()), "sha256": replay["selection_sha256"]},
        "replay": {"schema": replay["schema"], "status": replay["status"]},
        "capture": observed,
    }
    require_verified_execution(document)
    return document


def attest_sealed_m2m_session(replay: Mapping[str, Any], *, selection_path: Path, capture_path: Path) -> dict[str, Any]:
    """Compatibility entry point that refuses a v3 single-program capture."""
    if replay.get("capture_kind") != "session":
        raise AttestationNotVerified("the session attester requires a multi-program root")
    return attest_sealed_m2m_v3(replay, selection_path=selection_path, capture_path=capture_path)


def assess_sealed_m2m_capture(
    run_dir: Path,
    model_path: Path,
    *,
    model_sha256: str,
    capture_receipt_sha256: str,
    sealed_receipt_sha256: str,
    bwrap_binary: Path | None = None,
) -> dict[str, Any]:
    """Replay a selected CPU capture against independently selected Phase 0 bytes.

    The sealed receipt commits to the actual command, selected source/runtime
    snapshots, sandbox policy, and output. Require its independently selected
    digest as well as the model/materialized-receipt digests: replaying a
    different pending record cannot silently satisfy the same selection.
    This is a read-only assessment. The M2M receipt and its source/runtime plan
    remain unsigned, so even a passing replay cannot grant historical issuance
    or Phase 0 source-closure admission.
    """
    from merlin_experiments.capture_execution.sealed_m2m import (
        SCHEMA as M2M_V2_SCHEMA,
    )
    from merlin_experiments.capture_execution.sealed_m2m import (
        replay_verify,
    )
    from merlin_experiments.capture_execution.sealed_static import _canonical_path

    result: dict[str, Any] = {
        "schema": SEALED_M2M_ASSESSMENT_SCHEMA,
        "status": "unverified",
        "phase0_admission": "not_granted",
        "source_closure_verified": False,
        "fresh_execution": False,
        "capture": {
            "model_sha256": model_sha256,
            "receipt_sha256": capture_receipt_sha256,
            "sealed_receipt_sha256": sealed_receipt_sha256,
        },
        "replay": None,
        "blockers": [],
    }

    def blocked(reason: str) -> dict[str, Any]:
        result["blockers"].append(reason)
        return result

    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (model_sha256, capture_receipt_sha256, sealed_receipt_sha256)
    ):
        return blocked("selected model, capture receipt and sealed receipt require exact SHA-256 byte identities")
    try:
        run = _canonical_path(Path(run_dir), exists=True)
        model = _canonical_path(Path(model_path), exists=True)
    except ValueError as exc:
        return blocked(f"selected capture path is absent or indirect: {exc}")
    if model != run / "capture/model.mlir" or not model.is_file():
        return blocked("selected model is not the exact sealed M2M run capture/model.mlir")
    receipt_path = model.parent / "capture_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        return blocked("selected M2M capture receipt is absent or indirect")
    pending = run / "sealed_m2m_pending.json"
    if pending.is_symlink() or not pending.is_file():
        return blocked("selected M2M sealed receipt is absent or indirect")
    try:
        _, observed_model = _sha256(model)
        _, observed_receipt = _sha256(receipt_path)
        _, observed_sealed = _sha256(pending)
    except (OSError, ValueError) as exc:
        return blocked(f"selected M2M evidence bytes cannot be read consistently: {exc}")
    if (observed_model, observed_receipt, observed_sealed) != (
        model_sha256,
        capture_receipt_sha256,
        sealed_receipt_sha256,
    ):
        return blocked("selected model, capture receipt or sealed receipt bytes differ from the Phase 0 selection")
    try:
        materialized = verify_capture_receipt(model)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return blocked(f"selected M2M materialized receipt is malformed: {exc}")
    if materialized.get("status") != "verified_materialized" or materialized.get("receipt_sha256") != observed_receipt:
        return blocked(f"selected M2M materialized receipt is unverified: {materialized.get('errors')}")
    try:
        replay = replay_verify(run, bwrap_binary=bwrap_binary)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        return blocked(f"sealed M2M replay failed: {exc}")
    if (
        not isinstance(replay, dict)
        or replay.get("schema") != M2M_V2_SCHEMA
        or replay.get("status") != "verified_sandbox_replay"
        or replay.get("sealed_source_closure_replayed") is not True
        or replay.get("phase0_admission") != "not_granted"
    ):
        return blocked("sealed M2M replay has no supported CPU v2 proof")
    try:
        _, final_sealed = _sha256(pending)
        _, final_model = _sha256(model)
        _, final_receipt = _sha256(receipt_path)
    except (OSError, ValueError) as exc:
        return blocked(f"sealed M2M evidence changed after replay: {exc}")
    if (
        replay.get("receipt_sha256") != sealed_receipt_sha256
        or final_sealed != sealed_receipt_sha256
        or final_model != model_sha256
        or final_receipt != capture_receipt_sha256
    ):
        return blocked("sealed M2M replay or selected capture bytes changed during assessment")
    result["status"] = "replay_verified_nonadmissible"
    result["replay"] = {
        "schema": replay["schema"],
        "status": replay["status"],
        "receipt_sha256": sealed_receipt_sha256,
        "capture_dtype": replay.get("capture_dtype"),
    }
    result["blockers"] = [
        "unsigned M2M receipt cannot authenticate the original source/runtime selection or clean pinned revision",
        "selected Python runtime, framework dependencies and model data lack an independently verified closure pin",
    ]
    return result
