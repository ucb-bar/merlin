"""Explicit prepare, inspect and operator-reviewed seal for phase-0 corpus releases.

This host-only workflow binds scientific inputs; it never approves them on behalf
of an operator or calls an agent/oracle. Native admission and grading still own
their verdicts. Review is an attributed local acknowledgement, not a signature.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from ..spec import SpecError
from .preparation import admission, assemble, ordinary_tree, private_json, scaffold, source_run


def _read(path: Path) -> dict:
    from ..runner import _read_json

    ordinary_tree(path)
    if path.stat().st_mode & 0o077:
        raise SpecError("corpus release metadata must be owner-only")
    document = _read_json(path)
    if document.get("schema_version") != 1:
        raise SpecError("unsupported corpus release record version")
    return document


def _digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _private_root(path: Path) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise SpecError("corpus release root may not be a symlink")
    root = lexical.resolve(strict=True)
    if (root / "private").is_symlink() or not (root / "private").is_dir():
        raise SpecError("corpus release private metadata directory is missing or symlinked")
    if root.stat().st_mode & 0o077 or (root / "private").stat().st_mode & 0o077:
        raise SpecError("corpus release and private metadata directories must be owner-only")
    return root


@contextmanager
def _review_lock(root: Path):
    descriptor = os.open(root / "private" / ".review.lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpecError("another process is reviewing this corpus release") from exc
        yield


def _content(root: Path, prepared: dict) -> dict:
    from ..runner import fingerprint

    ordinary_tree(root / "payload")
    observed = fingerprint(root / "payload")
    if prepared.get("payload_sha256") != observed:
        raise SpecError("prepared corpus payload changed; prepare a new release")
    return {"preparation_sha256": _digest(prepared), "payload_sha256": observed}


def _freeze_payload(root: Path) -> None:
    from merlin.common.content_store import is_shared

    payload = root / "payload"
    ordinary_tree(payload)
    for path in sorted(payload.rglob("*"), key=lambda member: len(member.parts), reverse=True):
        if path.is_file() and is_shared(path):
            if path.stat().st_mode & 0o222:
                raise SpecError("shared corpus payload is unexpectedly writable")
            continue
        mode = 0o500 if path.is_dir() or path.stat().st_mode & 0o111 else 0o400
        path.chmod(mode)
    payload.chmod(0o500)


def inspect_release(path: Path) -> dict:
    """Return only safe counts/commitments, never member identities or review text."""
    root = _private_root(path)
    prepared = _read(root / "private" / "preparation.json")
    identity = _content(root, prepared)
    sealed = (root / "private" / "seal.json").is_file()
    if sealed:
        verify(root / "private" / "seal.json", root / "payload" / "experiment" / "target_experiment.yaml")
    return {
        "schema_version": 1,
        "release": str(root),
        "target": prepared["target"],
        "state": "sealed" if sealed else "awaiting_operator_review",
        "review_digest": _digest(identity),
        "payload_sha256": identity["payload_sha256"],
        "descriptor": str(root / "payload" / "experiment" / "target_experiment.yaml"),
        "counts": prepared["admission"],
        "engine_readiness": "not_executed",
    }


def prepare(run_dir: Path, output: Path) -> dict:
    """Assemble a fresh complete source pool; neither canonical inputs nor approval change."""
    from merlin.common.paths import out_dir
    from merlin.common.storage_lifecycle import lease
    from merlin.targetgen.target_experiment import load_target_experiment

    from ..runner import fingerprint

    source = run_dir.expanduser().resolve(strict=True)
    root = output.expanduser().absolute()
    if root.is_symlink() or root.exists():
        raise SpecError("corpus release output already exists; choose a fresh destination")
    root = root.resolve()
    generated_root = out_dir().resolve()
    if not root.is_relative_to(generated_root / "artifacts"):
        raise SpecError("corpus releases must live below the configured artifact root")
    plan, attempt, generated = source_run(source)
    te = load_target_experiment(plan["phases"]["0"]["inputs"]["descriptor"])
    for existing in (source, te.capsule_corpus.parent):
        if root == existing or root.is_relative_to(existing) or existing.is_relative_to(root):
            raise SpecError("corpus release output overlaps an immutable source")
    with lease(root, owner="corpus-release-preparation"):
        root.mkdir(parents=True, mode=0o700)
        (root / "private").mkdir(mode=0o700)
        (root / "payload").mkdir(mode=0o700)
        payload = root / "payload"
        try:
            assembly = assemble(te, generated, payload / "corpus")
            scaffolding = scaffold(te, payload / "corpus", payload / "experiment", private=root / "private")
            descriptor = payload / "experiment" / "target_experiment.yaml"
            checked = admission(descriptor)
            source_run(source)  # derivation/input drift during preparation is not accepted
            private_json(
                root / "private" / "preparation.json",
                {
                    "schema_version": 1,
                    "target": te.target,
                    "source_run": str(source),
                    "source_plan_sha256": fingerprint(source / "resolved-plan.json"),
                    "source_output_sha256": attempt["output_sha256"],
                    "source_descriptor_sha256": fingerprint(te.path),
                    "assembly": assembly,
                    "scaffolding": scaffolding,
                    "admission": checked,
                    "payload_sha256": fingerprint(payload),
                    "prepared_at": datetime.now(UTC).isoformat(),
                },
            )
        except Exception as exc:
            private_json(
                root / "private" / "failure.json",
                {
                    "schema_version": 1,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            raise SpecError(
                "corpus preparation failed; host-only diagnostics are in the release private directory"
            ) from None
    return inspect_release(root)


def seal(path: Path, *, expected_digest: str, reviewed_by: str, review_note: str) -> dict:
    """Record an explicit operator review of these exact bytes; never called by prepare/run."""
    from merlin.common.storage_lifecycle import lease, pin

    if not reviewed_by.strip() or not review_note.strip():
        raise SpecError("sealing requires an explicit reviewer and review note")
    root = _private_root(path)
    with _review_lock(root), lease(root, owner="corpus-release-sealing"):
        if (root / "private" / "seal.json").exists():
            raise SpecError("corpus release is already sealed; review records are not overwritten")
        report = inspect_release(root)
        if report["review_digest"] != expected_digest:
            raise SpecError("review digest does not match the prepared release")
        prepared = _read(root / "private" / "preparation.json")
        try:
            current = admission(Path(report["descriptor"]))
        except Exception:
            raise SpecError("native corpus admission no longer validates the reviewed release") from None
        if current != prepared["admission"] or inspect_release(root)["review_digest"] != expected_digest:
            raise SpecError("reviewed inputs or native admission changed during sealing")
        _freeze_payload(root)
        token = pin(root, reason="explicitly reviewed corpus release")
        private_json(
            root / "private" / "seal.json",
            {
                "schema_version": 1,
                "review_digest": expected_digest,
                "identity": _content(root, prepared),
                "review": {"reviewed_by": reviewed_by, "note": review_note, "at": datetime.now(UTC).isoformat()},
                "retention_pin": token,
            },
        )
    result = inspect_release(root)
    result["seal"] = str(root / "private" / "seal.json")
    return result


def verify(seal_path: Path, descriptor: Path) -> dict:
    """Validate a sealed release selected explicitly by a functional experiment."""
    path = seal_path.expanduser().absolute()
    if path.name != "seal.json" or path.parent.name != "private":
        raise SpecError("corpus seal must name the release's private seal.json record")
    root = _private_root(path.parent.parent)
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise SpecError("corpus seal must be an ordinary owner-only file")
    expected_descriptor = root / "payload" / "experiment" / "target_experiment.yaml"
    if descriptor.resolve() != expected_descriptor:
        raise SpecError("functional descriptor is not the one bound by the corpus seal")
    prepared = _read(root / "private" / "preparation.json")
    sealed = _read(path)
    identity = _content(root, prepared)
    if any(member.stat().st_mode & 0o222 for member in [root / "payload", *(root / "payload").rglob("*")]):
        raise SpecError("sealed corpus payload became writable")
    if sealed.get("identity") != identity or sealed.get("review_digest") != _digest(identity):
        raise SpecError("corpus seal does not bind this prepared payload")
    review = sealed.get("review")
    if not isinstance(review, dict) or not all(
        isinstance(review.get(k), str) and review[k].strip() for k in ("reviewed_by", "note", "at")
    ):
        raise SpecError("corpus seal has no explicit operator review")
    return {
        "release": str(root),
        "review_digest": sealed["review_digest"],
        "payload_sha256": identity["payload_sha256"],
    }


def verify_snapshot(seal_path: Path, descriptor: Path, ws: Path, bundle: dict, *, repo: Path | None = None) -> dict:
    """Native pre-agent gate: bind actual private snapshot bytes to reviewed source bytes."""
    from merlin.targetgen.sandbox.bwrap import snapshot_input_paths
    from merlin.targetgen.target_experiment import load_target_experiment

    from ..runner import fingerprint

    identity = verify(seal_path, descriptor)
    te = load_target_experiment(descriptor)
    sources = [*te.graded_roots(), *te.hidden_roots()]
    sources += [te.resource_path("task"), te.resource_path("scripts/agent_selfcheck.py")]
    sources += [Path(identity["release"]) / "private"]
    snapshots = snapshot_input_paths(ws, bundle, sources, repo=repo)
    for source, snapshot in zip(sources, snapshots, strict=True):
        if fingerprint(source) != fingerprint(snapshot):
            raise SpecError("native corpus snapshot differs from the operator-reviewed release")
    # Review metadata is a host-only snapshot input too. Read its frozen copy,
    # never an absent-copy fallback to mutable release metadata on resume.
    from ..runner import _read_json

    frozen_prepared = _read_json(snapshots[-1] / "preparation.json")
    frozen_sealed = _read_json(snapshots[-1] / "seal.json")
    frozen_identity = {
        "preparation_sha256": _digest(frozen_prepared),
        "payload_sha256": identity["payload_sha256"],
    }
    if (
        frozen_sealed.get("identity") != frozen_identity
        or frozen_sealed.get("review_digest") != identity["review_digest"]
    ):
        raise SpecError("native review snapshot does not bind the operator-reviewed release")
    # A change while the native snapshot was checked cannot borrow its seal.
    if verify(seal_path, descriptor) != identity:
        raise SpecError("corpus release changed while verifying its native snapshot")
    return identity
