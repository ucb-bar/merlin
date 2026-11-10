"""Host-private semantic-search census over a reviewed public capsule snapshot.

The census is advisory evidence. It never selects a treatment, serves an agent
tool or prompt, or contributes to a Phase 1 grade or certification decision.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from merlin.development.backend.source_inputs import fingerprint
from merlin.target.spec.instructions import validate_normalized_instruction_model
from merlin.targetgen.capsule_common import discover_capsules
from merlin.targetgen.contract.linalg_iface import is_linalg_on_tensors, parse_linalg_mlir
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.semantic_search import SearchLimits, search_linalg_inventory

_SCHEMA = "merlin.phase1.semantic_search_diagnostic.v1"
_RECEIPT = "semantic_search_diagnostic.json"
_MAX_PAYLOAD_OPS = 32
_MAX_LINALG_BYTES = 2 * 1024 * 1024
_SEARCH_LIMITS = SearchLimits(timeout_ms=250, solver_timeout_ms=50, max_candidates=64)


def _ordinary_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"semantic-search input is not an ordinary file: {path}")
    return path.read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assert_host_private_location(run_dir: Path, *, workspace: Path, public_root: Path) -> None:
    """Keep the owner receipt out of every directly bound agent input tree.

    Mode 0600 is insufficient: the sandboxed agent runs under the operator's UID.
    Resolve parent symlinks before testing both containment directions so a
    configured run root cannot alias an agent-visible workspace or public view.
    """
    run = run_dir.resolve()
    for label, root in (("workspace", workspace), ("public corpus", public_root)):
        visible = root.resolve()
        if run.is_relative_to(visible) or visible.is_relative_to(run):
            raise RuntimeError(f"semantic-search host-private run directory overlaps the agent {label}")


def assert_private_mounts(argv: list[str], run_dir: Path) -> None:
    """Reject a final agent mount plan that exposes run metadata at any bind alias.

    Call after *all* runtime/toolchain binds and answer masks have been composed,
    immediately before launching an agent. A source bound at another destination
    needs checking too; testing only the receipt's original path misses aliases.
    """
    private_trees = [path.resolve(strict=True) for path in (run_dir / "qa_history",) if path.exists()]
    private = [
        path.resolve(strict=True)
        for path in (run_dir, run_dir / "environment.yaml", run_dir / _RECEIPT)
        if path.exists()
    ] + private_trees
    if not private:
        raise RuntimeError("semantic-search host-private run directory is missing before agent launch")
    if any(BW.is_exposed(argv, path) for path in private):
        raise RuntimeError("semantic-search host-private run metadata is visible in the agent sandbox")
    for state, source, destination in BW._mounts(argv):
        if state != "expose":
            continue
        source_path = Path(source).resolve()
        for path in private:
            if path.is_relative_to(source_path):
                alias = Path(destination) / path.relative_to(source_path)
                if BW.is_exposed(argv, alias):
                    raise RuntimeError(
                        "semantic-search host-private run metadata is visible through an agent bind alias"
                    )
        # A direct bind of a private descendant does not include its parent at
        # the alias. Protect history trees without rejecting intentionally
        # public frozen views elsewhere below the host-owned run directory.
        if any(source_path.is_relative_to(tree) for tree in private_trees) and BW.is_exposed(argv, Path(destination)):
            raise RuntimeError("host-private QA history is visible through an agent bind alias")


def _public_linalg_files(public_root: Path, contract_root: Path) -> list[tuple[str, Path]]:
    """Select declared linalg files from public capsules in the frozen view only."""
    selected = []
    for capsule in discover_capsules(public_root, labels={"public"}, contract=contract_root):
        relative = capsule.get("linalg_mlir")
        if relative is None:
            continue
        relpath = Path(str(relative))
        directory = Path(capsule["__dir__"])
        if relpath.is_absolute() or ".." in relpath.parts or not relpath.parts:
            raise RuntimeError("public capsule declares an escaping linalg input")
        if not directory.is_relative_to(public_root):
            raise RuntimeError("public capsule escaped the frozen public view")
        path = directory / relpath
        if not path.resolve(strict=True).is_relative_to(directory.resolve(strict=True)):
            raise RuntimeError("public capsule linalg input escaped its capsule directory")
        selected.append((path.relative_to(public_root).as_posix(), path))
    names = [name for name, _ in selected]
    if len(set(names)) != len(names):
        raise RuntimeError("public linalg input names are not unique")
    return sorted(selected)


def create(
    run_dir: Path,
    *,
    workspace: Path,
    model_path: Path | None,
    public_root: Path,
    contract_root: Path,
) -> dict:
    """Write an owner-only receipt when the reviewed release has a model."""
    _assert_host_private_location(run_dir, workspace=workspace, public_root=public_root)
    if model_path is None:
        return {"schema": _SCHEMA, "status": "unavailable", "reason": "reviewed release has no instruction model"}

    model_bytes = _ordinary_bytes(model_path)
    model_sha = _sha256(model_bytes)
    public_sha = fingerprint(public_root)
    rows = []
    try:
        model = json.loads(model_bytes)
        if not isinstance(model, Mapping):
            raise ValueError("instruction model must be a JSON object")
        # Phase 0 emits this one non-selectable stub when no OOT instruction
        # description was selected. Every other model must carry the reviewed
        # SW-spec/CIRCT-fact identities and a self-consistent normalized body.
        stub_fields = {"schema", "target", "status", "unknowns", "instructions"}
        if model.get("status") == "UNKNOWN" and model.get("instructions") == [] and set(model) == stub_fields:
            if (
                model.get("schema") != "merlin.instruction_semantics.v1"
                or not isinstance(model.get("target"), str)
                or not model["target"]
                or not isinstance(model.get("unknowns"), list)
                or not model["unknowns"]
            ):
                raise ValueError("Phase 0 instruction-model stub is malformed")
        else:
            model = validate_normalized_instruction_model(model)
        model_error = None
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        model = None
        model_error = f"{type(exc).__name__}: {exc}"
    try:
        selected = _public_linalg_files(public_root, contract_root)
        selection_error = None
    except Exception as exc:  # noqa: BLE001 — optional discovery is diagnostic, not Phase 1 admission
        selected = []
        selection_error = f"{type(exc).__name__}: {exc}"
    searched_ops = 0
    for name, path in selected:
        row = {"capsule_linalg": name}
        try:
            raw = _ordinary_bytes(path)
        except Exception as exc:  # noqa: BLE001 — the native corpus owner retains admission authority
            row.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            continue
        row["linalg_sha256"] = _sha256(raw)
        if model_error is not None:
            row.update(status="unavailable", reason=model_error)
        elif len(raw) > _MAX_LINALG_BYTES:
            row.update(status="limit_reached", reason="declared linalg input exceeds diagnostic byte limit")
        else:
            try:
                text = raw.decode("utf-8")
                if not is_linalg_on_tensors(text):
                    raise ValueError("declared linalg input lacks linalg-on-tensors provenance")
                parsed = parse_linalg_mlir(text)
                operations = parsed["ops"]
                if not any(str(op.get("operation", "")).startswith("linalg.") for op in operations):
                    raise ValueError("declared linalg input has no linalg payload operation")
                if searched_ops + len(operations) > _MAX_PAYLOAD_OPS:
                    row.update(status="limit_reached", reason="public payload operation budget exhausted")
                else:
                    searched_ops += len(operations)
                    row.update(
                        status="observed",
                        inventory=search_linalg_inventory(parsed, model, limits=_SEARCH_LIMITS),
                    )
            except Exception as exc:  # noqa: BLE001 — diagnostic failure cannot change the experiment treatment
                row.update(status="diagnostic_error", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    if fingerprint(public_root) != public_sha or _sha256(_ordinary_bytes(model_path)) != model_sha:
        raise RuntimeError("semantic-search inputs changed during diagnostic collection")
    receipt = {
        "schema": _SCHEMA,
        "scope": "host-private public capsule linalg files; advisory only; no compiler or execution verdict",
        "instruction_model_sha256": model_sha,
        "public_corpus_sha256": public_sha,
        "selection_error": selection_error,
        "limits": {
            "max_payload_operations": _MAX_PAYLOAD_OPS,
            "max_linalg_bytes": _MAX_LINALG_BYTES,
            "search_timeout_ms_per_operation": _SEARCH_LIMITS.timeout_ms,
            "solver_timeout_ms": _SEARCH_LIMITS.solver_timeout_ms,
        },
        "inputs": len(rows),
        "rows": rows,
    }
    data = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
    destination = run_dir / _RECEIPT
    with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(data)
    return {
        "schema": _SCHEMA,
        "status": "recorded",
        "instruction_model_sha256": model_sha,
        "public_corpus_sha256": public_sha,
        "receipt_sha256": _sha256(data),
        "inputs": len(rows),
    }


def verify(record: Mapping, run_dir: Path, *, workspace: Path, model_path: Path | None, public_root: Path) -> None:
    """Refuse drift of a new run's optional model, public inputs, or receipt."""
    _assert_host_private_location(run_dir, workspace=workspace, public_root=public_root)
    if not isinstance(record, Mapping) or record.get("schema") != _SCHEMA:
        raise RuntimeError("semantic-search diagnostic identity is missing or malformed")
    if record.get("status") == "unavailable":
        if model_path is not None or (run_dir / _RECEIPT).exists():
            raise RuntimeError("semantic-search diagnostic availability changed on resume")
        return
    if record.get("status") != "recorded" or model_path is None:
        raise RuntimeError("semantic-search diagnostic model selection changed on resume")
    model_sha = _sha256(_ordinary_bytes(model_path))
    if model_sha != record.get("instruction_model_sha256"):
        raise RuntimeError("semantic-search instruction model changed on resume")
    public_sha = fingerprint(public_root)
    if public_sha != record.get("public_corpus_sha256"):
        raise RuntimeError("semantic-search public corpus changed on resume")
    receipt_path = run_dir / _RECEIPT
    receipt = _ordinary_bytes(receipt_path)
    if receipt_path.stat().st_mode & 0o077:
        raise RuntimeError("semantic-search diagnostic receipt is not owner-only")
    if _sha256(receipt) != record.get("receipt_sha256"):
        raise RuntimeError("semantic-search diagnostic receipt changed on resume")
    try:
        document = json.loads(receipt)
    except json.JSONDecodeError as exc:
        raise RuntimeError("semantic-search diagnostic receipt is malformed") from exc
    if (
        document.get("schema") != _SCHEMA
        or document.get("instruction_model_sha256") != model_sha
        or document.get("public_corpus_sha256") != public_sha
        or document.get("inputs") != record.get("inputs")
        or not isinstance(document.get("rows"), list)
    ):
        raise RuntimeError("semantic-search diagnostic receipt identity is inconsistent")
