"""Capture-cache source observations, locking and atomic attempt publication.

No framework imports or capture execution. Callers own validation and must hold the
slot lock across lookup, capture, identity rechecks and publication. Source observations
are deliberately narrower than complete framework/loader/data provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

# ---- capture_cache -----------------------------------------------------------------------------

# The decomposed source producers replace the former capsule_source monolith.
# This is a source commitment only, not an import or candidate access grant.
CAPTURE_IMPLEMENTATIONS = (
    "merlin.capture.artifacts",
    "merlin.capture.receipts",
    "merlin.capture.loaders",
    "merlin.capture.pytorch",
    "merlin.capture.demands",
    "merlin.capture.spec",
    "merlin.experiments.corpus.capsules.model_sources",
    "merlin.experiments.corpus.capsules.framework_sources",
    "merlin.experiments.corpus.capsules.model_inputs",
)


def implementation_identity() -> dict:
    """Bind direct Merlin capture/normalization owners without importing their implementations.

    This inventory is not the upstream framework stack or arbitrary loader dependency/data
    closure. Unreadable source refuses cache use through the caller's uncached path.
    """
    from merlin.common.digest import sha256_file
    from merlin.common.paths import module_source_path

    owners = (
        *CAPTURE_IMPLEMENTATIONS,
        "merlin.development.targetgen.capsule_source",
        "merlin.capture.preparation.cache",
        "merlin.capture.preparation.worker",
        "merlin.capture.preparation.arguments",
        "merlin.capture.preparation.values",
        "merlin.compiler.quantization.recipe",
        "merlin.capture.preparation.state",
        "merlin.capture.preparation.artifacts",
        "merlin.capture.preparation.deployment",
        "merlin.capture.preparation.reference",
        "merlin.development.targetgen.applications._aten_opset_worker",
        "merlin.compiler.quantization.torchao._recipe_quantizer",
        "merlin.compiler.quantization.torchao",
        "merlin.compiler.quantization.torchao.application",
        "merlin.compiler.quantization.torchao.eager",
        "merlin.compiler.quantization.torchao.pt2e",
        "merlin.compiler.quantization.torchao.annotation",
        "merlin.compiler.quantization.torchao.graphs",
        "merlin.compiler.quantization.torchao.effects",
        "merlin.compiler.quantization.torchao.calibration",
        "merlin.compiler.quantization.torchao.portable",
        "merlin.compiler.quantization.torchao.rewrites._activation_contractions",
        "merlin.compiler.quantization.torchao.rewrites._recipe_specs",
        "merlin.compiler.quantization.torchao.rewrites._recipe_agreement",
        "merlin.compiler.quantization.torchao.rewrites._capture_integer_mm",
        "merlin.compiler.quantization.torchao.rewrites._recipe_observers",
        "merlin.compiler.quantization.torchao.rewrites._integer_epilogue",
        "merlin.compiler.quantization.torchao.rewrites._integer_operand_sum",
        "merlin.compiler.quantization.torchao.rewrites._recipe_extensions",
        "merlin.compiler.quantization.torchao.rewrites._integer_residual",
        "merlin.compiler.quantization.torchao.rewrites._integer_pool",
        "merlin.compiler.quantization.torchao.rewrites._projection_input_reuse",
        "merlin.compiler.quantization.torchao.rewrites._projection_context",
        "merlin.compiler.quantization.torchao.rewrites._dynamic_contractions",
        "merlin.compiler.quantization.torchao.rewrites._static_attention",
        "merlin.compiler.quantization.torchao.rewrites._static_planes",
        "merlin.compiler.quantization.torchao.rewrites._gptq",
        "merlin.compiler.quantization.quant_layer_plan",
        "merlin.target.spec.software",
        "merlin.target.capabilities.semantic_families",
        "merlin.common.target_data",
        "merlin.runtime.engine.command_buffer",
        "merlin.compiler.frontend.patterns.capture_normalization",
        "merlin.compiler.frontend.linalg_mlir",
        "merlin.compiler.quantization.torchao.affine",
        "merlin.compiler.dialects.xdsl_context",
        "merlin.compiler.ir.xdsl_io",
        "merlin.common.digest",
    )
    result = {}
    for owner in owners:
        # Facade aliases deliberately retain object identity. Pin their real
        # source owners and the physical portable entry independently of which
        # spelling was imported first; find_spec would follow the live alias.
        root = Path(__file__).resolve().parent.parent.parent
        # Read the physical source, including the facade entry itself. Import
        # aliases may point at an application module and are not file identities.
        path = root.joinpath(*owner.split(".")[1:]).with_suffix(".py")
        if not path.is_file():
            initializer = root.joinpath(*owner.split(".")[1:], "__init__.py")
            path = initializer if initializer.is_file() else module_source_path(owner)
        result[owner] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def observed_sources_match(meta: dict) -> bool:
    """Recheck observed source bytes, without claiming a complete loader dependency closure."""
    from merlin.common.digest import sha256_file

    records = meta.get("loader_dependency_sources")
    if not isinstance(records, list):
        return False
    try:
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                return False
            path = Path(record["path"])
            if not path.is_absolute() or path.is_symlink() or sha256_file(path) != record.get("sha256"):
                return False
    except (OSError, ValueError):
        return False
    return True


def frontend_trace_matches(meta: dict, attempt: Path) -> bool:
    """Do not reuse a capture whose recorded frontend evidence disappeared or changed."""
    documents = {}
    for name in ("frontend_trace", "framework_catalog"):
        record = meta.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            return False
        selected = Path(record["path"])
        try:
            if selected.is_symlink() or not selected.resolve().is_relative_to(attempt.resolve()):
                return False
            if hashlib.sha256(selected.read_bytes()).hexdigest() != record.get("sha256"):
                return False
            documents[name] = json.loads(selected.read_bytes())
        except (OSError, ValueError):
            return False
    trace = documents["frontend_trace"]
    if not isinstance(trace, dict):
        return False
    declared = trace.get("mlir") or {}
    if not isinstance(declared, dict):
        return False
    try:
        source = Path(meta.get("frontend_raw_mlir_path") or attempt / "linalg.mlir")
        if source.is_symlink() or not source.resolve().is_relative_to(attempt.resolve()):
            return False
        raw = source.read_bytes()
        if declared.get("sha256") is not None and hashlib.sha256(raw).hexdigest() != declared["sha256"]:
            return False
        if declared.get("bytes") is not None and len(raw) != declared["bytes"]:
            return False
        normalization = meta.get("capture_normalization") or {}
        if normalization:
            if hashlib.sha256(raw).hexdigest() != normalization.get("input_sha256"):
                return False
            normalized = attempt / "linalg.mlir"
            if normalized.is_symlink() or hashlib.sha256(normalized.read_bytes()).hexdigest() != normalization.get(
                "output_sha256"
            ):
                return False
    except (OSError, ValueError, TypeError):
        return False
    return True


@contextmanager
def slot_lock(slot: Path | None):
    """Serialize a shared slot; unavailable locking means no shared cache use.

    Keep the lock beside the slot, never unlink it: replacing a locked inode
    would let another process acquire a different lock for the same capture.
    """
    if slot is None:
        yield None
        return
    stream = None
    try:
        import fcntl

        slot.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(slot.with_name(slot.name + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        stream = os.fdopen(descriptor, "a")
        fcntl.flock(stream, fcntl.LOCK_EX)
    except (ImportError, AttributeError, OSError):
        if stream is not None:
            stream.close()
        yield None
        return
    try:
        yield slot
    finally:
        stream.close()


def commit(slot: Path, attempt: Path) -> None:
    """Atomically publish the parent-validated metadata identity, never worker readiness."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=slot, prefix=".complete-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(
                {
                    "version": 1,
                    "attempt": attempt.name,
                    "meta_sha256": hashlib.sha256((attempt / "meta.json").read_bytes()).hexdigest(),
                },
                stream,
            )
        os.replace(temporary, slot / "capture_complete.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def committed_attempt(slot: Path) -> Path | None:
    try:
        document = json.loads((slot / "capture_complete.json").read_text())
        name = document.get("attempt") if isinstance(document, dict) else None
        if not isinstance(name, str) or not name.startswith("attempt-") or Path(name).name != name:
            return None
        attempt = slot / name
        if attempt.is_symlink() or not attempt.is_dir():
            return None
        expected = {
            "version": 1,
            "attempt": name,
            "meta_sha256": hashlib.sha256((attempt / "meta.json").read_bytes()).hexdigest(),
        }
        return attempt if document == expected else None
    except (OSError, ValueError):
        return None
