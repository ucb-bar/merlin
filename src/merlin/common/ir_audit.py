"""Invocation-local exact-byte lowering evidence, not a compiler certificate.

Declared sidecars are bound in place, without converting or duplicating tensor bytes.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType


def audit_mode(value: bool | str) -> str:
    """Normalize the public option without accepting arbitrary truthy objects."""
    if value is False:
        return "off"
    if value is True:
        return "exact"
    if isinstance(value, str) and value in {"exact", "compact", "both"}:
        return value
    raise ValueError("ir_audit must be False, True, 'exact', 'compact', or 'both'")


def _identity(path: Path) -> dict:
    """Observed bytes, not an atomic snapshot or a concurrent-writer exclusion."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = stream.tell()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def read_tensor_payload(directory: str | Path, descriptor: Mapping) -> bytes:
    """Read a recorded dense payload, verifying the returned bytes against its descriptor.

    The caller supplies the trusted descriptor; this is not authentication of an
    arbitrary audit index or a concurrent-writer exclusion. No numeric decoding occurs.
    """
    fields = {"file", "sha256", "bytes", "format", "element_type", "shape"}
    if not isinstance(descriptor, Mapping) or set(descriptor) != fields:
        raise ValueError("malformed inspection tensor descriptor")
    digest, size, shape = descriptor["sha256"], descriptor["bytes"], descriptor["shape"]
    if (
        descriptor["format"] != "xdsl-dense-bytes"
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(size) is not int
        or size < 0
        or not isinstance(shape, list)
        or any(type(dim) is not int or dim < 0 for dim in shape)
        or not isinstance(descriptor["element_type"], str)
        or not descriptor["element_type"].strip()
        or descriptor["file"] != f"tensors/{digest}.bin"
    ):
        raise ValueError("malformed inspection tensor descriptor")
    root = Path(directory)
    tensor_directory = root / "tensors"
    path = root / descriptor["file"]
    if root.is_symlink() or tensor_directory.is_symlink() or path.is_symlink():
        raise ValueError("inspection tensor path must not be a symlink")
    if not root.is_dir() or not tensor_directory.is_dir() or not path.is_file():
        raise ValueError("inspection tensor payload is absent or not an ordinary file")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("inspection tensor payload escapes audit directory")
    if path.stat().st_size != size:
        raise ValueError("inspection tensor payload size or hash differs")
    with path.open("rb") as stream:
        payload = stream.read(size + 1)
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("inspection tensor payload size or hash differs")
    return payload


class IrAudit:
    """Fresh audit directory beneath an explicit lowering workdir.

    Disabled/absent-workdir sessions perform no IO. Enabled sessions are strict:
    missing sidecars or unwritable evidence refuse the invocation. No pass runs on
    archived copies; serialization never replaces the executable input.
    """

    def __init__(
        self,
        workdir: str | Path | None,
        *,
        enabled: bool | str = False,
        producer: str,
        source: str | Path,
        sidecars: tuple[str | Path, ...] = (),
    ):
        self.mode = audit_mode(enabled)
        self.workdir = Path(workdir) if workdir is not None and self.mode != "off" else None
        self.directory: Path | None = None
        self.source = Path(source)
        self.sidecars = tuple(Path(path) for path in sidecars)
        self._tensor_descriptors: set[str] = set()
        self._tensor_files: dict[str, dict] = {}
        self.record = {"schema_version": 1, "producer": producer, "mode": self.mode, "outcome": "running", "stages": []}

    def _flush(self) -> None:
        assert self.directory is not None
        pending = self.directory / "index.json.pending"
        pending.write_text(json.dumps(self.record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pending.replace(self.directory / "index.json")

    def __enter__(self) -> IrAudit:
        if self.workdir is None:
            return self
        # Validate declared dependencies before writing anything or running a pass.
        self.record["sidecars"] = [_identity(path) for path in self.sidecars]
        self.record["producer_source"] = _identity(self.source)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="ir-audit-", dir=self.workdir))
        self._flush()
        return self

    def tensor(self, payload: bytes, *, element_type: str, shape: tuple[int, ...]) -> dict:
        """Store exact caller-owned dense storage, deduplicated within this audit.

        Type/shape describe the producer's representation, not a portable runtime
        ABI or safetensors conversion. No numeric unpacking or recasting occurs.
        """
        if self.directory is None or self.mode not in {"compact", "both"}:
            raise ValueError("tensor inspection requires an enabled compact or both audit")
        if not isinstance(payload, bytes) or not isinstance(element_type, str) or not element_type:
            raise ValueError("tensor inspection requires raw bytes and an element type")
        if not isinstance(shape, tuple) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError("tensor inspection requires a static nonnegative shape")
        digest = hashlib.sha256(payload).hexdigest()
        directory = self.directory / "tensors"
        if directory.is_symlink():
            raise ValueError("tensor inspection directory must not be a symlink")
        directory.mkdir(exist_ok=True)
        path = directory / f"{digest}.bin"
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or _identity(path)["sha256"] != digest:
                raise ValueError("tensor inspection payload changed")
        else:
            with path.open("xb") as stream:
                stream.write(payload)
        record = {
            "file": str(path.relative_to(self.directory)),
            "sha256": digest,
            "bytes": len(payload),
            "format": "xdsl-dense-bytes",
            "element_type": element_type,
            "shape": list(shape),
        }
        self._tensor_files[digest] = dict(record)
        self._tensor_descriptors.add(json.dumps(record, sort_keys=True))
        return record

    def stage(
        self,
        name: str,
        content: str,
        *,
        format: str = "mlir",
        inspection: str | None = None,
        inspection_tensors: list[dict] | None = None,
    ) -> None:
        """Record exact stage identity and an optional caller-printed inspection view.

        Inspection text is never an executable replacement for content. The caller
        owns elision/printing; this module only labels and binds the observed bytes.
        """
        if self.directory is None:
            return
        if inspection_tensors:
            if inspection is None or self.mode not in {"compact", "both"}:
                raise ValueError("tensor descriptors require an inspection view")
            if any(json.dumps(record, sort_keys=True) not in self._tensor_descriptors for record in inspection_tensors):
                raise ValueError("tensor descriptor was not produced by this audit")
        if not name or any(not (c.isascii() and (c.isalnum() or c in "-_")) for c in name):
            raise ValueError("audit stage names must be plain filename tokens")
        if format not in {"mlir", "llvm-ir"}:
            raise ValueError("unsupported executable IR format")
        stages = self.record["stages"]
        path = self.directory / f"{len(stages):03d}-{name}.{'mlir' if format == 'mlir' else 'll'}"
        payload = content.encode("utf-8")
        if self.mode != "compact":
            path.write_bytes(payload)
        stage = {
            "name": name,
            "file": path.name if self.mode != "compact" else None,
            "format": format,
            "representation": "exact-ir",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "compact_status": "unsupported-printer" if self.mode in {"compact", "both"} else "not-requested",
        }
        if inspection is not None and self.mode in {"compact", "both"}:
            view = self.directory / f"{len(stages):03d}-{name}.inspection.mlir.txt"
            view_payload = (
                f"// INSPECTION ONLY — NOT EXECUTABLE\n// Exact parent SHA-256: {stage['sha256']}\n" + inspection
            ).encode("utf-8")
            view.write_bytes(view_payload)
            stage["inspection"] = {
                "file": view.name,
                "sha256": hashlib.sha256(view_payload).hexdigest(),
                "bytes": len(view_payload),
                "representation": "inspection-only",
                "executable": False,
                "parent_sha256": stage["sha256"],
                "tensors": json.loads(json.dumps(inspection_tensors or [])),
            }
            stage["compact_status"] = "recorded"
        stages.append(stage)
        self._flush()

    def collect_views(self) -> None:
        """Collect records emitted by the native printer; failed child prefixes survive."""
        if self.directory is None or self.mode not in {"compact", "both"}:
            return
        native = self.directory / "native"
        self.record["inspection_views"] = [json.loads(path.read_text()) for path in sorted(native.glob("*.json"))]
        passes = self.directory / "passes"
        records = []
        for path in sorted(passes.rglob("*.mlir")):
            if not path.is_file():
                continue
            identity = _identity(path)
            records.append(
                {
                    "file": str(path.relative_to(self.directory)),
                    "bytes": identity["bytes"],
                    "sha256": identity["sha256"],
                    "representation": "native-pass-inspection-only",
                    "executable": False,
                }
            )
        self.record["pass_inspection"] = {
            "scope": "native pass-manager invocations; not Python rewrites",
            "segments": [
                {
                    "directory": str(path.parent.relative_to(self.directory)),
                    "pipeline": path.read_text(encoding="utf-8"),
                }
                for path in sorted(passes.glob("*/pipeline.txt"))
            ],
            "limits": {"large_elements_limit": 64, "large_resource_limit": 64},
            "files": records,
        }
        self._flush()

    def command(self, argv: list[str], *, sources: tuple[str | Path, ...] = (), provenance: dict | None = None) -> None:
        """Record the actual launch arguments, not a claim of complete tool closure."""
        if self.directory is None:
            return
        self.record.setdefault("commands", []).append(
            {
                "argv": argv,
                "after_stage_count": len(self.record["stages"]),
                "executable": _identity(Path(argv[0])),
                "sources": [_identity(Path(path)) for path in sources],
                "toolchain_observation": provenance or {},
            }
        )
        self._flush()

    def __exit__(self, exc_type: type | None, exc: BaseException | None, tb: TracebackType | None) -> bool:
        if self.directory is None:
            return False
        changed = False
        tensors_changed = False
        stages_changed = False
        if exc_type is None:
            try:
                changed = [_identity(path) for path in self.sidecars] != self.record["sidecars"]
            except OSError:
                changed = True
            try:
                for digest, record in self._tensor_files.items():
                    path = self.directory / record["file"]
                    if path.parent.is_symlink() or path.is_symlink() or _identity(path)["sha256"] != digest:
                        tensors_changed = True
            except OSError:
                tensors_changed = True
            try:
                for stage in self.record["stages"]:
                    for entry in (stage, stage.get("inspection", {})):
                        if not entry.get("file"):
                            continue  # Compact-only exact parents are deliberately hash-only.
                        path = self.directory / entry["file"]
                        if path.is_symlink() or not path.is_file():
                            stages_changed = True
                            continue
                        observed = _identity(path)
                        if any(observed[key] != entry[key] for key in ("sha256", "bytes")):
                            stages_changed = True
            except OSError:
                stages_changed = True
        self.record["outcome"] = (
            "failed" if exc_type is not None or changed or tensors_changed or stages_changed else "completed"
        )
        if exc_type is not None:
            self.record["failure_type"] = exc_type.__name__
        if changed:
            self.record["failure_type"] = "SidecarChanged"
        elif tensors_changed:
            self.record["failure_type"] = "TensorPayloadChanged"
        elif stages_changed:
            self.record["failure_type"] = "StagePayloadChanged"
        try:
            self._flush()
        except OSError as audit_error:
            if exc is None:
                raise
            exc.add_note(f"IR audit outcome could not be written: {type(audit_error).__name__}")
        if changed:
            raise ValueError("declared IR sidecar changed during lowering")
        if tensors_changed:
            raise ValueError("IR inspection tensor payload changed during lowering")
        if stages_changed:
            raise ValueError("IR audit stage payload changed during lowering")
        # Warning filters can raise. Persist the lowering's terminal outcome first,
        # and never replace an already-active lowering exception with a warning.
        if (
            self.mode == "compact"
            and not self.record.get("inspection_views")
            and not any(stage.get("inspection") for stage in self.record["stages"])
        ):
            try:
                warnings.warn(
                    "Compact IR audit has no supported printer views; stage records are hash-only.",
                    UserWarning,
                    stacklevel=2,
                )
            except Warning as warning:
                if exc is None:
                    raise
                exc.add_note(str(warning))
        return False
