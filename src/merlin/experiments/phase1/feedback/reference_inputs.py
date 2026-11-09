"""Admit and pin selected author-visible reference and candidate bytes without importing their code."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath

_FORMATS = (
    "auto",
    "c_source",
    "python_schedule",
    "listing",
    "disassembly",
    "elf",
    "schedule_mlir",
    "semantic_region",
    "interface_mlir",
    "source_mlir",
    "emitted_mlir",
    "command_buffer",
    "artifact_views",
    "candidate_build",
)


_MAX_BYTES = 256 << 20


_MAX_TEXT = 32 << 20


_MAX_ROWS = 2_000_000


class _AnalysisGap(ValueError):
    """Bounded actionable parser/schema detail, separate from source evidence."""

    def __init__(self, code, **detail):
        super().__init__(code)
        self.detail = detail


def _relative(value):
    if not isinstance(value, str) or not value or len(value) > 1024 or "\\" in value:
        raise ValueError("input_requires_a_visible_relative_path")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in (".", "..") for part in path.parts):
        raise ValueError("input_requires_a_visible_relative_path")
    return path


def _path(workspace, value, *, candidate=False, reference_root=None):
    relative = _relative(value)
    parts = relative.parts
    allowed = (
        parts[0] in ("reference-kernels", "cca-evidence")
        if not candidate
        else parts[0] in ("submission", "selfcheck_out", "artifacts")
        or (len(parts) > 2 and parts[0] == "variants" and parts[1] not in (".", ".."))
    )
    if not allowed:
        raise ValueError("input_outside_selected_visible_roots")
    root = workspace if candidate or reference_root is None or parts[0] == "cca-evidence" else reference_root
    path = workspace / relative if root == workspace else root.joinpath(*parts[1:])
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.resolve().is_relative_to(root):
        raise ValueError("linked_or_foreign_analysis_input")
    return path, relative.as_posix()


def _reference_root(workspace, bundle_manifest, repo):
    """Resolve the original read-only grant to frozen bytes, never its live alias."""
    if bundle_manifest is None:
        # Physical copies remain useful for explicit copy-mode diagnostics.
        # A sandbox alias cannot acquire authority through this fallback.
        if (workspace / "reference-kernels").is_symlink():
            raise ValueError("reference_alias_requires_original_bundle")
        return None
    import yaml

    from merlin.targetgen.sandbox import bwrap as BW

    manifest = Path(bundle_manifest)
    if (
        manifest.is_relative_to(workspace)
        or manifest.is_symlink()
        or not manifest.is_file()
        or manifest.stat().st_size > 4 << 20
        or repo is None
    ):
        raise ValueError("reference_analysis_requires_original_bundle")
    bundle = yaml.safe_load(manifest.read_text())
    grants = [row for row in bundle.get("allowed", []) if row.get("as") == "reference-kernels"]
    if len(grants) != 1 or grants[0].get("mode") != "ro":
        raise ValueError("reference_analysis_requires_one_readonly_reference_grant")
    source = BW.resolve_grant(grants[0]["path"], Path(repo)).absolute()
    [frozen] = BW.snapshot_input_paths(workspace, bundle, [source], repo=Path(repo))
    alias = workspace / "reference-kernels"
    if not alias.is_symlink() or alias.readlink() != source:
        raise ValueError("reference_alias_does_not_match_original_grant")
    return frozen


def _read(path, label, *, allow_elf=False):
    # Pin each ancestor by directory FD. A concurrently replaced workspace
    # parent cannot redirect the host reader through a symlink into private data.
    from merlin.analysis.artifact_reader import PinnedELF, open_ordinary

    descriptor = open_ordinary(path)
    try:
        is_elf = allow_elf and os.pread(descriptor, 4, 0) == b"\x7fELF"
    except BaseException:
        os.close(descriptor)
        raise
    if is_elf:
        artifact = PinnedELF(descriptor, label)
        return artifact, artifact.pin
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_BYTES:
            raise ValueError("input_not_a_bounded_ordinary_file")
        raw = stream.read(_MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if len(raw) > _MAX_BYTES or any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError("analysis_input_changed")
    return raw, dict(path=label, sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


def _member(raw, name):
    _relative(name)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        matches = [row for row in archive.infolist() if row.filename == name]
        if (
            len(matches) != 1
            or matches[0].is_dir()
            or matches[0].file_size > _MAX_BYTES
            or stat.S_ISLNK(matches[0].external_attr >> 16)
        ):
            raise ValueError("archive_member_not_unique_bounded_or_ordinary")
        with archive.open(matches[0]) as stream:
            payload = stream.read(_MAX_BYTES + 1)
        if len(payload) > _MAX_BYTES:
            raise ValueError("archive_member_exceeds_bound")
        return payload


def _entry(value):
    """A selected map is a locator/hypothesis source, never a semantic proof."""
    identifier = value.get("entry")
    if not identifier:
        return None
    from . import reference_map

    document = reference_map.selected()
    entries = [row for row in (document or {}).get("entries", []) if row["id"] == identifier]
    if len(entries) != 1:
        raise ValueError("analysis_entry_unavailable")
    entry = entries[0]
    path = "reference-kernels/" + entry["file"]
    if value.get("path") and value["path"] != path:
        raise ValueError("analysis_entry_path_mismatch")
    return entry


# Preserve established callable pickle identities through the explicit public exports.
for _export in (
    _AnalysisGap,
    _relative,
    _path,
    _reference_root,
    _read,
    _member,
    _entry,
):
    _export.__module__ = "merlin.experiments.phase1.feedback.reference_analysis"
