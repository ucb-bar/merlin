"""Bound regular-file IO and append-only records for the new private context.

No SDK is selected or opened at import. Installed/archive roots are forbidden
before filesystem traversal. Linux directory descriptors and loader reporting
remain trusted host mechanisms, not formal isolation theorems.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

if __package__:
    from .lean_graph_original_image_context_data import ContextRefusal, FileRef, absolute
else:
    from lean_graph_original_image_context_data import ContextRefusal, FileRef, absolute


class BindingError(ContextRefusal):
    def __init__(self, reason, evidence):
        self.evidence = evidence
        super().__init__(reason)


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContextRefusal("duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ContextRefusal("nonfinite JSON constant")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _stats(value):
    return {name: getattr(value, name) for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def _read(path, forbidden, *, failure_directory=None, expected=None):
    path = absolute(str(path))
    # These components name the still-held installation/archive in this task.
    if ".elan" in path.parts or "fresh-environment" in path.parts:
        raise BindingError("held installed/archive path; not opened", {"path": str(path)})
    if any(path.is_relative_to(absolute(root)) for root in forbidden):
        raise BindingError("forbidden original/archive root; not opened", {"path": str(path)})
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    descriptor = None
    chunks = []
    before = None
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BindingError("not a regular selected file", {"path": str(path), "stat": _stats(before)})
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        live = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        evidence = {"path": str(path), "before": _stats(before), "after": _stats(after), "live": _stats(live)}
        if _stats(before) != _stats(after) or _stats(after) != _stats(live):
            fail_with_buffer(BindingError("selected descriptor/path changed during read", evidence),
                             data, failure_directory, expected=expected, complete=True)
        return data, evidence
    except OSError as error:
        evidence = {"path": str(path), "error": str(error),
                    "before": _stats(before) if before is not None else None}
        fail_with_buffer(BindingError("selected file unavailable/nonregular/symlink", evidence),
                         b"".join(chunks), failure_directory, expected=expected, complete=False)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def read_bound(reference: FileRef, forbidden=(), *, failure_directory=None):
    data, evidence = _read(reference.path, forbidden, failure_directory=failure_directory,
                           expected=reference.record())
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != reference.bytes or digest != reference.sha256:
        failure = BindingError(
            "selected bytes differ; no refreshed expectation",
            {
                **evidence,
                "expected": reference.record(),
                "observed_bytes": len(data),
                "observed_sha256": digest,
            },
        )
        fail_with_buffer(failure, data, failure_directory, expected=reference.record(), complete=True)
    return data


def output_reference(path, workspace):
    """Observe ONLY a newly produced project artifact, never a runtime image."""
    path, workspace = absolute(str(path)), absolute(str(workspace))
    if not path.is_relative_to(workspace):
        raise ContextRefusal("new output outside private workspace")
    data, _ = _read(path, ())
    return FileRef(str(path), hashlib.sha256(data).hexdigest(), len(data))


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def save(path, value):
    write_new(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def fail_with_buffer(error, raw, directory, *, expected, complete):
    """NEW diagnostic-only sink. Always raise the SAME error after preservation.

    The caller supplies its private workspace. No source is reopened; raw is the
    already-read immutable bytes. Never replace the original observation or pin.
    A dump failure is supplementary evidence and never permits acceptance.
    """
    if directory is not None:
        original = {"type": type(error).__name__, "reason": str(error),
                    "evidence": error.evidence, "expected": expected,
                    "read_reached_eof": complete}
        try:
            directory = Path(directory)
            directory.mkdir(exist_ok=False)
            diagnostic_json(directory / "ORIGINAL-FAILURE.json", original)
            target = directory / "already-read-buffer.bin"
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
            with os.fdopen(descriptor, "wb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            diagnostic = {
                "path": str(target), "bytes": len(raw),
                "buffer_sha256_at_preservation": hashlib.sha256(raw).hexdigest(),
                "original_observed_sha256": error.evidence.get("observed_sha256"),
                "source_reopens": 0, "primary_failure_preserved": True,
                "holds_cleared": False, "native_authorized": False,
            }
            diagnostic_json(directory / "BUFFER.json", diagnostic)
            error.evidence = {**error.evidence, "forensic_buffer": diagnostic}
        except Exception as sink_error:
            error.evidence = {**error.evidence, "forensic_sink_failed": {
                "type": type(sink_error).__name__, "error": str(sink_error)}}
    raise error


def diagnostic_json(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(descriptor, "wb") as output:
        output.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        output.flush()
        os.fsync(output.fileno())
