"""A capsule's golden: ``golden.yaml`` (the document) plus ``golden.npz`` (its arrays), bound by digest.

A golden used to be one YAML file. For a deep contraction that file reached 129 MB -- two
million operand codes spelled as text, then two million decoded floats -- and dumping it, and every
reader parsing it again, cost minutes per capsule. The document stays YAML, because its metadata
(``golden_source``, ``oracle_provenance``, the qualification blocks) is what people and tools read.
The arrays move to one ``golden.npz`` beside it: every array under ``outputs``, and any other list
larger than :data:`INLINE_LIMIT` elements (operand provenance). In the document each array is replaced
by a reference naming its key, dtype, shape and SHA-256, so the document's own digest binds the arrays
and :func:`load_golden` refuses an archive whose bytes are not the ones referenced. A list mixing
ints and floats is stored as float64, so it reads back as floats of the same values.

:func:`load_golden` returns exactly the document a reader got before -- nested Python lists of the same
ints, floats and strings -- so readers change only how they open the golden, never what they read. A
golden written inline (hand-authored fixtures, older corpora) loads unchanged.

Both files are answer surfaces: the sandbox masks ``golden.*`` and the transcript audit names both.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DOCUMENT = "golden.yaml"
ARRAYS = "golden.npz"
FILES = (DOCUMENT, ARRAYS)
REFERENCE = "$array"
#: Lists outside ``outputs`` with more elements than this go to the archive; smaller ones (shapes,
#: scales, short provenance) stay readable in the document.
INLINE_LIMIT = 64
_KINDS = {"i": "int", "u": "int", "f": "float", "U": "str", "b": "bool"}


class GoldenStoreError(ValueError):
    """A golden's archive is missing, altered, or not the one its document references."""


def _array(value) -> np.ndarray | None:
    """A rectangular array of one scalar kind, or None when ``value`` is not one (kept inline)."""
    if not isinstance(value, list) or not value:
        return None
    try:
        array = np.array(value)
    except (ValueError, OverflowError):
        return None
    if array.dtype.kind not in _KINDS or array.ndim == 0:
        return None
    flat = [value]
    for _ in range(array.ndim):
        flat = [item for row in flat for item in row]
    kinds = {type(item) for item in flat}
    if array.dtype.kind in "iu" and kinds != {int}:
        return None  # e.g. bools mixed with ints would not round-trip
    if array.dtype.kind == "f" and not kinds <= {float, int}:
        return None
    if array.dtype.kind == "f" and int in kinds and any(isinstance(i, int) and abs(i) > 2**53 for i in flat):
        return None  # an integer a float64 cannot hold exactly stays inline
    if array.dtype.kind in "iu":
        array = array.astype(np.int64)
    return array


def _digest(array: np.ndarray) -> str:
    body = hashlib.sha256()
    body.update(f"{array.dtype.str}|{list(array.shape)}|".encode())
    view = memoryview(np.ascontiguousarray(array)).cast("B")
    for begin in range(0, len(view), 1 << 20):
        body.update(view[begin:begin + (1 << 20)])
    return body.hexdigest()


def split(document: dict) -> tuple[dict, dict[str, np.ndarray]]:
    """``(document with references, {key: array})`` -- the archive part of a golden."""
    arrays: dict[str, np.ndarray] = {}

    def visit(value, *, under_outputs: bool):
        if isinstance(value, dict):
            return {k: visit(v, under_outputs=under_outputs) for k, v in value.items()}
        if isinstance(value, list):
            array = _array(value)
            if array is not None and (under_outputs or array.size > INLINE_LIMIT):
                key = f"a{len(arrays):04d}"
                arrays[key] = array
                return {
                    REFERENCE: ARRAYS,
                    "key": key,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "sha256": _digest(array),
                }
            return [visit(item, under_outputs=under_outputs) for item in value]
        return value

    out = {key: visit(value, under_outputs=(key == "outputs")) for key, value in (document or {}).items()}
    return out, arrays


def _archive_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """A deterministic npz: fixed member order and timestamps, no pickled objects."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, member.getvalue())
    return buffer.getvalue()


def write_golden(directory: str | Path, document: dict) -> Path:
    """Write ``golden.yaml`` and, when it has arrays, ``golden.npz``; return the document's path."""
    directory = Path(directory)
    referenced, arrays = split(document)
    archive = directory / ARRAYS
    if arrays:
        archive.write_bytes(_archive_bytes(arrays))
    elif archive.exists():
        archive.unlink()  # a rewrite with no arrays must not leave a stale archive behind
    path = directory / DOCUMENT
    # The document is small once its arrays are archived, so the standard safe dumper/loader are used:
    # their output and their parse diagnostics are the ones every existing reader already expects.
    path.write_text(yaml.safe_dump(referenced, sort_keys=False), encoding="utf-8")
    return path


def read_document(directory: str | Path):
    """The raw ``golden.yaml`` document, references unresolved (metadata readers); None if absent.

    Returned exactly as parsed: a malformed (non-mapping) document is not normalized, so a caller
    that expects a mapping fails as loudly as it did when it parsed the file itself.
    """
    from merlin.common.yaml import safe_load_text

    path = Path(directory) / DOCUMENT
    if not path.is_file():
        return None
    # Parsed once per its bytes and copied to each reader: a whole-model build reads one capsule's
    # golden from several places, and a legacy inline golden is tens of megabytes of YAML.
    return safe_load_text(path.read_text(encoding="utf-8"))


def _resolve(document, directory: Path, *, compact_arrays=False):
    archive: dict[str, np.ndarray] = {}
    opened = False

    def fetch(ref: dict):
        nonlocal opened
        if ref.get(REFERENCE) != ARRAYS:
            raise GoldenStoreError(f"unknown golden array store {ref.get(REFERENCE)!r}")
        if not opened:
            path = directory / ARRAYS
            if path.is_symlink() or not path.is_file():
                raise GoldenStoreError(f"{directory / DOCUMENT} references arrays but {ARRAYS} is absent")
            with np.load(path, allow_pickle=False) as data:
                archive.update({key: data[key] for key in data.files})
            opened = True
        array = archive.get(str(ref.get("key")))
        if array is None:
            raise GoldenStoreError(f"{ARRAYS} has no array {ref.get('key')!r}")
        if array.dtype.str != ref.get("dtype") or list(array.shape) != ref.get("shape"):
            raise GoldenStoreError(f"{ARRAYS}:{ref.get('key')} is not the referenced dtype/shape")
        if _digest(array) != ref.get("sha256"):
            raise GoldenStoreError(f"{ARRAYS}:{ref.get('key')} does not match the digest its golden names")
        if compact_arrays:
            array.flags.writeable = False
            return array
        return array.tolist()

    def visit(value):
        if isinstance(value, dict):
            if REFERENCE in value:
                return fetch(value)
            return {k: visit(v) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return visit(document)


def load_golden(directory: str | Path, *, compact_arrays=False):
    """The golden document with every array resolved and digest-checked; None if there is no golden."""
    document = read_document(directory)
    if document is None:
        return None
    if type(compact_arrays) is not bool:
        raise GoldenStoreError("golden compact-array selection must be boolean")
    return _resolve(document, Path(directory), compact_arrays=compact_arrays)


def load_golden_file(path: str | Path) -> dict | None:
    """:func:`load_golden` for a caller holding the ``golden.yaml`` path itself."""
    path = Path(path)
    if path.name != DOCUMENT:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
        return _resolve(document, path.parent) if isinstance(document, dict) else document
    return load_golden(path.parent)


def update_golden(directory: str | Path, **fields: Any) -> dict:
    """Add top-level ``fields`` to an existing golden without disturbing its arrays."""
    document = load_golden(directory)
    if document is None:
        raise GoldenStoreError(f"no golden to update in {directory}")
    document.update(fields)
    write_golden(directory, document)
    return document
