"""Issue hash-only private exclusions from original trusted operator inventories.

There is no candidate-source hook, public file discovery, empty-list fallback or
compiler qualification. Output is an exclusive owner-only leaf outside every
author/publication root. A complete issuance refuses on any missing source.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import operator_exclusion as E
from . import operator_exclusion_intake as I
from . import operator_typed_exclusion as T


class IssuanceRefusal(ValueError):
    """Operator metadata only; never include protected names, shapes or answers."""


def _destination(path, forbidden_roots):
    path = Path(path)
    if (not path.is_absolute() or ".." in path.parts or path.exists()
            or any(p.is_symlink() for p in (path, *path.parents)) or not path.parent.is_dir()
            or path.parent.stat().st_mode & 0o077
            or not isinstance(forbidden_roots, (list, tuple)) or not forbidden_roots):
        raise IssuanceRefusal("private exclusion destination is not a fresh owner-only operator leaf")
    for root in forbidden_roots:
        if not Path(root).is_absolute() or path.is_relative_to(Path(root).resolve()):
            raise IssuanceRefusal("private exclusion destination overlaps an author or publication root")
    return path


def issue(pointer_path, pointer_sha256, destination, *, forbidden_roots, max_ir_bytes, max_total_ir_bytes):
    """Resolve original lineage and hash all typed source programs under bounds.

    No hashes or file are issued until the complete selected source roster has
    passed. These limits deny work; increasing them is an explicit operator
    resource choice and never a semantic waiver. The returned cost receipt has
    only aggregate counts/bytes and pins, and is not a numerical admission.
    """
    destination = _destination(destination, forbidden_roots)
    if (type(max_ir_bytes) is not int or max_ir_bytes < 1
            or type(max_total_ir_bytes) is not int or max_total_ir_bytes < max_ir_bytes):
        raise IssuanceRefusal("protected IR intake needs explicit finite parse ceilings")
    try:
        sources = I.resolve(pointer_path, pointer_sha256)
    except Exception:
        raise IssuanceRefusal(
            "protected original inventory lineage or complete source ownership was not verified"
        ) from None
    try:
        if not sources or len(sources) > 8192:
            raise ValueError("incomplete or oversized source roster")
        paths = {I.ordinary(path): sha for path, sha in sources.items()}
        sizes = {path: path.stat().st_size for path in paths}
    except Exception:
        raise IssuanceRefusal("complete protected original source roster was not verified") from None
    if max(sizes.values()) > max_ir_bytes or sum(sizes.values()) > max_total_ir_bytes:
        raise IssuanceRefusal("complete protected typed IR exceeds the selected per-source or aggregate parse budget")
    source_hashes, typed_hashes = set(), set()
    try:
        for path, sha in paths.items():
            raw = path.read_bytes()
            if len(raw) != sizes[path] or hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError("source drift")
            source_hashes.add(sha)
            typed_hashes.update(T.signatures(raw, max_bytes=max_ir_bytes))
        if I.resolve(pointer_path, pointer_sha256) != sources:
            raise ValueError("inventory drift")
        for path, sha in paths.items():
            if hashlib.sha256(I.ordinary(path).read_bytes()).hexdigest() != sha:
                raise ValueError("original source drift before issuance")
    except Exception:
        raise IssuanceRefusal("complete protected typed IR projection was not verified") from None
    data = {"schema": E.SCHEMA_V2, "excluded_source_sha256": sorted(source_hashes),
            "excluded_contraction_sha256": [], "excluded_typed_ir_sha256": sorted(typed_hashes)}
    if len(source_hashes) > 8192 or len(typed_hashes) > 8192:
        raise IssuanceRefusal("complete protected hash roster exceeds the selected reader format")
    raw = (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()
    if len(raw) > 1024 * 1024:
        raise IssuanceRefusal("complete protected hash record exceeds the selected reader format")
    _destination(destination, forbidden_roots)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        sha = hashlib.sha256(raw).hexdigest()
        selection = E.select(destination, sha)
        E.require_selection(selection, *forbidden_roots)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return selection, {
        "schema": "merlin.phase0.operator_private_exclusion_issuance.v1",
        "status": "issued_hash_only_private_source_exclusion",
        "original_inventory_sha256": pointer_sha256,
        "selection_sha256": sha,
        "protected_source_count": len(paths), "protected_source_bytes": sum(sizes.values()),
        "distinct_source_hashes": len(source_hashes), "distinct_typed_ir_hashes": len(typed_hashes),
        "parse_limits": {"max_ir_bytes": max_ir_bytes, "max_total_ir_bytes": max_total_ir_bytes},
        "qualification": "exclusion only; no compiler, numerical, hardware, timing or cohort admission",
        "bundle_intake": "complete original leaf index and required source/ownership pins; tensor answers unread",
    }
