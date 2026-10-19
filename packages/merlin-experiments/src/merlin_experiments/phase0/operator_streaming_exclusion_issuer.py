"""Explicit bounded streaming issuer of the unchanged hash-only V2 exclusions.

Original source byte seals, complete roster and catalog-role aliases remain
mandatory. Streaming initializer intake changes allocation, never the typed
signature algorithm, selected numerical policy, or publication qualification.
"""

from __future__ import annotations

import hashlib
import json
import os

from merlin.common.digest import sha256_file

from . import operator_dense_projection as P
from . import operator_exclusion as E
from . import operator_exclusion_intake as I
from . import operator_exclusion_issuer as U
from . import operator_mixed_exclusion as M
from . import operator_typed_exclusion as T


def issue(pointer_path, pointer_sha256, destination, *, forbidden_roots,
          max_ir_bytes, max_total_ir_bytes, max_source_bytes, max_total_source_bytes, source_mode=None):
    """Use separate streaming-input bounds without increasing parser ceilings.

    No protected source, symbol, shape, initializer value or exclusion hash set
    enters returned diagnostics. No partial selection is written. The live
    selection is still the original V2 reader, and every hash uses the unchanged
    typed_ir_exclusion_signature.v1 projection.
    """
    destination = U._destination(destination, forbidden_roots)
    if source_mode not in {None, M.SCHEMA}:
        raise U.IssuanceRefusal("protected streaming intake has no selected reviewed source grammar")
    limits = (max_ir_bytes, max_total_ir_bytes, max_source_bytes, max_total_source_bytes)
    if (any(type(n) is not int or n < 1 for n in limits) or max_total_ir_bytes < max_ir_bytes
            or max_total_source_bytes < max_source_bytes):
        raise U.IssuanceRefusal("protected streaming intake needs explicit finite input and parse ceilings")
    try:
        sources = I.resolve_streaming_v2(pointer_path, pointer_sha256)
        if not sources or len(sources) > 8192:
            raise ValueError("source roster unavailable")
        for row in sources.values():
            I.ordinary(row["read_path"])
            if (set(row) != {"sha256", "read_path", "alias"} or len(row["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in row["sha256"])):
                raise ValueError("source leaf binding invalid")
    except Exception:
        raise U.IssuanceRefusal(
            "protected original streamed inventory or complete catalog ownership was not verified"
        ) from None
    source_hashes, typed_hashes = set(), set()
    source_bytes = compact_bytes = constants = hex_bytes = 0
    try:
        for row in sources.values():
            path = I.ordinary(row["read_path"])
            before = path.stat()
            available = min(max_source_bytes, max_total_source_bytes - source_bytes)
            compact_available = min(max_ir_bytes, max_total_ir_bytes - compact_bytes)
            if available < before.st_size or compact_available < 1:
                raise ValueError("streaming source or aggregate parse cost denied")
            with path.open("rb") as stream:
                prefix = stream.read(4096)
                stream.seek(0)
                if source_mode == M.SCHEMA and b"merlin_iface.version" in prefix:
                    if before.st_size > compact_available:
                        raise ValueError("raw interface source exceeds unchanged compact parse budget")
                    raw = stream.read(compact_available + 1)
                    if len(raw) > compact_available:
                        raise ValueError("raw interface source changed beyond the compact parse budget")
                    projected = P.Projection(hashlib.sha256(raw).hexdigest(), len(raw), raw, 0, 0)
                else:
                    projected = P.project(stream, max_input_bytes=available, max_compact_bytes=compact_available)
            after = path.stat()
            if (projected.source_sha256 != row["sha256"] or projected.source_bytes != before.st_size
                    or any(getattr(before, key) != getattr(after, key)
                           for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))):
                raise ValueError("selected original source changed")
            source_bytes += projected.source_bytes
            compact_bytes += len(projected.compact)
            constants += projected.projected_constants
            hex_bytes += projected.hexadecimal_payload_bytes
            source_hashes.add(projected.source_sha256)
            signature = M.signatures if source_mode == M.SCHEMA else T.signatures
            typed_hashes.update(signature(projected.compact, max_bytes=max_ir_bytes))
            if len(source_hashes) > 8192 or len(typed_hashes) > 8192:
                raise ValueError("complete hash roster exceeds reader bound")
        if I.resolve_streaming_v2(pointer_path, pointer_sha256) != sources:
            raise ValueError("selected original inventory or alias binding changed")
        for row in sources.values():
            if sha256_file(I.ordinary(row["read_path"])) != row["sha256"]:
                raise ValueError("selected original source changed before issuance")
    except Exception:
        raise U.IssuanceRefusal("complete protected streamed typed projection was not verified") from None
    data = dict(schema=E.SCHEMA_V3 if source_mode == M.SCHEMA else E.SCHEMA_V2,
                excluded_source_sha256=sorted(source_hashes),
                excluded_contraction_sha256=[], excluded_typed_ir_sha256=sorted(typed_hashes))
    raw = (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()
    if len(raw) > 1 << 20:
        raise U.IssuanceRefusal("complete streamed private hash record exceeds the reader bound")
    U._destination(destination, forbidden_roots)
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
    return selection, dict(schema="merlin.phase0.operator_streaming_exclusion_issuance.v1",
        status="issued_complete_hash_only_private_source_exclusion",
        original_inventory_sha256=pointer_sha256, selection_sha256=sha,
        protected_source_count=len(sources), protected_source_bytes=source_bytes,
        distinct_source_hashes=len(source_hashes), distinct_typed_ir_hashes=len(typed_hashes),
        catalog_bound_source_aliases=sum(row["alias"] is not None for row in sources.values()),
        compact_source_bytes=compact_bytes, checked_initializer_count=constants,
        checked_hexadecimal_payload_bytes=hex_bytes,
        signature_schema=M.SCHEMA if source_mode == M.SCHEMA else T.SCHEMA, typed_signature_schema=T.SCHEMA,
        parse_limits=dict(max_ir_bytes=max_ir_bytes, max_total_ir_bytes=max_total_ir_bytes),
        streaming_limits=dict(max_source_bytes=max_source_bytes, max_total_source_bytes=max_total_source_bytes),
        qualification="private exclusion only; no numerical, compiler, hardware, timing or cohort admission",
        alias_scope=("original sealed source hash and catalog namespace; current ordinary target rechecked, "
                     "historical link spelling not attested"))
