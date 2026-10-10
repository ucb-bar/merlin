"""Closed, bounded operator requests for conditional plain source-word relations.

Every source and packing hypothesis is explicitly selected. This command emits
an observation only; it supplies no decoder, instruction or runtime authority.
"""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from dataclasses import fields
from pathlib import Path

from merlin.common.strict_json import loads

from . import plain_word_relation as W

REQUEST_SCHEMA = "merlin.plain_word_source_request.v1"


def _path(value):
    if not isinstance(value, Path) and (type(value) is not str or not value):
        raise W.WordRelationError("word observation requires an explicit file path")
    path = Path(value).absolute()
    if ".." in path.parts or any(member.is_symlink() for member in (path, *path.parents)):
        raise W.WordRelationError("word observation requires canonical unlinked paths")
    try:
        metadata = path.stat()
    except FileNotFoundError:
        # Only an absent output can be created. Missing inputs still refuse in
        # the bounded reader without acquiring source membership.
        return path
    except OSError as error:
        raise W.WordRelationError("word observation path is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise W.WordRelationError("word observation path must be a regular file")
    return path


def _bound(value):
    if type(value) is not int or value <= 0:
        raise W.WordRelationError("word observation byte bounds must be explicit positive integers")
    if value >= sys.maxsize:
        raise W.WordRelationError("word observation byte bound is unavailable to the bounded reader")


def _read(path, limit):
    _bound(limit)
    path = _path(path)
    try:
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
    except (OSError, OverflowError) as error:
        raise W.WordRelationError("word observation input is unavailable") from error
    if len(raw) > limit:
        raise W.WordRelationError("word observation input exceeds its byte bound")
    _path(path)
    return raw


def _record(cls, value):
    if type(value) is not dict or set(value) != {field.name for field in fields(cls)}:
        raise W.WordRelationError("word observation request has an incomplete or unsupported field roster")
    return dict(value)


def _span(value):
    raw = _record(W.WordSourceSpan, value)
    if type(raw["path"]) is not str or not Path(raw["path"]).is_absolute():
        raise W.WordRelationError("selected source members require explicit absolute paths")
    raw["path"] = _path(raw["path"])
    return W.WordSourceSpan(**raw)


def _selection(request):
    if (
        type(request) is not dict
        or set(request) != {"schema", "declaration", "cast", "semantics", "limits"}
        or request["schema"] != REQUEST_SCHEMA
    ):
        raise W.WordRelationError("word observation request schema is unsupported")
    limits = W.WordRelationLimits(**_record(W.WordRelationLimits, request["limits"]))
    limits.verify()
    declaration = _record(W.WordDeclarationSelection, request["declaration"])
    declaration["source"] = _span(declaration["source"])
    cast = _record(W.WordCastSelection, request["cast"])
    cast["source"] = _span(cast["source"])
    semantics = _record(W.WordSourceSemantics, request["semantics"])
    if (
        type(semantics["primitives"]) is not list
        or not 0 < len(semantics["primitives"]) <= limits.fields
        or type(semantics["premises"]) is not list
        or len(semantics["premises"]) != 6
    ):
        raise W.WordRelationError("word observation primitive/packing roster is unavailable")
    primitives = []
    for value in semantics["primitives"]:
        raw = _record(W.WordPrimitive, value)
        raw["construction_source"] = _span(raw["construction_source"])
        raw["width_source"] = _span(raw["width_source"])
        primitives.append(W.WordPrimitive(**raw))
    premises = []
    for value in semantics["premises"]:
        raw = _record(W.WordPackingPremise, value)
        raw["source"] = _span(raw["source"])
        premises.append(W.WordPackingPremise(**raw))
    semantics["primitives"], semantics["premises"] = tuple(primitives), tuple(premises)
    return {
        "declaration": W.WordDeclarationSelection(**declaration),
        "cast": W.WordCastSelection(**cast),
        "semantics": W.WordSourceSemantics(**semantics),
        "limits": limits,
    }


def _sources_unchanged(observation, limit):
    for pin in observation["source_membership"]["files"]:
        if hashlib.sha256(_read(Path(pin["path"]), limit)).hexdigest() != pin["sha256"]:
            raise W.WordRelationError("selected source bytes changed before publication")


def _render(value, limit):
    _bound(limit)
    raw = bytearray()
    for fragment in json.JSONEncoder(indent=2, sort_keys=True, allow_nan=False).iterencode(value):
        encoded = fragment.encode("utf-8")
        if len(raw) + len(encoded) + 1 > limit:
            raise W.WordRelationError("word observation output exceeds its byte bound")
        raw.extend(encoded)
    raw.extend(b"\n")
    return raw


def write_plain_word_observation(*, request, out, max_request_bytes, max_output_bytes):
    """Reopen an exact closed request/source roster and write one fresh observation.

    Paths and bytes are checked at this bounded command boundary. Namespace race
    isolation, source-semantic review and all decoder/ISA premises remain unknown.
    """
    request, out = _path(request), _path(out)
    _bound(max_output_bytes)
    raw = _read(request, max_request_bytes)
    try:
        selected = _selection(loads(raw, max_bytes=max_request_bytes))
    except (ValueError, RecursionError) as error:
        raise W.WordRelationError("word observation request cannot be interpreted exactly") from error
    observation = W.observe_plain_word_relation(**selected)
    observation["request"] = {
        "schema": REQUEST_SCHEMA,
        "path": str(request),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "max_request_bytes": max_request_bytes,
        "max_output_bytes": max_output_bytes,
    }
    rendered = _render(observation, max_output_bytes)
    if _read(request, max_request_bytes) != raw:
        raise W.WordRelationError("word observation request changed before publication")
    _sources_unchanged(observation, selected["limits"].source_bytes)
    _path(out)
    try:
        with out.open("xb") as stream:
            stream.write(rendered)
    except OSError as error:
        raise W.WordRelationError("word observation destination must be a fresh available file") from error
    _path(out)
    if _read(request, max_request_bytes) != raw:
        raise W.WordRelationError("word observation request changed during publication")
    _sources_unchanged(observation, selected["limits"].source_bytes)
    if _read(out, max_output_bytes) != rendered:
        raise W.WordRelationError("word observation output changed during publication")
    return observation
