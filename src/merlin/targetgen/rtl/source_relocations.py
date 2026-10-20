"""Explicit byte-identical ISA/FIRRTL source portability, never new facts.

The returned private view changes only selected input paths. Original facts and
their byte identity remain untouched; callers must recheck the roster after any
source observer or other callback. This establishes no semantic or execution
qualification and does not relocate hardware, compiler or admission receipts.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path


def _pin(value):
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError("source relocation requires closed source pins")
    path, digest = value["path"], value["sha256"]
    if (
        type(path) is not str
        or not Path(path).is_absolute()
        or "\0" in path
        or type(digest) is not str
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("source relocation requires absolute paths and exact SHA-256 pins")
    return path, digest


def relocate_operand_sources(facts, source_relocations=None):
    """Reopen a closed ISA/FIRRTL relocation roster and return a detached view.

    Rows contain exactly ``original`` and ``replacement`` byte pins. At most 16
    recorded ISA/FIRRTL sources may move; unknown sources, duplicate ownership,
    indirect paths and changed bytes refuse. No original facts are rewritten.
    """
    if type(facts) is not dict:
        raise ValueError("source relocation requires original facts")
    before = copy.deepcopy(facts)
    view = copy.deepcopy(before)
    if source_relocations is None:
        return view
    if type(source_relocations) not in (list, tuple) or len(source_relocations) > 16:
        raise ValueError("source relocation requires a bounded explicit roster")
    frozen = copy.deepcopy(source_relocations)
    inputs = view.get("inputs") or {}
    eligible = {}
    if "isa_path" in inputs:
        eligible[inputs["isa_path"]] = inputs.get("isa_sha256")
    firrtls = inputs.get("firrtl_inputs") or []
    for row in firrtls:
        path, digest = _pin(row)
        if path in eligible:
            raise ValueError("source relocation has ambiguous original source ownership")
        eligible[path] = digest
    seen, destinations = set(), set()
    for row in frozen:
        if type(row) is not dict or set(row) != {"original", "replacement"}:
            raise ValueError("source relocation has unsupported fields")
        old, digest = _pin(row["original"])
        new, replacement_digest = _pin(row["replacement"])
        if (
            old in seen
            or new in destinations
            or new in eligible
            or old not in eligible
            or eligible[old] != digest
            or replacement_digest != digest
        ):
            raise ValueError("source relocation must uniquely preserve an original ISA/FIRRTL byte pin")
        path = Path(new)
        if path.resolve() != path or path.is_symlink() or not path.is_file():
            raise ValueError("source relocation requires a canonical ordinary replacement source")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError("source relocation replacement bytes changed")
        seen.add(old)
        destinations.add(new)
        if inputs.get("isa_path") == old:
            inputs["isa_path"] = new
        for source in firrtls:
            if source["path"] == old:
                source["path"] = new
        if inputs.get("fir_path") == old:
            inputs["fir_path"] = new
    if source_relocations != frozen or facts != before:
        raise ValueError("source relocation original facts or roster changed while reopening its sources")
    return view
