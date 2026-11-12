"""Explicit bounded CFG observation selections, without layout or source authority.

The original LLVM input and layout declaration are caller-selected files. Their
bytes and limits are retained; neither declaration nor observation proves a
same-object DataLayout, ABI, storage, semantics, effects or runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import is_sha256

READER_SCHEMA = "merlin.source_control_flow_reader.v1"
LAYOUT_SCHEMA = "merlin.source_control_flow_layout_declaration.v1"


def _read(path, limit):
    try:
        return _regular_bytes(path, limit)
    except OSError as error:
        raise ValueError("CFG selection input is missing or unavailable") from error


def _regular_bytes(path, limit):
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path.resolve() != path
        or any(member.is_symlink() for member in (path, *path.parents))
        or not stat.S_ISREG(path.lstat().st_mode)
    ):
        raise ValueError("CFG selection requires canonical regular input files")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("CFG selection input ceased to be a regular file")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("CFG selection input exceeds its selected byte bound")
    return raw


def _members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("CFG layout declaration repeats a member")
        result[key] = value
    return result


@dataclass(frozen=True)
class ControlFlowObservationPlan:
    """Closed source/entry/layout declarations and complete reader budgets.

    A declaration's pointer width is an explicit interpretation premise. No
    target, width, entry, DataLayout or resource choice is inferred here.
    """

    lowered_source: Path
    source_sha256: str
    entry_symbol: str
    layout_source: Path
    layout_sha256: str
    pointer_bits: int
    max_source_bytes: int
    max_nesting: int
    max_blocks: int
    max_operations: int
    max_values: int
    max_edges: int
    max_integer_bits: int
    max_layout_bytes: int

    def __post_init__(self):
        self.record()

    def record(self):
        limits = {
            name: getattr(self, name)
            for name in (
                "max_source_bytes",
                "max_nesting",
                "max_blocks",
                "max_operations",
                "max_values",
                "max_edges",
                "max_integer_bits",
                "max_layout_bytes",
            )
        }
        if (
            any(type(value) is not int or value <= 0 for value in limits.values())
            or any(limits[name] >= sys.maxsize for name in ("max_source_bytes", "max_layout_bytes"))
            or limits["max_blocks"] > 1024
            or limits["max_operations"] > 100000
            or limits["max_integer_bits"] > 256
            or type(self.pointer_bits) is not int
            or not 1 <= self.pointer_bits <= 256
            or type(self.entry_symbol) is not str
            or not self.entry_symbol.isascii()
            or not self.entry_symbol.isidentifier()
            or not is_sha256(self.source_sha256)
            or not is_sha256(self.layout_sha256)
        ):
            raise ValueError("CFG observation requires explicit positive supported entry/width/reader limits")
        source = _read(self.lowered_source, self.max_source_bytes)
        layout = _read(self.layout_source, self.max_layout_bytes)
        if (
            hashlib.sha256(source).hexdigest() != self.source_sha256
            or hashlib.sha256(layout).hexdigest() != self.layout_sha256
        ):
            raise ValueError("CFG selected LLVM or layout declaration bytes changed")
        try:
            declaration = json.loads(layout, object_pairs_hook=_members)
        except (ValueError, UnicodeError) as error:
            raise ValueError("CFG layout declaration is not complete bounded JSON") from error
        if (
            type(declaration) is not dict
            or set(declaration) != {"schema", "data_layout", "pointer_bits"}
            or declaration["schema"] != LAYOUT_SCHEMA
            or type(declaration["pointer_bits"]) is not int
            or declaration["pointer_bits"] != self.pointer_bits
            or type(declaration["data_layout"]) is not str
            or not declaration["data_layout"]
            or not declaration["data_layout"].isascii()
            or any(character.isspace() for character in declaration["data_layout"])
        ):
            raise ValueError("CFG observation requires its exact explicit layout/width declaration")
        return {
            "schema": "merlin.source_control_flow_observation_plan.v1",
            "lowered_source": {"path": str(self.lowered_source), "sha256": self.source_sha256},
            "entry_symbol": self.entry_symbol,
            "layout_source": {"path": str(self.layout_source), "sha256": self.layout_sha256},
            "layout_declaration": declaration,
            "limits": limits,
            "unknown": ["same_object_data_layout", "abi_storage_correspondence", "compiled_resources"],
            "scope": "explicit source/layout declarations and observation limits only; no physical or semantic grant",
        }

    def emitted_text(self, path, *, entry_symbol, pointer_bits, max_operations):
        """Join actual ordinary input bytes to this independently selected plan."""
        self.record()
        if (
            type(entry_symbol) is not str
            or type(pointer_bits) is not int
            or type(max_operations) is not int
            or (entry_symbol, pointer_bits, max_operations)
            != (self.entry_symbol, self.pointer_bits, self.max_operations)
        ):
            raise ValueError("CFG caller changes the selected original entry/width/operation bound")
        raw = _read(path, self.max_source_bytes)
        if hashlib.sha256(raw).hexdigest() != self.source_sha256:
            raise ValueError("CFG caller changes the selected complete LLVM input")
        try:
            return raw.decode("utf-8")
        except UnicodeError as error:
            raise ValueError("CFG selected LLVM has no complete UTF-8 representation") from error
