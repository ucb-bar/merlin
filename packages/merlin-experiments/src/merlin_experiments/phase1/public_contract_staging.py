"""Snapshot-owned public ABI copies for restricted author profiles.

This module changes friendly aliases only. Canonical framework paths, host-only
inputs, candidate admission and the native permission policy stay unchanged.
"""
from __future__ import annotations

import hashlib
import stat
from pathlib import Path

from .fresh10_profiles import NAMES as FRESH10_PROFILES
from .author_profiles import NO_EXPERT_NAMES

PROFILES = frozenset(("xs-el-v1/el1", "xs-el-v1/el2", "xs-el-v1/el3", *FRESH10_PROFILES, *NO_EXPERT_NAMES))
PUBLIC_FILES = frozenset((
    "VERSION", "command_buffer_abi.yaml", "interface_dialect_contract.yaml",
    "interface_grammar.md", "integrity_policy.md", "mlir_oot_backend_contract.yaml",
    "target_dialect_contract.yaml",
))
PREFIX = "merlin/contract/"
SCHEMAS = PREFIX + "schemas/"


def selected(bundle, entry):
    """Only the reviewed scratch profiles use this narrow public-data projection."""
    profile = bundle.get("author_profile") or {}
    declared = str(entry["path"])
    return profile.get("name") in PROFILES and (declared == "merlin/contract" or declared.startswith(PREFIX))


def copy_grant(entry, frozen, destination):
    """Copy one exact RO public grant from already verified snapshot bytes.

    The caller must obtain ``frozen`` via ``snapshot_input_paths``; this function
    never resolves a grant against a mutable checkout. Unsupported contract
    grants, aliases, nested directories and links fail closed before any copy.
    """
    declared = str(entry["path"])
    is_schemas = declared == SCHEMAS
    if not is_schemas and declared not in {PREFIX + name for name in PUBLIC_FILES}:
        raise ValueError("unexpected restricted-profile public contract grant")
    alias = "schemas" if is_schemas else Path(declared).name
    if entry.get("mode") != "ro" or entry.get("as", alias) != alias:
        raise ValueError("public contract projection requires its exact RO alias")
    frozen, destination = Path(frozen), Path(destination)
    if frozen.resolve(strict=True) != frozen.absolute():
        raise ValueError("public contract snapshot source must be canonical")
    if destination.exists() or destination.is_symlink():
        raise ValueError("public contract alias must be a fresh destination")
    if destination.name != alias or destination.parent.resolve() != destination.parent.absolute():
        raise ValueError("public contract destination must be its canonical workspace alias")
    if is_schemas:
        if not stat.S_ISDIR(frozen.lstat().st_mode):
            raise ValueError("public schemas grant must be a directory")
        members = sorted(frozen.iterdir())
        if not members:
            raise ValueError("public schemas grant must not be empty")
        for member in members:
            if not stat.S_ISREG(member.lstat().st_mode) or not member.name.endswith(".schema.json"):
                raise ValueError("public schemas grant contains an unexpected member")
    else:
        if not stat.S_ISREG(frozen.lstat().st_mode):
            raise ValueError("public contract grant must be a regular file")
        members = [frozen]
    # Read and validate the complete grant before exposing any destination bytes.
    payloads = [(member.name, member.read_bytes()) for member in members]
    if is_schemas:
        destination.mkdir()
    record = {}
    for name, data in payloads:
        target = destination / name if is_schemas else destination
        target.write_bytes(data)
        target.chmod(0o444)
        if target.read_bytes() != data:
            raise RuntimeError("public contract copy differs from frozen input")
        record[name] = hashlib.sha256(data).hexdigest()
    if is_schemas:
        destination.chmod(0o555)
    return record
