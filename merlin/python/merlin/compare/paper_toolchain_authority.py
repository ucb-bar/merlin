"""Independent build-tool identity for paper package reconstruction.

Package templates are producer outputs, so their own ``build_tool`` hashes cannot establish which
compiler was trusted.  A paper freeze therefore takes a separately reviewed authority document and
pins its digest in the canonical frozen study.  Contracts retain that document, and every process
verifies both the authority digest and exact compiler identity before materializing an executable.
Recipes that rebuild execute it; the sealed ExecuTorch recipe verifies it only as host-producer
provenance because the x86-hosted cross compiler cannot and must not run on K1.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


AUTHORITY_KIND = "paper_toolchain_authority_v1"
TOOL_ROLE = "model_object_c_compiler"
_HEX = frozenset("0123456789abcdef")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _is_sha(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in _HEX for character in text)


def tool_identity(*, authority_id: str, target: str, sha256: str) -> str:
    """Derive the closed compiler identity; it is never an authored free-form value."""
    return _canonical_sha({
        "authority_id": authority_id, "target": target, "role": TOOL_ROLE,
        "sha256": sha256,
    })


def write_toolchain_authority(path: str | Path, *, authority_id: str, target: str,
                              build_tool: str | Path) -> Path:
    """Write a reviewable authority file outside package/template construction.

    This helper records bytes; it does not make them trustworthy.  Paper protocol review must pin
    the returned file independently and pass it explicitly to the freeze command.
    """
    path, build_tool = Path(path).resolve(), Path(build_tool).resolve()
    if not authority_id.strip() or not target.strip():
        raise ValueError("paper toolchain authority id/target must be non-empty")
    if (not build_tool.is_file() or build_tool.is_symlink()
            or not os.access(build_tool, os.X_OK)
            or not build_tool.read_bytes().startswith(b"\x7fELF")):
        raise ValueError("paper toolchain authority requires an executable ELF compiler")
    digest = _sha(build_tool)
    document = {
        "schema_version": 1, "kind": AUTHORITY_KIND, "status": "frozen",
        "authority_id": authority_id, "target": target,
        "tool": {
            "role": TOOL_ROLE, "path": str(build_tool), "sha256": digest,
            "identity_sha256": tool_identity(
                authority_id=authority_id, target=target, sha256=digest),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    return path


def load_toolchain_authority(path: str | Path, *, expected_sha256: str | None = None,
                             expected_target: str | None = None) -> dict[str, Any]:
    """Load one closed authority and verify its externally supplied document digest."""
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("paper toolchain authority is absent or unsafe")
    actual = _sha(path)
    if expected_sha256 is not None and (
            not _is_sha(expected_sha256) or actual != expected_sha256):
        raise ValueError("paper toolchain authority digest differs from frozen authority")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("paper toolchain authority is not valid JSON") from error
    fields = {"schema_version", "kind", "status", "authority_id", "target", "tool"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("paper toolchain authority is not a closed document")
    tool = value["tool"]
    tool_fields = {"role", "path", "sha256", "identity_sha256"}
    if not isinstance(tool, Mapping) or set(tool) != tool_fields:
        raise ValueError("paper toolchain authority tool is not a closed document")
    authority_id, target, digest = (str(value["authority_id"]), str(value["target"]),
                                    str(tool["sha256"]))
    if (value["schema_version"] != 1 or value["kind"] != AUTHORITY_KIND
            or value["status"] != "frozen" or not authority_id.strip() or not target.strip()
            or (expected_target is not None and target != expected_target)
            or tool["role"] != TOOL_ROLE or not _is_sha(digest)
            or tool["identity_sha256"] != tool_identity(
                authority_id=authority_id, target=target, sha256=digest)
            or not isinstance(tool["path"], str) or not tool["path"]):
        raise ValueError("paper toolchain authority identity is invalid")
    return dict(value)


def verify_build_tool(build_tool: str | Path, *, authority_path: str | Path,
                      authority_sha256: str, target: str,
                      expected_identity_sha256: str | None = None) -> str:
    """Verify authority, compiler bytes, and derived identity before compiler execution."""
    authority = load_toolchain_authority(
        authority_path, expected_sha256=authority_sha256, expected_target=target)
    build_tool = Path(build_tool).resolve()
    if (not build_tool.is_file() or build_tool.is_symlink()
            or not os.access(build_tool, os.X_OK)
            or not build_tool.read_bytes().startswith(b"\x7fELF")):
        raise ValueError("paper build tool is not an executable ELF")
    if _sha(build_tool) != authority["tool"]["sha256"]:
        raise ValueError("paper build tool differs from independent toolchain authority")
    identity = str(authority["tool"]["identity_sha256"])
    if expected_identity_sha256 is not None and identity != expected_identity_sha256:
        raise ValueError("paper build-tool identity differs from frozen authority identity")
    return identity


__all__ = [
    "AUTHORITY_KIND", "TOOL_ROLE", "load_toolchain_authority", "tool_identity",
    "verify_build_tool", "write_toolchain_authority",
]
