"""Reference artifact roles shared by capsule producers and candidate input views.

This classifies protected files, never reads their tensor payloads and grants no
capture, numerical or execution authority. Legacy capture schemas keep their
original reference role even when they predate the explicit closure field.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REFERENCE_SCHEMA = "merlin.capsule_reference_closure.v1"
_CAPTURE_SCHEMA = "merlin.materialized_public_capture.v2"
_CAPTURE_REFERENCE = "golden.npy"
_CAPTURE_PREFIX = "source-capture-"


def capture_reference_closure(identities: dict) -> dict:
    """Declare the exact expected-output artifact copied by the capture writer."""
    selected = identities[_CAPTURE_REFERENCE]
    return {
        "schema": REFERENCE_SCHEMA,
        "members": [{"path": _CAPTURE_PREFIX + _CAPTURE_REFERENCE, "role": "expected_output",
                     "sha256": selected["sha256"], "bytes": selected["bytes"]}],
    }


def _relative_member(value) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or "\x00" in value
        or Path(value).is_absolute() or any(part in {".", ".."} for part in value.split("/"))
        or any(not part for part in value.split("/"))):
        raise ValueError("capsule reference member must be a safe relative file")
    return value


def capsule_answer_members(directory: Path, *, capsule: dict | None = None) -> dict[str, str]:
    """Return relative protected members and their roles from this exact capsule.

    Read only capsule metadata and file names, so callers can inspect a frozen
    copy independently of missing or changed original sources. Declared byte pins
    describe identity; this privacy classification does not verify those bytes.
    """
    directory = Path(directory)
    if capsule is None:
        source = directory / "capsule.yaml"
        capsule = yaml.safe_load(source.read_bytes()) if source.is_file() else {}
    if not isinstance(capsule, dict):
        raise ValueError("capsule privacy requires a mapping")
    members = {path.name: "golden" for path in directory.glob("golden.*") if path.is_file()}
    coverage = directory / "expected_instruction_coverage.yaml"
    if coverage.is_file():
        members[coverage.name] = "golden"
    members.update({path.name: "weight" for path in directory.iterdir()
                    if path.is_file() and path.name.endswith((".safetensors", ".safetensors.manifest.json"))})
    capture = capsule.get("materialized_capture") or {}
    if not isinstance(capture, dict):
        raise ValueError("capsule materialized capture is malformed")
    if capture.get("schema") == _CAPTURE_SCHEMA and capture.get("workload_role") == "performance_scale":
        members[_CAPTURE_PREFIX + _CAPTURE_REFERENCE] = "golden"
    closure = capsule.get("reference_closure")
    if closure is None:
        return members
    if (not isinstance(closure, dict) or set(closure) != {"schema", "members"}
        or closure.get("schema") != REFERENCE_SCHEMA or not isinstance(closure.get("members"), list)
        or not closure["members"]):
        raise ValueError("capsule reference closure is malformed")
    seen = set()
    for row in closure["members"]:
        if not isinstance(row, dict) or set(row) != {"path", "role", "sha256", "bytes"}:
            raise ValueError("capsule reference artifact is malformed")
        path = _relative_member(row["path"])
        digest = row["sha256"]
        if (path in seen or row["role"] != "expected_output"
            or not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(row["bytes"]) is not int or row["bytes"] < 0):
            raise ValueError("capsule reference artifact role or identity is invalid")
        seen.add(path)
        members[path] = "golden"
    return members
