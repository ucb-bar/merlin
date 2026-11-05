"""Author-only numerical test views; original judge inputs are never modified.

The returned custody receipt is private operator metadata, not an author grant.
This projection removes generated target advice, not mathematical specifications.
It does not translate source languages or qualify a source as namespace-neutral.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import yaml

_REMOVED_KEYS = frozenset(
    {"source_reference", "software_screen", "integer_partial_sum_bound", "pass_requirements", "source_role"}
)
_OMITTED_NAMES = frozenset(
    {
        "expected_instruction_coverage.yaml",
        "frontend-evidence.json",
        "frontend-trace.json",
        "pytorch-opset.json",
        "source-capture-receipt.json",
    }
)
_METADATA_NAMES = frozenset({"capsule.yaml", "suite.json", "profile.json"})
_DATA_NAMES = frozenset({"inputs.npz", "inputs.npy"})
_SOURCE_NAMES = frozenset({"capsule.pytorch.py"})
_ANNOTATIONS = (*sorted(_REMOVED_KEYS), "generalization_axis", "rtl_facts", "facts.json")


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in mapping:
            raise ValueError("author metadata requires unique string mapping keys")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate author metadata key")
        result[key] = value
    return result


def _project(value, path, removed):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            current = (*path, key)
            if (
                key in _REMOVED_KEYS
                or (path and path[-1] == "expected" and key in {"instruction_classes", "modes"})
                or (path and path[-1] == "semantic" and key == "generalization_axis")
            ):
                removed.append("/".join(current))
            else:
                result[key] = _project(item, current, removed)
        return result
    if isinstance(value, list):
        return [_project(item, (*path, str(index)), removed) for index, item in enumerate(value)]
    return value


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kind(relative):
    name = relative.name
    # Keep parity with workspace_transport._is_answer_file even when the view's
    # canonical path is granted directly, bypassing friendly workspace copies.
    if any(
        part == "hidden"
        or part.startswith("golden.")
        or ".golden." in part
        or part.startswith("expected_command_buffer")
        or part == "expected_instruction_coverage.yaml"
        or part.endswith(".safetensors")
        or part.endswith(".safetensors.manifest.json")
        for part in relative.parts
    ):
        return "omitted"
    if name.lower() == "readme" or name.lower().startswith("readme.") or name in _OMITTED_NAMES:
        return "omitted"
    if name in _METADATA_NAMES:
        return "metadata"
    if name in _DATA_NAMES:
        return "data"
    if name in _SOURCE_NAMES or relative.suffix == ".mlir":
        return "source"
    raise ValueError(f"unclassified author input: {relative.as_posix()}")


def _view_relative(value):
    if not isinstance(value, str):
        raise ValueError("transformed author path must be a string")
    result = Path(value)
    if not result.parts or result.is_absolute() or ".." in result.parts or result.as_posix() != value:
        raise ValueError("transformed author path must be a canonical nonempty relative path")
    try:
        kind = _kind(result)
    except ValueError:
        if not result.name.endswith(".program.json"):
            raise
        kind = "source"
    if kind == "omitted":
        raise ValueError("transform cannot publish a withheld file family")
    return result


def project_tree(source: Path, destination: Path, *, file_transform=None) -> dict:
    """Publish a fresh readonly view with fail-closed inventory and byte custody.

    Only the explicit numerical corpus file families above are admitted. Unknown
    files, symlinks (including omitted files), and generated annotations in MLIR
    fail before a destination is created. All source and tensor bytes remain
    identical; metadata preserves every value outside the listed disclosure keys.
    An explicit trusted host transform may replace/omit admitted source files and
    rewrite their capsule references before publication. It cannot resurrect any
    answer file. Its new paths and hashes are recorded separately in the receipt.
    """
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if source.is_symlink() or not source.is_dir() or source.resolve() != source:
        raise ValueError("author input source must be a canonical ordinary directory")
    if destination.exists() or destination.is_symlink() or destination.parent.resolve() != destination.parent:
        raise ValueError("author input destination must be fresh with canonical parent")
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("author input source and destination must be disjoint")
    payloads, mapping = {}, []
    for directory, names, files in os.walk(source, followlinks=False):
        for name in sorted([*names, *files]):
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError(f"nonordinary author input: {path.relative_to(source).as_posix()}")
        names.sort()
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(source)
            kind = _kind(relative)
            row = {
                "path": relative.as_posix(),
                "kind": kind,
                "source_sha256": _file_sha(path),
                "source_bytes": path.stat().st_size,
            }
            if kind == "omitted":
                row.update(view_path=None, view_sha256=None, view_bytes=None, removed_fields=[], transformed=False)
            else:
                removed = []
                raw, view_relative = None, relative
                transformed = file_transform(relative.as_posix(), path.read_bytes()) if file_transform else None
                if file_transform and transformed is None:
                    row.update(view_path=None, view_sha256=None, view_bytes=None, removed_fields=[], transformed=True)
                    mapping.append(row)
                    continue
                if file_transform:
                    if (
                        not isinstance(transformed, tuple)
                        or len(transformed) != 2
                        or not isinstance(transformed[1], bytes)
                    ):
                        raise ValueError("author file transform must return (relative_path, bytes) or None")
                    view_relative, raw = _view_relative(transformed[0]), transformed[1]
                if kind == "metadata":
                    raw = path.read_bytes() if raw is None else raw
                    value = (
                        json.loads(raw, object_pairs_hook=_pairs)
                        if view_relative.suffix == ".json"
                        else yaml.load(raw, Loader=_UniqueLoader)
                    )
                    value = _project(value, (), removed)
                    raw = (
                        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
                        if view_relative.suffix == ".json"
                        else yaml.safe_dump(value, sort_keys=True, allow_unicode=True)
                    ).encode()
                elif kind == "source" and path.suffix == ".mlir":
                    text = (path.read_bytes() if raw is None else raw).decode("utf-8")
                    if any(marker in text for marker in _ANNOTATIONS):
                        raise ValueError(f"generated annotation in author source: {relative.as_posix()}")
                if view_relative in payloads:
                    raise ValueError("author transform has colliding destination paths")
                payloads[view_relative] = (relative, raw)
                row.update(
                    view_path=view_relative.as_posix(),
                    view_sha256=_sha(raw) if raw is not None else row["source_sha256"],
                    view_bytes=len(raw) if raw is not None else row["source_bytes"],
                    removed_fields=sorted(removed),
                    transformed=file_transform is not None,
                )
            mapping.append(row)
    mapping.sort(key=lambda row: row["path"])
    if any(parent in payloads for relative in payloads for parent in relative.parents if parent.parts):
        raise ValueError("author transform has conflicting file and directory paths")
    destination.mkdir(mode=0o700)
    by_path = {row["view_path"]: row for row in mapping if row["view_path"] is not None}
    for view_relative, (relative, raw) in sorted(payloads.items()):
        target = destination / view_relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with target.open("xb") as stream:
            if raw is not None:
                stream.write(raw)
            else:
                with (source / relative).open("rb") as original:
                    for chunk in iter(lambda: original.read(1024 * 1024), b""):
                        stream.write(chunk)
        if _file_sha(target) != by_path[view_relative.as_posix()]["view_sha256"]:
            raise ValueError("author input changed during projection")
        target.chmod(0o444)
    for directory, _, _ in os.walk(destination, topdown=False):
        Path(directory).chmod(0o555)
    identity = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "merlin.phase1.author_input_view.v1",
        "source": str(source),
        "destination": str(destination),
        "sha256": _sha(identity),
        "files": mapping,
    }
