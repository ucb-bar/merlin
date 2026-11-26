"""Read buffer mutation declarations from a byte-bound saved-program archive.

Only the reviewed serialized graph signature is interpreted. Tensor payloads and
graph code are never deserialized or executed, and no framework import is needed.
"""

from __future__ import annotations

import json
import os
import stat
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath

from merlin.common.digest import is_sha256

_SCHEMA_VERSION = (8, 17)
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_MEMBERS = 4096


@dataclass(frozen=True)
class SavedBufferSignature:
    argument_name: str
    buffer_target: str
    mutated: bool


@dataclass(frozen=True)
class SavedProgramSignature:
    archive_sha256: str
    signature_member: str
    schema_version: tuple[int, int]
    buffers: tuple[SavedBufferSignature, ...]
    buffer_mutations: tuple[str, ...]
    parameter_mutations: tuple[str, ...]
    user_input_mutations: tuple[str, ...]


def _fail(reason):
    raise ValueError("saved program signature: " + reason)


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        _fail(label + " has missing or unknown fields")
    return value


def _name(value):
    if type(value) is not str or not value.strip():
        _fail("argument and target names must be explicit strings")
    return value


def _tensor(value):
    return _name(_fields(value, {"name"}, "tensor argument")["name"])


def _user_tensor(value):
    return _tensor(_fields(value, {"as_tensor"}, "user tensor argument")["as_tensor"])


def _signature(document):
    version = document.get("schema_version") if isinstance(document, dict) else None
    if (
        not isinstance(version, dict)
        or set(version) != {"major", "minor"}
        or any(type(version[key]) is not int for key in ("major", "minor"))
        or (version["major"], version["minor"]) != _SCHEMA_VERSION
    ):
        _fail("unreviewed schema_version")
    module = document.get("graph_module")
    signature = _fields(
        module.get("signature") if isinstance(module, dict) else None,
        {"input_specs", "output_specs"},
        "graph signature",
    )
    if (
        not isinstance(signature["input_specs"], list)
        or not isinstance(signature["output_specs"], list)
        or not signature["output_specs"]
    ):
        _fail("signature inputs and nonempty outputs must be explicit lists")
    arguments, targets, inputs, buffers = set(), set(), {"parameter": set(), "buffer": set(), "user_input": set()}, []
    for spec in signature["input_specs"]:
        if not isinstance(spec, dict) or len(spec) != 1:
            _fail("input spec needs exactly one recognized variant")
        kind, value = next(iter(spec.items()))
        if kind == "user_input":
            argument = _user_tensor(_fields(value, {"arg"}, kind)["arg"])
            target = argument
        elif kind in {"parameter", "buffer"}:
            target_field = kind + "_name"
            _fields(value, {"arg", target_field, *({"persistent"} if kind == "buffer" else set())}, kind)
            argument, target = _tensor(value["arg"]), _name(value[target_field])
            if kind == "buffer" and type(value["persistent"]) is not bool:
                _fail("buffer persistent must be explicit Boolean")
            if target in targets:
                _fail("duplicate parameter or buffer target")
            targets.add(target)
        else:
            _fail("unrecognized input signature variant: " + str(kind))
        if argument in arguments:
            _fail("duplicate input argument")
        arguments.add(argument)
        inputs[kind].add(target)
        if kind == "buffer":
            buffers.append((argument, target))

    mutations = {"buffer": [], "parameter": [], "user_input": []}
    for spec in signature["output_specs"]:
        if not isinstance(spec, dict) or len(spec) != 1:
            _fail("output spec needs exactly one recognized variant")
        kind, value = next(iter(spec.items()))
        if kind == "user_output":
            _user_tensor(_fields(value, {"arg"}, kind)["arg"])
            continue
        mutation_kind = kind.removesuffix("_mutation")
        if kind != mutation_kind + "_mutation" or mutation_kind not in mutations:
            _fail("unrecognized output signature variant: " + str(kind))
        target_field = {"buffer": "buffer_name", "parameter": "parameter_name", "user_input": "user_input_name"}[
            mutation_kind
        ]
        _fields(value, {"arg", target_field}, kind)
        _tensor(value["arg"])
        target = _name(value[target_field])
        if target not in inputs[mutation_kind]:
            _fail("mutation target has no matching declared input")
        if target in mutations[mutation_kind]:
            _fail("duplicate mutation target")
        mutations[mutation_kind].append(target)
    return (
        tuple(SavedBufferSignature(argument, target, target in mutations["buffer"]) for argument, target in buffers),
        *(tuple(mutations[kind]) for kind in ("buffer", "parameter", "user_input")),
    )


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value):
    _fail("nonfinite JSON constant: " + value)


def _archive_signature(stream):
    with zipfile.ZipFile(stream) as archive:
        members = archive.infolist()
        if len(members) > _MAX_MEMBERS or len({info.filename for info in members}) != len(members):
            _fail("archive member inventory is oversized or duplicated")
        candidates = []
        for info in members:
            name = PurePosixPath(info.filename)
            if len(name.parts) >= 3 and name.parts[-2] == "models" and name.suffix == ".json":
                if (
                    name.is_absolute()
                    or name.as_posix() != info.filename
                    or any(part in {".", ".."} for part in name.parts)
                    or "\\" in info.filename
                    or "\x00" in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16)
                    or info.is_dir()
                ):
                    _fail("unsafe model JSON archive member")
                candidates.append(info)
        if len(candidates) != 1:
            _fail("archive requires exactly one models JSON member")
        member = candidates[0]
        if member.flag_bits & 1 or not 0 < member.file_size <= _MAX_JSON_BYTES:
            _fail("model JSON member is encrypted, empty or exceeds the byte budget")
        with archive.open(member) as payload:
            raw = payload.read(_MAX_JSON_BYTES + 1)
        if len(raw) != member.file_size or len(raw) > _MAX_JSON_BYTES:
            _fail("model JSON member size differs or exceeds the byte budget")
    document = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object, parse_constant=_invalid_constant)
    return member.filename, _signature(document)


def _digest(stream):
    stream.seek(0)
    digest = sha256()
    while block := stream.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def _identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def read_saved_program_signature(path, *, expected_sha256):
    """Read reviewed Torch serialization 8.17 with bounded JSON and exact custody.

    ``expected_sha256`` comes from the authenticated saved-program manifest.
    Missing or unknown signature variants refuse; readonly buffers follow only
    from explicit buffer inputs and the complete recognized mutation outputs.
    The archive is hashed before and after reading, without extracting weights.
    """
    if not is_sha256(expected_sha256):
        _fail("expected_sha256 must be an authenticated SHA-256")
    path = Path(path).absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        _fail("archive must be a real file without symlink parents")
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or _digest(stream) != expected_sha256:
                _fail("archive SHA-256 differs or archive is not a regular file")
            stream.seek(0)
            member, signature = _archive_signature(stream)
            if (
                _digest(stream) != expected_sha256
                or _identity(os.fstat(stream.fileno())) != _identity(before)
                or _identity(path.stat(follow_symlinks=False)) != _identity(before)
            ):
                _fail("archive changed while reading")
    except (
        OSError,
        zipfile.BadZipFile,
        RuntimeError,
        NotImplementedError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        _fail(str(error))
    return SavedProgramSignature(expected_sha256, member, _SCHEMA_VERSION, *signature)
