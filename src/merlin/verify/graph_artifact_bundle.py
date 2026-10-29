"""Stream-check pinned artifact transports as DATA, without extraction or execution.

``verify_bundle(archive, specification, metadata_directory=...)`` accepts the
portable transport-record schema. ``original_root_prefix`` rewrites logical
index names only; those names are never opened. Expected hashes authenticate
bytes only to the caller's supplied pins, not the sender or a proof claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import tarfile
import unicodedata
from pathlib import Path

_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
_MODES = {"original-full", "external-originals", "worked-selection", "transport-only"}
_LIMIT = 16 * 1024 * 1024


class BundleError(ValueError):
    """Malformed metadata, unsafe member, or a failed expected-byte association."""


def _require(condition, message):
    if not condition:
        raise BundleError(message)


def member_path(value):
    """Accept a normalized portable relative file name; never resolve it on disk."""
    _require(type(value) is str and bool(value) and not value.startswith("/"), "member path must be relative text")
    for part in value.split("/"):
        _require(part not in ("", ".", "..") and part[-1:] not in (" ", "."), "invalid member path component")
        _require(not any(ord(c) < 32 or ord(c) == 127 or c in '\\:<>"|?*' for c in part), "nonportable member path")
        _require(part.split(".")[0].casefold() not in _RESERVED, "reserved member path")
    _require(unicodedata.normalize("NFC", value) == value, "member path is not normalized")
    return value


def _count(value):
    _require(type(value) is int and value >= 0, "byte/count field must be a nonnegative integer")
    return value


def _pin(value):
    _require(type(value) is dict, "pin must be an object")
    digest = value.get("sha256")
    _require(
        type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
        "invalid SHA-256 pin",
    )
    return {"sha256": digest, "bytes": _count(value.get("bytes"))}


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def load_document(raw):
    """Decode metadata without accepting duplicate keys or nonfinite numbers."""
    try:
        return json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(BundleError("nonfinite JSON number")),
        )
    except (ValueError, UnicodeError) as error:
        raise BundleError(str(error)) from error


def _digest(stream):
    digest, size = hashlib.sha256(), 0
    while block := stream.read(1024 * 1024):
        digest.update(block)
        size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _bytes_pin(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _population(rows):
    result, folded = {}, set()
    for name, pin in rows:
        member_path(name)
        key = name.casefold()
        _require(key not in folded, "duplicate or portable-colliding member path")
        folded.add(key)
        result[name] = _pin(pin)
    for key in folded:
        parts = key.split("/")
        _require(
            not any("/".join(parts[:i]) in folded for i in range(1, len(parts))), "file/directory prefix collision"
        )
    return result


def _specification(value):
    _require(
        type(value) is dict and type(value.get("mode")) is str and value["mode"] in _MODES,
        "unknown or missing bundle mode",
    )
    result = dict(value)
    _pin(result.get("archive"))
    prefix = result.get("original_root_prefix", "")
    _require(type(prefix) is str, "logical root prefix must be text")
    if prefix:
        _require(
            prefix.startswith("/") and prefix.endswith("/"), "logical root prefix must be absolute with trailing slash"
        )
        if prefix != "/":
            member_path(prefix[1:-1])
    if "members" in result:
        _require(type(result["members"]) is dict, "members must be a path-to-pin object")
        result["members"] = _population(result["members"].items())
    if result["mode"] == "transport-only":
        _require("members" in result and "index" not in result, "transport-only requires members and no original index")
        return result
    index = result.get("index")
    _pin(index)
    _require(index.get("format") in ("files", "external-retention"), "unknown original index format")
    _count(index.get("indexed_files"))
    _require(("member" in index) != ("file" in index), "index must select exactly one member or metadata file")
    member_path(index.get("member", index.get("file")))
    if result["mode"] == "original-full":
        _require("member" in index, "full original index must be inside the transport")
    if result["mode"] == "worked-selection":
        _require("members" in result, "selection requires a complete transport member inventory")
        _count(result.get("selected_original_files"))
        selection = result.get("member_prefix", "files/")
        _require(type(selection) is str and selection.endswith("/"), "selection prefix must end in slash")
        member_path(selection[:-1])
    return result


def _original_entries(raw, specification):
    index = load_document(raw)
    if specification["index"]["format"] == "external-retention":
        _require(
            type(index) is dict and all(type(v) is dict for v in index.values()),
            "external retention groups must be objects",
        )
        rows = [row for group in index.values() for row in group.values()]
    else:
        rows = index.get("files") if type(index) is dict else index
        if type(rows) is dict:
            rows = list(rows.values())
        _require(type(rows) is list, "original files must be a list or object")
    _require(len(rows) == specification["index"]["indexed_files"], "original index population differs")
    prefix = specification.get("original_root_prefix", "")
    converted = []
    for row in rows:
        _require(type(row) is dict and type(row.get("path")) is str, "original file record requires a text path")
        name = row["path"]
        if prefix and name.startswith(prefix):
            name = name[len(prefix) :]
        converted.append((name, row))
    return _population(converted)


def _external_index_path(directory, reference):
    _require(directory is not None, "external original index requires explicit metadata_directory")
    root = Path(directory).resolve()
    path = root / reference["file"]
    _require(
        not path.is_symlink() and path.resolve().is_relative_to(root), "metadata file escaped its explicit directory"
    )
    _require(stat.S_ISREG(path.stat().st_mode), "metadata index must be a regular file")
    return path


def _external_index(directory, reference, limit):
    path = _external_index_path(directory, reference)
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    _require(len(raw) <= limit, "original index exceeds configured byte limit")
    return raw


def _protect_inputs(output, inputs):
    if output is None:
        return
    for path in inputs:
        _require(
            output.resolve() != path.resolve() and not (output.exists() and path.exists() and output.samefile(path)),
            "output must not overwrite an input",
        )


def verify_bundle(archive, specification, *, metadata_directory=None, max_index_bytes=_LIMIT):
    """Return integrity-only observations, or raise BundleError/OSError.

    A specification is one portable transport record, not an executable receipt.
    Only ``archive`` and an explicitly selected relative external index are read.
    The archive is streamed twice through one file descriptor: transport hash,
    then member hashes. Nothing is extracted, imported or executed.
    """
    spec = _specification(specification)
    _require(type(max_index_bytes) is int and max_index_bytes > 0, "index byte limit must be positive")
    path = Path(archive)
    _require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode), "archive must be a regular nonlink file")
    observed, names, index_raw = {}, set(), None
    reference = spec.get("index", {})
    with path.open("rb") as source:
        transport = _digest(source)
        _require(transport == _pin(spec["archive"]), "transport digest or size differs")
        source.seek(0)
        try:
            with tarfile.open(fileobj=source, mode="r|*") as stream:
                for member in stream:
                    name = member_path(member.name)
                    _require(
                        member.type in (tarfile.REGTYPE, tarfile.AREGTYPE)
                        and member.sparse is None
                        and not member.linkname,
                        "nonregular, linked or sparse archive member",
                    )
                    _require(name.casefold() not in names, "duplicate or portable-colliding archive member")
                    names.add(name.casefold())
                    with stream.extractfile(member) as payload:
                        if name == reference.get("member"):
                            _require(member.size <= max_index_bytes, "original index exceeds configured byte limit")
                            index_raw = payload.read(max_index_bytes + 1)
                            observed[name] = _bytes_pin(index_raw)
                        else:
                            observed[name] = _digest(payload)
                    _require(observed[name]["bytes"] == member.size, "archive member header size differs")
        except (tarfile.TarError, EOFError) as error:
            raise BundleError("invalid or truncated archive") from error
    observed = _population(observed.items())
    if "members" in spec:
        _require(observed == spec["members"], "transport member population or payload differs")
    original_count = 0
    if reference:
        if "file" in reference:
            index_raw = _external_index(metadata_directory, reference, max_index_bytes)
        _require(
            index_raw is not None and _bytes_pin(index_raw) == _pin(reference),
            "original index identity differs or is missing",
        )
        expected = _original_entries(index_raw, spec)
        if spec["mode"] == "original-full":
            if reference["member"] in expected:
                _require(expected[reference["member"]] == _pin(reference), "index conflicts with its own pin")
            expected[reference["member"]] = _pin(reference)
            _require(observed == expected, "original member population or pins differ")
            original_count = reference["indexed_files"]
        elif spec["mode"] == "external-originals":
            _require(observed == expected, "external original member population or pins differ")
            original_count = len(expected)
        else:
            prefix = spec.get("member_prefix", "files/")
            for name, actual in observed.items():
                if name.startswith(prefix):
                    required = _pin(reference) if name == reference.get("member") else expected.get(name[len(prefix) :])
                    _require(actual == required, "selected member original pin differs")
                    original_count += 1
            _require(original_count == spec["selected_original_files"], "selected original population differs")
    return {
        "schema": "merlin.graph-artifact-integrity.v1",
        "status": "integrity_matched",
        "integrity_only": True,
        "archive": transport,
        "archive_members": len(observed),
        "original_member_pins_checked": original_count,
        "proof_checked": False,
        "qualification_verified": False,
        "human_review_approved": False,
        "native_execution_verified": False,
        "sender_authenticated": False,
        "historical_replay_performed": False,
    }


def main(argv=None):
    """JSON CLI; metadata may be a single record or a transfer with --record ID."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--record", help="Select one id from a transfer document's transports list")
    parser.add_argument("--metadata-directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output_permitted = False
    try:
        inputs = [args.archive, args.specification]
        _protect_inputs(args.output, inputs)
        document = load_document(args.specification.read_bytes())
        if args.record is not None:
            _require(
                type(document) is dict and type(document.get("transports")) is list,
                "transfer must contain a transports list",
            )
            matches = [r for r in document["transports"] if type(r) is dict and r.get("id") == args.record]
            _require(len(matches) == 1, "record must identify exactly one transport")
            document = matches[0]
        document = _specification(document)
        metadata_directory = args.metadata_directory or args.specification.parent
        if "file" in document.get("index", {}):
            inputs.append(_external_index_path(metadata_directory, document["index"]))
        _protect_inputs(args.output, inputs)
        output_permitted = True
        report = verify_bundle(args.archive, document, metadata_directory=metadata_directory)
    except (BundleError, OSError) as error:
        report = {
            "schema": "merlin.graph-artifact-integrity.v1",
            "status": "unavailable" if isinstance(error, OSError) else "integrity_failed",
            "integrity_only": True,
            "error": str(error),
            "proof_checked": False,
            "qualification_verified": False,
            "human_review_approved": False,
            "native_execution_verified": False,
            "sender_authenticated": False,
            "historical_replay_performed": False,
        }
    output = json.dumps(report, indent=2) + "\n"
    if args.output and output_permitted:
        args.output.write_text(output)
    print(output, end="")
    return 0 if report["status"] == "integrity_matched" else 1


if __name__ == "__main__":
    raise SystemExit(main())
