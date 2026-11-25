#!/usr/bin/env python3
"""Index explicitly selected small metadata files without importing their verdicts.

Only the named files are read. Nested paths, tensor payloads and directory trees
are never followed or copied. Verification establishes file identity, not the
correctness, provenance or reproducibility of the indexed experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_bytes
from merlin.common.strict_json import loads

SCHEMA = "merlin.model_study.condition_index.v1"
SCOPE = "explicit_metadata_file_identity_only"
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_ENTRIES = 64
ROLES = frozenset({"result", "manifest", "config"})


def name(value):
    if (
        type(value) is not str
        or not 1 <= len(value) <= 80
        or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in value)
    ):
        raise ValueError("select a 1..80 character name using letters, digits, _, - or .")
    return value


def read_selected(path, expected, *, limit=MAX_FILE_BYTES):
    """Read a bounded canonical ordinary file and require the caller's digest."""
    if not is_sha256(expected):
        raise ValueError("a lowercase SHA-256 is required")
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("select canonical ordinary metadata files, without symlinks")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("selected metadata must be a bounded ordinary file")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    stamps = [tuple(getattr(s, key) for key in fields) for s in (before, opened, after, path.stat())]
    if len(raw) > limit or any(s != stamps[0] for s in stamps[1:]):
        raise ValueError("selected metadata changed while reading")
    if sha256_bytes(raw) != expected:
        raise ValueError("selected metadata digest differs")
    return raw, {"path": str(path), "sha256": expected, "bytes": len(raw)}


def entry(role, label, path, expected):
    if role not in ROLES:
        raise ValueError("unknown metadata role")
    suffixes = {".json", ".yaml", ".yml", ".toml"} if role == "config" else {".json"}
    if Path(path).suffix.lower() not in suffixes:
        raise ValueError("select JSON result/manifest or JSON/YAML/TOML config metadata")
    raw, selected = read_selected(path, expected)
    if Path(path).suffix.lower() == ".json" and type(loads(raw, max_bytes=MAX_FILE_BYTES)) is not dict:
        raise ValueError("JSON metadata must be an object")
    return {"name": name(label), "role": role, "pin": selected}


def make_index(label, selections):
    if not 1 <= len(selections) <= MAX_ENTRIES:
        raise ValueError("select 1..64 explicit metadata files")
    rows, seen, total = [], set(), 0
    for role, label_, path, expected in selections:
        if label_ in seen:
            raise ValueError("metadata names must be unique")
        seen.add(label_)
        row = entry(role, label_, path, expected)
        total += row["pin"]["bytes"]
        if total > MAX_TOTAL_BYTES:
            raise ValueError("selected metadata exceeds the total byte limit")
        rows.append(row)
    return {"schema": SCHEMA, "scope": SCOPE, "label": name(label), "entries": rows}


def verify_index(path, expected):
    raw, _ = read_selected(path, expected)
    document = loads(raw, max_bytes=MAX_FILE_BYTES)
    if (
        type(document) is not dict
        or set(document) != {"schema", "scope", "label", "entries"}
        or document["schema"] != SCHEMA
        or document["scope"] != SCOPE
        or type(document["entries"]) is not list
    ):
        raise ValueError("not an identity-only condition index")
    selections = []
    for row in document["entries"]:
        if type(row) is not dict or set(row) != {"name", "role", "pin"}:
            raise ValueError("invalid indexed metadata entry")
        selected = row["pin"]
        if (
            type(selected) is not dict
            or set(selected) != {"path", "sha256", "bytes"}
            or type(selected["path"]) is not str
            or type(selected["bytes"]) is not int
            or not 0 <= selected["bytes"] <= MAX_FILE_BYTES
        ):
            raise ValueError("invalid indexed file pin")
        selections.append((row["role"], row["name"], selected["path"], selected["sha256"]))
    if make_index(document["label"], selections) != document:
        raise ValueError("indexed file identity changed")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    create = commands.add_parser("create", help="pin only explicitly selected metadata")
    create.add_argument("--label", required=True)
    for role in sorted(ROLES):
        create.add_argument("--" + role, nargs=3, action="append", default=[], metavar=("NAME", "PATH", "SHA256"))
    create.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify a caller-pinned index and its named files")
    verify.add_argument("--index", type=Path, required=True)
    verify.add_argument("--index-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "create":
            selections = [(role, *values) for role in sorted(ROLES) for values in getattr(args, role)]
            document = make_index(args.label, selections)
            raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
            os.umask(0o077)
            with args.output.open("xb") as stream:
                stream.write(raw)
            result = {"status": "recorded", "index_sha256": sha256_bytes(raw)}
        else:
            document = verify_index(args.index, args.index_sha256)
            result = {"status": "identities_verified", "index_sha256": args.index_sha256}
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"conditions: {error}\n")
    print(json.dumps({**result, "entries": len(document["entries"]), "scope": SCOPE}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
