#!/usr/bin/env python3
"""Build an explicitly selected bubblewrap release with a larger argument bound.

This changes only bubblewrap 0.9.0's unique static MAX_ARGS declaration. It
neither installs nor selects the product, and establishes no sandbox policy or
transitive toolchain qualification. The operator supplies the archive pin,
build tools and optional already prepared dependency files.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

_ROOT = "bubblewrap-0.9.0"
_DECLARATION = b"static const int32_t MAX_ARGS = 9000;"
_MAX_MEMBERS = 4096
_MAX_SOURCE_BYTES = 64 << 20


def pin(path):
    selected = Path(path).absolute()
    if selected.resolve() != selected or selected.is_symlink() or not selected.is_file():
        raise ValueError("build input must be a canonical ordinary file")
    before = selected.stat()
    with selected.open("rb") as stream:
        identity = hashlib.file_digest(stream, "sha256").hexdigest()
    after = selected.stat()
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in keys):
        raise ValueError("build input changed while hashing")
    return {"path": str(selected), "sha256": identity, "bytes": after.st_size}


def pinned_archive(path, expected):
    if len(expected) != 64 or any(value not in "0123456789abcdef" for value in expected):
        raise ValueError("archive pin must be a lowercase SHA256")
    actual = pin(path)
    if actual["sha256"] != expected:
        raise ValueError("selected archive SHA256 mismatch")
    return actual


def inventory(root):
    rows = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            if not path.resolve().is_relative_to(root) or not path.resolve().is_file():
                raise ValueError("prepared source contains an escaping or unavailable alias")
            rows.append({"path": str(path.relative_to(root)), "kind": "symlink", "target": os.readlink(path)})
            continue
        if path.is_dir():
            continue
        row = pin(path)
        rows.append({**row, "path": str(path.relative_to(root)), "mode": path.stat().st_mode & 0o777})
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "members": rows}


def extract_source(archive, destination):
    """Refuse indirect, ambiguous or out-of-root members before writing any file."""
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        if len(members) > _MAX_MEMBERS or sum(member.size for member in members) > _MAX_SOURCE_BYTES:
            raise ValueError("source archive exceeds its bounded extraction budget")
        seen = set()
        by_name = {str(Path(member.name)): member for member in members}
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != _ROOT
                or str(path) in seen or not (member.isfile() or member.isdir() or member.issym())
            ):
                raise ValueError("source archive member is unsafe, indirect or duplicated")
            if member.issym():
                target = Path(member.linkname)
                referent = by_name.get(str(path.parent / target))
                if target.is_absolute() or ".." in target.parts or referent is None or not referent.isfile():
                    raise ValueError("source archive alias is not an in-root ordinary member")
            seen.add(str(path))
        stream.extractall(destination, members=members, filter="data")
    return Path(destination) / _ROOT


def patch_arguments(source, maximum, patch_path):
    if type(maximum) is not int or not 9001 <= maximum <= 131072:
        raise ValueError("argument bound must be an explicit integer in 9001..131072")
    path = Path(source) / "bubblewrap.c"
    before = path.read_bytes()
    declarations = [line.strip() for line in before.splitlines()
                    if line.strip().startswith(b"static const int32_t MAX_ARGS")]
    if declarations != [_DECLARATION] or before.count(_DECLARATION) != 1:
        raise ValueError("selected source lacks its unique original MAX_ARGS declaration")
    if b"version : '0.9.0'," not in (Path(source) / "meson.build").read_bytes():
        raise ValueError("selected source is not the supported bubblewrap release")
    after = before.replace(_DECLARATION, f"static const int32_t MAX_ARGS = {maximum};".encode(), 1)
    patch = "".join(difflib.unified_diff(
        before.decode().splitlines(keepends=True), after.decode().splitlines(keepends=True),
        fromfile="a/bubblewrap.c", tofile="b/bubblewrap.c",
    ))
    Path(patch_path).write_text(patch, encoding="utf-8")
    path.write_bytes(after)


def _run(argv, *, environment, cwd, log):
    with Path(log).open("wb") as output:
        completed = subprocess.run(argv, cwd=cwd, env=environment, stdout=output, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"selected build command failed ({completed.returncode}); inspect {log}")
    return {"argv": list(map(str, argv)), "log": pin(log)}


def build(args):
    archive = pinned_archive(args.archive, args.archive_sha256)
    dependency_archives = [pinned_archive(path, sha) for path, sha in args.dependency_archive]
    tools = {name: pin(getattr(args, name)) for name in ("meson", "ninja", "cc", "pkg_config", "build_python")}
    for tool in tools.values():
        if not os.access(tool["path"], os.X_OK) or any(value in tool["path"] for value in "'\\\r\n"):
            raise ValueError("selected build tool must be executable with a plain Meson path")
    dependency_files = [pin(path) for path in args.dependency_file]
    recipe = pin(Path(__file__).resolve())
    output = args.output.absolute()
    if output.resolve() != output or any(parent.is_symlink() for parent in output.parents):
        raise ValueError("build output must have a canonical independent owner")
    output.mkdir(parents=True, exist_ok=False)
    (output / "tmp").mkdir()
    source = extract_source(args.archive, output)
    original_source = inventory(source)
    patch_path = output / "max-arguments.patch"
    patch_arguments(source, args.max_arguments, patch_path)
    patched_source = inventory(source)
    original_members = {row["path"]: row for row in original_source["members"]}
    patched_members = {row["path"]: row for row in patched_source["members"]}
    if original_members.keys() != patched_members.keys() or [
        name for name in original_members if original_members[name] != patched_members[name]
    ] != ["bubblewrap.c"]:
        raise ValueError("source preparation changed more than the sole declared bound")
    native = output / "native.ini"
    native.write_text("[binaries]\n" + "\n".join(
        f"{key} = '{tools[name]['path']}'" for key, name in (("c", "cc"), ("pkg-config", "pkg_config"))
    ) + "\n", encoding="utf-8")
    native_pin = pin(native)
    environment = {**os.environ, "TMPDIR": str(output / "tmp"), "CC": tools["cc"]["path"],
                   "PKG_CONFIG": tools["pkg_config"]["path"], "NINJA": tools["ninja"]["path"]}
    if args.pkg_config_path:
        environment["PKG_CONFIG_PATH"] = os.pathsep.join(str(path.absolute()) for path in args.pkg_config_path)
    directory = output / "build"
    commands = [
        _run([tools["meson"]["path"], "setup", str(directory), str(source), "--native-file", str(native),
              "--buildtype=release", "-Dtests=false", "-Dman=disabled", "-Dbash_completion=disabled",
              "-Dzsh_completion=disabled", "-Dpython=" + tools["build_python"]["path"]],
             environment=environment, cwd=output, log=output / "meson.log"),
        _run([tools["ninja"]["path"], "-C", str(directory), "-j", str(args.jobs), "bwrap"],
             environment=environment, cwd=output, log=output / "ninja.log"),
    ]
    if inventory(source) != patched_source:
        raise ValueError("selected build changed prepared source bytes")
    binary = pin(directory / "bwrap")
    for original in (archive, recipe, native_pin, *dependency_archives, *dependency_files, *tools.values()):
        if pin(original["path"]) != original:
            raise ValueError("selected input or tool changed during build")
    record = {
        "schema": "merlin.bubblewrap_argument_build.v1", "version": "0.9.0",
        "maximum_arguments": args.max_arguments, "archive": archive,
        "original_source": original_source, "patched_source": patched_source, "patch": pin(patch_path),
        "recipe": recipe, "native_selection": native_pin, "tools": tools,
        "dependency_archives": dependency_archives, "dependency_files": dependency_files,
        "commands": commands, "binary": binary,
        "build_environment_sha256": hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
        "build_configuration": pin(directory / "meson-info" / "intro-buildoptions.json"),
        "scope": (
            "explicit local source preparation and build; complete dependency closure and sandbox policy unqualified"
        ),
        "selection": "operator_must_select_and_qualify_explicitly", "system_installation": "not_performed",
    }
    (output / "build-record.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--max-arguments", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("meson", "ninja", "cc", "pkg-config", "build-python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pkg-config-path", type=Path, action="append", default=[])
    parser.add_argument("--dependency-archive", nargs=2, metavar=("PATH", "SHA256"), action="append", default=[])
    parser.add_argument("--dependency-file", type=Path, action="append", default=[])
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 4 or not 9001 <= args.max_arguments <= 131072:
        parser.error("jobs must be 1..4 and max-arguments must be 9001..131072")
    record = build(args)
    print(json.dumps({"binary": record["binary"], "record": str(args.output.absolute() / "build-record.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
