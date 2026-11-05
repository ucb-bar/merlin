#!/usr/bin/env python3
"""Restore pinned dependency bytes without running any archived executable.

Original absolute symlinks retain their recorded targets. Use the artifact's
explicit namespace adapter to mount restored roots at those original locations.
Restoration alone does not relocate a virtual environment or qualify execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from restore_paper_artifact import digest, opened, relative


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid SHA256 declaration")
    return value


def _leaf(name):
    path = relative(name)
    if len(path.parts) != 1:
        raise ValueError("dependency ID and archive filenames must be single components")
    return str(path)


def _file(path, sha256, size=None):
    if (
        path.is_symlink()
        or not path.is_file()
        or size is not None
        and path.stat().st_size != size
        or digest(path) != _hash(sha256)
    ):
        raise ValueError(f"pinned file differs: {path}")


def _member(name, *, directory=False):
    if name.startswith("./"):
        name = name[2:]
    return str(relative(name.rstrip("/") if directory else name))


def selections(capture, capture_sha256, archive_dir, metadata_dir):
    _file(capture, capture_sha256)
    record = _json(capture)
    if record.get("schema") != "merlin.paper_dependency_capture.v1":
        raise ValueError("unsupported dependency capture schema")
    roots, ids, archives, inventories, selected = [], set(), set(), set(), []
    for row in record["archives"]:
        identity, filename, inventory = (_leaf(row[key]) for key in ("id", "archive", "inventory"))
        source = row["source"]
        if (
            not isinstance(source, str)
            or not source.startswith("/")
            or source == "/"
            or posixpath.normpath(source) != source
            or source in roots
            or identity in ids
            or filename in archives
            or inventory in inventories
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
        ):
            raise ValueError("invalid or duplicate dependency selection")
        roots.append(source)
        ids.add(identity)
        archives.add(filename)
        inventories.add(inventory)
        path = metadata_dir / inventory
        _file(path, row["inventory_sha256"])
        document = _json(path)
        members = document.get("members")
        if document.get("schema") != "merlin.paper_dependency_inventory.v1" or not isinstance(members, list):
            raise ValueError("unsupported dependency inventory")
        tree = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if tree != row["tree_sha256"] or tree != document.get("tree_sha256") or len(members) != row["members"]:
            raise ValueError("dependency inventory tree differs")
        by_name = {}
        for member in members:
            name = str(relative(member["path"]))
            mode = int(member["mode"], 8)
            kind = member["kind"]
            if (
                name == "."
                or name in by_name
                or mode < 0
                or mode & ~0o777
                or kind not in ("file", "directory", "symlink")
            ):
                raise ValueError("invalid dependency member")
            if kind == "file":
                _hash(member["sha256"])
                if type(member["bytes"]) is not int or member["bytes"] < 0:
                    raise ValueError("invalid dependency file length")
            elif kind == "symlink" and (not isinstance(member.get("target"), str) or not member["target"]):
                raise ValueError("invalid dependency symlink target")
            by_name[name] = member
        for name in by_name:
            for parent in PurePosixPath(name).parents:
                if str(parent) != "." and by_name.get(str(parent), {}).get("kind") != "directory":
                    raise ValueError("dependency parent is absent or not a directory")
        selected.append((row, by_name))
    if not selected:
        raise ValueError("dependency capture contains no archives")
    for row, members in selected:
        for name, member in members.items():
            if member["kind"] != "symlink":
                continue
            target = member["target"]
            absolute = posixpath.normpath(
                target if target.startswith("/") else posixpath.join(row["source"], posixpath.dirname(name), target)
            )
            if sum(absolute == root or absolute.startswith(root + "/") for root in roots) != 1:
                raise ValueError("dependency symlink leaves all declared source roots")
    # Authenticate every archive before creating any output tree.
    for row, _ in selected:
        _file(archive_dir / row["archive"], row["sha256"], row["bytes"])
    _file(capture, capture_sha256)
    return selected


def extract(path, row, members, destination):
    observed, files, links, directories, symlinks = set(), {}, {}, [], []
    root_seen = False
    with opened(path, "tar.zst") as stream:
        for item in stream:
            if item.name in (".", "./"):
                if root_seen or not item.isdir() or item.mode & ~0o777:
                    raise ValueError("duplicate or invalid archive root")
                root_seen = True
                continue
            name = _member(item.name, directory=item.isdir())
            expected = members.get(name)
            if name in observed or expected is None or item.mode != int(expected["mode"], 8):
                raise ValueError("undeclared, duplicate or changed archive member")
            observed.add(name)
            target = destination / name
            if expected["kind"] == "directory" and item.isdir():
                target.mkdir(parents=True, exist_ok=True)
                directories.append((target, item.mode))
            elif expected["kind"] == "symlink" and item.issym() and item.linkname == expected["target"]:
                symlinks.append((target, item.linkname))
            elif expected["kind"] == "file" and item.islnk():
                links[name] = _member(item.linkname)
            elif expected["kind"] == "file" and item.isfile():
                if item.size != expected["bytes"]:
                    raise ValueError("archive file length differs")
                target.parent.mkdir(parents=True, exist_ok=True)
                actual = hashlib.sha256()
                source = stream.extractfile(item)
                if source is None:
                    raise ValueError("unreadable archive member")
                with source, target.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        actual.update(block)
                if actual.hexdigest() != expected["sha256"]:
                    raise ValueError("archive file bytes differ")
                target.chmod(item.mode)
                files[name] = expected
            else:
                raise ValueError("archive type or symlink target differs")
    if observed != set(members):
        raise ValueError("archive membership is incomplete")
    for name, link in links.items():
        seen = {name}
        while link in links:
            if link in seen:
                raise ValueError("cyclic archive hardlink")
            seen.add(link)
            link = links[link]
        expected = members[name]
        source = files.get(link)
        if source is None or any(expected[key] != source[key] for key in ("sha256", "bytes", "mode")):
            raise ValueError("hardlink does not name matching ordinary archive bytes")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(destination / link, target)
    _file(path, row["sha256"], row["bytes"])
    return symlinks, directories


def _remove_private(path):
    if path is None:
        return
    for directory, _, _ in os.walk(path, followlinks=False):
        Path(directory).chmod(0o700)
    shutil.rmtree(path)


def restore(capture, capture_sha256, archive_dir, metadata_dir, destination):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination must not exist")
    if any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("destination has an indirect parent")
    selected = selections(Path(capture), capture_sha256, Path(archive_dir), Path(metadata_dir))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dependency-restore-", dir=destination.parent))
    owned_destination = False
    try:
        symlinks, directories, results = [], [], []
        for row, members in selected:
            target = staging / "dependencies" / row["id"]
            target.mkdir(parents=True)
            links, modes = extract(Path(archive_dir) / row["archive"], row, members, target)
            symlinks.extend(links)
            directories.extend(modes)
            results.append(
                dict(
                    id=row["id"],
                    source=row["source"],
                    tree_sha256=row["tree_sha256"],
                    members=len(members),
                    archive_sha256=row["sha256"],
                )
            )
        # No symlink is present during extraction of any dependency archive.
        for path, target in symlinks:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
        for path, mode in sorted(directories, key=lambda item: len(item[0].parts), reverse=True):
            path.chmod(mode)
        _file(Path(capture), capture_sha256)
        for row, _ in selected:
            _file(Path(metadata_dir) / row["inventory"], row["inventory_sha256"])
        destination.mkdir(mode=0o700)  # Exclusive creation prevents overwriting a late destination.
        owned_destination = True
        (staging / "dependencies").rename(destination / "dependencies")
        (destination / "dependency-restoration.json").write_text(
            json.dumps(
                dict(
                    schema="merlin.paper_dependency_restoration.v1",
                    capture_sha256=capture_sha256,
                    dependencies=results,
                    programs_executed=False,
                    relocation="Original symlink targets retained. Use only the explicit namespace adapter.",
                ),
                indent=2,
            )
            + "\n"
        )
        return results
    except BaseException:
        if owned_destination:
            _remove_private(destination)
        raise
    finally:
        _remove_private(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    results = restore(args.capture, args.capture_sha256, args.archive_dir, args.metadata_dir, args.destination)
    print(json.dumps(dict(restored_dependencies=results, programs_executed=False), indent=2))


if __name__ == "__main__":
    main()
