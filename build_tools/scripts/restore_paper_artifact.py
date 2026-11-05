#!/usr/bin/env python3
"""Verify and restore paper evidence without executing an archived program."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"unsafe archive path: {name!r}")
    return path


@contextmanager
def opened(path, archive_format):
    if archive_format == "tar.gz":
        with tarfile.open(path, "r|gz") as stream:
            yield stream
        return
    if archive_format != "tar.zst":
        raise ValueError("unsupported archive format")
    with subprocess.Popen(["zstd", "-dc", "--", str(path)], stdout=subprocess.PIPE) as process:
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
                yield stream
            process.stdout.close()
            if process.wait() != 0:
                raise ValueError("zstd could not decode the verified archive")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()


def archive_rows(manifest):
    record = json.loads(Path(manifest).read_bytes())
    if record.get("schema") != "merlin.paper_artifact_archives.v1":
        raise ValueError("unsupported archive manifest")
    rows = {}
    filenames = set()
    for row in record["archives"]:
        filename = relative(row["filename"])
        if len(filename.parts) != 1 or row["id"] in rows or str(filename) in filenames:
            raise ValueError("archive IDs and filenames must be unique")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError("invalid archive size")
        if len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
            raise ValueError("invalid archive SHA256")
        rows[row["id"]] = row
        filenames.add(str(filename))
    return rows


def inspect_archive(path, row, destination=None):
    """Check full membership, safe paths, hardlink custody and expanded length.

    Hardlinks are permitted only to other ordinary members of this same archive.
    Symlinks, devices, FIFOs, traversal and privileged modes are refused.
    """
    if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
        raise ValueError(f"archive checksum or size differs: {path}")
    if row["format"] == "git-bundle":
        return {"archive": row["id"], "sha256": row["sha256"], "restored": False}
    names, files, links, directories = {}, set(), {}, []
    expanded = 0
    with opened(path, row["format"]) as stream:
        for member in stream:
            name = str(relative(member.name.rstrip("/") if member.isdir() else member.name))
            if name in names or name.split("/")[0] not in row["top_level"]:
                raise ValueError("duplicate or undeclared archive member")
            if member.mode & ~0o777 or not (member.isfile() or member.isdir() or member.islnk()):
                raise ValueError("unsupported archive member type or mode")
            names[name] = member
            expanded += member.size
            if len(names) > row["members"] or expanded > row["expanded_bytes"]:
                raise ValueError("archive exceeds its declared expansion")
            target = destination / name if destination else None
            if member.isdir():
                directories.append((name, member.mode))
                if target:
                    target.mkdir(parents=True, exist_ok=True)
            elif member.islnk():
                links[name] = str(relative(member.linkname))
            else:
                files.add(name)
                source = stream.extractfile(member)
                if source is None:
                    raise ValueError("unreadable archive file")
                with source:
                    if target:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open("xb") as output:
                            shutil.copyfileobj(source, output)
                        target.chmod(member.mode)
                    else:
                        # Consume each file to detect truncated compressed streams.
                        for _ in iter(lambda: source.read(1024 * 1024), b""):
                            pass
    if len(names) != row["members"] or expanded != row["expanded_bytes"]:
        raise ValueError("archive membership or expanded size differs")
    for name in names:
        if any(parent.as_posix() in files or parent.as_posix() in links
               for parent in PurePosixPath(name).parents):
            raise ValueError("archive file is used as a parent directory")
    for name, link in links.items():
        seen = {name}
        while link in links:
            if link in seen:
                raise ValueError("cyclic archive hardlink")
            seen.add(link)
            link = links[link]
        if link not in files or any(parent.as_posix() in files or parent.as_posix() in links
                                    for parent in PurePosixPath(name).parents):
            raise ValueError("archive hardlink does not name an ordinary member")
        if destination:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(destination / link, target)
    if destination:
        for name, mode in sorted(directories, key=lambda item: len(PurePosixPath(item[0]).parts), reverse=True):
            (destination / name).chmod(mode)
    return {"archive": row["id"], "sha256": row["sha256"], "members": len(names),
            "expanded_bytes": expanded, "restored": destination is not None}


def restore(archive_dir, rows, destination):
    """Restore into a new tree, publishing it only after every archive succeeds."""
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination must not exist")
    for parent in destination.parents:
        if parent.is_symlink():
            raise ValueError("destination has an indirect parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".artifact-restore-", dir=destination.parent))
    try:
        results = []
        for row in rows:
            if row["format"] == "git-bundle":
                results.append(inspect_archive(archive_dir / row["filename"], row))
                continue
            suffix = row.get("destination", ".")
            target = staging if suffix == "." else staging / relative(suffix)
            target.mkdir(parents=True, exist_ok=True)
            results.append(inspect_archive(archive_dir / row["filename"], row, target))
        (staging / "artifact-restoration.json").write_text(json.dumps({
            "schema": "merlin.paper_artifact_restoration.v1", "archives": results,
            "programs_executed": False}, indent=2) + "\n")
        if destination.exists():
            raise ValueError("destination appeared during restoration")
        staging.rename(destination)
        return results
    except BaseException:
        # Only the fresh private staging tree belongs to this operation.
        for directory, subdirs, _ in os.walk(staging):
            Path(directory).chmod(0o700)
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--archive", action="append", help="select archive ID, repeat as needed")
    parser.add_argument("--destination", type=Path, help="new data directory, omitted for verification only")
    args = parser.parse_args()
    catalog = archive_rows(args.manifest)
    rows = [catalog[name] for name in (args.archive or catalog)]
    results = restore(args.archive_dir, rows, args.destination) if args.destination else [
        inspect_archive(args.archive_dir / row["filename"], row) for row in rows]
    print(json.dumps({"verified_archives": results, "programs_executed": False}, indent=2))


if __name__ == "__main__":
    main()
