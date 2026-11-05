#!/usr/bin/env python3
"""Verify an immutable reference archive, or restore its exact original members.

No compiler, experiment, simulator or executable archive member is run. Restoration
is deliberately separate from configuring a new experiment on another machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


def sha256_stream(source) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open('rb') as source:
        return sha256_stream(source)


def load_members(manifest: Path, archive: Path) -> tuple[dict, dict[str, dict]]:
    record = json.loads(manifest.read_bytes())
    if record.get('schema') != 'merlin.reference_archive.v1':
        raise ValueError('unsupported reference archive schema')
    expected = record['archive']
    if archive.stat().st_size != expected['bytes'] or sha256_file(archive) != expected['sha256']:
        raise ValueError('archive bytes do not match the manifest')
    rows = {}
    for row in record['files']:
        name = row['member']
        member = PurePosixPath(name)
        if (not isinstance(name, str) or member.is_absolute() or '..' in member.parts
                or not member.parts or member.as_posix() != name or name in rows
                or row['path'] != name):
            raise ValueError('invalid or duplicate archive member identity')
        rows[name] = row
    return record, rows


def verify(archive: Path, rows: dict[str, dict]) -> None:
    observed = set()
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            if not member.isfile() or member.name in observed or member.name not in rows:
                raise ValueError('unexpected, duplicate or non-file archive member')
            expected = rows[member.name]
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError('unreadable archive member')
            digest = sha256_stream(stream)
            if (member.size != expected['bytes'] or digest != expected['sha256']
                    or member.mode != int(expected['mode'], 8)):
                raise ValueError(f'archive member differs: {member.name}')
            observed.add(member.name)
    if observed != set(rows):
        raise ValueError('archive is missing declared members')


def safe_destination(destination: Path, name: str) -> Path:
    target = destination / name
    for parent in (target, *target.parents):
        if parent.is_symlink():
            raise ValueError(f'refusing symlink destination: {parent}')
    return target


def restore(archive: Path, destination: Path, rows: dict[str, dict]) -> int:
    # Reuse matching original files without rewriting them. Check every existing
    # member before writing anything, so a mixed or edited tree fails closed.
    pending = set()
    for name, row in rows.items():
        target = safe_destination(destination, name)
        if target.exists():
            if (not target.is_file() or target.stat().st_size != row['bytes']
                    or sha256_file(target) != row['sha256']
                    or target.stat().st_mode & 0o777 != int(row['mode'], 8)):
                raise ValueError(f'refusing to overwrite different bytes or mode: {target}')
        else:
            pending.add(name)
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            if member.name not in pending:
                continue
            target = safe_destination(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open('xb') as output:
                output.write(source.read())
            os.chmod(target, member.mode)
            if sha256_file(target) != rows[member.name]['sha256']:
                raise ValueError(f'restored file verification failed: {target}')
    return len(pending)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--destination', type=Path,
                        help='restore original relative paths beneath this separate directory')
    args = parser.parse_args()
    _, rows = load_members(args.manifest, args.archive)
    verify(args.archive, rows)
    restored = restore(args.archive, args.destination.absolute(), rows) if args.destination else 0
    print(json.dumps({'verified_files': len(rows), 'restored_files': restored,
                      'execution_started': False}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
