"""Positive public-data reads inside the ordinary native author permission probe.

Expected bytes come from the verified bundle snapshot, never the friendly aliases
being tested. This gate grants no path and changes no historical author profile.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
from pathlib import Path

from . import public_contract_staging as P

SCHEMA = "merlin.phase1_public_contract_preflight.v1"
MARKER = "PUBLIC_CONTRACT_READS_OK "
CHILD = r'''
import hashlib, json, sys
from pathlib import Path
expected = json.loads(sys.argv[1])
root = Path(expected['workspace'])
for relative, digest in expected['expected_files'].items():
    try:
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    except OSError as error:
        raise SystemExit('public contract input unreadable: ' + relative) from error
    if actual != digest:
        raise SystemExit('public contract input differs from frozen snapshot: ' + relative)
print('PUBLIC_CONTRACT_READS_OK ' + json.dumps(expected, sort_keys=True))
'''


def expected_reads(workspace, bundle, *, repo):
    """Select every required common ABI/schema byte from its frozen grant."""
    profile = (bundle.get("author_profile") or {}).get("name")
    if profile not in P.PROFILES:
        return None
    if repo is None:
        raise ValueError("public contract preflight requires the native composition owner's repository")
    entries = [entry for entry in bundle.get("allowed", ()) if P.selected(bundle, entry)]
    required = {*(P.PREFIX + name for name in P.PUBLIC_FILES), P.SCHEMAS}
    if len(entries) != len(required) or {entry["path"] for entry in entries} != required:
        raise ValueError("public contract preflight requires every exact public ABI/schema grant")
    from merlin.targetgen.sandbox import bwrap as BW

    sources = [BW.resolve_grant(entry["path"], Path(repo)) for entry in entries]
    frozen_paths = BW.snapshot_input_paths(Path(workspace), bundle, sources, repo=Path(repo))
    files = {}
    for entry, frozen in zip(entries, frozen_paths, strict=True):
        is_schemas = entry["path"] == P.SCHEMAS
        alias = "schemas" if is_schemas else Path(entry["path"]).name
        if entry.get("mode") != "ro" or entry.get("as", alias) != alias:
            raise ValueError("public contract preflight requires its exact RO workspace aliases")
        frozen = Path(frozen)
        if frozen.resolve(strict=True) != frozen.absolute():
            raise ValueError("public contract preflight requires canonical snapshot bytes")
        if is_schemas:
            if not stat.S_ISDIR(frozen.lstat().st_mode):
                raise ValueError("public schemas snapshot must be a directory")
            members = sorted(frozen.iterdir())
            if not members:
                raise ValueError("public schemas snapshot must not be empty")
        else:
            members = [frozen]
        for member in members:
            if not stat.S_ISREG(member.lstat().st_mode) or (is_schemas and not member.name.endswith(".schema.json")):
                raise ValueError("unexpected public contract snapshot member")
            relative = "schemas/" + member.name if is_schemas else alias
            files[relative] = hashlib.sha256(member.read_bytes()).hexdigest()
    return dict(schema=SCHEMA, profile=profile, workspace=str(workspace),
                expected_files=dict(sorted(files.items())), files_count=len(files))


def command(expected):
    """An isolated standard-library interpreter cannot import candidate modules."""
    return shlex.join(("python3", "-I", "-B", "-c", CHILD, json.dumps(expected, sort_keys=True)))


def record_success(expected, stdout, rounds, rnd):
    """Require the native positive receipt, then retain it outside the workspace."""
    records = [line[len(MARKER):] for line in stdout.splitlines() if line.startswith(MARKER)]
    if len(records) != 1 or json.loads(records[0]) != expected:
        raise RuntimeError("native public contract preflight did not verify the frozen input inventory")
    path = Path(rounds) / f"round_{rnd:02d}.public_contract_preflight.json"
    receipt = dict(expected, native_readability_verified=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
