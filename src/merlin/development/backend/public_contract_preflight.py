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

from merlin.development.backend import public_contract_staging as P

SCHEMA = "merlin.phase1_public_contract_preflight.v1"
MARKER = "PUBLIC_CONTRACT_READS_OK "
CHILD = r"""
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
"""


def expected_reads(workspace, bundle, *, repo):
    """Select every required common ABI/schema byte from its frozen grant."""
    profile = (bundle.get("author_profile") or {}).get("name")
    if profile not in P.PROFILES:
        return None
    if repo is None:
        raise ValueError("public contract preflight requires the native composition owner's repository")
    runtime_abi = None
    from merlin.development.backend.author_profiles import RAW_RTL_NAMES

    if profile in RAW_RTL_NAMES:
        from merlin.development.backend import public_runtime_abi as R

        runtime_abi = R.verify(bundle, workspace=workspace, repo=repo)
    if profile == "xs-el-raw-rtl-v2/el1":
        return _neutral_reads(workspace, bundle, repo=repo)
    protocol_path = None if runtime_abi is None else runtime_abi["protocol"]["path"]
    entries = [
        entry for entry in bundle.get("allowed", ()) if P.selected(bundle, entry) or entry.get("path") == protocol_path
    ]
    required = {*(P.PREFIX + name for name in P.PUBLIC_FILES), P.SCHEMAS}
    if runtime_abi is not None:
        required.remove(P.PREFIX + R.ALIAS)
        required.add(protocol_path)
    if len(entries) != len(required) or {entry["path"] for entry in entries} != required:
        raise ValueError("public contract preflight requires every exact public ABI/schema grant")
    from merlin.development.targetgen.sandbox import bwrap as BW

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
    # New raw-RTL conditions must positively read the exact hardware bytes,
    # not just pass a negative check for the withheld extracted summaries.
    _raw_reads(files, workspace, bundle, repo=repo)
    return dict(
        schema=SCHEMA,
        profile=profile,
        workspace=str(workspace),
        expected_files=dict(sorted(files.items())),
        files_count=len(files),
    )


def _raw_reads(files, workspace, bundle, *, repo):
    from merlin.development.targetgen.sandbox import bwrap as BW

    raw_pins = bundle.get("selected_raw_rtl_files", ())
    if raw_pins:
        raw_entries = {entry["path"]: entry for entry in bundle.get("allowed", ())}
        sources = [Path(pin["path"]) for pin in raw_pins]
        frozen = BW.snapshot_input_paths(Path(workspace), bundle, sources, repo=Path(repo))
        for pin, member in zip(raw_pins, frozen, strict=True):
            entry = raw_entries.get(pin["path"], {})
            relative = entry.get("as")
            if (
                entry.get("mode") != "ro"
                or not isinstance(relative, str)
                or Path(relative).parts[:1] != ("raw-rtl",)
                or ".." in Path(relative).parts
                or not stat.S_ISREG(Path(member).lstat().st_mode)
            ):
                raise ValueError("raw RTL preflight requires its exact read-only file alias")
            digest = hashlib.sha256(Path(member).read_bytes()).hexdigest()
            if digest != pin["sha256"] or relative in files:
                raise ValueError("raw RTL preflight differs from selected hardware pin")
            files[relative] = digest


def _neutral_reads(workspace, bundle, *, repo):
    from merlin.development.backend import neutral_public_contracts as N
    from merlin.development.targetgen.sandbox import bwrap as BW

    pins = bundle.get("neutral_public_contract_files")
    required = N.FILES | {N.INPUT_DOCUMENT} | {"schemas/" + name for name in N.SCHEMAS}
    if not isinstance(pins, list) or len(pins) != len(required) or {pin.get("alias") for pin in pins} != required:
        raise ValueError("neutral EL1 requires its complete external protocol pin inventory")
    if any(str(row["path"]).startswith(P.PREFIX) for row in bundle.get("allowed", ())):
        raise ValueError("neutral EL1 still exposes the original dialect contract directory")
    sources = [Path(pin["path"]) for pin in pins]
    frozen = BW.snapshot_input_paths(Path(workspace), bundle, sources, repo=Path(repo))
    files = {}
    for pin, member in zip(pins, frozen, strict=True):
        if not stat.S_ISREG(Path(member).lstat().st_mode):
            raise ValueError("neutral external contract must be a regular frozen file")
        raw = Path(member).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != pin["sha256"] or b"merlin_iface" in raw:
            raise ValueError("neutral external protocol is changed or exposes the hidden input dialect")
        files[pin["alias"]] = digest
    _raw_reads(files, workspace, bundle, repo=repo)
    return dict(
        schema=SCHEMA,
        profile=(bundle["author_profile"])["name"],
        workspace=str(workspace),
        expected_files=dict(sorted(files.items())),
        files_count=len(files),
    )


def command(expected):
    """An isolated standard-library interpreter cannot import candidate modules."""
    return shlex.join(("python3", "-I", "-B", "-c", CHILD, json.dumps(expected, sort_keys=True)))


def record_success(expected, stdout, rounds, rnd):
    """Require the native positive receipt, then retain it outside the workspace."""
    records = [line[len(MARKER) :] for line in stdout.splitlines() if line.startswith(MARKER)]
    if len(records) != 1 or json.loads(records[0]) != expected:
        raise RuntimeError("native public contract preflight did not verify the frozen input inventory")
    path = Path(rounds) / f"round_{rnd:02d}.public_contract_preflight.json"
    receipt = dict(expected, native_readability_verified=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
