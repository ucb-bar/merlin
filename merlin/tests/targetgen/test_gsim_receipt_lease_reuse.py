"""Strict gSIM receipt pins: one full read per file per process, never a weaker custody check.

A strict receipt pins thousands of toolchain, emitter and harness files, and one capsule's grade
reverifies it dozens of times (engine selection, every pre/post-launch identity check). These tests
pin what the reuse may and may not do: unchanged pins are not re-read, a duplicated path is read once,
and every way of changing a pinned file is detected exactly as the frozen-input lease cache detects it.
Real Linux read leases on the test's own temporary files; no simulator, compiler or target is involved.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path

import pytest

from merlin.common import provenance
from merlin.common import read_lease_digest as L
from merlin.targetgen import gsim_emulator as GE


@pytest.fixture
def leases(monkeypatch):
    L.close()
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    monkeypatch.delenv("MERLIN_SNAPSHOT_LEASE_MIN_BYTES", raising=False)
    monkeypatch.delenv("MERLIN_SNAPSHOT_LEASE_MAX_FILES", raising=False)
    yield
    L.close()


@pytest.fixture
def no_leases(monkeypatch):
    L.close()
    monkeypatch.delenv("MERLIN_SNAPSHOT_LEASE_CACHE", raising=False)
    yield
    L.close()


@pytest.fixture
def reads(monkeypatch):
    """Every full content read of a pin: lease admissions and ordinary full digests."""
    observed: list[Path] = []
    leased, full = L.hashlib.file_digest, provenance.file_digest

    def leased_read(stream, *args):
        observed.append(Path(os.readlink(f"/proc/self/fd/{stream.fileno()}")))
        return leased(stream, *args)

    def full_read(path):
        observed.append(Path(path))
        return full(path)

    monkeypatch.setattr(L.hashlib, "file_digest", leased_read)
    monkeypatch.setattr(provenance, "file_digest", full_read)
    return observed


def _pin(path: Path, data: bytes) -> dict:
    path.write_bytes(data)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def _receipt(root: Path) -> dict:
    """A strict v3 receipt whose wrapper and compiler name the SAME file, as a real receipt does."""
    root = root.resolve()
    compiler = _pin(root / "clang", b"compiler bytes " * 64)
    artifacts = {
        "firrtl": _pin(root / "design.fir", b"circuit Top :\n" * 32),
        "model_manifest": _pin(root / "model_manifest.json", b'{"model": 1}\n'),
        "binary": _pin(root / "emulator", b"\x7fELF emulator" * 128),
    }
    tools = {"gsim_emitter": _pin(root / "gsim", b"emitter" * 32), "cxx_wrapper": compiler, "cxx_compiler": compiler}
    inputs = [
        {"role": f"support:{name}", **_pin(root / name, f"// {name}\n".encode())} for name in ("a.h", "b.h", "c.cpp")
    ]
    commands = [
        {"stage": "emit", "cwd": str(root), "argv": [tools["gsim_emitter"]["path"], "design.fir"]},
        {"stage": "compile", "cwd": str(root), "argv": [compiler["path"], "model.cpp"]},
        {"stage": "link", "cwd": str(root), "argv": [compiler["path"], "model.o"]},
    ]
    return {
        "schema_version": GE.STRICT_RECEIPT_SCHEMA,
        "status": "complete",
        "provenance": {
            "firrtl_boundary": GE.FIRRTL_BOUNDARY_ADOPTED,
            "elaboration_performed": False,
            "warning": GE.ADOPTED_FIRRTL_WARNING,
        },
        "firrtl_sha256": artifacts["firrtl"]["sha256"],
        "model_manifest_sha256": artifacts["model_manifest"]["sha256"],
        "binary_sha256": artifacts["binary"]["sha256"],
        "artifacts": artifacts,
        "tools": tools,
        "inputs": inputs,
        "inputs_sha256": GE._canonical_sha(inputs),
        "commands": commands,
        "commands_sha256": GE._canonical_sha(commands),
    }


def _unique_pins(doc: dict) -> set[Path]:
    rows = [*doc["artifacts"].values(), *doc["tools"].values(), *doc["inputs"]]
    return {Path(row["path"]) for row in rows}


def _warm(doc: dict, reads: list) -> None:
    assert GE._strict_receipt_error(doc) is None
    assert sorted(reads) == sorted(_unique_pins(doc))  # each distinct pin read exactly once
    if L._CACHE is None or len(L._CACHE.entries) != len(_unique_pins(doc)):
        pytest.skip("Linux read leases unavailable for the selected temporary filesystem")
    reads.clear()


def test_unchanged_pins_are_read_once_per_process(tmp_path, leases, reads):
    doc = _receipt(tmp_path)
    _warm(doc, reads)
    for _ in range(13):  # one engine-selection call's worth of strict checks
        assert GE._strict_receipt_error(doc) is None
    assert reads == []


@pytest.mark.usefixtures("no_leases")
def test_duplicate_pin_path_is_read_once_per_check_without_the_cache(tmp_path, reads):
    doc = _receipt(tmp_path)
    assert doc["tools"]["cxx_wrapper"]["path"] == doc["tools"]["cxx_compiler"]["path"]
    for _ in range(2):
        assert GE._strict_receipt_error(doc) is None
        assert sorted(reads) == sorted(_unique_pins(doc))  # 7 rows name 7 paths; the 8th row repeats one
        reads.clear()


@pytest.mark.usefixtures("no_leases")
def test_duplicate_pin_with_a_different_claim_still_fails(tmp_path):
    doc = _receipt(tmp_path)
    doc["tools"]["cxx_compiler"] = {**doc["tools"]["cxx_compiler"], "sha256": "0" * 64}
    assert GE._strict_receipt_error(doc) == f"v3 receipt tool pin changed: {doc['tools']['cxx_compiler']['path']}"


def _rewrite_same_size_same_mtime(path: Path, data: bytes) -> None:
    before = path.stat()
    assert len(data) == before.st_size
    with path.open("r+b") as stream:  # a write-open breaks the read lease before any byte changes
        stream.write(data)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


@pytest.mark.parametrize("same_bytes", [False, True])
def test_in_place_rewrite_with_restored_mtime_refuses(tmp_path, leases, reads, same_bytes):
    doc = _receipt(tmp_path)
    _warm(doc, reads)
    target = Path(doc["inputs"][1]["path"])
    original = target.read_bytes()
    swapped = original if same_bytes else bytes(reversed(original))
    assert same_bytes or swapped != original
    _rewrite_same_size_same_mtime(target, swapped)
    with pytest.raises(RuntimeError, match="lease custody changed"):
        GE._strict_receipt_error(doc)
    # The refused process cache never turns into new hits; later checks are full reads again.
    expected = None if same_bytes else f"v3 receipt input pin changed: {target}"
    assert GE._strict_receipt_error(doc) == expected
    assert target in reads and len(reads) == len(set(reads))  # full reads, each pin at most once


def test_truncation_refuses(tmp_path, leases, reads):
    doc = _receipt(tmp_path)
    _warm(doc, reads)
    target = Path(doc["artifacts"]["binary"]["path"])
    try:
        os.close(os.open(target, os.O_WRONLY | os.O_TRUNC | os.O_NONBLOCK))
    except OSError as error:  # The custodian may not have released yet; the break is already pending.
        assert error.errno in (errno.EAGAIN, errno.EWOULDBLOCK)
    with pytest.raises(RuntimeError, match="lease custody changed"):
        GE._strict_receipt_error(doc)


@pytest.mark.parametrize("same_bytes", [False, True])
def test_path_replaced_by_another_inode_refuses(tmp_path, leases, reads, same_bytes):
    doc = _receipt(tmp_path)
    _warm(doc, reads)
    target = Path(doc["tools"]["cxx_compiler"]["path"])
    replacement = tmp_path / "replacement"
    replacement.write_bytes(target.read_bytes() if same_bytes else b"other compiler" * 70)
    os.replace(replacement, target)
    with pytest.raises(RuntimeError, match="lease custody changed"):
        GE._strict_receipt_error(doc)
    expected = None if same_bytes else f"v3 receipt tool pin changed: {target}"
    assert GE._strict_receipt_error(doc) == expected


def test_symlink_replacement_keeps_the_original_refusal(tmp_path, leases, reads):
    doc = _receipt(tmp_path)
    _warm(doc, reads)
    target = Path(doc["inputs"][0]["path"])
    copy = tmp_path / "copy.h"
    copy.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(copy)
    assert GE._strict_receipt_error(doc) == f"v3 receipt input pin is absent, relative, or a symlink: {target}"


def test_custodian_loss_refuses(tmp_path, leases, reads):
    doc = _receipt(tmp_path)
    _warm(doc, reads)
    L._CACHE.process.kill()
    L._CACHE.process.wait(timeout=3)
    with pytest.raises(RuntimeError, match="lease custody changed"):
        GE._strict_receipt_error(doc)
    assert GE._strict_receipt_error(doc) is None
    assert sorted(reads) == sorted(_unique_pins(doc))


@pytest.mark.usefixtures("no_leases")
def test_without_the_cache_a_same_size_same_mtime_swap_fails_by_content(tmp_path):
    doc = _receipt(tmp_path)
    target = Path(doc["inputs"][2]["path"])
    _rewrite_same_size_same_mtime(target, bytes(reversed(target.read_bytes())))
    assert GE._strict_receipt_error(doc) == f"v3 receipt input pin changed: {target}"


def test_partial_capacity_rechecks_reused_pins_after_slower_reads(tmp_path, leases, reads, monkeypatch):
    doc = _receipt(tmp_path)
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_MAX_FILES", "3")
    assert GE._strict_receipt_error(doc) is None
    if L._CACHE is None or len(L._CACHE.entries) != 3:
        pytest.skip("Linux read leases unavailable for the selected temporary filesystem")
    cached = {Path(path) for path in L._CACHE.paths}
    reads.clear()
    assert GE._strict_receipt_error(doc) is None
    assert sorted(reads) == sorted(_unique_pins(doc) - cached)  # only pins outside the cache
    reads.clear()

    # A writer that arrives while uncached pins are being read must still refuse the whole check.
    victim = sorted(cached)[0]
    full = provenance.file_digest

    def writer_during_slow_read(path):
        try:
            os.close(os.open(victim, os.O_WRONLY | os.O_NONBLOCK))
        except OSError as error:
            assert error.errno in (errno.EAGAIN, errno.EWOULDBLOCK)
        return full(path)

    monkeypatch.setattr(provenance, "file_digest", writer_during_slow_read)
    with pytest.raises(RuntimeError, match="lease custody changed"):
        GE._strict_receipt_error(doc)


def test_repeated_engine_resolution_reuses_pins(tmp_path, leases, reads):
    """``_validate_receipt`` is what engine selection and every identity check call."""
    doc = _receipt(tmp_path)
    receipt = tmp_path / GE.RECEIPT_NAME
    receipt.write_text(json.dumps(doc), encoding="utf-8")
    binary = Path(doc["artifacts"]["binary"]["path"])
    status, _, block = GE._validate_receipt("fixture_np", binary, doc["binary_sha256"], receipt)
    assert status == "bound" and block["firrtl_sha256"] == doc["firrtl_sha256"]
    if L._CACHE is None or not _unique_pins(doc) <= set(L._CACHE.paths):
        pytest.skip("Linux read leases unavailable for the selected temporary filesystem")
    reads.clear()
    for _ in range(13):
        assert GE._validate_receipt("fixture_np", binary, doc["binary_sha256"], receipt)[0] == "bound"
    assert not set(reads) & _unique_pins(doc)
