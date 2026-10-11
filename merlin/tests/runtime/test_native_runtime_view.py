"""Bounded snapshot and routing controls; no native engine or kernel setup.

The fake executable bytes below are never loaded. A copied view cannot establish
namespace adoption or reset/seed closure; these tests claim neither.
"""

import copy
import json
import os
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.runtime import _native_files as F
from merlin.runtime import _native_launch as L
from merlin.runtime import native_runtime as N
from merlin.runtime.backends import chipyard_rocc as C
from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend
from merlin.targetgen import rtl_engine_policy
from merlin.targetgen.contract import artifact_retesting as A


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "managed-output"))
    system = tmp_path / "system"
    (system / "usr/bin").mkdir(parents=True)
    (system / "bin").mkdir()
    (system / "config").mkdir()
    for path in (system / "bin/bash", system / "usr/bin/env", tmp_path / "engine", tmp_path / "bwrap"):
        path.write_bytes(b"\x7fELF source control only; never executed")
        path.chmod(0o555)
    (system / "config/selected.conf").write_text("original runtime configuration")
    output = tmp_path / "output"
    output.mkdir()
    engine = tmp_path / "engine"
    block = {
        "binary": {"path": str(engine), "sha256": F.digest(engine)},
        "argv": ["{elf}"],
        "environment": {},
        "cwd": str(system / "config"),
    }
    backend = BoundRoCCBackend(
        "synthetic", F.document({"contract": {"runner": {"chipyard_rocc": {"engines": {"gsim": block}}}}}), None
    )
    kwargs = {
        "backend": backend,
        "engine": engine,
        "readonly_roots": ((system / "usr", Path("/usr")), (system / "bin", Path("/bin")), system / "config"),
        "output_root": output,
        "bwrap": tmp_path / "bwrap",
        "max_files": 256,
        "max_bytes": 1024 * 1024,
        "max_console_bytes": 4096,
    }
    owner = N.prepare_native_runtime_view(tmp_path / "snapshot", **kwargs)
    yield SimpleNamespace(**locals())
    # A failed integrity check intentionally retains the source pin as UNKNOWN.
    try:
        owner.close()
    except ValueError:
        pass


def test_snapshot_is_shared_and_has_no_mutable_source_hardlinks(frozen):
    f = frozen
    selected = N._guest_member(f.owner.image, str(f.system / "config/selected.conf"))
    source = f.system / "config/selected.conf"
    assert selected.read_bytes() == source.read_bytes()
    assert selected.stat().st_ino != source.stat().st_ino and selected.stat().st_nlink == 1
    assert selected.stat().st_mode & 0o222 == 0
    source.write_text("later host source edit")
    assert f.owner.verify()["runtime_view_sha256"] == f.owner._tree["sha256"]
    assert selected.read_text() == "original runtime configuration"


@pytest.mark.parametrize("damage", ["bytes", "backdated", "added", "mode"])
def test_snapshot_disagreement_refuses(frozen, damage):
    f = frozen
    selected = N._guest_member(f.owner.image, str(f.system / "config/selected.conf"))
    before = selected.stat()
    if damage in {"bytes", "backdated"}:
        selected.chmod(0o600)
        selected.write_text("changed frozen runtime")
        selected.chmod(0o444)
        if damage == "backdated":
            os.utime(selected, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif damage == "mode":
        selected.chmod(0o644)
    else:
        f.owner.image.chmod(0o755)
        (f.owner.image / "injected").write_text("unexpected lookup member")
    with pytest.raises(ValueError, match="snapshot"):
        f.owner.verify()


def test_snapshot_bounds_refuse_before_large_copy(frozen):
    for field, value in (("max_files", 1), ("max_bytes", 1)):
        with pytest.raises(ValueError, match="bound"):
            N.prepare_native_runtime_view(frozen.tmp_path / ("bounded-" + field), **{**frozen.kwargs, field: value})


def test_view_cannot_be_reconstructed_and_has_no_native_completion(frozen):
    with pytest.raises(ValueError, match="exact live"):
        copy.copy(frozen.owner).verify()
    with pytest.raises(ValueError, match="exact live"):
        with N.executing_native_runtime(frozen.owner.verify()):
            pass
    with pytest.raises(ValueError, match="no actual completed"):
        frozen.owner.completed_observation(elf=frozen.engine, console="invented console")


def test_unknown_seed_remains_unknown_and_namespace_paths_are_fixed(frozen):
    f = frozen
    elf = f.output / "first.elf"
    elf.write_bytes(b"same complete ELF bytes")
    chosen = f.owner.execution_identity(f.backend, elf)
    assert chosen["guest_argv"] == [str(f.engine), "/run/merlin/input.elf"]
    assert chosen["reset_seed_closure"]["status"] == "UNKNOWN"
    second = f.output / "another.elf"
    second.write_bytes(elf.read_bytes())
    assert f.owner.execution_identity(f.backend, second) == chosen
    assert not tuple(f.output.rglob("invocation.json"))


def test_uncovered_original_cwd_stays_fresh(frozen):
    f = frozen
    elf = f.output / "requested.elf"
    elf.write_text("complete artifact")
    with N.executing_native_runtime(f.owner):
        assert (
            N.run_in_native_view(
                f.backend,
                elf,
                argv=[str(f.engine), str(elf)],
                environment={},
                cwd=f.output,
                timeout=1,
                capture_bytes=False,
                unlimited_stack=False,
            )
            is None
        )


def test_archive_context_owner_parent_and_forgery_controls(frozen, monkeypatch):
    f = frozen
    context = {"original_admission": "fixed"}
    index = A.prepare_artifact_retest_index(
        f.output / "archive", context=context, runtime_owner=f.owner, max_entries=2, max_bytes=4096
    )
    assert index.verify(runtime_owner=f.owner, context=context)["max_entries"] == 2
    with pytest.raises(ValueError, match="admission"):
        index.verify(context={"original_admission": "changed"})
    with pytest.raises(ValueError, match="exact live"):
        copy.copy(index).verify()
    parent = os.getpid()
    monkeypatch.setattr(os, "getpid", lambda: parent + 1)
    with pytest.raises(ValueError, match="creating host parent"):
        index.publish_completed_job(f.output)
    with pytest.raises(ValueError, match="creating host parent"):
        index.discard_cleaned_job(f.output)
    monkeypatch.setattr(os, "getpid", lambda: parent)
    key = "0" * 64
    (index.root / "entries" / (key + ".json")).write_text(json.dumps({"body": {}, "signature": "forged"}))
    with pytest.raises(ValueError, match="not issued"):
        index.original_observation(key)


def test_archive_factory_refuses_saved_runtime_identity(frozen):
    with pytest.raises(ValueError, match="live runtime"):
        A.prepare_artifact_retest_index(
            frozen.output / "archive", context={}, runtime_owner=frozen.owner.verify(), max_entries=2, max_bytes=4096
        )


def test_original_neutral_default_argv_environment_and_slot_remain(monkeypatch, tmp_path):
    expected = ["original-engine", str(tmp_path / "program.elf"), "+max-cycles=123", "+loadmem=original"]
    monkeypatch.setattr(C, "_gsim_argv", lambda elf: expected)
    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", lambda **kwargs: nullcontext())
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "DONE\n", "")

    monkeypatch.setattr(C.subprocess, "run", run)
    assert C.run_elf(tmp_path / "program.elf", simulator="gsim", timeout=1) == "DONE\n"
    assert calls[0][0] == expected
    assert calls[0][1]["env"] == dict(os.environ)
    assert calls[0][1]["preexec_fn"] is C._unlimited_stack


def test_uncredentialed_ready_packet_cannot_supply_namespace_adoption():
    control = SimpleNamespace(settimeout=lambda value: None, recvmsg=lambda *args: (b"READY", [], 0, None))
    with pytest.raises(ValueError, match="kernel-authenticated"):
        L._ready(control, SimpleNamespace(poll=lambda: None), timeout=1)


@pytest.mark.parametrize("field,value", [("CapEff", "1"), ("CapPrm", "1"), ("NoNewPrivs", "0"), ("NSpid", "9 2")])
def test_unsafe_process_status_refuses(field, value):
    rows = {"CapEff": "0", "CapPrm": "0", "NoNewPrivs": "1", "NSpid": "9 1"}
    rows[field] = value
    with pytest.raises(ValueError, match="unprivileged private namespace leader"):
        L._closed_status("\n".join(f"{key}: {item}" for key, item in rows.items()))


def test_archive_unsafe_key_and_pending_bounds_refuse(frozen):
    index = A.prepare_artifact_retest_index(
        frozen.output / "archive", context={}, runtime_owner=frozen.owner, max_entries=2, max_bytes=4
    )
    with pytest.raises(ValueError, match="complete SHA256"):
        index.original_observation("../unrelated")
    job = frozen.output / "job"
    job.mkdir()
    (job / "output").write_bytes(b"oversized")
    with pytest.raises(ValueError, match="byte budget"):
        index._job_bytes(job)
    assert index.discard_cleaned_job(job) == 0


def test_imported_owners_are_from_this_managed_source():
    source = Path.cwd() / "src"
    for module in (N, L, F, C, A):
        assert Path(module.__file__).resolve().is_relative_to(source)
