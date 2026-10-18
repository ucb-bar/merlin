"""Each simulator tier keeps the exact program it executed, and the capsule receipt names it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from merlin.targetgen import build_cache, executed_program, oracle_readback
from merlin.targetgen.capsule_source_identity import _capture_compiler_source, _finish_compiler_source


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build(workdir: Path, elf: bytes, harness: bytes, kernel: bytes = b"kernel object") -> dict:
    """One tier's link into the shared build directory, and its completed build receipt."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "package_kernel.elf").write_bytes(elf)
    (workdir / "harness.c").write_bytes(harness)
    (workdir / "kernel.o").write_bytes(kernel)
    return {
        "elf_sha256": _sha(elf),
        "harness_sha256": _sha(harness),
        "kernel_object_name": "kernel.o",
        "kernel_object_sha256": _sha(kernel),
        "build_identity_sha256": "b" * 64,
        "readback_policy": {"transport": "selected"},
    }


def _result(workdir: Path, receipt: dict, engine_sha256: str) -> dict:
    return {
        "elf": str(workdir / "package_kernel.elf"),
        "oracle": {"kind": "engine", "provenance": {"sha256": engine_sha256}},
        "readback_build": receipt,
    }


def test_each_tier_keeps_its_own_program_after_the_next_tier_relinks(tmp_path):
    work = tmp_path / "generated"
    first = _build(work, b"console caller ELF", b"console caller")
    executed_program.retain(_result(work, first, "1" * 64), sim="spike", workdir=work)
    second = _build(work, b"memory caller ELF", b"memory caller")
    executed_program.retain(_result(work, second, "2" * 64), sim="gsim", workdir=work)
    records = executed_program.collect(work)
    assert [(row["engine"], row["engine_sha256"], row["status"]) for row in records] == [
        ("spike", "1" * 64, "retained"),
        ("gsim", "2" * 64, "retained"),
    ]
    spike, gsim = records
    assert spike["elf"]["sha256"] == _sha(b"console caller ELF") != gsim["elf"]["sha256"]
    assert spike["kernel_object"] == gsim["kernel_object"] == {"name": "kernel.o", "sha256": _sha(b"kernel object")}
    assert (work / spike["elf"]["path"]).read_bytes() == b"console caller ELF"
    assert (work / spike["harness"]["path"]).read_bytes() == b"console caller"
    assert spike["elf"]["path"] == f"executed/spike/{_sha(b'console caller ELF')}/package_kernel.elf"
    assert (work / "package_kernel.elf").read_bytes() == b"memory caller ELF"


def test_unexecuted_or_inconsistent_runs_are_not_retained_as_programs(tmp_path):
    work = tmp_path / "generated"
    receipt = _build(work, b"elf", b"caller")
    executed_program.retain(
        {"status": "refused_before_execution", "elf": str(work / "package_kernel.elf")}, sim="spike", workdir=work
    )
    assert executed_program.collect(work) == []
    executed_program.retain(_result(work, dict(receipt, elf_sha256="0" * 64), "1" * 64), sim="spike", workdir=work)
    (record,) = executed_program.collect(work)
    assert record["status"] == "unretained" and "elf" in record["reason"]
    assert not list((work / "executed").glob("*/*/package_kernel.elf"))


def test_a_changed_copy_is_reported_rather_than_trusted(tmp_path):
    work = tmp_path / "generated"
    receipt = _build(work, b"elf", b"caller")
    executed_program.retain(_result(work, receipt, "1" * 64), sim="spike", workdir=work)
    (record,) = executed_program.collect(work)
    (work / record["elf"]["path"]).write_bytes(b"tampered")
    assert executed_program.collect(work)[0]["status"] == "copy_changed"


def test_retention_never_raises_into_the_grade(tmp_path):
    blocked = tmp_path / "generated"
    receipt = _build(blocked, b"elf", b"caller")
    (blocked / "executed").write_text("not a directory")
    res = _result(blocked, receipt, "1" * 64)
    assert executed_program.retain(res, sim="spike", workdir=blocked) is res


def test_compiler_receipt_names_only_this_invocations_programs(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "compiler.py").write_text("print('compiler')\n")
    work = tmp_path / "generated"
    stale = _build(work, b"an earlier grade's ELF", b"earlier caller")
    executed_program.retain(_result(work, stale, "1" * 64), sim="spike", workdir=work)
    captured = _capture_compiler_source(package, work)
    assert not (work / "executed").exists()
    current = _build(work, b"this grade's ELF", b"this caller")
    executed_program.retain(_result(work, current, "1" * 64), sim="spike", workdir=work)
    receipt = _finish_compiler_source(captured, work)
    assert receipt["status"] == "stable"
    assert [row["elf"]["sha256"] for row in receipt["executed_programs"]] == [_sha(b"this grade's ELF")]
    assert receipt["artifacts"]["package_kernel.elf"] == _sha(b"this grade's ELF")


def test_build_cache_never_stores_run_evidence(tmp_path):
    work = tmp_path / "generated"
    receipt = _build(work, b"elf", b"caller")
    executed_program.retain(_result(work, receipt, "1" * 64), sim="spike", workdir=work)
    stored = build_cache.contents(work)
    assert "package_kernel.elf" in stored and "harness.c" in stored
    assert not any(Path(name).parts[0] == "executed" for name in stored)


def test_the_readback_adapter_retains_what_each_oracle_run_executed(tmp_path, monkeypatch):
    from merlin.targetgen.contract import compile as oot_compile

    work = tmp_path / "generated"
    receipt = _build(work, b"elf", b"caller")

    def run_on_oracle(cb, llvm_text, *, simulator, workdir, **_):
        return _result(Path(workdir), receipt, "3" * 64)

    monkeypatch.setattr(oot_compile, "run_on_oracle", run_on_oracle)
    policy = SimpleNamespace(transport="serial-console")
    oracle_readback._run(
        {}, "", sim="spike", target="fixture", backend=object(), workdir=work, timeout=1, policy=policy
    )
    (record,) = executed_program.collect(work)
    assert (record["engine"], record["engine_sha256"]) == ("spike", "3" * 64)
    assert json.loads((work / Path(record["elf"]["path"]).parent / "executed_program.json").read_text()) == record
