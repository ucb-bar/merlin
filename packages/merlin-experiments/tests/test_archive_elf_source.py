"""Retained executable custody; no target program or simulator executes."""

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from merlin.common import invocation_record as I
from merlin_experiments.phase2.archive_elf_source import review_native_elf_relocation


@pytest.fixture
def linked(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    source, elf = original / "fixture.o", original / "program.elf"
    source.write_bytes(b"custody fixture only, never executed as an ELF")
    argv = [shutil.which("cp"), str(source), str(elf)]
    with I.observe(original, stage="elf", argv=argv, inputs=(source,), outputs=(elf,), env={}) as r:
        r.complete(subprocess.run(argv, check=True, capture_output=True))
    destination = tmp_path / "copy.elf"
    shutil.copyfile(elf, destination)
    return SimpleNamespace(source=source, elf=elf, destination=destination, record=r.path,
                           limits=SimpleNamespace(max_elf_bytes=1 << 20, max_bins=100))


def test_exact_copy_retains_original_producer_without_rewriting_it(linked):
    before = linked.record.read_bytes()
    result = review_native_elf_relocation(linked.elf, linked.destination, linked.limits)
    pin = result.producer(linked.destination, linked.limits)
    assert pin == result.destination
    assert result.record()["original"]["path"] == str(linked.elf)
    assert result.record()["new_native_producer"] is False
    assert linked.record.read_bytes() == before


@pytest.mark.parametrize("damage", ["source", "destination", "input", "record", "absent_record"])
def test_custody_mutation_refuses_before_reuse(linked, damage):
    result = review_native_elf_relocation(linked.elf, linked.destination, linked.limits)
    if damage == "source":
        linked.elf.write_bytes(b"different source executable")
    elif damage == "destination":
        linked.destination.write_bytes(b"different destination executable")
    elif damage == "input":
        linked.source.write_bytes(b"modified original link input")
    elif damage == "record":
        v = json.loads(linked.record.read_text())
        v["stage"] = "foreign_stage"
        linked.record.write_text(json.dumps(v))
    else:
        linked.record.unlink()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        result.producer(linked.destination, linked.limits)
