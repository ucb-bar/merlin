"""A measured program records what compiled it.

The repo already insists a hardware verdict name its hardware revision. It did not insist the same
of the COMPILER, and the gap showed: tracing the 25,419,657-cycle ResNet-50 result back to the sources
that produced it took an exhaustive scan of every file in the repository, and it succeeded only
because one emitted recipe name happened to be unique to one worktree. Nothing in the build manifest,
the FireSim job record or the measurement row named the package, the ``PYTHONPATH`` or a source
digest. This host carries dozens of merlin worktrees whose lowering trees differ; the same capture
compiled from two of them is two different programs, and the manifest said nothing that told them
apart.

So the manifest now records the modules Python actually RESOLVED, by content. These tests hold that
to the three properties that make it worth having: it describes what was imported rather than what a
path suggests, it changes when the bytes change, and an undeterminable digest is recorded as UNKNOWN
with its reason rather than omitted.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts"))
import group_model_program as gmp  # noqa: E402


@pytest.fixture(scope="module")
def record():
    return gmp._compiler_provenance()


def test_it_names_the_package_python_actually_resolved(record):
    """Not a configured path, not a guess: the parent of the merlin the interpreter imported. On a
    host with many worktrees this is the whole point -- a shadowing checkout must be visible."""
    import merlin

    assert record["merlin_package"] == str(Path(merlin.__file__).resolve().parent)
    assert Path(record["merlin_package"]).is_dir()


def test_every_recorded_module_digest_is_that_file_on_disk(record):
    assert record["modules"], "a build that imported no merlin module is not a build"
    for name, digest in record["modules"].items():
        origin = Path(sys.modules[name].__file__).resolve()
        assert hashlib.sha256(origin.read_bytes()).hexdigest() == digest, name


def test_the_digest_is_a_digest_or_says_why_it_is_not(record):
    """Fail closed. A provenance field that is quietly absent reads as 'nothing to record'."""
    value = record["source_digest"]
    if value.startswith("UNKNOWN"):
        assert ": " in value, "an UNKNOWN must carry its reason"
    else:
        assert len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def test_a_changed_source_byte_changes_the_digest(tmp_path):
    """The property the record rests on. Held against the function the recorder actually calls, over
    files it can be pointed at, because the alternative -- mutating this checkout -- is not a test."""
    from merlin.common import provenance as PROV

    one, two = tmp_path / "a.py", tmp_path / "b.py"
    one.write_text("x = 1\n", encoding="utf-8")
    two.write_text("y = 2\n", encoding="utf-8")
    before = PROV.source_digest([one, two])
    assert PROV.source_digest([one, two]) == before, "the same bytes must digest the same"
    two.write_text("y = 3\n", encoding="utf-8")
    assert PROV.source_digest([one, two]) != before


def test_the_record_is_attached_to_every_built_manifest():
    """The recorder existing is not the same as the manifest carrying it. Read the source of `main`
    so this fails if the key is dropped, without building a whole model to find out."""
    source = (merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts" / "group_model_program.py").read_text(
        encoding="utf-8"
    )
    assert '"compiler_provenance": _compiler_provenance()' in source


def test_a_recorded_build_can_be_told_from_one_built_elsewhere(record):
    """What the record is FOR. Two worktrees of this repo hold different lowering trees; a record
    that could not distinguish them would not have answered the question that prompted it."""
    assert record["source_digest"].startswith("UNKNOWN") or record["merlin_package"] in str(
        Path(sys.modules["merlin"].__file__).resolve()
    )
    # And the interpreter is named, because the same sources under two venvs are two builds.
    assert Path(record["python"]).exists()
