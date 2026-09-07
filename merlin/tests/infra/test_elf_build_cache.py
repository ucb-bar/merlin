"""The ELF build cache: what it keys on, and every direction in which it must refuse.

A build cache is only safe while its key covers every input that reaches the executable, so the tests
that matter are the ones that MUTATE one input and demand a different key. A test that only asserts
"same inputs hit" passes on a cache that always hits, which is the defect, not the feature.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import build_cache as BC

ELF = "package_kernel.elf"


class _Recipe:
    """The smallest object with the surface :func:`build_cache.recipe_token` reads."""

    def __init__(self, script: Path, support: Path, march: str = "-march=rv64gc",
                 load_address: int = 0x8000_0000, flag: str = "-O2"):
        self.link_script, self.support_sources = script, (support,)
        self.load_address, self._march, self._flag = load_address, march, flag

    def march(self):
        return self._march

    def compile_command(self, *, source, output):
        return ["/usr/bin/false-compiler", self._flag, "-c", str(source), "-o", str(output)]

    def link_command(self, *, objects, output, link_script):
        return ["/usr/bin/false-compiler", "-T", str(link_script), "-o", str(output),
                *[str(o) for o in objects]]


@pytest.fixture
def recipe(tmp_path):
    script = tmp_path / "link.ld"
    script.write_text("SECTIONS { . = 0x80000000; }")
    support = tmp_path / "crt.S"
    support.write_text(".globl _start\n_start: j _start\n")
    return _Recipe(script, support)


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "store"
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE", str(root))
    return root


#: Stands in for the build path in the key tests. A REAL file, so editing it must move the key --
#: which is the property :func:`test_the_code_that_performs_the_build_is_in_the_key` mutates.
LOWERING = repo_root() / "merlin/python/merlin/llvmlower/pipeline.py"


@pytest.fixture
def key_of(recipe, monkeypatch, store):
    """`build_identity` with the two host probes stubbed, so these tests need neither a cross-compiler
    nor a registered backend -- what is under test here is which inputs reach the key."""
    monkeypatch.setattr(BC, "toolchain_token", lambda compiler: "toolchain-sha")
    monkeypatch.setattr(BC, "build_path", lambda target=None: (LOWERING,))

    def make(**over):
        args = {"target": "some_target", "lowered_mlir_text": "module {}", "cb": {"commands": []},
                "inputs": None, "recipe": recipe}
        args.update(over)
        return BC.build_identity(**args)
    return make


def _build(work: Path, text: str = "elf-bytes") -> Path:
    """Stand in for a compile: leave an executable and the intermediates beside it."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "kernel.ll").write_text("; ir")
    (work / "harness.c").write_text("int main(void){return 0;}")
    elf = work / ELF
    elf.write_text(text)
    elf.chmod(0o755)
    return elf


# --------------------------------------------------------------------------------------------
# THE KEY -- every mutation must be visible, and the paired direction must not be
# --------------------------------------------------------------------------------------------

def test_identical_inputs_give_one_key(key_of):
    assert key_of() is not None and key_of() == key_of()


@pytest.mark.parametrize("mutation", [
    {"lowered_mlir_text": "module { func.func @k() { return } }"},
    {"cb": {"commands": [{"opcode": "MVIN"}]}},
    {"inputs": {"A": [1, 2, 3]}},
    {"target": "another_target"},
])
def test_every_input_that_reaches_the_executable_changes_the_key(key_of, mutation):
    assert key_of(**mutation) != key_of(), f"{sorted(mutation)} did not change the key"


def test_a_recipe_change_changes_the_key(key_of, recipe, tmp_path):
    base = key_of()
    assert key_of(recipe=_Recipe(recipe.link_script, recipe.support_sources[0], flag="-O0")) != base
    assert key_of(recipe=_Recipe(recipe.link_script, recipe.support_sources[0],
                                 march="-march=rv64gcv")) != base
    assert key_of(recipe=_Recipe(recipe.link_script, recipe.support_sources[0],
                                 load_address=0x9000_0000)) != base


def test_support_source_and_link_script_bytes_are_in_the_key(key_of, recipe):
    """They are compiled INTO the executable, so editing one is a different build."""
    base = key_of()
    recipe.support_sources[0].write_text(".globl _start\n_start: nop\n j _start\n")
    assert key_of() != base
    after_support = key_of()
    recipe.link_script.write_text("SECTIONS { . = 0x90000000; }")
    assert key_of() != after_support


def test_the_code_that_performs_the_build_is_in_the_key(key_of):
    """An edit to the lowering pipeline emits different code; a stored build must not answer for it."""
    src = LOWERING
    base = key_of()
    original = src.read_bytes()
    try:
        src.write_bytes(original + b"\n# mutation\n")
        assert key_of() != base
    finally:
        src.write_bytes(original)
    assert key_of() == base


def test_no_key_without_an_establishable_toolchain(recipe, monkeypatch, store):
    """The toolchain is outside the repo and outside every hardware pin: unresolved means no cache."""
    monkeypatch.setattr(BC, "build_path", lambda target=None: (LOWERING,))
    monkeypatch.setattr(BC, "toolchain_token", lambda compiler: None)
    assert BC.build_identity(target="t", lowered_mlir_text="m", cb={}, inputs=None,
                             recipe=recipe) is None


def test_no_key_without_an_establishable_build_path(recipe, monkeypatch, store):
    """A build path missing a member would key executables on a partial compiler."""
    monkeypatch.setattr(BC, "toolchain_token", lambda compiler: "toolchain-sha")
    monkeypatch.setattr(BC, "build_path", lambda target=None: None)
    assert BC.build_identity(target="t", lowered_mlir_text="m", cb={}, inputs=None,
                             recipe=recipe) is None


def test_the_real_build_path_refuses_a_target_with_no_backend():
    """The target's backend owns the harness renderer and the recipe, so it is part of the builder.
    A target that resolves to none leaves the builder unestablished -- not partially established."""
    assert BC.build_path() is not None, "merlin's own build modules must be locatable"
    assert BC.build_path("no_such_target_is_registered") is None


def test_no_key_when_a_support_source_cannot_be_read(key_of, recipe):
    recipe.support_sources[0].unlink()
    assert key_of() is None


def test_no_key_when_switched_off(key_of, monkeypatch):
    assert key_of() is not None
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE", "0")
    assert BC.disabled() and key_of() is None and BC.cache_root() is None


def test_unresolvable_toolchain_reports_none(monkeypatch):
    BC._TOOLCHAIN_MEMO.clear()
    monkeypatch.delenv("MERLIN_ELF_BUILD_CACHE", raising=False)
    assert BC.toolchain_token("no-such-compiler-anywhere-on-this-host") is None


# --------------------------------------------------------------------------------------------
# THE STORE -- a hit reproduces the whole build; anything doubtful is a miss
# --------------------------------------------------------------------------------------------

def test_round_trip_restores_every_file_the_build_produced(key_of, tmp_path):
    key, first = key_of(), tmp_path / "first"
    before = BC.snapshot(first)
    _build(first)
    BC.store(key, first, before, ELF)

    second = tmp_path / "second"
    restored = BC.reuse(second, key, ELF)
    assert restored is not None and restored == second / ELF
    assert restored.read_text() == "elf-bytes"
    assert (second / "kernel.ll").read_text() == "; ir"
    assert (second / "harness.c").read_text() == "int main(void){return 0;}"
    assert os.stat(restored).st_mode & 0o111, "the executable must come back executable"


def test_a_miss_is_a_miss(key_of, tmp_path):
    _build(tmp_path / "first")
    BC.store(key_of(), tmp_path / "first", {}, ELF)
    other = key_of(lowered_mlir_text="module { // different }")
    assert BC.reuse(tmp_path / "second", other, ELF) is None


def test_a_corrupted_entry_is_a_miss_and_leaves_nothing_behind(key_of, tmp_path, store):
    key = key_of()
    _build(tmp_path / "first")
    BC.store(key, tmp_path / "first", {}, ELF)
    stored = next(p for p in store.rglob(f"files/{ELF}"))
    stored.write_text("tampered")

    second = tmp_path / "second"
    assert BC.reuse(second, key, ELF) is None
    assert not (second / ELF).exists(), "a partial restore must not look like a build"


def test_a_truncated_entry_is_a_miss(key_of, tmp_path, store):
    key = key_of()
    _build(tmp_path / "first")
    BC.store(key, tmp_path / "first", {}, ELF)
    next(p for p in store.rglob("files/kernel.ll")).unlink()
    assert BC.reuse(tmp_path / "second", key, ELF) is None


def test_a_record_from_another_version_is_not_read(key_of, tmp_path, store):
    key = key_of()
    _build(tmp_path / "first")
    BC.store(key, tmp_path / "first", {}, ELF)
    rec = next(store.rglob("record.json"))
    payload = json.loads(rec.read_text())
    payload["version"] = BC.RECORD_VERSION + 1
    rec.write_text(json.dumps(payload))
    assert BC.reuse(tmp_path / "second", key, ELF) is None


def test_no_key_never_hits_and_never_writes(tmp_path, store):
    _build(tmp_path / "w")
    BC.store(None, tmp_path / "w", {}, ELF)
    assert BC.reuse(tmp_path / "w2", None, ELF) is None
    assert not store.exists() or not any(store.rglob("record.json"))


# --------------------------------------------------------------------------------------------
# THE MARKER -- one ladder builds once, and a marker never speaks for a previous grade
# --------------------------------------------------------------------------------------------

def test_the_marker_makes_a_second_tier_free(key_of, tmp_path, store, monkeypatch):
    key, work = key_of(), tmp_path / "w"
    _build(work)
    BC.store(key, work, {}, ELF)
    assert (work / BC.KEY_MARKER).read_text() == key
    # With the store made unreachable, only the marker can answer -- so a hit here proves the ladder's
    # second tier never touches it.
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE", str(tmp_path / "elsewhere"))
    assert BC.reuse(work, key, ELF) == work / ELF


def test_the_marker_does_not_answer_for_a_different_key(key_of, tmp_path, monkeypatch):
    work = tmp_path / "w"
    _build(work)
    BC.store(key_of(), work, {}, ELF)
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE", str(tmp_path / "elsewhere"))
    assert BC.reuse(work, key_of(lowered_mlir_text="other"), ELF) is None


def test_forget_drops_the_claim(key_of, tmp_path, monkeypatch):
    work = tmp_path / "w"
    _build(work)
    BC.store(key_of(), work, {}, ELF)
    BC.forget(work)
    assert not (work / BC.KEY_MARKER).exists()
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE", str(tmp_path / "elsewhere"))
    assert BC.reuse(work, key_of(), ELF) is None
    BC.forget(work)                      # idempotent: a directory with no marker is not an error


def test_snapshot_separates_what_the_build_produced(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    (work / "pre_existing.json").write_text("{}")
    before = BC.snapshot(work)
    _build(work)
    after = BC.snapshot(work)
    produced = {rel for rel, sha in after.items() if before.get(rel) != sha}
    assert ELF in produced and "kernel.ll" in produced
    assert "pre_existing.json" not in produced


def test_the_store_stays_bounded(key_of, tmp_path, store, monkeypatch):
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE_ENTRIES", "3")
    for i in range(6):
        work = tmp_path / f"w{i}"
        _build(work, text=f"elf-{i}")
        BC.store(key_of(lowered_mlir_text=f"module {i}"), work, {}, ELF)
    entries = [d for shard in store.iterdir() if shard.is_dir() for d in shard.iterdir()]
    assert len(entries) <= 3
