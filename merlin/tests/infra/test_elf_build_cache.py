"""The ELF build cache: what it keys on, and every direction in which it must refuse.

A build cache is only safe while its key covers every input that reaches the executable, so the tests
that matter are the ones that MUTATE one input and demand a different key. A test that only asserts
"same inputs hit" passes on a cache that always hits, which is the defect, not the feature.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import module_source_path
from merlin.targetgen import build_cache as BC
from merlin.targetgen.contract.build_recipe import KernelStackFramePolicy

ELF = "package_kernel.elf"


class _Recipe:
    """The smallest object with the surface :func:`build_cache.recipe_token` reads."""

    def __init__(
        self,
        script: Path,
        support: Path,
        march: str = "-march=rv64gc",
        load_address: int = 0x8000_0000,
        flag: str = "-O2",
    ):
        self.link_script, self.support_sources = script, (support,)
        self.load_address, self._march, self._flag = load_address, march, flag

    def march(self):
        return self._march

    def compile_command(self, *, source, output):
        return ["/usr/bin/false-compiler", self._flag, "-c", str(source), "-o", str(output)]

    def link_command(self, *, objects, output, link_script):
        return ["/usr/bin/false-compiler", "-T", str(link_script), "-o", str(output), *[str(o) for o in objects]]


@pytest.fixture
def recipe(tmp_path):
    script = tmp_path / "link.ld"
    script.write_text("SECTIONS { . = 0x80000000; }")
    support = tmp_path / "crt.S"
    support.write_text(".globl _start\n_start: j _start\n")
    return _Recipe(script, support)


def test_stack_policy_is_part_of_recipe_identity(recipe):
    recipe.kernel_stack_frame = KernelStackFramePolicy("entry", 4096)
    first = BC.recipe_token(recipe)
    recipe.kernel_stack_frame = KernelStackFramePolicy("entry", 8192)
    second = BC.recipe_token(recipe)
    assert first["kernel_stack_frame"] == {"entry_symbol": "entry", "max_static_bytes": 4096}
    assert second["kernel_stack_frame"] == {"entry_symbol": "entry", "max_static_bytes": 8192}
    assert first != second


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "store"
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE", str(root))
    return root


@pytest.fixture
def lowering(tmp_path):
    """Copy selected imported source; mutation must not edit a checkout or installed wheel."""
    source = tmp_path / "pipeline.py"
    source.write_bytes(module_source_path("merlin.llvmlower.pipeline").read_bytes())
    return source


@pytest.fixture
def key_of(recipe, monkeypatch, store, lowering):
    """`build_identity` with the two host probes stubbed, so these tests need neither a cross-compiler
    nor a registered backend -- what is under test here is which inputs reach the key."""
    monkeypatch.setattr(BC, "toolchain_token", lambda compiler: "toolchain-sha")
    monkeypatch.setattr(BC, "build_path", lambda target=None: (lowering,))

    def make(**over):
        args = {
            "target": "some_target",
            "lowered_mlir_text": "module {}",
            "cb": {"commands": []},
            "inputs": None,
            "recipe": recipe,
        }
        args.update(over)
        return BC.build_identity(**args)

    return make


def _store(key, work, elf, metadata=None):
    BC.store(key, work, BC.BuildProducts(elf, products=tuple(BC.contents(work))), metadata)


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


def test_same_version_driver_replacement_invalidates_toolchain_token(tmp_path):
    driver = tmp_path / "compiler"
    driver.write_text("#!/bin/sh\n# one\necho fixed-version\n")
    driver.chmod(0o755)
    first = BC.toolchain_token(driver)
    assert first is not None
    stamp = driver.stat().st_mtime_ns
    driver.write_text("#!/bin/sh\n# two\necho fixed-version\n")
    os.utime(driver, ns=(stamp, stamp))
    assert BC.toolchain_token(driver) != first


def test_toolchain_path_selection_tracks_replaced_executable(tmp_path, monkeypatch):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    for directory, comment in ((one, "one"), (two, "two")):
        driver = directory / "compiler"
        driver.write_text(f"#!/bin/sh\n# {comment}\necho fixed-version\n")
        driver.chmod(0o755)
    monkeypatch.setenv("PATH", str(one))
    first = BC.toolchain_token("compiler")
    monkeypatch.setenv("PATH", str(two))
    assert BC.toolchain_token("compiler") != first


def test_backdated_same_size_source_edit_invalidates_memo(tmp_path):
    source = tmp_path / "lower.py"
    source.write_text("VALUE=1\n")
    stamp = 1_000_000_000
    os.utime(source, ns=(stamp, stamp))
    first = BC._build_path_digest((source,))
    source.write_text("VALUE=2\n")
    os.utime(source, ns=(stamp, stamp))
    assert BC._build_path_digest((source,)) != first


def test_explicit_zero_input_roster_does_not_reuse_an_absent_input_build(key_of, store, tmp_path):
    absent, explicit = key_of(inputs=None), key_of(inputs={})
    assert absent is not None and explicit is not None and absent != explicit
    work = tmp_path / "built"
    _build(work)
    _store(absent, work, ELF)
    assert BC.reuse(tmp_path / "zero_input", explicit, ELF) is None
    assert not (tmp_path / "zero_input" / ELF).exists()
    assert BC.reuse(tmp_path / "same_absent", absent, ELF) is not None


@pytest.mark.parametrize(
    "mutation",
    [
        {"lowered_mlir_text": "module { func.func @k() { return } }"},
        {"cb": {"commands": [{"opcode": "MVIN"}]}},
        {"inputs": {"A": [1, 2, 3]}},
        {"target": "another_target"},
    ],
)
def test_every_input_that_reaches_the_executable_changes_the_key(key_of, mutation):
    assert key_of(**mutation) != key_of(), f"{sorted(mutation)} did not change the key"


def test_a_recipe_change_changes_the_key(key_of, recipe, tmp_path):
    base = key_of()
    assert key_of(recipe=_Recipe(recipe.link_script, recipe.support_sources[0], flag="-O0")) != base
    assert key_of(recipe=_Recipe(recipe.link_script, recipe.support_sources[0], march="-march=rv64gcv")) != base
    assert key_of(recipe=_Recipe(recipe.link_script, recipe.support_sources[0], load_address=0x9000_0000)) != base


def test_distinct_link_driver_is_content_bound(key_of, recipe, monkeypatch):
    compile_token, link_token = "compile-one", "link-one"
    recipe.link_command = lambda **kwargs: ["link-driver", "-o", str(kwargs["output"])]
    monkeypatch.setattr(BC, "toolchain_token", lambda driver: link_token if driver == "link-driver" else compile_token)
    first = key_of()
    link_token = "link-two"
    assert key_of() != first
    monkeypatch.setattr(BC, "toolchain_token", lambda driver: None if driver == "link-driver" else compile_token)
    assert key_of() is None


@pytest.mark.parametrize("mutation", ["bytes", "path", "version", "missing", "unsupported", "selector_error"])
def test_selected_kernel_clang_invalidates_unchanged_recipe_build(
    recipe, store, lowering, tmp_path, monkeypatch, mutation
):
    """Source-only driver probes: never invoke a compiler or simulator."""
    from merlin.llvmlower import codegen, toolchain

    driver = tmp_path / "kernel-clang"
    driver.write_text("source-only selected compiler one\n")
    driver.chmod(0o755)
    version = "selected-clang-version-one"

    def probe(argv, **kwargs):
        assert argv[0] == str(driver)
        assert argv[1] in {"--version", "-dumpmachine"}
        return SimpleNamespace(returncode=0, stdout=version if argv[1] == "--version" else "riscv64-unknown-elf")

    import subprocess

    monkeypatch.setattr(subprocess, "run", probe)
    token = BC.toolchain_token
    monkeypatch.setattr(
        BC,
        "toolchain_token",
        lambda compiler: "unchanged-recipe-driver" if str(compiler) == "/usr/bin/false-compiler" else token(compiler),
    )
    monkeypatch.setattr(BC, "build_path", lambda target=None: (lowering,))
    # codegen imports its selector: selecting a different toolchain-module
    # alias must not change which compiler this actual original path consumes.
    monkeypatch.setattr(codegen, "clang", lambda: driver)
    monkeypatch.setattr(toolchain, "clang", lambda: pytest.fail("used a different compiler selector"))

    def key():
        return BC.build_identity(target="t", lowered_mlir_text="module {}", cb={}, inputs={}, recipe=recipe)

    initial = key()
    assert initial is not None
    work = tmp_path / "original-build"
    _build(work)
    BC.store(initial, work, ELF)
    assert BC.reuse(tmp_path / "same-selection", key(), ELF) is not None
    if mutation == "bytes":
        stamp = driver.stat().st_mtime_ns
        driver.write_text("source-only selected compiler two\n")
        os.utime(driver, ns=(stamp, stamp))
    elif mutation == "path":
        other = tmp_path / "other-clang"
        other.write_bytes(driver.read_bytes())
        other.chmod(0o755)
        driver = other
    elif mutation == "version":
        version = "selected-clang-version-two"
        driver.chmod(0o755)  # Invalidate the stat memo while preserving path and bytes.
    elif mutation == "missing":
        driver.unlink()
    elif mutation == "unsupported":
        monkeypatch.setattr(codegen, "clang", lambda: object())
    else:
        monkeypatch.setattr(codegen, "clang", lambda: (_ for _ in ()).throw(ValueError("no selected compiler")))
    changed = key()
    assert changed != initial
    if mutation in {"missing", "unsupported", "selector_error"}:
        assert changed is None
    assert BC.reuse(tmp_path / "changed-selection", changed, ELF) is None
    assert not (tmp_path / "changed-selection" / ELF).exists()


def test_support_source_and_link_script_bytes_are_in_the_key(key_of, recipe):
    """They are compiled INTO the executable, so editing one is a different build."""
    base = key_of()
    recipe.support_sources[0].write_text(".globl _start\n_start: nop\n j _start\n")
    assert key_of() != base
    after_support = key_of()
    recipe.link_script.write_text("SECTIONS { . = 0x90000000; }")
    assert key_of() != after_support


def test_declared_harness_header_bytes_are_in_the_key_and_missing_header_disables_cache(key_of, recipe, tmp_path):
    included = tmp_path / "include"
    included.mkdir()
    header = included / "transport.h"
    header.write_text("#define TRANSPORT_VERSION 1\n")
    recipe.include_roots = (included,)
    recipe.header_dependencies = (header,)
    first = key_of()
    assert first is not None
    header.write_text("#define TRANSPORT_VERSION 2\n")
    assert key_of() != first
    header.unlink()
    assert key_of() is None


def test_shadowed_harness_header_cannot_be_hashed_as_compiled_bytes(key_of, recipe, tmp_path):
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "transport.h").write_text("#define VERSION 1\n")
    second = second_root / "transport.h"
    second.write_text("#define VERSION 2\n")
    recipe.include_roots = (first_root, second_root)
    recipe.header_dependencies = (second,)
    assert key_of() is None


def test_the_code_that_performs_the_build_is_in_the_key(key_of, lowering):
    """An edit to the lowering pipeline emits different code; a stored build must not answer for it."""
    src = lowering
    base = key_of()
    original = src.read_bytes()
    try:
        src.write_bytes(original + b"\n# mutation\n")
        assert key_of() != base
    finally:
        src.write_bytes(original)
    assert key_of() == base


def test_no_key_without_an_establishable_toolchain(recipe, monkeypatch, store, lowering):
    """The toolchain is outside the repo and outside every hardware pin: unresolved means no cache."""
    monkeypatch.setattr(BC, "build_path", lambda target=None: (lowering,))
    monkeypatch.setattr(BC, "toolchain_token", lambda compiler: None)
    assert BC.build_identity(target="t", lowered_mlir_text="m", cb={}, inputs=None, recipe=recipe) is None


def test_no_key_without_an_establishable_build_path(recipe, monkeypatch, store):
    """A build path missing a member would key executables on a partial compiler."""
    monkeypatch.setattr(BC, "toolchain_token", lambda compiler: "toolchain-sha")
    monkeypatch.setattr(BC, "build_path", lambda target=None: None)
    assert BC.build_identity(target="t", lowered_mlir_text="m", cb={}, inputs=None, recipe=recipe) is None


def test_the_real_build_path_refuses_a_target_with_no_backend():
    """The target's backend owns the harness renderer and the recipe, so it is part of the builder.
    A target that resolves to none leaves the builder unestablished -- not partially established."""
    assert BC.build_path() is not None, "merlin's own build modules must be locatable"
    assert BC.build_path("no_such_target_is_registered") is None


def test_build_path_follows_imported_package_when_checkout_root_is_absent(tmp_path, monkeypatch):
    """Installed wheels have no legacy merlin/python links or checkout markers."""
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    paths = BC.build_path()
    assert paths is not None
    assert Path(BC.__file__).resolve() in paths
    assert all(path.is_relative_to(Path(BC.__file__).resolve().parents[2]) for path in paths)


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
    _build(first)
    _store(key, first, ELF)

    second = tmp_path / "second"
    restored = BC.reuse(second, key, ELF)
    assert restored is not None and restored == second / ELF
    assert restored.read_text() == "elf-bytes"
    assert (second / "kernel.ll").read_text() == "; ir"
    assert (second / "harness.c").read_text() == "int main(void){return 0;}"
    assert os.stat(restored).st_mode & 0o111, "the executable must come back executable"


def test_a_miss_is_a_miss(key_of, tmp_path):
    _build(tmp_path / "first")
    _store(key_of(), tmp_path / "first", ELF)
    other = key_of(lowered_mlir_text="module { // different }")
    assert BC.reuse(tmp_path / "second", other, ELF) is None


def test_a_corrupted_entry_is_a_miss_and_leaves_nothing_behind(key_of, tmp_path, store):
    key = key_of()
    _build(tmp_path / "first")
    _store(key, tmp_path / "first", ELF)
    stored = next(p for p in store.rglob(f"files/{ELF}"))
    stored.write_text("tampered")

    second = tmp_path / "second"
    assert BC.reuse(second, key, ELF) is None
    assert not (second / ELF).exists(), "a partial restore must not look like a build"


def test_a_truncated_entry_is_a_miss(key_of, tmp_path, store):
    key = key_of()
    _build(tmp_path / "first")
    _store(key, tmp_path / "first", ELF)
    next(p for p in store.rglob("files/kernel.ll")).unlink()
    assert BC.reuse(tmp_path / "second", key, ELF) is None


def test_a_record_from_another_version_is_not_read(key_of, tmp_path, store):
    key = key_of()
    _build(tmp_path / "first")
    _store(key, tmp_path / "first", ELF)
    rec = next(store.rglob("record.json"))
    payload = json.loads(rec.read_text())
    payload["version"] = BC.RECORD_VERSION + 1
    rec.write_text(json.dumps(payload))
    assert BC.reuse(tmp_path / "second", key, ELF) is None


def test_no_key_never_hits_and_never_writes(tmp_path, store):
    _build(tmp_path / "w")
    _store(None, tmp_path / "w", ELF)
    assert BC.reuse(tmp_path / "w2", None, ELF) is None
    assert not store.exists() or not any(store.rglob("record.json"))


# --------------------------------------------------------------------------------------------
# THE MARKER -- one ladder builds once, and a marker never speaks for a previous grade
# --------------------------------------------------------------------------------------------


def test_the_marker_makes_a_second_tier_free(key_of, tmp_path, store, monkeypatch):
    key, work = key_of(), tmp_path / "w"
    _build(work)
    _store(key, work, ELF)
    assert (work / BC.KEY_MARKER).read_text() == key
    # With the store made unreachable, only the marker can answer -- so a hit here proves the ladder's
    # second tier never touches it.
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE", str(tmp_path / "elsewhere"))
    assert BC.reuse(work, key, ELF) == work / ELF


def test_the_marker_does_not_answer_for_a_different_key(key_of, tmp_path, monkeypatch):
    work = tmp_path / "w"
    _build(work)
    _store(key_of(), work, ELF)
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE", str(tmp_path / "elsewhere"))
    assert BC.reuse(work, key_of(lowered_mlir_text="other"), ELF) is None


def test_forget_drops_the_claim(key_of, tmp_path, monkeypatch):
    work = tmp_path / "w"
    _build(work)
    _store(key_of(), work, ELF)
    BC.forget(work)
    assert not (work / BC.KEY_MARKER).exists()
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE", str(tmp_path / "elsewhere"))
    assert BC.reuse(work, key_of(), ELF) is None
    BC.forget(work)  # idempotent: a directory with no marker is not an error


def test_the_whole_directory_is_stored_not_a_diff(key_of, tmp_path):
    """A restore must reproduce the directory, including what the emit step left there before the
    build. A stat diff could miss a rewrite landing on the same size inside one mtime tick, which
    would restore a stale file beside a fresh executable; storing everything cannot err that way."""
    work = tmp_path / "w"
    work.mkdir()
    (work / "command_buffer.json").write_text('{"commands": []}')  # written by the emit step
    _build(work)
    assert set(BC.contents(work)) == {"command_buffer.json", "kernel.ll", "harness.c", ELF}
    _store(key_of(), work, ELF)
    second = tmp_path / "second"
    assert BC.reuse(second, key_of(), ELF) is not None
    assert (second / "command_buffer.json").read_text() == '{"commands": []}'


def test_contents_never_stores_the_key_marker(key_of, tmp_path):
    """The marker names the key its directory holds; storing it would ship one key inside another."""
    work = tmp_path / "w"
    _build(work)
    _store(key_of(), work, ELF)
    assert (work / BC.KEY_MARKER).is_file()
    assert BC.KEY_MARKER not in BC.contents(work)
    second = tmp_path / "second"
    BC.reuse(second, key_of(), ELF)
    assert (second / BC.KEY_MARKER).read_text() == key_of()


def test_a_restore_skips_what_is_already_in_place(key_of, tmp_path, monkeypatch):
    """On a hit the emit step's output is already in the directory. Re-copying it is the whole reason
    a restore was slower than it needed to be, so an identical file must be left alone -- and a file
    that DIFFERS must still be replaced, or a restore would leave a stale byte stream behind."""
    work = tmp_path / "w"
    work.mkdir()
    (work / "lowered.llvm.mlir").write_text("module { /* emitted */ }")
    _build(work)
    _store(key_of(), work, ELF)

    second = tmp_path / "second"
    second.mkdir()
    (second / "lowered.llvm.mlir").write_text("module { /* emitted */ }")  # identical: skip
    (second / "kernel.ll").write_text("; STALE, wrong length")  # differs: replace
    copied = []
    real = BC.shutil.copyfile
    monkeypatch.setattr(BC.shutil, "copyfile", lambda s, d: (copied.append(Path(d).name), real(s, d))[1])
    assert BC.reuse(second, key_of(), ELF) is not None
    assert "lowered.llvm.mlir" not in copied, "an identical file was copied anyway"
    assert "kernel.ll" in copied and (second / "kernel.ll").read_text() == "; ir"
    assert (second / ELF).read_text() == "elf-bytes"


def test_a_same_length_but_different_file_is_replaced(key_of, tmp_path):
    """Size is only the cheap negative; the digest decides. Skipping on length alone would restore a
    directory whose IR does not match its executable."""
    work = tmp_path / "w"
    work.mkdir()
    (work / "kernel.ll").write_text("; ir")
    _build(work)
    _store(key_of(), work, ELF)
    second = tmp_path / "second"
    second.mkdir()
    (second / "kernel.ll").write_text("; XX")  # same length, different bytes
    assert BC.reuse(second, key_of(), ELF) is not None
    assert (second / "kernel.ll").read_text() == "; ir"


def test_contents_does_not_read_file_contents(tmp_path, monkeypatch):
    """Walked on every miss, before compiling. It must not re-read what is already on disk."""
    work = tmp_path / "w"
    work.mkdir()
    (work / "big.ll").write_text("x" * 4096)
    monkeypatch.setattr(BC, "_file_sha", lambda p: pytest.fail("contents read a file's contents"))
    assert "big.ll" in BC.contents(work)


def test_the_build_path_digest_is_memoized_but_not_stale(tmp_path):
    """Memoized on the covering files' stat signature, so an edit misses rather than being missed."""
    src = tmp_path / "builder.py"
    src.write_text("# v1")
    first = BC._build_path_digest((src,))
    assert BC._build_path_digest((src,)) == first
    src.write_text("# v2 -- edited")
    assert BC._build_path_digest((src,)) != first


def test_the_store_stays_bounded(key_of, tmp_path, store, monkeypatch):
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE_ENTRIES", "3")
    for i in range(6):
        work = tmp_path / f"w{i}"
        _build(work, text=f"elf-{i}")
        _store(key_of(lowered_mlir_text=f"module {i}"), work, ELF)
    entries = [d for shard in store.iterdir() if shard.is_dir() for d in shard.iterdir()]
    assert len(entries) <= 3


def test_interface_input_changes_build_identity(key_of):
    assert key_of(interface_mlir_text="module { // first }") != key_of(interface_mlir_text="module { // second }")
    assert key_of() != key_of(interface_mlir_text="")


def test_default_store_does_not_replace_caller_inputs_or_invocations(key_of, tmp_path):
    first, second = tmp_path / "donor", tmp_path / "receiver"
    _build(first)
    second.mkdir()
    for directory, content in ((first, "donor"), (second, "receiver")):
        (directory / "input.interface.mlir").write_text(content)
        (directory / "invocations").mkdir()
        (directory / "invocations/invocation.json").write_text(content)
    BC.store(key_of(), first, ELF)
    assert BC.reuse(second, key_of(), ELF) == second / ELF
    assert (second / "input.interface.mlir").read_text() == "receiver"
    assert (second / "invocations/invocation.json").read_text() == "receiver"


@pytest.mark.parametrize("product", ["input.interface.mlir", "invocations/invocation.json", "../outside"])
def test_store_refuses_nonproduct_members(key_of, tmp_path, product):
    first = tmp_path / "donor"
    _build(first)
    BC.store(key_of(), first, BC.BuildProducts(ELF, products=(product,)))
    assert BC.reuse(tmp_path / "receiver", key_of(), ELF) is None


def test_certification_attempts_have_separate_stores(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_ELF_BUILD_CACHE", str(tmp_path / "shared"))
    ambient = BC.cache_root()

    @BC.certification_cache
    def grade(*, runs_root):
        return BC.cache_root()

    first = grade(runs_root=tmp_path / "grade")
    second = grade(runs_root=tmp_path / "grade")
    assert first != second
    assert first.is_relative_to(tmp_path / "grade")
    assert second.is_relative_to(tmp_path / "grade")
    assert BC.cache_root() == ambient


def test_certification_rebuilds_previous_attempt_in_same_workdir(tmp_path, store):
    work = tmp_path / "generated"
    _build(work, "ambient")
    BC.store("key", work, ELF)
    builds = []

    @BC.certification_cache
    def grade(*, runs_root):
        assert BC.reuse(work, "key", ELF) is None
        builds.append(len(builds))
        _build(work, str(builds[-1]))
        BC.store("key", work, ELF)
        assert BC.reuse(work, "key", ELF) == work / ELF

    grade(runs_root=tmp_path / "grade")
    grade(runs_root=tmp_path / "grade")
    assert builds == [0, 1]
    assert (work / ELF).read_text() == "1"


@pytest.mark.parametrize("fails", [False, True])
def test_certification_removes_attempt_store_on_exit(tmp_path, store, fails):
    roots = []

    @BC.certification_cache
    def grade(*, runs_root):
        roots.append(BC.cache_root())
        work = tmp_path / "generated"
        _build(work)
        BC.store("key", work, ELF)
        assert list(roots[-1].rglob("record.json"))
        if fails:
            raise RuntimeError("failed attempt")

    if fails:
        with pytest.raises(RuntimeError, match="failed attempt"):
            grade(runs_root=tmp_path / "grade")
    else:
        grade(runs_root=tmp_path / "grade")
    assert not roots[0].exists()
    assert BC.cache_root() == store
    assert not list((tmp_path / "grade" / ".elf-build-cache").iterdir())


def test_same_named_support_sources_both_change_identity(recipe, key_of, tmp_path):
    first, second = tmp_path / "one/crt.S", tmp_path / "two/crt.S"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first")
    second.write_text("second")
    recipe.support_sources = (first, second)
    before = key_of()
    first.write_text("changed")
    assert key_of() != before


def test_compile_rebuilds_when_only_interface_changes(recipe, key_of, tmp_path, monkeypatch):
    from merlin.runtime.backends import base
    from merlin.targetgen.contract import compile as compiler

    recipe.with_effective_abi = lambda: recipe
    monkeypatch.setattr(base, "harness_build_recipe", lambda target: recipe)
    builds = []

    def object_file(text, work, **kwargs):
        builds.append(text)
        (work / "kernel.o").write_text(text)
        return work / "kernel.o"

    def link(cb, obj, work, **kwargs):
        return _build(work)

    monkeypatch.setattr(compiler, "llvm_mlir_to_object", object_file)
    monkeypatch.setattr(compiler, "link_elf", link)
    work = tmp_path / "compile"
    work.mkdir()
    interface = work / "input.interface.mlir"
    interface.write_text("first")
    compiler.compile_lowered_to_elf({}, "module {}", work, target="fixture", inputs={})
    compiler.compile_lowered_to_elf({}, "module {}", work, target="fixture", inputs={})
    assert len(builds) == 1
    interface.write_text("other")
    compiler.compile_lowered_to_elf({}, "module {}", work, target="fixture", inputs={})
    assert len(builds) == 2
    assert interface.read_text() == "other"
