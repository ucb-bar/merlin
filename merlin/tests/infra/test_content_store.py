"""The store itself: an ignore rule is honored exactly, and two frozen trees share their bytes.

``merlin.common.content_store`` replaced a deep copy in two places -- an agent run's declared input
closure and a performance suite's source snapshot. Both had a reason to copy (the frozen bytes must
not follow a later in-place edit of the source) and no reason to copy *per consumer*.
"""

from __future__ import annotations

import errno
import importlib
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from merlin.common import content_store as CS
from merlin.common.paths import merlin_dir


def _tree(root: Path) -> Path:
    (root / "keep").mkdir(parents=True)
    (root / "keep" / "wanted.txt").write_text("wanted\n", encoding="utf-8")
    (root / "top.txt").write_text("top\n", encoding="utf-8")
    return root


def test_two_frozen_trees_of_the_same_source_share_their_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    source = _tree(tmp_path / "src")

    CS.place_tree(source, tmp_path / "first", CS.store_root())
    CS.place_tree(source, tmp_path / "second", CS.store_root())

    for relative in ("keep/wanted.txt", "top.txt"):
        first, second = tmp_path / "first" / relative, tmp_path / "second" / relative
        assert first.read_bytes() == second.read_bytes()
        assert first.stat().st_ino == second.stat().st_ino, f"{relative} was stored twice"


def test_disabling_the_store_still_freezes_the_same_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv(CS.LOCATION_ENV, "")
    source = _tree(tmp_path / "src")
    assert CS.store_root() is None

    CS.place_tree(source, tmp_path / "copied", CS.store_root())
    landed = tmp_path / "copied" / "top.txt"
    assert landed.read_text() == "top\n"
    assert landed.stat().st_nlink == 1


def test_permission_preserving_copy_refuses_cas_and_shared_destination(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"owner-only selected input\n")
    source.chmod(0o400)
    store = tmp_path / "store"
    for placer in (CS.place_file, CS.place_tree):
        with pytest.raises(ValueError, match="cannot use the shared content store"):
            placer(source, tmp_path / "absent", store, preserve_permissions=True)
    assert not store.exists() and not (tmp_path / "absent").exists()
    shared = tmp_path / "shared"
    other = tmp_path / "other"
    shared.write_bytes(b"unrelated retained bytes\n")
    shared.chmod(0o444)
    os.link(shared, other)
    with pytest.raises(ValueError, match="shared permission-preserving destination"):
        CS.place_file(source, shared, None, preserve_permissions=True)
    assert other.read_bytes() == b"unrelated retained bytes\n"
    assert other.stat().st_mode & 0o7777 == 0o444


@pytest.mark.parametrize("existing", ["directory", "symlink", "dangling_symlink"])
def test_permission_preserving_tree_refuses_existing_destination_before_copy(tmp_path, existing):
    source = _tree(tmp_path / "source")
    destination, retained = tmp_path / "destination", tmp_path / "retained"
    if existing == "directory":
        destination.mkdir()
    else:
        if existing == "symlink":
            retained.mkdir()
        destination.symlink_to(retained, target_is_directory=True)
    with pytest.raises(ValueError, match="tree destination must be new"):
        CS.place_tree(source, destination, None, preserve_permissions=True)
    assert not (retained / "top.txt").exists() and not (destination / "top.txt").exists()
    assert destination.is_symlink() == (existing != "directory")


@pytest.mark.parametrize("executable", [False, True])
@pytest.mark.parametrize("fallback", ["disabled", "unavailable", "failed_link"])
def test_copy_fallback_retains_typed_readonly_mode_and_original_inputs(tmp_path, monkeypatch, executable, fallback):
    source = tmp_path / "source"
    source.write_bytes(b"unchanged frozen input bytes\n")
    original_mode = 0o750 if executable else 0o664
    source.chmod(original_mode)
    original = source.read_bytes(), source.stat().st_mode & 0o7777, source.stat().st_ino
    store = None if fallback == "disabled" else tmp_path / "store"
    destination = tmp_path / "consumer"
    if fallback == "unavailable":
        store.write_bytes(b"not a directory")
        store.chmod(0o640)
    elif fallback == "failed_link":
        native_link = os.link

        def fail_only_consumer(src, dst, *args, **kwargs):
            if Path(dst) == destination:
                raise OSError(errno.EOPNOTSUPP, "independent fixture: destination links unsupported")
            return native_link(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "link", fail_only_consumer)
    assert not CS.place_file(source, destination, store)
    expected_mode = 0o555 if executable else 0o444
    assert destination.read_bytes() == original[0]
    assert destination.stat().st_mode & 0o7777 == expected_mode
    assert destination.stat().st_nlink == 1
    assert destination.stat().st_ino != source.stat().st_ino
    assert (source.read_bytes(), source.stat().st_mode & 0o7777, source.stat().st_ino) == original
    if fallback == "failed_link":
        obj = CS.object_for(store, source)
        assert obj.read_bytes() == original[0] and obj.stat().st_mode & 0o7777 == expected_mode
        assert not destination.samefile(obj)
    elif fallback == "unavailable":
        assert store.read_bytes() == b"not a directory" and store.stat().st_mode & 0o7777 == 0o640


@pytest.mark.parametrize("executable", [False, True])
def test_native_cross_filesystem_copy_freezes_mode_without_changing_source_or_cas(tmp_path, executable):
    """Opt-in native EXDEV control; release qualifiers supply an owned other volume.

    Ordinary CI may have only one writable filesystem. The qualifier must set
    MERLIN_TEST_CONTENT_STORE_EXDEV_ROOT to its owned source/CAS directory and
    keep pytest's tmp_path on a different volume; a configured same-volume
    root fails rather than converting this proof into a synthetic link error.
    """
    selected = os.environ.get("MERLIN_TEST_CONTENT_STORE_EXDEV_ROOT")
    if not selected:
        pytest.skip("native EXDEV gate requires an explicit owned source/CAS test root")
    selected_root = Path(selected)
    selected_root.mkdir(parents=True, exist_ok=True)
    assert selected_root.stat().st_dev != tmp_path.stat().st_dev
    with tempfile.TemporaryDirectory(dir=selected_root, prefix="native-exdev-") as source_dir:
        source = Path(source_dir) / "source"
        source.write_bytes(b"native cross-filesystem frozen input\n")
        original_mode = 0o750 if executable else 0o664
        source.chmod(original_mode)
        store = Path(source_dir) / "cas"
        obj = CS.object_for(store, source)
        expected_mode = 0o555 if executable else 0o444
        assert obj is not None and obj.stat().st_mode & 0o7777 == expected_mode
        before_source = source.read_bytes(), source.stat().st_mode & 0o7777, source.stat().st_ino
        before_cas = obj.read_bytes(), obj.stat().st_mode & 0o7777, obj.stat().st_ino
        with pytest.raises(OSError) as native:
            os.link(obj, tmp_path / "native-link-probe")
        assert native.value.errno == errno.EXDEV
        destination = tmp_path / "consumer"
        assert not CS.place_file(source, destination, store)
        assert destination.stat().st_dev != obj.stat().st_dev
        assert destination.read_bytes() == before_source[0]
        assert destination.stat().st_mode & 0o7777 == expected_mode
        assert destination.stat().st_nlink == 1
        assert (source.read_bytes(), source.stat().st_mode & 0o7777, source.stat().st_ino) == before_source
        assert (obj.read_bytes(), obj.stat().st_mode & 0o7777, obj.stat().st_ino) == before_cas


@pytest.mark.parametrize("executable", [False, True])
def test_disabled_store_tree_freezes_dereferenced_file_modes(tmp_path, executable):
    source = tmp_path / "source"
    source.mkdir()
    actual = source / "data"
    actual.write_bytes(b"selected input\n")
    original_mode = 0o750 if executable else 0o664
    actual.chmod(original_mode)
    (source / "alias").symlink_to("data")
    CS.place_tree(source, tmp_path / "consumer", None)
    for name in ("data", "alias"):
        landed = tmp_path / "consumer" / name
        assert not landed.is_symlink() and landed.read_bytes() == actual.read_bytes()
        assert landed.stat().st_mode & 0o7777 == (0o555 if executable else 0o444)
    assert actual.stat().st_mode & 0o7777 == original_mode
    assert (source / "alias").is_symlink()


def test_copy_fallback_keeps_caller_owned_regular_overwrite_behavior(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "consumer"
    source.write_bytes(b"new source")
    source.chmod(0o664)
    destination.write_bytes(b"old consumer")
    inode = destination.stat().st_ino
    assert not CS.place_file(source, destination, None)
    assert destination.read_bytes() == b"new source" and destination.stat().st_mode & 0o7777 == 0o444
    assert destination.stat().st_ino == inode
    assert source.stat().st_mode & 0o7777 == 0o664


@pytest.mark.parametrize("executable", [False, True])
def test_failed_copy_never_changes_destination_mode(tmp_path, monkeypatch, executable):
    source, destination = tmp_path / "source", tmp_path / "consumer"
    source.write_bytes(b"complete source")
    source.chmod(0o750 if executable else 0o664)
    before_source = source.read_bytes(), source.stat().st_mode & 0o7777
    destination.write_bytes(b"caller-owned destination")
    destination.chmod(0o600)

    def interrupted_copy(src, dst, **kwargs):
        Path(dst).write_bytes(b"partial copied bytes")
        raise OSError("independent fixture: interrupted copy")

    monkeypatch.setattr(CS.shutil, "copy2", interrupted_copy)
    with pytest.raises(OSError, match="interrupted copy"):
        CS.place_file(source, destination, None)
    assert destination.read_bytes() == b"partial copied bytes"
    assert destination.stat().st_mode & 0o7777 == 0o600
    assert (source.read_bytes(), source.stat().st_mode & 0o7777) == before_source


@pytest.mark.parametrize("store_enabled", [False, True])
@pytest.mark.parametrize("dangling", [False, True])
def test_destination_symlink_refuses_before_mutating_foreign_target_or_creating_cas(tmp_path, store_enabled, dangling):
    source, foreign = tmp_path / "source", tmp_path / "foreign"
    source.write_bytes(b"source bytes")
    source.chmod(0o664)
    if not dangling:
        foreign.write_bytes(b"foreign bytes")
        foreign.chmod(0o600)
    destination = tmp_path / "consumer"
    destination.symlink_to(foreign)
    store = tmp_path / "store" if store_enabled else None
    with pytest.raises(ValueError, match="destination.*symlink"):
        CS.place_file(source, destination, store)
    assert destination.is_symlink() and source.read_bytes() == b"source bytes"
    assert source.stat().st_mode & 0o7777 == 0o664
    if dangling:
        assert not foreign.exists()
    else:
        assert foreign.read_bytes() == b"foreign bytes" and foreign.stat().st_mode & 0o7777 == 0o600
    if store is not None:
        assert not store.exists()


@pytest.mark.parametrize("executable", [False, True])
def test_cache_reuse_restores_immutable_modes_for_every_link(tmp_path, executable):
    source = tmp_path / "source"
    source.write_bytes(b"unchanged input bytes")
    source.chmod(0o755 if executable else 0o644)
    store = tmp_path / "store"
    first, second = tmp_path / "first", tmp_path / "second"
    assert CS.place_file(source, first, store)
    expected_mode = 0o555 if executable else 0o444
    assert first.stat().st_mode & 0o777 == expected_mode

    # TemporaryDirectory cleanup can chmod a hardlink after an unlink failure.
    # Reusing the validated object must not hand another consumer a writable inode.
    first.chmod(0o700)
    assert CS.place_file(source, second, store)
    assert first.samefile(second)
    assert first.read_bytes() == second.read_bytes() == source.read_bytes()
    assert first.stat().st_mode & 0o777 == expected_mode
    assert second.stat().st_mode & 0o777 == expected_mode
    assert source.stat().st_mode & 0o777 == (0o755 if executable else 0o644)


def test_cache_mode_repair_never_chmods_a_symlink_target(tmp_path):
    source, foreign = tmp_path / "source", tmp_path / "foreign"
    source.write_bytes(b"same bytes")
    foreign.write_bytes(source.read_bytes())
    foreign.chmod(0o600)
    digest, _ = CS.digest_file(source)
    store = tmp_path / "store"
    obj = store / digest[:2] / digest
    obj.parent.mkdir(parents=True)
    obj.symlink_to(foreign)

    destination = tmp_path / "copy"
    assert not CS.place_file(source, destination, store)
    assert destination.read_bytes() == source.read_bytes()
    assert not destination.is_symlink()
    assert foreign.stat().st_mode & 0o777 == 0o600
    assert obj.is_symlink()


def test_observer_tracks_each_invocation_even_when_bytes_hit_cache(tmp_path):
    store = tmp_path / "store"
    public, private = tmp_path / "public", tmp_path / "private"
    public.mkdir()
    private.mkdir()
    for root in (public, private):
        (root / "data").write_bytes(b"identical bytes")
    alias = tmp_path / "alias"
    alias.symlink_to(public, target_is_directory=True)
    observations = []
    CS.place_tree(alias, tmp_path / "first", store, observe=lambda *row: observations.append(row))
    alias.unlink()
    alias.symlink_to(private, target_is_directory=True)
    CS.place_tree(alias, tmp_path / "second", store, observe=lambda *row: observations.append(row))
    assert (alias / "data", public / "data", tmp_path / "first/data") in observations
    assert (alias / "data", private / "data", tmp_path / "second/data") in observations
    assert (tmp_path / "first/data").stat().st_ino == (tmp_path / "second/data").stat().st_ino


def test_file_observer_preserves_symlink_before_parent_resolution(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    actual = tmp_path / "actual"
    (actual / "inner").mkdir(parents=True)
    (actual / "data").write_text("selected bytes")
    (source / "data").write_text("wrong lexical normalization")
    (source / "pivot").symlink_to(actual / "inner", target_is_directory=True)
    link = source / "link"
    link.symlink_to("pivot/../data")
    observations = []
    CS.place_file(link, tmp_path / "copy", None, observe=lambda *row: observations.append(row))
    assert (tmp_path / "copy").read_text() == "selected bytes"
    assert all(canonical == actual / "data" for _, canonical, _ in observations)
    assert any(lexical == source / "pivot" for lexical, _, _ in observations)


def test_observer_refuses_alias_substitution_during_copy(tmp_path):
    original, replacement = tmp_path / "original", tmp_path / "replacement"
    original.write_text("original")
    replacement.write_text("replacement")
    alias = tmp_path / "alias"
    alias.symlink_to(original)

    def changed(lexical, canonical, destination):
        if lexical == alias:
            alias.unlink()
            alias.symlink_to(replacement)

    with pytest.raises(RuntimeError, match="ownership changed"):
        CS.place_file(alias, tmp_path / "copy", None, observe=changed)
    assert not (tmp_path / "copy").exists()


def test_a_freeze_that_verifies_by_FILE_MODE_must_not_use_the_store(tmp_path, monkeypatch):
    """The boundary of this mechanism, kept as a test because it is not obvious.

    A store object's mode belongs to its inode, so it is shared by every consumer linking it and any
    one of them can change it for all the others -- including the chmod a caller has to perform to
    delete its own frozen tree. That is harmless for a freeze verified by CONTENT (the sandbox's
    bundle closure re-digests the bytes), and unsound for one that treats "this file is not writable"
    as evidence of its own integrity: the perf-bench source snapshot does exactly that, and adopting
    the store there made a second snapshot fail verification because an unrelated first one had been
    chmodded during its teardown. Reproduced here so the saving does not tempt someone to weaken the
    check instead.
    """
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    source = _tree(tmp_path / "src")
    CS.place_tree(source, tmp_path / "first", CS.store_root())
    CS.place_tree(source, tmp_path / "second", CS.store_root())

    first = tmp_path / "first" / "top.txt"
    second = tmp_path / "second" / "top.txt"
    assert first.stat().st_ino == second.stat().st_ino
    first.chmod(0o600)  # what a caller does to remove its own frozen tree
    assert second.stat().st_mode & 0o222, "a mode change reached the other consumer -- as documented"
    assert second.read_text() == "top\n", "the BYTES, which is what the store actually promises"


def test_the_perf_bench_snapshot_still_owns_its_own_bytes(tmp_path, monkeypatch):
    """Consequence of the above: that snapshot deliberately still deep-copies. Pinned so the
    mechanism is not quietly reintroduced there."""
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    scripts = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    snapshot = importlib.import_module("merlin_experiments.source_snapshot")

    source = tmp_path / "repo"
    _tree(source / "merlin" / "python")
    output_root = tmp_path / "out"
    output_root.mkdir()

    try:
        for name in ("run1", "run2"):
            snapshot.create(
                source,
                tmp_path / name,
                output_root=output_root,
                source_roots=("merlin/python",),
                python_roots=("merlin/python",),
                legacy_roots=(),
            )
            assert snapshot.verify(tmp_path / name)["schema"] == snapshot.SCHEMA

        first = tmp_path / "run1" / "merlin/python/keep/wanted.txt"
        second = tmp_path / "run2" / "merlin/python/keep/wanted.txt"
        assert first.read_text() == "wanted\n"
        assert first.stat().st_ino != second.stat().st_ino
        assert first.stat().st_nlink == 1, "the snapshot must own its mode, so it owns its inode"
    finally:
        # A finished snapshot is mode 0500 all the way down, so nothing -- including pytest's own
        # temp reclaim -- can remove it until the write bits come back. Leaving it is how a suite
        # accumulates unreclaimable trees, which is the thing this whole change is about.
        for name in ("run1", "run2"):
            root = tmp_path / name
            if not root.exists():
                continue
            root.chmod(0o700)
            for path in sorted(root.rglob("*"), key=lambda p: len(p.parts)):
                if not path.is_symlink():
                    path.chmod(0o700 if path.is_dir() else 0o600)


def test_a_source_edit_after_the_freeze_does_not_reach_either_snapshot(tmp_path, monkeypatch):
    """The property both callers copy for. A link to the SOURCE would lose it; a link to a store
    object does not, because the store holds its own copy."""
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    source = _tree(tmp_path / "src")
    CS.place_tree(source, tmp_path / "frozen", CS.store_root())

    (source / "top.txt").write_bytes(b"edited in place")

    assert (tmp_path / "frozen" / "top.txt").read_text() == "top\n"


def test_a_non_regular_file_is_refused_rather_than_silently_dropped(tmp_path, monkeypatch):
    # The store is redirected here for the same reason as in every other test in this file, and this
    # one went without it: `store_root()` with no override is the LIVE store that ~13 sessions share,
    # so the two files this places before it reaches the fifo were landing in it for real.
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    source = _tree(tmp_path / "src")
    os.mkfifo(source / "pipe")
    with pytest.raises(RuntimeError, match="not a regular file or directory"):
        CS.place_tree(source, tmp_path / "out", CS.store_root())


def test_two_links_to_one_shared_directory_are_not_a_cycle(tmp_path, monkeypatch):
    """A tree may legitimately declare the same corpus twice. Only a directory that contains
    ITSELF makes the walk unbounded, and that is what must be refused."""
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    source = _tree(tmp_path / "src")
    (source / "shared").mkdir()
    (source / "shared" / "corpus.txt").write_text("corpus\n", encoding="utf-8")
    (source / "keep" / "as_a").symlink_to(source / "shared")
    (source / "keep" / "as_b").symlink_to(source / "shared")

    CS.place_tree(source, tmp_path / "out", CS.store_root())
    out = tmp_path / "out" / "keep"
    assert (out / "as_a" / "corpus.txt").read_text() == "corpus\n"
    assert (out / "as_b" / "corpus.txt").read_text() == "corpus\n"
    assert (out / "as_a" / "corpus.txt").stat().st_ino == (out / "as_b" / "corpus.txt").stat().st_ino

    (source / "shared" / "back").symlink_to(source / "keep")
    with pytest.raises(RuntimeError, match="symlink cycle"):
        CS.place_tree(source, tmp_path / "looped", CS.store_root())


def test_a_dangling_symlink_is_named_rather_than_dropped(tmp_path, monkeypatch):
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "store"))
    source = _tree(tmp_path / "src")
    (source / "keep" / "gone").symlink_to(source / "never_existed")

    with pytest.raises(RuntimeError, match="not a regular file or directory"):
        CS.place_tree(source, tmp_path / "out", CS.store_root())


def _place(job: tuple[str, str, str]) -> tuple[str, int, int]:
    """One worker's freeze, run in its own process (module level so it is picklable)."""
    source, dst, store = (Path(p) for p in job)
    CS.place_file(source, dst, store)
    stat = dst.stat()
    return dst.read_text(), stat.st_size, stat.st_ino


def test_runs_racing_on_the_same_content_all_get_the_same_correct_bytes(tmp_path, monkeypatch):
    """~13 sessions share this host, so two campaigns materializing the same grant at the same
    moment is the normal case, not the edge one. A half-written object must never be linked: the
    store stages under a name no other writer can hold and renames it into place, which is atomic.
    """
    store = tmp_path / "store"
    monkeypatch.setenv(CS.LOCATION_ENV, str(store))
    source = tmp_path / "grant.bin"
    source.write_text("x" * 200_000, encoding="utf-8")

    jobs = []
    for n in range(12):
        run = tmp_path / f"run{n}"
        run.mkdir()
        jobs.append((str(source), str(run / "grant.bin"), str(store)))

    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_place, jobs))

    assert {text for text, _, _ in results} == {"x" * 200_000}, "a worker saw partial bytes"
    assert {size for _, size, _ in results} == {200_000}
    # Whichever writer won the rename, every run ends up on ONE object -- that is the saving.
    assert len({ino for _, _, ino in results}) == 1
    assert not any(p.name.endswith(".pending") for p in store.rglob("*")), "staging file left behind"


def test_no_test_in_this_file_writes_to_the_live_store():
    """The store is shared by every session on this host, and a test that forgets to redirect it
    leaves objects in it for real -- which is how a 7-byte object whose bytes disagreed with its own
    name came to sit in the live store. Checked structurally, because the symptom is invisible: the
    test passes either way, and only the shared store is worse off."""
    import ast

    source = Path(__file__).read_text(encoding="utf-8")
    leaking = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        body = ast.dump(node)
        if ("store_root" in body or "default_root" in body) and "LOCATION_ENV" not in body:
            leaking.append(node.name)
    assert leaking == [], f"these resolve the live store without redirecting it: {leaking}"


def test_place_tree_skips_what_the_ignore_rule_names(tmp_path):
    import shutil

    source = tmp_path / "src"
    (source / "pkg" / "build").mkdir(parents=True)
    (source / "pkg" / "keep.py").write_text("kept\n")
    (source / "pkg" / "build" / "out.o").write_bytes(b"object")
    destination = tmp_path / "dst"
    CS.place_tree(source, destination, tmp_path / "cas", ignore=shutil.ignore_patterns("build"))
    assert (destination / "pkg" / "keep.py").read_text() == "kept\n"
    assert not (destination / "pkg" / "build").exists()


def test_a_store_namespace_is_a_separate_sibling_store(tmp_path, monkeypatch):
    monkeypatch.setenv(CS.LOCATION_ENV, str(tmp_path / "cas"))
    base, own = CS.store_root(), CS.store_root("compiler-source")
    assert base == tmp_path / "cas" and own == tmp_path / "cas.compiler-source" and own.is_dir()
    source = tmp_path / "file"
    source.write_bytes(b"identical bytes")
    CS.place_file(source, tmp_path / "frozen-input", base)
    CS.place_file(source, tmp_path / "capture", own)
    frozen, capture = (tmp_path / "frozen-input").stat(), (tmp_path / "capture").stat()
    assert frozen.st_ino != capture.st_ino and frozen.st_nlink == 2  # the capture never touched it
    with pytest.raises(ValueError):
        CS.store_root("../escape")
    monkeypatch.setenv(CS.LOCATION_ENV, "")
    assert CS.store_root("compiler-source") is None
