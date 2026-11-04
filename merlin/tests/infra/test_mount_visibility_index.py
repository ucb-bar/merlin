"""Mount replay equivalence and per-call indexing, without executing bwrap."""

import random
from pathlib import Path

import pytest

from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface


def surface(path, kind="file"):
    return AnswerSurface(label="test", path=Path(path), kind=kind, origin="test")


def replayed_exposure(argv, path):
    """Linear replay of bwrap's mount order, an independent oracle for the index.

    Each operation stacks over its destination, so the last one on the path or an ancestor serves it.
    """
    best = None
    for state, src, dest in BW._mounts(argv):
        base = Path(dest)
        if path == base or base in path.parents:
            best = (state, src, dest)
    if best is None or best[0] == "hide":
        return False
    _, src, dest = best
    mapped = Path(src) / path.relative_to(dest) if str(path) != dest else Path(src)
    try:
        return mapped.exists()
    except PermissionError:
        return True
    except OSError:
        return False


def test_coverage_parses_mounts_once_per_batch(tmp_path, monkeypatch):
    source = tmp_path / "data"
    source.write_text("test")
    argv = ["bwrap", "--ro-bind", str(tmp_path), "/visible"]
    original = BW._mounts
    calls = []

    def counted(args):
        calls.append(tuple(args))
        return original(args)

    monkeypatch.setattr(BW, "_mounts", counted)
    assert len(BW.coverage_gap(argv, [surface("/visible/data")] * 5)) == 5
    assert len(calls) == 1


def test_appending_masks_parses_once_and_preserves_incremental_hide(tmp_path, monkeypatch):
    (tmp_path / "data").write_text("test")
    argv = ["bwrap", "--ro-bind", str(tmp_path), "/visible"]
    original = BW._mounts
    calls = []

    def counted(args):
        calls.append(tuple(args))
        return original(args)

    monkeypatch.setattr(BW, "_mounts", counted)
    result = BW.apply_answer_masks(argv, [surface("/visible/data")] * 5)
    assert result == argv + ["--ro-bind", "/dev/null", "/visible/data"]
    assert len(calls) == 1


def test_latest_operation_wins_over_a_longer_spelling_of_the_same_destination(tmp_path):
    (tmp_path / "data").write_text("test")
    # Path('/visible/.') names the same directory as '/visible'; bwrap stacks the
    # later tmpfs over the earlier bind whatever the destination's spelling.
    argv = ["--ro-bind", str(tmp_path), "/visible/.", "--tmpfs", "/visible"]
    assert not replayed_exposure(argv, Path("/visible/data"))
    assert not BW.is_exposed(argv, Path("/visible/data"))
    tied = ["--tmpfs", "/visible", "--ro-bind", str(tmp_path), "/visible/."]
    assert BW.is_exposed(tied, Path("/visible/data"))


def test_snapshot_mapping_and_stat_state_are_not_cached_across_calls(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    argv = ["--ro-bind", str(source), "/live"]
    query = surface("/live/later")
    assert not BW.coverage_gap(argv, [query])
    (source / "later").write_text("created")
    assert BW.coverage_gap(argv, [query]) == [query]
    argv.extend(["--ro-bind", "/dev/null", "/live/later"])
    assert not BW.coverage_gap(argv, [query])


@pytest.mark.parametrize("exception, exposed", [(PermissionError, True), (OSError, False)])
def test_mapped_permission_error_semantics_preserved(monkeypatch, exception, exposed):
    def denied(path):
        raise exception("test")

    monkeypatch.setattr(Path, "exists", denied)
    assert BW.is_exposed(["--ro-bind", "/actual", "/visible"], Path("/visible/x")) is exposed


def test_ordered_and_aliased_mounts_match_linear_replay(tmp_path):
    (tmp_path / "data").write_text("test")
    (tmp_path / "child").mkdir()
    (tmp_path / "child" / "data").write_text("nested")
    randomizer = random.Random(4982)
    destinations = ["/v", "/v/", "/v/.", "/v/child", "/v/child/", "/v/../child", "//v", ".", "relative"]
    queries = [Path(name) for name in destinations] + [Path(name) / "data" for name in destinations]
    sources = [str(tmp_path), str(tmp_path / "child"), str(tmp_path / "missing"), "/dev/null"]
    for _ in range(35):
        argv = ["bwrap", "--unshare-pid"]
        for _ in range(12):
            destination = randomizer.choice(destinations)
            if randomizer.randrange(3):
                argv += [randomizer.choice(BW._EXPOSE_OPS), randomizer.choice(sources), destination]
            else:
                argv += [randomizer.choice(["--tmpfs", "--dev", "--proc"]), destination]
        rows = [surface(path) for path in queries]
        expected = [row for row in rows if replayed_exposure(argv, row.path)]
        assert BW.coverage_gap(argv, rows) == expected
        old_masked = list(argv)
        for row in rows:
            if replayed_exposure(old_masked, row.path):
                old_masked += ["--ro-bind", "/dev/null", str(row.path)]
        assert BW.apply_answer_masks(argv, rows) == old_masked
