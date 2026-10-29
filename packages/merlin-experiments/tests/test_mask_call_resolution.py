"""Mask planning keeps fresh aliases while sharing one call's path prefixes."""
from pathlib import Path

import pytest

from merlin.targetgen.sandbox import bwrap as B
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface


def test_resolution_matches_normal_paths_links_missing_and_parent_steps(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "child").mkdir()
    (actual / "child/data").write_text("payload")
    (tmp_path / "relative").symlink_to("actual/child")
    (tmp_path / "absolute").symlink_to(actual)
    resolver = B._CallPathResolution()
    for path in (actual, actual / "child/data", tmp_path / "relative/data",
                 tmp_path / "relative/../child/data", tmp_path / "absolute/child",
                 tmp_path / "absent/child", tmp_path / "absent/../actual",
                 actual / "child/data/not-a-directory"):
        assert resolver.resolve(path) == path.resolve()
    (tmp_path / "loop").symlink_to("loop")
    with pytest.raises(RuntimeError):
        resolver.resolve(tmp_path / "loop/child")


def test_mask_calls_observe_retargeted_alias_and_later_creation(tmp_path):
    private = tmp_path / "private"
    public = tmp_path / "public"
    private.mkdir()
    public.mkdir()
    (private / "answer").write_text("secret")
    (public / "input").write_text("operand")
    alias = tmp_path / "alias"
    alias.symlink_to(public)
    guest = tmp_path / "guest"
    argv = ["--ro-bind", str(alias), str(guest)]
    surfaces = [AnswerSurface("secret", private, "dir", "hidden")]
    assert B.is_exposed(B.apply_answer_masks(argv, surfaces), guest / "input")
    alias.unlink()
    alias.symlink_to(private)
    assert B.coverage_gap(argv, surfaces)
    masked = B.apply_answer_masks(argv, surfaces)
    assert not B.is_exposed(masked, guest / "answer")
    assert B.coverage_gap(masked, surfaces) == []
    late = private / "late"
    leaves = [AnswerSurface("late", late, "file", "golden")]
    direct = ["--ro-bind", str(private), str(private)]
    B.apply_answer_masks(direct, leaves)
    late.write_text("new secret")
    assert not B.is_exposed(B.apply_answer_masks(direct, leaves), late)


def test_masks_match_independent_resolution_with_many_shared_prefixes(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    surfaces = []
    for index in range(32):
        path = private / f"case{index}" / "answer"
        path.parent.mkdir()
        path.write_text(str(index))
        surfaces.append(AnswerSurface(str(index), path, "file", "golden"))
    argv = ["--ro-bind", str(private), str(tmp_path / "guest")]
    shared = B.apply_answer_masks(argv, surfaces)
    monkeypatch.setattr(B._CallPathResolution, "resolve", lambda self, path: Path(path).resolve())
    assert B.apply_answer_masks(argv, surfaces) == shared
    assert B.coverage_gap(shared, surfaces) == []


def test_host_location_check_reopens_live_alias_each_call(tmp_path):
    workspace = tmp_path / "workspace"
    private = tmp_path / "private"
    workspace.mkdir()
    private.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(private)
    hosts = [("secret", alias / "answer")]
    B._validate_host_locations(workspace, hosts)
    alias.unlink()
    alias.symlink_to(workspace)
    with pytest.raises(RuntimeError, match="host-only input overlaps"):
        B._validate_host_locations(workspace, hosts)
