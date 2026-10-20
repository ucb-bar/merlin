"""Per-call privacy projection reuse preserves aliases and ordered masks."""

from __future__ import annotations

from pathlib import Path

import pytest

from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import host_surfaces as HS
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface


def _project(kind, argv, paths):
    if kind == "host":
        return HS._private_path_surfaces(argv, [(str(path), path, path) for path in paths], set())
    surfaces = [AnswerSurface("private fixture", path, "file", "hidden") for path in paths]
    return BW._support_bind_surfaces(BW._mounts(argv), surfaces)


@pytest.mark.parametrize("kind", ["host", "support"])
def test_mount_source_resolution_is_once_per_unique_source(tmp_path, monkeypatch, kind):
    root = tmp_path / "source"
    root.mkdir()
    paths = [root / f"private-{index}" for index in range(256)]
    for path in paths:
        path.write_text("private")
    lexical = tmp_path / "lexical"
    lexical.symlink_to(root, target_is_directory=True)
    argv = ["--ro-bind", str(lexical), str(tmp_path / "first"),
            "--tmpfs", str(tmp_path / "first"),
            "--ro-bind", str(lexical), str(tmp_path / "second")]
    resolves = []
    parses = []
    original_resolve, original_mounts = Path.resolve, HS._mounts

    def resolve(path, *args, **kwargs):
        if path == lexical:
            resolves.append(path)
        return original_resolve(path, *args, **kwargs)

    def mounts(arguments):
        parses.append(arguments)
        return original_mounts(arguments)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(HS, "_mounts", mounts)
    actual = _project(kind, argv, paths)
    assert resolves == [lexical]
    if kind == "host":
        assert parses == [argv]
    assert {tmp_path / alias / path.name for alias in ("first", "second") for path in paths} <= {
        surface.path for surface in actual
    }


@pytest.mark.parametrize("kind", ["host", "support"])
def test_alias_mutation_between_calls_is_observed(tmp_path, kind):
    roots = [tmp_path / label for label in ("first", "second")]
    paths = []
    for root in roots:
        root.mkdir()
        path = root / "secret"
        path.write_text("private")
        paths.append(path)
    lexical, destination = tmp_path / "lexical", tmp_path / "view"
    lexical.symlink_to(roots[0], target_is_directory=True)
    argv = ["--ro-bind", str(lexical), str(destination)]
    assert destination / "secret" in {surface.path for surface in _project(kind, argv, [paths[0]])}
    lexical.unlink()
    lexical.symlink_to(roots[1], target_is_directory=True)
    assert destination / "secret" not in {surface.path for surface in _project(kind, argv, [paths[0]])}
    assert destination / "secret" in {surface.path for surface in _project(kind, argv, [paths[1]])}


@pytest.mark.parametrize("kind", ["host", "support"])
def test_all_destinations_and_late_reexposures_remain_masked(tmp_path, kind):
    root = tmp_path / "source"
    root.mkdir()
    private = root / "secret"
    private.write_text("private")
    lexical = tmp_path / "lexical"
    lexical.symlink_to(root, target_is_directory=True)
    first, second, direct = (tmp_path / label for label in ("first", "second", "direct"))
    argv = ["--ro-bind", str(lexical), str(first), "--tmpfs", str(first),
            "--ro-bind", str(root), str(first), "--ro-bind", str(lexical), str(second),
            "--ro-bind", str(private), str(direct)]
    actual = _project(kind, argv, [private])
    aliases = {first / "secret", second / "secret", direct}
    assert aliases <= {surface.path for surface in actual}
    assert all(BW.is_exposed(argv, alias) for alias in aliases)
    masked = BW.apply_answer_masks(argv, actual)
    assert masked[:len(argv)] == argv
    assert all(not BW.is_exposed(masked, alias) for alias in aliases)
    assert BW.coverage_gap(masked, actual) == []


def test_support_public_exemption_and_original_query_order_are_preserved(tmp_path):
    root = tmp_path / "source"
    public = root / "contracts"
    public.mkdir(parents=True)
    (public / "spec").write_text("public")
    (root / "secret").write_text("private")
    surface = AnswerSurface("private package", root, "dir", "support", ("contracts",))
    view, public_view = tmp_path / "view", tmp_path / "public-view"
    argv = ["--ro-bind", str(root), str(view), "--ro-bind", str(public), str(public_view)]
    actual = BW._support_bind_surfaces(BW._mounts(argv), [surface, surface])
    assert actual[:2] == [surface, surface]
    assert view in {item.path for item in actual}
    assert public_view not in {item.path for item in actual}
    masked = BW.apply_answer_masks(argv, actual)
    assert not BW.is_exposed(masked, view / "secret")
    assert BW.is_exposed(masked, public_view / "spec")


def test_shared_parse_preserves_strict_hardlink_checks(tmp_path, monkeypatch):
    private = tmp_path / "secret"
    private.write_text("private")
    public = tmp_path / "runtime"
    public.mkdir()
    (public / "alias").hardlink_to(private)
    destination = tmp_path / "view"
    argv = ["--ro-bind", str(public), str(destination)]
    original_mounts, parses = HS._mounts, []

    def mounts(arguments):
        parses.append(arguments)
        return original_mounts(arguments)

    monkeypatch.setattr(HS, "_mounts", mounts)
    actual = HS._private_path_surfaces(argv, [(str(private), private, private)], set())
    assert parses == [argv]
    assert destination / "alias" in {surface.path for surface in actual}
    assert not BW.is_exposed(BW.apply_answer_masks(argv, actual), destination / "alias")
