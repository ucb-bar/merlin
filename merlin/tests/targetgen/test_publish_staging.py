"""Publication staging ownership only: no git, publication or network execution."""

from types import SimpleNamespace

import pytest

from merlin.common.yaml import write_yaml
from merlin.targetgen import publish as P


@pytest.fixture
def staging(tmp_path, monkeypatch):
    source = tmp_path / "artifacts" / "targets" / "fixture"
    source.mkdir(parents=True)
    (source / "sentinel").write_text("source")
    repo = tmp_path / "merlin"
    repo.mkdir()
    (repo / "LICENSE").write_text("license")
    monkeypatch.setattr(P.paths, "repo_root", lambda: repo)
    monkeypatch.setattr(P, "index_entries", lambda *a, **k: [])
    monkeypatch.setattr(P, "_index_readme", lambda *a: "index")
    return source


def assemble(kind, source, dest):
    if kind == "repo":
        return P.assemble_repo_tree(SimpleNamespace(package_dir=source), dest, layout_version=P.LAYOUT_VERSION)
    return P.assemble_index_tree("fixture", dest, artifacts_root=source.parent.parent)


@pytest.mark.parametrize("kind", ["repo", "index"])
@pytest.mark.parametrize(
    "case",
    [
        "existing",
        "empty",
        "file",
        "source",
        "ancestor",
        "descendant",
        "alias",
        "dangling",
        "parent_alias",
        "disjoint_alias",
    ],
)
def test_unsafe_destination_refuses_without_mutation(staging, tmp_path, kind, case):
    source = staging
    dest = tmp_path / "destination"
    if case in ("existing", "empty"):
        dest.mkdir()
        if case == "existing":
            (dest / "sentinel").write_text("destination")
    elif case == "file":
        dest.write_text("destination")
    elif case == "source":
        dest = source
    elif case == "ancestor":
        dest = source.parent
    elif case == "descendant":
        dest = source / "new" / "repo"
    elif case == "alias":
        dest.symlink_to(source, target_is_directory=True)
    elif case == "dangling":
        dest.symlink_to(tmp_path / "missing", target_is_directory=True)
    elif case == "parent_alias":
        dest.symlink_to(source, target_is_directory=True)
        dest = dest / "new" / "repo"
    else:
        other = tmp_path / "other"
        other.mkdir()
        dest.symlink_to(other, target_is_directory=True)
        dest = dest / "repo"
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    with pytest.raises(P.PublishError, match="staging destination"):
        assemble(kind, source, dest)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before
    assert (source / "sentinel").read_text() == "source"
    if case == "existing":
        assert (dest / "sentinel").read_text() == "destination"
    if case == "file":
        assert dest.read_text() == "destination"


def test_fresh_index_generation_and_repeat_refusal(staging, tmp_path):
    dest = tmp_path / "stage" / "repo"
    assemble("index", staging, dest)
    assert (dest / "README.md").read_text() == "index"
    assert (dest / "LICENSE").read_text() == "license"
    with pytest.raises(P.PublishError, match="fresh"):
        assemble("index", staging, dest)


def test_unique_stage_parents_preserve_same_timestamp(staging, tmp_path, monkeypatch):
    monkeypatch.setattr(P.paths, "build_dir", lambda: tmp_path / "build")
    monkeypatch.setattr(P, "utc_stamp", lambda: "same-time")
    first = P._new_stage_root("fixture", sources=(staging,))
    (first / "sentinel").write_text("retained")
    second = P._new_stage_root("fixture", sources=(staging,))
    assert first != second
    assert not (first / "repo").exists()
    assert not (second / "repo").exists()
    assert (first / "sentinel").read_text() == "retained"


def test_stage_parent_inside_source_refused_before_creation(staging, monkeypatch):
    monkeypatch.setattr(P.paths, "build_dir", lambda: staging / "new-build")
    with pytest.raises(P.PublishError, match="overlaps"):
        P._new_stage_root("fixture", sources=(staging,))
    assert not (staging / "new-build").exists()


def test_fresh_repo_generation_preserves_source(staging, tmp_path, monkeypatch, pytestconfig):
    monkeypatch.setattr(P.paths, "repo_root", lambda: pytestconfig.rootpath)
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": "fixture",
        "language": "python",
        "authoring": {"mode": "hand_curated"},
        "integrity_exempt": False,
        "entrypoints": {"tool": "fixture-opt"},
        "commands": {
            name: {"argv": ["{tool}", "{input_mlir}"]}
            for name in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
        },
    }
    write_yaml(staging / "manifest.yaml", manifest)
    tool = staging / "fixture-opt"
    tool.write_text("#!/usr/bin/env python3\nprint('fixture')\n")
    tool.chmod(0o755)
    monkeypatch.setattr(P, "git_sha7", lambda: "fixture")
    monkeypatch.setattr(P, "resolve_repo_name", lambda target: f"{target}-mlir")
    selection = P._build_selection("fixture", staging, manifest)
    dest = tmp_path / "stage" / "repo"
    result = P.assemble_repo_tree(selection, dest, layout_version=P.LAYOUT_VERSION)
    assert result["entrypoints"]["tool"] == "fixture-opt"
    assert (dest / "fixture-opt").read_bytes() == tool.read_bytes()
    assert (dest / "fixture-opt").stat().st_mode & 0o111
    assert (staging / "sentinel").read_text() == "source"
    assert not (dest / "CMakeLists.txt").exists()


@pytest.mark.parametrize("target", ["", ".", "..", "../escape", "/absolute", "nested/name"])
def test_stage_target_cannot_escape_build_root(staging, tmp_path, monkeypatch, target):
    build = tmp_path / "absent-build"
    monkeypatch.setattr(P.paths, "build_dir", lambda: build)
    with pytest.raises(P.PublishError, match="single staging path component"):
        P._new_stage_root(target, sources=(staging,))
    assert not build.exists()


def test_existing_clone_is_not_deleted_or_contacted(staging, tmp_path, monkeypatch):
    parent = tmp_path / "stage"
    clone = parent / "clone"
    clone.mkdir(parents=True)
    (clone / "sentinel").write_text("retained")

    def forbidden(*args, **kwargs):
        pytest.fail("unsafe staging must refuse before any git invocation")

    monkeypatch.setattr(P, "_git", forbidden)
    with pytest.raises(P.PublishError, match="fresh"):
        P._git_publish("unused", parent / "repo", SimpleNamespace(package_dir=staging), {}, "", "", parent, "unused")
    assert (clone / "sentinel").read_text() == "retained"
