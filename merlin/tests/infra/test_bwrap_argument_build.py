"""The explicit source recipe preserves all code except its argument bound."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import tarfile

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def recipe():
    path = repo_root() / "build_tools/scripts/build_bwrap_argument_budget.py"
    spec = importlib.util.spec_from_file_location("bwrap_argument_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "bubblewrap.c").write_bytes(
        b"before;\n  static const int32_t MAX_ARGS = 9000;\nafter;\n"
    )
    (root / "meson.build").write_bytes(b"project('bubblewrap', 'c', version : '0.9.0',)\n")
    return root


def test_only_unique_declaration_changes(recipe, source, tmp_path):
    before = recipe.inventory(source)
    recipe.patch_arguments(source, 65536, tmp_path / "change.patch")
    after = recipe.inventory(source)
    assert [a["path"] for a, b in zip(before["members"], after["members"], strict=True) if a != b] == [
        "bubblewrap.c"
    ]
    assert (source / "bubblewrap.c").read_bytes() == (
        b"before;\n  static const int32_t MAX_ARGS = 65536;\nafter;\n"
    )
    assert "-  static const int32_t MAX_ARGS = 9000;" in (tmp_path / "change.patch").read_text()


@pytest.mark.parametrize("value", [9000, 131073, -1, True, "65536"])
def test_invalid_bound_refuses_before_source_mutation(recipe, source, tmp_path, value):
    before = recipe.inventory(source)
    with pytest.raises(ValueError, match="argument bound"):
        recipe.patch_arguments(source, value, tmp_path / "change.patch")
    assert recipe.inventory(source) == before
    assert not (tmp_path / "change.patch").exists()


@pytest.mark.parametrize("extra", [b"", b"static const int32_t MAX_ARGS = 9000;\n"])
def test_changed_or_duplicate_original_declaration_refuses(recipe, source, tmp_path, extra):
    (source / "bubblewrap.c").write_bytes(b"static const int32_t MAX_ARGS = 9001;\n" + extra)
    with pytest.raises(ValueError, match="unique original"):
        recipe.patch_arguments(source, 65536, tmp_path / "change.patch")


def _archive(path, entries):
    with tarfile.open(path, "w:xz") as stream:
        for name, kind, target in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                data = target.encode()
                member.size = len(data)
                stream.addfile(member, io.BytesIO(data))
            else:
                member.linkname = target
                stream.addfile(member)


def test_original_in_root_license_alias_is_preserved(recipe, tmp_path):
    path = tmp_path / "release.tar.xz"
    _archive(path, [("bubblewrap-0.9.0/COPYING", tarfile.REGTYPE, "license"),
                    ("bubblewrap-0.9.0/LICENSE", tarfile.SYMTYPE, "COPYING")])
    root = recipe.extract_source(path, tmp_path / "extracted")
    assert (root / "LICENSE").is_symlink()
    assert (root / "LICENSE").read_text() == "license"
    assert recipe.inventory(root)["members"][1] == {
        "path": "LICENSE", "kind": "symlink", "target": "COPYING"
    }


@pytest.mark.parametrize("entry", [
    ("../escape", tarfile.REGTYPE, "untrusted"),
    ("bubblewrap-0.9.0/alias", tarfile.SYMTYPE, "/etc/passwd"),
    ("bubblewrap-0.9.0/alias", tarfile.SYMTYPE, "../outside"),
    ("bubblewrap-0.9.0/alias", tarfile.LNKTYPE, "bubblewrap-0.9.0/original"),
    ("bubblewrap-0.9.0/pipe", tarfile.FIFOTYPE, ""),
])
def test_unsafe_member_refuses_before_any_extraction(recipe, tmp_path, entry):
    path = tmp_path / "release.tar.xz"
    _archive(path, [("bubblewrap-0.9.0/original", tarfile.REGTYPE, "original"), entry])
    with pytest.raises(ValueError, match="archive"):
        recipe.extract_source(path, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_archive_sha_mismatch_is_not_replaced_with_observed_pin(recipe, tmp_path):
    path = tmp_path / "release"
    path.write_bytes(b"selected release")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert recipe.pinned_archive(path, actual)["sha256"] == actual
    path.write_bytes(b"changed release")
    with pytest.raises(ValueError, match="mismatch"):
        recipe.pinned_archive(path, actual)
