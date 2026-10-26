import hashlib
import zipfile
from dataclasses import replace

import pytest
from merlin_experiments.phase1.reference_inputs import ReferenceFile, ReferenceOrigin, stage_references


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_public_repository_copies_only_exact_reviewed_files(tmp_path):
    source = tmp_path / "repository"
    source.mkdir()
    (source / "schedule.py").write_bytes(b"public scheduling example\n")
    (source / "not_selected.txt").write_bytes(b"not exported")
    origin = ReferenceOrigin("public_example", "public_repository", source, "1" * 40, "LICENSE", (
        ReferenceFile("schedule.py", "schedule.py", digest((source / "schedule.py").read_bytes())),),
        public_url="https://example.org/compiler")
    result = stage_references((origin,), tmp_path / "stage", review_id="review-1")
    assert result["origins"][0]["public_url"] == origin.public_url
    assert not (tmp_path / "stage/public_example/not_selected.txt").exists()
    assert str(source) not in (tmp_path / "stage/reference-selection.json").read_text()
    with pytest.raises(ValueError, match="fresh"):
        stage_references((origin,), tmp_path / "stage", review_id="review-1")


def archive_origin(tmp_path):
    archive = tmp_path / "authorized.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("permitted/code.py", b"example\n")
        stream.writestr("other/data", b"not selected")
    return ReferenceOrigin("explicit_archive", "user_permitted_archive", archive,
        digest(archive.read_bytes()), "LICENSE", (ReferenceFile("permitted/code.py", "code.py", digest(b"example\n")),),
        user_authorization="explicit user permission record")


def test_archive_is_explicit_permission_not_relabelled_public_source(tmp_path):
    origin = archive_origin(tmp_path)
    result = stage_references((origin,), tmp_path / "stage", review_id="archive-review")
    assert result["origins"][0]["kind"] == "user_permitted_archive"
    assert (tmp_path / "stage/explicit_archive/code.py").read_bytes() == b"example\n"
    assert not (tmp_path / "stage/explicit_archive/other").exists()
    with pytest.raises(ValueError, match="authorization"):
        replace(origin, user_authorization="")
    with pytest.raises(ValueError, match="positively"):
        replace(origin, kind="coworker_private_archive")


def test_changed_archive_and_source_pins_refuse_before_output(tmp_path):
    origin = archive_origin(tmp_path)
    bad = replace(origin, identity="0" * 64)
    with pytest.raises(ValueError, match="archive differs"):
        stage_references((bad,), tmp_path / "bad", review_id="review")
    assert not (tmp_path / "bad").exists()
    bad = replace(origin, files=(replace(origin.files[0], sha256="0" * 64),))
    with pytest.raises(ValueError, match="bytes differ"):
        stage_references((bad,), tmp_path / "bad", review_id="review")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("path", ["../x", "/x", "a/../x", "a//x", "a\\x", ".git/config"])
def test_no_archive_traversal_or_implicit_directory_grant(path):
    with pytest.raises(ValueError):
        ReferenceFile(path, "code.py", "0" * 64)


def test_archive_duplicate_and_symlink_members_refuse(tmp_path):
    origin = archive_origin(tmp_path)
    with zipfile.ZipFile(origin.container, "a") as stream:
        with pytest.warns(UserWarning):
            stream.writestr("permitted/code.py", b"example\n")
    origin = replace(origin, identity=digest(origin.container.read_bytes()))
    with pytest.raises(ValueError, match="ambiguous"):
        stage_references((origin,), tmp_path / "duplicate", review_id="review")
    assert not (tmp_path / "duplicate").exists()
    with zipfile.ZipFile(origin.container, "w") as stream:
        link = zipfile.ZipInfo("permitted/code.py")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        stream.writestr(link, "other/data")
    origin = replace(origin, identity=digest(origin.container.read_bytes()))
    with pytest.raises(ValueError, match="indirect"):
        stage_references((origin,), tmp_path / "symlink", review_id="review")
