import hashlib
import stat
import zipfile
from dataclasses import replace

import pytest
from merlin_experiments.phase1 import reference_inputs as R
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


def raw_origin(tmp_path, payload=b"\x7fELF\x00\xff", *, name="raw_example"):
    archive = tmp_path / (name + ".zip")
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("artifact.bin", payload)
    return ReferenceOrigin(name, "user_permitted_archive", archive, digest(archive.read_bytes()),
        "user-provided artifact", (ReferenceFile("artifact.bin", "artifact.bin", digest(payload)),),
        user_authorization="explicit user permission for raw evidence", content_kind="raw_evidence")


def test_binary_requires_explicit_raw_archive_permission_and_stays_inert(tmp_path):
    origin = raw_origin(tmp_path)
    with pytest.raises(ValueError, match="source text|binary payload"):
        stage_references((replace(origin, content_kind="source_text"),), tmp_path / "old", review_id="review")
    assert not (tmp_path / "old").exists()
    result = stage_references((origin,), tmp_path / "raw", review_id="review")
    staged = tmp_path / "raw/raw_example/artifact.bin"
    assert staged.read_bytes() == b"\x7fELF\x00\xff"
    assert stat.S_IMODE(staged.stat().st_mode) == 0o444
    assert result["origins"][0]["content_kind"] == "raw_evidence"
    assert result["origins"][0]["allowed_purpose"] == "reference_reading"
    assert "execution" in result["scope"]
    with pytest.raises(ValueError, match="explicit user-permitted"):
        replace(origin, kind="public_repository")
    with pytest.raises(ValueError, match="authorization"):
        replace(origin, user_authorization="")


@pytest.mark.parametrize("pin", ["archive", "member"])
def test_raw_evidence_keeps_exact_archive_and_member_pins(tmp_path, pin):
    origin = raw_origin(tmp_path)
    if pin == "archive":
        origin = replace(origin, identity="0" * 64)
    else:
        origin = replace(origin, files=(replace(origin.files[0], sha256="0" * 64),))
    with pytest.raises(ValueError, match="differs|bytes differ"):
        stage_references((origin,), tmp_path / "bad", review_id="review")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("bound", ["file", "total", "source_total", "source_file"])
def test_raw_bounds_never_relax_source_bounds_and_refuse_before_writing(tmp_path, monkeypatch, bound):
    monkeypatch.setattr(R, "RAW_MAX_FILE_BYTES", 16)
    monkeypatch.setattr(R, "RAW_MAX_TOTAL_BYTES", 24)
    monkeypatch.setattr(R, "MAX_TOTAL_BYTES", 8)
    if bound == "source_file":
        monkeypatch.setattr(R, "MAX_FILE_BYTES", 8)
    if bound == "file":
        origins = (raw_origin(tmp_path, b"x" * 17),)
    elif bound == "total":
        origins = (raw_origin(tmp_path, b"x" * 13), raw_origin(tmp_path, b"x" * 13, name="second"))
    else:
        origins = (raw_origin(tmp_path, b"x"),
            replace(raw_origin(tmp_path, b"s" * 9, name="source"), content_kind="source_text"))
    with pytest.raises(ValueError, match="oversized|staging bounds"):
        stage_references(origins, tmp_path / "bad", review_id="review")
    assert not (tmp_path / "bad").exists()


def test_raw_evidence_does_not_relax_archive_symlink_protection(tmp_path):
    origin = raw_origin(tmp_path)
    with zipfile.ZipFile(origin.container, "w") as stream:
        link = zipfile.ZipInfo("artifact.bin")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        stream.writestr(link, b"\x7fELF\x00\xff")
    origin = replace(origin, identity=digest(origin.container.read_bytes()))
    with pytest.raises(ValueError, match="indirect"):
        stage_references((origin,), tmp_path / "bad", review_id="review")
    assert not (tmp_path / "bad").exists()
