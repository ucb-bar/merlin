"""Large explicitly permitted containers are inert, bounded, owned copies."""

import hashlib
import json
import stat
from dataclasses import replace

import pytest
from merlin_experiments.corpus.public_references import stage
from merlin_experiments.phase1.reference_inputs import ReferenceFile, ReferenceOrigin, stage_references


def origin(tmp_path):
    path = tmp_path / "models.tar.zst"
    path.write_bytes(b"opaque-not-a-zip\0" * 100)
    path.chmod(0o664)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return ReferenceOrigin(
        "models",
        "user_permitted_archive",
        path,
        sha,
        "user supplied",
        (ReferenceFile(path.name, path.name, sha, path.stat().st_size),),
        user_authorization="explicit user authorization",
        content_kind="opaque_container",
    )


def test_copy_never_chmods_original_archive_or_hardlink_inode(tmp_path):
    selected = origin(tmp_path)
    before = selected.container.stat()
    stage_references((selected,), tmp_path / "bank", review_id="review")
    copy = tmp_path / "bank/models/models.tar.zst"
    assert copy.read_bytes() == selected.container.read_bytes()
    assert copy.stat().st_ino != before.st_ino
    assert stat.S_IMODE(copy.stat().st_mode) == 0o444
    assert stat.S_IMODE(selected.container.stat().st_mode) == 0o664


@pytest.mark.parametrize("changed", ["size", "pin", "member"])
def test_opaque_pin_size_and_container_identity_are_jointly_required(tmp_path, changed):
    selected = origin(tmp_path)
    member = selected.files[0]
    if changed == "size":
        member = replace(member, size_bytes=member.size_bytes + 1)
    elif changed == "pin":
        member = replace(member, sha256="0" * 64)
    else:
        member = replace(member, member="unselected-member")
    with pytest.raises(ValueError):
        stage_references((replace(selected, files=(member,)),), tmp_path / "bank", review_id="review")
    assert not (tmp_path / "bank").exists()


def test_existing_manifest_stage_admits_explicit_opaque_size_pin(tmp_path):
    selected = origin(tmp_path)
    review = dict(
        schema="operator.allowed_reference_sources.v2",
        review_id="review",
        origins=[
            dict(
                name=selected.name,
                kind=selected.kind,
                container=selected.container.name,
                identity=selected.identity,
                license=selected.license,
                user_authorization=selected.user_authorization,
                allowed_purpose=selected.allowed_purpose,
                content_kind=selected.content_kind,
                files=[
                    dict(
                        member=selected.container.name,
                        destination=selected.container.name,
                        sha256=selected.identity,
                        size_bytes=selected.container.stat().st_size,
                    )
                ],
            )
        ],
    )
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    record = stage(path, tmp_path / "bank")
    assert record["members"] == 1
    assert stat.S_IMODE(selected.container.stat().st_mode) == 0o664
