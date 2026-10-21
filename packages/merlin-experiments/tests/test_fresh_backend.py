"""Fresh author origin is explicit, empty and immutable across resume."""

from dataclasses import replace

import pytest

from merlin_experiments.phase1 import fresh_backend
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.session import validate_options


def selection():
    return parse_options(["--run-id", "fresh", "--fresh-backend"], environ={})


@pytest.mark.parametrize("option", ["seed_submission", "qualify_submission"])
def test_native_admission_refuses_seed_before_launch(option):
    with pytest.raises(ValueError, match="empty implementation"):
        validate_options(replace(selection(), **{option: "/previous/compiler"}))


def test_existing_backend_cannot_be_claimed_empty(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "lower.py").write_text("# previous compiler")
    with pytest.raises(ValueError, match="empty directory"):
        fresh_backend.origin(selection(), tmp_path)


def test_resume_preserves_new_author_work_and_original_origin(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    record = fresh_backend.origin(selection(), tmp_path)
    (submission / "lower.py").write_text("# new author implementation")
    assert fresh_backend.origin(selection(), tmp_path, prior={"backend_origin": record}) == record
    with pytest.raises(ValueError, match="original run"):
        fresh_backend.origin(selection(), tmp_path, prior={})
    with pytest.raises(ValueError, match="original run"):
        fresh_backend.origin(replace(selection(), fresh_backend=False), tmp_path, prior={"backend_origin": record})


def test_linked_or_unisolated_backend_refuses(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (tmp_path / "submission").symlink_to(old, target_is_directory=True)
    with pytest.raises(ValueError, match="empty directory"):
        fresh_backend.origin(selection(), tmp_path)
    with pytest.raises(ValueError, match="isolated"):
        validate_options(replace(selection(), sandbox="none"))
