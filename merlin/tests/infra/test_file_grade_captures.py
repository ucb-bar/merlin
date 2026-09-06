"""A finished grade already holds the inputs a cross-validation capture needs.

The expensive half of a capture is the REFERENCE engine, and a performance campaign pays for it in
front of a run that cannot start until it finishes -- while the functional grade, which needs no
certificate at all, has already produced the command buffer, the lowered module and the ELF for every
capsule it lowered. These tests pin the selection rule, which is the whole of the logic: a capsule is
capturable exactly when the grade left both artifacts, so a capsule the backend DECLINED to lower -- it
emitted neither -- is absent rather than attempted and failed.
"""
from __future__ import annotations

import sys
from pathlib import Path

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts"))

import file_grade_captures as F  # noqa: E402


def _grade(root: Path, capsule: str, *, buffer: bool = True, lowered: bool = True) -> None:
    generated = root / capsule / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    if buffer:
        (generated / "command_buffer.json").write_text("{}", encoding="utf-8")
    if lowered:
        (generated / "lowered.llvm.mlir").write_text("module {}", encoding="utf-8")


def test_a_capsule_the_grade_lowered_is_capturable(tmp_path):
    _grade(tmp_path, "lowered_one")
    assert [name for name, _ in F._graded_capsules(tmp_path)] == ["lowered_one"]


def test_a_declined_capsule_emitted_nothing_and_is_absent(tmp_path):
    """The backend STATED it does not lower these, so there is no program to cross-validate."""
    _grade(tmp_path, "kept")
    (tmp_path / "declined" / "generated").mkdir(parents=True)      # the dir exists, empty
    assert [name for name, _ in F._graded_capsules(tmp_path)] == ["kept"]


def test_half_an_emission_is_not_capturable(tmp_path):
    """Either artifact alone cannot build the ELF, and a capture is about ONE ELF run twice."""
    _grade(tmp_path, "no_buffer", buffer=False)
    _grade(tmp_path, "no_lowered", lowered=False)
    assert F._graded_capsules(tmp_path) == []


def test_the_generated_dir_is_returned_so_capture_case_reads_the_same_bytes(tmp_path):
    _grade(tmp_path, "capsule_a")
    (name, generated), = F._graded_capsules(tmp_path)
    assert generated == tmp_path / "capsule_a" / "generated"
    assert (generated / "command_buffer.json").is_file()
