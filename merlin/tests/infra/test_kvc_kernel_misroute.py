"""An agent CLI may write the kernel to a path the harness does not watch.

This is pinned because the failure is INVISIBLE in the score: a kernel written to the wrong root and
a model that could not write a kernel both surface as `correct=False`, and the study would publish the
second explanation for the first fact. It has happened once here -- opencode resolved a relative path
against the enclosing checkout rather than the workspace, and three arms were recorded 0/3 for kernels
they had written.
"""
import json
import sys

import pytest

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin/experiments/llm_kernel_vs_compiler_v0/scripts"))
import run_kernel_agent as R  # noqa: E402


def _transcript(tmp_path, records, name="t.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


def _expected(tmp_path):
    return tmp_path / "ws" / "submission" / "kernel.llvm.mlir"


def test_a_kernel_written_to_another_root_is_found(tmp_path):
    stray = tmp_path / "repo" / "submission" / "kernel.llvm.mlir"
    stray.parent.mkdir(parents=True)
    stray.write_text("module {}")
    t = _transcript(tmp_path, [{"part": {"state": {"input": {"filePath": str(stray)}}}}])
    assert R._find_misrouted_kernel(t, expected=_expected(tmp_path)) == stray


@pytest.mark.parametrize("dumps", [
    lambda r: json.dumps(r),                        # compact
    lambda r: json.dumps(r, indent=None, separators=(", ", ": ")),   # spaced
])
def test_detection_does_not_depend_on_the_emitter_s_spacing(tmp_path, dumps):
    """A text match would go blind the day a CLI pretty-printed its output."""
    stray = tmp_path / "elsewhere" / "kernel.llvm.mlir"
    stray.parent.mkdir(parents=True)
    stray.write_text("module {}")
    p = tmp_path / "t.jsonl"
    p.write_text(dumps({"type": "tool_use", "part": {"tool": "write",
                                                     "state": {"input": {"filePath": str(stray)}}}}))
    assert R._find_misrouted_kernel(p, expected=_expected(tmp_path)) == stray


def test_an_agent_that_wrote_nothing_is_not_excused(tmp_path):
    """The guard must not invent a kernel for a model that genuinely produced none."""
    t = _transcript(tmp_path, [{"type": "text", "part": {"text": "I was unable to write it"}}])
    assert R._find_misrouted_kernel(t, expected=_expected(tmp_path)) is None


def test_a_write_to_the_expected_path_is_not_a_misroute(tmp_path):
    exp = _expected(tmp_path)
    exp.parent.mkdir(parents=True)
    exp.write_text("module {}")
    t = _transcript(tmp_path, [{"part": {"state": {"input": {"filePath": str(exp)}}}}])
    assert R._find_misrouted_kernel(t, expected=exp) is None


def test_a_path_named_in_the_transcript_but_absent_on_disk_is_not_a_kernel(tmp_path):
    """An attempted-then-failed write must not be scored as a kernel that exists."""
    t = _transcript(tmp_path, [{"part": {"state": {"input": {
        "filePath": str(tmp_path / "never" / "kernel.llvm.mlir")}}}}])
    assert R._find_misrouted_kernel(t, expected=_expected(tmp_path)) is None


def test_a_malformed_line_does_not_abort_the_scan(tmp_path):
    stray = tmp_path / "x" / "kernel.llvm.mlir"
    stray.parent.mkdir(parents=True)
    stray.write_text("module {}")
    p = tmp_path / "t.jsonl"
    p.write_text("not json at all\n{\"broken\": \n"
                 + json.dumps({"part": {"state": {"input": {"filePath": str(stray)}}}}))
    assert R._find_misrouted_kernel(p, expected=_expected(tmp_path)) == stray
