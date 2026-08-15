"""The transcript answer-access audit must distinguish a BLOCKED probe of a masked answer file (the mask
bound it to /dev/null, so the read returned nothing) from a real LEAK (content actually returned). A
thorough model that merely `cat`s a masked golden and gets empty output must NOT mark the run unclean —
only a read that returns withheld content, or oracle USE, is a violation.
"""
from __future__ import annotations

import json
import sys

import pytest

from merlin.common.paths import merlin_dir

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
_ATLAS = merlin_dir() / "experiments/capsule_bench/targets/atlas/target_experiment.yaml"


@pytest.fixture()
def audit(monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(_ATLAS))
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    import run_baseline_qa_loop as L  # noqa: PLC0415
    return L.audit_transcript


def _transcript(tmp_path, cmd, stdout, tid="t1", tool="Bash"):
    inp = {"command": cmd} if tool == "Bash" else {"file_path": cmd}
    lines = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": tid, "name": tool, "input": inp}]}},
        {"type": "user", "tool_use_result": {"stdout": stdout},
         "message": {"content": [{"type": "tool_result", "tool_use_id": tid,
                                  "content": stdout or "(Bash completed with no output)"}]}},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines))
    return p


def test_blocked_probe_of_masked_golden_stays_clean(audit, tmp_path):
    # cat a masked golden -> empty result (the /dev/null mask). Advisory, NOT a violation.
    tp = _transcript(tmp_path, "cat isa/AT0_config_smoke/golden.yaml 2>/dev/null | head -10", "")
    r = audit(tp, arm="merlin_assisted")
    assert r["clean"] is True
    assert r["blocked_probes"] == 1
    assert [h["kind"] for h in r["hits"]] == ["blocked_probe"]


def test_real_leak_breaks_clean(audit, tmp_path):
    # Same command but the read RETURNED content -> the mask failed -> a genuine violation.
    tp = _transcript(tmp_path, "cat isa/AT0_config_smoke/golden.yaml", "expected_out: [1.0, 2.0, 3.0]")
    r = audit(tp, arm="merlin_assisted")
    assert r["clean"] is False
    assert any(h["kind"] == "path_read" for h in r["hits"])


def test_read_tool_blocked_probe_stays_clean(audit, tmp_path):
    tp = _transcript(tmp_path, "isa/AT0_config_smoke/golden.yaml", "", tool="Read")
    r = audit(tp, arm="merlin_assisted")
    assert r["clean"] is True
    assert r["blocked_probes"] == 1


@pytest.mark.parametrize("err", [
    "Error: ENOENT: no such file or directory, open 'isa/AT0_config_smoke/golden.yaml'",
    "cat: isa/AT0_config_smoke/golden.yaml: Permission denied",
    "cat: isa/AT0_config_smoke/golden.yaml: No such file or directory",
])
def test_read_that_errored_is_blocked_not_leak(audit, tmp_path, err):
    # The mask blocks an answer path by absence / mode-000 too, not only by binding to /dev/null: the read
    # then returns an ERROR, no answer bytes. That attempt must classify as blocked_probe, never path_read.
    tp = _transcript(tmp_path, "isa/AT0_config_smoke/golden.yaml", err, tool="Read")
    r = audit(tp, arm="merlin_assisted")
    assert r["clean"] is True
    assert [h["kind"] for h in r["hits"]] == ["blocked_probe"]


def test_no_answer_reference_is_clean(audit, tmp_path):
    tp = _transcript(tmp_path, "ls submission", "manifest.yaml\natlas_opt.py")
    r = audit(tp, arm="merlin_assisted")
    assert r["clean"] is True
    assert r["hits"] == []


def test_grep_l_finding_own_tool_is_recon_not_leak(audit, tmp_path):
    # `grep -l capsule_grade` (path-listing search) that surfaces only the agent's OWN granted shim is
    # advisory reconnaissance — it returned a filename, not answer content. Must NOT break clean.
    tp = _transcript(
        tmp_path,
        'find . -name "*.py" | xargs grep -l "capsule_grade" 2>/dev/null | head',
        "/scratch/.../targets/atlas/scripts/agent_selfcheck.py")
    r = audit(tp, arm="merlin_assisted")
    assert r["clean"] is True
    assert r["recon_probes"] == 1
    assert [h["kind"] for h in r["hits"]] == ["recon_probe"]


def test_search_that_surfaces_an_answer_path_is_a_violation(audit, tmp_path):
    # A path-listing search that names an answer file AND returns it (a mask failure would expose the golden
    # path) is NOT benign recon — the answer-token in the result keeps it a violation.
    tp = _transcript(tmp_path, 'grep -l "config" isa/AT0_config_smoke/golden.yaml',
                     "isa/AT0_config_smoke/golden.yaml")
    r = audit(tp, arm="merlin_assisted")
    assert r["clean"] is False


def test_content_grep_that_returns_data_is_a_violation(audit, tmp_path):
    # grep WITHOUT -l returns matching content lines; if a masked answer file ever returned data, that is a
    # real leak (mask breach), not path-listing recon.
    tp = _transcript(tmp_path, 'grep -h "." isa/AT0_config_smoke/golden.yaml', "expected: [1.0, 2.0]")
    r = audit(tp, arm="merlin_assisted")
    assert r["clean"] is False
