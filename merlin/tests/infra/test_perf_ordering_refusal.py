"""Screening refuses unqualified ordering without inventing empirical target evidence."""

from __future__ import annotations

import inspect
import json
import sys
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import emission_diagnostics as ED

from merlin.common.paths import merlin_dir
from merlin.targetgen.sandbox import toolchain as TC

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from merlin_experiments.phase2 import prompt as PP  # noqa: E402


@pytest.mark.parametrize("target", ["synthetic_array", "synthetic_vector"])
def test_ordering_refusal_never_attributes_unbound_experiment_statistics(tmp_path, target):
    from test_perf_analysis_action import _buffer

    path = _buffer(tmp_path, "buffer.json", (16, 16, 16))
    buffer = json.loads(path.read_text())
    buffer["target"] = target
    path.write_text(json.dumps(buffer))
    result = ED.analyze_command_buffers(
        path, path, peak_macs_per_cycle=None, achievable_macs_per_cycle=None, target=target
    )
    assert result["schema_version"] == 3
    ordering = result["ordering_signals"]
    assert ordering["status"] == "refused_unqualified_ordering"
    assert ordering["measured"] == {}
    assert ordering["artifact"] is None
    assert "qualified" in ordering["reason"]
    assert result["differential"]["basis"] == "not_attempted"


def test_prompt_does_not_present_one_experiment_as_universal_evidence():
    source = inspect.getsource(PP)
    assert "says so with the numbers" not in source
    assert "0.273" not in source
    assert "qualified ordering" in source


def test_the_registry_description_promises_only_what_is_returned(tmp_path, monkeypatch):
    """It must not name a verdict the action stopped returning."""
    (tmp_path / "manifest.yaml").write_text("entrypoints: {tool: tool.py}\ncommands: {compile: {argv: ['{tool}']}}\n")
    (tmp_path / "tool.py").write_text("# candidate entrypoint; never executed\n")
    monkeypatch.setattr(TC, "required_tool_probes", lambda _target: [])
    actions = BP.action_registry(BP.CORPUS_FEEDBACK_V1, tmp_path, SimpleNamespace())
    description = next(action.purpose for action in actions if action.name == BP.ANALYSIS_ACTION)
    assert "differential verdict" not in description, (
        "the differential was removed and returns not_attempted; the description must not promise it"
    )
    assert "cannot certify" in description or "never certify" in description, (
        "a screen that may only eliminate has to say so where the agent reads it"
    )


def test_the_prompt_documents_the_free_action_and_its_limit():
    source = inspect.getsource(PP)
    assert BP.ANALYSIS_ACTION in source, "the prompt never mentions the free screening action"
    assert "no oracle time" in source, "the agent must be told the screen is free, or it will not use it"
    assert "never certify" in source, (
        "the screen may eliminate and may never certify; the prompt has to carry the asymmetry"
    )


def test_every_verdict_the_harness_can_emit_is_explained_to_the_agent():
    """A field the agent receives but is never told about is a field it cannot act on.

    `verdict` is the per-member "is this one finished" signal -- the only principled basis for giving
    up on a capsule -- and it reached the agent for a whole campaign with no mention anywhere in the
    prompt. Pin every value the module can emit, so adding a sixth cannot silently go undocumented.
    """
    from merlin_experiments.phase2 import capsule_verdict as CV

    prompt_source = inspect.getsource(PP)
    emitted = {CV.NO_HEADROOM, CV.IMPROVED, CV.HEADROOM_OPEN, CV.REGRESSED, CV.REFUSED}
    missing = sorted(v for v in emitted if v not in prompt_source)
    assert not missing, f"verdict value(s) the agent is never told about: {missing}"
    assert "verdict_reason" in prompt_source
