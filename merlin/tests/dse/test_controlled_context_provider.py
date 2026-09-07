"""Refusal paths run before any compiler, target code or simulator invocation."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.perf.controlled_context_provider import ControlledSourcePrefixProvider


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -1, 0, True])
def test_provider_refuses_invalid_deadline_before_accessing_experiment(tmp_path, deadline):
    provider = ControlledSourcePrefixProvider(target="test-target", adapter=None, output=tmp_path)
    with pytest.raises(ValueError, match="finite positive deadline"):
        provider(candidate=Path("unused"), experiment=None, timeout_s=deadline)


def test_provider_refuses_stale_actual_source_before_parse_or_prepare(tmp_path):
    provider = ControlledSourcePrefixProvider(target="test-target", adapter=None, output=tmp_path)
    experiment = SimpleNamespace(current_probe_binding=lambda candidate: object(),
                                 current_artifacts=lambda candidate: {
                                     "lowered_text": "changed source", "candidate_lowered_sha256": "0" * 64})
    with pytest.raises(ValueError, match="retained full-model source artifact changed"):
        provider(candidate=Path("unused"), experiment=experiment)
