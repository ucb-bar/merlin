"""Paired diagnostics fail before target execution when identity/deadlines are unresolved."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.perf.fixed_work_context import project_fixed_work_context
from merlin.perf.paired_context_provider import PairedControlledContextProvider


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -1, 0, True])
def test_invalid_deadline_never_accesses_target(tmp_path, deadline):
    provider = PairedControlledContextProvider(target="fixture", adapter=None, output=tmp_path)
    with pytest.raises(ValueError, match="finite positive deadline"):
        provider(candidate=Path("unused"), probes=None, timeout_s=deadline)


def test_changed_graph_never_prepares_pair(tmp_path):
    provider = PairedControlledContextProvider(target="fixture", adapter=None, output=tmp_path)
    probes = SimpleNamespace(
        analysis=SimpleNamespace(
            session=SimpleNamespace(
                previous_probe_binding=lambda candidate: SimpleNamespace(graph_digest="old", target_digest="same"),
                current_probe_binding=lambda candidate: SimpleNamespace(graph_digest="new", target_digest="same"),
            )
        )
    )
    with pytest.raises(ValueError, match="changed logical graph or target"):
        provider(candidate=Path("unused"), probes=probes)


def test_projection_checks_actual_source_before_decoder():
    with pytest.raises(ValueError, match="source hash differs"):
        project_fixed_work_context(
            {"lowered_text": "changed", "candidate_lowered_sha256": "0" * 64}, {}, {}, target="fixture"
        )
