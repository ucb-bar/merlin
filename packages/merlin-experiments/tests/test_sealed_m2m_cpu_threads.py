"""A capture may select its intra-op CPU thread count; the count binds the command and the replay.

Reference bytes depend on how parallel reductions split their work, so a thread count is part of the
selected numeric policy, never an ambient default. A TinyLlama capture at four threads replayed byte
for byte on three different CPU sets; its float-stage bytes differ from one thread. Absent, the
command, plan and policy bytes are exactly those of a one-thread capture.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from merlin_experiments import cli
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_m2m import SealedM2MError
from merlin_experiments.phase0 import capture_selection


def test_the_selected_thread_count_is_part_of_the_command_and_policy():
    output = Path("/capture-out")
    plain = sealed_m2m._command_v2(output, dtype="int8", recipe=True)
    assert plain == sealed_m2m._command_v2(output, dtype="int8", recipe=True, options={})
    assert "--cpu-threads" not in plain[-1]
    threaded = sealed_m2m._command_v2(output, dtype="int8", recipe=True, options={"cpu_threads": 4})
    assert "'--cpu-threads', '4'" in threaded[-1]
    assert sealed_m2m._policy(threaded, output) != sealed_m2m._policy(plain, output)
    assert sealed_m2m._command_v2(output, dtype="int8", recipe=True, options={"cpu_threads": 1}) != plain


@pytest.mark.parametrize("bad", [0, 257, True, 4.0, "4", None])
def test_an_unbounded_or_non_integer_thread_count_is_refused(bad):
    with pytest.raises(SealedM2MError, match="cpu_threads"):
        sealed_m2m._worker_options({"cpu_threads": bad})


def test_selection_flag_selects_the_worker_option_and_absence_keeps_the_plan(tmp_path, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        capture_selection, "select", lambda **kwargs: calls.append(kwargs) or {"sha256": "a" * 64, "schema": "x"}
    )
    base = [
        "corpus",
        "capture",
        "select",
        "--m2m-root",
        str(tmp_path / "m2m"),
        "--workload-root",
        str(tmp_path / "workload"),
        "--venv",
        str(tmp_path / "venv"),
        "--run-dir",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "selection"),
    ]
    assert cli.main([*base, "--cpu-threads", "4"]) == 0
    assert cli.main([*base, "--cpu-threads", "4", "--stage-fp32"]) == 0
    assert cli.main(base) == 0
    assert calls[0]["worker_options"] == {"cpu_threads": 4}
    assert calls[1]["worker_options"] == {"stage_fp32": True, "cpu_threads": 4}
    assert "worker_options" not in calls[2]
