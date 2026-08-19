"""The comparison table must carry the variables that explain the outcome, not just the score.

Measured on the 2026-08 gemmini campaign: two runs both read `0/20`, but one reached the numeric and trace
tiers on 17 of 20 capsules and lost only on the hardware encoding, while the other reached no tier at all.
The run that scored 20/20 used the RTL-derived tooling every round and read for 73% of its actions before
its first edit; the runs that scored zero read for 9% and then rewrote heavily. None of that was in the
table, so the table said the two zeros were the same result.
"""
from __future__ import annotations

import importlib.util
import json

import pytest

from merlin.common.paths import merlin_dir

_SRC = merlin_dir() / "experiments/capsule_bench/harness/agg_by_model.py"


@pytest.fixture(scope="module")
def agg():
    if not _SRC.is_file():
        pytest.skip(f"{_SRC} not present")
    import sys
    sys.path.insert(0, str(_SRC.parent))
    try:
        spec = importlib.util.spec_from_file_location("agg_by_model_under_test", _SRC)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


def _run_dir(tmp_path, per_capsule, extra_results=0):
    d = tmp_path / "run"
    g = d / "grading_public"
    g.mkdir(parents=True)
    (g / "score_capsule.json").write_text(json.dumps({"per_capsule": per_capsule}))
    # leftovers from a wider grading pass: present on disk, not part of the graded set
    for i in range(extra_results):
        p = g / "runs" / "bench" / f"leftover_{i}"
        p.mkdir(parents=True)
        (p / "capsule_result.json").write_text(json.dumps({"capsule": f"leftover_{i}",
                                                           "status": "fail", "tiers": {}}))
    return d


def test_partial_credit_distinguishes_two_identical_zeros(agg, tmp_path):
    reached = [{"capsule": f"c{i}", "tiers": {"L0": {"status": "pass"}, "L1": {"status": "pass"},
                                              "L2": {"status": "fail"}}} for i in range(17)]
    reached += [{"capsule": f"d{i}", "tiers": {}} for i in range(3)]
    never = [{"capsule": f"e{i}", "tiers": {"L0": {"status": "fail"}}} for i in range(20)]

    got_reached = agg._tier_reach(_run_dir(tmp_path / "a", reached))
    got_never = agg._tier_reach(_run_dir(tmp_path / "b", never))
    assert got_reached == {"L1": 17, "none": 3}
    assert got_never == {"none": 20}
    assert got_reached != got_never, "two runs that both score 0/20 must not look identical"


def test_leftover_result_files_do_not_inflate_the_graded_set(agg, tmp_path):
    """A regrade or a wider pass leaves extra capsule dirs; the graded set is the grader's own record."""
    per = [{"capsule": f"c{i}", "tiers": {"L3": {"status": "pass"}}} for i in range(20)]
    reach = agg._tier_reach(_run_dir(tmp_path, per, extra_results=123))
    assert reach == {"L3": 20}, f"graded set inflated by leftovers: {reach}"


def test_behaviour_reads_both_driver_event_shapes(agg, tmp_path):
    """Codex records actions as items; the CLI drivers record tool_use blocks. Reading one shape reported
    the codex runs as making zero edits -- false, and backwards from the point of the metric."""
    d = tmp_path / "codexrun" / "rounds"
    d.mkdir(parents=True)
    ev = [{"event": {"type": "item.completed", "item": {"type": "command_execution"}}} for _ in range(3)]
    ev += [{"event": {"type": "item.completed", "item": {"type": "file_change"}}}]
    (d / "round_00.codex_events.timestamped.jsonl").write_text(
        "\n".join(json.dumps(e) for e in ev))
    b = agg._behaviour(tmp_path / "codexrun")
    assert b["actions"] == 4 and b["writes"] == 1
    assert b["recon_before_first_write"] == 3, "recon is the actions taken before the first edit"


def test_conformance_reports_whether_the_tooling_was_ever_used(agg, tmp_path):
    import yaml
    d = tmp_path / "run"
    d.mkdir()
    (d / "qa_loop_state.yaml").write_text(yaml.safe_dump({"rounds": [
        {"conformance": {"conformant": False, "checks": {"isa_tools_used": False, "cca_used": False}}},
        {"conformance": {"conformant": True, "checks": {"isa_tools_used": True, "cca_used": False}}},
    ]}))
    c = agg._conformance(d)
    assert c["ever"]["isa_tools_used"] is True, "used in any round counts as used"
    assert c["ever"]["cca_used"] is False
    assert c["conformant_rounds"] == 1
