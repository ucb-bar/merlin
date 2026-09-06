"""The run-report readers, tested where they can silently return nothing.

Every test here is paired: one direction proves the reader FINDS the thing, the other proves it
REFUSES when the thing is not there. A reader that always returns zero passes a one-directional
suite, and this repo has shipped several of those.
"""
from __future__ import annotations

import json

import pytest

from merlin.agentreport.availability import (Availability, DERIVED, MEASURED, UNAVAILABLE, Status,
                                             derived, measured, unavailable)
from merlin.agentreport.index import (ArmSpec, UNKNOWN, arm_from_prefix, build_index, resolve_arm)
from merlin.agentreport.passes import read_passes, _rebase
from merlin.agentreport.spans import (FLUSH_FLOOR_S, SOURCE_RAW_ITEMS, SOURCE_TRANSCRIPT, Span,
                                      SpanSet, concurrency, read_spans)
from merlin.agentreport.capsule_time import read_capsule_timings, summarize
from merlin.agentreport.phase2 import read_phase2
from merlin.agentreport.tokens import METERED, NOTIONAL, UNPRICED, normalize_model, read_tokens

ARMS = (
    ArmSpec("arm1", "raw_baseline", "rb", ("raw_baseline_hwbringup_v0", "raw_baseline_public_v0")),
    ArmSpec("arm2", "cpp_merlininfra", "rbinfra", ("cpp_merlininfra_hwbringup_v0",)),
    ArmSpec("arm3", "merlin_assisted", "merlin",
            ("merlin_assisted_hwbringup_v0", "merlin_assisted_public_v0")),
    ArmSpec("arm4", "merlin_rtlchecks", "merlincirct",
            ("merlin_assisted_rtlchecks_hwbringup_v0", "merlin_assisted_rtlchecks_public_v0")),
    ArmSpec("eqsat", "merlin_eqsat", "merlineqsat", ("merlin_assisted_eqsat_hwbringup_v0",)),
)
PHASES = {"capsule-bench": "phase1", "perf-bench": "phase2"}


# ---------------------------------------------------------------- availability

def test_unavailable_must_carry_a_reason():
    """A blank refusal renders as an empty cell, which reads as 'nothing to say'."""
    with pytest.raises(ValueError):
        Status(UNAVAILABLE)
    assert Status(UNAVAILABLE, reason="because").reason == "because"


def test_availability_score_ranks_a_plottable_run_above_a_stub():
    full = Availability({"a": measured(), "b": derived("x")})
    half = Availability({"a": measured(), "b": unavailable("x")})
    assert full.score == 1.0 and half.score == 0.5
    assert half.reasons() == {"x": ["b"]}


# ---------------------------------------------------------------- arm resolution

@pytest.mark.parametrize("run_id,expected", [
    ("rb_x", "arm1"), ("rbinfra_x", "arm2"), ("merlin_x", "arm3"),
    ("merlincirct_x", "arm4"), ("merlineqsat_x", "eqsat"), ("nothing_x", UNKNOWN),
])
def test_prefix_match_is_longest_wins(run_id, expected):
    assert arm_from_prefix(run_id, ARMS) == expected


def test_prefix_match_does_not_depend_on_spec_order():
    """`merlin` vs `merlincirct` is decided by length, never by which was checked first."""
    for rotation in range(len(ARMS)):
        rotated = ARMS[rotation:] + ARMS[:rotation]
        assert arm_from_prefix("merlincirct_x", rotated) == "arm4"
        assert arm_from_prefix("merlin_x", rotated) == "arm3"


def test_bundle_beats_prefix_and_the_disagreement_is_reported():
    """An arm IS its grant set, so the bundle decides -- but a conflict is a fact, not noise.

    Real instance: `merlincirct_atlas_operands_v2` carries the arm-3 bundle under the arm-4 prefix.
    Resolving that silently would attribute one arm's result to another."""
    arm, source, conflict = resolve_arm("merlincirct_x", "merlin_assisted_public_v0", ARMS)
    assert (arm, source) == ("arm3", "bundle_id")
    assert "arm3" in conflict and "arm4" in conflict

    arm, source, conflict = resolve_arm("merlincirct_x", "merlin_assisted_rtlchecks_public_v0", ARMS)
    assert (arm, source, conflict) == ("arm4", "bundle_id", "")


def test_unidentifiable_run_is_kept_not_dropped(tmp_path):
    """The denominator is part of the result: a run we cannot classify still counts."""
    run = tmp_path / "tgt" / "capsule-bench" / "grp" / "mystery_run"
    (run / "rounds").mkdir(parents=True)
    (run / "rounds" / "round_00.transcript.jsonl").write_text("{}\n")
    refs = build_index([tmp_path], ARMS, PHASES)
    assert len(refs) == 1
    assert refs[0].arm == UNKNOWN
    assert refs[0].availability.get("arm").kind == UNAVAILABLE
    assert "mystery_run" in refs[0].availability.get("arm").reason


# ---------------------------------------------------------------- spans

def _transcript(path, *, with_ids: bool):
    """Two tool calls, 10 s and 30 s, the second starting before the first ends."""
    rows = [
        {"type": "assistant", "arrived_at": "2026-01-01T00:00:00+00:00",
         "message": {"content": [dict({"type": "tool_use", "name": "bash",
                                       "input": {"command": "sim"}},
                                      **({"id": "a"} if with_ids else {}))]}},
        {"type": "assistant", "arrived_at": "2026-01-01T00:00:05+00:00",
         "message": {"content": [dict({"type": "tool_use", "name": "bash",
                                       "input": {"command": "build"}},
                                      **({"id": "b"} if with_ids else {}))]}},
        {"type": "user", "arrived_at": "2026-01-01T00:00:10+00:00",
         "message": {"content": [{"type": "tool_result", "tool_use_id": "a"}]}},
        {"type": "user", "arrived_at": "2026-01-01T00:00:35+00:00",
         "message": {"content": [{"type": "tool_result", "tool_use_id": "b"}]}},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _raw_items(path):
    """The same two calls as the driver's own item stream."""
    rows = [
        ("2026-01-01T00:00:00+00:00", "item.started", "i1", "command_execution"),
        ("2026-01-01T00:00:05+00:00", "item.started", "i2", "command_execution"),
        ("2026-01-01T00:00:10+00:00", "item.completed", "i1", "command_execution"),
        ("2026-01-01T00:00:35+00:00", "item.completed", "i2", "command_execution"),
    ]
    path.write_text("".join(
        json.dumps({"arrived_at": t, "event": {"type": k, "item": {"id": i, "type": kind,
                                                                   "command": "x"}}}) + "\n"
        for t, k, i, kind in rows))


def test_spans_come_from_the_transcript_when_the_ids_are_there(tmp_path):
    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    _transcript(run / "rounds" / "round_00.transcript.jsonl", with_ids=True)
    ss = read_spans(run)
    assert ss.source == SOURCE_TRANSCRIPT and len(ss.spans) == 2
    assert ss.availability.get("spans").kind == MEASURED


def test_spans_are_recovered_from_the_raw_stream_when_tool_use_has_no_id(tmp_path):
    """The 2026-09-04 defect: tool_use blocks with no id, so the join key exists on one side only.

    Before this fallback, radiance had 49 stamped runs and zero usable ones."""
    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    _transcript(run / "rounds" / "round_00.transcript.jsonl", with_ids=False)
    _raw_items(run / "rounds" / "round_00.codex_events.timestamped.jsonl")
    ss = read_spans(run)
    assert ss.source == SOURCE_RAW_ITEMS and len(ss.spans) == 2
    status = ss.availability.get("spans")
    assert status.kind == DERIVED
    # The reader must SAY the transcript was unusable, and say how unusable.
    assert "no id" in status.reason and "2" in status.reason


def test_spans_refuse_when_neither_stream_can_supply_them(tmp_path):
    """A refusal, not an empty list: an empty list plots as a finished, blank chart."""
    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    _transcript(run / "rounds" / "round_00.transcript.jsonl", with_ids=False)
    ss = read_spans(run)
    assert ss.spans == [] and ss.source == ""
    assert ss.availability.get("spans").kind == UNAVAILABLE
    assert "no id" in ss.availability.get("spans").reason


# ---------------------------------------------------------------- concurrency

def _spanset(pairs, source=SOURCE_TRANSCRIPT):
    spans = [Span(a, b) for a, b in pairs]
    return SpanSet(spans=spans, source=source, wall_s=max(b for _, b in pairs),
                   availability=Availability({"spans": measured(source)}))


def test_concurrency_is_zero_when_the_calls_are_serial():
    c = concurrency(_spanset([(0, 10), (10, 20), (20, 30)]))
    assert c.overlap_s == 0.0 and c.max_concurrent == 1
    assert c.availability.get("concurrency").kind == MEASURED


def test_concurrency_finds_real_overlap():
    """The other direction: a function that always returns 0 must fail here."""
    c = concurrency(_spanset([(0, 30), (5, 10), (5, 35)]))
    assert c.max_concurrent == 3
    # [5,10) is three-deep, [10,30) two-deep, [30,35) back to one: 5 + 20 = 25 s over threshold.
    assert c.overlap_s == pytest.approx(25.0)
    assert c.overlap_share > 0


def test_concurrency_refuses_when_many_long_spans_share_one_end_stamp():
    """The second, independent failure mode. Five long calls that all END at the same instant look
    deeply concurrent, but an end stamp records when the harness READ a completion: a pile of them is
    one flush. Measured once at 8 calls sharing a stamp, putting 169 min of tool time inside 43.6."""
    spans = [(float(i) * 10, 500.0) for i in range(5)]
    c = concurrency(_spanset(spans))
    status = c.availability.get("concurrency")
    assert status.kind == UNAVAILABLE
    assert "share an end stamp" in status.reason


def test_concurrency_survives_when_the_short_spans_do_not_carry_it():
    """Robustness, the publishable case: real long-call overlap that a threshold does not move."""
    tiny = FLUSH_FLOOR_S / 2
    spans = [(0.0, 100.0), (50.0, 150.0)] + [(200.0 + i, 200.0 + i + tiny) for i in range(5)]
    c = concurrency(_spanset(spans, source=SOURCE_RAW_ITEMS))
    assert c.overlap_s == pytest.approx(c.overlap_s_trusted)
    assert c.availability.get("concurrency").kind == DERIVED


def test_concurrency_refuses_without_spans():
    empty = SpanSet(availability=Availability({"spans": unavailable("nothing to read")}))
    c = concurrency(empty)
    assert c.availability.get("concurrency").kind == UNAVAILABLE
    assert c.max_concurrent == 0


# ---------------------------------------------------------------- passes

def test_wall_offset_is_rebased_across_round_resets():
    assert _rebase([10, 20, 30, 5, 15]) == [10.0, 20.0, 30.0, 35.0, 45.0]
    assert _rebase([5, 5, 6]) == [5.0, 5.0, 6.0]          # equal values are not a reset


def _selfcheck(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_rows_without_a_denominator_do_not_drag_the_curve_to_zero(tmp_path):
    """THE must-fail test. A self-check that could not run writes n_capsules=0, n_passed=0. Charted
    as written it puts the curve on the floor and reads as the agent destroying its own work.

    Asserting only that the series is non-decreasing would pass trivially on an empty series, so the
    exclusion must also be COUNTED and the surviving points checked."""
    run = tmp_path / "run"
    run.mkdir()
    _selfcheck(run / "selfcheck_log.jsonl", [
        {"wall_offset_s": 10, "capsules": "all", "n_passed": 5, "n_capsules": 20, "failing": ["x"] * 15},
        {"wall_offset_s": 20, "capsules": "all", "n_passed": 0, "n_capsules": 0,
         "failing": [], "build_failed": True},
        {"wall_offset_s": 30, "capsules": "all", "n_passed": 9, "n_capsules": 20, "failing": ["x"] * 11},
    ])
    s = read_passes(run)
    assert [p.n_passed for p in s.points] == [5, 9]
    assert s.n_no_denominator == 1 and s.n_build_failed == 1
    assert s.n_regressions == 0
    assert s.best == (9, 20)


def test_a_genuine_regression_is_recorded_not_smoothed(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _selfcheck(run / "selfcheck_log.jsonl", [
        {"wall_offset_s": 10, "capsules": "all", "n_passed": 9, "n_capsules": 20, "failing": ["x"] * 11},
        {"wall_offset_s": 20, "capsules": "all", "n_passed": 4, "n_capsules": 20, "failing": ["x"] * 16},
    ])
    s = read_passes(run)
    assert [p.n_passed for p in s.points] == [9, 4]      # kept as measured
    assert s.n_regressions == 1
    assert [p.n_passed for p in s.envelope()] == [9, 9]  # the plot may flatten it; the count remains


def test_a_row_disagreeing_with_its_own_failing_list_is_counted(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _selfcheck(run / "selfcheck_log.jsonl", [
        {"wall_offset_s": 10, "capsules": "all", "n_passed": 5, "n_capsules": 20, "failing": ["x"] * 3},
    ])
    assert read_passes(run).n_inconsistent == 1


def test_passes_refuse_when_the_run_kept_no_progress_record(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    s = read_passes(run)
    assert not s.ok and s.availability.get("passes").kind == UNAVAILABLE


def test_verdict_fallback_is_marked_derived_because_its_clock_is_an_mtime(tmp_path):
    """A continuous-schedule run writes verdicts, not a self-check log. Usable, but the x-axis is
    file metadata -- and several of these runs live in a tree that has been copied."""
    run = tmp_path / "run"
    (run / "qa_history").mkdir(parents=True)
    for i, n in enumerate((3, 7)):
        (run / "qa_history" / f"verdict_round_0{i}.json").write_text(
            json.dumps({"n_passed": n, "n_capsules": 10}))
    s = read_passes(run)
    assert [p.n_passed for p in s.points] == [3, 7]
    status = s.availability.get("passes")
    assert status.kind == DERIVED and "mtime" in status.reason


# ---------------------------------------------------------------- tokens and cost

@pytest.mark.parametrize("raw,expected", [
    ("amazon-bedrock/us.anthropic.claude-opus-4-6-v1", "claude-opus-4-6"),
    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5-20251001"),
    ("us.anthropic.claude-opus-4-8", "claude-opus-4-8"),
    ("zai.glm-5", "glm-5"),
    ("amazon-bedrock/nvidia.nemotron-super-3-120b", "nemotron-super-3-120b"),
    ("gpt-5.6-sol", "gpt-5.6-sol"),
])
def test_model_ids_normalize_structurally(raw, expected):
    """Stacked routing prefixes are stripped repeatedly, so a new deployment needs no edit here."""
    assert normalize_model(raw)[0] == expected


def test_a_bare_family_name_is_flagged_rather_than_resolved():
    """`opus` names no particular Opus. Folding it into one would invent a fact."""
    assert normalize_model("opus") == ("opus", True)
    assert normalize_model("claude-opus-4-8")[1] is False


def _cost_yaml(run, doc):
    import yaml
    (run / "cost_time_toolcalls.yaml").write_text(yaml.safe_dump(doc))


def test_metered_notional_and_unpriced_stay_three_different_things(tmp_path):
    run = tmp_path / "m"
    run.mkdir()
    _cost_yaml(run, {"model": "claude-opus-4-8", "estimated_cost_usd": 12.5, "tokens_total": 10})
    m = read_tokens(run)
    assert (m.cost_kind, m.cost_usd, m.notional_usd) == (METERED, 12.5, None)

    run = tmp_path / "n"
    run.mkdir()
    _cost_yaml(run, {"model": "gpt-5.6-sol", "estimated_cost_usd": None,
                     "subscription_notional_usd": 8.2, "tokens_total": 10,
                     "cost_unavailable_reason": "a seat is not billed per token"})
    n = read_tokens(run)
    assert (n.cost_kind, n.cost_usd, n.notional_usd) == (NOTIONAL, None, 8.2)
    assert n.availability.get("cost").kind == DERIVED

    run = tmp_path / "u"
    run.mkdir()
    _cost_yaml(run, {"model": "who-knows", "tokens_total": 10})
    u = read_tokens(run)
    assert u.cost_kind == UNPRICED
    # The invariant that matters: unpriced is None, never 0.0, or a free run and an unpriceable one
    # become the same row.
    assert u.cost_usd is None and u.notional_usd is None
    assert u.availability.get("cost").kind == UNAVAILABLE


def test_a_notional_run_never_contributes_to_metered_spend(tmp_path):
    runs = []
    for name, doc in (("a", {"estimated_cost_usd": 10.0}), ("b", {"subscription_notional_usd": 99.0}),
                      ("c", {})):
        run = tmp_path / name
        run.mkdir()
        _cost_yaml(run, dict(doc, model="x", tokens_total=1))
        runs.append(read_tokens(run))
    assert sum(r.cost_usd or 0.0 for r in runs) == 10.0
    assert sum(r.notional_usd or 0.0 for r in runs) == 99.0


def test_cache_read_and_write_are_split_when_the_run_recorded_them(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _cost_yaml(run, {"model": "x", "tokens_native_by_model": {
        "x": {"input": 100, "output": 20, "cache_read": 9000, "cache_create": 300, "reasoning": 5}}})
    t = read_tokens(run)
    assert (t.cache_read_tokens, t.cache_creation_tokens) == (9000, 300)
    assert t.availability.get("token_split").kind == MEASURED
    assert t.cached_share == pytest.approx(9000 / 9400)


def test_a_summed_cache_column_declares_that_it_cannot_be_split(tmp_path):
    """Reads and writes are billed an order of magnitude apart; the sum cannot be undone later."""
    run = tmp_path / "run"
    run.mkdir()
    _cost_yaml(run, {"model": "x", "tokens_input": 100, "tokens_output": 20, "tokens_cached": 9300})
    t = read_tokens(run)
    assert t.availability.get("token_split").kind == UNAVAILABLE


def test_tokens_refuse_when_the_run_recorded_none(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _cost_yaml(run, {"model": "x", "available": False, "reason": "no usage metadata in transcript"})
    t = read_tokens(run)
    assert t.total_tokens == 0
    assert t.availability.get("tokens").kind == UNAVAILABLE
    assert "no usage metadata" in t.availability.get("tokens").reason


def test_longest_match_decides_when_one_prefix_extends_another():
    """The separator settles `merlin` vs `merlincirct`; it does NOT settle `merlin` vs `merlin_rtl`.

    Without a length comparison the winner there depends on spec order, which is a silent
    misattribution of one arm's results to another."""
    extended = (ArmSpec("armA", "a", "merlin", ()), ArmSpec("armB", "b", "merlin_rtl", ()))
    for specs in (extended, tuple(reversed(extended))):
        assert arm_from_prefix("merlin_rtl_x", specs) == "armB"
        assert arm_from_prefix("merlin_plain_x", specs) == "armA"


# ---------------------------------------------------------------- per-capsule tier cost

def _capsule(dirpath, name, tiers):
    d = dirpath / name
    d.mkdir(parents=True)
    (d / "capsule_result.json").write_text(json.dumps({"capsule": name, "tiers": tiers}))


def _timing(build, sim, wait, wall):
    return {"build_s": build, "sim_active_s": sim, "oracle_wait_s": wait, "adapter_wall_s": wall}


def test_a_carried_tier_is_recorded_as_carried_and_never_costed(tmp_path):
    """A reused certificate records `timing: null` on purpose -- copying a duration forward would
    fabricate a measurement. It must not become a zero in the distribution."""
    _capsule(tmp_path, "A0", {
        "L2": {"status": "pass", "timing": _timing(0.1, 0.5, 0.0, 0.6)},
        "L3": {"status": "pass", "timing": None,
               "reason": "verdict carried: already certified at this tier on this instrument"}})
    rows = read_capsule_timings(tmp_path)
    l3 = [r for r in rows if r.tier == "L3"][0]
    assert l3.carried is True and l3.has_timing is False and l3.active_s is None

    s = summarize(rows, tier="L3", status="pass")
    assert s.n == 1 and s.n_carried == 1 and s.median_active_s is None
    status = s.availability.get("tier_cost")
    assert status.kind == UNAVAILABLE and "carried" in status.reason


def test_a_prefetched_measurement_is_flagged_and_its_wall_is_not_used(tmp_path):
    """The performance lane measures ahead of the loop, so adapter_wall_s reads an already-computed
    result: sim 52 s against a wall of 0.03 s. Reading that wall as the cost understates it ~2000x."""
    _capsule(tmp_path, "P0", {"L3": {"status": "pass", "timing": _timing(0.0, 52.3, 0.0, 0.026)}})
    rows = read_capsule_timings(tmp_path)
    assert rows[0].wall_is_consistent is False
    assert rows[0].active_s == pytest.approx(52.3)      # sim + build, never the wall
    s = summarize(rows, tier="L3", status="pass")
    assert s.wall_inconsistent == 1
    assert "adapter wall" in s.availability.get("tier_cost").reason


def test_a_consistent_wall_is_not_flagged(tmp_path):
    """The other direction: the functional lane really did wait, and must not be marked suspect."""
    _capsule(tmp_path, "F0", {"L3": {"status": "pass", "timing": _timing(1.4, 30.7, 0.1, 32.2)}})
    rows = read_capsule_timings(tmp_path)
    assert rows[0].wall_is_consistent is True
    assert summarize(rows, tier="L3", status="pass").wall_inconsistent == 0


def test_pass_and_fail_populations_are_never_pooled(tmp_path):
    """A failing capsule aborts in hundredths of a second while a passing one simulates for tens.
    Pooling them yields a median about the pass rate, not about cost."""
    _capsule(tmp_path, "ok1", {"L3": {"status": "pass", "timing": _timing(1.0, 30.0, 0.0, 31.0)}})
    _capsule(tmp_path, "ok2", {"L3": {"status": "pass", "timing": _timing(1.0, 34.0, 0.0, 35.0)}})
    for i in range(6):
        _capsule(tmp_path, f"bad{i}", {"L3": {"status": "fail", "timing": _timing(0.0, 0.01, 0.0, 0.02)}})
    rows = read_capsule_timings(tmp_path)
    passing = summarize(rows, tier="L3", status="pass")
    failing = summarize(rows, tier="L3", status="fail")
    assert passing.median_active_s == pytest.approx(33.0)   # median of 1+30 and 1+34
    assert failing.median_active_s == pytest.approx(0.01)
    # The pooled median would be ~0.01 -- an eight-capsule suite would look 3000x cheaper than it is.
    assert passing.median_active_s > failing.median_active_s * 100


def test_oracle_wait_is_not_counted_as_work(tmp_path):
    """Queueing for a simulator slot is not simulation. It is kept, separately."""
    _capsule(tmp_path, "Q0", {"L3": {"status": "pass", "timing": _timing(1.0, 10.0, 90.0, 101.0)}})
    row = read_capsule_timings(tmp_path)[0]
    assert row.active_s == pytest.approx(11.0)
    assert row.oracle_wait_s == pytest.approx(90.0)


# ---------------------------------------------------------------- the performance lane

def _stage(tmp_path, *, tools=None, receipts=None, actions=None):
    stage = tmp_path / "stage"
    if tools is not None:
        (stage / "agent").mkdir(parents=True)
        (stage / "agent" / "tools.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in tools))
    if receipts is not None:
        (stage / "control" / "round_00").mkdir(parents=True)
        (stage / "control" / "round_00" / "receipts.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in receipts))
    if actions is not None:
        (stage / "agent_workspaces" / "round_00").mkdir(parents=True)
        (stage / "agent_workspaces" / "round_00" / "STAGE_CONTEXT.json").write_text(
            json.dumps({"broker_actions": actions}))
    stage.mkdir(parents=True, exist_ok=True)
    return stage


def test_point_events_are_excluded_from_span_math(tmp_path):
    """`file_change` rows are logged at an instant, not over one. Counting them as spans would put
    99% of the lane's 'tool calls' at zero duration and drag every occupancy figure down."""
    stage = _stage(tmp_path, tools=[
        {"kind": "command_execution", "t_start_s": 0.0, "t_end_s": 10.0, "command": "sim"},
        {"kind": "file_change", "t_start_s": 3.0, "t_end_s": 3.0},
        {"kind": "file_change", "t_start_s": 4.0, "t_end_s": 4.0},
    ])
    facts = read_phase2(stage)
    assert len(facts.spanset.spans) == 1
    assert facts.n_point_events == 2


def test_broker_time_is_attributed_per_action(tmp_path):
    """Where the brokered half of a performance run goes: the measurement, not the compiles."""
    stage = _stage(tmp_path, receipts=[
        {"action": "tuning-gsim-feedback", "elapsed_s": 40.0, "returncode": 0, "index": 0},
        {"action": "tuning-gsim-feedback", "elapsed_s": 60.0, "returncode": 0, "index": 1},
        {"action": "candidate-parse", "elapsed_s": 3.0, "returncode": 0, "index": 2},
        {"action": "analyze-command-buffers", "elapsed_s": 0.002, "returncode": 0, "index": 3},
    ])
    totals = read_phase2(stage).action_totals()
    assert totals["tuning-gsim-feedback"] == (2, pytest.approx(100.0))
    assert totals["analyze-command-buffers"][1] < 0.01     # free by construction


def test_the_declared_tool_surface_is_read_never_assumed(tmp_path):
    """The action set is derived per run from the candidate's own manifest, so a hardcoded list would
    describe one campaign and mislabel the next."""
    stage = _stage(tmp_path, actions=["candidate-parse", "tuning-gsim-feedback", "probe-spike"])
    facts = read_phase2(stage)
    assert facts.broker_actions == ["candidate-parse", "probe-spike", "tuning-gsim-feedback"]
    assert facts.availability.get("broker_actions").kind == MEASURED


def test_a_stage_without_a_stage_context_refuses_to_state_its_tool_surface(tmp_path):
    stage = _stage(tmp_path, tools=[])
    facts = read_phase2(stage)
    assert facts.broker_actions == []
    status = facts.availability.get("broker_actions")
    assert status.kind == UNAVAILABLE and "derived per run" in status.reason


def test_a_serial_performance_stage_reports_no_concurrency(tmp_path):
    """Measured over the real corpus: 4 overlapping pairs in 1,336. The lane is serial, and saying so
    is a finding -- but it must come from the spans, not from an assumption."""
    stage = _stage(tmp_path, tools=[
        {"kind": "command_execution", "t_start_s": 0.0, "t_end_s": 10.0},
        {"kind": "command_execution", "t_start_s": 12.0, "t_end_s": 20.0},
    ])
    c = concurrency(read_phase2(stage).spanset)
    assert c.max_concurrent == 1 and c.overlap_s == 0.0


def test_a_negligible_flush_overlap_is_reported_as_zero_not_as_unknown(tmp_path):
    """A ten-hour run whose only 'overlap' is 0.9 s of flush artifact ran serially. Saying so is a
    measurement; calling it unknown discards a run that demonstrably did not parallelise."""
    tiny = FLUSH_FLOOR_S / 2
    long_run = [(0.0, 36000.0)]
    flush = [(1000.0 + i, 1000.0 + i + tiny) for i in range(9)]
    c = concurrency(_spanset(long_run + flush, source=SOURCE_RAW_ITEMS))
    assert c.overlap_s == 0.0 and c.overlap_share == 0.0
    status = c.availability.get("concurrency")
    assert status.kind == DERIVED and "serially" in status.reason


def test_two_calls_finishing_together_is_a_coincidence_not_a_flush():
    """The guard must not fire on a plausible tie, or every genuinely concurrent run is discarded."""
    c = concurrency(_spanset([(0.0, 100.0), (10.0, 100.0)]))
    assert c.availability.get("concurrency").kind == MEASURED
    assert c.max_concurrent == 2 and c.overlap_s == pytest.approx(90.0)


def test_a_minority_of_tied_spans_yields_the_uncontaminated_figure_not_a_refusal():
    """Discarding a ten-hour run because a handful of its spans share an end stamp throws away a real
    measurement. Report what survives their removal, and say that is what is being reported."""
    real = [(0.0, 1000.0), (100.0, 1000.5)]           # 900 s of genuine overlap
    tied = [(1900.0 + i, 1950.0) for i in range(4)]   # four ends within the tie window, ~50 s worth
    c = concurrency(_spanset(real + tied))
    status = c.availability.get("concurrency")
    assert status.kind == DERIVED
    assert "were excluded as a flush" in status.reason
    assert c.overlap_s > 0                            # the real overlap is still reported


def test_joinable_but_unstamped_events_say_the_clock_is_missing_not_the_key(tmp_path):
    """The two refusals have different fixes and must not share a message.

    One driver's transcripts carry 995 tool_use blocks every one of which has an id, and no time
    field anywhere in the file. That is not recoverable by any reader: the driver has to stamp events
    on arrival. Reporting it as a missing join key would send someone looking for a raw stream that
    would not help."""
    run = tmp_path / "run"
    (run / "rounds").mkdir(parents=True)
    rows = [
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "a", "name": "bash", "input": {"command": "x"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "a"}]}},
    ]
    (run / "rounds" / "round_00.transcript.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    ss = read_spans(run)
    assert ss.spans == []
    reason = ss.availability.get("spans").reason
    assert "no time field" in reason and "stamping events on arrival" in reason
    assert "no id" not in reason


def test_a_stamped_verdict_beats_the_file_mtime(tmp_path):
    """`graded_at` is a property of the RUN; an mtime is a property of the file and does not survive
    a copy between trees — which these run trees have had."""
    run = tmp_path / "run"
    (run / "qa_history").mkdir(parents=True)
    (run / "qa_history" / "verdict_round_00.json").write_text(json.dumps(
        {"n_passed": 3, "n_capsules": 10, "graded_at": "2026-01-01T00:00:00+00:00"}))
    (run / "qa_history" / "verdict_round_01.json").write_text(json.dumps(
        {"n_passed": 8, "n_capsules": 10, "graded_at": "2026-01-01T02:00:00+00:00"}))
    s = read_passes(run)
    assert [p.n_passed for p in s.points] == [3, 8]
    # Two hours apart by the STAMP, whatever the files' mtimes happen to be.
    assert s.points[-1].t_s == pytest.approx(7200.0)
    assert s.availability.get("passes").kind == MEASURED


def test_a_partly_stamped_history_says_which_half_is_metadata(tmp_path):
    run = tmp_path / "run"
    (run / "qa_history").mkdir(parents=True)
    (run / "qa_history" / "verdict_round_00.json").write_text(json.dumps(
        {"n_passed": 1, "n_capsules": 10, "graded_at": "2026-01-01T00:00:00+00:00"}))
    (run / "qa_history" / "verdict_round_01.json").write_text(json.dumps(
        {"n_passed": 4, "n_capsules": 10}))
    status = read_passes(run).availability.get("passes")
    assert status.kind == DERIVED and "mtime" in status.reason and "1 carry" in status.reason


def test_zero_length_spans_do_not_become_an_idle_run():
    """One driver reports a tool only once completed, so the call and its result share one stamp and
    every span has zero length. Those spans carry real ordering and identity — but binning them
    yields a flat zero band that reads as 'this agent ran no tools', and sweeping them reports 'no
    overlap' as though that had been measured. Both must refuse instead."""
    from merlin.agentreport.spans import occupancy_bins
    flat = SpanSet(spans=[Span(1.0, 1.0), Span(2.0, 2.0), Span(3.0, 3.0)],
                   source=SOURCE_TRANSCRIPT, wall_s=3.0,
                   availability=Availability({"spans": measured(SOURCE_TRANSCRIPT)}))
    assert flat.durations_measurable is False
    assert occupancy_bins(flat) == ([], [])
    status = concurrency(flat).availability.get("concurrency")
    assert status.kind == UNAVAILABLE and "zero length" in status.reason


def test_a_set_with_real_durations_is_still_binned():
    """The other direction, so the guard cannot pass by refusing everything."""
    from merlin.agentreport.spans import occupancy_bins
    live = _spanset([(0.0, 30.0), (40.0, 90.0)])
    centres, shares = occupancy_bins(live, bins=9)
    assert len(centres) == 9 and max(shares) > 0
    assert concurrency(live).availability.get("concurrency").kind == MEASURED


def test_rows_graded_against_a_different_suite_are_not_mixed_in(tmp_path):
    """A pilot-slice row and a full-suite row are fractions of different wholes.

    Measured on the corpus: 8 of 82 runs carry more than one denominator, one of them four. Plotting
    them on a single 'capsules passed' axis compares suites rather than progress."""
    run = tmp_path / "run"
    run.mkdir()
    _selfcheck(run / "selfcheck_log.jsonl", [
        {"wall_offset_s": 10, "capsules": "all", "n_passed": 5, "n_capsules": 10, "failing": ["x"] * 5},
        {"wall_offset_s": 20, "capsules": "all", "n_passed": 13, "n_capsules": 36, "failing": ["x"] * 23},
        {"wall_offset_s": 30, "capsules": "all", "n_passed": 20, "n_capsules": 36, "failing": ["x"] * 16},
        {"wall_offset_s": 40, "capsules": "all", "n_passed": 31, "n_capsules": 36, "failing": ["x"] * 5},
    ])
    s = read_passes(run)
    assert s.suite_size == 36
    assert [p.n_passed for p in s.points] == [13, 20, 31]
    assert s.n_other_suite == 1
    assert "different suite size" in s.availability.get("passes").reason


def test_the_suite_size_is_the_mode_not_a_literal(tmp_path):
    """An eleven-capsule suite must read exactly as well as a twenty-capsule one."""
    run = tmp_path / "run"
    run.mkdir()
    _selfcheck(run / "selfcheck_log.jsonl", [
        {"wall_offset_s": 10, "capsules": "all", "n_passed": 2, "n_capsules": 11, "failing": ["x"] * 9},
        {"wall_offset_s": 20, "capsules": "all", "n_passed": 9, "n_capsules": 11, "failing": ["x"] * 2},
    ])
    s = read_passes(run)
    assert s.suite_size == 11 and [p.n_passed for p in s.points] == [2, 9]
