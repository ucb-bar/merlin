"""The liveness screen's verdict must reach the agent, and nothing answer-bearing may ride with it.

`capsule_runner` runs an advisory L2.5 silicon-liveness screen on every emitted stream and writes
`generated/liveness_report.json` beside the result. MEASURED across 3446 reports on disk: 313 `stall`
verdicts and 229 `scratchpad-overflow` findings -- a program the screen had already judged unable to make
progress on silicon. The redacted verdict row carried none of it, so the author saw an opaque cert
failure where a rule name existed.

These tests pin both halves, the way `test_verdict_carries_emitted_cost` does: the signal arrives (rule
names, severities, counts, the derived DRAM window), and the allowlist drops everything else -- no
message, no `where`, no `evidence`, and no provenance sentence echoed out of a file that lives inside the
agent's own work tree.

They ALSO pin that this is SURFACED, NOT GATED. `epilogue_applicability` went advisory-to-gating and
instantly failed 10 capsules on a plane the other arms had never been assessed on, invalidating the
cross-arm comparison; `REFUSING_SEVERITIES` exists so a future caller can make that decision explicitly,
and no caller does.
"""
from __future__ import annotations

import json
import sys

from merlin.common.paths import merlin_dir
from merlin.targetgen.dram_facts import dram_window_for

sys.path.insert(0, str(merlin_dir() / "experiments/capsule_bench/harness"))
import qa_check as Q  # noqa: E402

CARD_TARGET = "atlas"          # ships a memory-map card with both DRAM addresses
NO_CARD_TARGET = "gemmini"     # ships no memory map at all

REPORT = {
    "target": CARD_TARGET,
    "program": "SY_x",
    "verdict": "stall",
    "findings": [
        {"rule": "scratchpad-overflow", "severity": "stall",
         "message": "resident scratchpad footprint reaches row 2147483664",
         "where": "MVIN #7", "evidence": {"max_row": 2147483664, "capacity_rows": 16384}},
        {"rule": "scratchpad-overflow", "severity": "stall", "message": "again"},
        {"rule": "dram-window-unknown", "severity": "unknown", "message": "no upper bound"},
        {"rule": "visibility-no-drain", "severity": "warn", "message": "no closing FENCE"},
    ],
    "resource_peaks": {"dram_movements": 18624,
                       "dram_window_bytes": dram_window_for(CARD_TARGET)[1]},
}


def _stage(tmp_path, report=None, *, write_report=True, raw=None):
    cap = tmp_path / "runs" / "suite" / "SY_x"
    (cap / "generated").mkdir(parents=True)
    (cap / "capsule_result.json").write_text(json.dumps({"capsule": "SY_x", "status": "pass"}))
    if raw is not None:
        (cap / "generated" / "liveness_report.json").write_text(raw)
    elif write_report:
        (cap / "generated" / "liveness_report.json").write_text(
            json.dumps(REPORT if report is None else report))
    return cap / "capsule_result.json"


# --- the signal arrives -----------------------------------------------------------------------------

def test_the_verdict_reaches_the_row(tmp_path):
    got = Q._liveness_screen(_stage(tmp_path))
    assert got is not None
    assert got["verdict"] == "stall", "the one word that says the program cannot make progress"


def test_rule_names_are_counted_per_severity(tmp_path):
    got = Q._liveness_screen(_stage(tmp_path))
    assert got["rules"]["stall"] == {"scratchpad-overflow": 2}, \
        "the rule that already identified the defect, and how many times it fired"
    assert got["rules"]["unknown"] == {"dram-window-unknown": 1}
    assert got["rules"]["warn"] == {"visibility-no-drain": 1}


def test_the_derived_dram_window_rides_along_with_its_provenance(tmp_path):
    got = Q._liveness_screen(_stage(tmp_path))
    assert got["dram_window_bytes"] == 32 * 1024 ** 3
    assert got["dram_window_provenance"] == dram_window_for(CARD_TARGET)[2]


def test_the_provenance_is_derived_here_not_echoed_from_the_report(tmp_path):
    """That file sits in the agent's own tree. A size it invents must not buy it a provenance string."""
    hostile = json.loads(json.dumps(REPORT))
    hostile["resource_peaks"]["dram_window_bytes"] = 12345          # not the derived size
    hostile["resource_peaks"]["dram_window_provenance"] = "GOLDEN=[1,2,3]"
    got = Q._liveness_screen(_stage(tmp_path, hostile))
    assert got["dram_window_bytes"] == 12345, "the reported size is a statistic of its own program"
    assert "dram_window_provenance" not in got, \
        "a provenance that does not match this repo's derivation must be dropped"
    assert "GOLDEN" not in json.dumps(got)


def test_a_crafted_target_name_never_reaches_the_path_lookup(tmp_path, monkeypatch):
    """The target name is read from the same file, and it selects a DESCRIPTOR PATH.

    So it is constrained to a bare identifier before anything is looked up -- asserted at the call, not
    at the output, because a traversal that resolves to nothing looks identical to one that was refused.
    """
    import merlin.targetgen.dram_facts as DF

    calls: list[str] = []

    def _spy(target):
        calls.append(target)
        return (0, None, "spy")

    monkeypatch.setattr(DF, "dram_window_for", _spy)
    hostile = json.loads(json.dumps(REPORT))
    hostile["target"] = "../../../contract/capsules"
    got = Q._liveness_screen(_stage(tmp_path, hostile))
    assert calls == [], f"a non-identifier target name reached the descriptor lookup: {calls}"
    assert "dram_window_provenance" not in got
    assert got["dram_window_bytes"] == 32 * 1024 ** 3, "the program's own statistic still rides"


def test_a_plain_target_name_does_reach_the_derivation(tmp_path, monkeypatch):
    """The guard must not be a blanket refusal -- the ordinary case still derives."""
    import merlin.targetgen.dram_facts as DF

    calls: list[str] = []
    monkeypatch.setattr(DF, "dram_window_for",
                        lambda t: (calls.append(t), (0, 32 * 1024 ** 3, "spy-why"))[1])
    got = Q._liveness_screen(_stage(tmp_path))
    assert calls == [CARD_TARGET]
    assert got["dram_window_provenance"] == "spy-why"


def test_a_target_with_no_memory_map_gets_no_window_field(tmp_path):
    """UNKNOWN stays UNKNOWN: no default window is manufactured for a target that ships no map."""
    rep = json.loads(json.dumps(REPORT))
    rep["target"] = NO_CARD_TARGET
    rep["resource_peaks"]["dram_window_bytes"] = None
    got = Q._liveness_screen(_stage(tmp_path, rep))
    assert "dram_window_bytes" not in got and "dram_window_provenance" not in got
    assert got["rules"]["unknown"] == {"dram-window-unknown": 1}, "the gap is still surfaced"


def test_the_field_is_on_the_redacted_row_beside_emitted_cost(tmp_path):
    """The reader that builds the row must actually call it -- a helper nobody calls is silence."""
    src = (merlin_dir() / "experiments/capsule_bench/harness/qa_check.py").read_text()
    assert '"liveness": _liveness_screen(cr),' in src
    assert '"emitted_cost": _emitted_cost(cr),' in src


# --- nothing else survives --------------------------------------------------------------------------

def test_no_message_where_or_evidence_can_ride_in(tmp_path):
    got = Q._liveness_screen(_stage(tmp_path))
    blob = json.dumps(got)
    for banned in ("message", "where", "evidence", "2147483664", "MVIN #7", "FENCE"):
        assert banned not in blob, f"answer-bearing/free-form content survived: {banned!r} in {blob}"


def test_the_allowlist_drops_an_unexpected_key(tmp_path):
    hostile = json.loads(json.dumps(REPORT))
    hostile["expected_outputs"] = [1, 2, 3]
    hostile["golden"] = {"values": [4, 5]}
    hostile["reference_outputs"] = "secret"
    hostile["resource_peaks"]["golden"] = [7, 8]
    got = Q._liveness_screen(_stage(tmp_path, hostile))
    assert set(got) <= {"verdict", "rules", "dram_window_bytes", "dram_window_provenance"}, \
        f"unexpected field survived: {sorted(got)}"
    assert "secret" not in json.dumps(got)


def test_an_unrecognised_rule_slug_is_bucketed_not_echoed(tmp_path):
    rep = {"target": CARD_TARGET, "verdict": "fault", "findings": [
        {"rule": "expected=[3,1,4,1,5]", "severity": "fault", "message": "x"}]}
    got = Q._liveness_screen(_stage(tmp_path, rep))
    assert got["rules"] == {"fault": {"other": 1}}, \
        "only slugs this repo defines may be named; anything else is counted, never echoed"


def test_a_bogus_severity_or_verdict_is_dropped(tmp_path):
    rep = {"target": CARD_TARGET, "verdict": "golden=[1,2]", "findings": [
        {"rule": "scratchpad-overflow", "severity": "expected=7", "message": "x"}]}
    assert Q._liveness_screen(_stage(tmp_path, rep)) is None


def test_absence_is_absence_not_a_crash(tmp_path):
    assert Q._liveness_screen(_stage(tmp_path, write_report=False)) is None


def test_an_unreadable_report_is_tolerated(tmp_path):
    assert Q._liveness_screen(_stage(tmp_path, raw="{not json")) is None


def test_a_non_dict_report_is_tolerated(tmp_path):
    assert Q._liveness_screen(_stage(tmp_path, raw="[1, 2, 3]")) is None


# --- surfaced, not gated ----------------------------------------------------------------------------

def test_the_fatal_set_exists_and_names_unknown(tmp_path):
    assert Q.REFUSING_SEVERITIES == frozenset({"fault", "stall", "unknown"})
    assert Q.REFUSING_SEVERITIES <= set(Q.LIVENESS_SEVERITIES)


def test_nothing_gates_on_the_fatal_set():
    """SURFACE, do not gate. If this ever becomes a gate it must be a reviewed decision, not a diff."""
    harness = merlin_dir() / "experiments" / "capsule_bench" / "harness"
    users = []
    for path in sorted(harness.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if "REFUSING_SEVERITIES" not in line:
                continue
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("REFUSING_SEVERITIES"):
                continue          # the definition itself, or a comment about it
            users.append(f"{path.name}:{lineno}: {stripped}")
    assert not users, f"the liveness screen must stay advisory; found consumers: {users}"


def test_the_liveness_field_never_changes_a_capsule_status(tmp_path):
    """The row is a report of the result, not an input to it."""
    cr = _stage(tmp_path)
    result = json.loads(cr.read_text())
    assert result["status"] == "pass"
    got = Q._liveness_screen(cr)
    assert got["verdict"] == "stall"
    assert json.loads(cr.read_text())["status"] == "pass", \
        "reading the screen must not rewrite the capsule result"
