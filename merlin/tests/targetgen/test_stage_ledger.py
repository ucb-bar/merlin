"""The ledger must separate an INERT edit from a WRONG one — the case that cost six rounds unread."""

from __future__ import annotations

import json

from merlin.common.provenance import UNKNOWN
from merlin.targetgen import stage_ledger as SL


def _tree(root, files: dict[str, str]):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def _round(tmp_path, name, sub: dict[str, str], emitted: dict[str, dict[str, str]], prev=None):
    """One round: a graded submission copy + per-capsule emitted artifacts, as the loop lays them out."""
    base = tmp_path / name
    subdir = _tree(base / "submission", sub)
    roots = {cap: _tree(base / "emitted" / cap, files) for cap, files in emitted.items()}
    return SL.build(submission_dir=subdir, emitted_roots=roots, previous=prev)


def test_first_round_has_nothing_to_compare(tmp_path):
    led = _round(tmp_path, "r0", {"pipeline.py": "v1"}, {"AF6": {"cb.json": "{}"}})
    assert led["diagnosis"] == SL.NOTHING_TO_COMPARE
    assert led["submission_moved"] is None
    assert set(led["submission_verdicts"].values()) == {SL.NEW}


def test_edit_that_changes_nothing_is_no_submission_change(tmp_path):
    r0 = _round(tmp_path, "r0", {"pipeline.py": "v1"}, {"AF6": {"cb.json": "{}"}})
    r1 = _round(tmp_path, "r1", {"pipeline.py": "v1"}, {"AF6": {"cb.json": "{}"}}, prev=r0)
    assert r1["diagnosis"] == SL.NO_SUBMISSION_CHANGE
    assert r1["submission_moved"] is False


def test_the_six_round_case_edit_lands_but_emission_is_identical(tmp_path):
    """The real failure: the agent edits its emitter every round and every emitted artifact is
    byte-identical, so no numeric verdict CAN move. This is the line that was never printed."""
    r0 = _round(tmp_path, "r0", {"pipeline.py": "v1"}, {"AF6": {"cb.json": "{}"}, "AT7": {"cb.json": "[]"}})
    r1 = _round(tmp_path, "r1",
                {"pipeline.py": "v2 — a real edit"},                      # submission DID change
                {"AF6": {"cb.json": "{}"}, "AT7": {"cb.json": "[]"}},     # emission did NOT
                prev=r0)
    assert r1["submission_moved"] is True
    assert r1["emit_moved"] is False
    assert r1["diagnosis"] == SL.EMIT_INSENSITIVE_TO_EDIT
    assert SL.capsules_that_did_not_move(r1) == ["AF6", "AT7"]


def test_one_byte_of_emitted_change_is_emit_moved(tmp_path):
    r0 = _round(tmp_path, "r0", {"pipeline.py": "v1"}, {"AF6": {"cb.json": "{}"}, "AT7": {"cb.json": "[]"}})
    r1 = _round(tmp_path, "r1", {"pipeline.py": "v2"},
                {"AF6": {"cb.json": "{ }"}, "AT7": {"cb.json": "[]"}}, prev=r0)
    assert r1["diagnosis"] == SL.EMIT_MOVED
    assert r1["capsules"]["AF6"]["verdicts"]["cb.json"] == SL.CHANGED
    # the capsule that really did not move is still named, so a partial move is not read as a full one
    assert SL.capsules_that_did_not_move(r1) == ["AT7"]


def test_emission_that_stops_producing_a_file_is_absent_not_silence(tmp_path):
    r0 = _round(tmp_path, "r0", {"p.py": "v1"}, {"AF6": {"cb.json": "{}", "kernel.S": ".word 0"}})
    r1 = _round(tmp_path, "r1", {"p.py": "v2"}, {"AF6": {"cb.json": "{}"}}, prev=r0)
    assert r1["capsules"]["AF6"]["verdicts"]["kernel.S"] == SL.ABSENT
    assert r1["emit_moved"] is True          # losing an artifact IS movement, and must be visible


def test_scratch_dirs_do_not_manufacture_movement(tmp_path):
    """A __pycache__ rewrite on every import would report CHANGED forever and destroy the signal."""
    r0 = _round(tmp_path, "r0", {"p.py": "v1", "__pycache__/p.pyc": "a"}, {"AF6": {"cb.json": "{}"}})
    r1 = _round(tmp_path, "r1", {"p.py": "v1", "__pycache__/p.pyc": "DIFFERENT"}, {"AF6": {"cb.json": "{}"}},
                prev=r0)
    assert all("__pycache__" not in k for k in r1["submission_files"])
    assert r1["diagnosis"] == SL.NO_SUBMISSION_CHANGE


def test_unreadable_is_recorded_not_skipped(tmp_path):
    base = tmp_path / "r0"
    sub = _tree(base / "submission", {"p.py": "v1"})
    bad = sub / "locked.bin"
    bad.write_text("x")
    bad.chmod(0o000)
    try:
        led = SL.build(submission_dir=sub, emitted_roots={}, previous=None)
        # fail closed: surfaced with a count, never dropped from the fingerprint set
        assert led["submission_files"]["locked.bin"] == UNKNOWN
        assert led["n_unreadable"] == 1
    finally:
        bad.chmod(0o600)


def test_an_unchanged_unknown_never_reads_as_unchanged(tmp_path):
    """Two UNKNOWNs are not evidence of sameness — that would call an unreadable artifact stable."""
    v = SL.compare({"a": UNKNOWN}, {"a": UNKNOWN})
    assert v["a"] == SL.CHANGED


def test_ledger_is_json_serializable_and_summarizes(tmp_path):
    r0 = _round(tmp_path, "r0", {"p.py": "v1"}, {"AF6": {"cb.json": "{}"}})
    r1 = _round(tmp_path, "r1", {"p.py": "v2"}, {"AF6": {"cb.json": "{}"}}, prev=r0)
    json.dumps(r1)                                     # lands in the run dir as JSON
    line = SL.summarize(r1)
    assert "emit_insensitive_to_edit" in line and "unmoved=1" in line


def test_no_target_name_or_regex_in_the_module():
    """The cardinal rule: this must stay a byte comparator, with no target fact and no regex."""
    from merlin.common.paths import repo_root
    src = (repo_root() / "merlin/python/merlin/targetgen/stage_ledger.py").read_text()
    assert "import re" not in src and "from re " not in src
    for name in ("gemmini", "atlas", "radiance", "saturn", "muon"):
        assert name not in src.lower(), f"target name {name!r} leaked into shared code"


def test_run_level_boolean_hides_a_plateau_but_failing_and_frozen_does_not(tmp_path):
    """The replayed real case: SOME capsule moves every round, so a run-level flag says 'moved' while
    the failing set is frozen. The join with the verdict is what names the actionable class."""
    r0 = _round(tmp_path, "r0", {"p.py": "v1"},
                {"AF6": {"cb": "x"}, "AF7": {"cb": "y"}, "AT7": {"cb": "z"}})
    r1 = _round(tmp_path, "r1", {"p.py": "v2"},
                {"AF6": {"cb": "x"}, "AF7": {"cb": "y"}, "AT7": {"cb": "MOVED"}},   # only the PASSING one
                prev=r0)
    assert r1["emit_moved"] is True                     # run-level: "something moved" -- misleading
    assert SL.frozen_fraction(r1) == 2 / 3
    verdict = {"per_capsule": [{"capsule": "AF6", "status": "fail"},
                               {"capsule": "AF7", "status": "fail"},
                               {"capsule": "AT7", "status": "pass"}]}
    assert SL.failing_and_frozen(r1, verdict) == ["AF6", "AF7"]
    assert "frozen=67%" in SL.summarize(r1)


def test_failing_and_frozen_is_empty_when_the_failing_set_moved(tmp_path):
    r0 = _round(tmp_path, "r0", {"p.py": "v1"}, {"AF6": {"cb": "x"}, "AT7": {"cb": "z"}})
    r1 = _round(tmp_path, "r1", {"p.py": "v2"}, {"AF6": {"cb": "MOVED"}, "AT7": {"cb": "z"}}, prev=r0)
    verdict = {"per_capsule": [{"capsule": "AF6", "status": "fail"},
                               {"capsule": "AT7", "status": "pass"}]}
    assert SL.failing_and_frozen(r1, verdict) == []     # it moved and was still wrong -- a real attempt


def test_gated_capsules_are_not_counted_as_failing(tmp_path):
    """A whole-model capsule deferred by its op-pass gate never emitted, so it is not 'failing and
    frozen' -- calling it that would point the agent at work it cannot do yet."""
    r0 = _round(tmp_path, "r0", {"p.py": "v1"}, {"M0": {"cb": "x"}})
    r1 = _round(tmp_path, "r1", {"p.py": "v2"}, {"M0": {"cb": "x"}}, prev=r0)
    verdict = {"per_capsule": [{"capsule": "M0", "status": "gated"}]}
    assert SL.failing_and_frozen(r1, verdict) == []


def test_an_unrecognized_status_counts_as_an_attempt(tmp_path):
    """Fail closed: a status nobody anticipated must be surfaced, not silently treated as a non-attempt."""
    r0 = _round(tmp_path, "r0", {"p.py": "v1"}, {"X0": {"cb": "x"}})
    r1 = _round(tmp_path, "r1", {"p.py": "v2"}, {"X0": {"cb": "x"}}, prev=r0)
    verdict = {"per_capsule": [{"capsule": "X0", "status": "some_new_failure_mode"}]}
    assert SL.failing_and_frozen(r1, verdict) == ["X0"]
