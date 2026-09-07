"""A partial self-check must not report a complete one, and re-checking failures must be easy.

These two belong together. Re-checking only what is broken was always possible via the
comma-separated form, but it required reading a verdict and pasting names, so the reflex was
`--capsules all` -- 25 full sweeps at ~6 min each in one 6.1 h run, while a typical edit leaves 82% of
emitted programs byte-identical. Making the cheap path easy is only safe once a partial verdict says
what it is: checking two capsules of ninety-six and passing both reported `all_pass: true` beside a
note whose first sentence defines "done".
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import merlin_dir

SRC = merlin_dir() / "experiments" / "capsule_bench" / "harness" / "agent_selfcheck.py"


@pytest.fixture
def mod(tmp_path, monkeypatch):
    """The self-check module, with its capsule corpus pointed at a fixture and cwd in a temp dir."""
    corpus = tmp_path / "public"
    for name in ("A0", "A1", "B0", "B1"):
        (corpus / name).mkdir(parents=True)
        (corpus / name / "capsule.yaml").write_text("name: " + name)
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("agent_selfcheck_under_test", SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("agent_selfcheck_under_test", m)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "PUBLIC_CAPSULES", corpus)
    return m


# --------------------------------------------------------------------------------------------
# how big the suite is
# --------------------------------------------------------------------------------------------

def test_the_suite_size_is_counted_from_the_corpus(mod):
    assert mod._suite_size() == 4


def test_an_unreadable_corpus_reports_zero_not_a_guess(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "PUBLIC_CAPSULES", tmp_path / "absent")
    assert mod._suite_size() == 0


# --------------------------------------------------------------------------------------------
# --capsules failing
# --------------------------------------------------------------------------------------------

def _prev(tmp_path, rows):
    out = tmp_path / "selfcheck_out"
    out.mkdir(exist_ok=True)
    (out / "last.json").write_text(json.dumps({"per_capsule": rows}))


def test_failing_reads_the_previous_self_check(mod, tmp_path):
    _prev(tmp_path, [{"capsule": "A0", "pass": True}, {"capsule": "A1", "pass": False},
                     {"capsule": "B0", "pass": False}])
    names, why = mod._previously_failing()
    assert names == {"A1", "B0"} and why == ""


def test_with_no_previous_run_it_refuses_rather_than_grading_nothing(mod):
    """The paired direction. An empty set would grade ZERO capsules and report success -- a check that
    cannot fail, which is this repo's most-repeated defect."""
    names, why = mod._previously_failing()
    assert names == set() and "no previous self-check" in why


def test_when_everything_passed_it_says_so_instead_of_returning_empty(mod, tmp_path):
    _prev(tmp_path, [{"capsule": "A0", "pass": True}, {"capsule": "A1", "pass": True}])
    names, why = mod._previously_failing()
    assert names == set() and "nothing to re-check" in why and "--capsules all" in why


def test_an_unreadable_previous_run_refuses(mod, tmp_path):
    out = tmp_path / "selfcheck_out"
    out.mkdir(exist_ok=True)
    (out / "last.json").write_text("{ truncated")
    names, why = mod._previously_failing()
    assert names == set() and "could not be read" in why


def test_a_previous_run_with_no_rows_refuses(mod, tmp_path):
    _prev(tmp_path, [])
    names, why = mod._previously_failing()
    assert names == set() and "no per-capsule rows" in why


def test_a_declined_capsule_counts_as_failing(mod, tmp_path):
    """A decline is a shape the backend never lowered -- exactly what the next iteration targets."""
    _prev(tmp_path, [{"capsule": "A0", "pass": False, "declined": True}])
    names, _ = mod._previously_failing()
    assert names == {"A0"}


def test_the_flag_is_documented_where_the_agent_reads_it(mod):
    import argparse
    ap = argparse.ArgumentParser()
    # mirror the real declaration by scraping it, so the help text cannot silently drop the mode
    text = SRC.read_text()
    i = text.index('ap.add_argument("--capsules"')
    decl = text[i:i + 400]
    assert "'failing'" in decl, "the mode exists but is not offered in --help"
    assert "fast iteration" in decl


# --------------------------------------------------------------------------------------------
# what a partial verdict may claim
# --------------------------------------------------------------------------------------------

def test_a_subset_check_cannot_report_completion():
    """`certified_complete` is the field an agent may key "done" on, and a subset must never set it."""
    for scope, ncert, n, expected in (("all", 4, 4, True), ("subset", 2, 2, False),
                                      ("all", 3, 4, False), ("all", 0, 0, False)):
        assert bool(scope == "all" and ncert == n and n > 0) is expected, (scope, ncert, n)


def test_the_partial_warning_leads_the_note_and_names_the_unchecked_count():
    """It has to be the FIRST thing read: the sentence it precedes defines "done"."""
    text = SRC.read_text()
    i = text.index('"note": ((f"⚠ PARTIAL:')
    note = text[i:i + 700]
    assert "were NOT checked" in note and "UNKNOWN, not" in note
    assert "certified_complete" in note, "the note must point at the field that means done"
    assert "can also BREAK a capsule you did not check" in note, (
        "a partial re-check can miss a regression it caused; the note must say so")
    assert note.index("PARTIAL") < note.index("Self-check on"), "the warning must come first"


def test_all_pass_keeps_its_meaning():
    """Left alone deliberately: existing readers key on it, and changing what it means silently is how
    a fix becomes a second defect. The new field is additive."""
    text = SRC.read_text()
    assert '"all_pass": ncert == n and n > 0' in text
    assert '"certified_complete": bool(scope == "all" and ncert == n and n > 0)' in text


def test_the_suite_size_is_read_from_the_corpus_not_from_the_subset():
    """A wiring check, and it earns its keep: taking `suite_size = n` makes every partial check claim
    it covered the whole suite, and the PARTIAL warning then reads "you checked 2 of 2". The helper is
    tested above; this pins that the caller actually asks it."""
    text = SRC.read_text()
    assert "suite_size = _suite_size()" in text, (
        "suite_size must come from the corpus; deriving it from the checked subset makes a partial "
        "check indistinguishable from a complete one")
    assert "suite_size = n" not in text
