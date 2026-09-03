"""How a result is allowed to SPELL the hardware revision it was measured on.

WHY THIS EXISTS. A bare 40-character sha reads as "pinned" whether or not the bytes that were read are
that commit's bytes, and this repo shipped both shapes under the same spelling: a perf record whose
``toolchain_shas`` named one revision while the RTL source that was actually elaborated carried an
uncommitted edit, and a capability surface whose ISA header was a different revision's entirely. Neither
record was wrong about the sha; both were wrong about what the sha meant.

``provenance.citation`` is the one place that knows the three forms, so these tests are about the
FORMS -- one positive case per state, each asserting the string a claim would carry, plus the registry
fact that a declared local edit stays declared.
"""
from __future__ import annotations

import textwrap

import pytest

from merlin.common import provenance as P

PINNED_SHA = "1111111111111111111111111111111111111111"
EDIT_DIGEST = "a" * 64
PIN_DIGEST = "b" * 64


def _registry(tmp_path):
    """A two-pin registry on disk: a coarse pin that COVERS a nested one, as the real one is shaped."""
    p = tmp_path / "pins.yaml"
    p.write_text(textwrap.dedent(f"""
        version: 1
        pins:
          outer:
            commit: "{PINNED_SHA}"
            root_env: SOME_ROOT
            path: gen/outer
            requires_paths: [src/a.scala]
            covers: [inner]
          inner:
            commit: "{'2' * 40}"
            root_env: SOME_ROOT
            path: gen/outer/sw
            requires_paths: [include/h.h]
        """).lstrip(), encoding="utf-8")
    return p


def _verification(*, sources=(), covered=(), commit=PINNED_SHA, present=True, pin_name="outer",
                  drift=()):
    return P.Verification(
        pin=pin_name,
        observed=P.Observation(path="/nowhere", commit=commit, present=present),
        drift=tuple(drift), sources=tuple(sources), covered=tuple(covered))


def _status(rel, status, *, digest=EDIT_DIGEST, reason="because"):
    return P.SourceStatus(pin="outer", rel=rel, status=status, digest=digest,
                          pinned_digest=PIN_DIGEST, reason=reason)


def test_all_read_paths_pinned_cites_the_bare_commit(tmp_path):
    got = _verification(sources=[_status("src/a.scala", P.PINNED, digest=PIN_DIGEST)])
    cited = P.citation(got, path=_registry(tmp_path))
    assert cited == f"outer {PINNED_SHA} (pinned)"


def test_a_reviewed_off_pin_file_is_cited_as_commit_plus_bytes(tmp_path):
    """THE POSITIVE CASE THIS FILE EXISTS FOR: an off-pin read path must not be spelled 'pinned'."""
    got = _verification(sources=[_status("src/a.scala", P.OFF_PIN)])
    cited = P.citation(got, path=_registry(tmp_path))
    assert cited == f"outer {PINNED_SHA} plus these bytes: src/a.scala@{EDIT_DIGEST[:16]}"
    assert "(pinned)" not in cited
    # The digest of the bytes that were READ has to be in the string: "plus these bytes" without saying
    # WHICH bytes names no revision at all, which is the ambiguity the whole registry exists to remove.
    assert EDIT_DIGEST[:16] in cited


def test_an_undeterminable_read_path_refuses_publication(tmp_path):
    got = _verification(sources=[_status("src/a.scala", P.UNDETERMINABLE, reason="no object store")])
    cited = P.citation(got, path=_registry(tmp_path))
    assert "UNDETERMINABLE" in cited and "do not publish" in cited
    assert "(pinned)" not in cited and "plus these bytes" not in cited


def test_an_unverified_read_set_is_not_citable_as_pinned(tmp_path):
    """No content verdicts at all is a THIRD thing, and it must not read as a clean citation."""
    cited = P.citation(_verification(), path=_registry(tmp_path))
    assert "not citable as pinned" in cited
    assert "(pinned)" not in cited


def test_an_absent_checkout_cites_nothing(tmp_path):
    cited = P.citation(_verification(present=False, commit=""), path=_registry(tmp_path))
    assert "CHECKOUT ABSENT" in cited


def test_a_checkout_off_its_pin_says_which_revision_it_is_actually_on(tmp_path):
    got = _verification(commit="3" * 40, sources=[_status("src/a.scala", P.PINNED, digest=PIN_DIGEST)])
    cited = P.citation(got, path=_registry(tmp_path))
    assert "checkout HEAD is 333333333333" in cited


def test_a_covered_pin_is_folded_into_the_citation(tmp_path):
    """A nested surface a claim rests on may not drop out of the sentence the claim is written in."""
    inner = _verification(pin_name="inner", commit="2" * 40,
                          sources=[_status("include/h.h", P.OFF_PIN)])
    got = _verification(sources=[_status("src/a.scala", P.PINNED, digest=PIN_DIGEST)], covered=[inner])
    cited = P.citation(got, path=_registry(tmp_path))
    assert cited.startswith(f"outer {PINNED_SHA} (pinned); ")
    assert "inner" in cited and "plus these bytes: include/h.h@" in cited


def test_record_embeds_the_citation_beside_the_verification():
    """Every artifact that records a pin gets the spelling for free; no caller has to remember.

    Emitting it in `record` rather than at each call site is the point: the callers that got this wrong
    got it wrong by omission, and an opt-in field is omitted by exactly the reports that need it.
    """
    # `ok` is driven by drift, not by the per-file verdicts, so a real off-pin verification carries
    # both -- mirrored here, because a test that let the two disagree would assert nothing.
    got = _verification(sources=[_status("src/a.scala", P.OFF_PIN)],
                        drift=["src/a.scala is OFF-PIN"])
    rec = P.record(pins={"outer": got})
    assert rec["all_pins_ok"] is False
    assert "plus these bytes: src/a.scala@" in rec["pin_citations"]["outer"]
    assert "(pinned)" not in rec["pin_citations"]["outer"]
    # No pins at all must not fabricate a citation, for the same reason `all_pins_ok` stays None.
    assert P.record()["pin_citations"] == {} and P.record()["all_pins_ok"] is None


def test_pins_declaring_a_local_edit_keep_it_declared():
    """Registry-level regression guard, and it needs no external checkout.

    Every path a pin declares a ``local_edits`` digest for must also be in the set that pin VERIFIES --
    otherwise the reviewed digest is never compared to anything and the declaration is decoration. This
    is exactly how an elaborated-but-unread RTL source stayed invisible: the file was dirty, nothing
    declared it, and `verify` reported "none of them a source this reads".
    """
    for name, declared in P.load_pins().items():
        for rel, digest in declared.local_edits:
            assert len(digest) == 64, f"{name}: local_edits[{rel}] is not a sha256"
            assert rel in declared.requires_paths, (
                f"{name}: declares a local edit for {rel!r} but does not read it, so the digest is "
                "never compared and the declaration means nothing")
            if declared.nested_in:
                # A nested checkout's HEAD can itself be off the pin, and then `git status` answers a
                # question with the wrong subject: a file can be clean-vs-HEAD and still be the wrong
                # revision's bytes. Only a content check is right in both directions there.
                assert declared.checks_content, (
                    f"{name}: is nested and declares a local edit for {rel!r} but does not check "
                    "content, so the edit is compared against a HEAD that may itself be off the pin")


@pytest.mark.parametrize("form", ["(pinned)", "plus these bytes", "UNDETERMINABLE"])
def test_the_three_forms_are_mutually_exclusive(tmp_path, form):
    """One state per clause. A citation that carried two of these would license the wrong reading."""
    by_form = {
        "(pinned)": _status("src/a.scala", P.PINNED, digest=PIN_DIGEST),
        "plus these bytes": _status("src/a.scala", P.OFF_PIN),
        "UNDETERMINABLE": _status("src/a.scala", P.UNDETERMINABLE),
    }
    cited = P.citation(_verification(sources=[by_form[form]]), path=_registry(tmp_path))
    assert form in cited
    assert sum(other in cited for other in by_form) == 1
