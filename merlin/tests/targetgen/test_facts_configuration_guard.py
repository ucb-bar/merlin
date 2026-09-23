"""One cache slot per target name, and a design has many configurations.

WHY THIS EXISTS, measured 2026-09-19. A hardware design is elaborated in many configurations -- a
different mesh edge, bank count, accumulator depth, dtype set -- and `rtl_cache_dir` is
`artifacts/cache/rtl_introspect/<target>/` with NO configuration component. Every artifact already
records which elaboration it describes, in `facts.source.config`, and nothing read it.

So regenerating a target under a second configuration did not add a configuration. It overwrote the
first one's facts in place, and passed every guard on the way: `_written_by_another_family` compares the
EXTRACTOR that wrote the artifact, not the device it describes, and the hollowing check only refuses a
write that loses fields -- a DIM=32 elaboration carries the same keys as a DIM=16 one, so it is not
weaker, it is about something else. Downstream, every capacity and width would then be attributed to an
elaboration that did not produce it, which is the failure the hardware-pin registry's whole discipline
exists to prevent: "cycles are not comparable across designs."

The repo's real answer to a second configuration today is a second target name -- `gemmini_universal`
is `UniversalResNet50GemminiRocketConfig` with its own package, contract and pin, and its descriptor
says in its own words that it must never be mixed with `gemmini`. These tests do not change that. They
make the alternative -- quietly reusing one slot for two devices -- impossible rather than merely
discouraged, which is the precondition for a configuration axis being added later.
"""

from __future__ import annotations

import json

import pytest

from merlin.targetgen.rtl.facts import (
    FactsConfigMismatch,
    FactsDowngrade,
    _configuration_of,
    write_facts_guarded,
)

A = "GemminiRocketConfig"
B = "UniversalResNet50GemminiRocketConfig"


def _doc(config: str | None, *, memories: int = 2) -> dict:
    source: dict = {"kind": "firrtl"}
    if config:
        source["config"] = config
    return {"facts": {"source": source, "memories": [{"name": f"m{i}"} for i in range(memories)]}}


def test_a_second_configuration_does_not_silently_take_the_slot(tmp_path):
    """THE POINT. Both artifacts are well-formed, neither is weaker, and they describe different
    devices -- so nothing but the configuration itself can tell them apart."""
    p = tmp_path / "facts.json"
    write_facts_guarded(p, _doc(A))
    with pytest.raises(FactsConfigMismatch) as caught:
        write_facts_guarded(p, _doc(B))
    message = str(caught.value)
    assert A in message and B in message, "the refusal must name BOTH configurations, not just one"
    assert json.loads(p.read_text())["facts"]["source"]["config"] == A, "the first artifact survived"


def test_the_same_configuration_may_be_re_extracted(tmp_path):
    """The control. Without it this suite would pass for a guard that refuses every second write, which
    would break the regeneration these artifacts exist to support."""
    p = tmp_path / "facts.json"
    write_facts_guarded(p, _doc(A, memories=1))
    write_facts_guarded(p, _doc(A, memories=3))
    assert len(json.loads(p.read_text())["facts"]["memories"]) == 3


def test_an_artifact_recording_no_configuration_is_not_invalidated_on_a_guess(tmp_path):
    """Three-state, the same way the family guard treats an artifact that records no generator: an
    older or simpler extractor that never stamped a configuration is accepted, because "this is a
    different device" and "this one does not say" are different claims and only one of them is checked.
    """
    p = tmp_path / "facts.json"
    write_facts_guarded(p, _doc(None))
    write_facts_guarded(p, _doc(A))
    assert json.loads(p.read_text())["facts"]["source"]["config"] == A


def test_losing_a_configuration_stamp_is_a_hollowing_not_a_mismatch(tmp_path):
    """The other direction, and it belongs to the OLDER guard rather than this one.

    Asserted because I got it wrong first: an artifact that used to name its configuration and now does
    not has LOST a fact, which is what `hollowed_facts` is for -- and it is the signature of the
    degraded extraction path, where a missing toolchain produces a well-formed artifact that knows less.
    The two guards divide cleanly: a DIFFERENT configuration is a different device, an ABSENT one is a
    weaker read of the same device, and neither is silent.
    """
    q = tmp_path / "other.json"
    write_facts_guarded(q, _doc(A))
    with pytest.raises(FactsDowngrade, match=r"source\.config"):
        write_facts_guarded(q, _doc(None))


def test_the_slot_can_be_reassigned_deliberately(tmp_path):
    """A refusal with no way past it would be a wall rather than a guard -- the same escape hatch the
    hollowing check offers, and for the same reason: the tool cannot know that a human has decided this
    slot now describes that device."""
    p = tmp_path / "facts.json"
    write_facts_guarded(p, _doc(A))
    write_facts_guarded(p, _doc(B), allow_downgrade=True)
    assert json.loads(p.read_text())["facts"]["source"]["config"] == B


def test_hollowing_is_still_refused_within_one_configuration(tmp_path):
    """The pre-existing guard must survive the new one. Two checks on one path is exactly where one
    quietly stops running."""
    p = tmp_path / "facts.json"
    write_facts_guarded(p, _doc(A, memories=2))
    with pytest.raises(FactsDowngrade):
        write_facts_guarded(p, _doc(A, memories=0))


def test_a_configuration_mismatch_is_catchable_as_a_downgrade(tmp_path):
    """Callers already handle one refusal to overwrite; they keep handling this one. The distinct type
    exists for a caller that wants to tell "weaker facts about this device" from "another device"."""
    assert issubclass(FactsConfigMismatch, FactsDowngrade)
    p = tmp_path / "facts.json"
    write_facts_guarded(p, _doc(A))
    with pytest.raises(FactsDowngrade):
        write_facts_guarded(p, _doc(B))


@pytest.mark.parametrize(
    "doc,expect",
    [
        ({"facts": {"source": {"config": A}}}, A),
        ({"source": {"config": A}}, A),
        ({"facts": {"source": {"kind": "firrtl"}}}, None),
        ({"facts": {"source": "a prose description of the elaboration"}}, None),
        ({"facts": {}}, None),
        ({}, None),
        (None, None),
    ],
    ids=["nested", "bare", "no config key", "source is a string", "no source", "empty", "none"],
)
def test_the_configuration_is_read_without_assuming_a_shape(doc, expect):
    """`source` is a dict for some families and a bare STRING for others -- the shape divergence the
    facts schema was written to absorb. A reader that assumed the dict would raise on half the roster,
    and one that caught the raise would report "no configuration" for an artifact that has one."""
    assert _configuration_of(doc) == expect
