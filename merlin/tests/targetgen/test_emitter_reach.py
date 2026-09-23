"""A performance family must be emittable somewhere its traits admit it.

The double gate checks the traits and the emitter's status independently, and neither asks the question
between them: can that emitter emit for THAT target? This pins the answer, because the gap is real and
it read as something else entirely.

The original PC failure depended on an ISA definition that did not resolve at the time.
Provider availability can change. Test that requirement independently of target names;
the corpus checks below still exercise the actual declared families and target facts.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import yaml
from merlin_experiments.phase0 import sweeps as SWEEPS

from merlin.common.paths import repo_root
from merlin.perf import emitter_reach as ER

pytestmark = pytest.mark.target("gemmini", "atlas", "mx_gemmini", "radiance")


_TARGETS = ("gemmini", "atlas", "radiance", "mx_gemmini")


def _shared():
    return (
        yaml.safe_load((repo_root() / "experiments/templates/phase0/performance.yaml").read_text(encoding="utf-8"))
        or {}
    )


def _families() -> dict[str, str]:
    return {
        s["base"]["performance"]["family"]: s["base"]["performance"]["emitter"]["entry"]
        for s in _shared().get("sweeps") or []
    }


def _satisfying_targets() -> dict[str, tuple[str, ...]]:

    out: dict[str, tuple[str, ...]] = {}
    for sweep in _shared().get("sweeps") or []:
        perf = sweep["base"]["performance"]
        sat = []
        for t in _TARGETS:
            try:
                ok, _ = SWEEPS.evaluate_gate(perf["gate"], SWEEPS._performance_facts(t))
            except Exception:  # noqa: BLE001 -- an unresolvable target is not a pass
                ok = False
            if ok:
                sat.append(t)
        out[perf["family"]] = tuple(sat)
    return out


def test_a_capsule_builder_reaches_every_target():
    """It emits the interface grammar, so no ISA is involved — which is why PK/PF/PL run."""
    for t in _TARGETS:
        r = ER.can_emit("merlin.targetgen.corpus_spec.build", t)
        assert r.can_emit and not r.needs_isa


@pytest.mark.parametrize("failure", [ValueError("no ISA definition"), RuntimeError("no compute array")])
def test_the_program_generator_requires_current_machine_facts(monkeypatch, failure):
    """A changing provider is queried each time; failures retain their distinct repair reason."""
    provider = ModuleType("merlin.perf.workload_gen")
    calls = []
    available = False

    def machine_facts(target):
        calls.append(target)
        if not available:
            raise failure
        return object()

    provider.machine_facts = machine_facts
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    entry = "merlin.perf.workload_gen.plan_matmul"
    target = "synthetic-device"
    for available in (False, True, False):
        reach = ER.can_emit(entry, target)
        assert (reach.entry, reach.target) == (entry, target)
        assert reach.needs_isa and reach.established
        assert reach.can_emit is available
        if not available:
            assert reach.reason == f"{type(failure).__name__}: {failure}"
    assert calls == [target, target, target]


def test_no_family_is_admitted_where_its_emitter_cannot_reach():
    """The invariant, now that PC satisfies it too.

    It did not. PC's traits are satisfied on gemmini alone, and its emitter was declared as
    `perf.workload_gen.plan_matmul` -- a program generator that needs a self-hosted ISA, which gemmini
    (a RoCC command-buffer machine) does not have. So the claim was unsatisfiable while reading as "the
    emitter is not wired yet". Repointing it at `perf.command_stream_gen`, the sibling written for this
    archetype, resolved it: that emitter consumes the ordered command list the interface grammar
    already carries and needs no ISA.

    This assertion is what makes the repair stick. A family declared against the wrong archetype again
    fails here rather than looking merely unfinished.
    """
    families = _families()
    blocked = ER.unreachable_where_admitted(_satisfying_targets(), families)
    assert not blocked, (
        f"family/families {sorted(blocked)} are admitted by their traits only on targets their "
        f"emitter cannot reach: {blocked}. Either the emitter is declared against the wrong archetype "
        f"or the gate admits a machine this emitter was never for."
    )


def test_pc_reaches_the_one_target_its_traits_admit():
    """The specific case, kept explicit because it is the one that was wrong."""
    families = _families()
    if "PC" not in families:
        pytest.skip("PC is not declared")
    sat = _satisfying_targets().get("PC", ())
    assert "gemmini" in sat, f"PC's traits are no longer satisfied on gemmini (satisfied on {sat})"
    r = ER.can_emit(families["PC"], "gemmini")
    assert r.can_emit, f"PC's emitter cannot reach gemmini: {r.reason}"
    assert not r.needs_isa, "PC's declared command-stream emitter must stay independent of ISA-definition availability"


def test_no_other_family_is_silently_unreachable():
    """The families that DO run must not be in the blocked set, or the check is mis-scoped."""
    families = _families()
    blocked = ER.unreachable_where_admitted(_satisfying_targets(), families)
    for fam in ("PK", "PF", "PL"):
        if fam in families:
            assert fam not in blocked, (
                f"{fam} is reported unreachable, but it materializes capsules today; the reachability rule is wrong"
            )


def test_a_family_with_no_admitting_target_is_not_reported_here():
    """PS is refuted/unestablished everywhere. That is the TRAIT gate's story, told with evidence.

    Reporting it as an emitter-reachability problem too would attribute a trait result to the emitter
    and send the repair to the wrong place.
    """
    families = _families()
    sat = _satisfying_targets()
    if "PS" not in families:
        pytest.skip("PS is not declared")
    assert sat.get("PS") == (), f"PS is satisfied somewhere now: {sat.get('PS')}"
    assert "PS" not in ER.unreachable_where_admitted(sat, families)


# --- an unclassified emitter kind is not a reachable one -------------------------------------------
#
# `can_emit` used to answer "capsule builder: emits the interface grammar, no ISA needed" for every
# entry that merely was not the program generator -- an allowlist of one, inverted into a permissive
# default. Measured on the entries `_perf.yaml` actually declares, that produced a confident True with
# a fabricated reason for the barrier pair, for both `new:` placeholders that name no module at all,
# and for a nonsense string. Nothing consumed the module, so nothing noticed.


def test_an_unclassified_entry_is_not_reported_as_reaching_everything():
    r = ER.can_emit("utter.nonsense.that.does.not.exist", "gemmini")
    assert r.can_emit is False
    assert r.established is False, "an unrecognised emitter kind is UNMEASURED, not shown to be unreachable"
    assert "unclassified" in r.reason


def test_a_placeholder_entry_naming_no_module_does_not_claim_reach():
    """`new:` entries are declared before an emitter exists; they must not read as working."""
    r = ER.can_emit("new:host_seam_differential", "gemmini")
    assert (r.can_emit, r.established) == (False, False)


def test_the_target_emitter_driver_is_unestablished_and_says_why():
    """`pair_from_emitter` drives the TARGET's own emitter through a knob on its signature.

    Whether a given target's emitter accepts that knob is a fact about its backend, and
    `merlin/targets` is not on the import path -- so this module must not answer either way.
    """
    r = ER.can_emit("merlin.perf.barrier_arms.pair_from_emitter", "gemmini")
    assert (r.can_emit, r.established) == (False, False)
    assert "knob" in r.reason


def test_both_grammar_emitters_still_reach_every_target():
    """The two archetypes that genuinely need no ISA: one produces the grammar, one consumes it."""
    for entry in ("merlin.targetgen.corpus_spec.build", "merlin.perf.command_stream_gen.pair_from_interface"):
        for target in _TARGETS:
            r = ER.can_emit(entry, target)
            assert r.can_emit is True and r.established is True, f"{entry} on {target}: {r.reason}"


def test_unestablished_is_not_reported_as_unreachable():
    """The mirror of the old bug: "nobody classified this" must not become a named defect either."""
    families = {"PBar": "merlin.perf.barrier_arms.pair_from_emitter"}
    gates = {"PBar": tuple(_TARGETS)}
    assert ER.unreachable_where_admitted(gates, families) == {}, (
        "an unestablished reach is not evidence the family cannot be emitted"
    )
    assert "PBar" in ER.unestablished_reach(families, _TARGETS), (
        "but it must be surfaced somewhere, not silently dropped from both reports"
    )
