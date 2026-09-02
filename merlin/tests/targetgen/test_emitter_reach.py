"""A performance family must be emittable somewhere its traits admit it.

The double gate checks the traits and the emitter's status independently, and neither asks the question
between them: can that emitter emit for THAT target? This pins the answer, because the gap is real and
it read as something else entirely.

⚠️ MEASURED. `PC`'s traits are satisfied on exactly one target — gemmini, whose
`independent_engine_ports` and `explicit_completion` both derive True from its own facts — and gemmini
is the one target `workload_gen` cannot encode a kernel for: it ships no ISA definition, being a RoCC
command-buffer machine rather than a self-hosted-ISA one. The only target the program generator CAN
encode is atlas, whose `explicit_completion` is unestablished. So the family is unsatisfiable as
declared. Before this it presented as "the emitter is not wired yet", which sent a reader off to build
something that already existed.
"""
from __future__ import annotations

import sys

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.perf import emitter_reach as ER

sys.path.insert(0, str(merlin_dir() / "contract" / "capsules"))

_TARGETS = ("gemmini", "atlas", "radiance", "mx_gemmini")


def _shared():
    return yaml.safe_load(
        (merlin_dir() / "contract" / "capsules" / "profiles" / "_perf.yaml").read_text(
            encoding="utf-8")) or {}


def _families() -> dict[str, str]:
    return {s["base"]["performance"]["family"]: s["base"]["performance"]["emitter"]["entry"]
            for s in _shared().get("sweeps") or []}


def _satisfying_targets() -> dict[str, tuple[str, ...]]:
    import generate_corpus as GC
    out: dict[str, tuple[str, ...]] = {}
    for sweep in _shared().get("sweeps") or []:
        perf = sweep["base"]["performance"]
        sat = []
        for t in _TARGETS:
            try:
                ok, _ = GC.evaluate_gate(perf["gate"], GC._performance_facts(t))
            except Exception:                      # noqa: BLE001 -- an unresolvable target is not a pass
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


def test_the_program_generator_needs_a_self_hosted_isa():
    """And it is honest about which targets have one, carrying the reason rather than a bare False."""
    encodable = {t: ER.can_emit("merlin.perf.workload_gen.plan_matmul", t) for t in _TARGETS}
    assert all(r.needs_isa for r in encodable.values())
    assert encodable["gemmini"].can_emit is False
    assert "no ISA definition" in encodable["gemmini"].reason
    assert encodable["atlas"].can_emit is True, (
        "atlas is the target this generator can encode; if that changed, the reachability story below "
        "changed with it")


def test_pc_is_unsatisfiable_as_declared_and_that_is_recorded():
    """⚠️ WHEN THIS FAILS, PC BECAME SATISFIABLE — re-read the family rather than deleting the test.

    Either gemmini gained an ISA definition, or another target's `explicit_completion` and
    `independent_engine_ports` became established. Both are good news and both change what PC needs.
    """
    families = _families()
    if "PC" not in families:
        pytest.skip("PC is not declared in the shared template")
    blocked = ER.unreachable_where_admitted(_satisfying_targets(), families)
    assert "PC" in blocked, (
        "PC is no longer emitter-unreachable on every target its traits admit — the family may now be "
        "runnable; re-read its gate and emitter rather than assuming this test is stale")
    assert blocked["PC"] == ("gemmini",), f"expected gemmini alone, got {blocked['PC']}"


def test_no_other_family_is_silently_unreachable():
    """The families that DO run must not be in the blocked set, or the check is mis-scoped."""
    families = _families()
    blocked = ER.unreachable_where_admitted(_satisfying_targets(), families)
    for fam in ("PK", "PF", "PL"):
        if fam in families:
            assert fam not in blocked, (
                f"{fam} is reported unreachable, but it materializes capsules today; the reachability "
                f"rule is wrong")


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
