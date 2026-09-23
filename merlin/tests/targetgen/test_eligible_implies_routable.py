"""The denominator and the router must answer one question the same way.

Eligibility is the denominator of acceleratable recall and routing is its numerator. When the first
says yes and the second says no, the region is unwinnable by construction: it sits in every
compiler's denominator and no compiler can move it.
"""

from __future__ import annotations

import pytest

from merlin.common import quant_formats as qf
from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen import eligibility as E
from merlin.targetgen import routing as R
from merlin.targetgen import semantic_families as sf
from merlin.targetgen import target_registry as tr
from merlin.targetgen.compute_units import SemanticCapability


def _caps(**composed) -> dict[str, SemanticCapability]:
    families = {
        "contraction": (),
        "elementwise_map": ("contraction",),
        "reduction": ("contraction", "movement"),
        **composed,
    }
    return {
        name: SemanticCapability(family=name, dtypes=("int8",), composed_with=tuple(attach))
        for name, attach in families.items()
    }


def _verdict(family: str, caps, **kwargs) -> E.EligibilityVerdict:
    return E.is_eligible(E.RegionDescriptor(family=family, in_dtype="int8"), caps, **kwargs)


def test_a_fused_only_primitive_does_not_license_a_composite_with_no_producer() -> None:
    caps = _caps()
    for composite in ("normalization", "softmax"):
        verdict = _verdict(composite, caps)
        assert not verdict.eligible
        assert "no such producer feeds this region" in verdict.reason
    # The mutation that proves the rule is what excludes them: make the primitives standalone.
    standalone = _caps(elementwise_map=(), reduction=())
    assert _verdict("normalization", standalone).eligible


def test_a_fused_only_capability_admits_a_region_its_producer_feeds() -> None:
    caps = _caps()
    assert not _verdict("elementwise_map", caps).eligible
    attached = _verdict("elementwise_map", caps, fused_with=("contraction",))
    assert attached.eligible and "attached to ['contraction']" in attached.reason
    # A producer the capability does not attach to is not a licence.
    assert not _verdict("elementwise_map", caps, fused_with=("movement",)).eligible
    assert _verdict("reduction", caps, fused_with=("movement",)).eligible


def _known_disagreements() -> set[tuple[str, str, str]]:
    ledger = repo_root() / "build_tools/scripts/eligible_unroutable_ratchet.txt"
    rows = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        fields = line.split("#", 1)[0].split()
        if len(fields) == 3:
            rows.add((fields[0], fields[1], fields[2]))
    return rows


def _resolvable_targets() -> list[str]:
    found = []
    # Every target the tree declares, found rather than listed: a target added later is held to
    # the agreement without anyone remembering to add it here.
    declared = sorted(
        path.parent.parent.name for path in (merlin_dir() / "targets").glob("*/contracts/target_contract.yaml")
    )
    for name in declared:
        try:
            E.capability_map_for_target(name)
        except Exception:  # noqa: BLE001 -- not onboarded here
            continue
        found.append(name)
    return found


@pytest.mark.parametrize("target", _resolvable_targets())
def test_a_family_eligible_standalone_is_routable(target: str) -> None:
    cap_map = E.capability_map_for_target(target)
    disagreements = []
    for family, capability in cap_map.items():
        for dtype in capability.dtypes:
            if not qf.has(dtype):
                continue
            verdict = E.is_eligible(E.RegionDescriptor(family=family, in_dtype=dtype), cap_map)
            plan = R.route_plan([R.OpDemand(op=family, in_fmt=dtype, family=family)], target)
            routed = bool(plan["mesh"] or plan["fallback"])
            if verdict.eligible and not routed:
                disagreements.append((target, family, dtype))
    known = {row for row in _known_disagreements() if row[0] == target}
    assert set(disagreements) - known == set(), "a NEW eligible-but-unroutable format"
    assert known - set(disagreements) == set(), "a ledger entry that no longer reproduces: remove it"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the router places ops one at a time and never decomposes a composite into the primitives "
        "that license it, so a target whose primitives are all standalone reads eligible for a softmax "
        "it will not route; closed when placement works on groups rather than single ops"
    ),
)
def test_a_composite_eligible_through_its_primitives_is_routable() -> None:
    disagreements = []
    for target in _resolvable_targets():
        cap_map = E.capability_map_for_target(target)
        for composite in sf.COMPOSITES:
            if composite in cap_map:
                continue
            dtypes = [d for p in sf.primitives_of(composite) if p in cap_map for d in cap_map[p].dtypes if qf.has(d)]
            for dtype in dict.fromkeys(dtypes):
                region = E.RegionDescriptor(family=composite, in_dtype=dtype)
                if not E.is_eligible(region, cap_map).eligible:
                    continue
                plan = R.route_plan([R.OpDemand(op=composite, in_fmt=dtype, family=composite)], target)
                if not (plan["mesh"] or plan["fallback"]):
                    disagreements.append((target, composite, dtype))
    assert not disagreements, disagreements
