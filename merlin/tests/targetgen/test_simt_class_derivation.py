"""The datapath KIND a target's class is built on, and where it is allowed to come from.

``merlin.perf.profile.archetype_of`` composes a class as ``dispatch/datapath_kind``, and the datapath
half comes from the target's compute units. Those used to be read from the residual side-input ALONE,
so a target that ships no residual file was credited with no compute units at all: its class degraded
to ``unknown-datapath`` and it stopped being asked the questions its kind exists to ask — an absent
OVERRIDE read as an absent DECLARATION.

These tests pin the fallback, its precedence, and the thing that makes the fallback safe: a kind read
from a contract is tiered as DECLARED, never as RTL-grounded.
"""
from __future__ import annotations

from merlin.perf import profile as P

_SIMT_CONTRACT = {"compute_units": [{"name": "cluster", "kind": "simt"},
                                    {"name": "pe", "kind": "systolic"}]}
_VECTOR_RESIDUAL = {"compute_units": [{"name": "vec", "kind": "vector"}]}


def _sources(**kw):
    # facts={} keeps the RTL out of it: this is about the DECLARED half of the class.
    kw.setdefault("facts", {})
    kw.setdefault("residual", {})
    kw.setdefault("contract", {})
    return P.load_sources("phantom_target_under_test", **kw)


class TestContractFallback:
    def test_a_target_with_no_residual_still_has_its_kinds(self):
        src = _sources(contract=_SIMT_CONTRACT)
        assert src.unit_kinds() == ("simt", "systolic")
        assert src.units_source() == "contract"

    def test_the_class_no_longer_degrades_to_unknown_datapath(self):
        arch = P.archetype_of(_sources(contract=_SIMT_CONTRACT))
        assert arch.datapath_kind == "simt"
        assert arch.label.endswith("/simt")

    def test_the_kind_restores_the_questions_that_kind_makes_worth_asking(self):
        with_kind = P.archetype_of(_sources(contract=_SIMT_CONTRACT))
        without = P.archetype_of(_sources())
        assert set(P._QUESTIONS_FOR_KIND["simt"]) <= set(with_kind.questions)
        assert len(with_kind.questions) > len(without.questions)

    def test_nothing_declared_anywhere_is_still_unknown_datapath(self):
        # The fallback must not invent a kind. A target that declares none in EITHER place has none.
        arch = P.archetype_of(_sources())
        assert arch.datapath_kind is None
        assert arch.label.endswith("/unknown-datapath")
        assert "no compute unit is declared" in arch.evidence["datapath_kind"]


class TestPrecedence:
    def test_the_residual_overrides_the_contract(self):
        src = _sources(residual=_VECTOR_RESIDUAL, contract=_SIMT_CONTRACT)
        assert src.unit_kinds() == ("vector",)
        assert src.units_source() == "residual"

    def test_an_explicitly_empty_contract_suppresses_the_on_disk_read(self):
        # Without this a test that DELETES a declaration would have it quietly restored from disk,
        # and would pass while proving nothing.
        assert _sources().unit_kinds() == ()


class TestTierHonesty:
    def test_a_contract_declared_kind_is_not_rtl_evidence(self):
        src = _sources(contract=_SIMT_CONTRACT)
        assert src.units_tier() == P.TIER_CONTRACT
        assert src.units_tier() != P.TIER_FACTS

    def test_a_contract_declared_kind_is_not_reported_as_a_residual_one(self):
        # Both are DECLARED, but a reader checking one opens a different file than a reader checking
        # the other, so collapsing them would send them to the wrong place.
        assert _sources(contract=_SIMT_CONTRACT).units_tier() != P.TIER_RESIDUAL
        assert _sources(residual=_VECTOR_RESIDUAL).units_tier() == P.TIER_RESIDUAL

    def test_the_trait_that_rests_on_the_units_carries_the_contract_tier(self):
        trait, tier = P._t_multiple_engine_groups(_sources(contract=_SIMT_CONTRACT))
        assert trait.satisfied is True
        assert tier == P.TIER_CONTRACT
        assert "contract" in trait.evidence

    def test_the_class_evidence_names_the_source_and_the_tier(self):
        ev = P.archetype_of(_sources(contract=_SIMT_CONTRACT)).evidence["datapath_kind"]
        assert "contract" in ev and P.TIER_CONTRACT in ev

    def test_the_table_gives_a_contract_tier_its_own_letter(self):
        prof = P.derive_profile("phantom_target_under_test", facts={}, residual={},
                                contract=_SIMT_CONTRACT)
        table = P.profile_table([prof])
        # `+c` -- satisfied, contract-declared. Never `+f`, which would read as RTL-grounded.
        assert "+c" in table
        assert "+f" not in table
