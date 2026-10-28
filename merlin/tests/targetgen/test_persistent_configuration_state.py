"""Where ``persistent_configuration_state`` may come from, and what it must never come from.

The trait is the encoding ABI's ``config_subtype`` table: a configuration op class whose sub-types
each set state later commands read. A data-only Phase 0 selection supplies the target's contract and
no residual, and the trait used to read the residual alone, so a contract that plainly declares the
table left the trait UNKNOWN -- an absent OVERRIDE read as an absent DECLARATION, the same collapse
``Sources.units`` already guards against. These tests pin the contract fallback, the residual's
precedence over it, the DECLARED tier, and that nothing else establishes the trait.
"""
from __future__ import annotations

from merlin.perf import profile as P

_TABLE = {0: "CFG_A", 1: "CFG_B"}


def _trait(**kw):
    kw.setdefault("facts", {})
    kw.setdefault("residual", {})
    kw.setdefault("contract", {})
    return P._t_persistent_configuration_state(P.load_sources("phantom_target_under_test", **kw))


def test_a_contract_encoding_table_establishes_it_as_contract_declared():
    trait, tier = _trait(contract={"encoding": {"config_subtype": _TABLE}})
    assert trait.satisfied is True
    assert tier == P.TIER_CONTRACT
    assert "contract" in trait.evidence and "CFG_A" in trait.evidence


def test_the_residual_wins_where_both_declare_a_table():
    trait, tier = _trait(
        residual={"encoding": {"config_subtype": {0: "RES_ONLY"}}},
        contract={"encoding": {"config_subtype": _TABLE}},
    )
    assert trait.satisfied is True
    assert tier == P.TIER_RESIDUAL
    assert "RES_ONLY" in trait.evidence and "CFG_A" not in trait.evidence


def test_no_table_anywhere_stays_unknown_not_false():
    trait, tier = _trait(contract={"encoding": {"semantic_class": {0: "CONFIG"}}})
    assert trait.satisfied is None
    assert tier == P.TIER_NONE
    assert "semantic_class" in trait.evidence


def test_a_malformed_table_does_not_establish_it():
    trait, _ = _trait(contract={"encoding": {"config_subtype": {0: ""}}})
    assert trait.satisfied is None
    trait, _ = _trait(contract={"encoding": {"config_subtype": ["CFG_A"]}})
    assert trait.satisfied is None


def test_the_rtl_facts_alone_never_establish_it():
    facts = {"facts": {"interfaces": [{"name": "funct_decode_table", "evidence": "funct 0"}]}}
    trait, _ = _trait(facts=facts)
    assert trait.satisfied is None
