"""Can each declared performance family reach a verdict at all, from its own declaration?

Two families in the shipped template cannot, and neither reads that way. `PC`'s gate asks for at least
two separation regimes while its comparand holds the separation regime EQUAL, so no admissible pair
exists. `PL`'s falsifier fires when a saving changes beyond "the declared band", and no band is
declared anywhere -- not in the family, not in a target contract. Both look finished, both materialise
capsules, and both would sit in a campaign producing members forever without ever producing an answer.

That is the failure mode `merlin.perf.emitter_reach` was written for one level up, and its docstring
records the measured case: a family admissible only where its emitter could not reach. These are the
same species -- a contract that is internally unsatisfiable -- caught by reading the relation between
the fields rather than each field alone.

The roster assertion is deliberately over the WHOLE template rather than over a list of families this
file knows. A family added later that carries the same contradiction fails here, which is the point;
re-listing the known-good names would only move the staleness.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.perf import claim_reach as CR

_TEMPLATE = merlin_dir() / "contract" / "capsules" / "profiles" / "_perf.yaml"


def _declarations() -> dict:
    document = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8")) or {}
    out = {str(sweep["id"]): sweep["base"]["performance"]
           for sweep in (document.get("sweeps") or [])}
    out.update({str(row["family"]): row["performance"]
                for row in (document.get("blocked_unimplemented") or [])})
    return out


def test_exactly_the_three_known_contradictions_are_unsatisfiable():
    """The audit's verdict over the whole shipped template, derived from the template itself."""
    verdicts = {name: CR.family_reach(perf) for name, perf in _declarations().items()}
    unsatisfiable = {name for name, v in verdicts.items() if not v.satisfiable}

    assert unsatisfiable == {"PS", "PC", "PL"}, (
        "a family became unsatisfiable (or one was repaired) without this audit being updated: "
        + repr({n: [o["rule"] for o in v.obstructions] for n, v in sorted(verdicts.items())
                if not v.satisfiable}))


def test_PC_cannot_vary_the_quantity_its_own_comparand_holds_equal():
    verdict = CR.family_reach(_declarations()["PC"])

    assert [o["rule"] for o in verdict.obstructions] == [CR.CAPACITY_CONTRADICTS_DEMAND_EQUAL]
    assert "separation_regime" in verdict.obstructions[0]["detail"]


def test_PL_fires_on_a_band_nothing_declares():
    verdict = CR.family_reach(_declarations()["PL"])

    assert [o["rule"] for o in verdict.obstructions] == [CR.UNDECLARED_FIRING_QUANTITY]
    assert "band" in verdict.obstructions[0]["detail"]


def test_only_the_families_with_a_frozen_analyzer_have_a_decision_procedure_today():
    """Not a contradiction -- a wiring state -- but the reason a satisfiable family still has no verdict.

    PK has had one since its contract was frozen. PR joined it when the residency family froze its own
    acceptance contract and named ``perf_pr_claim``. Every other family here is satisfiable and simply
    has nothing that turns its rows into a verdict.
    """
    verdicts = {name: CR.family_reach(perf) for name, perf in _declarations().items()}
    decidable = {name for name, v in verdicts.items() if v.decidable_today}

    assert decidable == {"PK", "PR"}, (
        "a family gained or lost its acceptance.analyzer; the per-family audit must say so: "
        + repr(sorted(decidable)))


def test_PF_is_satisfiable_as_declared():
    """PF's obstruction is measurement, not declaration, and the two must not read alike."""
    verdict = CR.family_reach(_declarations()["PF"])

    assert verdict.satisfiable is True
    assert verdict.obstructions == ()


def test_the_gemmini_audit_table_separates_the_three_ways_a_family_stops():
    """One place that states, per family, WHICH of the three things is missing on this target.

    They are genuinely different repairs and this tree has repeatedly confused them:

    * the target's own traits refuse the family -- an answer about the hardware, and final;
    * the declaration contradicts itself -- an answer about the contract, and it must be re-declared;
    * the family is admitted, materialises members, and nothing computes its verdict -- a wiring gap,
      and the only one of the three that finishing the work resolves.
    """
    import sys

    from merlin.common.paths import merlin_dir as _merlin_dir

    generator_dir = _merlin_dir() / "contract" / "capsules"
    if str(generator_dir) not in sys.path:
        sys.path.insert(0, str(generator_dir))
    import generate_corpus as GC  # noqa: PLC0415

    profile = GC.load_profile("gemmini", include_holdouts=False)
    shared = [s for s in profile["sweeps"] if (s.get("base") or {}).get("cat") == "_perf"]
    skips, blocked, errors = [], [], []
    binding = GC.CS.derive_binding(
        GC.load_target_experiment(GC._descriptor_for("gemmini")), profile.get("datapath") or {})
    entries = GC.expand_sweeps({"sweeps": shared}, binding,
                               trait_facts=GC._performance_facts("gemmini"), skipped=skips,
                               blocked_unimplemented=blocked, errors=errors)
    assert errors == []

    refused = {row["family"]: row["gate"]["outcome"] for row in skips}
    built = {entry["performance"]["family"] for entry in entries}
    declarations = _declarations()
    table = {}
    for name, perf in declarations.items():
        verdict = CR.family_reach(perf)
        if name in refused:
            state = f"trait_{refused[name]}"
        elif not verdict.satisfiable:
            state = "declaration_contradicted"
        elif name not in built:
            state = "not_materialized"
        else:
            state = "decidable" if verdict.decidable_today else "no_analyzer"
        table[name] = state

    assert table == {
        # The one family with a frozen acceptance contract and an analyzer. Its verdict on the
        # measured cohort is REFUTED -- see test_perf_pk_refutation_mechanism.
        "PK": "decidable",
        # Refuted by the hardware: this is not a self-hosted program.
        "PS": "trait_refuted",
        # Traits pass and members build, but the gate asks for two of what the comparand pins.
        "PC": "declaration_contradicted",
        # Builds a complete group; nothing turns the group into a verdict yet.
        "PF": "no_analyzer",
        # Builds its regime pair; the falsifier fires on a band nothing declares.
        "PL": "declaration_contradicted",
        # Refuted by the hardware: one declared contraction encoding, so there is no choice to price.
        "PG": "trait_refuted",
        # Builds a residency ladder across every reachable band, and now names the analyzer that
        # decides it -- its acceptance contract is frozen.
        "PR": "decidable",
        # Declared blocked on per-lane cycle accounting, so it never enters the sweep expansion.
        "PB": "not_materialized",
        # PK's successor: the same affine claim with the same two bounds, re-declared to sit PAST the
        # overlap fill transient that refuted PK. Blocked on the derivation rather than on the claim
        # -- its starting depth is a measured property of the target, and no materializer can yet read
        # a recorded measurement -- so like PB it never enters the sweep expansion.
        "PT": "not_materialized",
    }, table


# ---------------------------------------------------------------------------------------------------
# the rules themselves, pinned away from the shipped template
# ---------------------------------------------------------------------------------------------------


def _declaration(*, capacity="at_least_two_widget_regimes", demand_equal=("operation", "M"),
                 fires_when="the_delta_is_zero", acceptance=None) -> dict:
    return {
        "family": "FX", "level": "L1_test", "lever": "test", "claim": "DIFFERENTIAL",
        "comparand": {"kind": "paired_run", "against": "control", "cancels": ["shape"],
                      "demand_equal": list(demand_equal)},
        "falsifier": {"observation": "cycles", "fires_when": fires_when,
                      "negative_control": "control"},
        "gate": {"traits": ["t"], "instrument": "cycle_count", "capacity": capacity,
                 "on_missing": "skip_with_evidence"},
        "regime": {"separation": "fixed", "layout": "identical"},
        "emitter": {"status": "existing", "entry": "merlin.targetgen.corpus_spec.build", "knobs": {}},
        "cost": {"tier": 1, "runs": 2, "projected_cycles": "preflight", "basis": "two"},
        **({"acceptance": acceptance} if acceptance else {}),
    }


def test_a_capacity_naming_a_plural_of_a_held_equal_quantity_is_a_contradiction():
    verdict = CR.family_reach(_declaration(demand_equal=("operation", "widget_regime")))

    assert [o["rule"] for o in verdict.obstructions] == [CR.CAPACITY_CONTRADICTS_DEMAND_EQUAL]


def test_a_capacity_over_a_quantity_nobody_holds_equal_is_fine():
    verdict = CR.family_reach(_declaration(demand_equal=("operation", "operand_dtype")))

    assert verdict.satisfiable is True


def test_a_single_token_demand_equal_entry_never_establishes_the_contradiction():
    """``K`` landing inside an unrelated capacity is a coincidence, not a contradiction."""
    verdict = CR.family_reach(_declaration(capacity="at_least_two_widget_regimes",
                                           demand_equal=("widget",)))

    assert verdict.satisfiable is True


def test_a_completeness_capacity_demands_no_axis_and_so_contradicts_nothing():
    verdict = CR.family_reach(_declaration(capacity="complete_fused_and_part_comparison_group",
                                           demand_equal=("fused", "part")))

    assert verdict.satisfiable is True


def test_a_firing_condition_may_not_satisfy_its_own_declared_quantity():
    """The exclusion without which the rule is vacuous: every ``declared_X`` mentions an X."""
    verdict = CR.family_reach(_declaration(fires_when="saving_changes_beyond_the_declared_band"))

    assert [o["rule"] for o in verdict.obstructions] == [CR.UNDECLARED_FIRING_QUANTITY]


def test_a_declared_quantity_the_contract_carries_elsewhere_does_not_fire():
    verdict = CR.family_reach(
        _declaration(fires_when="the_saving_exceeds_the_declared_residual_bound",
                     acceptance={"analyzer": "x/v1",
                                 "thresholds": {"residual_bound": {"absolute_floor_cycles": 8}}}))

    assert verdict.satisfiable is True
    assert verdict.decidable_today is True


def test_a_firing_condition_that_declares_nothing_is_out_of_the_rule_s_scope():
    """Narrow on purpose: an unquantified "noise band" is terse, not self-contradictory."""
    verdict = CR.family_reach(
        _declaration(fires_when="the_two_costs_agree_within_the_noise_band"))

    assert verdict.satisfiable is True


# --------------------------------------------------------------------------------------------
# analyzer_identity: the parsing a claim dispatcher rests on.  A family names the module, function
# and version that decide it, so a dispatcher resolves its decision procedure from the DECLARATION
# instead of from a table of family names -- and a family that names none is refused rather than
# handed some other family's arithmetic.
# --------------------------------------------------------------------------------------------

def test_a_declared_analyzer_splits_into_module_function_and_version():
    identity = CR.analyzer_identity(
        {"acceptance": {"analyzer": "pkg.sub.mod.analyze_x/v12"}})

    assert (identity.module, identity.function, identity.version) == (
        "pkg.sub.mod", "analyze_x", "v12")
    assert identity.declared == "pkg.sub.mod.analyze_x/v12"


@pytest.mark.parametrize("performance", [
    {},
    {"acceptance": {}},
    {"acceptance": {"analyzer": ""}},
    {"acceptance": "not a mapping"},
])
def test_a_family_that_declares_no_analyzer_reads_as_absent_not_as_broken(performance):
    """``None`` is the same wiring state ``has_decision_procedure`` reports, not an error."""
    assert CR.analyzer_identity(performance) is None
    assert CR.has_decision_procedure(performance) is False


@pytest.mark.parametrize("declared", [
    "perf_pk_claim",                       # no function, no version
    "perf_pk_claim/v1",                    # no function
    "analyze/v1",                          # no module
    "perf_pk_claim.analyze_pk_claim",      # no version
    "perf_pk_claim.analyze pk/v1",         # not an identifier
    "perf_pk_claim..analyze/v1",           # empty module segment
    "perf_pk_claim.analyze/",              # empty version
    17,                                    # not a string at all
])
def test_a_malformed_analyzer_raises_rather_than_reading_as_absent(declared):
    """A malformed contract and an absent one are different states and must not collapse."""
    with pytest.raises(ValueError):
        CR.analyzer_identity({"acceptance": {"analyzer": declared}})
