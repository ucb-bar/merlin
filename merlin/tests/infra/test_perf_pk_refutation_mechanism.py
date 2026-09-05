"""The PK adapter: a REFUTED decision plus the run's own counters becomes a first-class result.

The arithmetic and the mechanism live in :mod:`merlin.perf.fill_transient` and are pinned against the
measured cohort in ``test_perf_fill_transient`` -- including the evidence for NOT minting an
overlap-term successor. This file covers only the thin adapter in the campaign's claim analyzer: that
it reads a decision's own cohort and fit, that it refuses rather than averaging when the
deterministic-simulator control does not hold, and that it points the repair at the cohort block
rather than at the thresholds the frozen contract forbids moving.

Kept beside its subject: ``perf_pk_claim`` is one of the campaign scripts that is not tracked in this
working tree, and a tracked test importing an untracked module would red the suite for everyone else.
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.perf import fill_transient as FT
from test_perf_fill_transient import MEASURED, counters, readings

_SCRIPTS = merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import perf_pk_claim as PK  # noqa: E402


def _decision(*, replicate_cycles=None) -> dict:
    """A PK decision shaped exactly as :func:`perf_pk_claim.analyze_pk_claim` returns one."""
    caps = list(MEASURED)
    return {
        "family": "PK", "claim": "PREDICTS", "status": "REFUTED",
        "cohort": {"K_values": [MEASURED[c][0] for c in caps], "capsules": caps},
        "fit": {"cycles_by_K": [
            {"K": MEASURED[c][0],
             "replicate_cycles": (replicate_cycles or {}).get(c, [MEASURED[c][1]] * 3)}
            for c in caps]},
        "refutation_reasons": ["r_squared_at_least_0_995",
                              "every_residual_within_max_of_8_cycles_and_3_percent"],
    }


def test_the_mechanism_locates_the_repair_in_the_cohort_and_not_in_the_thresholds():
    mechanism = PK.refutation_mechanism(_decision(), readings(), counters())

    assert mechanism["decision_status"] == "REFUTED"
    assert mechanism["mechanism"]["state"] == FT.IN_FILL_TRANSIENT
    assert mechanism["mechanism"]["affine_form_contradicted"] is True
    # Not `thresholds`. The acceptance block's own comment forbids moving those, and the marginals say
    # a wider bound would accept a model this cohort contradicts.
    assert mechanism["repair_locus"] == "cohort.K_multipliers_of_tile"
    assert mechanism["capsules_without_counter_readings"] == []


def test_a_capsule_with_no_counter_reading_makes_the_mechanism_undeterminable():
    partial = readings()
    del partial["PK02_k64"]
    mechanism = PK.refutation_mechanism(_decision(), partial, counters())

    assert mechanism["capsules_without_counter_readings"] == ["PK02_k64"]
    assert mechanism["mechanism"]["state"] == FT.UNDETERMINABLE
    assert mechanism["repair_locus"] == "undetermined"
    assert mechanism["mechanism"]["affine_form_contradicted"] is True


def test_disagreeing_replicates_refuse_rather_than_averaging_into_a_point():
    with pytest.raises(ValueError, match="replicates disagree"):
        PK.refutation_mechanism(_decision(replicate_cycles={"PK01_k32": [373, 375, 373]}),
                                readings(), counters())


def test_a_decision_from_another_family_is_refused():
    other = _decision()
    other["family"] = "PF"
    with pytest.raises(ValueError, match="only be read off a PK decision"):
        PK.refutation_mechanism(other, readings(), counters())
