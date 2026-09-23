"""A zero-offload gate whose exemption is manufactured by the failure it exists to catch.

MEASURED, on three separate ResNet-50 captures, byte-identically: 231 regions, 4,089,184,256 MACs,
ZERO on the accelerator, 160 contractions refused on one clause -- the operand precision read off
the module -- and the gate that refuses "eligible work with none of it on a unit" did not fire. Its
exemption ("a model with NO eligible work passes") is sound, and it applied, because `eligible` was
derived DOWNSTREAM of the refusal: the same clause that put 100% of the work on the host also made
that work ineligible. The report read `coverage 0.000, admitted True`.

The same captures' router places 54 integer contractions (53 convolutions and one fully-connected,
which is what the shipped bundle's own provenance records), so the number the gate needed was
available; it was reading the wrong population.

These tests pin the three-way split that makes the gate able to fail, and each of the three matters:
work a unit exists for and refused (the gap), work no unit exists for (the only exemption), and work
nobody decided (never either). A gate with no failing input is a gate that cannot fail, so the first
test below is the mutation.
"""

from __future__ import annotations

import pytest

from merlin.perf import lowering_coverage as LC
from merlin.perf.gate_phase import PHASE_FAIL, PHASE_REPORT


def _on_unit(macs: int = 1):
    return {"placement": LC.ACCELERATOR, "macs": macs}


def _off(cause: str | None, macs: int = 1):
    row = {"placement": LC.HOST, "macs": macs}
    if cause is not None:
        row["off_accelerator_cause"] = cause
    return row


# --------------------------------------------------------------------------------------------
# The mutation: the gate MUST fail on the population it was blind to
# --------------------------------------------------------------------------------------------


def test_work_a_unit_exists_for_and_refused_on_a_property_fails_the_gate() -> None:
    """THE CASE THAT SURVIVED. Every contraction eligible by class and shape, all of it refused for
    a property it carries, none of it on a unit. Refusing an INSTANCE is something only a unit that
    exists can do, so this population is addressable by construction."""
    verdict = LC.offload_verdict(
        [_off(LC.REFUSED_ON_PROPERTY, macs=4_089_184_256 // 160) for _ in range(160)], phase=PHASE_FAIL
    )
    assert verdict["status"] == LC.STATUS_ZERO_OFFLOAD
    assert verdict["blocking"] is True
    assert verdict["admitted"] is False
    assert verdict["work_addressable"] > 0
    with pytest.raises(LC.ZeroOffloadError):
        LC.require_offload(verdict)


def test_the_same_population_with_one_region_on_a_unit_does_not_fail() -> None:
    """The counterpart the mutation needs: the rule fires on zero offload and on nothing else."""
    verdict = LC.offload_verdict(
        [_on_unit(10)] + [_off(LC.REFUSED_ON_PROPERTY, 10) for _ in range(159)], phase=PHASE_FAIL
    )
    assert verdict["status"] == LC.STATUS_OFFLOADED
    assert verdict["blocking"] is False
    assert verdict["offload_of_addressable"] == pytest.approx(10 / 1600, rel=1e-6)
    LC.require_offload(verdict)  # does not raise


# --------------------------------------------------------------------------------------------
# The exemption, and the only thing that may earn it
# --------------------------------------------------------------------------------------------


def test_a_model_whose_work_has_no_unit_at_all_is_exempt() -> None:
    """Sound and kept: the absence of an accelerator's work is only a defect when there was some."""
    verdict = LC.offload_verdict([_off(LC.NO_UNIT, 1_000) for _ in range(20)], phase=PHASE_FAIL)
    assert verdict["status"] == LC.STATUS_OFFLOADED
    assert verdict["blocking"] is False
    assert verdict["work_addressable"] == 0
    assert verdict["offload_of_addressable"] is None, (
        "a ratio over an empty denominator is not a measured zero -- a reader must be able to tell "
        "a model with no acceleratable work from a compiler that accelerated none of it"
    )


def test_a_property_refusal_may_not_buy_the_no_unit_exemption() -> None:
    """The bug, stated as a rule: one region refused on a property is enough to make the population
    addressable, and the exemption must not apply to it."""
    verdict = LC.offload_verdict(
        [_off(LC.NO_UNIT, 1_000) for _ in range(20)] + [_off(LC.REFUSED_ON_PROPERTY, 1)], phase=PHASE_FAIL
    )
    assert verdict["status"] == LC.STATUS_ZERO_OFFLOAD
    assert verdict["work_addressable"] == 1


# --------------------------------------------------------------------------------------------
# Unknown: never an exemption, never a zero
# --------------------------------------------------------------------------------------------


def test_an_undecided_eligibility_makes_the_verdict_incomplete() -> None:
    """The live case: a capture whose precision is not yet in the form the backend will place from.
    The module says one dtype because integer preparation has not run, and reading 0% offload off it
    is not a conservative measurement -- it is a wrong one, and it is the kind that gets cited."""
    verdict = LC.offload_verdict([_off(LC.ELIGIBILITY_UNKNOWN, 4_089_184_256)], phase=PHASE_FAIL)
    assert verdict["status"] == "incomplete"
    assert verdict["blocking"] is False
    assert verdict["admitted"] is False, "incomplete is never a pass, at any phase"
    assert "nobody decided" in verdict["reason"]


def test_an_unknown_region_is_not_drowned_out_by_decided_ones() -> None:
    """One undecided region makes the population unestablished; reporting a ratio over the rest
    would present a coverage figure as a measurement."""
    verdict = LC.offload_verdict(
        [_on_unit(100), _off(LC.REFUSED_ON_PROPERTY, 10), _off(LC.ELIGIBILITY_UNKNOWN, 1)], phase=PHASE_FAIL
    )
    assert verdict["status"] == "incomplete"


@pytest.mark.parametrize("cause", [None, "", "some_new_refusal_kind"])
def test_an_absent_or_unrecognised_cause_is_unknown_not_an_exemption(cause) -> None:
    """A gate must not be able to acquire an exemption by being told nothing. A new refusal kind
    that nobody classified is undecided, so it cannot quietly become "no unit exists"."""
    verdict = LC.offload_verdict([_off(cause, 10)], phase=PHASE_FAIL)
    assert verdict["status"] == "incomplete"
    assert verdict["off_accelerator_by_cause"] == {LC.ELIGIBILITY_UNKNOWN: 1}


def test_a_region_with_unknown_extents_still_counts_as_one_unit_of_work() -> None:
    """Weighing it zero would let a model pass by having been unreadable."""
    verdict = LC.offload_verdict(
        [{"placement": LC.HOST, "off_accelerator_cause": LC.REFUSED_ON_PROPERTY}], phase=PHASE_FAIL
    )
    assert verdict["status"] == LC.STATUS_ZERO_OFFLOAD
    assert verdict["work_addressable"] == 1


# --------------------------------------------------------------------------------------------
# The classification itself -- one function, because two derivations of it will differ
# --------------------------------------------------------------------------------------------


def test_a_class_with_no_unit_is_the_exemption() -> None:
    assert LC.off_accelerator_cause(class_has_unit=False) == LC.NO_UNIT


def test_a_class_with_a_unit_that_refused_this_instance_is_the_gap() -> None:
    assert LC.off_accelerator_cause(class_has_unit=True) == LC.REFUSED_ON_PROPERTY


def test_an_underivable_capability_is_unknown_not_an_exemption() -> None:
    """An unestablished capability read as "no unit" is an exemption granted by not looking."""
    assert LC.off_accelerator_cause(class_has_unit=None) == LC.ELIGIBILITY_UNKNOWN


def test_a_refusal_decided_on_a_precision_the_backend_has_not_seen_yet_is_unknown() -> None:
    """THE ROOT CAUSE, as a rule. The census reads the precision off the module; the router is TOLD
    it. When quantization is still expressed as quantize/dequantize pairs around wide tensors, the
    module's own type is the pre-preparation one and a refusal decided on it decides nothing --
    measured, 0 contractions eligible against 54 the router placed."""
    assert LC.off_accelerator_cause(class_has_unit=True, precision_stated=False) == LC.ELIGIBILITY_UNKNOWN


def test_an_unstated_precision_does_not_buy_the_exemption_either() -> None:
    """It must not collapse into NO_UNIT on the way past REFUSED_ON_PROPERTY."""
    assert LC.off_accelerator_cause(class_has_unit=True, precision_stated=False) != LC.NO_UNIT


def test_the_classification_feeds_the_verdict_end_to_end() -> None:
    """The population the ResNet-50 census reported as exempt, classified correctly, fails."""
    rows = [
        {
            "placement": LC.HOST,
            "macs": 25_557_402,
            "off_accelerator_cause": LC.off_accelerator_cause(class_has_unit=True),
        }
        for _ in range(160)
    ]
    assert LC.offload_verdict(rows, phase=PHASE_FAIL)["status"] == LC.STATUS_ZERO_OFFLOAD
    # ...and the same population, decided on a precision that has not been prepared yet, is
    # incomplete rather than a zero-offload claim the evidence does not support.
    unknown = [
        dict(row, off_accelerator_cause=LC.off_accelerator_cause(class_has_unit=True, precision_stated=False))
        for row in rows
    ]
    assert LC.offload_verdict(unknown, phase=PHASE_FAIL)["status"] == "incomplete"


# --------------------------------------------------------------------------------------------
# Phasing
# --------------------------------------------------------------------------------------------


def test_the_report_phase_computes_the_whole_verdict_and_blocks_nothing() -> None:
    """Where this lands. Flipping it to fail today would fail every whole-model compile against the
    current dtype situation, which is the real state and not a regression this change introduced."""
    verdict = LC.offload_verdict([_off(LC.REFUSED_ON_PROPERTY, 10)], phase=PHASE_REPORT)
    assert verdict["status"] == LC.STATUS_ZERO_OFFLOAD
    assert verdict["blocking"] is False
    LC.require_offload(verdict)  # does not raise at the report phase


def test_an_unknown_phase_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="phase must be one of"):
        LC.offload_verdict([], phase="enforce")


def test_the_gate_has_a_production_caller() -> None:
    """A gate nothing calls is indistinguishable from a gate that always passes."""
    from merlin.targetgen import capsule_grade as CG  # noqa: PLC0415

    out = CG.model_execution_check(
        {
            "mesh_execution": {
                "dispatch_ledger": [
                    {"ordinal": 0, "symbol": "k0", "lane": "on_mesh", "status": "pass"},
                    {
                        "ordinal": 1,
                        "symbol": "k1",
                        "lane": "host_fallback",
                        "status": "pass",
                        "mesh_decline": "reduction depth exceeds the tile edge",
                    },
                ]
            }
        }
    )
    assert out["offload"]["schema"] == "offload_verdict_v1"
    assert out["offload"]["phase"] == PHASE_REPORT


def test_the_grader_cannot_decide_eligibility_from_a_dispatch_ledger_alone() -> None:
    """A ledger says where calls WENT, not what could have taken them. Only the runtime's own
    selector clause proves a unit existed and refused; everything else is undecided, and saying so
    is better than a zero-offload claim the evidence does not support."""
    from merlin.targetgen import capsule_grade as CG  # noqa: PLC0415

    out = CG.model_execution_check(
        {
            "mesh_execution": {
                "dispatch_ledger": [
                    {
                        "ordinal": 0,
                        "symbol": "k0",
                        "lane": "scalar_rvv_lane",
                        "status": "pass",
                        "executor": "a host library",
                    },
                ]
            }
        }
    )
    assert out["offload"]["status"] == "incomplete"
