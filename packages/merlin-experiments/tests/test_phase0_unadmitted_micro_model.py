import pytest

from merlin_experiments.phase0.generation import withhold_unadmitted_micro_model as withhold

MICRO = {"name": "SY_micro_model", "micro_model": True, "source_role": "derived_sweep"}
UNSUPPORTED = {
    "status": "unsupported",
    "reason": "m2m_q.quantize.default: operand_dtype 'fp32' not in ['int8']",
    "placements": {"host": 3},
}


def test_a_synthesized_micro_model_the_host_lane_cannot_admit_is_withheld_with_its_verbatim_reason():
    row = withhold(MICRO, UNSUPPORTED, must_refuse=False)
    assert row["capsule"] == "SY_micro_model"
    assert row["placements"] == {"host": 3} and row["placements"] is not UNSUPPORTED["placements"]
    assert row["reason"].endswith(UNSUPPORTED["reason"])
    assert row["reason"].startswith("emitted program not admitted by the selected host lane")


@pytest.mark.parametrize(
    "entry,observed,must_refuse",
    [
        ({**MICRO, "micro_model": False}, UNSUPPORTED, False),
        ({"name": "SY_x", "source_role": "derived_sweep"}, UNSUPPORTED, False),
        ({**MICRO, "source_role": "model_derived"}, UNSUPPORTED, False),
        ({**MICRO, "source_role": "materialized_iteration_capture"}, UNSUPPORTED, False),
        ({k: v for k, v in MICRO.items() if k != "source_role"}, UNSUPPORTED, False),
        (MICRO, UNSUPPORTED, True),
        (MICRO, {**UNSUPPORTED, "status": "unknown"}, False),
        (MICRO, {**UNSUPPORTED, "status": "refused"}, False),
        (MICRO, {**UNSUPPORTED, "status": "admitted"}, False),
        (MICRO, {"reason": "no status"}, False),
    ],
)
def test_every_other_unadmitted_emission_stays_fatal(entry, observed, must_refuse):
    assert withhold(entry, observed, must_refuse=must_refuse) is None
