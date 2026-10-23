"""The native host-exact gate keeps sealed metadata, aliases and mutation."""

import copy

import pytest


def test_native_policy_keeps_sealed_metadata_aliases_and_mutation():
    pytest.importorskip("torch")
    from merlin.targetgen.core_aten_eval import _validate_observation

    original = dict(
        kind="tensor", dtype="float32", shape=[1], stride=[1], storage_offset=0, requires_grad=False, values=[0.0]
    )
    case = dict(
        case_id="independent",
        expected=original,
        post_arguments={},
        comparison="torch_close",
        comparison_parameters=dict(rtol=0.0, atol=0.0),
        mutated_arguments=[],
        output_input_aliases=[],
    )
    observation = dict(
        status="executed",
        execution_kind="simulator",
        output={**original, "values": [1.0]},
        post_arguments={},
        mutated_arguments=[],
        output_input_aliases=[],
        numeric_reference=dict(
            policy="native_host_exact",
            admission_status="admitted",
            gate=dict(
                case_id="independent",
                status="pass",
                results=[dict(status="pass")],
                comparison="exact_rational_error_not_greater_than_native_per_component",
            ),
        ),
    )
    _validate_observation(case, observation)
    for field, value in [("stride", [2]), ("storage_offset", 1), ("requires_grad", True)]:
        changed = copy.deepcopy(observation)
        changed["output"][field] = value
        with pytest.raises(AssertionError):
            _validate_observation(case, changed)
    for field, value in [("case_id", "other"), ("status", "mismatch"), ("comparison", "torch_close")]:
        changed = copy.deepcopy(observation)
        changed["numeric_reference"]["gate"][field] = value
        with pytest.raises(AssertionError, match="independent exactness gate"):
            _validate_observation(case, changed)
    for field in ["mutated_arguments", "output_input_aliases"]:
        changed = copy.deepcopy(observation)
        changed[field] = [0]
        with pytest.raises(AssertionError):
            _validate_observation(case, changed)
