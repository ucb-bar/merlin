"""The external functional rules are commitments, never observed placement."""

import copy
import json

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.functional_truth import expectation
from merlin.targetgen.software_spec import load_software_spec
from merlin.targetgen.spec_fact_drift import resolve_spec


@pytest.fixture(scope="module")
def admission():
    """The authored spec with its standalone affine form resolved as the selected facts resolve it."""
    spec = load_software_spec(repo_root() / "examples/gemmini/target/software-spec.yaml")
    row = next(row for row in spec["operations"] if row["id"] == "elementwise_map_standalone")
    row.pop("hardware")
    row["placement"] = "accelerator"
    row["signature"]["dtypes"] = ["int8"]
    return {"spec": resolve_spec(spec, {})[0]}


def test_lane_commitments_are_value_and_placement_independent(admission):
    data = json.loads((repo_root() / "merlin/tests/data/exactness/bounded_final_gate.json").read_text())
    for case in data["selected_cases"]:
        baseline = expectation(case, admission=admission)
        changed = copy.deepcopy(case)

        def mutate(v):
            if isinstance(v, dict):
                if v.get("kind") == "tensor":
                    v["values"] = ["intentionally unavailable"]
                else:
                    for value in v.values():
                        mutate(value)
            elif isinstance(v, list):
                for value in v:
                    mutate(value)

        mutate(changed)
        changed["current_run13_lane"] = "arbitrary"
        assert expectation(changed, admission=admission) == baseline
        assert baseline["lane"] in {"device", "no-work", "host-guard"}


def test_strict_placement_check_refuses_both_directions_and_device_no_work():
    from merlin.targetgen.functional_truth import require_placement

    for lane in ("device", "host-guard", "no-work"):
        commitment = dict(lane=lane, reason="semantic test commitment")
        require_placement(commitment, device=(lane == "device"))
        with pytest.raises(ValueError, match="truth mismatch"):
            require_placement(commitment, device=(lane != "device"))


def _scalar_case(overload, schema, dtype, other):
    tensor = {"kind": "tensor", "dtype": dtype, "shape": [2], "stride": [1], "storage_offset": 0, "values": [0.0, 1.0]}
    return {
        "overload": overload,
        "case_id": overload + "::ruling",
        "schema": schema,
        "arguments": {"kind": "tuple", "args": {"kind": "tuple", "items": [tensor, other]}, "kwargs": {}},
        "expected": dict(tensor),
        "partition_assignment": {"dtype": dtype},
    }


def test_exact_identities_are_bit_exact_and_near_identities_are_not(admission):
    # Ruling 2026-10-20: real x*1 and x/1 are exact identities (movement), graded on eager bits.
    mul = "aten::mul.Scalar(Tensor self, Scalar other) -> Tensor"
    div = "aten::div.Scalar(Tensor self, Scalar other) -> Tensor"
    for dtype in ("float32", "float16", "bfloat16"):
        for overload, schema in (("aten.mul.Scalar", mul), ("aten.div.Scalar", div)):
            assert (
                expectation(_scalar_case(overload, schema, dtype, 1.0), admission=admission)["precision_class"]
                == "bit_exact_vs_eager"
            )
            assert (
                expectation(_scalar_case(overload, schema, dtype, 2.0), admission=admission)["precision_class"]
                == "quantized_exact"
            )
    # A complex scale by (1+0j) follows eager's full complex product: not an exact identity.
    assert (
        expectation(_scalar_case("aten.mul.Scalar", mul, "complex64", 1.0), admission=admission)["precision_class"]
        == "quantized_exact"
    )
