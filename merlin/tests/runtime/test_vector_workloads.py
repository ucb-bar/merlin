"""Synthetic vector fixtures carry caller-selected identity, not a default accelerator."""

import pytest
from vector_workloads import RUNGS, build

from merlin.runtime import reference_outputs, simulate


@pytest.mark.parametrize("rung", sorted(RUNGS))
@pytest.mark.parametrize("length", [1, 7, 65])
def test_vector_workload_identity_is_explicit_and_shapes_are_parameterized(rung, length):
    first = build(rung, target="fixture_vector_a", backend="fixture_engine_a", n=length)
    second = build(rung, target="fixture_vector_b", backend="fixture_engine_b", n=length)
    assert first["target"] == "fixture_vector_a"
    assert second["target"] == "fixture_vector_b"
    assert first["backend"] == "fixture_engine_a"
    assert second["backend"] == "fixture_engine_b"
    assert first["tensors"] == second["tensors"]
    assert first["commands"] == second["commands"]
    assert all(tensor["shape"] == [length] for tensor in first["tensors"].values() if tensor["role"] == "input")
    assert simulate(first)["outputs"] == reference_outputs(first)


def test_no_implicit_target_or_backend():
    with pytest.raises(TypeError):
        build("VEC0")
    with pytest.raises(TypeError):
        build("VEC0", target="fixture_vector")
