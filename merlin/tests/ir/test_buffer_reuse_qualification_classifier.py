"""SDK-free classifier controls; these do not mock positive proof authority."""

import pytest
from buffer_reuse_qualification_classifier import classify

PATH = "/bound/Materialization.lean"
QUERY = """namespace GraphMemoryAfterMaterialization
theorem accepted : True := by
  trivial
end GraphMemoryAfterMaterialization
"""
ERROR = PATH + ":3:2: error: unsolved goals\n"


@pytest.mark.parametrize(
    "marker",
    [
        "maximum number of heartbeats",
        "maximum recursion depth",
        "out of memory",
        "stack overflow",
        "stack space",
        "panic",
        "internal error",
        "failed to import",
        "unknown constant",
        "unknown identifier",
        "object file",
        "failed to open",
        "uncaught exception",
    ],
)
def test_resource_or_infrastructure_is_unavailable(marker):
    assert classify("final_wrong_seed", 1, ERROR + marker, "", QUERY, PATH)["status"] == "UNAVAILABLE"


@pytest.mark.parametrize("code", [-11, -9, None, 2, 0])
def test_signals_unknown_and_success_are_not_refusals(code):
    assert classify("final_wrong_seed", code, ERROR, "", QUERY, PATH)["status"] == "UNAVAILABLE"


def test_exact_root_and_source_required():
    assert classify("source_wrong_seed", 1, ERROR, "", QUERY, PATH)["status"] == "UNAVAILABLE"
    assert classify("final_wrong_seed", 1, ERROR, "", QUERY, "/other.lean")["status"] == "UNAVAILABLE"
    assert classify("unknown", 1, ERROR, "", QUERY, PATH)["status"] == "UNAVAILABLE"


def test_known_normal_refusal_is_attributed():
    result = classify("final_wrong_seed", 1, ERROR, "", QUERY, PATH)
    assert result["status"] == "EXPECTED_COMPONENT_OBLIGATION_REFUSAL"
    assert result["root"] == "GraphMemoryAfterMaterialization.accepted"


def test_unknown_diagnostic_is_not_semantic_refusal():
    assert (
        classify("final_wrong_seed", 1, ERROR.replace("unsolved goals", "mystery failure"), "", QUERY, PATH)["status"]
        == "UNAVAILABLE"
    )
