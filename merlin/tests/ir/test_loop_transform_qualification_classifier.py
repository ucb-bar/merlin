"""SDK-free refusal/resource classification; no mocked proof acceptance."""

import pytest
from loop_transform_qualification_classifier import classify

PATH = "/bound/Query.lean"
QUERY = """namespace Merlin.GraphStatePipelineInstance
namespace F4
theorem body_group : True := by
  fail_if_success trivial
end F4
end Merlin.GraphStatePipelineInstance
"""
ERROR = PATH + ":4:2: error: unsolved goals\n"


@pytest.mark.parametrize(
    "marker",
    [
        "maximum number of heartbeats",
        "maximum recursion depth",
        "out of memory",
        "stack overflow",
        "internal error",
        "panic",
        "unknown module prefix",
        "failed to read file",
        "object file unavailable",
        "unknown identifier",
    ],
)
def test_resource_or_infrastructure_never_counts_as_refusal(marker):
    assert classify("middle_wrong_call", 1, ERROR + marker, "", QUERY, PATH)["classification"] == "UNAVAILABLE"


@pytest.mark.parametrize("code", [-11, -9, 2, None, 0])
def test_signal_timeout_unknown_exit_is_unavailable(code):
    assert classify("middle_wrong_call", code, ERROR, "", QUERY, PATH)["classification"] == "UNAVAILABLE"


def test_requires_the_expected_actual_bound_obligation():
    assert classify("final_wrong_capture", 1, ERROR, "", QUERY, PATH)["classification"] == "UNAVAILABLE"
    assert (
        classify("middle_wrong_call", 1, ERROR, "", QUERY, "/different/Query.lean")["classification"] == "UNAVAILABLE"
    )


def test_ordinary_obligation_failure_is_attributed():
    result = classify("middle_wrong_call", 1, ERROR, "", QUERY, PATH)
    assert result["classification"] == "LEAN_COMPONENT_OBLIGATION_REFUSAL"
    assert result["root"] == "Merlin.GraphStatePipelineInstance.F4.body_group"


def test_generic_error_text_is_not_sufficient():
    error = ERROR.replace("unsolved goals", "something unexpected")
    assert classify("middle_wrong_call", 1, error, "", QUERY, PATH)["classification"] == "UNAVAILABLE"
