"""Measured choices never bypass variant correctness or compilation identity."""

import pytest

from merlin.semantic_compiler.codegen import CodegenOptions, choose_reduction_unroll, codegen_options


def profile():
    return {
        "schema": "merlin.native_tuning_profile.v1",
        "key": {"source": "a"},
        "objective": "device_cycles",
        "variants": [
            {
                "factor": f,
                "status": "passed",
                "correctness": "passed",
                "bytes": 100 * f,
                "device_cycles": [100 // f] * 5,
            }
            for f in (1, 2, 4, 8)
        ],
    }


def test_profile_selection_rejects_failed_variants_and_obeys_identity():
    value = profile()
    value["variants"][-1] = {"factor": 8, "status": "numerical_mismatch"}
    with codegen_options(CodegenOptions(tuning_profile=value)):
        assert choose_reduction_unroll({"source": "a"})[0] == 4
        assert choose_reduction_unroll({"source": "b"})[0] == 1
    assert choose_reduction_unroll({"source": "a"})[0] == 1
    with codegen_options(CodegenOptions("2", tuning_profile=value)):
        assert choose_reduction_unroll({"source": "a"})[0] == 2


def test_profile_ties_prefer_smaller_program_and_options_restore_on_error():
    value = profile()
    for row in value["variants"]:
        row["device_cycles"] = [50] * 5
    with codegen_options(CodegenOptions(tuning_profile=value)):
        assert choose_reduction_unroll(value["key"])[0] == 1
    with pytest.raises(ValueError):
        with codegen_options(CodegenOptions("4")):
            raise ValueError("test failure")
    assert choose_reduction_unroll({})[0] == 1


def test_matching_all_failed_profile_cannot_publish_an_unchecked_default():
    value = profile()
    for row in value["variants"]:
        row["status"] = "search_timeout"
    with codegen_options(CodegenOptions(tuning_profile=value)):
        with pytest.raises(ValueError, match="no checked variant"):
            choose_reduction_unroll(value["key"])
        assert choose_reduction_unroll({"source": "different"})[0] == 1


@pytest.mark.parametrize("changed", ("samples", "correctness", "factor", "missing"))
def test_malformed_profiles_cannot_select_a_variant(changed):
    value = profile()
    if changed == "samples":
        value["variants"][0]["device_cycles"] = [1]
    elif changed == "correctness":
        value["variants"][0]["correctness"] = "unknown"
    elif changed == "factor":
        value["variants"][0]["factor"] = True
    else:
        value["variants"].pop()
    with pytest.raises(ValueError):
        CodegenOptions(tuning_profile=value)
