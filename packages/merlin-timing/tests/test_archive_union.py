"""Synthetic group/identity contracts; no timing or hardware qualification."""

import copy

import pytest
from data_helpers import sample
from merlin_timing import archive_union as U
from merlin_timing import contracts as C


def another_variant(document):
    other = copy.deepcopy(document)
    cases = {}
    for case in other["cases"]:
        previous = case["case_id"]
        case["case_id"] = "new-" + previous
        case["elf_sha256"] = C.digest(["new original ELF", case["elf_sha256"]])
        cases[previous] = case
    for row in other["records"]:
        case = cases[row["case_id"]]
        row["case_id"] = case["case_id"]
        row["record_id"] = "new-" + row["record_id"]
        row["feature_provenance"]["execution_sha256"] = C.execution_identity(case)
        row["label"]["execution_sha256"] = C.execution_identity(case)
    return C.dataset(other)


def test_union_preserves_complete_rosters_original_splits_and_missing_obligations():
    first = sample()
    second = another_variant(first)
    second["records"].pop()
    before = copy.deepcopy((first, second))
    joined = U.join((first, second))
    assert (first, second) == before
    assert joined["cases"] == first["cases"] + second["cases"]
    assert joined["records"] == first["records"] + second["records"]
    assert {c["source_group_sha256"] for c in joined["cases"]} == {c["source_group_sha256"] for c in first["cases"]}
    joined["cases"][0]["case_id"] = "mutated result"
    assert first == before[0]


@pytest.mark.parametrize("damage", ["source_split", "source_group", "model", "record", "execution", "family"])
def test_union_never_resplits_repeats_or_reinterprets_original_evidence(damage):
    first = sample()
    second = another_variant(first)
    if damage in {"source_split", "source_group", "family"}:
        group = second["cases"][0]["source_group_sha256"]
        for case in second["cases"]:
            if case["source_group_sha256"] != group:
                continue
            if damage == "source_split":
                case["split"] = "calibration"
            elif damage == "source_group":
                case["source_group_sha256"] = C.digest("different leakage group")
            else:
                case["mechanism_family"] = "new family"
    elif damage == "model":
        second["domain"]["target_model_sha256"] = C.digest("different model")
        for row in second["records"]:
            row["feature_provenance"]["model_sha256"] = second["domain"]["target_model_sha256"]
    elif damage == "record":
        second["records"][0]["record_id"] = first["records"][0]["record_id"]
    else:
        case = second["cases"][0]
        case["elf_sha256"] = first["cases"][0]["elf_sha256"]
        second["records"][0]["label"]["execution_sha256"] = C.execution_identity(case)
        second["records"][0]["feature_provenance"]["execution_sha256"] = C.execution_identity(case)
    C.dataset(second)
    with pytest.raises(C.ContractError):
        U.join((first, second))


@pytest.mark.parametrize("roster", [[], (), (sample(),), tuple(sample() for _ in range(17))])
def test_union_is_bounded_and_never_an_implicit_single_archive_reinterpretation(roster):
    with pytest.raises(C.ContractError):
        U.join(roster)
