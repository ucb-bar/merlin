from __future__ import annotations

import importlib.util

import pytest

from merlin.targetgen.core_aten_bounded import (
    BoundedCoverageProfile,
    _greedy_pairwise_rows,
    _one_overload_pool,
    bounded_summary,
)
from merlin.targetgen.core_aten_cases import case_document, core_aten_cases, eager_case_observation
from merlin.targetgen.core_aten_eval import _validate_observation

requires_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")


def test_pairwise_rows_cover_every_nonbaseline_value_pair() -> None:
    domains = {"a": ["a0", "a1", "a2"], "b": ["b0", "b1"], "c": ["c0", "c1", "c2"]}
    baseline = {"a": "a0", "b": "b0", "c": "c0"}
    rows = _greedy_pairwise_rows(domains, baseline)
    for left, right in (("a", "b"), ("a", "c"), ("b", "c")):
        for left_value in domains[left][1:]:
            for right_value in domains[right][1:]:
                assert any(row[left] == left_value and row[right] == right_value for row in rows)


@requires_torch
def test_bounded_pool_is_deterministic_exact_and_smaller_than_candidates() -> None:
    overload = "aten.add.Tensor"
    canonical = case_document(core_aten_cases()[overload])
    profile = BoundedCoverageProfile()
    first = _one_overload_pool(overload, canonical, profile)
    second = _one_overload_pool(overload, canonical, profile)
    assert first["selected_case_ids"] == second["selected_case_ids"]
    assert first["cover"]["status"] == "optimal"
    assert first["cover"]["selected_union_matches_denominator"] is True
    assert first["selected_count"] < first["candidate_count"]
    assert {"dtype", "shape", "layout", "values", "control:alpha"} <= set(first["feasible_domains"])
    assert "broadcast_pair" in first["feasible_domains"]["shape"]


def test_v2_profile_includes_int8_and_refuses_a_misleading_custom_bound() -> None:
    dtypes = ("bool", "int8", "int32", "int64", "float16", "bfloat16", "float32", "complex64")
    assert BoundedCoverageProfile().dtypes == dtypes
    with pytest.raises(ValueError, match="fixed ranks"):
        BoundedCoverageProfile(ranks=(0,))


def test_bounded_summary_qualifies_the_claim() -> None:
    document = {
        "complete": True,
        "pytorch_version": "test",
        "overload_count": 193,
        "raw_candidate_count": 110,
        "candidate_count": 100,
        "eqsat_eliminated_candidate_count": 10,
        "selected_count": 20,
        "witnessed_obligation_count": 500,
        "rejected_attempt_count": 7,
        "profile_sha256": "abc",
        "generator_sha256": "def",
        "engines": {"minimum_certificate": "unit-test certificate"},
        "candidate_budget_exhausted_overloads": [],
    }
    text = bounded_summary(document)
    assert "exact minimum" in text
    assert "does not prove correctness for arbitrary PyTorch models" in text


@requires_torch
@pytest.mark.parametrize(
    "overload,repeat,replay",
    [
        pytest.param("aten._native_batch_norm_legit.default", False, True, id="batch_norm_layout_variants"),
        pytest.param("aten.native_layer_norm_backward.default", True, False, id="layer_norm_backward"),
        pytest.param("aten.native_group_norm_backward.default", True, True, id="group_norm_backward"),
        pytest.param("aten.resize_.default", True, False, id="metadata_only_resize"),
    ],
)
def test_pools_are_reproducible_and_replayable(overload, repeat, replay) -> None:
    """Pools rebuild identically; selected cases replay (layout variants do not overlap running statistics)."""
    canonical = case_document(core_aten_cases()[overload])
    first = _one_overload_pool(overload, canonical, BoundedCoverageProfile())
    if repeat:
        assert first == _one_overload_pool(overload, canonical, BoundedCoverageProfile())
    for identifier in first["selected_case_ids"] if replay else ():
        case = first["candidates"][identifier]
        _validate_observation(case, eager_case_observation(case))
    if overload == "aten.resize_.default":
        assert all(
            "values" not in tensor
            for case in first["candidates"].values()
            for tensor in case["post_arguments"]["args"]
            if isinstance(tensor, dict) and tensor.get("kind") == "tensor"
        )


@requires_torch
@pytest.mark.parametrize(
    ("overload", "reason"),
    [
        ("aten._fft_c2r.default", "FFT overflow"),
        ("aten._fft_r2c.default", "FFT overflow"),
        ("aten.atan2.out", "mutation witness"),
    ],
)
def test_known_nonreplayable_cells_are_rejected(overload: str, reason: str) -> None:
    canonical = case_document(core_aten_cases()[overload])
    pool = _one_overload_pool(overload, canonical, BoundedCoverageProfile())
    assert any(reason in rejection["reason"] for rejection in pool["rejected"])


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("bad pointer 0x7f12ab34cd56ef here", "bad pointer 0x<runtime-address> here"),
        ("short 0x1234567 stays", "short 0x1234567 stays"),
        ("0xABCDEF0123 and 0xabcdef0123456789", "0x<runtime-address> and 0x<runtime-address>"),
        ("size 12345678901 bytes", "size <runtime-large-value> bytes"),
        ("value -12345678901;", "value <runtime-large-value>;"),
        ("value -12345678901.", "value -12345678901."),
        ("range [-1234567890, 9999999999]", "range [<runtime-large-value>, <runtime-large-value>]"),
        ("x-1234567890", "x-<runtime-large-value>"),
        ("1234567890.5 and v1234567890 and 1234567890abc", "1234567890.5 and v1234567890 and 1234567890abc"),
        ("small 123456789 stays", "small 123456789 stays"),
        ("", ""),
    ],
)
def test_runtime_value_masking_matches_the_documented_patterns(message: str, expected: str) -> None:
    from merlin.targetgen.core_aten_bounded import mask_runtime_values

    assert mask_runtime_values(message) == expected
