"""The shared successor selects explicit portable f32 suppliers and preserves f64."""

from copy import deepcopy

import pytest
import yaml
from merlin_experiments.phase1.feedback import private_linkage_support as linkage

from merlin.common.paths import repo_root
from merlin.targetgen.host_capabilities import validate_host_capabilities
from merlin.targetgen.host_linkage_contract import RUNTIME_SUPPLIER, runtime_recipe


def test_shared_portable_host_view_preserves_every_unrelated_native_declaration():
    root = repo_root() / "examples/gemmini/platform"
    old = yaml.safe_load((root / "host-capabilities-attention-v2.yaml").read_bytes())
    new = yaml.safe_load((root / "host-capabilities-portable-v3.yaml").read_bytes())
    validate_host_capabilities(new)
    a = {row["id"]: row for row in old["operations"]}
    b = {row["id"]: row for row in new["operations"]}
    changed = {name for name in a if a[name] != b[name]}
    assert changed == {
        "host_review_selected_libm_sinf_f32",
        "host_review_selected_libm_cosf_f32",
        "host_review_literal_base_pow_f32",
        "host_review_tanh_gelu_f32",
        "host_review_attention_exp_f32",
    }
    assert set(a) == set(b)
    for name in changed:
        original, successor = deepcopy(a[name]), deepcopy(b[name])
        for field in ("linkage_contract", "numerical_contract"):
            original.pop(field, None)
            successor.pop(field, None)
        assert original == successor
        assert b[name]["numerical_contract"]["status"] == "unreviewed"
        assert b[name]["linkage_contract"]["supplier"] == RUNTIME_SUPPLIER
    for name, row in b.items():
        if (row.get("source_body") or {}).get("operation", "").startswith("f64_"):
            assert row == a[name]


def _source(row):
    pending = linkage.begin("a" * 64, "b" * 64, 1)
    pending.update(
        count=1,
        occurrences=[
            {
                "schema": row["source_body"]["schema"],
                "operation": row["source_body"]["operation"],
                "contract": row["linkage_contract"],
            }
        ],
    )
    return {linkage.FIELD: pending}


def test_same_host_view_selects_tiny_smol_and_native_f64_without_policy_guessing():
    root = repo_root() / "examples/gemmini/platform"
    data = yaml.safe_load((root / "host-capabilities-portable-v3.yaml").read_bytes())
    rows = {row["id"]: row for row in data["operations"]}
    for policy in ("portable_exp_trig", "portable_neural_f32"):
        for name in (
            "host_review_selected_libm_sinf_f32",
            "host_review_selected_libm_cosf_f32",
            "host_review_attention_exp_f32",
        ):
            row = rows[name]
            assert runtime_recipe(row["linkage_contract"], policy)["policy"] == policy
            assert linkage.build_options(_source(row), policy) == {
                "host_math_symbols": row["linkage_contract"]["symbols"]
            }
    for name in ("host_review_literal_base_pow_f32", "host_review_tanh_gelu_f32"):
        row = rows[name]
        assert linkage.build_options(_source(row), "portable_neural_f32")["host_math_symbols"]
        with pytest.raises(ValueError):
            linkage.build_options(_source(row), "portable_exp_trig")
    for row in rows.values():
        if (row.get("source_body") or {}).get("operation") == "f64_math_sin":
            for policy in ("native", "portable_exp_trig", "portable_neural_f32"):
                assert linkage.build_options(_source(row), policy) == {"math_archive_symbols": ["sin"]}
    empty = {linkage.FIELD: linkage.begin("a" * 64, "b" * 64, 1)}
    assert linkage.build_options(empty, "native") == {}
