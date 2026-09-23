"""A capsule that tests a sign-sensitive stage must be able to produce a negative value."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from merlin.common.paths import repo_root
from merlin.runtime.commandbuffer import DEFAULT_STIMULUS_RANGE, SIGNED_STIMULUS_RANGE, STIMULUS_RANGE_KEY


def _gate():
    spec = importlib.util.spec_from_file_location(
        "check_sign_coverage_under_test", repo_root() / "build_tools/scripts/check_sign_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capsule(root: Path, name: str, **fields) -> None:
    (root / "layers" / name).mkdir(parents=True)
    (root / "layers" / name / "capsule.yaml").write_text(
        yaml.safe_dump({"name": name, "label": "public", **fields}), encoding="utf-8"
    )


def test_only_a_sign_sensitive_capsule_on_a_non_negative_stimulus_is_flagged(tmp_path: Path) -> None:
    relu = {"operation": {"op": "matmul", "attributes": {"epilogue": ["relu"]}}}
    _capsule(tmp_path, "relu_default", **relu)
    _capsule(tmp_path, "relu_signed", **relu, **{STIMULUS_RANGE_KEY: list(SIGNED_STIMULUS_RANGE)})
    _capsule(tmp_path, "relu_positive_range", **relu, **{STIMULUS_RANGE_KEY: [0, 7]})
    _capsule(tmp_path, "plain_matmul", operation={"op": "matmul", "attributes": {"epilogue": []}})
    _capsule(tmp_path, "standalone_stage", operation={"op": "requant", "attributes": {}})
    _capsule(tmp_path, "hidden", label="hidden", **relu)
    assert _gate().unsigned(tmp_path) == ["relu_default", "relu_positive_range", "standalone_stage"]


def test_the_signed_range_is_the_defaults_own_negative_half() -> None:
    low, high = SIGNED_STIMULUS_RANGE
    assert high == DEFAULT_STIMULUS_RANGE[1] and low == -(high + 1)


def test_the_ledger_matches_the_tree() -> None:
    gate = _gate()
    assert sorted(gate._ledger()) == gate.unsigned(), (
        "regenerate with: python build_tools/scripts/check_sign_coverage.py --write"
    )


def test_an_entry_that_asks_for_a_signed_stimulus_gets_a_capsule_that_has_one() -> None:
    from merlin.targetgen import corpus_spec as CSP

    binding = CSP.CorpusBinding(
        target="t",
        tile_dim=16,
        operand_dtype="int8",
        accum_dtype="int32",
        integer=True,
        tiers=["L2", "L3"],
        compare="exact",
    )
    entry = {
        "cat": "layers",
        "name": "signed_member",
        "kind": "layer",
        "op": "matmul",
        "lhs": "A0",
        "weight": "W",
        "out": "Y0",
        "M": 16,
        "K": 16,
        "N": 16,
        "epilogue": ["relu"],
        "label": "public",
        "source_role": "derived_sweep",
        "source_reference": "test",
    }
    plain, _ = CSP.build(dict(entry), binding)
    assert STIMULUS_RANGE_KEY not in plain  # absent stays absent: no capsule changes
    signed, _ = CSP.build({**entry, "stimulus_range": list(SIGNED_STIMULUS_RANGE)}, binding)
    assert signed[STIMULUS_RANGE_KEY] == list(SIGNED_STIMULUS_RANGE)
    import pytest

    with pytest.raises(ValueError, match="two-element"):
        CSP.build({**entry, "stimulus_range": [0]}, binding)
