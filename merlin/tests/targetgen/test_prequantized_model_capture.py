"""An authored integer graph with floating host islands must keep its own arithmetic.

The MicroViT example has floating image/sensor inputs and 12 explicit int8 GEMMs.
Applying a static PT2E recipe to it attaches a histogram observer to an int8 GEMM
operand and crashes during calibration. The example's declaration must select the
generic already-materialized path and the captured program must still prove int8
contractions, rather than merely avoiding the observer crash.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common import mlir_query as query
from merlin.common.paths import repo_root
from merlin.targetgen import capsule_golden as golden
from merlin.targetgen.capsule_source import (
    M2MUnavailable,
    PytorchRefSource,
    _model_output_numeric_policy,
    _require_integer_contraction,
    write_model_capsule,
)
from merlin.targetgen.corpus_spec import CorpusBinding


def test_model_numeric_policy_uses_captured_output_not_operand_dtype() -> None:
    binding = SimpleNamespace(atol=0.03125, rtol=0.02)
    assert _model_output_numeric_policy([{"shape": [1], "dtype": "f32"}], [[1.0]], binding) == {
        "compare": "tolerance_float",
        "dtype": "f32",
        "atol": 0.03125,
        "rtol": 0.02,
    }
    assert _model_output_numeric_policy([{"shape": [1], "dtype": "i8"}], [[1.0]], binding) == {
        "compare": "exact_int",
        "dtype": "i8",
    }
    with pytest.raises(M2MUnavailable, match="inexact/out-of-range"):
        _model_output_numeric_policy([{"shape": [1], "dtype": "i8"}], [[1.5]], binding)
    with pytest.raises(M2MUnavailable, match="different output dtypes"):
        _model_output_numeric_policy(
            [{"shape": [1], "dtype": "i8"}, {"shape": [1], "dtype": "f32"}], [[1.0], [1.0]], binding
        )


def test_microvit_example_captures_its_existing_integer_graph(tmp_path: Path) -> None:
    root = repo_root()
    recipe = yaml.safe_load((root / "examples/gemmini/phase0/recipe.yaml").read_text())
    entry = next(row for row in recipe["capsules"] if row["name"] == "M2_microvit_gemmini")
    assert entry["capture_quantization"] == "already_materialized"

    source = PytorchRefSource(timeout=900)
    if not source.available():
        pytest.skip("model2MLIR capture environment unavailable")
    captured = source.capture_loader(
        root / entry["loader"], entry["operand_dtype"], workdir=tmp_path, already_quantized=True
    )
    assert captured.meta["capture_quantization"] == "already_materialized"
    assert captured.meta["quantization_stats"]["applied"] is False
    assert captured.meta["opaque"] == 0
    module = query.parse(captured.linalg_mlir)
    integer_matmuls = [
        op
        for op in query.walk(module, "linalg.matmul")
        if len(op.operands) >= 2 and all(query.type_shape_dtype(value.type)[1] == "i8" for value in op.operands[:2])
    ]
    assert len(integer_matmuls) >= 12


def test_host_island_example_preserves_its_authored_integer_matmuls(tmp_path: Path) -> None:
    """An int8→float host island→int8 graph is not an unconverted PT2E model."""
    root = repo_root()
    recipe = yaml.safe_load((root / "examples/gemmini/phase0/recipe.yaml").read_text())
    entry = next(row for row in recipe["capsules"] if row["name"] == "M3_host_island_seam_gemmini")
    assert entry["capture_quantization"] == "already_materialized"
    assert entry["loader"] == "examples/gemmini/phase0/inputs/host_island_seam.py"

    source = PytorchRefSource(timeout=900)
    if not source.available():
        pytest.skip("model2MLIR capture environment unavailable")
    captured = source.capture_loader(
        root / entry["loader"], entry["operand_dtype"], workdir=tmp_path, already_quantized=True
    )
    assert captured.meta["capture_quantization"] == "already_materialized"
    assert captured.meta["quantization_stats"]["applied"] is False
    assert captured.meta["materialized_contractions"] >= 2
    assert captured.meta["opaque"] == 0
    assert captured.meta["input_abi"] == [{"shape": [16, 32], "dtype": "i8"}]
    assert captured.meta["output_abi"] == [{"shape": [16, 32], "dtype": "i8"}]

    binding = CorpusBinding(
        target="gemmini",
        tile_dim=16,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0"],
        compare="exact_int",
        classes_for=lambda **_: ["MATMUL"],
    )
    cached = SimpleNamespace(m2m_dir=source.m2m_dir, capture_loader=lambda *args, **kwargs: captured)
    output = write_model_capsule({**entry, "cat": "model"}, binding, tmp_path / "corpus", source=cached)
    capsule = yaml.safe_load((output / "capsule.yaml").read_text())
    assert capsule["numeric_policy"] == {"compare": "exact_int", "dtype": "i8"}
    assert capsule["operation"]["attributes"]["materialized_contractions"] == 2
    assert golden.golden(capsule, output) == yaml.safe_load((output / "golden.yaml").read_text())["outputs"]


def test_static_w8a8_recipe_cannot_accept_narrow_result_as_wide_accumulation() -> None:
    """The same integer graph is not proof that a static PT2E recipe ran.

    Its authored GEMMs return i8; a recipe claiming W8A8 with i32 accumulation
    needs to show that different arithmetic in the captured program.
    """
    program = (repo_root() / "merlin/contract/capsules/model/M2_microvit_gemmini/capsule.interface.mlir").read_text()
    recipe = {"activation": {"dtype": "int8", "mode": "static"}, "weight": {"dtype": "int8"}}
    with pytest.raises(M2MUnavailable, match="no i8×i8→i32 integer contraction"):
        _require_integer_contraction(program, scheme=None, recipe=recipe)
