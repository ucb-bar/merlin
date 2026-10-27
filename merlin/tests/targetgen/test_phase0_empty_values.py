"""Empty declarations and rank-zero tensors retain their Phase 0 meaning."""

import pytest
import yaml

from merlin.targetgen import applications, cert_cost, conformance, corpus_spec, corpus_synth, memory_regime
from merlin.targetgen.address_space import Store


@pytest.mark.parametrize("budget_source", ["argument", "workload"])
def test_zero_synthesis_budget_refuses_required_members(budget_source):
    requirement = {
        "target": "synthetic",
        "cells": [{"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8", "alignment": "aligned"}],
        "boundaries": {
            "tile_edge": 16,
            "extent_probes": [{"M": "tile", "K": "tile", "N": "tile", "alignment": "aligned"}],
        },
    }
    selection = {"budget": 0} if budget_source == "argument" else {"workload_spec": {"max_synthesized_capsules": 0}}
    with pytest.raises(corpus_synth.SynthesisError, match="budget of 0"):
        corpus_synth.synthesize(requirement, capability_contract={}, **selection)


def test_zero_budget_allows_an_empty_synthesis():
    result = corpus_synth.synthesize(
        {"target": "synthetic", "cells": []},
        workload_spec={"max_synthesized_capsules": 0},
        capability_contract={},
    )
    assert result["capsules"] == []
    assert result["provenance"]["budget"] == 0


@pytest.mark.parametrize("tile", [0, 16])
def test_empty_admission_never_requires_an_application_anchor(monkeypatch, tile):
    monkeypatch.setattr(conformance, "admitted", lambda _target: {})
    monkeypatch.setattr(corpus_spec, "_tile_dim", lambda *_args: tile)
    from merlin.targetgen import target_registry

    monkeypatch.setattr(target_registry, "load_contract", lambda _target: {})
    monkeypatch.setattr(
        applications,
        "classify_captures",
        lambda *_args, **_kwargs: {
            "classes": [
                {
                    "family": "contraction",
                    "dtype": "i8",
                    "alignment": "aligned",
                    "regime": "fits_double",
                    "rank": 2,
                    "geometry": "square",
                    "M": 16,
                    "K": 16,
                    "N": 16,
                    "batch": 1,
                    "multiplicity": 1,
                    "work": 4096,
                    "work_complete": True,
                    "source": "capture",
                }
            ]
        },
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("a non-admitted class reached accelerator sizing")

    monkeypatch.setattr(applications, "size_class", unexpected)
    result = conformance._application_axis("synthetic", captures={"capture": "unused"}, certification_floor="L3")
    assert result["required"] == []
    assert len(result["refused"]) == 1
    assert "does not admit contraction/i8" in result["refused"][0]


@pytest.mark.parametrize("shape,rows", [([], 1), ([0], 0), ([16], 1)])
def test_memory_regime_sizes_scalar_and_zero_extent(tmp_path, shape, rows):
    (tmp_path / "capsule.yaml").write_text(yaml.safe_dump({"inputs": [{"name": "X", "shape": shape, "dtype": "i8"}]}))
    store = _store()
    result = memory_regime.capsule_regime(tmp_path, "synthetic", store=store, capacity=16)
    assert result["rows"] == rows
    assert "unsized_inputs" not in result


@pytest.mark.parametrize("shape", [None, ["unknown"]])
def test_memory_regime_keeps_unavailable_shape_unknown(tmp_path, shape):
    (tmp_path / "capsule.yaml").write_text(yaml.safe_dump({"inputs": [{"name": "X", "shape": shape, "dtype": "i8"}]}))
    result = memory_regime.capsule_regime(tmp_path, "synthetic", store=_store(), capacity=16)
    assert result["regime"] == memory_regime.UNKNOWN


@pytest.mark.parametrize(
    "results,elements", [("tensor<f32>", 1), ("tensor<0xf32>", 0), ("(tensor<f32>, tensor<2xf32>)", 3)]
)
def test_certification_counts_rank_zero_results(results, elements):
    source = f"module {{ func.func private @result() -> {results} }}"
    assert cert_cost._linalg_result_elements(source) == elements


def _store():
    return Store(
        name="operand",
        nbytes=256,
        depth=16,
        row_elems=16,
        element_dtype="i8",
        element_bits=8,
        row_bytes=16,
        total_rows=16,
        banks=1,
    )
