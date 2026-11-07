"""The pre-extraction feature vocabulary survives backend composition."""

from __future__ import annotations

import json
from pathlib import Path

from merlin.llvmlower.backend_spi import load_vector_codegen
from merlin.llvmlower.shared_feature_registry import eager_features, get_shared_feature

_BASELINE = Path(__file__).with_name("data") / "pre_extraction_feature_names.json"


def test_pre_extraction_names_are_owned_by_shared_or_vector_provider():
    names = set(json.loads(_BASELINE.read_text(encoding="utf-8")))
    assert len(names) == 95
    shared = {feature.name for feature in eager_features()}
    for name in names - shared:
        try:
            shared.add(get_shared_feature(name).name)
        except KeyError:
            pass
    vector = set(load_vector_codegen("rvv").feature_names())
    assert names - (shared | vector) == {"opu_matmul"}
    assert "opu_matmul" not in vector


def test_shared_and_vector_hooks_still_change_the_selected_pipeline():
    shared = get_shared_feature("fuse_elementwise_after_generalize")
    passes = ["func.func(linalg-generalize-named-ops)", "one-shot-bufferize"]
    assert shared.edit_pipeline is not None
    assert shared.edit_pipeline(passes) == [
        "func.func(linalg-generalize-named-ops)",
        "func.func(linalg-fuse-elementwise-ops)",
        "one-shot-bufferize",
    ]
    vector = load_vector_codegen("rvv")
    features = vector.normalize_features({"fused_vfmacc_tiled"})
    baseline = vector.baseline_schedule_text()
    assert vector.apply_schedule(baseline, features) != baseline


def test_chip_marker_is_only_in_the_chip_bound_backend():
    chip = load_vector_codegen("kodiak")
    assert "opu_matmul" in chip.feature_names()
    assert "opu_matmul" in chip.normalize_features({"opu_matmul"})
    assert "opu_matmul" not in load_vector_codegen("rvv").feature_names()
