"""Version-2 whole-model bundles preserve capture ABI and every model result."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import capsule_source
from merlin.targetgen.capsule_runner import _model_runtime_bundle, load_capsule


_ROOT = merlin_dir() / "contract/capsules/atlas/model"


def _capsule(name: str) -> dict:
    root = _ROOT / name
    required = (root / "capsule.weights.safetensors",
                root / "capsule.weights.safetensors.manifest.json")
    if not all(path.is_file() for path in required):
        pytest.skip("local regenerated model answer surfaces are unavailable")
    if not capsule_source._m2m_python().is_file():
        pytest.skip("model2MLIR torch interpreter is unavailable")
    return load_capsule(root)


@pytest.mark.parametrize("name", ["M0_small_llama_atlas", "M1_lstmnetvit_atlas"])
def test_capture_manifest_materializes_without_reexporting_loader_abi(name):
    cap = _capsule(name)
    source = Path(cap["__dir__"])
    with _model_runtime_bundle(cap, timeout=120) as (bundle, provenance, verify):
        assert provenance["version"] == 2
        assert provenance["construction"] == "frozen_capsule_assets_v2"
        assert provenance["validation"]["capture_manifest_validated"] is True
        assert provenance["validation"]["torch_export"] is False
        assert ((bundle / "weights.safetensors.manifest.json").read_bytes() ==
                (source / "capsule.weights.safetensors.manifest.json").read_bytes())
        verify()


def test_stateful_model_bundle_preserves_all_results_in_declared_order():
    cap = _capsule("M1_lstmnetvit_atlas")
    assert cap["operation"]["attributes"]["outs"] == ["Y0", "Y1", "Y2"]
    with _model_runtime_bundle(cap, timeout=120) as (bundle, _provenance, verify):
        order = json.loads((bundle / "output_order.json").read_text(encoding="utf-8"))
        goldens = np.load(bundle / "goldens.npz", allow_pickle=False)
        assert order == ["Y0", "Y1", "Y2"]
        assert [list(goldens[name].shape) for name in order] == [[1, 3], [3, 128], [3, 128]]
        assert np.array_equal(np.load(bundle / "golden.npy"), goldens["Y0"])
        verify()
