"""Bundle layout, approximation declarations and content-addressed storage behave as the contract states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.model_check import bundle as B
from merlin.model_check import catalog as CAT
from merlin.model_check.approximations import ApproximationError, declared
from merlin.model_check.models import load


def _models():
    return load(repo_root() / "examples" / "gemmini" / "numerics" / "models.yaml")


def test_stage_directories_follow_the_sealed_capture_layout(tmp_path):
    models = _models()
    assert B.stage_dir(tmp_path, models.model("resnet50"), "model") == tmp_path / "resnet50/run/capture"
    assert B.stage_dir(tmp_path, models.model("smolvla"), "flow_denoise") == (
        tmp_path / "smolvla/run/capture/stages/flow_denoise"
    )


def test_blobs_are_stored_once_by_content_and_typed_by_header(tmp_path):
    store = CAT.BlobStore(tmp_path)
    a = np.arange(12, dtype=np.int8).reshape(3, 4)
    k1, k2 = store.put(a), store.put(a.copy())
    assert k1 == k2 and len(list(tmp_path.glob("*.npy"))) == 1
    assert store.put(a.astype(np.int32)) != k1  # same values, different dtype: a different blob
    assert np.array_equal(store.get(k1), a)


def _manifest(tmp_path: Path, text: str) -> Path:
    (tmp_path / "manifest.yaml").write_text(text)
    return tmp_path


def test_no_numerics_block_means_every_kernel_is_exact(tmp_path):
    assert declared(_manifest(tmp_path, "artifact_type: x\n")) == []


def test_a_well_formed_approximation_is_returned(tmp_path):
    rows = declared(
        _manifest(
            tmp_path,
            "numerics:\n  approximations:\n    - id: attn_split\n      opcodes: [ATTENTION_QK]\n"
            "      description: d\n      reason: r\n",
        )
    )
    assert rows[0]["id"] == "attn_split"


@pytest.mark.parametrize(
    "text",
    [
        "numerics:\n  exactness: off\n",
        "numerics:\n  approximations:\n    - id: A\n      opcodes: [X]\n      description: d\n      reason: r\n",
        "numerics:\n  approximations:\n    - id: a\n      opcodes: []\n      description: d\n      reason: r\n",
        "numerics:\n  approximations:\n    - id: a\n      opcodes: [X]\n      description: d\n",
    ],
)
def test_a_malformed_declaration_is_refused(tmp_path, text):
    with pytest.raises(ApproximationError):
        declared(_manifest(tmp_path, text))
