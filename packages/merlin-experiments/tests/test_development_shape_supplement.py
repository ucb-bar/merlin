"""Public default/connector evidence adds shapes without rewriting old members."""

import hashlib
import json
import shutil

import pytest
from merlin_experiments.phase0 import development_cohort as F
from merlin_experiments.phase0 import development_shape_supplement as S

from merlin.common.paths import repo_root


def source_root():
    return repo_root() / "experiments/reference-data/phase0/public-architectures"


def test_public_implementation_proves_vision_defaults_and_connector_geometry():
    result = S.derive(source_root(), base_manifest_sha256="a" * 64)
    assert result["effective_vision_defaults"] == {"intermediate_size": 3072, "num_hidden_layers": 12}
    assert result["geometry"] == dict(
        vision_tokens=1024,
        post_shuffle_image_tokens=64,
        connector_reduction_width=12288,
        image_plus_text_prefix=112,
        expert_width=720,
    )
    assert len(result["requests"]) == 15 and result["original_members_removed"] == []
    assert result["invocation_audit"]["denoising"] == 10 and result["invocation_audit"]["decoder_decode"] == 32
    for row in result["requests"]:
        if row["role"] == "upper_geometry_control" or row["layer_count"] is None:
            assert row["occurrence_proxy"] is None
    assert any(row["declaration"]["geometry"] == dict(M=64, K=12288, N=960) for row in result["requests"])
    assert not result["phase0_admitted"] and not result["performance_objective_changed"]


def test_missing_effective_default_stays_unknown_instead_of_assuming_a_width(tmp_path):
    root = tmp_path / "inputs"
    shutil.copytree(source_root(), root)
    file = root / "defaults-sources/configuration_smolvlm.py.source"
    raw = file.read_bytes().replace(b"intermediate_size=3072,", b"intermediate_size=None,")
    file.write_bytes(raw)
    meta = json.loads((root / "defaults-provenance.json").read_bytes())
    for row in meta:
        if "configuration_smolvlm" in row["path"]:
            row.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    (root / "defaults-provenance.json").write_text(json.dumps(meta))
    result = S.derive(root, base_manifest_sha256="a" * 64)
    assert result["effective_vision_defaults"]["intermediate_size"] is None
    assert any(g["member"] == "vision_MLP_up_down" and g["status"] == "UNKNOWN" for g in result["gaps"])
    file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bytes changed"):
        S.derive(root, base_manifest_sha256="a" * 64)


def test_additive_projection_sources_materialize_with_expensive_refs_preserved(tmp_path):
    result = S.derive(source_root(), base_manifest_sha256="a" * 64)
    manifest = F.freeze(
        result["requests"],
        tmp_path / "sources",
        provenance=result,
        budgets=dict(
            max_members=32,
            max_source_bytes=65536,
            max_reference_elements=1,
            max_reference_work=1,
            max_total_reference_work=1,
            max_total_reference_elements=1,
        ),
    )
    assert len(manifest["members"]) == 15
    assert all(row["status"] == "source_frozen_reference_budget_denied" for row in manifest["members"])
