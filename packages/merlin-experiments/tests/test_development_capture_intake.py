"""Complete retained references feed capture without widening its admission."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from merlin_experiments.phase0 import development_capture as C
from merlin_experiments.phase0 import development_capture_intake as I
from merlin_experiments.phase0 import development_cohort as F
from merlin_experiments.phase0 import development_envelopes as E
from merlin_experiments.phase0 import development_reference_supplement as S


def _pin(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    requests = [
        {"declaration": E.declaration("matmul_rhs_nk", dict(M=3, K=17, N=5), ordinal=0)},
        {"status": "source_refused", "reason": "original numerical policy unavailable"},
    ]
    root = tmp_path / "source"
    F.freeze(
        requests,
        root,
        provenance={},
        budgets=dict(
            max_members=4,
            max_source_bytes=20000,
            max_reference_elements=10000,
            max_reference_work=1,
            max_total_reference_work=1,
            max_total_reference_elements=10000,
        ),
    )
    manifest_sha256 = _pin(root / "manifest.json")
    output = tmp_path / "reference"
    S.supplement(
        root,
        manifest_sha256=manifest_sha256,
        output=output,
        ordinals=[0],
        block=dict(M=2, N=3, K=7),
        budgets=dict(
            max_workspace_bytes=8 * 1024**2,
            max_output_bytes=10000,
            max_total_output_bytes=10000,
            max_member_work=10000,
            max_total_work=10000,
            max_dense_source_bytes=100000,
            member_timeout_s=30,
            total_timeout_s=60,
        ),
    )
    return root, manifest_sha256, (output / "index.json", _pin(output / "index.json"))


def test_complete_reference_intake_preserves_original_refusals_and_detects_tampering(tmp_path):
    root, digest, supplement = fixture(tmp_path)
    before = (root / "manifest.json").read_bytes()
    manifest, catalog = I.references(root, manifest_sha256=digest, supplements=[supplement])
    assert set(catalog) == {0}
    assert len(I.load_outputs(manifest["members"][0], catalog[0])) == 1
    report = I.plan(manifest, catalog, max_work=10000, max_elements=10000, max_seed_bytes=100000, max_contractions=1)
    assert len(report["members"]) == 2
    assert report["members"][0]["status"] == "capture_cost_screen_passed_not_admission"
    assert report["members"][1]["original_status"] == "source_refused"
    assert not report["phase0_admitted"] and not report["performance_objective_changed"]
    assert (root / "manifest.json").read_bytes() == before
    output = Path(catalog[0]["outputs"][0]["path"])
    output.chmod(0o600)
    np.save(output, np.ones((3, 5), dtype=np.int32))
    with pytest.raises(ValueError, match="bytes changed"):
        I.load_outputs(manifest["members"][0], catalog[0])


def test_cost_refusals_keep_seed_and_graph_work_separate_from_reference_availability(tmp_path):
    root, digest, supplement = fixture(tmp_path)
    manifest, catalog = I.references(root, manifest_sha256=digest, supplements=[supplement])
    report = I.plan(manifest, catalog, max_work=254, max_elements=10000, max_seed_bytes=100, max_contractions=1)
    assert report["members"][0]["status"] == "capture_cost_refused"
    assert len(report["members"][0]["reasons"]) == 2
    with pytest.raises(ValueError, match="duplicate selected"):
        I.references(root, manifest_sha256=digest, supplements=[supplement, supplement])


def test_selected_frozen_outputs_feed_ordinary_capture_without_recomputing_reference(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_source

    source = capsule_source.PytorchRefSource(timeout=120)
    if not source.available():
        pytest.skip("selected original Model2MLIR runtime unavailable")
    root, digest, supplement = fixture(tmp_path)
    monkeypatch.setattr(C.R, "evaluate", lambda *_: pytest.fail("retained complete reference was recomputed"))
    report = C.capture(
        root,
        manifest_sha256=digest,
        ordinals=[0],
        output=tmp_path / "captured",
        max_work=10000,
        max_elements=10000,
        frozen_references=True,
        reference_supplements=[supplement],
    )
    row = report["members"][0]
    assert row["status"] == "complete_capture_and_independent_reference_match"
    assert row["independent_reference_execution"] == "reused_byte_verified_complete_outputs"
    assert not report["phase0_admitted"] and len(report["capture_intake"]["members"]) == 2
    assert "capsule.yaml" in row["files"]
