"""Phase1 cannot authorize a native check, copied token or saved receipt."""

import copy
import hashlib
import json
import os
import pickle
from dataclasses import replace
from pathlib import Path

import pytest
from configuration_elision_qualification_cases import declarations

from merlin.common.paths import repo_root
from merlin.verify import configuration_hoist_gate as gate
from merlin.verify.configuration_hoist_intake import lean_root, prepare, proof_sources

ROOT = repo_root() / "out/artifacts/verification/merlin-passes/loop-scope-20261028/NEXT-HOIST/Q"


def forbidden(*args, **kwargs):
    pytest.fail("phase1 source-only record reached native command/consumer")


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    r = json.loads((ROOT / "actual-baseline-v1/ACTUAL.json").read_text())
    inputs = tmp_path_factory.mktemp("hoist-original-inputs")
    paths = {}
    for name, payload in zip(("facts", "software"), declarations(), strict=True):
        assert hashlib.sha256(payload).hexdigest() == r[name]["sha256"]
        path = inputs / ("facts.json" if name == "facts" else "software.yaml")
        path.write_bytes(payload)
        paths[name] = path
    return prepare(
        r["source"],
        r["final"],
        reference=os.environ.get("MERLIN_HOIST_REFERENCE", repo_root().parent / "merlin-dce-cse-integration"),
        facts=paths["facts"],
        software=paths["software"],
        target=r["isa"]["target"],
        source_manifest=ROOT.parent / "COORDINATOR/PRODUCER-SOURCE-MANIFEST-e0484f488.json",
    )


def test_candidate_stage_cannot_select_private_native_or_artifact_fallback(candidate, tmp_path, monkeypatch):
    # Canonical intake may have completed since preparation of the original
    # phase1 prototype. A phase1-tagged request still cannot gain authority.
    candidate = replace(candidate, proof_stage="isolated_phase1_candidate")
    monkeypatch.setattr(gate, "_run", forbidden)
    result = gate.check(candidate, lean=Path("/must-not-open/lean"), destination=tmp_path / "new")
    assert result.status == "unavailable" and result.completion is None
    assert not (tmp_path / "new").exists() and not gate._LIVE


@pytest.mark.parametrize("kind", ["saved", "copied", "constructed"])
def test_no_live_consumer_authority_from_data(candidate, kind):
    token = gate.Completion(candidate.identity, "/saved")
    value = {"saved": {"status": "checked"}, "copied": copy.copy(token), "constructed": token}[kind]
    with pytest.raises(gate.Unavailable):
        gate.deliver(value, candidate, forbidden)
    with pytest.raises(TypeError):
        pickle.dumps(token)


def test_source_only_request_never_claims_runtime_success(candidate):
    report = candidate.describe()
    assert report["kernel_outcome"] == "UNOBSERVED" and not report["authority"]
    assert not report["ISA_effect_hardware_approved"]
    assert len(json.loads(candidate.proof_sources_json)) == 9
    assert candidate.query == (ROOT / "query-v1/ActualConfigurationHoist.lean").read_text()


@pytest.mark.parametrize("kind", ["query", "roots", "domain", "saved"])
def test_forged_requests_refused_before_native(candidate, tmp_path, monkeypatch, kind):
    values = {
        "query": replace(candidate, query=candidate.query + "\n"),
        "roots": replace(candidate, roots=candidate.roots[:-1]),
        "domain": replace(candidate, domain="unqualified-other-profile"),
        "saved": {"status": "checked", "producer_revision": "e0484f488"},
    }
    monkeypatch.setattr(gate, "_run", forbidden)
    result = gate.check(values[kind], lean=Path("/never-open/lean"), destination=tmp_path / "new")
    assert result.status == "unavailable" and result.completion is None
    assert not (tmp_path / "new").exists()


def test_repository_environment_cannot_select_trusted_proof_sources(tmp_path, monkeypatch):
    expected_root = lean_root()
    expected_sources = proof_sources()
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "caller-selected-tree"))
    assert lean_root() == expected_root
    assert proof_sources() == expected_sources
    assert all(Path(row["path"]).is_relative_to(expected_root) for row in proof_sources())
