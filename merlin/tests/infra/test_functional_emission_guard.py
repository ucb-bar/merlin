"""The differential guard reads the admitted public grading view, not live corpora."""

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import corpus_inputs as CI
from merlin_experiments.phase2 import functional_inputs as FI
from test_perf_agent_stage import PAS


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native launch"))
    frozen = tmp_path / "frozen"
    for name in ("public/visible", "policy", "contract/schemas", "hidden/answer", "tuning/performance"):
        (frozen / name).mkdir(parents=True)
    for name in ("public/visible", "hidden/answer", "tuning/performance"):
        (frozen / name / "capsule.yaml").write_text("interface_mlir: capsule.interface.mlir\n")
        (frozen / name / "capsule.interface.mlir").write_text("module {}\n")
    (frozen / "source_commitments.json").write_text(json.dumps({"version": 1, "sources": []}))
    record = {
        "version": 1,
        "staging_path": str(tmp_path / "original-deleted"),
        "content_sha256": CI.fingerprint(frozen),
    }
    environment = tmp_path / "environment.yaml"
    environment.write_text(yaml.safe_dump({"public_corpus_input": record}))
    provenance = {
        "environment": str(environment),
        "environment_sha256": hashlib.sha256(environment.read_bytes()).hexdigest(),
    }
    verified = {"workspace": tmp_path / "workspace", "bundle": {}, "repo": tmp_path / "old-repo"}
    monkeypatch.setattr(FI, "_verify_private_functional_provenance", lambda value: verified)
    monkeypatch.setattr(CI.bwrap, "snapshot_input_paths", lambda *a, **k: [frozen])
    monkeypatch.setattr(CI.bwrap, "verify_bundle_snapshot", lambda *a, **k: {"grants": []})
    from merlin.targetgen import oot_runner

    monkeypatch.setattr(oot_runner, "load_package", lambda path: object())
    observed = []

    def emit(package, interface, scratch, tag, timeout):
        observed.append(interface)
        return 0, "same LLVM", "same buffer"

    monkeypatch.setattr(PAS.EA, "emit_pair", emit)
    return SimpleNamespace(host_provenance=provenance), environment, frozen, observed


def test_public_view_ignores_live_roots_hidden_and_tuning(inputs, tmp_path):
    frozen_inputs, environment, frozen, observed = inputs

    class NoLiveDiscovery:
        @property
        def capsule_corpus(self):
            pytest.fail("live corpus queried")

        def corpus_siblings(self):
            pytest.fail("live siblings queried")

    result = PAS.functional_emission_guard(tmp_path, tmp_path, NoLiveDiscovery(), frozen_functional=frozen_inputs)
    assert result["proved_unchanged"] == 1
    assert observed == [frozen / "public/visible/capsule.interface.mlir"] * 2
    assert not (tmp_path / "original-deleted").exists()


@pytest.mark.parametrize("missing", ["provenance", "declaration", "snapshot_mapping", "changed_frozen_view"])
def test_guard_refuses_missing_admission_before_emission(inputs, tmp_path, monkeypatch, missing):
    frozen_inputs, environment, frozen, observed = inputs
    if missing == "provenance":
        frozen_inputs.host_provenance = None
    elif missing == "declaration":
        environment.write_text("{}\n")
        frozen_inputs.host_provenance["environment_sha256"] = hashlib.sha256(environment.read_bytes()).hexdigest()
    elif missing == "changed_frozen_view":
        (frozen / "public/visible/capsule.interface.mlir").write_text("changed after admission\n")
    else:

        def absent(*a, **k):
            raise RuntimeError("undeclared frozen input")

        monkeypatch.setattr(CI.bwrap, "snapshot_input_paths", absent)
    with pytest.raises(PAS.StageGateError):
        PAS.functional_emission_guard(tmp_path, tmp_path, object(), frozen_functional=frozen_inputs)
    assert not observed


@pytest.mark.parametrize(
    "baseline_rc,candidate_rc,kind",
    [(1, 1, "baseline_emission_failed"), (1, 0, "baseline_emission_failed"), (0, 1, "lowering_regressed")],
)
def test_failed_emission_never_proves_unchanged(inputs, tmp_path, monkeypatch, baseline_rc, candidate_rc, kind):
    frozen_inputs, _, _, _ = inputs
    monkeypatch.setattr(
        PAS.EA,
        "emit_pair",
        lambda package, interface, scratch, tag, timeout: (baseline_rc if tag == "baseline" else candidate_rc, "", ""),
    )
    result = PAS.functional_emission_guard(tmp_path, tmp_path, object(), frozen_functional=frozen_inputs)
    assert result["status"] == "offending"
    assert result["proved_unchanged"] == 0
    assert result["offenders"] == [
        {"capsule": "visible", "kind": kind, "baseline_rc": baseline_rc, "candidate_rc": candidate_rc}
    ]


def test_missing_declared_interface_is_absence_of_proof(inputs, tmp_path):
    frozen_inputs, environment, frozen, observed = inputs
    (frozen / "public/visible/capsule.interface.mlir").unlink()
    # Construct an admitted synthetic view that was already missing the interface,
    # distinct from a post-admission byte mutation.
    document = yaml.safe_load(environment.read_text())
    document["public_corpus_input"]["content_sha256"] = CI.fingerprint(frozen)
    environment.write_text(yaml.safe_dump(document))
    frozen_inputs.host_provenance["environment_sha256"] = hashlib.sha256(environment.read_bytes()).hexdigest()
    result = PAS.functional_emission_guard(tmp_path, tmp_path, object(), frozen_functional=frozen_inputs)
    assert result["status"] == "offending"
    assert result["proved_unchanged"] == 0
    assert result["offenders"] == [{"capsule": "visible", "kind": "interface_missing"}]
    assert not observed
