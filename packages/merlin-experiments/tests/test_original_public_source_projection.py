"""Original source projection reuse retains owner, profile and frozen-byte custody."""
import hashlib
from types import SimpleNamespace

import pytest

from merlin.common import bundle_snapshot as S
from merlin.targetgen.sandbox import bwrap as B
from merlin_experiments.phase2 import compiler_libraries as L
from merlin_experiments.phase2.contracts import StageGateError


@pytest.fixture
def original(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    repo = tmp_path / "repo"
    source = repo / "merlin" / "portable.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"def identity(value): return value\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bundle = {"allowed": [{"path": "merlin/portable.py", "mode": "ro"}], "host_inputs": [], "denied": []}
    B.materialize_bundle_inputs(workspace, bundle, repo=repo)
    frozen = S.snapshot_path(S.bundle_snapshot_root(workspace), source, repo)
    frozen.chmod(0o600)
    owner = SimpleNamespace(workspace=workspace, repo=repo, bundle=bundle, checks=0, refuse=False,
                            profile={"common_sources": [{"path": "merlin/portable.py",
                                      "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
                                     "treatment_tools": []})

    def verify():
        owner.checks += 1
        if owner.refuse:
            raise ValueError("original owner expired")
        S.verify_bundle_snapshot(workspace, bundle, repo=repo)

    owner.verify = verify
    projections = []
    project = L._original_source_projection

    def counted(*args):
        projections.append(True)
        return project(*args)

    monkeypatch.setattr(L, "_original_source_projection", counted)
    entries = []

    def entry():
        result = S.shared_entry_verification(workspace, bundle, repo=repo)
        if result is None:
            pytest.skip("kernel guard unavailable; production retains fresh projections")
        entries.append(result)
        return result

    yield owner, frozen, projections, entry
    for selected in entries:
        selected.close()


def test_guarded_reuse_is_detached_and_reopens_owner_and_profile(original):
    owner, frozen, projections, entry = original
    with S.shared_entry_scope(entry()):
        result = L._original_sources(owner)
        result["merlin/forged.py"] = frozen
        assert L._original_sources(owner) == {"merlin/portable.py": frozen}
        assert len(projections) == 1 and owner.checks == 2
        owner.profile["common_sources"][0]["sha256"] = "0" * 64
        with pytest.raises(StageGateError, match="source identity"):
            L._original_sources(owner)
        assert len(projections) == 2
        owner.refuse = True
        with pytest.raises(ValueError, match="owner expired"):
            L._original_sources(owner)
        assert owner.checks == 4


def test_frozen_source_mutation_refuses_after_a_warm_projection(original):
    owner, frozen, projections, entry = original
    with S.shared_entry_scope(entry()):
        L._original_sources(owner)
        frozen.write_bytes(b"def identity(value): return None\n")
        with pytest.raises(RuntimeError, match="content differs"):
            L._original_sources(owner)
        assert owner.checks == 2 and len(projections) == 1


def test_projection_does_not_cross_request_or_unscoped_boundaries(original):
    owner, _frozen, projections, entry = original
    first, second = entry(), entry()
    with S.shared_entry_scope(first):
        L._original_sources(owner)
        L._original_sources(owner)
    with S.shared_entry_scope(second):
        L._original_sources(owner)
    assert len(projections) == 2
    L._original_sources(owner)
    L._original_sources(owner)
    assert len(projections) == 4 and owner.checks == 5


def test_cached_mapping_still_checks_live_host_locations(original, monkeypatch):
    owner, _frozen, projections, entry = original
    with S.shared_entry_scope(entry()):
        L._original_sources(owner)
        def refuse(*args, **kwargs):
            raise RuntimeError("host-only input overlaps the writable workspace")
        monkeypatch.setattr(B, "_validate_host_locations", refuse)
        with pytest.raises(RuntimeError, match="host-only input overlaps"):
            L._original_sources(owner)
        assert owner.checks == 2 and len(projections) == 1
