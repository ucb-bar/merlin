"""Cleanup refuses what a seal, release, run record or derivation cites, and still reclaims the rest.

Twice a disk cleanup deleted sealed Phase 0 evidence whose path and digest sat in a record that
outlived the cleanup. These tests hold the storage tool to the registry built from those records:
the cited file survives ``merlin-storage prune --apply`` and the uncited orphan beside it does not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from merlin.common import cited_evidence as CE
from merlin.common import content_store as CS
from merlin.common import storage_cli as SC
from merlin.common import storage_lifecycle as lifecycle


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "out" / "artifacts" / "cache" / CS.NAMESPACE))
    monkeypatch.delenv(CE.ROOTS_ENV, raising=False)
    (tmp_path / "out" / "artifacts" / "cache").mkdir(parents=True)
    declared = dict(SC.contract(), scan_roots=[])
    monkeypatch.setattr(SC, "contract", lambda: declared)
    CE.clear()
    yield tmp_path
    CE.clear()


def _terminal(path: Path) -> None:
    with lifecycle.lease(path, owner="fixture-cache-writer"):
        pass


def test_a_cited_cache_is_refused_and_an_uncited_orphan_is_pruned(rooted, capsys):
    cache = rooted / "out" / "artifacts" / "cache"
    cited = cache / "captures-v5" / "app"
    cited.mkdir(parents=True)
    (cited / "pytorch-opset.json").write_text("{}")
    orphan = cache / "orphan-scratch"
    orphan.mkdir()
    (orphan / "bulk.bin").write_bytes(b"x" * 4096)
    for path in (cited.parent, orphan):
        _terminal(path)
    # A derivation outside the cache pins the capture member by path and digest, as meta.json does.
    derivation = rooted / "out" / "artifacts" / "probes" / "derivation" / "materialized" / "app"
    derivation.mkdir(parents=True)
    (derivation / "meta.json").write_text(
        json.dumps({"framework_catalog": {"path": str(cited / "pytorch-opset.json"), "sha256": "a" * 64}})
    )

    assert SC.main(["prune", "caches"]) == 0
    plan = capsys.readouterr().out
    assert str(orphan) in plan and str(cited.parent) not in plan

    assert SC.main(["prune", "caches", "--apply"]) == 0
    capsys.readouterr()
    assert not orphan.exists(), "the uncited orphan was not reclaimed"
    assert (cited / "pytorch-opset.json").is_file(), "prune deleted a file a derivation cites"

    with pytest.raises(PermissionError, match="cited evidence"):
        SC._remove(cited.parent)
    assert (cited / "pytorch-opset.json").is_file()


def test_a_sealed_release_is_protected_whole_even_against_its_own_seal(rooted):
    release = rooted / "out" / "artifacts" / "protocols" / "t" / "phase0-x"
    (release / "private").mkdir(parents=True)
    (release / "payload" / "corpus").mkdir(parents=True)
    (release / "payload" / "corpus" / "golden.yaml").write_text("value: 1\n")
    (release / "private" / "seal.json").write_text(json.dumps({"schema_version": 1}))
    registry = CE.build([rooted / "out"])
    member = release / "payload" / "corpus" / "golden.yaml"
    assert [c.kind for c in registry.citations(member)] == ["anchor"]
    assert registry.citations(release), "removing the release with its seal must still be refused"
    assert not registry.citations(rooted / "out" / "artifacts" / "protocols" / "t" / "other")


def test_a_run_record_does_not_pin_the_unit_it_lives_in(rooted):
    """Retention may drop a whole run, record included; it may not drop what a surviving record cites."""
    run = rooted / "out" / "runs" / "t" / "suite" / "20260101T000000Z_run"
    run.mkdir(parents=True)
    (run / "result.json").write_text("{}")
    elsewhere = rooted / "out" / "artifacts" / "inputs" / "facts.json"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("{}")
    (run / "run_record.json").write_text(json.dumps({"inputs": [str(run / "result.json"), str(elsewhere)]}))
    registry = CE.build([rooted / "out"])
    assert registry.citations(run) == [], "a run's own record blocked removing the run"
    assert registry.citations(run / "result.json"), "a file the surviving record cites is unprotected"
    assert registry.citations(elsewhere.parent), "a directory holding a cited file is unprotected"


def test_a_bare_mention_of_a_directory_does_not_claim_every_file_in_it(rooted):
    store = rooted / "out" / "artifacts" / "cache" / "objects"
    store.mkdir(parents=True)
    (store / "ab").write_text("orphan")
    pinned = rooted / "out" / "artifacts" / "captures" / "run"
    pinned.mkdir(parents=True)
    (pinned / "model.mlir").write_text("module {}")
    record = rooted / "out" / "artifacts" / "attempt" / "resolved-plan.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"store": str(store), "capture": {"path": str(pinned), "sha256": "b" * 64}}))
    registry = CE.build([rooted / "out"])
    assert registry.citations(store), "the cited directory itself is unprotected"
    assert not registry.citations(store / "ab"), "a bare mention pinned every file under it"
    assert registry.citations(pinned / "model.mlir"), "a digest-pinned tree left its members unprotected"


def test_relative_members_resolve_against_their_record(rooted):
    root = rooted / "out" / "artifacts" / "probes" / "d"
    (root / "materialized" / "app").mkdir(parents=True)
    (root / "materialized" / "app" / "capture_receipt.json").write_text(
        json.dumps({"artifacts": {"frontend-trace.json": {"sha256": "c" * 64, "bytes": 2}}})
    )
    (root / "synthesis.yaml").write_text(
        "capsules:\n- materialized_capture:\n    loader_path: materialized/app/loader.py\n"
        f"    loader_sha256: {'d' * 64}\n"
    )
    registry = CE.build([rooted / "out"])
    assert registry.citations(root / "materialized" / "app" / "frontend-trace.json")
    assert registry.citations(root / "materialized" / "app" / "loader.py")
    assert not registry.citations(root / "materialized" / "app" / "golden.npz")


def test_extra_roots_come_from_the_environment(rooted, monkeypatch):
    ledger = rooted / "papers" / "ledger"
    ledger.mkdir(parents=True)
    target = rooted / "out" / "artifacts" / "cache" / "grades" / "harness.c"
    target.parent.mkdir(parents=True)
    target.write_text("int main;")
    (ledger / "harvest-receipt.json").write_text(json.dumps({"read": [str(target)]}))
    assert not CE.registry(refresh=True).citations(target)
    monkeypatch.setenv(CE.ROOTS_ENV, str(rooted / "papers"))
    assert CE.registry(refresh=True).citations(target)


def test_the_cited_subcommand_answers_before_a_manual_delete(rooted, capsys, monkeypatch):
    ledger = rooted / "ledger"
    ledger.mkdir()
    target = rooted / "evidence" / "golden.npz"
    target.parent.mkdir()
    target.write_bytes(b"g")
    (ledger / "release-seal.json").write_text(json.dumps({"protects": [str(target.parent)]}))
    monkeypatch.setenv(CE.ROOTS_ENV, str(ledger))
    assert SC.main(["cited", str(target), "--json"]) == 1
    rows = json.loads(capsys.readouterr().out)["paths"][str(target)]
    assert rows and rows[0]["kind"] == "anchor"
    assert SC.main(["cited", str(rooted / "unrelated")]) == 0
    assert os.path.exists(target)
