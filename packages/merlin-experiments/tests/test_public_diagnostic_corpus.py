"""Source/reference copy custody, with a neutral accelerator projection only."""

import copy
import hashlib
import shutil

import pytest
import test_operator_private_exclusion as EXCLUSION
import test_public_program_curriculum as PUBLIC_FIXTURE
from merlin_experiments.phase2 import corpus as FORMAL
from merlin_experiments.phase2 import public_diagnostic_corpus as D
from merlin_experiments.phase2.contracts import StageGateError


@pytest.fixture
def published(tmp_path, monkeypatch):
    selected, private = EXCLUSION.selected(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    _, member = PUBLIC_FIXTURE.write(source, selected)
    # PP.verify_written and the real exclusion owner still check the complete
    # captured copy program/reference closure. Only accelerator applicability
    # is neutral here; no test supplies target/source qualification.
    original_projection = D.PUBLIC.original_contractions
    monkeypatch.setattr(D.PUBLIC, "original_contractions", lambda *a, **kw: [{"neutral_projection": True}])
    planning = tmp_path / "planning.json"
    planning.write_text('{"members":[],"scope":"external synthetic planning only"}')
    args = dict(
        sources=(member,),
        destination=tmp_path / "diagnostic-snapshot",
        target="fixture",
        operator_exclusion=selected,
        planning_inventory=planning,
        planning_inventory_sha256=hashlib.sha256(planning.read_bytes()).hexdigest(),
    )
    return args, member, selected, private, original_projection


def test_fresh_snapshot_reopens_original_public_writer_and_real_exclusion(published, monkeypatch):
    args, source, selection, private, _ = published
    observed = []
    original = D.EXCLUSION.OperatorPrivateExclusion.check_member

    def check(self, directory):
        assert self is selection
        observed.append(directory)
        return original(self, directory)

    monkeypatch.setattr(D.EXCLUSION.OperatorPrivateExclusion, "check_member", check)
    owner = D.freeze_public_diagnostic_corpus(**args)
    record = owner.verify()
    copied = owner.capsules[0].source_dir
    assert source in observed and copied in observed and len(observed) >= 4
    assert record["performance_cohort_admitted"] is False and record["formal_admission"] is False
    assert record["promotion_authority"] == "NONE" and record["timing"] == "UNKNOWN"
    assert not owner.root.stat().st_mode & 0o222
    assert (
        source != copied
        and (source / "capsule.interface.mlir").read_bytes() == (copied / "capsule.interface.mlir").read_bytes()
    )
    assert str(private).encode() not in owner.manifest_path.read_bytes()
    assert private.read_bytes() not in owner.manifest_path.read_bytes()
    with pytest.raises(StageGateError):
        FORMAL.verify_frozen_performance_corpus(owner)
    with pytest.raises(StageGateError):
        FORMAL.load_frozen_performance_corpus(
            owner.root, manifest_sha256=owner.manifest_sha256, capsules_sha256=owner.capsules_sha256
        )
    with pytest.raises(StageGateError, match="original host"):
        copy.copy(owner).verify()


@pytest.mark.parametrize("fault", ["source", "reference", "planning", "extra_file", "writable"])
def test_snapshot_or_parent_obligation_drift_refuses(published, fault):
    args, *_ = published
    owner = D.freeze_public_diagnostic_corpus(**args)
    if fault == "planning":
        args["planning_inventory"].write_text('{"changed":true}')
    elif fault == "extra_file":
        owner.root.chmod(0o755)
        (owner.root / "foreign").write_text("not an original source member")
        owner.root.chmod(0o555)
    else:
        source = owner.capsules[0].source_dir
        name = "capsule.interface.mlir" if fault == "source" else "source-capture-golden.npy"
        path = source / name
        path.chmod(0o644)
        if fault != "writable":
            path.write_bytes(b"changed")
    with pytest.raises((StageGateError, ValueError)):
        owner.verify()


def test_real_source_and_reference_verifier_refusal_blocks_issuance(published):
    args, source, *_ = published
    (source / "source-capture-golden.npy").write_bytes(b"forged reference")
    with pytest.raises((StageGateError, ValueError)):
        D.freeze_public_diagnostic_corpus(**args)


def test_host_only_source_cannot_be_a_diagnostic_accelerator_member(published, monkeypatch):
    args, *_, original_projection = published
    monkeypatch.setattr(D.PUBLIC, "original_contractions", original_projection)
    with pytest.raises(StageGateError, match="host-only"):
        D.freeze_public_diagnostic_corpus(**args)


def test_aggregate_file_budget_refuses_before_creating_or_copying_snapshot(published, tmp_path, monkeypatch):
    args, source, *_ = published
    second = tmp_path / "second-source"
    shutil.copytree(source, second)
    original = D.source_inventory

    def inventory(path, **kwargs):
        result = original(path, **kwargs)
        return dict(result, n_files=8192) if path in (source, second) else result

    monkeypatch.setattr(D, "source_inventory", inventory)
    with pytest.raises(StageGateError, match="aggregate copy budget"):
        D.freeze_public_diagnostic_corpus(**dict(args, sources=(source, second)))
    assert not args["destination"].exists()
