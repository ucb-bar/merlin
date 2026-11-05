"""Check metadata boundaries independently of any target or timing authority."""

import contextlib
import importlib.util
import json
import os

import pytest
import test_compiler_profile_child as PROFILE
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD

from merlin.common.digest import sha256_file
from merlin.llvmlower.reviewed_metadata import _metadata_verified
from merlin.targetgen.compiler_pass_selection import (
    METADATA_PROTOCOL,
    CompilerPassSelection,
    OriginalPassFactory,
    scoped_compiler_passes,
)

FACTORY = '''from pathlib import Path
from merlin.common.digest import sha256_file
from merlin.llvmlower.reviewed_pass_capabilities import (
 PassCapabilityBinding, ReviewedPassCapabilities, ReviewedPassCapability)
class EvidenceOwner:
 def __init__(self,path):
  self.path=Path(path);self.identity=sha256_file(self.path);self.checks=0
 def revalidate(self):
  self.checks+=1
  if sha256_file(self.path)!=self.identity: raise ValueError("original metadata evidence changed")
def restore_metadata_fixture(inputs,*,work_root):
 if sha256_file(inputs["evidence"])!=inputs["evidence_sha256"]: raise ValueError("original metadata evidence changed")
 return ReviewedPassCapabilities((ReviewedPassCapability("metadata_fixture",
  PassCapabilityBinding("rtl","sw","numeric","runtime","library"),EvidenceOwner(inputs["evidence"])),))
'''


@pytest.fixture
def original(tmp_path, monkeypatch):
    return PROFILE.original.__wrapped__(tmp_path, monkeypatch)


def selection(original, *, enabled=True):
    path = original.private / "metadata_factory.py"
    path.write_text(FACTORY)
    spec = importlib.util.spec_from_file_location("original_metadata_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evidence = original.private / "metadata-evidence"
    evidence.write_text("original evidence")
    inputs = {"evidence": str(evidence), "evidence_sha256": sha256_file(evidence)}
    if enabled:
        inputs["metadata_verification_protocol"] = METADATA_PROTOCOL
    factory = OriginalPassFactory(module.restore_metadata_fixture, inputs)
    registry = factory.create(work_root=original.private / "metadata-admission")
    capability = registry._capabilities[0]
    passes = CompilerPassSelection(registry, capability.binding, mechanisms=("metadata_fixture",), factory=factory)
    return passes, capability


@pytest.mark.parametrize("enabled", [False, True])
def test_child_transport_closes_opted_metadata_before_caller_can_launch(original, monkeypatch, enabled):
    passes, capability = selection(original, enabled=enabled)
    owner = capability.owner
    sealed, observations = CHILD._sealed_transport, []

    @contextlib.contextmanager
    def observe(*args, **kwargs):
        observations.append(_metadata_verified(capability))
        with sealed(*args, **kwargs) as transport:
            yield transport

    monkeypatch.setattr(CHILD, "_sealed_transport", observe)
    with scoped_compiler_passes(passes), PROFILE._scope(original):
        owner.checks = 0
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            assert observations == [enabled]
            assert not _metadata_verified(capability)
            # The active-selection getter has its own two boundaries. The
            # transport document then has exactly two, across all its getters.
            assert (owner.checks == 4 if enabled else owner.checks > 2), owner.checks
            assert transport.document["schema"] == CHILD.PASSES_SCHEMA
            # These are actual post-issuance getters: none inherits the closed
            # metadata operation, and no child result is reused or manufactured.
            before = owner.checks
            passes.revalidate()
            assert owner.checks > before


@pytest.mark.parametrize("damage", ["evidence", "registry"])
def test_final_original_check_prevents_transport_escape_and_closes_fd(original, monkeypatch, damage):
    passes, capability = selection(original)
    owner = capability.owner
    sealed, descriptors = CHILD._sealed_transport, []

    @contextlib.contextmanager
    def change_at_boundary(*args, **kwargs):
        with sealed(*args, **kwargs) as transport:
            descriptors.append(transport.fd)
            if damage == "evidence":
                owner.path.write_text("changed original evidence")
            else:
                passes.registry._identity = "changed registry"
            yield transport

    monkeypatch.setattr(CHILD, "_sealed_transport", change_at_boundary)
    with scoped_compiler_passes(passes), PROFILE._scope(original):
        original_identity = passes.registry._identity
        try:
            with pytest.raises(ValueError, match="changed"):
                with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                               package_dir=original.candidate):
                    pytest.fail("transport escaped its failed final original check")
            assert not _metadata_verified(capability)
            assert len(descriptors) == 1
            with pytest.raises(OSError):
                os.fstat(descriptors[0])
        finally:
            owner.path.write_text("original evidence")
            passes.registry._identity = original_identity


def test_actual_child_still_reopens_current_evidence_after_metadata_scope(original):
    passes, capability = selection(original)
    owner = capability.owner
    with scoped_compiler_passes(passes), PROFILE._scope(original):
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            assert not _metadata_verified(capability)
            owner.path.write_text("changed after parent preparation")
            try:
                result = PROFILE._run(transport, original)
                assert result.returncode != 0
                assert "evidence" in result.stderr
            finally:
                owner.path.write_text("original evidence")


def test_actual_child_restores_original_selection_after_metadata_scope(original):
    passes, capability = selection(original)
    with scoped_compiler_passes(passes), PROFILE._scope(original):
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            result = PROFILE._run(transport, original)
            assert result.returncode == 0, result.stderr
            transport.verify_result(json.loads(result.stdout))
            assert not _metadata_verified(capability)
