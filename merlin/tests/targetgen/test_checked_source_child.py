"""Real sealed-child selection transport; all target effects remain synthetic."""

import hashlib
import importlib.util
import json
import os

import pytest
import test_compiler_profile_child as PROFILE
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD
from merlin_experiments.phase2.contracts import StageGateError

from merlin.targetgen import checked_source_model_diagnostic as S
from merlin.targetgen.compiler_pass_selection import (
    CompilerPassSelection,
    OriginalPassFactory,
    scoped_compiler_passes,
)


@pytest.fixture
def original(tmp_path, monkeypatch):
    return PROFILE.original.__wrapped__(tmp_path, monkeypatch)


def _selection(original, *, native_execution_protocol=None):
    source = original.private / "issuer.py"
    source.write_text('''from merlin.llvmlower.reviewed_pass_capabilities import (
 PassCapabilityBinding, ReviewedPassCapabilities, ReviewedPassCapability)
class Issuer:
 identity="synthetic-source-effects-transport-control"
 def revalidate(self): pass
 def create_source_execution_owner(self,**kw):
  raise AssertionError("no target effects or qualification run in transport controls")
def restore(inputs,*,work_root):
 return ReviewedPassCapabilities((ReviewedPassCapability("source_execution_diagnostic",
  PassCapabilityBinding(**inputs["binding"]),Issuer()),))
''')
    spec = importlib.util.spec_from_file_location("synthetic_checked_source_issuer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fir = original.private / "synthetic.fir"
    fir.write_text("synthetic nonexecuted source fixture")
    fir_sha = hashlib.sha256(fir.read_bytes()).hexdigest()
    facts = original.private / "facts.json"
    facts.write_text(json.dumps(dict(facts=dict(source=dict(config="fixture")), inputs=dict(
        target=original.target.target, fir_sha256=fir_sha,
        firrtl_inputs=[dict(path=str(fir), sha256=fir_sha)]))))
    binding = dict(hardware=hashlib.sha256(facts.read_bytes()).hexdigest(), software="software",
                   numerical_policy="numeric", runtime="runtime", library="library")
    factory = OriginalPassFactory(module.restore, dict(binding=binding))
    from merlin.llvmlower.reviewed_pass_capabilities import PassCapabilityBinding

    passes = CompilerPassSelection(
        factory.create(work_root=original.private / "admission"), PassCapabilityBinding(**binding),
        mechanisms=("optional_transform",), factory=factory)
    selected = S.select_checked_source_diagnostic(passes, target=original.target.target,
                                                 rtl_facts=facts, board_config="fixture",
                                                 native_execution_protocol=native_execution_protocol)
    job = json.loads(original.spec.read_bytes())
    job["capsule"].update(kind="model", label="dev", oracle_tier_ceiling="L2",
                          required_oracle_tiers=["L0", "L1", "L2", "L3"])
    job["candidate_context"] = {S.CONTEXT_FIELD: selected.record()}
    original.spec.write_text(json.dumps(job))
    return passes, selected


def test_actual_v4_child_rejoins_only_original_live_registry_and_context(original):
    passes, selected = _selection(original)
    script = PROFILE._CONTROL.replace("    job=json.loads(raw)", '''    job=json.loads(raw)
    from merlin.targetgen import checked_source_model_diagnostic as D
    selection=D.selection_from_context(job['candidate_context'][D.CONTEXT_FIELD],
      capsule=job['capsule'],target=job['target'],package_dir=job['package_dir'])
    assert selection.record()['formal_admission'] is False
    assert selection.record()['promotion_authority']=='NONE'
    assert receipt['schema']==CHILD.CHECKED_SOURCE_SCHEMA''')
    with (scoped_compiler_passes(passes), S.scoped_checked_source_diagnostic(selected), PROFILE._scope(original)):
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            assert transport.document["schema"] == CHILD.CHECKED_SOURCE_SCHEMA
            result = PROFILE._run(transport, original, script=script)
            assert result.returncode == 0, result.stderr
            transport.verify_result(json.loads(result.stdout))


def test_actual_child_preserves_explicit_pointer_selector_without_target_execution(original):
    passes, selected = _selection(original, native_execution_protocol=S.POINTER_PROTOCOL)
    script = PROFILE._CONTROL.replace("    job=json.loads(raw)", '''    job=json.loads(raw)
    from merlin.targetgen import checked_source_model_diagnostic as D
    selection=D.selection_from_context(job['candidate_context'][D.CONTEXT_FIELD],
      capsule=job['capsule'],target=job['target'],package_dir=job['package_dir'])
    assert selection.record()['schema']==D.POINTER_SCHEMA
    assert selection.record()['native_execution_protocol']==D.POINTER_PROTOCOL
    assert selection.record()['formal_admission'] is False''')
    with (scoped_compiler_passes(passes), S.scoped_checked_source_diagnostic(selected), PROFILE._scope(original)):
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            result = PROFILE._run(transport, original, script=script)
            assert result.returncode == 0, result.stderr
            transport.verify_result(json.loads(result.stdout))


def test_crafted_context_without_actual_parent_selection_refuses(original):
    passes, _selected = _selection(original)
    with (scoped_compiler_passes(passes), PROFILE._scope(original),
          pytest.raises(StageGateError, match="actual original")):
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate):
            pytest.fail("crafted context cannot mint a diagnostic selection")


def test_plain_context_record_outside_actual_restored_worker_refuses(original):
    passes, selected = _selection(original)
    with scoped_compiler_passes(passes), S.scoped_checked_source_diagnostic(selected):
        with pytest.raises(StageGateError, match="restored sealed worker"):
            S.selection_from_context(selected.record(), capsule={}, target=original.target.target,
                                     package_dir=original.candidate)
