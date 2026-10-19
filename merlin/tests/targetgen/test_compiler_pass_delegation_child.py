"""Real sealed-parent delegation controls; no target capability is qualified."""

import importlib.util
import json
import os

import pytest
import test_compiler_profile_child as PROFILE
from merlin_experiments.phase1.feedback import compiler_pass_delegation as D
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD

from merlin.llvmlower.reviewed_pass_capabilities import PassCapabilityBinding
from merlin.targetgen.compiler_pass_selection import (
    CompilerPassSelection,
    OriginalPassFactory,
    scoped_compiler_passes,
)


@pytest.fixture
def original(tmp_path, monkeypatch):
    return PROFILE.original.__wrapped__(tmp_path, monkeypatch)


def _selection(original, *, opt_in=True):
    source = original.private / "delegating_fixture.py"
    source.write_text('''from pathlib import Path
from merlin.llvmlower.reviewed_pass_capabilities import (
 PassCapabilityBinding, ReviewedPassCapabilities, ReviewedPassCapability)
from merlin_experiments.phase1.feedback.compiler_pass_delegation import (
 active_original_parent_pass_qualification)
class Owner:
 identity="synthetic-parent-custody-control"
 def revalidate(self): pass
 def create_source_execution_owner(self,**kw):
  raise AssertionError("transport control has no target qualification")
def registry(inputs):
 return ReviewedPassCapabilities((ReviewedPassCapability(inputs.get("mechanism","synthetic"),
   PassCapabilityBinding(**inputs["binding"]),Owner()),))
def restore(inputs,*,work_root):
 token=active_original_parent_pass_qualification()
 if token is not None:
  return token.redeem(restore,inputs,decoder=redeem,work_root=work_root)
 with Path(inputs["log"]).open("a") as stream: stream.write("full\\n")
 return registry(inputs)
def export(registry):
 registry.revalidate()
 return {"registry":registry.record(),"payload":"original live owner"}
def redeem(snapshot,*,token,work_root):
 context=token.require(snapshot)
 assert set(context)=={"registry","factory_inputs"}
 assert snapshot["registry"]==context["registry"]
 with Path(context["factory_inputs"]["log"]).open("a") as stream: stream.write("redeem\\n")
 return registry(context["factory_inputs"])
''')
    spec = importlib.util.spec_from_file_location("original_delegating_fixture", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binding = dict(hardware="hardware", software="software", numerical_policy="numeric",
                   runtime="runtime", library="library")
    inputs = dict(binding=binding, log=str(original.private / "admission.log"))
    if opt_in:
        inputs["parent_delegation_protocol"] = D.PROTOCOL
    factory = OriginalPassFactory(module.restore, inputs)
    selected = CompilerPassSelection(factory.create(work_root=original.private / "parent-admission"),
                                     PassCapabilityBinding(**binding), mechanisms=("synthetic",), factory=factory)
    return selected, module


def test_v5_actual_parent_qualification_is_inherited_without_repeating_admission(original):
    passes, module = _selection(original)
    delegation = D.OriginalParentPassDelegation(passes, exporter=module.export, redeemer=module.redeem)
    script = PROFILE._CONTROL.replace("    job=json.loads(raw)", '''    job=json.loads(raw)
    from merlin_experiments.phase1.feedback import compiler_pass_delegation as D
    token=D.active_original_parent_pass_qualification()
    assert token.receipt()['fresh_pair_evaluation'] is False
    assert receipt['schema']==CHILD.DELEGATED_PASSES_SCHEMA
    assert receipt['inherited_pass_qualification']['fresh_pair_evaluation'] is False
    try: token.require({'payload':'copied success'})
    except ValueError: pass
    else: raise AssertionError('snapshot outside redemption accepted')''')
    with (scoped_compiler_passes(passes), D.scoped_original_parent_pass_delegation(delegation),
          PROFILE._scope(original)):
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            assert transport.document["schema"] == CHILD.DELEGATED_PASSES_SCHEMA
            result = PROFILE._run(transport, original, script=script)
            assert result.returncode == 0, result.stderr
            transport.verify_result(json.loads(result.stdout))
    assert (original.private / "admission.log").read_text().splitlines() == ["full", "redeem"]


def test_default_v3_still_runs_full_original_factory_in_child(original):
    passes, _module = _selection(original)
    with scoped_compiler_passes(passes), PROFILE._scope(original):
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            assert transport.document["schema"] == CHILD.PASSES_SCHEMA
            result = PROFILE._run(transport, original)
            assert result.returncode == 0, result.stderr
    assert (original.private / "admission.log").read_text().splitlines() == ["full", "full"]


def test_factory_must_explicitly_opt_in_and_parent_snapshot_cannot_drift(original):
    passes, module = _selection(original, opt_in=False)
    with pytest.raises(ValueError, match="explicitly opt in"):
        D.OriginalParentPassDelegation(passes, exporter=module.export, redeemer=module.redeem)


def test_plain_json_and_foreign_source_cannot_supply_live_qualification(original):
    with pytest.raises(Exception, match="ordinary JSON"):
        CHILD.verified_parent_pass_delegation({"success": True})
    passes, module = _selection(original)
    delegation = D.OriginalParentPassDelegation(passes, exporter=module.export, redeemer=module.redeem)
    delegation._record["snapshot"]["payload"] = "forged"
    with pytest.raises(ValueError, match="changed"):
        delegation.revalidate()


def test_separately_sealed_copy_of_v5_record_is_not_the_inherited_parent_object(original):
    passes, module = _selection(original)
    delegation = D.OriginalParentPassDelegation(passes, exporter=module.export, redeemer=module.redeem)
    with (scoped_compiler_passes(passes), D.scoped_original_parent_pass_delegation(delegation),
          PROFILE._scope(original)):
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            # Copy identical sealed bytes, while the document still names the
            # actual original parent's distinct descriptor. Content is equal;
            # kernel-object custody is deliberately different.
            fd = CHILD._memfd()
            try:
                os.write(fd, CHILD._encode_selection(transport.document))
                CHILD.fcntl.fcntl(fd, CHILD._ADD_SEALS, CHILD._SEALS)
                readonly = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
                try:
                    from types import SimpleNamespace

                    fake = SimpleNamespace(fd=readonly, pass_fds=(readonly,))
                    result = PROFILE._run(fake, original)
                finally:
                    os.close(readonly)
            finally:
                os.close(fd)
            assert result.returncode != 0
            assert "inherited sealed kernel object" in result.stderr


def test_expired_child_token_cannot_be_redeemed_after_original_scope(original):
    passes, module = _selection(original)
    delegation = D.OriginalParentPassDelegation(passes, exporter=module.export, redeemer=module.redeem)
    script = PROFILE._CONTROL.replace("    job=json.loads(raw)", '''    job=json.loads(raw)
    from merlin_experiments.phase1.feedback import compiler_pass_delegation as D
    token=D.active_original_parent_pass_qualification()''')
    script = script.replace("print(json.dumps", '''try: token.revalidate()
except ValueError: pass
else: raise AssertionError('expired child token accepted')
print(json.dumps''')
    with (scoped_compiler_passes(passes), D.scoped_original_parent_pass_delegation(delegation),
          PROFILE._scope(original)):
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            result = PROFILE._run(transport, original, script=script)
            assert result.returncode == 0, result.stderr


def test_foreign_decoder_or_inputs_and_direct_decoder_call_are_refused(original):
    passes, module = _selection(original)
    delegation = D.OriginalParentPassDelegation(passes, exporter=module.export, redeemer=module.redeem)
    script = PROFILE._CONTROL.replace("    job=json.loads(raw)", '''    job=json.loads(raw)
    from merlin_experiments.phase1.feedback import compiler_pass_delegation as D
    from merlin.targetgen.compiler_pass_selection import active_compiler_passes
    token=D.active_original_parent_pass_qualification()
    factory=active_compiler_passes().factory
    for inputs in (factory.inputs,dict(factory.inputs,foreign=True)):
      try: token.redeem(factory.callback,inputs,decoder=factory.callback,work_root=spec.parent)
      except ValueError: pass
      else: raise AssertionError('foreign decoder or inputs accepted')''')
    with (scoped_compiler_passes(passes), D.scoped_original_parent_pass_delegation(delegation),
          PROFILE._scope(original)):
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            result = PROFILE._run(transport, original, script=script)
            assert result.returncode == 0, result.stderr


def test_v5_preserves_checked_source_diagnostic_context_and_pending_l3(original):
    import hashlib

    from merlin.targetgen import checked_source_model_diagnostic as S

    passes, module = _selection(original)
    fir = original.private / "source.fir"
    fir.write_text("synthetic nonexecuted fixture")
    fir_sha = hashlib.sha256(fir.read_bytes()).hexdigest()
    facts = original.private / "facts.json"
    facts.write_text(json.dumps(dict(facts=dict(source=dict(config="fixture")), inputs=dict(
        target=original.target.target, fir_sha256=fir_sha,
        firrtl_inputs=[dict(path=str(fir), sha256=fir_sha)]))))
    inputs = dict(passes.factory.inputs, mechanism="source_execution_diagnostic")
    inputs["binding"]["hardware"] = hashlib.sha256(facts.read_bytes()).hexdigest()
    factory = OriginalPassFactory(module.restore, inputs)
    passes = CompilerPassSelection(factory.create(work_root=original.private / "source-admission"),
                                   PassCapabilityBinding(**inputs["binding"]),
                                   mechanisms=("synthetic",), factory=factory)
    selected = S.select_checked_source_diagnostic(passes, target=original.target.target,
                                                 rtl_facts=facts, board_config="fixture")
    job = json.loads(original.spec.read_bytes())
    job["capsule"].update(kind="model", label="dev", oracle_tier_ceiling="L2",
                          required_oracle_tiers=["L0", "L1", "L2", "L3"])
    job["candidate_context"] = {S.CONTEXT_FIELD: selected.record()}
    original.spec.write_text(json.dumps(job))
    delegation = D.OriginalParentPassDelegation(passes, exporter=module.export, redeemer=module.redeem)
    script = PROFILE._CONTROL.replace("    job=json.loads(raw)", '''    job=json.loads(raw)
    from merlin.targetgen import checked_source_model_diagnostic as D
    checked=D.selection_from_context(job['candidate_context'][D.CONTEXT_FIELD],
      capsule=job['capsule'],target=job['target'],package_dir=job['package_dir'])
    assert checked.record()['formal_admission'] is False
    assert checked.record()['promotion_authority']=='NONE'
    assert job['capsule']['required_oracle_tiers']==['L0','L1','L2','L3']''')
    with (scoped_compiler_passes(passes), S.scoped_checked_source_diagnostic(selected),
          D.scoped_original_parent_pass_delegation(delegation), PROFILE._scope(original)):
        with CHILD.prepare_model_child(spec_path=original.spec, environment=dict(os.environ),
                                       package_dir=original.candidate) as transport:
            result = PROFILE._run(transport, original, script=script)
            assert result.returncode == 0, result.stderr
            transport.verify_result(json.loads(result.stdout))
