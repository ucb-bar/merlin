"""Startup reconstructs the same explicit reviewed selection as preparation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.targetgen.rocc.selected_operand_protocol import selected_inputs
from merlin_experiments.corpus import operand_revision as R
from merlin_experiments.phase1.corpus_inputs import require_reviewed_bundle
from test_operand_revision import fixture


@pytest.mark.parametrize('tamper',[False,True])
def test_startup_uses_preparation_review_pin(tmp_path,monkeypatch,tamper):
    contract,after,review,pin,_=fixture(tmp_path)
    contract['runner']={'chipyard_rocc':{'engines':{'spike':{}}}}
    experiment=tmp_path/'release/payload/experiment'
    (experiment/'contracts').mkdir(parents=True)
    (experiment/'rtl_facts').mkdir()
    raw=Path(after['original_facts']['path']).read_bytes()
    after['original_facts']=pin('release/payload/experiment/contracts/operand-facts.json',raw)
    (experiment/'rtl_facts/facts.json').write_bytes(raw)
    after['sidecar']=pin('release/payload/experiment/contracts/operand-protocol.json',Path(after['sidecar']['path']).read_bytes())
    revision_pin=pin('release/payload/experiment/contracts/operand-revision.json',json.dumps(review).encode())
    released=R.select(contract,after,revision_pin)
    path=experiment/'contracts/target_contract.yaml';path.write_text(yaml.safe_dump(released))
    private=tmp_path/'release/private';private.mkdir()
    if tamper:revision_pin['sha256']='0'*64
    (private/'preparation.json').write_text(json.dumps({'scaffolding':{'capability_contract':{'operand_protocol_revision':revision_pin}}}))
    coverage={'capability_contract':contract,'raw_facts_utf8':raw.decode(),'evidence':{'raw_facts_sha256':after['original_facts']['sha256']}}
    monkeypatch.setattr('merlin_experiments.phase0.coverage_commitment.read_inputs',lambda _:coverage)
    monkeypatch.setattr('merlin_experiments.corpus.publication.require_public_capsules',lambda *_:None)
    te=SimpleNamespace(path=experiment/'target_experiment.yaml',capsule_corpus=tmp_path/'release/payload/corpus/public',declared_contract_path=lambda:path,graded_roots=lambda:[])
    bundle={'bundle_id':'checked','host_inputs':[{'path':x} for x in selected_inputs(after)],'allowed':[{'path':str(path),'mode':'ro'}]}
    manifest=experiment/'input_bundles/checked/input_bundle_manifest.yaml'
    if tamper:
        with pytest.raises(ValueError,match='coverage inputs'):require_reviewed_bundle(te,manifest,bundle)
    else:require_reviewed_bundle(te,manifest,bundle)
