import hashlib
import json

from merlin_experiments.phase1.feedback.joint_diagnostics import project


def test_diagnostics_keep_missing_and_mismatched_evidence_unknown(tmp_path):
    path = tmp_path / 'receipt.json'
    path.write_text('{"emitted":{"pack_calls":2}}')
    performance = {'status':'PROVISIONAL','scope_sha256':'s','model_sha256':'m',
                   'modeled_resource_activity':{'scope':'whole_trace'}}
    record = {'pins':{'compiler_sha256':'c','capsule_sha256':'x','scope_sha256':'s','model_sha256':'m'},
              'artifact':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
              'planned':{'pack_calls':1},'emitted':{'pack_calls':2,'raw_log':'do not emit'}}
    report = project(performance, compiler_sha256='c',capsule_sha256='x',correctness='pass',
                     observations={'packing_reuse':record,'wholechain':record})
    assert report['sections']['packing_reuse']['status'] == 'OBSERVED_PARTIAL'
    assert report['sections']['packing_reuse']['emitted'] == {'pack_calls':2}
    assert report['sections']['wholechain']['status'] == 'UNKNOWN'
    assert not report['optimization_savings_estimated'] and not report['resource_cycles_additive']
    assert len(json.dumps(report)) < 8000
    record['pins']['scope_sha256']='wrong'
    assert project(performance,compiler_sha256='c',capsule_sha256='x',correctness='fail',
                   observations={'packing_reuse':record})['sections']['packing_reuse']['status']=='UNKNOWN'
    record['pins']['scope_sha256']='s'
    path.write_text('changed')
    assert project(performance,compiler_sha256='c',capsule_sha256='x',correctness='pass',
                   observations={'packing_reuse':record})['sections']['packing_reuse']['status']=='UNKNOWN'
