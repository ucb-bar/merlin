"""Legacy source review never certifies old arrays or invents a producer."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import operator_legacy_source_review as R
from merlin_experiments.phase0 import operator_exclusion_intake as I
import test_operator_exclusion_issuer as U


@pytest.fixture
def owner(tmp_path):
    root=tmp_path/'bundle';root.mkdir()
    program=root/'synthetic/programs/unit';(program/'ref').mkdir(parents=True)
    source=program/'ref/model.prepared.mlir';source.write_text('module {}')
    (program/'catalog.json').write_text(json.dumps(dict(schema='merlin.model_check.catalog.v1',ref_prepared=str(source))))
    (program/'census.json').write_text('{}')
    array=root/'answer.npy';array.write_bytes(b'unread original tensor')
    binding=root/'binding.json';binding.write_text('{}')
    leaves={str(p.relative_to(root)):U.pin(p)['sha256'] for p in root.rglob('*') if p.is_file()}
    sums=root/'SHA256SUMS';sums.write_text(''.join(f'{sha}  {name}\n' for name,sha in sorted(leaves.items())))
    capture=tmp_path/'selected-capture'
    manifest=root/'MANIFEST.json';U.write_json(manifest,dict(schema='merlin.model_check.bundle.v1',target='synthetic',
        models={'synthetic':dict(model='synthetic',programs={'unit':{'source_stage':str(capture/'unit')}})},
        binding={'status':'pass','receipt':'binding.json','receipt_sha256':U.pin(binding)['sha256']},
        seal={'sums':'SHA256SUMS','sums_sha256':U.pin(sums)['sha256'],'files':len(leaves)}))
    spec=tmp_path/'private-spec.json';U.write_json(spec,dict(target='synthetic',models=[{'id':'synthetic','capture':str(capture)}]))
    launch=tmp_path/'launch.json';U.write_json(launch,dict(execution_started=True,
        argv=['python','--private-full-model-spec',str(spec)],environment={'MERLIN_MODEL_CHECK_BUNDLE':str(root)}))
    private=tmp_path/'owner';private.mkdir(mode=0o700)
    args=dict(manifest=U.pin(manifest),sums=U.pin(sums),original_launch=U.pin(launch),private_spec=U.pin(spec),
              review_id='new-source-only-owner-review',output=private/'review.json')
    return SimpleNamespace(root=root,program=program,source=source,array=array,manifest=manifest,
        spec=spec,launch=launch,args=args)


def test_new_review_closes_source_metadata_without_reading_tensor_or_claiming_producer(owner,monkeypatch):
    f=owner;original=Path.read_bytes
    def read(path):
        assert path!=f.array,'unrelated tensor bytes must remain unread'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    pin,count=R.prepare(**f.args)
    assert count==1 and set(R.resolve(pin))=={str(f.source)}
    record=json.loads(original(Path(pin['path'])))
    assert record['qualification_issued'] is False
    assert 'producer_receipt' not in record and 'selected_capture_roots' not in record
    assert I.validation_sources_v2({'prospective_legacy_source_review':pin})==R.resolve(pin)
    assert Path(pin['path']).stat().st_mode&0o777==0o600


@pytest.mark.parametrize('damage',['launch','private-spec','stage','catalogue','namespace','missing-source','binding','index'])
def test_incomplete_or_changed_original_ownership_refuses_without_partial_review(owner,damage):
    f=owner
    if damage=='launch':
        d=json.loads(f.launch.read_text());d['environment']['MERLIN_MODEL_CHECK_BUNDLE']='other';U.write_json(f.launch,d);f.args['original_launch']=U.pin(f.launch)
    elif damage=='private-spec':
        U.write_json(f.spec,dict(target='synthetic',models=[]));f.args['private_spec']=U.pin(f.spec)
    elif damage=='stage':
        d=json.loads(f.manifest.read_text());d['models']['synthetic']['programs']['unit']['source_stage']='/other';U.write_json(f.manifest,d);f.args['manifest']=U.pin(f.manifest)
    elif damage=='catalogue':(f.program/'catalog.json').write_text('{}')
    elif damage=='namespace':(f.root/'extra').write_text('extra')
    elif damage=='missing-source':f.source.unlink()
    elif damage=='binding':(f.root/'binding.json').write_text('changed')
    else:(f.root/'SHA256SUMS').write_text('')
    with pytest.raises(ValueError):R.prepare(**f.args)
    assert not f.args['output'].exists()


def test_existing_review_reopens_exact_original_metadata(owner):
    f=owner;pin,_=R.prepare(**f.args);f.launch.write_text('{}')
    with pytest.raises(ValueError):R.resolve(pin)
