"""A real public cohort projection retains declared inputs and refuses corrupt ones."""
import hashlib
import pytest
import yaml
from merlin.targetgen.contract.materialize import materialize_public_capsules
from merlin_experiments.phase1.corpus_inputs import _resources


def fixture_capsule(tmp_path, *, profile=True):
    root = tmp_path / 'source'
    capsule = root / 'sample'
    capsule.mkdir(parents=True)
    raw = b'{"sequence_length":12}\n'
    attrs = {'loader_profile':'profile.json','loader_profile_sha256':hashlib.sha256(raw).hexdigest()}
    doc = {'name':'sample','kind':'model','label':'public','operation':{'op':'model','attributes':attrs},
           'required_oracle_tiers':['L2','L3']}
    (capsule/'capsule.yaml').write_text(yaml.safe_dump(doc))
    if profile:
        (capsule/'profile.json').write_bytes(raw)
    return root,capsule,doc,raw


def test_projection_preserves_exact_declared_profile_and_tier_policy(tmp_path):
    root,source,doc,raw=fixture_capsule(tmp_path)
    destination=tmp_path/'projected'
    assert materialize_public_capsules(destination,corpus_roots=[root],tier_ceiling='L3') == ['sample']
    assert (destination/'sample/profile.json').read_bytes() == raw
    projected=yaml.safe_load((destination/'sample/capsule.yaml').read_text())
    assert projected['operation']['attributes'] == doc['operation']['attributes']
    assert projected['required_oracle_tiers'] == ['L2','L3']
    assert (source/'profile.json').read_bytes() == raw


@pytest.mark.parametrize('damage',['missing','wrong_bytes','symlink','traversal'])
def test_projection_refuses_missing_or_unbound_dependency(tmp_path,damage):
    root,source,doc,raw=fixture_capsule(tmp_path,profile=damage!='missing')
    if damage=='wrong_bytes':
        (source/'profile.json').write_bytes(b'{}')
    elif damage=='symlink':
        (source/'profile.json').unlink()
        outside=tmp_path/'outside.json';outside.write_bytes(raw)
        (source/'profile.json').symlink_to(outside)
    elif damage=='traversal':
        doc['operation']['attributes']['loader_profile']='../profile.json'
        (source/'capsule.yaml').write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError,match='loader profile'):
        materialize_public_capsules(tmp_path/'projected',corpus_roots=[root],tier_ceiling='L3')


def test_freeze_resources_refuses_missing_declared_profile(tmp_path):
    root,source,doc,raw=fixture_capsule(tmp_path,profile=False)
    with pytest.raises(ValueError,match='loader_profile'):
        _resources([{**doc,'__dir__':str(source)}])
