"""Reviewed source-only handoff starts a new clock without inherited verdicts."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from merlin_experiments.phase1 import joint_control as J, joint_seed as S, fresh_backend as F
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.run_inputs import graded_source_hash


@pytest.fixture
def selected(tmp_path, monkeypatch):
    # J.configure assigns os.environ[J.ENV] itself; setenv records the original state first so teardown
    # restores it (delenv alone records nothing when unset, and the selection leaks into later tests).
    monkeypatch.setenv(J.ENV, '')
    monkeypatch.delenv(J.ENV)
    monkeypatch.delenv('MERLIN_JOINT_PERFORMANCE_SELECTION', raising=False)
    source = tmp_path / 'checkpoint' / 'submission'
    source.mkdir(parents=True)
    (source / 'manifest.yaml').write_text('version: 1\n')
    (source / 'compiler.py').write_text('authored = True\n')
    overlay = source / 'compiler_library/merlin/frontend.py'
    overlay.parent.mkdir(parents=True)
    overlay.write_text('overlay = True\n')
    alternative = tmp_path / 'alternative'
    alternative.mkdir()
    (alternative / 'manifest.yaml').write_text('version: 1\n')
    review = {'schema':'operator.joint_seed_review.v1','approved':True,
        'review_id':'explicit-user-amendment','source':str(source),'sha256':graded_source_hash(source),
        'inherit_clock':False,'inherit_results':False,
        'variants':[{'name':'authored_alternative','path':str(alternative),'sha256':graded_source_hash(alternative)}]}
    path = tmp_path / 'review.json'; path.write_text(json.dumps(review))
    origin = {k:v for k,v in review.items() if k not in ('schema','approved')}
    origin.update(schema=S.SCHEMA,review={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps({'version':1,'duration_s':21600,'final_model_s':3600,'drain_s':3600,'compiler_origin':origin}))
    options = replace(parse_options(['--run-id','new']),joint_campaign=str(policy),seed_submission=str(source),
                      fresh_backend=False,sandbox='bwrap',schedule='continuous')
    return options, source, origin


def test_exact_seed_stages_source_overrides_and_alternatives_without_old_credit(selected, tmp_path):
    options, source, origin = selected
    J.validate(options); F.validate(options)
    workspace=tmp_path/'new-workspace'; workspace.mkdir()
    run=tmp_path/'new-run'; run.mkdir()
    result=S.stage(options,workspace,run)
    assert (workspace/'submission/compiler.py').read_bytes()==(source/'compiler.py').read_bytes()
    assert (workspace/'submission/compiler_library/merlin/frontend.py').is_file()
    assert (workspace/'variants/authored_alternative/manifest.yaml').is_file()
    assert result['current_credit_granted'] is False and result['inherited_clock'] is False
    assert result['selected_source_tree_sha256']==origin['sha256']
    J.configure(options,run)
    state=json.loads((run/'joint_campaign.json').read_text())
    assert state['started_at_epoch'] is None
    assert 'gsim_ledger' not in state and not list(run.glob('*budget*'))
    assert F.origin(options,workspace) is None  # Never manufactures empty origin.


@pytest.mark.parametrize('change',[{'fresh_backend':True},{'resume':True},{'joint_epoch':'old'}, {'seed_submission':'other'}, {'allow_unsandboxed':True}])
def test_conflicting_origin_refused(selected,change):
    options,_,_=selected
    with pytest.raises(ValueError): J.validate(replace(options,**change))


def test_source_and_review_mutation_refused(selected):
    options, source, origin = selected
    p=source/'compiler.py';old=p.read_bytes();p.write_text('changed=True\n')
    with pytest.raises(ValueError,match='bytes changed'):J.validate(options)
    p.write_bytes(old)
    Path(origin['review']['path']).write_text('{}')
    with pytest.raises(ValueError,match='review changed'):J.validate(options)


def test_unbound_seed_does_not_weaken_empty_origin(selected):
    options,_,_=selected
    p=Path(options.joint_campaign);value=json.loads(p.read_text());del value['compiler_origin'];p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='fresh-backend'):J.validate(options)
    with pytest.raises(ValueError,match='empty implementation'):F.validate(replace(options,fresh_backend=True))
