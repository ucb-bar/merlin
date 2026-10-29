import json
from pathlib import Path
import pytest
import yaml
from merlin_experiments.phase1 import joint_epochs as E
from merlin_experiments.phase1 import joint_interrupted_epoch as R


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    run = tmp_path/'run'; ws=tmp_path/'workspace'; out=tmp_path/'review'/'epoch.json'
    out.parent.mkdir()
    put(ws/'submission'/'manifest.yaml', {'commands': {}})
    (ws/'submission'/'compiler.py').write_text('original compiler\n')
    (ws/'variants'/'alternative').mkdir(parents=True)
    (ws/'variants'/'alternative'/'compiler.py').write_text('alternative\n')
    put(ws/'qa'/'tier_state.json', {'case': {'L3': {'status':'unknown'}}})
    put(ws/'qa'/'tier_instrument_history'/'history.json', {'old':'evidence'})
    put(run/'joint_campaign.json', {'started_at_epoch':123,'policy':{'duration_s':43200},'gsim_ledger':'original-budget','gsim_correctness_ledger':'original-correctness-budget'})
    put(run/'joint_epoch_stop.json', {'schema':'merlin.joint_epoch_stop.v1','reason':'repair blocked verification'})
    run.joinpath('environment.yaml').write_text(yaml.safe_dump({'workspace_path':str(ws),'backend_origin':{'schema':'merlin.empty_backend_origin.v1','initial_files':0},'lineage_id':'same-origin','run_id':'old'}))
    summary=put(run/'rounds'/'round_00.codex_summary.json', {'operator_interrupted':True,'turn_incomplete':True,'exit_code':130,'process_exit_code':-15})
    cfg=put(tmp_path/'config.json',{'reservation':{'cgroup':'/dedicated'}})
    selected=put(tmp_path/'selection.json',{'configuration':R._pin(cfg),'run_dir':str(run),'argv':['python','native'],'taskset_cpus':[9,10]})
    native=put(tmp_path/'native.json',{'pid':11,'argv':['/usr/bin/taskset','-c','9,10','python','native']})
    cal=put(tmp_path/'cal.json',{'pid':12,'argv':['python','cal','--selection',str(selected)]})
    plan=put(tmp_path/'plan.json',{'schema':'operator.owned_termination_plan.v1','selection':R._pin(selected),'native_owner':R._pin(native),'calibration_owner':R._pin(cal),'author_summary':R._pin(summary),'cgroup':'/dedicated','preserved_residents':[],'processes':[{'pid':11,'start_ticks':'1'},{'pid':12,'start_ticks':'2'}]})
    receipt=put(tmp_path/'termination.json',{'schema':'operator.owned_termination_result.v1','applied':True,'owned_terminal':True,'completion_claimed':False,'unfinished_evaluations':'UNKNOWN','plan':R._pin(plan)})
    monkeypatch.setattr(R.O,'process_identity',lambda row:None)
    monkeypatch.setattr(R,'_process',lambda pid:None)
    monkeypatch.setattr(R,'_members',lambda cg:[])
    return run,ws,out,receipt


def prepare(fixture):
    run,ws,out,receipt=fixture
    return R.prepare(run,owner_capture=receipt,contract_sha256='a'*64,review_id='reviewed interruption',output=out)


def test_interruption_preserves_source_variants_clock_and_unknown(stopped):
    run,ws,out,receipt=stopped
    record=prepare(stopped)
    assert not (run/'joint_campaign_summary.json').exists()
    state,origin=E._verify(record)
    assert state['started_at_epoch']==123 and state['gsim_correctness_ledger']=='original-correctness-budget'
    assert record['interruption']['unfinished_evaluations']=='UNKNOWN'
    assert record['interruption']['current_credit_granted'] is False
    assert Path(record['source'],'compiler.py').read_text()=='original compiler\n'
    assert Path(record['variants'][0]['path'],'compiler.py').read_text()=='alternative\n'
    assert json.loads(Path(record['tier_ledger']['path']).read_text())['case']['L3']['status']=='unknown'
    # A new epoch can occupy the same reservation after stopped custody was established.
    R._members=lambda _: [999]
    E._verify(record)


@pytest.mark.parametrize('mutation',['compiler','variant','ledger','author','state'])
def test_changed_custody_rejected(stopped,mutation):
    run,ws,out,receipt=stopped;record=prepare(stopped)
    path={'compiler':Path(record['source'])/'compiler.py','variant':Path(record['variants'][0]['path'])/'compiler.py','ledger':ws/'qa/tier_state.json','author':run/'rounds/round_00.codex_summary.json','state':run/'joint_campaign.json'}[mutation]
    path.write_text('{}')
    with pytest.raises(ValueError):E._verify(record)


def test_live_or_unowned_process_refused(stopped,monkeypatch):
    monkeypatch.setattr(R,'_members',lambda cg:[55])
    monkeypatch.setattr(R,'_process',lambda pid:{'pid':pid,'start_ticks':'new'} if pid==55 else None)
    with pytest.raises(ValueError,match='owned or unknown'):prepare(stopped)
    assert not stopped[2].exists()


def test_live_captured_owner_refused(stopped,monkeypatch):
    monkeypatch.setattr(R,'_process',lambda pid:{'pid':pid,'start_ticks':'1'} if pid==11 else None)
    with pytest.raises(ValueError,match='remains alive'):prepare(stopped)


def test_exact_author_variants_reach_new_workspace_without_credit(stopped):
    record = prepare(stopped)
    destination = stopped[2].parent / 'new_workspace'
    destination.mkdir()
    rows = R.seed_variants(record, destination)
    assert (destination/'variants/alternative/compiler.py').read_text() == 'alternative\n'
    assert rows[0]['current_credit_granted'] is False
    with pytest.raises(FileExistsError):
        R.seed_variants(record, destination)


def test_pending_claims_require_separate_proven_reconciliation(stopped):
    put(stopped[1]/'qa/tier_state.json', {'case': {'records': {'L3': {'identity': {'status':'pending'}}}}})
    with pytest.raises(ValueError, match='separate proven reconciliation'):
        prepare(stopped)
    assert not (stopped[2].parent/'interrupted_source').exists()


def test_source_run_cannot_be_relabelled(stopped):
    record = prepare(stopped)
    record['source_run_id'] = 'different'
    with pytest.raises(ValueError, match='source run identity'):
        E._verify(record)
