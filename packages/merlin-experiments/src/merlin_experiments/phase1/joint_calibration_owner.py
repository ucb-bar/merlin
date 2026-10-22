#!/usr/bin/env python3
"""Operator continuation owner; adopts an already launched native joint campaign.

No initial author launch, code edits, new policy or library selection. Every
continuation requires an accepted immutable model and the original native drain.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def write(path,value,*,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    else:
        temp=path.with_suffix('.next.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def verify_pin(value):
    if pin(value['path']) != value:
        raise ValueError('operator input changed: '+value['path'])


def replace(argv,key,value):
    if argv.count(key)>1:raise ValueError('repeated native option '+key)
    if key in argv:argv[argv.index(key)+1]=str(value)
    else:argv.extend([key,str(value)])


def environment(selection):
    policy=selection['environment_policy']
    env={k:v for k,v in os.environ.items() if not any(k.startswith(p) for p in policy['remove_prefixes']) and k not in policy['unset']}
    env.update(selection['environment'])
    return env


def process_identity(owner):
    """No PID reuse assumption: a live PID must still be the original native argv."""
    pid=owner.get('pid')
    if type(pid) is not int or pid<=0:raise ValueError('native owner lacks launched PID')
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':return None
        argv=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
    except FileNotFoundError:return None
    expected=owner['argv'];native=expected[3:] if expected[:2]==['/usr/bin/taskset','-c'] else expected
    if argv not in (expected,native):
        raise ValueError('PID no longer identifies the selected native process')
    return {'pid':pid,'start_ticks':stat[19],'session':stat[3]}


def validate_source(selection):
    if not selection.get('ready_for_native_admission'):
        raise ValueError('original native selection was not ready')
    for value in selection['pins']+[selection['configuration'],selection['helper']]:verify_pin(value)
    repo=selection['cwd']
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=selection['source_commit']:
        raise ValueError('native evaluator HEAD changed; automatic continuation requires operator review')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip():
        raise ValueError('native evaluator tracked sources changed')
    if not set(selection['taskset_cpus'])<=os.sched_getaffinity(0):
        raise ValueError('original CPU reservation unavailable')


def prepare_continuation(initial,previous,response,index):
    """Only epoch identity/model selection may change. Original pool paths stay shared."""
    if response.get('status')!='ready_for_native_epoch':raise ValueError('native epoch not drained')
    from merlin_experiments.phase1 import joint_epochs as E
    epoch=read(response['joint_epoch']);state,_=E._verify(epoch)
    if epoch['previous_run']!=previous['run_dir']:raise ValueError('epoch targets another native run')
    if epoch.get('performance_selection_sha256')!=response['performance_selection']['sha256']:
        raise ValueError('epoch and model selection differ')
    if state['policy_sha256']!=read(Path(initial['run_dir'])/'joint_campaign.json')['policy_sha256']:
        raise ValueError('continuation changed original policy')
    if time.time()>=state['started_at_epoch']+state['policy']['duration_s']-state['policy']['drain_s']:
        raise ValueError('original authoring clock expired')
    new=copy.deepcopy(initial)
    name=initial['run_id']+'-model-'+str(index).zfill(3)
    new['run_id']=name;new['run_dir']=str(Path(initial['run_dir']).parent/name)
    replace(new['argv'],'--run-id',name)
    replace(new['argv'],'--joint-epoch',response['joint_epoch'])
    replace(new['argv'],'--codex-home-root',Path(initial['task'])/'private'/('codex-'+name))
    for forbidden in ('--resume','--seed-submission','--qualify-submission'):
        if forbidden in new['argv']:raise ValueError('continuation contains unsupported launch mode')
    if '--fresh-backend' not in new['argv']:raise ValueError('empty-origin flag was lost')
    new['environment'][E.PERFORMANCE_ENV]=json.dumps(response['performance_selection'],sort_keys=True)
    new['pins'] += [pin(response['joint_epoch']),response['performance_selection']]
    new['continuation']={'previous_run':previous['run_dir'],'epoch':pin(response['joint_epoch']),
                         'original_started_at_epoch':state['started_at_epoch'],'index':index}
    return new


def launch_continuation(selection,selection_path,directory):
    validate_source(selection)
    if Path(selection['run_dir']).exists():raise ValueError('new native epoch directory already exists')
    # Keep the existing shared pool/slot namespace; never create a second pool.
    for key in ('MERLIN_COMPUTE_CPU_SLOTS','MERLIN_GSIM_SLOT_ROOT'):
        if not Path(selection['environment'][key]).is_dir():raise ValueError('original shared pool is unavailable')
    directory.mkdir(parents=True,exist_ok=False)
    write(directory/'launch-claim.json',{'selection':pin(selection_path),'claimed_at':time.time()},exclusive=True)
    command=['/usr/bin/taskset','-c',','.join(map(str,selection['taskset_cpus'])),*selection['argv']]
    record={'schema':'operator.joint_native_owner.v1','selection':pin(selection_path),
            'started_at_unix':time.time(),'status':'starting','argv':command,
            'cwd':selection['cwd'],'native_run_dir':selection['run_dir']}
    write(directory/'owner.json',record,exclusive=True)
    with (directory/'native.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=selection['cwd'],env=environment(selection),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record.update(pid=child.pid,process_group=child.pid,status='native_runner_started')
    write(directory/'owner.json',record)
    return child,record


def start_watcher(selection,profile,output,directory):
    directory.mkdir(parents=True,exist_ok=False)
    command=['/usr/bin/taskset','-c',','.join(map(str,selection['taskset_cpus'])),selection['argv'][0],'-B','-m',
        'merlin_experiments.phase1.joint_calibration_service','--profile',profile['path'],
        '--profile-sha256',profile['sha256'],'--state',str(Path(selection['run_dir'])/'joint_campaign.json'),
        '--descriptor',selection['argv'][selection['argv'].index('--descriptor')+1],
        '--repo',selection['cwd'],'--output',str(output),'--poll-seconds','30']
    with (directory/'watcher.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=selection['cwd'],env=environment(selection),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    write(directory/'owner.json',{'pid':child.pid,'argv':command,'started_at':time.time()},exclusive=True)
    return child


def refresh_model_priority(selection, state, output, questions_pin=None):
    """Advisory report failure cannot stop native authoring or grant calibration."""
    from .joint_calibration_priority import refresh
    from .joint_calibration_admission import ENV
    try:
        denial = selection['environment'].get(ENV)
        refresh(state=state,run_dir=selection['run_dir'],output=Path(output)/'model_priority.json',
                denial_pin=json.loads(denial) if denial else None,questions_pin=questions_pin)
    except (ValueError,OSError,RuntimeError,KeyError,TypeError) as error:
        write(Path(output)/'model_priority.json',dict(status='blocked_coverage_review',
            reason=str(error),dispatch_authorized=False,fit_authorized=False))


def validate_replaced_owners(selection, selection_pin, owners, *, preview=False):
    """Bind both retired processes to this selection and its watcher directory."""
    if len(owners)!=2:raise ValueError('two previous calibration owners required')
    for value in owners:verify_pin(value)
    records=[read(value['path']) for value in owners]
    supervisors=[value for value in records if value.get('selection')==selection_pin]
    if len(supervisors)!=1 or len({value['pid'] for value in records})!=2:
        raise ValueError('previous calibration owners do not bind this native selection')
    supervisor=supervisors[0]
    watcher=next(value for value in records if value is not supervisor)
    argv=supervisor['argv'];watch_argv=watcher['argv']
    def option(command,key):
        if command.count(key)!=1:raise ValueError('previous calibration owner option is ambiguous')
        return command[command.index(key)+1]
    if (option(argv,'-m')!='merlin_experiments.phase1.joint_calibration_owner'
            or option(argv,'--selection')!=selection_pin['path']
            or option(argv,'--selection-sha256')!=selection_pin['sha256']
            or option(watch_argv,'-m')!='merlin_experiments.phase1.joint_calibration_service'
            or option(watch_argv,'--state')!=str(Path(selection['run_dir'])/'joint_campaign.json')
            or option(watch_argv,'--output')!=option(argv,'--calibration-output')):
        raise ValueError('previous watcher does not bind the original calibration owner')
    watcher_pin=next(pin for pin,record in zip(owners,records,strict=True) if record is watcher)
    if not Path(watcher_pin['path']).is_relative_to(Path(option(argv,'--output'))):
        raise ValueError('previous watcher receipt is outside its original owner')
    if not preview and any(process_identity(value) is not None for value in records):
        raise ValueError('previous calibration supervisor or watcher is still running')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('selection','selection-sha256','native-owner','output','calibration-output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--max-epochs',type=int,default=16)
    p.add_argument('--coverage-questions',help='Optional operator JSON {path,sha256}; advisory only')
    p.add_argument('--admission-review-successor',
                   help='Operator JSON {path,sha256}; missing-evidence repair only, no native continuation')
    p.add_argument('--superseded-owner',action='append',default=[],
                   help='Pinned previous calibration supervisor/watcher receipt; both must be stopped')
    p.add_argument('--once',action='store_true',help='inspect adoption only; never launch a watcher or continuation')
    a=p.parse_args(argv)
    if not 1<=a.max_epochs<=32:p.error('max epochs must be1..32')
    questions_pin=json.loads(a.coverage_questions) if a.coverage_questions else None
    if questions_pin:verify_pin(questions_pin)
    initial_pin={'path':str(Path(a.selection).resolve()),'sha256':a.selection_sha256};verify_pin(initial_pin)
    initial=read(initial_pin['path']);validate_source(initial)
    owner=read(a.native_owner)
    if owner.get('selection')!=initial_pin or owner.get('native_run_dir')!=initial['run_dir']:
        raise ValueError('owner does not identify the initial prepared selection')
    if a.admission_review_successor:
        from .joint_calibration_review_successor import select
        initial=select(initial,json.loads(a.admission_review_successor))
        if len(a.superseded_owner)!=2:
            raise ValueError('review replacement requires both previous calibration owner receipts')
        superseded=[json.loads(value) for value in a.superseded_owner]
        for value in superseded:verify_pin(value)
        validate_replaced_owners(initial,initial_pin,superseded,preview=a.once)
    elif a.superseded_owner:
        raise ValueError('superseded owners require an evidence-only review successor')
    output=Path(a.output).resolve();calibration=Path(a.calibration_output).resolve()
    if output==calibration or output.is_relative_to(calibration) or calibration.is_relative_to(output):
        raise ValueError('operator ownership and calibration evidence roots must be separate')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=output/'adoption.json'
        if manifest.exists():
            raise ValueError('continuation owner already claimed; inspect its retained state rather than retrying blindly')
        identity=process_identity(owner)
        adoption={'initial_selection':initial_pin,'initial_owner':pin(a.native_owner),
                  'native_process':identity,'script':pin(__file__),'calibration_output':str(calibration),'coverage_questions':questions_pin}
        if initial.get('calibration_review_successor'):
            adoption['calibration_review_successor']=initial['calibration_review_successor']
            adoption['superseded_owners']=superseded
        if a.once:
            print(json.dumps({'status':'adoption_checked_no_launch',**adoption}));return 0
        if identity is None:raise ValueError('initial native process already exited; no fresh launch permitted')
        write(manifest,adoption,exclusive=True)
        current=initial;index=0;watcher=None;native_child=None;notified_exit=False
        while True:
            if native_child is not None:native_child.poll()
            alive=process_identity(owner)
            if alive is not None and alive != identity:
                raise ValueError('native PID start identity changed; refuse automatic continuation')
            state_path=Path(current['run_dir'])/'joint_campaign.json'
            state=read(state_path) if state_path.exists() else None
            request=state_path.with_name('joint_model_epoch_request.json')
            if state and state.get('started_at_epoch') is not None:
                refresh_model_priority(current,state,output,questions_pin)
                deadline=state['started_at_epoch']+state['policy']['duration_s']
                if time.time()>deadline+1200:
                    raise ValueError('original campaign plus bounded calibration shutdown window exhausted')
                if watcher is None and alive and not request.exists():
                    profile=state['policy']['native_calibration'];verify_pin(profile)
                    watcher=start_watcher(current,profile,calibration,output/f'watcher-{index:03d}')
                if request.exists() and (watcher is None or watcher.poll() is not None):
                    if initial.get('calibration_review_successor'):
                        raise ValueError('evidence-only review owner cannot activate a native model epoch')
                    proposal=read(request)
                    proposed=read(proposal['selection']['path'])
                    prior_selection=state.get('performance_selection')
                    prior=read(prior_selection['path']) if prior_selection else {}
                    if (proposed.get('runtime_revisions', []) != prior.get('runtime_revisions', [])
                            and proposed.get('timing_epochs', []) == prior.get('timing_epochs', [])):
                        from merlin_experiments.phase1.joint_runtime_revision import request_promotion
                    else:
                        from merlin_experiments.phase1.joint_calibration import request_promotion
                    response=request_promotion(proposal['selection'],previous_run=current['run_dir'],
                        review_id=proposal['review_id'],output=proposal['output'])
                    if response['status']=='ready_for_native_epoch' and alive is None:
                        if index>=a.max_epochs:raise ValueError('reviewed model epoch count exhausted')
                        new=prepare_continuation(initial,current,response,index+1)
                        target=output/f'epoch-{index+1:03d}.json';write(target,new,exclusive=True)
                        native_child,owner=launch_continuation(new,target,output/f'native-{index+1:03d}')
                        identity=process_identity(owner)
                        if identity is None:raise ValueError('continued native startup exited immediately; no automatic retry')
                        current=new;index+=1;watcher=None;notified_exit=False
                        write(output/'current.json',{'selection':pin(target),'owner':pin(output/f'native-{index:03d}'/'owner.json'),'epoch':index})
                        continue
                elif alive is None:
                    # Ask the watcher to finish its bounded original operation;
                    # never kill a simulator/observer process or leave it behind.
                    if watcher is not None and watcher.poll() is None:
                        if not state_path.with_name('joint_epoch_stop.json').exists():
                            from merlin_experiments.phase1.joint_epochs import request_stop
                            request_stop(current['run_dir'],reason='native owner exited; close calibration without continuation')
                        notified_exit=True
                    else:
                        write(output/'status.json',{'status':'native_campaign_finished','run_dir':current['run_dir'],'epochs':index});return 0
                elif watcher is not None and watcher.poll() is not None and not request.exists():
                    status=read(calibration/'status.json') if (calibration/'status.json').exists() else {}
                    if status.get('status')!='native_drain_or_campaign_stop':
                        raise ValueError('calibration watcher exited without native drain request; inspect watcher log')
            elif alive is None:
                raise ValueError('native startup exited before its authoring clock began; no automatic retry')
            watcher_status=read(calibration/'status.json') if (calibration/'status.json').exists() else {}
            write(output/'status.json',{'status':'waiting_for_native_start' if not state or state.get('started_at_epoch') is None
                else 'closing_watcher' if notified_exit else 'calibration_supervision',
                'run_dir':current['run_dir'],'epochs':index,'native_alive':alive is not None,
                'watcher_pid':watcher.pid if watcher else None,
                'calibration_status':watcher_status.get('status'),'calibration_reason':watcher_status.get('reason')})
            time.sleep(15)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,RuntimeError,KeyError,TypeError,subprocess.SubprocessError) as error:
        # No automatic relaunch after uncertain process ownership.
        print(json.dumps({'status':'blocked_operator_continuation','reason':str(error),'exception':type(error).__name__}),flush=True)
        raise SystemExit(1)
