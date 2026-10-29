"""No compiler or simulator runs: exercise actual broker ownership and drain."""
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from merlin_experiments.phase1.brokers import model_check as B


def request(channel, name, model='resnet50'):
    (channel / f'req_{name}.json').write_text(json.dumps(dict(action='check',model=model,split='visible')))


def until(predicate):
    end=time.monotonic()+3
    while not predicate():
        assert time.monotonic()<end
        time.sleep(.005)


def test_two_independent_requests_overlap_and_stop_drains_without_starting_third(tmp_path, monkeypatch):
    channel=tmp_path/'.model_check_channel';channel.mkdir()
    for i in range(3):request(channel,str(i))
    entered=[];finished=[];lock=threading.Lock();started=[];joined=[]
    context=object();manifest=tmp_path/'selected-manifest'
    def execute(ws, payload, **kw):
        assert kw['context'] is context and kw['compiler_bundle_manifest']==manifest
        assert 'snapshot_source' not in kw  # Every original execute snapshots independently.
        with lock:entered.append(threading.current_thread().name)
        while not kw['stop']():time.sleep(.005)
        finished.append(threading.current_thread().name)
        return {'status':'cancelled'}
    original_thread=B.library_thread;original_join=B.join_background
    def owned_thread(function, **kw):
        started.append(kw['name']);return original_thread(function,**kw)
    def owned_join(thread):joined.append(thread.name);return original_join(thread)
    monkeypatch.setattr(B,'execute_visible',execute)
    monkeypatch.setattr(B,'library_thread',owned_thread);monkeypatch.setattr(B,'join_background',owned_join)
    broker=threading.Thread(target=lambda:B.serve(tmp_path,poll=.01,max_active=2,stop=lambda:(channel/'STOP').exists(),
        context=context,compiler_bundle_manifest=manifest))
    broker.start();until(lambda:len(entered)==2)
    (channel/'STOP').touch();broker.join(3)
    assert not broker.is_alive() and len(entered)==len(finished)==2
    assert sorted(started)==sorted(joined)
    assert len(list(channel.glob('resp_*.json')))==3
    assert all(json.loads(p.read_text())['status']=='cancelled' for p in channel.glob('resp_*.json'))


def test_bounded_queue_defers_overflow_and_stop_drains_active_workers(tmp_path, monkeypatch):
    channel=tmp_path/'.model_check_channel';channel.mkdir()
    monkeypatch.setattr(B,'MAX_QUEUED_REQUESTS',2)
    for i in range(5):request(channel,str(i))
    entered=[];finished=[]
    def execute(ws,payload,**kw):
        entered.append(1)
        while not kw['stop']():time.sleep(.005)
        finished.append(1);return {'status':'cancelled'}
    monkeypatch.setattr(B,'execute_visible',execute)
    broker=threading.Thread(target=lambda:B.serve(tmp_path,poll=.01,max_active=2,stop=lambda:(channel/'STOP').exists()))
    broker.start();until(lambda:len(entered)==2)
    until(lambda:any(json.loads(p.read_text())['status']=='deferred' for p in channel.glob('resp_*.json')))
    (channel/'STOP').touch();broker.join(3)
    assert not broker.is_alive() and len(finished)==2


def test_original_execution_uses_distinct_scratch_and_explicit_custody_arguments(tmp_path, monkeypatch):
    source=tmp_path/'submission';source.mkdir();(source/'manifest.yaml').write_text('commands: {}\n')
    monkeypatch.setenv('TMPDIR',str(tmp_path))
    monkeypatch.setattr(B,'target_from_contract',lambda:'gemmini')
    monkeypatch.setattr(B.Bundle,'default',lambda target:SimpleNamespace(root=tmp_path/'bundle'))
    monkeypatch.setattr(B,'default_models_path',lambda target:tmp_path/'models.yaml')
    monkeypatch.setattr(B,'child_arguments',lambda:('--compiler-library-root','selected-library'))
    monkeypatch.setattr(B,'context_argv',lambda context:('--descriptor','selected-descriptor'))
    barrier=threading.Barrier(2);seen=[]
    class Done:
        returncode=0
        def __init__(self,argv,**kw):
            seen.append(argv);barrier.wait(timeout=3)
            Path(argv[argv.index('--out')+1]).write_text(json.dumps(dict(schema=B.REPORT_SCHEMA,
                command='check',model='resnet50',split='visible',agent_view=True,status='pass',exit_code=0)))
        def poll(self):return 0
    monkeypatch.setattr(B.subprocess,'Popen',Done)
    results=[]
    def run():results.append(B.execute_visible(tmp_path,dict(action='check',model='resnet50',split='visible'),
        context=object(),compiler_bundle_manifest=tmp_path/'manifest.json'))
    threads=[threading.Thread(target=run) for _ in range(2)]
    for t in threads:t.start()
    for t in threads:t.join(4)
    assert len(results)==2 and all(r['status']=='pass' for r in results)
    assert len({argv[argv.index('--submission')+1] for argv in seen})==2
    assert len({argv[argv.index('--work')+1] for argv in seen})==2
    assert all('--compiler-library-root' in argv and '--compiler-bundle-manifest' in argv and '--descriptor' in argv for argv in seen)


def test_configured_four_active_requests_are_not_serialized(tmp_path, monkeypatch):
    channel=tmp_path/'.model_check_channel';channel.mkdir()
    for i in range(5):request(channel,str(i))
    entered=[]
    def execute(ws,payload,**kw):
        entered.append(1)
        while not kw['stop']():time.sleep(.005)
        return {'status':'cancelled'}
    monkeypatch.setattr(B,'execute_visible',execute)
    broker=threading.Thread(target=lambda:B.serve(tmp_path,poll=.01,max_active=4,
        stop=lambda:(channel/'STOP').exists()))
    broker.start();until(lambda:len(entered)==4)
    (channel/'STOP').touch();broker.join(3)
    assert not broker.is_alive() and len(entered)==4
    assert len(list(channel.glob('resp_*.json')))==5
