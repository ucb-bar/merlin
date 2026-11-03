"""Overlap original47 backfill with a running attention-v2 capture; final closure is separate."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
MERLIN_ROOT = HERE.parents[2]
ATLAS_ROOT = Path(os.environ['ATLAS_ROOT']).absolute()
ROOT = Path(os.environ['TINY_WORK_ROOT']).absolute()
ATLAS_HELPERS = ATLAS_ROOT/'tools/tiny_full'
MERLIN_HELPERS = MERLIN_ROOT/'examples/atlas/comparisons'
MODEL_ROOT = Path(os.environ['ATLAS_NPU_MODEL_ROOT']).absolute()
spec = importlib.util.spec_from_file_location('d1_v2_stream_worker', MERLIN_HELPERS / 'd1-v2-stream-worker.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
s, w = v.s, v.w
legacy = s.load('d1-stream-original')
# Discovery is unchanged: complete native projections only, original attention skipped.
legacy.w = w


def freeze(out, resume):
    path = s.safe(out/'source-freeze.json')
    current = w.source_state()
    if path.exists():
        if not resume:
            raise ValueError('initialized stream requires --resume')
        saved = json.loads(path.read_bytes())
        w.verify_freeze(saved)
        if saved != current:
            raise ValueError('source freeze differs')
        return saved
    w.save(path,current)
    return current


def result_path(out, path):
    return s.safe(out/'records'/(path.parent.name+'.json'))


def map_bundle(args, out, frozen):
    qualify = s.load('d1-v2-qualify')
    q = qualify.q
    bundle = q.Bundle(args.map_assets,args.smoke)
    model = q.load_model()
    model_sources,harness_sources = model.source_binding(),q.harness()
    candidates = {}
    projections,partial = legacy.discover(s.safe(args.capture))
    if partial:
        raise ValueError('final mapping refuses incomplete capture records')
    for path in projections:
        target = result_path(out,path)
        if not target.exists():
            raise ValueError('missing actual original47 projection backfill: '+str(path))
        record = w.audit(args.capture,path,target,frozen)
        key = (record['program_sha256'],record['arena_initial']['sha256'],record['arena_expected']['sha256'])
        candidates.setdefault(key,[]).append(dict(root=str(s.safe(args.capture)),record=str(target)))
    calls = s.safe(Path(args.capture)/'calls')
    with os.scandir(calls) as entries:
        inventory = sorted(entries,key=lambda entry:entry.name)
    for entry in inventory:
        if entry.is_symlink():
            raise ValueError('capture inventory contains symlink')
        if not entry.is_dir(follow_symlinks=False):
            continue
        path = s.safe(Path(entry.path)/'record.json')
        if not path.exists():
            continue
        record = json.loads(path.read_bytes())
        if record.get('schema') == 'atlas.tiny.attention_launch.v2':
            key = (record['program_sha256'],record['arena_initial']['sha256'],record['arena_expected']['sha256'])
            candidates.setdefault(key,[]).append(dict(root=str(s.safe(args.capture)),record=str(path)))
    mapping = {}
    for item in bundle.launches:
        source,binary,initial,expected,final = bundle.payloads(item)
        selected = candidates.get((s.sha(binary),s.sha(initial),s.sha(final)),[])
        if not selected:
            raise ValueError('missing exact original47 candidate for final launch '+str(item['index']))
        q.from_capture(selected[0],bundle,item,model,model_sources,harness_sources)
        mapping[str(item['index'])] = selected[0]
    bundle.pins.verify()
    w.verify_freeze(frozen)
    w.save(out/'reuse-records.json',mapping)
    print(json.dumps(dict(status='FINAL_BUNDLE_MAPPING_AUDITED',mapped=len(mapping),required=len(bundle.launches),
        mapping=str(out/'reuse-records.json'),overall_acceptance=False)),flush=True)
    return 0


def main(args):
    capture,out = s.safe(args.capture),s.safe(args.out)
    if not out.is_relative_to(ROOT/'d1-v2-stream-original') or out.is_relative_to(capture) or capture.is_relative_to(out):
        raise ValueError('output must be in D/d1-v2-stream-original and disjoint from capture')
    if not 1 <= args.workers <= 16 or not 1 <= args.poll_seconds <= 30 or args.expected_native_records < 1:
        raise ValueError('invalid worker, poll interval, or expected inventory')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('records','logs'):
        s.safe(out/name).mkdir(exist_ok=True)
    frozen = freeze(out,args.resume)
    if args.map_assets:
        return map_bundle(args,out,frozen)
    if args.freeze_only:
        print(json.dumps(dict(status='FROZEN_NO_EXECUTION',freeze=str(out/'source-freeze.json'))),flush=True)
        return 0
    if (out/'failures.json').exists():
        raise ValueError('prior failure requires investigation and a fresh output directory')
    completed,identities,failed,running = {},{},{},{}

    def discover():
        records,partial = legacy.discover(capture)
        if len(records) > args.expected_native_records:
            raise ValueError('projection inventory exceeds expected count')
        if args.record:
            wanted = s.safe(args.record)
            if wanted not in records:
                raise ValueError('record must name a completed native projection')
            records = [wanted]
        return records,partial

    discovered,partial = discover()
    for path in discovered:
        target = result_path(out,path)
        if target.exists():
            if not args.resume:
                raise ValueError('existing receipt requires --resume')
            result = w.audit(capture,path,target,frozen)
            completed[str(path)] = str(target)
            identities[str(path)] = result['source_capture_record_sha256']

    def progress(status):
        w.save(out/'progress.json',dict(schema='atlas.d1.v2.stream_original_progress.v1',status=status,
            model_pin=w.PIN,numerics='rtl',accelerated=False,native_dispatch_enabled=False,
            speculative_capture_input_numerical_evidence=True,final_full_source_capture_replay_closure_required=True,
            overall_acceptance=False,capture_root=str(capture),completed=completed,failed=failed,
            running=sorted(str(path) for path in running.values()),discovered=len(discovered),incomplete_records=partial,
            source_freeze=str(out/'source-freeze.json'),command=sys.argv,updated_unix=time.time()))

    def execute(path):
        command = [sys.executable,'-B',str(MERLIN_HELPERS / 'd1-v2-stream-worker.py'),'--capture',str(capture),'--record',str(path),
                   '--out',str(result_path(out,path)),'--freeze',str(out/'source-freeze.json')]
        environment = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
                           ATLAS_NPU_MODEL_NATIVE='0',NPU_MODEL_RTL_NATIVE_MXU0='0',TMPDIR=str(out))
        environment.pop('PYTHONPATH',None)
        environment.pop('NPU_MODEL_RTL_NATIVE_BUILD_DIR',None)
        with s.safe(out/'logs'/(path.parent.name+'.log')).open('w') as log:
            code = subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:
            raise ValueError('original47 worker failed; inspect '+str(out/'logs'/(path.parent.name+'.log')))
        return w.audit(capture,path,result_path(out,path),frozen)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            w.verify_freeze(frozen)
            for path in discovered:
                key,digest = str(path),s.sha(path.read_bytes())
                if key in identities and identities[key] != digest:
                    raise ValueError('capture record changed after discovery')
                if key not in completed and path not in running.values() and len(running) < args.workers:
                    identities[key] = digest
                    running[pool.submit(execute,path)] = path
            progress('RUNNING_SPECULATIVE_NUMERICS')
            if running:
                ready,_ = wait(running,timeout=args.poll_seconds,return_when=FIRST_COMPLETED)
                for future in ready:
                    path = running.pop(future)
                    try:
                        future.result()
                        completed[str(path)] = str(result_path(out,path))
                        print(path.parent.name+' PASS original47',flush=True)
                    except Exception as error:
                        failed[str(path)] = str(error)
                        w.save(out/'failures.json',failed)
                if failed:
                    break
            if not running and all(str(path) in completed for path in discovered):
                if args.once or args.record or len(completed) == args.expected_native_records and not partial:
                    break
            if not args.once and not args.record:
                if not running:
                    time.sleep(args.poll_seconds)
                discovered,partial = discover()
    w.verify_freeze(frozen)
    progress('FAILED' if failed else 'PASS_DISCOVERED_SPECULATIVE_NUMERICS')
    w.save(out/'receipt.json',json.loads((out/'progress.json').read_bytes()))
    return 1 if failed else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',required=True)
    parser.add_argument('--out',default=str(ROOT/'d1-v2-stream-original/full'))
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--poll-seconds',type=float,default=10)
    parser.add_argument('--expected-native-records',type=int,default=840)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--record')
    parser.add_argument('--freeze-only',action='store_true')
    parser.add_argument('--map-assets')
    parser.add_argument('--smoke',action='store_true')
    raise SystemExit(main(parser.parse_args()))
