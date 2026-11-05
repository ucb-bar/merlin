"""Backfill immutable capture events using the clean original RTL model.

PASS reuse and final admission share the strict auditor and require a durable,
receipt-bound successful worker exit. Archived gate files remain immutable.
"""
import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import struct
from smol_capture_audit import PIN, bindings, digest as sha, payload, require, safe, validate_completed as _validate_completed


def fp8_observation(event):
    stats=event.get('fp8_stats')
    if event.get('output_dtype')=='fp8':
        require(isinstance(stats,dict) and stats.get('scale_exponent')==event['output_scale_exponent']
            and stats.get('physical_values')==event['output_bytes'],'required FP8 observation missing or differs')
    return stats


def validate_completed(event, frozen, receipt, completion, current):
    saved=_validate_completed(event,frozen,receipt,completion,current)
    fp8_observation(event)
    require(saved.get('fp8_stats')==event.get('fp8_stats'),'original FP8 observation differs')
    return saved


def validate_projection_observations(events, manifest):
    """Derive FP8 requirements from the independent complete projection inventory."""
    kernels=manifest.get('projection_reference',{}).get('kernels',[])
    for event in events:
        if event['kind']!='projection':continue
        index=event['projection_index']
        require(type(index) is int and 0<=index<len(kernels),'projection observation inventory differs')
        fusion=kernels[index].get('fusion',{})
        if fusion.get('output_dtype')=='fp8':
            stats=event.get('fp8_stats')
            require(isinstance(stats,dict) and stats.get('scale_exponent')==fusion['output_scale_exponent']
                and stats.get('physical_values')==fusion['output_bytes']==event['output_bytes'],
                'required manifest FP8 observation missing or differs')
            # Legacy M3 events have observations but no typed event marker.
            require(event.get('output_dtype','fp8')=='fp8'
                and event.get('output_scale_exponent',fusion['output_scale_exponent'])==fusion['output_scale_exponent'],
                'event FP8 type or scale differs from manifest')
        else:
            require(event.get('output_dtype')!='fp8' and 'fp8_stats' not in event,
                'unexpected FP8 observation outside manifest producer inventory')

def interpreter(value):
    path = Path(value).absolute()
    for _ in range(32):
        require('..' not in path.parts, 'parent traversal in interpreter path')
        for prefix in reversed([path, *path.parents]):
            if prefix.is_symlink():
                target = Path(os.readlink(prefix))
                if not target.is_absolute():
                    target = prefix.parent / target
                path = Path(os.path.normpath(target / path.relative_to(prefix)))
                break
        else:
            safe(path)
            return Path(value).absolute()
    raise ValueError('interpreter symlink loop')


def worker(event_path, out, args):
    import importlib.util
    os.environ["ATLAS_NPU_MODEL_ROOT"] = str(args.model_root)
    os.environ["ATLAS_NPU_MODEL_PIN"] = PIN
    os.environ.pop("NPU_MODEL_RTL_NATIVE_MXU0", None)
    os.environ.pop("NPU_MODEL_RTL_NATIVE_BUILD_DIR", None)
    spec = importlib.util.spec_from_file_location("smol_original_harness", args.harness)
    unit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = unit
    spec.loader.exec_module(unit)
    binding=unit.model_binding()
    if binding!=dict(commit=PIN,numerics='rtl',root=str(args.model_root)):
        raise ValueError('backfill requires original model')
    if subprocess.check_output(['git','-C',str(args.model_root),'status','--porcelain'],text=True).strip():
        raise ValueError('original model sources are dirty')
    current=bindings(args.model_root,__file__,args.harness)
    event=json.loads(safe(event_path).read_text())
    started=time.perf_counter()
    record=dict(index=event['index'],kind=event['kind'],status='RUNNING',model=binding,
        event_path=str(event_path),event_sha256=sha(safe(event_path).read_bytes()),
        runner_sha256=sha(Path(__file__).read_bytes()),harness_sha256=sha(args.harness.read_bytes()),
        source_binding_schema=current.source_binding_schema,helper_source_sha256=current.helper_source_sha256,
        original_model=True,native_backend=False,math_override=False)
    try:
        canonical_source=payload(event['source']).decode()
        canonical_program=payload(event['program'])
        source=canonical_source
        raw_program=canonical_program
        if 'runtime_program' in event:
            raw_program=payload(event['runtime_program'])
            source=payload(event['runtime_source']).decode()
            epilogue=struct.pack('<6I',0x13,0xc1102ff3,0x13,0xc10f9073,0x13,0x73)
            if raw_program!=canonical_program[:-4]+epilogue:
                raise ValueError('runtime completion program differs from canonical compute plus fixed epilogue')
            if event['runtime_epilogue_sha256']!=sha(epilogue):
                raise ValueError('runtime epilogue binding differs')
        record['canonical_program_sha256']=sha(canonical_program)
        record['runtime_program_executed']='runtime_program' in event
        record['initial_csrs']={'0xc11':event['sequence']}
        initial=payload(event['initial'])
        expected_output=payload(event['output'])
        offset,size=event['output_offset'],event['output_bytes']
        if len(expected_output)!=size or not 0<=offset<=len(initial)-size:
            raise ValueError('expected output extent differs')
        expected=initial[:offset]+expected_output+initial[offset+size:]
        if sha(expected)!=event['expected_arena_sha256']:
            raise ValueError('expected full arena reconstruction differs')
        functional=unit._functional
        def initialize_csr(program,state,max_instructions):
            state.write_csrf(0xc11,event['sequence'])
            return functional(program,state,max_instructions)
        unit._functional=initialize_csr
        fp8_stats=fp8_observation(event)
        observation=contextlib.nullcontext()
        if fp8_stats is not None:
            from smol_capture.stats import observe_fp8
            observation=observe_fp8(unit,fp8_stats['scale_exponent'],size)
        try:
            with observation as stats:
                result=unit.execute(source,initial,offset,size,timed=False)
            if fp8_stats is not None:
                actual_stats=stats.report(result.output)
                if actual_stats!=fp8_stats:
                    raise ValueError('original FP8 preclip and emitted statistics differ')
                record['fp8_stats']=actual_stats
        finally:
            unit._functional=functional
        if result.program_sha256!=sha(raw_program): raise ValueError('assembled program differs from capture')
        if result.arena!=expected or result.output!=expected_output:
            if result.output!=expected_output:
                (out.parent/f'launch-{event["index"]:04d}-actual-output.bin').write_bytes(result.output)
            raise ValueError('original RTL complete arena or output differs')
        requested=event['requested_dma']
        if any(result.observed.get(key)!=requested[key] for key in ('load_bytes','store_bytes')):
            raise ValueError('original requested DMA differs')
        require(unit.model_binding()==binding,'model binding changed during original execution')
        require(bindings(args.model_root,__file__,args.harness)==current,'gate source bindings changed during execution')
        record.update(status='PASS',full_arena_exact=True,output_exact=True,execution=result.metadata(),
            initial_sha256=sha(initial),expected_arena_sha256=sha(expected),actual_arena_sha256=sha(result.arena),
            program_sha256=sha(raw_program),expected_output_sha256=sha(expected_output))
    except BaseException as error:
        record.update(status='FAILED',error=dict(type=type(error).__name__,message=str(error)))
    record['elapsed_seconds']=time.perf_counter()-started
    temporary=out.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(out)
    print(json.dumps({key:record[key] for key in ('index','kind','status','elapsed_seconds')}),flush=True)
    return record['status']=='PASS'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=safe)
    parser.add_argument('--out', type=safe, required=True)
    parser.add_argument('--model-root', type=safe, required=True)
    parser.add_argument('--harness', type=safe, default=Path(__file__).parent / 'smol_capture/model.py')
    parser.add_argument('--python', type=interpreter, default=Path(sys.executable))
    parser.add_argument('--workers', type=int, default=40)
    parser.add_argument('--expected-launches', type=int, default=2915, help='explicit complete inventory size; 2915 for full attention capture')
    parser.add_argument('--event', type=safe)
    parser.add_argument('--worker-out', type=safe)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.event:
        require(args.workers > 0, 'workers must be positive')
        bindings(args.model_root, __file__, args.harness)
        require(args.worker_out is not None, '--event requires --worker-out')
        return 0 if worker(args.event, args.worker_out, args) else 1
    args.out.mkdir(parents=True, exist_ok=True)
    # Never leave a prior PASS authoritative while a resumed invocation runs.
    summary_path = args.out / 'gate.json'
    summary_path.write_text(json.dumps(dict(status='INCOMPLETE', authority='progress only')) + '\n')
    try:
        require(args.workers > 0, 'workers must be positive')
        require(args.ledger is not None, '--ledger is required')
        current = bindings(args.model_root, __file__, args.harness)
    except Exception as error:
        report = dict(status='INCOMPLETE', authority='entry validation failed',
                      failures=[dict(type=type(error).__name__, error=str(error))])
        summary_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        return 1
    scheduled, failures = {}, []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', ATLAS_NPU_MODEL_ROOT=str(args.model_root), ATLAS_NPU_MODEL_PIN=PIN)
    env.pop('NPU_MODEL_RTL_NATIVE_MXU0', None)
    env.pop('NPU_MODEL_RTL_NATIVE_BUILD_DIR', None)

    def paths(index):
        return (args.out / f'event-{index:04d}.json', args.out / f'launch-{index:04d}.json',
                args.out / f'worker-{index:04d}.json')

    def execute_event(event):
        frozen, target, completion = paths(event['index'])
        command = [str(args.python), '-B', str(Path(__file__).absolute()), '--event', str(frozen),
                   '--worker-out', str(target), '--out', str(args.out), '--model-root', str(args.model_root),
                   '--harness', str(args.harness), '--python', str(args.python)]
        with target.with_suffix('.log').open('w') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record = dict(index=event['index'], command=command, returncode=result.returncode,
                      event_sha256=sha(frozen.read_bytes()),
                      receipt_sha256=sha(target.read_bytes()) if target.exists() else None)
        completion.write_text(json.dumps(record, indent=2) + '\n')
        require(result.returncode == 0, 'worker failed: ' + str(event['index']))
        validate_completed(event, frozen.read_bytes(), target, completion,
                           bindings(args.model_root, __file__, args.harness))
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            rows = []
            if args.ledger.exists():
                rows = [json.loads(line) for line in args.ledger.read_text().splitlines(keepends=True)
                        if line.strip() and line.endswith('\n')]
            for index, event in enumerate(rows):
                require(event['index'] == index and event['sequence'] == index + 1, 'noncontiguous device ledger')
                if index in scheduled:
                    continue
                frozen, target, completion = paths(index)
                raw = (json.dumps(event, indent=2) + '\n').encode()
                try:
                    require(frozen.read_bytes() == raw, 'frozen event changed')
                    validate_completed(event, raw, target, completion,
                                       bindings(args.model_root, __file__, args.harness))
                    scheduled[index] = None
                    continue
                except (OSError, ValueError, KeyError, TypeError):
                    # Remove stale PASS before dispatch; failed workers cannot leave it reusable.
                    target.unlink(missing_ok=True)
                    completion.unlink(missing_ok=True)
                    frozen.write_bytes(raw)
                scheduled[index] = pool.submit(execute_event, event)
            for index, future in list(scheduled.items()):
                if future is not None and future.done():
                    try:
                        future.result()
                    except Exception as error:
                        failures.append(dict(index=index, error=str(error)))
                    scheduled[index] = None
            manifest_path = safe(args.ledger.parent / 'manifest.json')
            try:
                manifest = json.loads(manifest_path.read_bytes())
            except (OSError, ValueError):
                manifest = {}
            pending = sum(future is not None for future in scheduled.values())
            complete = manifest.get('status') == 'COMPLETE_REFERENCE_GENERATION'
            if (complete or manifest.get('status') == 'INCOMPLETE' or not args.watch) and not pending:
                break
            time.sleep(5)
    # The same strict validator is used again after every worker has exited.
    valid = []
    ledger_raw = args.ledger.read_bytes()
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    complete = manifest.get('status') == 'COMPLETE_REFERENCE_GENERATION'
    final_rows = [json.loads(line) for line in ledger_raw.decode().splitlines() if line.strip()]
    require(final_rows == rows, 'ledger changed during final aggregation')
    require(manifest.get('device_events') == rows, 'manifest and ledger differ')
    validate_projection_observations(rows,manifest)
    for event in rows:
        frozen, target, completion = paths(event['index'])
        try:
            validate_completed(event, frozen.read_bytes(), target, completion,
                               bindings(args.model_root, __file__, args.harness))
            valid.append(event['index'])
        except Exception as error:
            failures.append(dict(index=event['index'], error=str(error)))
    require(bindings(args.model_root, __file__, args.harness) == current, 'source bindings changed during backfill')
    require(ledger_raw == args.ledger.read_bytes() and manifest_raw == manifest_path.read_bytes(),
            'capture changed during final aggregation')
    counts = dict(projection=0, attention=0)
    for event in rows:
        require(event['kind'] in counts, 'unknown launch kind')
        require(event[event['kind'] + '_index'] == counts[event['kind']], 'noncontiguous typed launch inventory')
        counts[event['kind']] += 1
    if args.expected_launches == 2915:
        require(counts == dict(projection=2715, attention=200), 'full launch inventory differs')
        require(len(manifest['projection_reference']['calls']) == 1421 and
                len(manifest['projection_reference']['kernels']) == 2715 and
                len(manifest['attention_reference']['kernels']) == 200, 'full reference inventory differs')
    passed = (complete and manifest.get('errors') == [] and len(valid) == len(rows) == args.expected_launches
              and args.expected_launches > 0 and not failures)
    report = dict(status='PASS_AUDITED_ORIGINAL_ALL_LAUNCHES' if passed else 'INCOMPLETE',
                  authority='strict current-run backfill audit; archived M2 gate is separate',
                  launches=len(rows), expected_launches=args.expected_launches, typed_launches=counts,
                  validated=len(valid), failures=failures, model=current.model,
                  runner_sha256=current.runner_sha256, harness_sha256=current.harness_sha256,
                  source_binding_schema=current.source_binding_schema, helper_source_sha256=current.helper_source_sha256,
                  ledger_sha256=sha(ledger_raw), manifest_sha256=sha(manifest_raw), all_full_arenas_exact=passed)
    summary_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
