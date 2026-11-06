"""Qualify captured Multi-Task DiT launch boundaries without running a connected policy.

Preparation reads inventory.py's authenticated boundaries.json. Projection panels
retain full K, split M/N at 128, and pad only the tail to 32. Each physical kind
uses real captured operands at its first matching panel. Counts include all such
panels in the full logical geometry and all model occurrences. Fidelity therefore
qualifies that representative boundary, not every layer, panel, or diffusion step.
Independent projection goldens cover the entire physical output including padding.

Run --prepare first with the 47b8599 environment, then --run --mode rtl and
--run --mode official (the latter imports the unmodified 74982e7 performance API
in separate processes). --report joins those receipts and records missing gates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

RTL_PIN = '47b8599d8ffbe5ddd96059ffe2b5be2427a410f5'
PERF_PIN = '74982e7c68bd28ba6505b4df37290f198102ccce'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def checked(path):
    path = Path(path).absolute()
    if any(any(word in part.lower() for word in ('ham' + 'mer', 'vl' + 'si')) for part in path.parts):
        raise ValueError('excluded path')
    if any(item.is_symlink() for item in (*path.parents, path)):
        raise ValueError('qualification requires paths without symbolic links')
    return path


def revision(root, expected):
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise ValueError(f'model revision {actual}, expected {expected}')
    return actual


def authenticate(root, paths):
    """Check explicitly selected files against their HEAD blobs without walking history."""
    hashes = {}
    for name in sorted(set(paths)):
        path = checked(root / name)
        raw = path.read_bytes()
        original = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{name}'])
        if raw != original:
            raise ValueError('modified model dependency: ' + name)
        hashes[name] = sha(raw)
    return hashes


def model_binding(root):
    paths = []
    for name, module in list(sys.modules.items()):
        if name in ('npu_model','npu_performance') or name.startswith(('npu_model.', 'npu_performance.')):
            source = getattr(module, '__file__', None)
            if source:
                path = checked(source)
                if not path.is_relative_to(root):
                    raise ValueError('model import outside selected root: ' + name)
                paths.append(str(path.relative_to(root)))
    return authenticate(root, paths)


def imports(args, root):
    sys.path[:0] = [str(args.library), str(args.library / 'coverage'), str(root)]
    import torch
    torch.set_num_threads(1)
    import atlas_kernels_path  # noqa: F401
    import bench_model
    import kernel_bench
    return bench_model, kernel_bench, torch


def projection_kinds(record):
    """Return each distinct padded/tail physical panel and its exact multiplicity."""
    kinds = {}
    m, k, n = (int(record[key]) for key in ('m', 'k', 'n'))
    if min(m, k, n) <= 0 or int(record['count']) <= 0:
        raise ValueError('projection dimensions and count must be positive')
    pad = lambda x: (x + 31) // 32 * 32
    for r in range(0, m, 128):
        for c in range(0, n, 128):
            rows, width = min(128, m-r), min(128, n-c)
            key = rows, width
            if key not in kinds:
                kinds[key] = dict(m=pad(rows), k=pad(k), n=pad(width), rows=rows,
                                  logical_k=k, logical_n=width, row_start=r, column_start=c,
                                  count=0)
            kinds[key]['count'] += int(record['count'])
    return list(kinds.values())


def load_artifact(record, inventory_path):
    import numpy as np
    artifact = record['artifact']
    path = checked(artifact['path'])
    if not path.exists():
        path = checked(inventory_path.parent / artifact['path'])
    raw = path.read_bytes()
    if sha(raw) != artifact['sha256'] or len(raw) != artifact['bytes']:
        raise ValueError('captured artifact identity differs: ' + str(path))
    return np.load(io.BytesIO(raw), allow_pickle=False)


def receipt_matches(receipt, spec_path):
    return (receipt.get('status')=='PASS' and receipt.get('spec_sha256')==sha(spec_path.read_bytes())
            and receipt.get('runner_sha256')==sha(Path(__file__).read_bytes()))


def prepare(args):
    import numpy as np
    revision(args.model_root, RTL_PIN)
    bench, _, _ = imports(args, args.model_root)
    inventory = json.loads(args.inventory.read_text())
    args.out.mkdir(parents=True, exist_ok=args.resume)
    previous=json.loads((args.out/'campaign.json').read_text()) if args.resume else {}
    if previous and previous['inventory_sha256']!=sha(args.inventory.read_bytes()):
        raise ValueError('resume inventory differs')
    specs, pending = previous.get('specs',[]), []
    existing={spec['label'] for spec in specs}
    records = [(r, 'projection_gemm') for r in inventory['projections']]
    records += [(r, 'attention') for r in inventory.get('attention', inventory.get('attentions', []))]
    for ordinal, (record, family) in enumerate(records):
        data = load_artifact(record, args.inventory)
        kinds = projection_kinds(record) if family == 'projection_gemm' else [dict(record, count=int(record['count']))]
        for panel_index, kind in enumerate(kinds):
            label = f'{ordinal:03d}-{panel_index:02d}-{family}'
            if label in existing:continue
            folder = args.out / label
            folder.mkdir(exist_ok=args.resume)
            try:
                if family == 'projection_gemm':
                    p = {k:kind[k] for k in ('m','k','n','rows','logical_k','logical_n')}
                    a = np.zeros((p['m'],p['k']),dtype=np.float32)
                    w = np.zeros((p['k'],p['n']),dtype=np.float32)
                    r,c = kind['row_start'],kind['column_start']
                    a[:p['rows'],:p['logical_k']] = data['A'][r:r+p['rows']]
                    w[:p['logical_k'],:p['logical_n']] = data['W'][:,c:c+p['logical_n']]
                    expected = data['expected_without_bias'][r:r+p['rows'],c:c+p['logical_n']]
                    built = bench.build(dict(family=family,key=label,params=p), folder,args.model_root,projection_operands=(a,w))
                else:
                    p = {key:record[key] for key in ('rows','keys','heads','kv_heads','head_dim','mask')}
                    if args.attention_sync_dma:p['variant']='synchronous-dma'
                    batch=int(record.get('batch',1))
                    p.update(heads=p['heads']*batch,kv_heads=p['kv_heads']*batch,output_fp8=False)
                    if p['mask']=='causal':p['query_start']=p['keys']-p['rows']
                    operands=tuple(np.ascontiguousarray(data[key].reshape(-1,data[key].shape[-2],data[key].shape[-1]),dtype=np.float32) for key in ('Q','K','V'))
                    expected = data['original_output'].reshape(p['heads'],p['rows'],p['head_dim'])
                    mask_values = None
                    if p['mask']=='arbitrary':
                        mask=data['mask'].reshape(-1,p['rows'],p['keys'])
                        if not np.array_equal(mask,np.broadcast_to(mask[:1],mask.shape)):
                            raise ValueError('per-head masks require separate launches')
                        mask=mask[0]
                        if mask.dtype==np.bool_:
                            mask_values=mask
                        else:
                            if not np.all((mask==0)|(mask<=-1e4)):
                                raise ValueError('nonboolean attention bias has no matching kernel route')
                            mask_values=mask==0
                    built = bench.build(dict(family=family,key=label,params=p),folder,args.model_root,attention_operands=operands,attention_mask_values=mask_values)
                identity=bench.program_identity(built,args.atlas_bin,'verified')
                golden=built.golden() if built.golden else None
                (folder/'program.S').write_text(built.source)
                (folder/'arena.bin').write_bytes(built.arena)
                if golden is not None:(folder/'golden.bin').write_bytes(golden)
                np.save(folder/'reference.npy',expected,allow_pickle=False)
                spec=dict(label=label,family=family,params=p,count=kind['count'],logical_source=record,
                          representative_panel=kind,identity=identity,output_offset=built.output_offset,
                          output_bytes=built.output_bytes,variant=built.variant,calibration=built.meta,
                          golden_sha256=None if golden is None else sha(golden),reference_sha256=sha((folder/'reference.npy').read_bytes()),
                          inventory_sha256=sha(args.inventory.read_bytes()),folder=str(folder),
                          library=str(args.library),model_root=str(args.model_root),perf_root=str(args.perf_root),
                          max_cycles=args.max_cycles,max_instructions=args.max_instructions)
                save(folder/'spec.json',spec);specs.append(spec)
                print(label,p,'PREPARED',flush=True)
            except Exception as error:
                traceback.print_exc()
                pending.append(dict(label=label,family=family,source=record,status='PENDING',error=f'{type(error).__name__}: {error}'))
    save(args.out/'campaign.json',dict(schema='merlin.multi_task_dit.qualification.v1',inventory=str(args.inventory),inventory_sha256=sha(args.inventory.read_bytes()),specs=specs,pending=pending))
    return int(bool(pending))


def source_and_arena(spec):
    folder=checked(spec['folder']);source=(folder/'program.S').read_text();arena=(folder/'arena.bin').read_bytes()
    if sha(source.encode())!=spec['identity']['source_sha256'] or sha(arena)!=spec['identity']['arena_sha256']:
        raise ValueError('staged program or arena differs')
    golden=(folder/'golden.bin').read_bytes() if spec['golden_sha256'] else None
    if golden is not None and sha(golden)!=spec['golden_sha256']:raise ValueError('golden differs')
    return folder,source,arena,golden


def worker(args):
    started=time.monotonic();spec=json.loads(args.spec.read_text());result=dict(status='PENDING',spec_sha256=sha(args.spec.read_bytes()),runner_sha256=sha(Path(__file__).read_bytes()))
    try:
        root=checked(spec['model_root'] if args.mode=='rtl' else spec['perf_root'])
        revision(root,RTL_PIN if args.mode=='rtl' else PERF_PIN)
        args.library=checked(spec['library'])
        bench,kernel_bench,torch=imports(args,root)
        import numpy as np
        folder,source,arena,golden=source_and_arena(spec)
        if args.mode=='rtl':
            from atlas_execution import Executor
            functional=Executor(root).execute(source,arena,spec['output_offset'],spec['output_bytes'],max_instructions=spec['max_instructions'])
            if golden is not None and functional.output!=golden:raise AssertionError('independent full-extent projection golden differs')
            cycles,timed=kernel_bench.timed_cycles(root,source,arena,spec['max_cycles'])
            end=spec['output_offset']+spec['output_bytes']
            if timed[:spec['output_offset']]!=arena[:spec['output_offset']] or timed[end:]!=arena[end:]:raise AssertionError('timed guard changed')
            if timed[spec['output_offset']:end]!=functional.output:raise AssertionError('timed/functional output differs')
            (folder/'rtl-output.bin').write_bytes(functional.output)
            if spec['family']=='projection_gemm':
                p=spec['params'];actual=kernel_bench.bf16_values(kernel_bench.unpack_bf16_tiles(functional.output,p['m'],p['n']))[:p['rows'],:p['logical_n']]
            else:
                from atlas_attention_v2_program import AttentionV2Config,unpack_output_v2
                from atlas_attention_wide_program import AttentionWideConfig,unpack_output_wide
                p=spec['params'];typ,unpack=(AttentionV2Config,unpack_output_v2) if p['head_dim']==64 else (AttentionWideConfig,unpack_output_wide)
                config=typ(p['rows'],p['keys'],p['heads'],p['kv_heads'],head_dim=p['head_dim'],mask=p['mask'],query_start=p.get('query_start'),output_fp8=False,**spec['calibration'])
                actual=kernel_bench.bf16_values(unpack(config,functional.output))
            reference=(folder/'reference.npy').read_bytes()
            if sha(reference)!=spec['reference_sha256']:raise ValueError('reference changed')
            result.update(status='PASS',rtl_cycles=cycles,independent_golden_exact=None if golden is None else True,full_extent_exact=None if golden is None else True,timed_functional_exact=True,guards_exact=True,output_sha256=sha(functional.output),execution=functional.metadata(),fidelity=kernel_bench.fidelity(actual,np.load(io.BytesIO(reference),allow_pickle=False)))
        else:
            import dataclasses
            import npu_model.hardware  # noqa: F401
            from npu_model.configs.hardware.default import DefaultHardwareConfig
            from npu_model.util.converter import input_to_program
            from npu_model.simulation import Simulation
            from npu_performance import speed_of_light as sol,roofline as roof
            program=input_to_program(io.StringIO(source));program.dram_base=0x90000000
            program.memory_regions=[(0,torch.frombuffer(bytearray(arena),dtype=torch.uint8))]
            words=b''.join(int(w).to_bytes(4,'little') for w in program.assemble())
            if sha(words)!=spec['identity']['words_sha256']:raise ValueError('official assembler differs')
            config=DefaultHardwareConfig()
            config.arch_state_config=dataclasses.replace(config.arch_state_config,numerics='rtl',randomize_init=True,init_seed=42)
            sol.configure_memory(config,'curve')
            rtl=json.loads((folder/'rtl.json').read_text())
            if not receipt_matches(rtl,args.spec):raise ValueError('RTL receipt is stale or incomplete')
            expected=(folder/'rtl-output.bin').read_bytes()
            if sha(expected)!=rtl['output_sha256']:raise ValueError('RTL output identity differs')
            validations=[]
            class CheckedSimulation(Simulation):
                def close(self):
                    try:
                        if self.core is not None and self.core.is_finished():
                            if self.get_stats().runtime_errors:raise AssertionError('official runtime errors')
                            actual=bytes(self.core.arch_state.read_dram(program.dram_base,len(arena)).tolist())
                            end=spec['output_offset']+spec['output_bytes']
                            if actual[:spec['output_offset']]!=arena[:spec['output_offset']] or actual[end:]!=arena[end:]:raise AssertionError('official guard differs')
                            if actual[spec['output_offset']:end]!=expected:raise AssertionError('official/RTL output differs')
                            validations.append(True)
                    finally:super().close()
            before=model_binding(root)
            rtl_sources=authenticate(checked(spec['model_root']),[name for name in before if name.startswith('npu_model/')])
            if any(before[name]!=digest for name,digest in rtl_sources.items()):
                raise ValueError('official execution-model source differs from pinned RTL model')
            # Both untouched public APIs run their own complete cycle simulation.
            original=sol.Simulation
            sol.Simulation=CheckedSimulation
            try:report=sol.speed_of_light(program,config,max_cycles=spec['max_cycles'])
            finally:sol.Simulation=original
            original=roof.Simulation if hasattr(roof,'Simulation') else None
            # roof.measure calls its imported execute, whose Simulation global is sol's.
            sol.Simulation=CheckedSimulation
            try:point=roof.measure(spec['label'],program,config,max_cycles=spec['max_cycles'])
            finally:sol.Simulation=Simulation
            if len(validations)!=2:raise AssertionError('official output validation did not run twice')
            if point.cycles!=report.measured or point.mxu_flops!=report.flops or point.dram_bytes!=report.dram_bytes:raise AssertionError('official API execution counters differ')
            if rtl.get('status')!='PASS' or rtl['rtl_cycles']!=report.measured:raise AssertionError('official/RTL cycles differ')
            after=model_binding(root)
            if any(after.get(k)!=v for k,v in before.items()):raise ValueError('official model bookend differs')
            result.update(status='PASS',speed_of_light=asdict(report),bound=asdict(report.bound),roofline=asdict(point),machine=asdict(roof.ceilings(config)),fraction=report.bound.cycles/report.measured,memory=sol.describe_memory(config),numerics=config.arch_state_config.numerics,official_execution_runs=2,official_rtl_cycles_equal=True,official_rtl_output_equal=True,guards_exact=True,model_source_sha256=after,model_revision=PERF_PIN,imported_execution_sources_equal_rtl_pin=True)
    except Exception as error:
        traceback.print_exc();result.update(status='ERROR',error=f'{type(error).__name__}: {error}')
    result['elapsed_seconds']=time.monotonic()-started
    save(args.spec.parent/f'{args.mode}.json',result)
    print(spec['label'],args.mode,result['status'],flush=True)
    return int(result['status']!='PASS')


def run(args):
    campaign=json.loads((args.out/'campaign.json').read_text())
    specs=campaign['specs']
    def task(spec):
        folder=Path(spec['folder'])
        if args.only and spec['label'] not in args.only:return None
        old=folder/f'{args.mode}.json'
        if old.exists() and not args.force:
            result=json.loads(old.read_text())
            if receipt_matches(result,folder/'spec.json'):return result
        if shutil.disk_usage('/tmp').free<20*1024**3:raise RuntimeError('/tmp below 20 GiB free')
        with (folder/f'{args.mode}.log').open('w') as log:
            try:
                process=subprocess.run(['nice','-n','10',sys.executable,'-B',str(Path(__file__).absolute()),'--worker','--mode',args.mode,'--spec',str(folder/'spec.json')],stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
                if process.returncode and not old.exists():save(old,dict(status='ERROR',error=f'worker exit {process.returncode}'))
            except subprocess.TimeoutExpired:save(old,dict(status='PENDING',error=f'wall timeout {args.timeout} seconds'))
        return json.loads(old.read_text())
    failures=[]
    for start in range(0,len(specs),args.workers):
        load=os.getloadavg()[0]
        workers=1 if load>112 else args.workers
        print(f'batch {start}: load={load:.2f}, workers={workers}',flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in as_completed([pool.submit(task,s) for s in specs[start:start+args.workers]]):
                value=result.result()
                if value and value['status']!='PASS':failures.append(value)
    return int(bool(failures))


def report(args):
    campaign=json.loads((args.out/'campaign.json').read_text());rows=[]
    for spec in campaign['specs']:
        folder=Path(spec['folder']);row=dict(label=spec['label'],family=spec['family'],params=spec['params'],count=spec['count'],logical_source=spec['logical_source'],identity=spec['identity'])
        spec_path=folder/'spec.json'
        spec_matches=spec_path.exists() and json.loads(spec_path.read_text())==spec
        for mode in ('rtl','official'):
            path=folder/f'{mode}.json'
            receipt=json.loads(path.read_text()) if path.exists() else dict(status='PENDING')
            if receipt.get('status')=='PASS' and (not spec_matches or not receipt_matches(receipt,spec_path)):
                receipt=dict(status='PENDING',error='stale receipt spec or runner identity')
            if mode=='rtl' and receipt.get('status')=='PASS':
                try:
                    source_and_arena(spec)
                    if sha((folder/'rtl-output.bin').read_bytes())!=receipt['output_sha256']:
                        raise ValueError('RTL output identity differs')
                except (ValueError,OSError) as error:
                    receipt=dict(status='PENDING',error=str(error))
            row[mode]=receipt
        row['status']='PASS' if all(row[mode]['status']=='PASS' for mode in ('rtl','official')) else 'PENDING'
        rows.append(row)
    qualified=[r for r in rows if r['status']=='PASS']
    cycles=sum(r['count']*r['rtl']['rtl_cycles'] for r in qualified)
    sol=sum(r['count']*r['official']['bound']['cycles'] for r in qualified)
    total=dict(schema='merlin.multi_task_dit.qualification_report.v1',rows=rows,pending=campaign['pending'],distinct_kinds=len(rows),qualified_kinds=len(qualified),count_weighted_cycles=cycles,count_weighted_sol_cycles=sol,sol_fraction=None if not cycles else sol/cycles,count_weighted_headroom=cycles-sol,scope='isolated count-weighted launch estimates; representative real captured boundaries; no connected policy execution',remaining_gates=['connected quantized policy execution','trained task accuracy','hardware RTL execution','host dispatch and transfer timing'])
    save(args.out/'qualification.json',total)
    print(json.dumps({k:v for k,v in total.items() if k not in ('rows','pending')},indent=2))
    return int(len(qualified)!=len(rows) or bool(campaign['pending']))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    action=p.add_mutually_exclusive_group(required=True)
    for name in ('prepare','run','worker','report'):action.add_argument('--'+name,action='store_true')
    p.add_argument('--inventory',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--library',type=Path);p.add_argument('--model-root',type=Path);p.add_argument('--perf-root',type=Path);p.add_argument('--atlas-bin',type=Path)
    p.add_argument('--mode',choices=('rtl','official'),default='rtl');p.add_argument('--spec',type=Path)
    p.add_argument('--resume',action='store_true',help='prepare missing kinds in the identical existing campaign')
    p.add_argument('--attention-sync-dma',action='store_true',help='opt-in attention schedule with a wait after every DMA')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--timeout',type=int,default=14400);p.add_argument('--only',nargs='*');p.add_argument('--force',action='store_true')
    p.add_argument('--max-cycles',type=int,default=100_000_000);p.add_argument('--max-instructions',type=int,default=50_000_000)
    args=p.parse_args()
    if not 1<=args.workers<=8:p.error('--workers must be between 1 and 8')
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,checked(value))
    if args.prepare and any(getattr(args,k) is None for k in ('inventory','out','library','model_root','perf_root','atlas_bin')):p.error('--prepare requires --inventory --out --library --model-root --perf-root --atlas-bin')
    if args.worker and args.spec is None:p.error('--worker requires --spec')
    if (args.run or args.report) and args.out is None:p.error('--run and --report require --out')
    return prepare(args) if args.prepare else worker(args) if args.worker else run(args) if args.run else report(args)


if __name__=='__main__':raise SystemExit(main())
