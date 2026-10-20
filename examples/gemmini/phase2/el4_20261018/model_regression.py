"""Final compiler regression: reuse saved host objects, replace only device schedules.

This checks against the previously executed compiled chain, not a quantized
reference. Its Spike counters have no target timing authority.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import common as C
C.configure()
import numpy as np
from merlin.common.cpu_pool import shared_compute_pool
from merlin.llvmlower import c_runtime, device_build
from merlin.model_check import chain as CH, full_build as FB, models as MODELS
from merlin.model_check.full import carried
from merlin.targetgen.contract.compile import llvm_mlir_to_object
from merlin.targetgen.target_registry import observed_contract

HANDOFF=C.P1/'.tmp/handoff/functional-phase1-baseline-v1'


def arrays(path):
    with np.load(path) as archive:
        return {k:np.asarray(archive[k]) for k in archive.files}


def prepare(name, package, root):
    package_sha=C.tree_hash(package)[0]
    model=MODELS.load(HANDOFF/'models.yaml').model(name)
    records={}
    for program in model.programs:
        prefix='tiny32' if name=='tiny_llama' else 'smolvla'
        status_path=HANDOFF/f'{prefix}-{program.name}-compile-v2/status.json'
        status=json.loads(status_path.read_text())
        assert status['stage']=='finished' and status['compilation_status']=='compiled'
        receipt=status['receipt']
        assert receipt['inputs']['readback']=='full'
        assert status['candidate_tree']['sha256']==C.initialize()['baseline_sha256']
        original=Path(receipt['output']['elf']).parent
        assert FB.sha256(original/'model.elf')==receipt['output']['elf_sha256']
        build=root/'programs'/program.name/'build'
        record_path=build.parent/'build.json'
        if record_path.exists():
            record=json.loads(record_path.read_text())
            assert record['candidate_sha256']==package_sha
            assert all(C.file_hash(x['object'])==x['object_sha256'] for x in record['device_objects'])
            records[program.name]=record
            continue
        if build.exists():
            assert build.is_relative_to(C.OUT/'model-regression') and not build.is_symlink()
            shutil.rmtree(build)  # incomplete task-owned adapter products only
        build.mkdir(parents=True,exist_ok=True)
        for item in original.iterdir():
            if item.name not in {'device','compilation_recipe.json','model.elf'}:
                (build/item.name).symlink_to(item.resolve())
        device=build/'device'
        device.mkdir()
        sources=sorted((original/'device').glob('*.lowering/input.interface.mlir'))
        replaced={source.parent.name.removesuffix('.lowering')+'.o' for source in sources}
        for item in (original/'device').iterdir() if (original/'device').is_dir() else []:
            if item.name not in replaced:
                (device/item.name).symlink_to(item.resolve())
        objects=[]
        for source in sources:
            stem=source.parent.name.removesuffix('.lowering')
            work=build.parent/'candidate-lowering'/stem
            work.mkdir(parents=True,exist_ok=True)
            with shared_compute_pool(timeout_s=120):
                result=subprocess.run([str(C.PYTHON),str(package/'compiler.py'),
                    '--convert-iface-to-gemmini',f'--emit-command-buffer={work}/commands.json',
                    '--emit-target-artifact',str(source)],capture_output=True,text=True,
                    check=True,timeout=120,env=C.environment())
            (work/'compiler.stderr').write_text(result.stderr)
            (work/'emitted.llvm.mlir').write_text(result.stdout)
            original_cb=json.loads((source.parent/'command_buffer.json').read_text())
            candidate_cb=json.loads((work/'commands.json').read_text())
            assert C.logical_command_buffer(original_cb)==C.logical_command_buffer(candidate_cb),stem
            raw=llvm_mlir_to_object(result.stdout,work,target='gemmini')
            symbols=subprocess.check_output([device_build._nm(),'--defined-only',str(original/'device'/f'{stem}.o')],text=True)
            strong=[line.split()[-1] for line in symbols.splitlines() if len(line.split())==3 and line.split()[1]=='T']
            assert len(strong)==1,(stem,strong)
            obj=device/f'{stem}.o'
            with shared_compute_pool(timeout_s=120):
                subprocess.run([device_build._objcopy(),f'--redefine-sym=gemmini_kernel={strong[0]}',str(raw),str(obj)],check=True)
            objects.append({'source':str(source),'source_sha256':C.file_hash(source),
                'logical_command_buffer_sha256':C.digest(C.logical_command_buffer(candidate_cb)),
                'old_object_sha256':C.file_hash(original/'device'/f'{stem}.o'),
                'object':str(obj),'object_sha256':C.file_hash(obj),'entry_symbol':strong[0]})
        # Rebase build products only. Host sources/objects/weights stay read-only.
        def rebase(value):
            if isinstance(value,str):
                return str(build)+value[len(str(original)):] if value.startswith(str(original)+'/') else value
            if isinstance(value,list):return [rebase(x) for x in value]
            if isinstance(value,dict):return {k:rebase(v) for k,v in value.items()}
            return value
        original_recipe=original/'compilation_recipe.json'
        recipe=rebase(json.loads(original_recipe.read_text()))
        C.save(build/'compilation_recipe.json',recipe)
        weights=build/'cgen/weights.bin'
        record={'build_dir':str(build),'elf':str(build/'model.elf'),
            'build_hash':receipt['output']['build_hash'],
            'build_hash_scope':'original host/readback protocol header; candidate identity is separate',
            'weights_bin_bytes':weights.stat().st_size,'weights_bin_sha256':FB.sha256(weights),
            'candidate_sha256':package_sha,'original_elf':str(original/'model.elf'),
            'original_elf_sha256':receipt['output']['elf_sha256'],
            'original_recipe_sha256':C.file_hash(original_recipe),'recipe_sha256':C.file_hash(build/'compilation_recipe.json'),
            'device_objects':objects}
        assert C.tree_hash(package)[0]==package_sha
        C.save(record_path,record)
        records[program.name]=record
        print(json.dumps({'prepared':program.name,'objects':len(objects),'candidate_sha256':package_sha}),flush=True)
    return records


def main():
    if (C.EVIDENCE/'manual-stop.json').exists():
        raise SystemExit('Manual campaign stopped by user workflow correction; no independent model job admitted.')
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['tiny_llama','smolvla'],required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    name=args.model
    package=args.candidate.resolve()
    package_sha=C.tree_hash(package)[0]
    root=C.OUT/'model-regression'/package_sha/name
    root.mkdir(parents=True,exist_ok=True)
    if (root/'status.json').exists():
        previous=json.loads((root/'status.json').read_text())
        assert not previous.get('steps'),'refuse to overwrite a previously executed chain'
        C.save(root/f'status-before-{time.time_ns()}.json',previous)
    state={'schema':'phase2.compiled_chain_regression.v1','model':name,'candidate_sha256':package_sha,
        'comparison':'bitwise against saved compiled baseline outputs, not quantized reference equivalence',
        'timing_authority':'NONE: functional untuned Spike','all_steps_completed':False,'steps':[]}
    started=time.monotonic()
    def save():
        state['elapsed_s']=time.monotonic()-started
        C.save(root/'status.json',state)
    try:
        with C.stage('model_candidate_prepare',model=name,candidate=package_sha):
            records=prepare(name,package,root)
        state.update(builds=records,status='candidate_programs_prepared',driver_sha256=C.file_hash(__file__),
            facts_sha256=C.file_hash(C.environment()['MERLIN_RTL_FACTS']),
            contract_sha256=C.file_hash(C.environment()['MERLIN_TARGET_CONTRACT']),
            spike_sha256=C.file_hash(Path(C.environment()['MERLIN_FIRE_ALIGNED_SPIKE_RUNTIME'])/'bin/spike'),
            extension_sha256=C.file_hash(Path(C.environment()['MERLIN_FIRE_ALIGNED_SPIKE_RUNTIME'])/'lib/libgemmini.so'))
        save()
        if args.prepare_only:return
        # Measured prior execution cost plus preparation/link margin; no deadline extension.
        estimate=3900 if name=='tiny_llama' else 3100
        assert C.remaining(final=True)>estimate,'insufficient campaign time for complete model chain'
        assert shutil.disk_usage(root).free>6*1024**3,'insufficient bounded relink space'
        baseline_root=HANDOFF/f'{name}-full-readback-spike-v1'
        baseline_status=json.loads((baseline_root/'status.json').read_text())
        assert baseline_status['all_steps_completed']
        model=MODELS.load(HANDOFF/'models.yaml').model(name)
        input_root=C.P1/f'private/gemmini/numerics/derived/{name}/visible/v00'
        input_path=next(p for p in [input_root/'inputs.npz',input_root/'observation.npz'] if p.is_file())
        initial=arrays(input_path)
        state.update(input_path=str(input_path),input_sha256=C.file_hash(input_path),
            baseline_status_sha256=C.file_hash(baseline_root/'status.json'),status='executing')
        capture=(C.P1/'private/gemmini/captures/functional-phase1-v1/tiny_llama/run/capture/stages'
            if name=='tiny_llama' else C.P1/'private/gemmini/captures/launch-v14-methodology/smolvla/run/capture/stages')
        chain=CH.ChainRun()
        contract=Path(C.environment()['MERLIN_TARGET_CONTRACT'])
        with observed_contract('gemmini',json.loads(contract.read_text()),source_path=contract):
            for program in model.programs:
                record=records[program.name]
                stage=capture/program.name
                specs=c_runtime._out_specs(stage/'model.mlir')
                chain.steps[program.name]=[]
                for step in range(program.steps):
                    state.update(active_program=program.name,active_step=step)
                    save()
                    with C.stage('model_final_verification',model=name,program=program.name,step=step,candidate=package_sha):
                        bound=CH.bind(program,model,initial,step,chain)
                        work=root/'steps'/f'{program.name}.{step}'
                        relink_started=time.monotonic()
                        elf=FB.relink(record,stage,bound,work)
                        elf_sha=FB.sha256(elf)
                        relink_s=time.monotonic()-relink_started
                        # Retain one representative image per program, no large environment copy.
                        if step==0:os.link(elf,Path(record['elf']))
                        try:
                            console,wall=FB.run('gemmini',elf,timeout_s=max(1,C.remaining(final=True)-120))
                            values,info=FB.read(console,stage,record['build_hash'])
                        finally:FB.discard(work)
                        np.savez(work/'outputs.npz',**{f'out{i}':v for i,v in enumerate(values)})
                        original_path=baseline_root/f'{program.name}.{step}'/'outputs.npz'
                        old=arrays(original_path)
                        assert set(old)=={f'out{i}' for i in range(len(values))}
                        mismatches=[i for i,v in enumerate(values) if v.dtype!=old[f'out{i}'].dtype
                            or v.shape!=old[f'out{i}'].shape or v.tobytes()!=old[f'out{i}'].tobytes()]
                        old_row=next(r for r in baseline_status['steps'] if r['program']==program.name and r['step']==step)
                        profile=info.get('group_profile')
                        required_profile=old_row.get('group_profile')
                        assert (profile is None)==(required_profile is None)
                        if profile is not None:
                            assert profile['calls']==required_profile['calls'] and profile['dropped']==0
                        row={'program':program.name,'step':step,'elf_sha256':elf_sha,'relink_s':relink_s,
                            'simulation_and_wait_s':wall,'all_outputs_bit_exact':not mismatches,
                            'mismatches':mismatches,'baseline_outputs_sha256':C.file_hash(original_path),
                            'outputs_sha256':C.file_hash(work/'outputs.npz'),'output_shapes':[list(v.shape) for v in values],
                            'group_profile':profile,'cycles_functional_uncalibrated':info['metrics'].get('cycles')}
                        state['steps'].append(row)
                        save()
                        print(json.dumps(row),flush=True)
                        assert not mismatches,'compiled-chain regression: '+str(mismatches)
                        chain.steps[program.name].append(carried(values,specs))
        state.update(status='complete_bitwise_compiled_chain_regression',all_steps_completed=True,
            program_counts=CH.summary(chain),quantized_reference_equivalence='Phase1 owner pending; not established by this check')
    except BaseException as error:
        state.update(status='refused',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise
    finally:save()


if __name__=='__main__':main()
