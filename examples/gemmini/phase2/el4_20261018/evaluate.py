"""Bounded paired public experiments using Merlin's existing compiler/linker/runtime."""
from __future__ import annotations

import argparse
import collections
import json
from math import prod
from pathlib import Path
import re
import subprocess
import sys
import time

import common as C
C.configure()
import numpy as np
from merlin.common.cpu_pool import shared_compute_pool
from merlin.runtime.backends.base import get_backend
from merlin.runtime.reference import reference_outputs
from merlin.targetgen.contract.compile import llvm_mlir_to_object, link_elf
from merlin_experiments.phase1.annotated_spike_evidence import review_elf
from merlin.targetgen.contract.harness_render import explicit_whole_program, logical_abi
from merlin.targetgen.contract.readback_policy import ReadbackPolicy, require_current_build_receipt
from merlin_experiments.phase1.feedback.native_memory_readback import NativeLogicalMemoryReadback

OBJDUMP = C.P1 / '.tmp/chipyard-fire-aligned/.conda-env/riscv-tools/bin/riscv64-unknown-elf-objdump'
MANIFEST_PATH = C.EVIDENCE/'capsule-manifest.json'
DRIVER_SHA=C.file_hash(__file__)
PROFILE='selected_engine_full_values_v2'


def signed_inputs(cb, seed):
    values = {}
    for index, (name, t) in enumerate(cb['tensors'].items()):
        if t.get('role') not in {'input', 'weight', 'bias'}:
            continue
        dtype = np.int8 if t['dtype'] == 'i8' else np.int32
        size = prod(t['shape'])
        # seed=0 preserves the stopped session's exact signed inputs.
        x = ((np.arange(size, dtype=np.int64)*5 + index*3 + 1 + seed*11) % 7 - 3)
        if t.get('role') == 'bias':
            x = ((np.arange(size, dtype=np.int64)*13 + seed*7) % 33 - 16)
        values[name] = x.astype(dtype).reshape(t['shape'])
    return values


def reference(cb, inputs):
    """Accelerate only plain contraction references; other semantics use existing interpreter."""
    if not all(c['opcode'] in {'RES_PACK', 'EVICT', 'MATMUL', 'MATMUL_RESIDENT', 'COMMIT'} for c in cb['commands']):
        return reference_outputs(cb, {k:v.tolist() for k,v in inputs.items()})
    if any('maxpool' in c.get('attributes', {}).get('epilogue', []) for c in cb['commands']):
        return reference_outputs(cb, {k:v.tolist() for k,v in inputs.items()})
    env = {k:v.astype(np.int64) for k,v in inputs.items()}
    outputs = {}
    for cmd in cb['commands']:
        op, o, at = cmd['opcode'], cmd.get('operands', {}), cmd.get('attributes', {})
        if op == 'RES_PACK':
            env[o['dst']] = env[o['src']]
        elif op in {'MATMUL', 'MATMUL_RESIDENT'}:
            env[o['dst']] = env[o['lhs']] @ env[o['rhs']]
        elif op == 'COMMIT':
            v = env[o['src']].copy()
            for stage in at.get('epilogue', []):
                if stage in {'bias', 'bias_add'}:
                    v += env[o.get('bias', at.get('bias'))]
                elif stage == 'acc_scale':
                    v = np.rint(v.astype(np.float32) * np.float32(at['acc_scale'])).astype(np.int64)
                elif stage == 'relu':
                    v = np.maximum(v, 0)
                else:
                    raise ValueError('reference refuses unimplemented stage '+stage)
            dtype = cb['tensors'][o['dst']]['dtype']
            if dtype == 'i8':
                v = np.clip(v, -128, 127)
            elif dtype != 'i32':
                raise ValueError('reference dtype '+dtype)
            env[o['dst']] = v
            outputs[o['dst']] = v
    # Cross-check the independent NumPy projection on affordable programs.
    costs = [prod(t['shape']) for t in cb['tensors'].values()]
    if sum(costs) < 6000:
        original = reference_outputs(cb, {k:v.tolist() for k,v in inputs.items()})
        assert set(original) == set(outputs)
        for key, value in original.items():
            assert np.array_equal(np.asarray(value).reshape(np.asarray(outputs[key]).shape), outputs[key]), key
    return outputs


def build(member, package, arm, seed, engine='spike'):
    if C.file_hash(__file__)!=DRIVER_SHA:
        raise ValueError('experiment driver changed after process start; restart at a job boundary')
    compiler_hash, _ = C.tree_hash(package)
    retained=C.OUT/'retained-compilers'/compiler_hash
    if not retained.exists():
        import shutil
        shutil.copytree(package,retained,ignore=shutil.ignore_patterns('__pycache__','build'))
    if C.tree_hash(retained)[0]!=compiler_hash:
        raise ValueError('retained compiler source differs from the measured candidate')
    backend=get_backend('gemmini')
    policy=backend.engine_readback_policy(engine)
    source = Path(member['source'])
    context = {'compiler_sha256':compiler_hash, 'source_sha256':C.file_hash(source), 'seed':seed,
               'facts_sha256':C.file_hash(C.environment()['MERLIN_RTL_FACTS']),
               'contract_sha256':C.file_hash(C.environment()['MERLIN_TARGET_CONTRACT']),
               'clang_sha256':C.file_hash(C.environment()['MERLIN_CLANG']),
               'driver_sha256':DRIVER_SHA, 'evaluator_profile':PROFILE, 'engine':engine, 'readback_policy':policy.record(),
               'environment_sha256':C.digest({k:v for k,v in C.environment().items()
                    if k.startswith('MERLIN_') or k in {'PYTHONPATH','TMPDIR','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'}})}
    key = C.digest(context)
    work = C.OUT / 'build' / member['identity'].replace('/', '__') / arm / key[:16]
    work.mkdir(parents=True, exist_ok=True)
    receipt_path = work / 'build-receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['context'] == context and C.file_hash(receipt['elf']) == receipt['elf_sha256']:
            cb = json.loads((work/'commands.json').read_text())
            data = dict(np.load(work/'inputs.npz'))
            return Path(receipt['elf']), cb, data, dict(receipt,build_reused=True)
    started = time.monotonic()
    with shared_compute_pool(timeout_s=90):
        actual_compile_cpus = sorted(__import__('os').sched_getaffinity(0))
        run = subprocess.run([str(C.PYTHON), str(package/'compiler.py'), '--convert-iface-to-gemmini',
                              f'--emit-command-buffer={work}/commands.json', '--emit-target-artifact', str(source)],
                             capture_output=True, text=True, check=False, timeout=120, env=C.environment())
    (work/'compiler.stderr').write_text(run.stderr)
    if run.returncode:
        raise ValueError('compiler refusal: '+run.stderr.strip())
    cb = json.loads((work/'commands.json').read_text())
    if cb.get('declined'):
        raise ValueError('compiler refusal: '+str(cb['declined']))
    # The candidate must retain precisely the source's logical work/readout contract.
    if member.get('command_buffer_sha256') and C.digest(cb) != member['command_buffer_sha256']:
        original_cb=json.loads(Path(member['command_buffer_path']).read_text())
        if C.logical_command_buffer(cb)!=C.logical_command_buffer(original_cb):
            raise ValueError('candidate changes logical command buffer semantics/source/placement/ABI')
    inputs = signed_inputs(cb, seed)
    np.savez(work/'inputs.npz', **inputs)
    (work/'emitted.llvm.mlir').write_text(run.stdout)
    if C.tree_hash(package)[0]!=compiler_hash:
        raise ValueError('compiler changed during emission')
    harness_cb=explicit_whole_program(cb,logical_abi()) if engine=='gsim' else cb
    C.save(work/'harness-commands.json',harness_cb)
    obj = llvm_mlir_to_object(run.stdout, work, target='gemmini')
    elf = link_elf(harness_cb, obj, work, target='gemmini', inputs={k:v.tolist() for k,v in inputs.items()},readback_policy=policy)
    audit = review_elf(elf, objdump=OBJDUMP, counter_asm='csrr %0, mcycle',
                       entry_symbol='kernel_entry', completion_symbol='kernel_complete')
    disasm = subprocess.run([str(OBJDUMP), '-d', '--no-show-raw-insn', str(elf)], capture_output=True, text=True, check=True).stdout
    (work/'final-elf.disassembly').write_text(disasm)
    completion = re.search(r'<kernel_complete>:\n(.*?)(?=\n\n|\Z)', disasm, re.S)
    if completion is None or 'fence' not in completion.group(1):
        raise ValueError('linked completion entry has no fence')
    audit['completion_contains_fence'] = True
    audit['completion_semantics'] = 'Rocket fence completion behavior requires selected RTL protocol evidence; static audit alone is not certification'
    C.save(work/'counter-protocol-audit.json', audit)
    receipt = {'context':context, 'elf':str(elf), 'elf_sha256':C.file_hash(elf),
               'object_sha256':C.file_hash(obj), 'inputs_sha256':C.file_hash(work/'inputs.npz'),
               'command_buffer_sha256':C.digest(cb), 'compile_link_s':time.monotonic()-started,
               'logical_command_buffer_sha256':C.digest(C.logical_command_buffer(cb)),
               'counter_audit_sha256':C.file_hash(work/'counter-protocol-audit.json'),
               'lowering_recipe_sha256':C.file_hash(work/'lowering_recipe.json'),
               'actual_compile_cpu_grant':actual_compile_cpus,
               'harness_command_buffer':str(work/'harness-commands.json'), 'readback_policy':policy.record(),
               'simulators': {'gsim':C.file_hash(C.environment()['MERLIN_GEMMINI_GSIM_EMU'])}}
    C.save(receipt_path, receipt)
    return elf, cb, inputs, dict(receipt,build_reused=False)


def evaluate(member, package, arm, seed, engine, repeat, *, timeout=180):
    if C.remaining() < timeout + 120:
        raise TimeoutError('campaign refuses job that cannot finish before verification reserve')
    with C.stage('compile_and_verify', member=member['identity'], arm=arm, engine=engine, repeat=repeat) as status:
        elf, cb, inputs, build_receipt = build(member, package, arm, seed,engine)
        outputs = reference(cb, inputs)
        started = time.monotonic()
        status['stage'] = 'simulate'
        backend = get_backend('gemmini')
        policy=ReadbackPolicy.from_record(build_receipt['readback_policy'])
        memory_evidence=None
        if engine=='gsim':
            harness_cb=json.loads(Path(build_receipt['harness_command_buffer']).read_text())
            invocation=(elf.parent/'measurements'/f'{arm}-{engine}-seed{seed}-r{repeat:03}').resolve()
            invocation.mkdir(parents=True,exist_ok=False)
            reader=NativeLogicalMemoryReadback(facts_path=Path(C.environment()['MERLIN_RTL_FACTS']).resolve(),policy=policy)
            require_current_build_receipt(cb=harness_cb,target='gemmini',workdir=elf.parent,elf_path=elf,policy=policy)
            kwargs=reader.prepare(cb=harness_cb,target='gemmini',elf_path=elf.resolve(),workdir=invocation,simulator=engine,backend=backend)
            console=backend.run_elf(elf,simulator=engine,timeout=timeout,**kwargs)
            serial_outputs,metrics=backend.parse_output(console,readback_policy=policy)
            actual,memory_evidence=reader.decode(console)
            require_current_build_receipt(cb=harness_cb,target='gemmini',workdir=elf.parent,elf_path=elf,policy=policy)
        else:
            console=backend.run_elf(elf,simulator=engine,timeout=timeout)
            actual,metrics=backend.parse_output(console)
        elapsed = time.monotonic() - started
        problems = []
        if set(actual) != set(outputs):
            problems.append(f'output roster mismatch: {set(actual)} versus {set(outputs)}')
        count = 0
        for name, expected in outputs.items():
            a, e = np.asarray(actual.get(name, [])), np.asarray(expected)
            count += e.size
            if a.size != e.size or not np.array_equal(a.reshape(-1), e.reshape(-1)):
                problems.append(name)
        cycles = metrics.get('cycles')
        if not isinstance(cycles, int) or cycles <= 0:
            problems.append('missing positive cycle metric')
        folder = elf.parent / 'measurements'
        folder.mkdir(exist_ok=True)
        path = folder / f'{engine}-seed{seed}-r{repeat:03}.json'
        if path.exists():
            raise ValueError('refuse to replace an existing measurement '+str(path))
        log = path.with_suffix('.log')
        log.write_text(console)
        result = {'schema':'phase2.exploratory.measurement.v1', 'member':member['identity'], 'arm':arm,
                  'seed':seed, 'repeat':repeat, 'engine':engine, 'cycles':cycles, 'exact':not problems,
                  'problems':problems, 'outputs':int(count), 'simulation_and_wait_s':elapsed,
                  'build':build_receipt, 'console_sha256':C.file_hash(log), 'record_path':str(path),
                  'timing_qualification':'exploratory_gsim_kernel_window' if engine=='gsim' else 'functional_only_uncalibrated_cycles',
                  'cold_warm':'UNKNOWN', 'roofline':member.get('roofline'),
                  'evaluator_profile':PROFILE,'readback_policy':policy.record(),'memory_evidence':memory_evidence,
                  'source_sha256':member['source_sha256'], 'cohort_sha256':C.file_hash(MANIFEST_PATH)}
        C.save(path, result)
        with (C.OUT/'measurements.jsonl').open('a') as f:
            f.write(json.dumps(result, sort_keys=True)+'\n')
        print(json.dumps({k:result[k] for k in ('member','arm','engine','repeat','cycles','exact','simulation_and_wait_s')}), flush=True)
        return result


def main():
    global MANIFEST_PATH
    if (C.EVIDENCE/'manual-stop.json').exists():
        raise SystemExit('Manual campaign stopped by user workflow correction. Use the existing automatic sandboxed Phase2 experiment route; historical evidence remains readable.')
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', required=True, help='comma-separated exact manifest identities')
    ap.add_argument('--manifest', type=Path, default=MANIFEST_PATH)
    ap.add_argument('--candidate', type=Path, default=C.CANDIDATE)
    ap.add_argument('--label', default='rhs_residency')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--repeat-offset', type=int, default=0)
    ap.add_argument('--engines', default='spike,gsim')
    ap.add_argument('--timeout', type=int, default=180)
    args = ap.parse_args()
    MANIFEST_PATH = args.manifest
    manifest = json.loads(MANIFEST_PATH.read_text())
    members = {m['identity']:m for m in manifest['members']}
    failures = []
    for identity in args.members.split(','):
        for engine in args.engines.split(','):
            for repeat in range(args.repeat_offset, args.repeat_offset + (args.repeats if engine=='gsim' else 1)):
                arms = [('baseline',C.BASE), (args.label,args.candidate)]
                if repeat % 2:
                    arms.reverse()
                for arm, package in arms:
                    try:
                        evaluate(members[identity], package, arm, args.seed, engine, repeat, timeout=args.timeout)
                    except Exception as error:
                        failure = {'member':identity,'arm':arm,'engine':engine,'repeat':repeat,'error':str(error)}
                        failures.append(failure)
                        with (C.OUT/'failures.jsonl').open('a') as f:
                            f.write(json.dumps(failure)+'\n')
                        print(json.dumps(failure),flush=True)
                        # Retain the failing member, avoid spending repetitions on a broken arm.
                        break
                if failures and failures[-1]['member']==identity and failures[-1]['engine']==engine:
                    break
    batch_id = C.digest([args.members,args.engines,args.repeats,args.seed,args.repeat_offset])[:10]
    C.save(C.EVIDENCE/f'batch-{args.label}-seed{args.seed}-{batch_id}.json',
           {'members':args.members.split(','),'engines':args.engines,'repeats':args.repeats,
            'candidate_sha256':C.tree_hash(args.candidate)[0], 'failures':failures,
            'status':'incomplete' if failures else 'completed_exploratory'})


if __name__ == '__main__':
    main()
